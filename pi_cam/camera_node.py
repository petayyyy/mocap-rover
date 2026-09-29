#!/usr/bin/env python3
"""IMX219 camera node for a Raspberry Pi CM4/CM5: windows on request over LAN.

The sensor runs continuously at 1640x1232 / 8 bit / ~83 fps whether or not a
laptop is connected.  A full frame at that rate does not fit 1 GbE, so the
node sends what the laptop asks for: a list of windows cut from every frame
(``set_windows``), an occasional full frame (``full_frame``), or a decimated
stream of full frames for dataset recording (``stream_full``).  All windows of
one sensor frame carry the same ``stamp_ns``.

Threads (the capture thread never blocks on the network):

    capture  -- picamera2 request loop; reads metadata, cuts windows from the
                mapped buffer, hands a job to the sender; drops the job when
                the sender is behind and counts it
    sender   -- JPEG encoding and socket writes; the only writer of the socket
    server   -- accept loop; one client at a time, commands parsed here
    status   -- once a second: sensor fps, drops, latencies, PTP, temperature
    ptp      -- polls ``pmc`` for offsetFromMaster and the port state

Time.  libcamera's ``SensorTimestamp`` is CLOCK_BOOTTIME.  The node converts
it to CLOCK_REALTIME, which PTP (``ptp4l`` + ``phc2sys``) keeps common with the
laptop, using an offset sampled around every frame.  What SensorTimestamp
marks on the frame (start of exposure, start of readout, end of frame) is a
platform fact to be measured with ``led_timestamp_probe.py``; the node applies
the measured reference from its config so that ``stamp_ns`` is always the
exposure start of the first row.  The rolling-shutter row period
``line_time_ns`` goes in every header and is verified by the same probe.

Run with ``--synthetic`` on any machine to exercise the protocol without a
camera; the tests do exactly that.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import queue
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Started as ``python3 camera_node.py`` from the pi_cam directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_cam import lan_protocol as proto  # noqa: E402

log = logging.getLogger("camera_node")

SENSOR_WIDTH, SENSOR_HEIGHT = 1640, 1232
NS = 1_000_000_000

# IMX219 in the 1640x1232 2x2-binned 8-bit mode (kernel imx219.c): line length
# 3448 pixel clocks at 182.4 MHz, and the binned 8-bit readout doubles the
# effective line rate.  9452 ns per row, 1232 rows in 11.6 ms.  These are the
# prior; the LED probe measures the real value and the config overrides it.
IMX219_LINE_LENGTH_PCK = 3448
IMX219_PIXEL_RATE_HZ = 182_400_000
IMX219_BINNED_8BIT_RATE_FACTOR = 2

THROTTLE_FLAGS = {
    0: "under_voltage_now", 1: "arm_freq_capped_now", 2: "throttled_now",
    3: "soft_temp_limit_now", 16: "under_voltage_occurred",
    17: "arm_freq_capped_occurred", 18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}


# ----------------------------------------------------------------- config

@dataclass
class NodeConfig:
    camera_id: str = "camera_1"
    host: str = "0.0.0.0"
    port: int = 5600
    sensor: str = "picamera2"                 # or "synthetic"
    width: int = SENSOR_WIDTH
    height: int = SENSOR_HEIGHT
    bit_depth: int = 8
    fps: float = 83.0
    exposure_us: int = 800
    analogue_gain: float = 4.0
    buffer_count: int = 6
    jpeg_quality: int = 90
    default_windows: list = field(default_factory=list)
    full_frame_format: str = "y8"
    status_period_s: float = 1.0
    send_queue: int = 3
    # What SensorTimestamp marks; set from the LED probe result.
    stamp_reference: str = "readout_start_first_row"
    stamp_correction_ns: int = 0
    # Rolling shutter row period; null = computed from the register model.
    line_time_ns: int | None = None
    line_length_pck: int = IMX219_LINE_LENGTH_PCK
    pixel_rate_hz: int = IMX219_PIXEL_RATE_HZ
    line_rate_factor: int = IMX219_BINNED_8BIT_RATE_FACTOR
    ptp_enabled: bool = True
    ptp_pmc: str = "pmc"
    ptp_uds: str = "/var/run/ptp4l"
    ptp_period_s: float = 1.0
    synthetic_fps: float = 83.0
    synthetic_width: int = SENSOR_WIDTH
    synthetic_height: int = SENSOR_HEIGHT

    @classmethod
    def load(cls, path=None, **overrides):
        data = {}
        if path:
            data = json.loads(Path(path).read_text())
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        cfg = cls(**data)
        if cfg.stamp_reference not in ("exposure_start_first_row", "readout_start_first_row", "frame_end"):
            raise ValueError(f"unknown stamp_reference {cfg.stamp_reference!r}")
        if len(cfg.camera_id.encode("ascii")) > 16:
            raise ValueError("camera_id must be at most 16 ASCII characters")
        return cfg

    def computed_line_time_ns(self) -> int:
        if self.line_time_ns:
            return int(self.line_time_ns)
        return int(round(self.line_length_pck * NS / (self.pixel_rate_hz * self.line_rate_factor)))

    def line_time_source(self) -> str:
        return "config" if self.line_time_ns else "register_model"


# ---------------------------------------------------------------- clocks

def boottime_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def realtime_minus_boottime_ns(samples: int = 3) -> int:
    """REALTIME - BOOTTIME, from the tightest of a few bracketed samples."""
    best_gap, best = None, 0
    for _ in range(samples):
        b0 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        r = time.clock_gettime_ns(time.CLOCK_REALTIME)
        b1 = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        gap = b1 - b0
        if best_gap is None or gap < best_gap:
            best_gap, best = gap, r - (b0 + b1) // 2
    return best


def stamp_reference_correction_ns(reference: str, exposure_ns: int, line_time_ns: int,
                                  height: int) -> int:
    """Offset that turns SensorTimestamp into the first row's exposure start."""
    if reference == "exposure_start_first_row":
        return 0
    if reference == "readout_start_first_row":
        return -int(exposure_ns)
    if reference == "frame_end":
        return -int(exposure_ns + height * line_time_ns)
    raise ValueError(reference)


class PtpMonitor(threading.Thread):
    """Poll ``pmc`` for the PTP offset and port state; tolerate its absence."""

    def __init__(self, cfg: NodeConfig):
        super().__init__(daemon=True, name="ptp")
        self.cfg = cfg
        self.lock = threading.Lock()
        self.offset_ns = None
        self.state = "disabled" if not cfg.ptp_enabled else "unknown"
        self.updated_mono = None
        self.error = None
        self.stop_event = threading.Event()

    def snapshot(self) -> dict:
        with self.lock:
            age = None if self.updated_mono is None else time.monotonic() - self.updated_mono
            return {"offset_ns": self.offset_ns, "state": self.state,
                    "age_s": age, "error": self.error}

    def poll_once(self):
        try:
            out = subprocess.run(
                [self.cfg.ptp_pmc, "-u", "-b", "0", "-s", self.cfg.ptp_uds,
                 "GET CURRENT_DATA_SET", "GET PORT_DATA_SET"],
                capture_output=True, text=True, timeout=2.0).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            with self.lock:
                self.state, self.error = "unavailable", str(exc)
            return
        offset = re.search(r"offsetFromMaster\s+(-?\d+(?:\.\d+)?)", out)
        state = re.search(r"portState\s+(\w+)", out)
        with self.lock:
            self.offset_ns = int(float(offset.group(1))) if offset else None
            self.state = state.group(1) if state else ("no_answer" if not out.strip() else "parsed")
            self.updated_mono = time.monotonic()
            self.error = None

    def run(self):
        if not self.cfg.ptp_enabled:
            return
        while not self.stop_event.is_set():
            self.poll_once()
            self.stop_event.wait(self.cfg.ptp_period_s)


# --------------------------------------------------------------- sensors

class Frame:
    """One captured frame: a view of the Y plane plus libcamera metadata."""

    __slots__ = ("y", "sensor_stamp_ns", "exposure_ns", "frame_duration_ns", "gain",
                 "sequence", "_release")

    def __init__(self, y, sensor_stamp_ns, exposure_ns, frame_duration_ns, gain,
                 sequence, release=None):
        self.y = y
        self.sensor_stamp_ns = int(sensor_stamp_ns)
        self.exposure_ns = int(exposure_ns)
        self.frame_duration_ns = int(frame_duration_ns)
        self.gain = float(gain)
        self.sequence = int(sequence)
        self._release = release

    def release(self):
        if self._release is not None:
            self._release()
            self._release = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class SyntheticSensor:
    """Stand-in for picamera2: a moving pattern with BOOTTIME stamps.

    Pixel (r, c) of frame ``seq`` is ``(7*r + 3*c + seq) & 0xFF``, so a
    receiver can check that a window really came from the coordinates and the
    frame the header claims.
    """

    def __init__(self, cfg: NodeConfig):
        self.width, self.height = cfg.synthetic_width, cfg.synthetic_height
        self.fps = float(cfg.synthetic_fps)
        self.exposure_ns = int(cfg.exposure_us) * 1000
        self.gain = float(cfg.analogue_gain)
        rows = np.arange(self.height, dtype=np.uint32)[:, None] * 7
        cols = np.arange(self.width, dtype=np.uint32)[None, :] * 3
        self.base = ((rows + cols) & 0xFF).astype(np.uint8)
        self.scratch = np.empty_like(self.base)
        self.sequence = 0
        self.next_due = None
        self.mode = {"size": [self.width, self.height], "bit_depth": 8, "format": "synthetic"}

    @staticmethod
    def expected_pixel(row, col, sequence):
        return (7 * row + 3 * col + sequence) & 0xFF

    def start(self):
        self.next_due = boottime_ns()

    def stop(self):
        pass

    def set_controls(self, exposure_us=None, gain=None, fps=None):
        if exposure_us is not None:
            self.exposure_ns = int(exposure_us) * 1000
        if gain is not None:
            self.gain = float(gain)
        if fps is not None:
            self.fps = float(fps)
        return {"exposure_us": self.exposure_ns // 1000, "gain": self.gain, "fps": self.fps}

    def capture(self) -> Frame:
        """Return the frame whose exposure started at ``next_due``.

        Like the real sensor it is delivered one frame period after its
        exposure start, when the readout has finished; ``sensor_stamp_ns`` is
        that exposure start.
        """
        period = int(NS / self.fps)
        stamp = self.next_due
        ready = stamp + period
        now = boottime_ns()
        if ready > now:
            time.sleep((ready - now) / NS)
        self.next_due += period
        if self.next_due < boottime_ns() - 5 * period:   # consumer stalled: resync
            self.next_due = boottime_ns() - period
        seq = self.sequence
        self.sequence += 1
        np.add(self.base, np.uint8(seq & 0xFF), out=self.scratch)
        return Frame(self.scratch, stamp, self.exposure_ns, period, self.gain, seq)


class PicameraSensor:
    """picamera2 in the 1640x1232 / 8-bit / YUV420 configuration, Y plane only."""

    def __init__(self, cfg: NodeConfig):
        from picamera2 import MappedArray, Picamera2
        self._MappedArray = MappedArray
        self.cfg = cfg
        self.width, self.height = cfg.width, cfg.height
        self.camera = Picamera2()
        duration_us = int(round(1_000_000 / cfg.fps))
        config = self.camera.create_video_configuration(
            main={"size": (cfg.width, cfg.height), "format": "YUV420"},
            sensor={"output_size": (cfg.width, cfg.height), "bit_depth": cfg.bit_depth},
            controls={
                "FrameDurationLimits": (duration_us, duration_us),
                "ExposureTime": int(cfg.exposure_us),
                "AnalogueGain": float(cfg.analogue_gain),
                "AeEnable": False, "AwbEnable": False,
            },
            buffer_count=cfg.buffer_count)
        self.camera.configure(config)
        actual = self.camera.camera_configuration()
        main = actual["main"]
        self.stride = int(main["stride"])
        sensor = actual.get("sensor", {})
        self.mode = {"size": list(sensor.get("output_size", (cfg.width, cfg.height))),
                     "bit_depth": sensor.get("bit_depth"), "format": main["format"],
                     "stride": self.stride, "raw": actual.get("raw", {}).get("format")}
        if tuple(self.mode["size"]) != (cfg.width, cfg.height) or self.mode["bit_depth"] != cfg.bit_depth:
            raise RuntimeError(f"libcamera picked sensor mode {self.mode}, not "
                               f"{cfg.width}x{cfg.height}/{cfg.bit_depth}-bit")
        self.sequence = 0
        self.last_stamp = None

    def start(self):
        self.camera.start()

    def stop(self):
        self.camera.stop()

    def set_controls(self, exposure_us=None, gain=None, fps=None):
        controls = {}
        if exposure_us is not None:
            controls["ExposureTime"] = int(exposure_us)
        if gain is not None:
            controls["AnalogueGain"] = float(gain)
        if fps is not None:
            duration_us = int(round(1_000_000 / float(fps)))
            controls["FrameDurationLimits"] = (duration_us, duration_us)
        if controls:
            self.camera.set_controls(controls)
        return {"exposure_us": exposure_us, "gain": gain, "fps": fps}

    def capture(self) -> Frame:
        request = self.camera.capture_request()
        metadata = request.get_metadata()
        try:
            mapped = self._MappedArray(request, "main", reshape=False, write=False)
        except TypeError:   # older picamera2 without the reshape flag
            mapped = self._MappedArray(request, "main")
        mapped.__enter__()
        array = mapped.array
        h, w = self.height, self.width
        if array.ndim == 1:
            y = array[: self.stride * h].reshape(h, self.stride)[:, :w]
        else:
            y = array[:h, :w]

        def release():
            mapped.__exit__(None, None, None)
            request.release()

        seq = getattr(getattr(request, "request", None), "sequence", None)
        if seq is None:
            seq = self.sequence
        self.sequence += 1
        return Frame(y, metadata["SensorTimestamp"], int(metadata["ExposureTime"]) * 1000,
                     int(metadata["FrameDuration"]) * 1000, metadata.get("AnalogueGain", 0.0),
                     seq, release)


def make_sensor(cfg: NodeConfig):
    if cfg.sensor == "synthetic":
        return SyntheticSensor(cfg)
    if cfg.sensor == "picamera2":
        return PicameraSensor(cfg)
    raise ValueError(f"unknown sensor {cfg.sensor!r}")


# --------------------------------------------------------------- encoding

def _jpeg_encoder():
    try:
        import simplejpeg

        def encode(gray, quality):
            return simplejpeg.encode_jpeg(np.ascontiguousarray(gray)[:, :, None],
                                          quality=quality, colorspace="GRAY")
        return encode, "simplejpeg"
    except ImportError:
        pass
    try:
        import cv2

        def encode(gray, quality):
            ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(gray),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            return buf.tobytes()
        return encode, "cv2"
    except ImportError:
        return None, "none"


JPEG_ENCODE, JPEG_BACKEND = _jpeg_encoder()


# ------------------------------------------------------------------ node

@dataclass
class Window:
    row0: int
    col0: int
    w: int
    h: int
    format: str
    request_id: int


@dataclass
class FrameJob:
    frame_seq: int
    sensor_stamp_ns: int
    stamp_ns: int
    exposure_ns: int
    frame_duration_ns: int
    clock_offset_ns: int
    windows: list          # of (Window, ndarray)


def read_soc_temperature_c():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except (OSError, ValueError):
        return None


def read_throttled():
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=1.0).stdout
        value = int(out.strip().split("=")[1], 16)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return {"raw": f"0x{value:x}", "flags": [name for bit, name in THROTTLE_FLAGS.items()
                                             if value & (1 << bit)]}


class CpuLoad:
    """CPU utilisation between two calls, from /proc/stat."""

    def __init__(self):
        self.last = self._read()

    @staticmethod
    def _read():
        try:
            fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            values = [int(v) for v in fields]
            return sum(values), values[3] + values[4]   # total, idle + iowait
        except (OSError, ValueError, IndexError):
            return None

    def percent(self):
        now = self._read()
        if now is None or self.last is None:
            return None
        total, idle = now[0] - self.last[0], now[1] - self.last[1]
        self.last = now
        return None if total <= 0 else round(100.0 * (total - idle) / total, 1)


class CameraNode:
    def __init__(self, cfg: NodeConfig, sensor=None):
        self.cfg = cfg
        self.sensor = sensor or make_sensor(cfg)
        self.sensor_width, self.sensor_height = self.sensor.width, self.sensor.height
        self.line_time_ns = cfg.computed_line_time_ns()
        self.started_mono = time.monotonic()
        self.stop_event = threading.Event()

        self.state_lock = threading.Lock()
        self.windows: list[Window] = []
        self.full_pending: Window | None = None
        self.full_stream: dict | None = None      # {"divisor", "format", "request_id"}
        self.request_counter = 0
        for w in cfg.default_windows:
            self._set_windows_locked([w], self._next_request_id())

        self.frame_queue: queue.Queue = queue.Queue(maxsize=max(1, cfg.send_queue))
        self.control_queue: queue.Queue = queue.Queue()
        self.client_lock = threading.Lock()
        self.client: socket.socket | None = None
        self.client_addr = None

        self.stats_lock = threading.Lock()
        self.frames_captured = 0
        self.frames_missed = 0
        self.frames_with_output = 0
        self.frames_sent = 0
        self.frames_dropped_queue = 0
        self.frames_no_client = 0
        self.windows_sent = 0
        self.full_sent = 0
        self.bytes_sent = 0
        self.send_errors = 0
        self.last_error = None
        self.frame_durations = collections.deque(maxlen=200)
        self.capture_to_send_ns = collections.deque(maxlen=400)
        self.capture_cost_ns = collections.deque(maxlen=400)
        self.last_exposure_ns = cfg.exposure_us * 1000
        self.last_gain = cfg.analogue_gain
        self.last_sensor_stamp = None
        self.cpu = CpuLoad()
        self.ptp = PtpMonitor(cfg)
        self.server_socket = None
        self.threads = []
        self.port = cfg.port

    # ----------------------------------------------------------- requests

    def _next_request_id(self):
        self.request_counter += 1
        return self.request_counter

    def _set_windows_locked(self, windows, request_id):
        normalized = [proto.normalize_window(w, self.sensor_width, self.sensor_height)
                      for w in windows]
        self.windows = [Window(n["row0"], n["col0"], n["w"], n["h"], n["format"], request_id)
                        for n in normalized]
        return [asdict(w) for w in self.windows]

    def handle_command(self, meta: dict) -> dict:
        cmd = meta.get("cmd")
        if cmd == "status":
            # status() takes the state lock itself, so answer outside it.
            self.control_queue.put(proto.encode_json(proto.MSG_STATUS, self.status()))
            return {"ok": True, "cmd": cmd}
        with self.state_lock:
            request_id = self._next_request_id()
            try:
                if cmd == "set_windows":
                    windows = self._set_windows_locked(meta.get("windows", []), request_id)
                    return {"ok": True, "cmd": cmd, "request_id": request_id, "windows": windows}
                if cmd == "full_frame":
                    fmt = meta.get("format", self.cfg.full_frame_format)
                    if fmt not in proto.FORMAT_CODES:
                        raise proto.ProtocolError(f"unknown format {fmt!r}")
                    self.full_pending = Window(0, 0, self.sensor_width, self.sensor_height, fmt, request_id)
                    return {"ok": True, "cmd": cmd, "request_id": request_id, "format": fmt}
                if cmd == "stream_full":
                    divisor = int(meta.get("divisor", 0))
                    fmt = meta.get("format", self.cfg.full_frame_format)
                    if fmt not in proto.FORMAT_CODES:
                        raise proto.ProtocolError(f"unknown format {fmt!r}")
                    self.full_stream = (None if divisor <= 0 else
                                        {"divisor": divisor, "format": fmt, "request_id": request_id})
                    return {"ok": True, "cmd": cmd, "request_id": request_id,
                            "divisor": divisor, "format": fmt}
                if cmd == "configure":
                    applied = self.sensor.set_controls(
                        exposure_us=meta.get("exposure_us"), gain=meta.get("gain"), fps=meta.get("fps"))
                    return {"ok": True, "cmd": cmd, "request_id": request_id, "applied": applied}
                raise proto.ProtocolError(f"unknown command {cmd!r}")
            except (proto.ProtocolError, ValueError, KeyError, TypeError) as exc:
                return {"ok": False, "cmd": cmd, "request_id": request_id, "error": str(exc)}

    # ------------------------------------------------------------ capture

    def convert_stamp(self, sensor_stamp_ns, exposure_ns, clock_offset_ns):
        correction = stamp_reference_correction_ns(
            self.cfg.stamp_reference, exposure_ns, self.line_time_ns, self.sensor_height)
        return sensor_stamp_ns + clock_offset_ns + correction + self.cfg.stamp_correction_ns

    def capture_loop(self):
        sensor = self.sensor
        while not self.stop_event.is_set():
            try:
                frame = sensor.capture()
            except Exception as exc:  # camera hiccup: keep the node alive
                self.last_error = f"capture: {exc}"
                log.warning("capture failed: %s", exc)
                time.sleep(0.05)
                continue
            begin = time.perf_counter_ns()
            with frame:
                clock_offset = realtime_minus_boottime_ns()
                stamp_ns = self.convert_stamp(frame.sensor_stamp_ns, frame.exposure_ns, clock_offset)
                with self.state_lock:
                    windows = list(self.windows)
                    full = self.full_pending
                    self.full_pending = None
                    stream = self.full_stream
                if stream is not None and frame.sequence % stream["divisor"] == 0 and full is None:
                    full = Window(0, 0, self.sensor_width, self.sensor_height,
                                  stream["format"], stream["request_id"])
                cuts = []
                for w in windows:
                    cuts.append((w, np.ascontiguousarray(frame.y[w.row0:w.row0 + w.h, w.col0:w.col0 + w.w])))
                if full is not None:
                    cuts.append((full, np.ascontiguousarray(frame.y)))
                job = FrameJob(frame.sequence, frame.sensor_stamp_ns, stamp_ns, frame.exposure_ns,
                               frame.frame_duration_ns, clock_offset, cuts)
            with self.stats_lock:
                self.frames_captured += 1
                self.frame_durations.append(frame.frame_duration_ns)
                self.last_exposure_ns = frame.exposure_ns
                self.last_gain = frame.gain
                if self.last_sensor_stamp is not None and frame.frame_duration_ns > 0:
                    gap = frame.sensor_stamp_ns - self.last_sensor_stamp
                    missed = int(round(gap / frame.frame_duration_ns)) - 1
                    if missed > 0:
                        self.frames_missed += missed
                self.last_sensor_stamp = frame.sensor_stamp_ns
                self.capture_cost_ns.append(time.perf_counter_ns() - begin)
            if not cuts:
                continue
            with self.stats_lock:
                self.frames_with_output += 1
            try:
                self.frame_queue.put_nowait(job)
            except queue.Full:
                with self.stats_lock:
                    self.frames_dropped_queue += 1

    # ------------------------------------------------------------- sender

    def _send_parts(self, sock, parts):
        views = [memoryview(p) for p in parts]
        while views:
            sent = sock.sendmsg(views)
            with self.stats_lock:
                self.bytes_sent += sent
            while views and sent >= len(views[0]):
                sent -= len(views[0])
                views.pop(0)
            if views and sent:
                views[0] = views[0][sent:]

    def sender_loop(self):
        while not self.stop_event.is_set():
            # Control messages (status, acks) go first; they are tiny.
            parts = None
            try:
                parts = [self.control_queue.get_nowait()]
            except queue.Empty:
                try:
                    job = self.frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                parts = self._encode_job(job)
                if parts is None:
                    continue
            with self.client_lock:
                sock = self.client
            if sock is None:
                with self.stats_lock:
                    self.frames_no_client += 1
                continue
            try:
                self._send_parts(sock, parts)
            except OSError as exc:
                with self.stats_lock:
                    self.send_errors += 1
                    self.last_error = f"send: {exc}"
                self._drop_client(sock)

    def _encode_job(self, job: FrameJob):
        parts = []
        count = len(job.windows)
        ptp = self.ptp.snapshot()["offset_ns"]
        for index, (w, array) in enumerate(job.windows):
            if w.format == "jpeg":
                if JPEG_ENCODE is None:
                    self.last_error = "no JPEG encoder (install simplejpeg or opencv)"
                    continue
                data = JPEG_ENCODE(array, self.cfg.jpeg_quality)
            else:
                data = memoryview(array).cast("B")
            node_send_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
            header = proto.FrameHeader(
                camera_id=self.cfg.camera_id, frame_seq=job.frame_seq, stamp_ns=job.stamp_ns,
                exposure_ns=job.exposure_ns, line_time_ns=self.line_time_ns,
                frame_duration_ns=job.frame_duration_ns, row0=w.row0, col0=w.col0,
                width=w.w, height=w.h, format=proto.FORMAT_CODES[w.format],
                window_index=index, window_count=count, sensor_width=self.sensor_width,
                sensor_height=self.sensor_height, request_id=w.request_id,
                node_send_ns=node_send_ns, sensor_stamp_ns=job.sensor_stamp_ns,
                clock_offset_ns=job.clock_offset_ns, ptp_offset_ns=ptp)
            parts.extend(proto.encode_frame(header, data))
            with self.stats_lock:
                if w.w == self.sensor_width and w.h == self.sensor_height:
                    self.full_sent += 1
                else:
                    self.windows_sent += 1
                self.capture_to_send_ns.append(node_send_ns - job.stamp_ns)
        if not parts:
            return None
        with self.stats_lock:
            self.frames_sent += 1
        return parts

    # ------------------------------------------------------------- server

    def _drop_client(self, sock):
        with self.client_lock:
            if self.client is sock:
                self.client = None
                self.client_addr = None
        for step in (lambda: sock.shutdown(socket.SHUT_RDWR), sock.close):
            try:
                step()
            except OSError:
                pass

    def server_loop(self):
        while not self.stop_event.is_set():
            try:
                sock, addr = self.server_socket.accept()
            except OSError:
                if self.stop_event.is_set():
                    return
                continue
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(5.0)
            with self.client_lock:
                old, self.client, self.client_addr = self.client, sock, addr
            if old is not None:
                log.info("replacing the previous client")
                self._drop_client(old)
            log.info("client %s connected", addr)
            self.control_queue.put(proto.encode_json(proto.MSG_HELLO, self.hello()))
            threading.Thread(target=self.client_loop, args=(sock,), daemon=True,
                             name="client").start()

    def client_loop(self, sock):
        """Read commands from one client until it disconnects or is replaced."""
        while not self.stop_event.is_set():
            with self.client_lock:
                if self.client is not sock:
                    return
            try:
                msg = proto.read_message(sock)
            except socket.timeout:
                continue            # quiet client; the sender keeps streaming
            except (ConnectionError, OSError, proto.ProtocolError, ValueError) as exc:
                log.info("client gone: %s", exc)
                self._drop_client(sock)
                return
            if msg.msg_type != proto.MSG_COMMAND:
                continue
            answer = self.handle_command(msg.meta)
            if "token" in msg.meta:
                answer["token"] = msg.meta["token"]
            self.control_queue.put(proto.encode_json(proto.MSG_ACK, answer))

    # ------------------------------------------------------------- status

    def hello(self):
        return {"camera_id": self.cfg.camera_id, "protocol_version": proto.PROTOCOL_VERSION,
                "sensor_width": self.sensor_width, "sensor_height": self.sensor_height,
                "mode": getattr(self.sensor, "mode", {}), "line_time_ns": self.line_time_ns,
                "line_time_source": self.cfg.line_time_source(),
                "stamp_reference": self.cfg.stamp_reference,
                "stamp_correction_ns": self.cfg.stamp_correction_ns,
                "jpeg_backend": JPEG_BACKEND, "sensor": self.cfg.sensor}

    def status(self):
        with self.stats_lock:
            durations = list(self.frame_durations)
            latencies = sorted(self.capture_to_send_ns)
            costs = sorted(self.capture_cost_ns)
            counters = {
                "frames_captured": self.frames_captured, "frames_missed": self.frames_missed,
                "frames_with_output": self.frames_with_output, "frames_sent": self.frames_sent,
                "frames_dropped_queue": self.frames_dropped_queue,
                "frames_no_client": self.frames_no_client, "windows_sent": self.windows_sent,
                "full_sent": self.full_sent, "bytes_sent": self.bytes_sent,
                "send_errors": self.send_errors}
            exposure, gain = self.last_exposure_ns, self.last_gain
            error = self.last_error
        with self.state_lock:
            windows = [asdict(w) for w in self.windows]
            stream = dict(self.full_stream) if self.full_stream else None
        with self.client_lock:
            client = None if self.client is None else f"{self.client_addr[0]}:{self.client_addr[1]}"

        def pct(values, q):
            if not values:
                return None
            return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]

        median_duration = statistics.median(durations) if durations else 0
        return {
            "camera_id": self.cfg.camera_id, "uptime_s": round(time.monotonic() - self.started_mono, 1),
            "sensor_fps": round(NS / median_duration, 2) if median_duration else 0.0,
            "frame_duration_ns": int(median_duration), "exposure_ns": exposure, "analogue_gain": gain,
            "line_time_ns": self.line_time_ns, "line_time_source": self.cfg.line_time_source(),
            "stamp_reference": self.cfg.stamp_reference,
            **counters,
            "capture_to_send_ms": {"p50": None if not latencies else pct(latencies, 0.5) / 1e6,
                                   "p95": None if not latencies else pct(latencies, 0.95) / 1e6},
            "capture_cost_us": {"p50": None if not costs else pct(costs, 0.5) / 1e3,
                                "p95": None if not costs else pct(costs, 0.95) / 1e3},
            "queue_depth": self.frame_queue.qsize(),
            "ptp": self.ptp.snapshot(), "clock_offset_ns": realtime_minus_boottime_ns(),
            "soc_temp_c": read_soc_temperature_c(), "throttled": read_throttled(),
            "cpu_percent": self.cpu.percent(), "windows": windows, "full_stream": stream,
            "client": client, "error": error, "jpeg_backend": JPEG_BACKEND,
            "status_time_ns": time.clock_gettime_ns(time.CLOCK_REALTIME),
        }

    def status_loop(self):
        while not self.stop_event.wait(self.cfg.status_period_s):
            with self.client_lock:
                connected = self.client is not None
            status = self.status()
            if connected:
                self.control_queue.put(proto.encode_json(proto.MSG_STATUS, status))
            log.debug("fps %.2f missed %d dropped %d client %s", status["sensor_fps"],
                      status["frames_missed"], status["frames_dropped_queue"], status["client"])

    # ---------------------------------------------------------- lifecycle

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.cfg.host, self.cfg.port))
        self.server_socket.listen(2)
        self.server_socket.settimeout(0.5)
        self.port = self.server_socket.getsockname()[1]
        self.sensor.start()
        self.ptp.start()
        for target, name in ((self.capture_loop, "capture"), (self.sender_loop, "sender"),
                             (self.server_loop, "server"), (self.status_loop, "status")):
            thread = threading.Thread(target=target, daemon=True, name=name)
            thread.start()
            self.threads.append(thread)
        log.info("camera %s: %dx%d, line_time %d ns (%s), port %d, jpeg via %s",
                 self.cfg.camera_id, self.sensor_width, self.sensor_height, self.line_time_ns,
                 self.cfg.line_time_source(), self.port, JPEG_BACKEND)

    def stop(self):
        self.stop_event.set()
        self.ptp.stop_event.set()
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass
        with self.client_lock:
            sock, self.client = self.client, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        for thread in self.threads:
            thread.join(timeout=2.0)
        try:
            self.sensor.stop()
        except Exception as exc:  # noqa: BLE001
            log.warning("sensor stop: %s", exc)

    def run_forever(self):
        self.start()
        try:
            while not self.stop_event.wait(1.0):
                pass
        finally:
            self.stop()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON config (see node_config.example.json)")
    p.add_argument("--camera-id")
    p.add_argument("--port", type=int)
    p.add_argument("--synthetic", action="store_true", help="run without a camera")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = NodeConfig.load(a.config, camera_id=a.camera_id, port=a.port,
                          sensor="synthetic" if a.synthetic else None)
    node = CameraNode(cfg)
    signal.signal(signal.SIGTERM, lambda *_: node.stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: node.stop_event.set())
    node.run_forever()


if __name__ == "__main__":
    main()
