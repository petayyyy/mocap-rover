#!/usr/bin/env python3
"""Measure the CM4's hardware H.264 encoder on the node's own frames.

Run it on a node with the camera free (``camera_node`` stopped).  It reads the
sensor exactly as the node does (raw SBGGR8 1640x1232 on a CM4), turns each
frame into the grey picture a stream would carry, feeds it to the VideoCore
encoder through V4L2 (``bcm2835-codec``, /dev/video11) and times every step:

* platform  -- exposure start of the first row -> frame in Python (known: 22.6 ms)
* prep      -- the reduction to 640x480 (``scale_luma``, as the small stream)
               or the demosaic to full-size grey, plus the copy into the
               encoder's buffer
* encoder   -- buffer queued -> its H.264 frame dequeued
* total     -- exposure start -> encoded frame ready to send

and counts what the encoder could not keep up with.

The question this answers first is whether the encoder HOLDS a frame (gives
frame k only once frame k+1 arrives).  If it does, a whole frame period sits
inside its latency: 12 ms at 83 Hz, 33 ms at 30 Hz.  Each case starts with a
hold test -- single frames 0.3 s apart, each waited for alone -- and the
continuous run then reports how often frame k came out before frame k+1 went
in.

The encoder is driven directly through V4L2 ioctls (ctypes, no extra
packages), so what is timed is the hardware and the kernel driver, not a
wrapper's queue.  The structure layouts are checked against the kernel
headers at import time.

Typical use on the node (cases are WxH@HZ, ``full`` = 1640x1232, HZ ``all``
= every sensor frame)::

    sudo systemctl stop camera_node
    PYTHONPATH=/opt/mocap-rover python3 h264_probe.py --config /etc/mocap-rover/node_config.json \\
        --exposure-us 3000 --seconds 30 --output h264_static.json
    sudo systemctl start camera_node
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import errno
import fcntl
import json
import mmap
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_cam.camera_node import NodeConfig, boottime_ns, make_sensor, scale_luma  # noqa: E402

try:
    import cv2 as _cv2
except ImportError:      # pragma: no cover - the node has OpenCV since the small stream
    _cv2 = None

FULL_W, FULL_H = 1640, 1232
DEFAULT_CASES = ("640x480@all", "640x480@30", "full@30", "full@all")


# ------------------------------------------------------------------ V4L2 ABI
# Layouts from <linux/videodev2.h>; identical on aarch64 and x86_64 (LP64).

def _ioc(direction, nr, size):
    return (direction << 30) | (size << 16) | (ord("V") << 8) | nr


_IOC_W, _IOC_R = 1, 2
_IOC_RW = _IOC_W | _IOC_R

BUF_TYPE_CAPTURE_MPLANE = 9
BUF_TYPE_OUTPUT_MPLANE = 10
MEMORY_MMAP = 1
CAP_VIDEO_M2M_MPLANE = 0x4000
CAP_DEVICE_CAPS = 0x80000000
BUF_FLAG_KEYFRAME = 0x8
PIX_FMT_YUV420 = int.from_bytes(b"YU12", "little")
PIX_FMT_H264 = int.from_bytes(b"H264", "little")

CID_CODEC_BASE = 0x00990900
CID_BITRATE_MODE = CID_CODEC_BASE + 206
CID_BITRATE = CID_CODEC_BASE + 207
CID_REPEAT_SEQ_HEADER = CID_CODEC_BASE + 226
CID_H264_I_PERIOD = CID_CODEC_BASE + 358
CID_H264_LEVEL = CID_CODEC_BASE + 359
CID_H264_PROFILE = CID_CODEC_BASE + 363
BITRATE_MODES = {"vbr": 0, "cbr": 1}
H264_LEVELS = {"4.0": 11, "4.1": 12, "4.2": 13, "5.0": 14, "5.1": 15}
# Macroblocks per second each level allows (H.264 table A-1).  The CM4's
# firmware refuses to start streaming above the level it is given: 1640x1232
# at 83 fps under 4.2 fails STREAMON with ESRCH ("Failed enabling i/p port").
H264_LEVEL_MBPS = {"4.0": 245760, "4.1": 245760, "4.2": 522240, "5.0": 589824, "5.1": 983040}


def pick_level(width, height, fps):
    """The lowest of 4.2 / 5.1 that covers the stream's macroblock rate."""
    mbps = ((width + 15) // 16) * ((height + 15) // 16) * fps
    return "4.2" if mbps <= H264_LEVEL_MBPS["4.2"] else "5.1"


H264_PROFILES = {"baseline": 0, "constrained_baseline": 1, "main": 2, "high": 4}


class Capability(ctypes.Structure):
    _fields_ = [("driver", ctypes.c_char * 16), ("card", ctypes.c_char * 32),
                ("bus_info", ctypes.c_char * 32), ("version", ctypes.c_uint32),
                ("capabilities", ctypes.c_uint32), ("device_caps", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32 * 3)]


class PlanePixFormat(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("sizeimage", ctypes.c_uint32), ("bytesperline", ctypes.c_uint32),
                ("reserved", ctypes.c_uint16 * 6)]


class PixFormatMplane(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
                ("pixelformat", ctypes.c_uint32), ("field", ctypes.c_uint32),
                ("colorspace", ctypes.c_uint32), ("plane_fmt", PlanePixFormat * 8),
                ("num_planes", ctypes.c_uint8), ("flags", ctypes.c_uint8),
                ("ycbcr_enc", ctypes.c_uint8), ("quantization", ctypes.c_uint8),
                ("xfer_func", ctypes.c_uint8), ("reserved", ctypes.c_uint8 * 7)]


class _FormatUnion(ctypes.Union):
    _fields_ = [("pix_mp", PixFormatMplane), ("raw_data", ctypes.c_uint8 * 200),
                ("_align", ctypes.c_uint64)]


class Format(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("fmt", _FormatUnion)]


class RequestBuffers(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("capabilities", ctypes.c_uint32),
                ("flags", ctypes.c_uint8), ("reserved", ctypes.c_uint8 * 3)]


class Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class Timecode(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("frames", ctypes.c_uint8), ("seconds", ctypes.c_uint8),
                ("minutes", ctypes.c_uint8), ("hours", ctypes.c_uint8),
                ("userbits", ctypes.c_uint8 * 4)]


class _PlaneM(ctypes.Union):
    _fields_ = [("mem_offset", ctypes.c_uint32), ("userptr", ctypes.c_ulong),
                ("fd", ctypes.c_int32)]


class Plane(ctypes.Structure):
    _fields_ = [("bytesused", ctypes.c_uint32), ("length", ctypes.c_uint32),
                ("m", _PlaneM), ("data_offset", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32 * 11)]


class _BufferM(ctypes.Union):
    _fields_ = [("offset", ctypes.c_uint32), ("userptr", ctypes.c_ulong),
                ("planes", ctypes.POINTER(Plane)), ("fd", ctypes.c_int32)]


class Buffer(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("bytesused", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("field", ctypes.c_uint32), ("timestamp", Timeval),
                ("timecode", Timecode), ("sequence", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("m", _BufferM),
                ("length", ctypes.c_uint32), ("reserved2", ctypes.c_uint32),
                ("request_fd", ctypes.c_int32)]


class Fract(ctypes.Structure):
    _fields_ = [("numerator", ctypes.c_uint32), ("denominator", ctypes.c_uint32)]


class OutputParm(ctypes.Structure):
    _fields_ = [("capability", ctypes.c_uint32), ("outputmode", ctypes.c_uint32),
                ("timeperframe", Fract), ("extendedmode", ctypes.c_uint32),
                ("writebuffers", ctypes.c_uint32), ("reserved", ctypes.c_uint32 * 4)]


class _ParmUnion(ctypes.Union):
    _fields_ = [("output", OutputParm), ("raw_data", ctypes.c_uint8 * 200)]


class StreamParm(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("parm", _ParmUnion)]


class Control(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("value", ctypes.c_int32)]


# Sizes from the kernel headers (checked with a C program on 2026-10-04).
ABI_SIZES = {Capability: 104, Format: 208, PixFormatMplane: 192, RequestBuffers: 20,
             Buffer: 88, Plane: 64, StreamParm: 204, Control: 8}
for _cls, _size in ABI_SIZES.items():
    if ctypes.sizeof(_cls) != _size:
        raise ImportError(f"V4L2 ABI mismatch: {_cls.__name__} is {ctypes.sizeof(_cls)} bytes, "
                          f"kernel has {_size}")

VIDIOC_QUERYCAP = _ioc(_IOC_R, 0, ctypes.sizeof(Capability))
VIDIOC_S_FMT = _ioc(_IOC_RW, 5, ctypes.sizeof(Format))
VIDIOC_REQBUFS = _ioc(_IOC_RW, 8, ctypes.sizeof(RequestBuffers))
VIDIOC_QUERYBUF = _ioc(_IOC_RW, 9, ctypes.sizeof(Buffer))
VIDIOC_QBUF = _ioc(_IOC_RW, 15, ctypes.sizeof(Buffer))
VIDIOC_DQBUF = _ioc(_IOC_RW, 17, ctypes.sizeof(Buffer))
VIDIOC_STREAMON = _ioc(_IOC_W, 18, ctypes.sizeof(ctypes.c_int))
VIDIOC_STREAMOFF = _ioc(_IOC_W, 19, ctypes.sizeof(ctypes.c_int))
VIDIOC_S_PARM = _ioc(_IOC_RW, 22, ctypes.sizeof(StreamParm))
VIDIOC_S_CTRL = _ioc(_IOC_RW, 28, ctypes.sizeof(Control))


def find_encoder_device():
    """The first /dev/video* whose driver card says it is an H.264 encoder."""
    for path in sorted(Path("/dev").glob("video*"), key=lambda p: (len(p.name), p.name)):
        try:
            fd = os.open(str(path), os.O_RDWR | os.O_NONBLOCK)
        except OSError:
            continue
        try:
            cap = Capability()
            fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap)
            card = cap.card.decode(errors="replace").lower()
            caps = cap.device_caps if cap.capabilities & CAP_DEVICE_CAPS else cap.capabilities
            if "encode" in card and "image" not in card and caps & CAP_VIDEO_M2M_MPLANE:
                return str(path)
        except OSError:
            pass
        finally:
            os.close(fd)
    return None


# ------------------------------------------------------------------ encoder

class V4L2H264Encoder:
    """A stateful V4L2 memory-to-memory H.264 encoder fed grey frames.

    Grey goes in as the Y plane of YUV420 with the chroma held at 128.  Every
    queued frame carries its id in the buffer timestamp, which the driver
    copies to the encoded buffer, so outputs are matched to inputs exactly.
    A reader thread dequeues as soon as the driver signals, and calls
    ``on_output(frame_id, done_ns, nbytes, keyframe, data)``.
    """

    def __init__(self, device, width, height, fps, bitrate, gop, profile="high",
                 level="auto", rate_mode="vbr", out_buffers=2, cap_buffers=4,
                 keep_data=False, on_output=None):
        self.width, self.height = width, height
        if level == "auto":
            level = pick_level(width, height, fps)
        self.on_output = on_output
        self.keep_data = keep_data
        self.fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
        self.maps_out, self.maps_cap = [], []
        self.lock = threading.Lock()
        self.free_out = collections.deque()
        self.stop_event = threading.Event()
        self.thread = None
        self.streaming = False
        cap = Capability()
        fcntl.ioctl(self.fd, VIDIOC_QUERYCAP, cap)
        self.info = {"device": device, "driver": cap.driver.decode(errors="replace"),
                     "card": cap.card.decode(errors="replace")}

        fmt = Format(type=BUF_TYPE_OUTPUT_MPLANE)
        pix = fmt.fmt.pix_mp
        pix.width, pix.height, pix.pixelformat, pix.num_planes = width, height, PIX_FMT_YUV420, 1
        pix.plane_fmt[0].bytesperline = width
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)
        self.stride = int(pix.plane_fmt[0].bytesperline)
        self.buf_height = int(pix.height)
        self.sizeimage = int(pix.plane_fmt[0].sizeimage)
        if pix.pixelformat != PIX_FMT_YUV420 or pix.width < width or self.stride < width:
            raise RuntimeError(f"encoder refused {width}x{height} YU12: got "
                               f"{pix.width}x{pix.height} stride {self.stride}")

        fmt = Format(type=BUF_TYPE_CAPTURE_MPLANE)
        pix = fmt.fmt.pix_mp
        pix.width, pix.height, pix.pixelformat, pix.num_planes = width, height, PIX_FMT_H264, 1
        pix.plane_fmt[0].sizeimage = max(512 * 1024, width * height)
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)
        self.info.update(stride=self.stride, buffer_height=self.buf_height,
                         sizeimage=self.sizeimage,
                         capture_sizeimage=int(pix.plane_fmt[0].sizeimage))

        parm = StreamParm(type=BUF_TYPE_OUTPUT_MPLANE)
        parm.parm.output.timeperframe.numerator = 1000
        parm.parm.output.timeperframe.denominator = int(round(fps * 1000))
        try:
            fcntl.ioctl(self.fd, VIDIOC_S_PARM, parm)
        except OSError as exc:
            self.info["s_parm_error"] = str(exc)

        self.controls = {}
        for name, cid, value in (("bitrate_mode", CID_BITRATE_MODE, BITRATE_MODES[rate_mode]),
                                 ("bitrate", CID_BITRATE, int(bitrate)),
                                 ("i_period", CID_H264_I_PERIOD, int(gop)),
                                 ("repeat_seq_header", CID_REPEAT_SEQ_HEADER, 1),
                                 ("profile", CID_H264_PROFILE, H264_PROFILES[profile]),
                                 ("level", CID_H264_LEVEL, H264_LEVELS[level])):
            try:
                fcntl.ioctl(self.fd, VIDIOC_S_CTRL, Control(id=cid, value=value))
                self.controls[name] = value
            except OSError as exc:
                self.controls[name] = f"refused ({exc.strerror})"

        self.maps_out = self._request(BUF_TYPE_OUTPUT_MPLANE, out_buffers)
        for index, m in enumerate(self.maps_out):
            y_size = self.stride * self.buf_height
            m[y_size:self.sizeimage] = b"\x80" * (self.sizeimage - y_size)
            self.free_out.append(index)
        self.maps_cap = self._request(BUF_TYPE_CAPTURE_MPLANE, cap_buffers)
        for index in range(len(self.maps_cap)):
            self._queue(BUF_TYPE_CAPTURE_MPLANE, index, 0)
        for buf_type in (BUF_TYPE_OUTPUT_MPLANE, BUF_TYPE_CAPTURE_MPLANE):
            fcntl.ioctl(self.fd, VIDIOC_STREAMON, ctypes.c_int(buf_type))
        self.streaming = True
        self.thread = threading.Thread(target=self._reader, daemon=True, name="h264-reader")
        self.thread.start()

    def _request(self, buf_type, count):
        req = RequestBuffers(count=count, type=buf_type, memory=MEMORY_MMAP)
        fcntl.ioctl(self.fd, VIDIOC_REQBUFS, req)
        maps = []
        for index in range(req.count):
            planes = (Plane * 1)()
            buf = Buffer(index=index, type=buf_type, memory=MEMORY_MMAP, length=1)
            buf.m.planes = ctypes.cast(planes, ctypes.POINTER(Plane))
            fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, buf)
            maps.append(mmap.mmap(self.fd, planes[0].length, mmap.MAP_SHARED,
                                  mmap.PROT_READ | mmap.PROT_WRITE,
                                  offset=planes[0].m.mem_offset))
        return maps

    def _queue(self, buf_type, index, bytesused, frame_id=0):
        planes = (Plane * 1)()
        planes[0].bytesused = bytesused
        planes[0].length = len((self.maps_out if buf_type == BUF_TYPE_OUTPUT_MPLANE
                                else self.maps_cap)[index])
        buf = Buffer(index=index, type=buf_type, memory=MEMORY_MMAP, length=1)
        buf.m.planes = ctypes.cast(planes, ctypes.POINTER(Plane))
        buf.timestamp.tv_sec, buf.timestamp.tv_usec = divmod(frame_id, 1_000_000)
        fcntl.ioctl(self.fd, VIDIOC_QBUF, buf)

    def _dequeue(self, buf_type):
        planes = (Plane * 1)()
        buf = Buffer(type=buf_type, memory=MEMORY_MMAP, length=1)
        buf.m.planes = ctypes.cast(planes, ctypes.POINTER(Plane))
        try:
            fcntl.ioctl(self.fd, VIDIOC_DQBUF, buf)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EINVAL, errno.EPIPE):
                return None
            raise
        frame_id = buf.timestamp.tv_sec * 1_000_000 + buf.timestamp.tv_usec
        return buf.index, frame_id, int(planes[0].bytesused), int(buf.flags)

    def submit(self, gray, frame_id):
        """Queue one grey frame; return the queue time, or None if no buffer is free."""
        with self.lock:
            if not self.free_out:
                return None
            index = self.free_out.popleft()
        view = np.frombuffer(self.maps_out[index], np.uint8, self.stride * self.buf_height)
        view.reshape(self.buf_height, self.stride)[:gray.shape[0], :gray.shape[1]] = gray
        queued = boottime_ns()
        self._queue(BUF_TYPE_OUTPUT_MPLANE, index, self.sizeimage, frame_id)
        return queued

    def _reader(self):
        poller = select.poll()
        poller.register(self.fd, select.POLLIN | select.POLLOUT | select.POLLERR | select.POLLPRI)
        while not self.stop_event.is_set():
            events = poller.poll(100)
            got = False
            while True:
                item = self._dequeue(BUF_TYPE_CAPTURE_MPLANE)
                if item is None:
                    break
                got = True
                index, frame_id, nbytes, flags = item
                done = boottime_ns()
                data = bytes(self.maps_cap[index][:nbytes]) if self.keep_data else None
                if self.on_output is not None:
                    self.on_output(frame_id, done, nbytes, bool(flags & BUF_FLAG_KEYFRAME), data)
                self._queue(BUF_TYPE_CAPTURE_MPLANE, index, 0)
            while True:
                item = self._dequeue(BUF_TYPE_OUTPUT_MPLANE)
                if item is None:
                    break
                got = True
                with self.lock:
                    self.free_out.append(item[0])
            if events and not got:
                time.sleep(0.0005)      # POLLERR with nothing queued: do not spin

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.streaming:
            for buf_type in (BUF_TYPE_OUTPUT_MPLANE, BUF_TYPE_CAPTURE_MPLANE):
                try:
                    fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, ctypes.c_int(buf_type))
                except OSError:
                    pass
        for m in self.maps_out + self.maps_cap:
            m.close()
        for buf_type in (BUF_TYPE_OUTPUT_MPLANE, BUF_TYPE_CAPTURE_MPLANE):
            try:
                fcntl.ioctl(self.fd, VIDIOC_REQBUFS,
                            RequestBuffers(count=0, type=buf_type, memory=MEMORY_MMAP))
            except OSError:
                pass
        os.close(self.fd)


# ------------------------------------------------------------------ frames

def parse_case(text):
    """``640x480@30`` / ``full@all`` -> (width, height, rate_hz or 0 for every frame)."""
    size, _, rate = text.partition("@")
    if size == "full":
        width, height = FULL_W, FULL_H
    else:
        width, height = (int(v) for v in size.split("x"))
    hz = 0.0 if rate in ("", "all") else float(rate)
    return width, height, hz


def full_gray(bayer):
    """Raw SBGGR8 -> full-size grey, as the node's ``gray`` full frame does."""
    if _cv2 is not None:
        return _cv2.cvtColor(np.ascontiguousarray(bayer), _cv2.COLOR_BayerBG2GRAY)
    rows, cols = (bayer.shape[0] // 2) * 2, (bayer.shape[1] // 2) * 2
    binned = bayer[0:rows:2, 0:cols:2].astype(np.uint16)
    binned += bayer[1:rows:2, 0:cols:2]
    binned += bayer[0:rows:2, 1:cols:2]
    binned += bayer[1:rows:2, 1:cols:2]
    out = ((binned + 2) >> 2).astype(np.uint8)
    return out.repeat(2, axis=0).repeat(2, axis=1)


def prepare(y, width, height, raw):
    """The grey picture a stream carries: the small stream's reduction, or the
    full frame (demosaiced when the node reads the raw Bayer stream)."""
    if (width, height) == (y.shape[1], y.shape[0]):
        return full_gray(y) if raw else np.ascontiguousarray(y)
    return scale_luma(y, width, height)


def summary(values, scale=1e6):
    if not values:
        return {"n": 0, "p50": None, "p95": None, "max": None, "min": None}
    a = np.asarray(values, dtype=float) / scale
    return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "max": round(float(a.max()), 3),
            "min": round(float(a.min()), 3)}


class CpuMeter:
    """Whole-board and this-process CPU over an interval, from /proc."""

    def __init__(self):
        self.t0 = self._read()

    @staticmethod
    def _read():
        try:
            fields = [int(v) for v in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
            idle, total = fields[3] + fields[4], sum(fields)
        except (OSError, ValueError, IndexError):
            idle = total = 0
        t = os.times()
        return idle, total, t.user + t.system, time.monotonic()

    def percent(self):
        idle0, total0, proc0, wall0 = self.t0
        idle1, total1, proc1, wall1 = self._read()
        board = 100.0 * (1 - (idle1 - idle0) / (total1 - total0)) if total1 > total0 else None
        cores = os.cpu_count() or 1
        proc = 100.0 * (proc1 - proc0) / max(wall1 - wall0, 1e-9) / cores
        return {"board_percent": None if board is None else round(board, 1),
                "probe_percent_of_board": round(proc, 1)}


def soc_state():
    out = {}
    try:
        out["temp_c"] = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        out["temp_c"] = None
    try:
        out["throttled"] = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                                          text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out["throttled"] = None
    return out


# ------------------------------------------------------------------ one case

class CaseRecorder:
    """Collects the encoder's outputs; one output per frame id, bytes summed
    when the driver splits a frame (stream headers before the first one)."""

    def __init__(self, stream_file=None, save_until_ns=None):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.done = {}          # frame_id -> [first_ns, last_ns, bytes, keyframe]
        self.untagged_bytes = 0
        self.stream_file = stream_file
        self.save_until_ns = save_until_ns

    def __call__(self, frame_id, done_ns, nbytes, keyframe, data):
        with self.cond:
            if frame_id == 0:
                self.untagged_bytes += nbytes
            else:
                entry = self.done.setdefault(frame_id, [done_ns, done_ns, 0, False])
                entry[1] = done_ns
                entry[2] += nbytes
                entry[3] = entry[3] or keyframe
            self.cond.notify_all()
        if self.stream_file is not None and data is not None and \
                (self.save_until_ns is None or done_ns <= self.save_until_ns):
            self.stream_file.write(data)

    def wait_for(self, frame_id, timeout_s):
        deadline = time.monotonic() + timeout_s
        with self.cond:
            while frame_id not in self.done:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self.cond.wait(left)
            return self.done[frame_id][1]


def hold_test(encoder, recorder, gray, first_id, count=8, gap_s=0.3):
    """Single frames, each waited for alone: does the encoder release frame k
    before frame k+1 arrives?  Returns per-frame latencies (ms, None = held)."""
    latencies = []
    for i in range(count):
        frame_id = first_id + i
        queued = None
        give_up = time.monotonic() + 1.0
        while queued is None:
            queued = encoder.submit(gray, frame_id)
            if queued is None:
                if time.monotonic() > give_up:
                    raise RuntimeError("the encoder never returned an input buffer")
                time.sleep(0.002)
        done = recorder.wait_for(frame_id, gap_s)
        latencies.append(None if done is None else round((done - queued) / 1e6, 3))
    time.sleep(gap_s)       # let a held last frame show up before the run starts
    released = [v for v in latencies if v is not None]
    return {"latencies_ms": latencies, "released_alone": len(released), "sent": count,
            "holds_a_frame": len(released) < count - 1}


def run_case(sensor, case, a, device, make_encoder=V4L2H264Encoder, first_id=1):
    width, height, hz = parse_case(case)
    rate = hz if hz > 0 else a.sensor_fps
    gop = max(1, int(round(rate * a.gop_s)))
    bitrate = int((a.bitrate_full if (width, height) == (FULL_W, FULL_H) else a.bitrate_small) * 1e6)
    stream_file = None
    save_until = None
    if a.save_dir:
        Path(a.save_dir).mkdir(parents=True, exist_ok=True)
        stream_file = open(Path(a.save_dir) / f"{case.replace('@', '_')}.h264", "wb")
    recorder = CaseRecorder(stream_file)
    result = {"case": case, "width": width, "height": height, "rate_hz": hz or "all",
              "gop_frames": gop, "bitrate_mbit_s": bitrate / 1e6}
    encoder = None
    try:
        encoder = make_encoder(device, width, height, rate, bitrate, gop, profile=a.profile,
                               level=a.level, rate_mode=a.rate_mode,
                               keep_data=stream_file is not None, on_output=recorder)
        result["encoder"] = dict(getattr(encoder, "info", {}), controls=getattr(encoder, "controls", {}))

        frame = sensor.capture()
        sample = prepare(frame.y, width, height, a.raw)
        frame.release()
        result["hold_test"] = hold_test(encoder, recorder, sample, first_id, gap_s=a.hold_gap_s)
        next_id = first_id + 1000
        for _ in range(int(a.sensor_fps / 2)):     # frames left queued during the hold test
            sensor.capture().release()

        rows = []               # (id, exp_start, in_python, prepared, queued)
        counts = {"skipped_busy": 0, "skipped_prep": 0}
        sensor_frames = 0
        seqs = []
        last_tick = None
        # As in the node: the capture thread only copies the frame out and
        # releases it; the reduction (or demosaic) and the encoder run on a
        # worker, so a slow step drops frames instead of queueing the camera.
        handoff = collections.deque(maxlen=1)
        ready = threading.Event()
        stopping = threading.Event()

        def worker():
            fid = next_id
            while not (stopping.is_set() and not handoff):
                if not ready.wait(0.05):
                    continue
                ready.clear()
                try:
                    raw_copy, exp_start, in_python = handoff.popleft()
                except IndexError:
                    continue
                gray = prepare(raw_copy, width, height, a.raw)
                prepared = boottime_ns()
                queued = encoder.submit(gray, fid)
                if queued is None:
                    counts["skipped_busy"] += 1
                    continue
                rows.append((fid, exp_start, in_python, prepared, queued))
                fid += 1

        thread = threading.Thread(target=worker, daemon=True, name="prep")
        thread.start()
        cpu = CpuMeter()
        start = time.monotonic()
        save_until = boottime_ns() + int(a.save_seconds * 1e9)
        recorder.save_until_ns = save_until
        while time.monotonic() - start < a.seconds:
            frame = sensor.capture()
            in_python = boottime_ns()
            sensor_frames += 1
            seqs.append(frame.sequence)
            stamp = frame.sensor_stamp_ns
            if hz > 0:
                tick = stamp // int(1e9 / hz)
                if tick == last_tick:
                    frame.release()
                    continue
                last_tick = tick
            raw_copy = np.array(frame.y, copy=True)
            exp_start = stamp - frame.exposure_ns
            frame.release()
            if handoff:
                counts["skipped_prep"] += 1      # the worker has not taken the last one
            handoff.append((raw_copy, exp_start, in_python))
            ready.set()
        stopping.set()
        ready.set()
        thread.join(timeout=2.0)
        submitted = len(rows)
        skipped_busy = counts["skipped_busy"]
        elapsed = time.monotonic() - start
        cpu_result = cpu.percent()
        time.sleep(0.5)         # the last frames in flight
    finally:
        if encoder is not None:
            encoder.close()
        if stream_file is not None:
            stream_file.close()
    if "hold_test" not in result:
        return result

    done = recorder.done
    platform, prep, enc, total, sizes, key_sizes = [], [], [], [], [], []
    before_next = 0
    for i, (fid, exp_start, in_python, prepared, queued) in enumerate(rows):
        platform.append(in_python - exp_start)
        prep.append(queued - in_python)
        out = done.get(fid)
        if out is None:
            continue
        enc.append(out[1] - queued)
        total.append(out[1] - exp_start)
        (key_sizes if out[3] else sizes).append(out[2])
        if i + 1 < len(rows) and out[1] <= rows[i + 1][4]:
            before_next += 1
    encoded = sum(1 for r in rows if r[0] in done)
    gaps = np.diff(np.asarray(seqs, dtype=np.int64)) if len(seqs) > 1 else np.zeros(0)
    all_sizes = sizes + key_sizes
    result.update({
        "seconds": round(elapsed, 2),
        "sensor_frames": sensor_frames,
        "sensor_missed": int(np.sum(np.clip(gaps - 1, 0, None))),
        "submitted": submitted, "skipped_encoder_busy": skipped_busy,
        "skipped_prep_busy": counts["skipped_prep"],
        "encoded": encoded, "lost_in_encoder": submitted - encoded,
        "input_fps": round(submitted / elapsed, 2), "encoded_fps": round(encoded / elapsed, 2),
        "released_before_next_input": round(before_next / max(len(rows) - 1, 1), 3),
        "latency_ms": {"platform": summary(platform), "prep_and_copy": summary(prep),
                       "encoder": summary(enc), "total_exposure_to_encoded": summary(total)},
        "frame_kbytes": {"p_frames": summary(sizes, 1e3), "keyframes": summary(key_sizes, 1e3)},
        "mbit_s": round(sum(all_sizes) * 8 / elapsed / 1e6, 2),
        "header_bytes_untagged": recorder.untagged_bytes,
        "cpu": cpu_result, "soc": soc_state(),
    })
    if stream_file is not None:
        result["saved_stream"] = stream_file.name
    return result


def print_case(r):
    if "error" in r:
        print(f"\n== {r['case']}\n  ERROR: {r['error']}")
        return
    print(f"\n== {r['case']}  ({r['width']}x{r['height']}, rate {r['rate_hz']}, "
          f"GOP {r['gop_frames']}, {r['bitrate_mbit_s']:g} Mbit/s target)")
    enc = r.get("encoder", {})
    print(f"  encoder          {enc.get('card')} {enc.get('device')}, stride {enc.get('stride')}, "
          f"controls {enc.get('controls')}")
    h = r["hold_test"]
    print(f"  hold test        {h['released_alone']}/{h['sent']} frames released alone "
          f"{h['latencies_ms']} -> {'HOLDS A FRAME' if h['holds_a_frame'] else 'no hold'}")
    print(f"  frames           sensor {r['sensor_frames']} (missed {r['sensor_missed']}), "
          f"prep busy {r['skipped_prep_busy']}, submitted {r['submitted']}, "
          f"encoder busy {r['skipped_encoder_busy']}, "
          f"encoded {r['encoded']}, lost {r['lost_in_encoder']}")
    print(f"  rate             in {r['input_fps']} fps, out {r['encoded_fps']} fps; "
          f"frame k out before k+1 in: {r['released_before_next_input'] * 100:.1f} %")
    for name, s in r["latency_ms"].items():
        if s["n"]:
            print(f"  {name:<16} P50 {s['p50']:.2f}  P95 {s['p95']:.2f}  max {s['max']:.2f} ms")
    fk = r["frame_kbytes"]
    print(f"  size             P-frame P50 {fk['p_frames']['p50']} KB P95 {fk['p_frames']['p95']} KB; "
          f"keyframe P50 {fk['keyframes']['p50']} KB; {r['mbit_s']} Mbit/s")
    print(f"  board            CPU {r['cpu']}, {r['soc']}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="node_config.json (sensor mode, stream: raw on CM4)")
    p.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES),
                   help="WxH@HZ or full@HZ; HZ 'all' = every sensor frame")
    p.add_argument("--seconds", type=float, default=30.0, help="per case")
    p.add_argument("--exposure-us", type=int, help="override the config (3000 for a lit picture)")
    p.add_argument("--gain", type=float)
    p.add_argument("--bitrate-small", type=float, default=10.0, help="Mbit/s target, small frames")
    p.add_argument("--bitrate-full", type=float, default=25.0, help="Mbit/s target, full frames")
    p.add_argument("--rate-mode", choices=sorted(BITRATE_MODES), default="vbr")
    p.add_argument("--gop-s", type=float, default=0.2, help="keyframe period, seconds")
    p.add_argument("--profile", choices=sorted(H264_PROFILES), default="high")
    p.add_argument("--level", choices=["auto"] + sorted(H264_LEVELS), default="auto",
                   help="auto: 4.2, or 5.1 when the stream needs it")
    p.add_argument("--device", help="encoder device; found by its card name if omitted")
    p.add_argument("--save-dir", help="keep the H.264 of each case here (first --save-seconds)")
    p.add_argument("--save-seconds", type=float, default=10.0)
    p.add_argument("--hold-gap-s", type=float, default=0.3,
                   help="hold test: wait for each single frame this long")
    p.add_argument("--synthetic", action="store_true", help="the node's synthetic sensor (dry run)")
    p.add_argument("--output", help="write every case as JSON")
    return p.parse_args(argv)


def main(argv=None, make_encoder=V4L2H264Encoder):
    a = parse_args(argv)
    device = a.device
    if make_encoder is V4L2H264Encoder:
        device = device or find_encoder_device()
        if device is None:
            raise SystemExit("no V4L2 H.264 encoder found (expected bcm2835-codec-encode, "
                             "/dev/video11 on a CM4); pass --device")
    cfg = NodeConfig.load(a.config)
    if a.synthetic:
        cfg.sensor = "synthetic"
        cfg.synthetic_width, cfg.synthetic_height = FULL_W, FULL_H
    a.sensor_fps = float(cfg.synthetic_fps if a.synthetic else cfg.fps)
    a.raw = cfg.stream == "raw" and not a.synthetic
    sensor = make_sensor(cfg)
    sensor.start()
    if a.exposure_us is not None or a.gain is not None:
        sensor.set_controls(exposure_us=a.exposure_us, gain=a.gain)
        for _ in range(20):
            sensor.capture().release()
    results = {"config": {"stream": cfg.stream, "buffer_count": cfg.buffer_count,
                          "exposure_us": a.exposure_us or cfg.exposure_us,
                          "mode": getattr(sensor, "mode", None)},
               "args": vars(a), "cases": []}
    try:
        for n, case in enumerate(a.cases):
            try:
                r = run_case(sensor, case, a, device, make_encoder, first_id=1 + n * 1_000_000)
            except Exception as exc:     # one case failing must not cost the others
                r = {"case": case, "error": f"{type(exc).__name__}: {exc}"}
            results["cases"].append(r)
            print_case(r)
            sys.stdout.flush()
            if a.output:        # after every case, so a later crash keeps the rest
                Path(a.output).write_text(json.dumps(results, indent=2, default=str))
    finally:
        sensor.stop()
    if a.output:
        Path(a.output).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwritten {a.output}")
    return results


if __name__ == "__main__":
    main()
