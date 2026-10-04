"""What the camera node and the gigabit port do to a window, emulated offline.

Replay-only: the live tract receives real windows from ``lan_capture`` and
never imports this.  Three parts, all deterministic:

``SensorPath``   what the node reads off the sensor, applied to a rendered
                 frame: ``cm5`` delivers luminance (the ISP's Y plane),
                 ``cm4`` reads the raw Bayer mosaic (its ISP drops every other
                 frame) and the laptop demosaics it to luminance with the same
                 ``demosaic_bayer_gray`` that ``lan_capture`` uses on hardware.
                 Gazebo renders RGB; the mosaic is built from its luminance
                 with per-channel gains of +/-5 %, the pessimistic end of the
                 0.8 % measured on a CM4 in the dark.
``jpeg_roundtrip`` a window the planner marked JPEG is encoded at q90 and
                 decoded again before the tract sees it; its real size is what
                 goes on the wire.
``LinkModel``    exposure -> arrival: the node's floor plus its work on the
                 windows, then one gigabit port shared by all six cameras and
                 the lidar, windows serialised in the order they are ready.

The numbers are the measured ones in docs/cm4_camera_node_report.md and
docs/handoff_window_size_answer.md; see ``NODE`` below.
"""
from __future__ import annotations

import collections
import time

import numpy as np

from .lan_capture import demosaic_bayer_gray

MODES = ("ideal", "cm5", "cm4")

# Exposure of the first row to the frame in the node's Python, measured:
# CM4 raw stream 22.6 ms, CM5 through the ISP 15.3 ms.
FLOOR_MS = {"cm5": 15.3, "cm4": 22.6}
# The node's own work, per window, as measured on CM5 (two windows per frame
# in the report's table, halved here; JPEG minus the lossless row of the same
# size).  CM4 does the same work 1.88 times slower (two 480 JPEG windows took
# 27.19 ms on CM4 against the CM5-scaled estimate), and before a JPEG it must
# demosaic the window, 0.5 ms for 480x480 on CM4.
JPEG_WORK_MS_CM5 = ((240 * 240, 0.15), (320 * 320, 0.40), (480 * 480, 1.00),
                    (640 * 640, 1.50), (1640 * 1232, 6.00))
CUT_MS_CM5_480 = 0.17             # cutting one 480x480 window out, CM5
CM4_SLOWDOWN = 1.88
CM4_DEMOSAIC_MS_480 = 0.5
PORT_MBPS = 940.0                 # usable gigabit payload, measured overhead 4.6 %
LIDAR_MBPS = 22.5                 # RoboSense Airy, constant
BAYER_GAINS = ((1.05, 1.00), (1.00, 0.95))   # B G / G R, +/-5 %
JPEG_QUALITY = 90


def _interp_work(pixels):
    area = np.array([a for a, _ in JPEG_WORK_MS_CM5], dtype=float)
    work = np.array([w for _, w in JPEG_WORK_MS_CM5], dtype=float)
    if pixels <= area[0]:
        return float(work[0] * pixels / area[0])
    return float(np.interp(pixels, area, work))


def node_work_ms(mode, windows):
    """The node's work on one frame's windows, beyond its floor.

    ``windows``: iterable of (width, height, fmt) with fmt "raw" or "jpeg".
    """
    total = 0.0
    for width, height, fmt in windows:
        pixels = float(width) * float(height)
        work = CUT_MS_CM5_480 * pixels / (480 * 480)
        if fmt == "jpeg":
            work += _interp_work(pixels)
        if mode == "cm4":
            work *= CM4_SLOWDOWN
            if fmt == "jpeg":
                work += CM4_DEMOSAIC_MS_480 * pixels / (480 * 480)
        total += work
    return total


class SensorPath:
    """Rendered RGB frame -> what the tract gets from this node, as 3-channel grey.

    The tract's consumers are built on 3-channel images; the grey plane is
    repeated so every channel carries the same luminance the node delivers.
    """

    def __init__(self, mode, gains=BAYER_GAINS):
        if mode not in ("cm4", "cm5"):
            raise ValueError(f"no sensor path for {mode!r}")
        self.mode = mode
        self.gains = np.asarray(gains, dtype=np.float32).reshape(2, 2)
        self._tile = None

    def gray(self, rgb):
        import cv2
        luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.mode == "cm5":
            return luma
        height, width = luma.shape
        if self._tile is None or self._tile.shape != luma.shape:
            self._tile = np.tile(self.gains, ((height + 1) // 2, (width + 1) // 2))[:height, :width]
        mosaic = np.clip(luma.astype(np.float32) * self._tile + 0.5, 0, 255).astype(np.uint8)
        return demosaic_bayer_gray(mosaic)

    def __call__(self, rgb):
        import cv2
        return cv2.cvtColor(self.gray(rgb), cv2.COLOR_GRAY2RGB)

    def color(self, rgb):
        """Rendered RGB -> the node's colour frame (RGB), for a colour stream.

        ``cm4``: each site of the SBGGR mosaic keeps only its own channel of
        the render, times the same +/-5 % gains (here a white-balance error),
        and the frame is demosaiced to colour -- not grey -- as the node's
        ISP-less path would before a colour encoder.  ``cm5``: the ISP's
        colour output, the render as is.
        """
        import cv2
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if self.mode == "cm5":
            return rgb
        height, width = rgb.shape[:2]
        if self._tile is None or self._tile.shape != (height, width):
            self._tile = np.tile(self.gains, ((height + 1) // 2, (width + 1) // 2))[:height, :width]
        mosaic = np.empty((height, width), np.float32)
        mosaic[0::2, 0::2] = rgb[0::2, 0::2, 2]          # B
        mosaic[0::2, 1::2] = rgb[0::2, 1::2, 1]          # G
        mosaic[1::2, 0::2] = rgb[1::2, 0::2, 1]          # G
        mosaic[1::2, 1::2] = rgb[1::2, 1::2, 0]          # R
        mosaic = np.clip(mosaic * self._tile + 0.5, 0, 255).astype(np.uint8)
        # OpenCV names the pattern one site off: a BGGR mosaic to RGB is its
        # ``BayerRG2RGB`` (the same operation as ``BayerBG2BGR``).
        return cv2.cvtColor(mosaic, cv2.COLOR_BayerRG2RGB)


def clip_roi(roi, width, height):
    """(x, y, w, h) clipped to the frame, or None when nothing is left."""
    x, y, w, h = (int(v) for v in roi)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def jpeg_roundtrip(image, roi, quality=JPEG_QUALITY):
    """Encode the grey window at ``roi`` as JPEG and write the decoded pixels back.

    Returns (bytes, decode_ms); ``image`` (3-channel grey) is changed in place.
    Only the decode is timed: the encode is the node's, in ``node_work_ms``.
    """
    import cv2
    height, width = image.shape[:2]
    clipped = clip_roi(roi, width, height)
    if clipped is None:
        return 0, 0.0
    x, y, w, h = clipped
    window = np.ascontiguousarray(image[y:y + h, x:x + w, 0])
    ok, data = cv2.imencode(".jpg", window, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    begin = time.perf_counter_ns()
    decoded = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    decode_ms = (time.perf_counter_ns() - begin) / 1e6
    image[y:y + h, x:x + w] = decoded[:, :, None]
    return int(data.size), decode_ms


class LinkModel:
    """Exposure -> arrival for every window, over one shared gigabit port.

    ``send(stamp_ns, camera_id, windows)`` takes one camera's windows for one
    frame, each (width, height, fmt, bytes), and returns the arrival time of
    the last of them on the laptop: the node is ready at stamp + floor + its
    work, then the port serialises everything ready in order (ties by camera
    name), at the port rate less the lidar's constant share.  Calls must come
    in nondecreasing ``stamp_ns``; within one stamp, in any order -- they are
    queued by ready time before the port serves them.
    """

    def __init__(self, mode, port_mbps=PORT_MBPS, lidar_mbps=LIDAR_MBPS, bin_ms=100.0):
        if mode not in FLOOR_MS:
            raise ValueError(f"no link model for {mode!r}")
        self.mode = mode
        self.floor_ns = int(round(FLOOR_MS[mode] * 1e6))
        self.lidar_mbps = float(lidar_mbps)
        self.camera_bps = (float(port_mbps) - self.lidar_mbps) * 1e6
        self.bin_ns = int(bin_ms * 1e6)
        self.port_free_ns = 0
        self.bins = collections.Counter()       # bin index -> camera bytes
        self.first_bin = None
        self.last_bin = None
        self.counts = collections.Counter()

    def ready_ns(self, stamp_ns, windows):
        work = node_work_ms(self.mode, [(w, h, f) for w, h, f, _ in windows])
        return int(stamp_ns) + self.floor_ns + int(round(work * 1e6))

    def serve(self, jobs):
        """``jobs``: list of (camera_id, stamp_ns, windows) of one instant.

        Returns {camera_id: (ready_ns, arrival_ns)}; cameras with no windows
        send nothing and are absent.
        """
        queue = sorted(((self.ready_ns(stamp, windows), cid, stamp, windows)
                        for cid, stamp, windows in jobs if windows),
                       key=lambda item: (item[0], item[1]))
        out = {}
        for ready, cid, stamp, windows in queue:
            start = max(ready, self.port_free_ns)
            size = sum(int(b) for _, _, _, b in windows)
            finish = start + int(round(size * 8 / self.camera_bps * 1e9))
            self.port_free_ns = finish
            self._account(start, finish, size)
            for width, height, fmt, _ in windows:
                self.counts[f"windows_{fmt}"] += 1
            out[cid] = (ready, finish)
        return out

    def _account(self, start, finish, size):
        """Spread a transfer's bytes over the 100 ms bins it spans."""
        if finish <= start:
            finish = start + 1
        first, last = start // self.bin_ns, (finish - 1) // self.bin_ns
        for b in range(first, last + 1):
            lo = max(start, b * self.bin_ns)
            hi = min(finish, (b + 1) * self.bin_ns)
            self.bins[b] += size * (hi - lo) / (finish - start)
        self.first_bin = first if self.first_bin is None else min(self.first_bin, first)
        self.last_bin = last if self.last_bin is None else max(self.last_bin, last)

    def series_mbps(self, start_ns=None, end_ns=None):
        """Port load per bin in Mbit/s, lidar included, over [start, end)."""
        if self.first_bin is None:
            return []
        first = self.first_bin if start_ns is None else int(start_ns) // self.bin_ns
        last = self.last_bin if end_ns is None else (int(end_ns) - 1) // self.bin_ns
        seconds = self.bin_ns / 1e9
        return [self.bins.get(b, 0.0) * 8 / seconds / 1e6 + self.lidar_mbps
                for b in range(first, last + 1)]

    def report(self, start_ns=None, end_ns=None):
        series = self.series_mbps(start_ns, end_ns)
        array = np.asarray(series, dtype=float) if series else np.zeros(1)
        return {
            "mode": self.mode, "floor_ms": self.floor_ns / 1e6, "port_mbps": PORT_MBPS,
            "lidar_mbps": self.lidar_mbps, "bin_ms": self.bin_ns / 1e6,
            "mbps_p50": float(np.percentile(array, 50)),
            "mbps_p95": float(np.percentile(array, 95)),
            "mbps_max": float(array.max()),
            "fraction_above_900_mbps": float((array > 900.0).mean()),
            "series_mbps": [round(v, 2) for v in series],
            **dict(self.counts),
        }
