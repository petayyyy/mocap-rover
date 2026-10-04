"""Read decoded camera frames that the C++ receiver (cpp/rx, ``mocap_rx --shm``)
publishes in shared memory, /dev/shm/mocap_<camera_id>.

Layout (cpp/rx/shm_ring.hpp): a 4096-byte header, then ``slots`` slots of
``slot_stride`` bytes, each a 128-byte slot header followed by Y
(max_width * max_height) and, for colour, U and V (a quarter of that each).
The writer fills slot ``write_count % slots`` under a sequence lock (odd while
writing) and then bumps ``write_count``; a reader copies the newest slot and
keeps it only if the lock did not move meanwhile.

    reader = ShmFrameReader("camera_1")
    frame = reader.latest()          # None until the first frame
    frame.y                          # (height, width) uint8, a copy
"""
from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 0x4853434D
HEADER = struct.Struct("<IIIIIIQQ32s")
SLOT = struct.Struct("<QQqqIIIIIIIIqqq")
HEADER_BYTES = 4096
SLOT_HEADER_BYTES = 128


@dataclass
class ShmFrame:
    camera_id: str
    frame_seq: int
    stamp_ns: int            # start of exposure of row 0, this host's REALTIME
    node_stamp_ns: int
    exposure_ns: int
    line_time_ns: int
    width: int
    height: int
    sensor_width: int
    sensor_height: int
    color: bool
    keyframe: bool
    receive_ns: int
    decoded_ns: int
    published_ns: int
    y: np.ndarray
    u: np.ndarray | None = None
    v: np.ndarray | None = None

    def bgr(self) -> np.ndarray:
        """The picture as BGR (grey replicated when the stream is grey)."""
        import cv2
        if not self.color:
            return cv2.cvtColor(self.y, cv2.COLOR_GRAY2BGR)
        i420 = np.concatenate([self.y.ravel(), self.u.ravel(), self.v.ravel()])
        return cv2.cvtColor(i420.reshape(self.height * 3 // 2, self.width), cv2.COLOR_YUV2BGR_I420)


class ShmFrameReader:
    def __init__(self, camera_id: str, root: str = "/dev/shm"):
        self.camera_id = camera_id
        self.path = Path(root) / f"mocap_{camera_id}"
        self._mm = None
        self.last_seq = None

    def _open(self):
        if self._mm is None:
            if not self.path.exists():
                return False
            with open(self.path, "rb") as f:
                self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, slots, max_w, max_h, _, stride, _, _ = HEADER.unpack_from(self._mm, 0)
        if magic != MAGIC:
            return False
        self.slots, self.max_w, self.max_h, self.stride = slots, max_w, max_h, stride
        return True

    def latest(self, retries: int = 4) -> ShmFrame | None:
        """The newest complete frame, or None (no writer yet, or always torn)."""
        if not self._open():
            return None
        for _ in range(retries):
            write_count = HEADER.unpack_from(self._mm, 0)[7]
            if write_count == 0:
                return None
            base = HEADER_BYTES + self.stride * ((write_count - 1) % self.slots)
            lock0 = struct.unpack_from("<Q", self._mm, base)[0]
            if lock0 & 1:
                continue
            (_, seq, stamp, node_stamp, exp, line, w, h, sw, sh, color, key,
             rx, dec, pub) = SLOT.unpack_from(self._mm, base)
            px = base + SLOT_HEADER_BYTES
            y = np.frombuffer(self._mm, np.uint8, w * h, px).reshape(h, w).copy()
            u = v = None
            if color:
                c = self.max_w * self.max_h // 4
                off = px + self.max_w * self.max_h
                u = np.frombuffer(self._mm, np.uint8, (w // 2) * (h // 2), off).reshape(h // 2, w // 2).copy()
                v = np.frombuffer(self._mm, np.uint8, (w // 2) * (h // 2), off + c).reshape(h // 2, w // 2).copy()
            if struct.unpack_from("<Q", self._mm, base)[0] != lock0:
                continue                      # the writer came back meanwhile: torn copy
            self.last_seq = seq
            return ShmFrame(self.camera_id, seq, stamp, node_stamp, exp, line, w, h, sw, sh, bool(color),
                            bool(key), rx, dec, pub, y, u, v)
        return None

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
