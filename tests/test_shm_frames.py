"""The shared-memory frame reader against a ring written the way cpp/rx/shm_ring.cpp does."""
from __future__ import annotations

import struct

import numpy as np

from localization_contracts.shm_frames import (
    HEADER, HEADER_BYTES, MAGIC, SLOT, SLOT_HEADER_BYTES, ShmFrameReader)


def write_ring(path, frames, slots=4, max_w=64, max_h=48, torn_last=False):
    stride = (SLOT_HEADER_BYTES + max_w * max_h * 3 // 2 + 4095) // 4096 * 4096
    buf = bytearray(HEADER_BYTES + stride * slots)
    for n, (seq, y, uv) in enumerate(frames):
        base = HEADER_BYTES + stride * (n % slots)
        h, w = y.shape
        lock = 2 * (n // slots + 1) + (1 if torn_last and n == len(frames) - 1 else 0)
        SLOT.pack_into(buf, base, lock, seq, 1000 + seq, 900 + seq, 800_000, 9452, w, h, 1640, 1232,
                       int(uv is not None), 0, 5, 6, 7)
        px = base + SLOT_HEADER_BYTES
        buf[px:px + w * h] = y.tobytes()
        if uv is not None:
            c = max_w * max_h // 4
            buf[px + max_w * max_h:px + max_w * max_h + uv[0].size] = uv[0].tobytes()
            buf[px + max_w * max_h + c:px + max_w * max_h + c + uv[1].size] = uv[1].tobytes()
    HEADER.pack_into(buf, 0, MAGIC, 1, slots, max_w, max_h, 0, stride, len(frames), b"camera_9")
    path.write_bytes(bytes(buf))


def test_newest_frame_grey_and_colour(tmp_path):
    rng = np.random.default_rng(0)
    y1 = rng.integers(0, 255, (48, 64), np.uint8)
    y2 = rng.integers(0, 255, (24, 32), np.uint8)
    u, v = rng.integers(0, 255, (12, 16), np.uint8), rng.integers(0, 255, (12, 16), np.uint8)
    write_ring(tmp_path / "mocap_camera_9", [(10, y1, None), (11, y2, (u, v))])
    f = ShmFrameReader("camera_9", root=str(tmp_path)).latest()
    assert f.frame_seq == 11 and f.stamp_ns == 1011 and f.color
    assert (f.width, f.height) == (32, 24)
    np.testing.assert_array_equal(f.y, y2)
    np.testing.assert_array_equal(f.u, u)
    np.testing.assert_array_equal(f.v, v)
    assert f.bgr().shape == (24, 32, 3)


def test_slot_being_written_is_not_returned(tmp_path):
    y = np.zeros((48, 64), np.uint8)
    write_ring(tmp_path / "mocap_camera_9", [(1, y, None)], torn_last=True)
    assert ShmFrameReader("camera_9", root=str(tmp_path)).latest() is None


def test_no_writer_yet(tmp_path):
    assert ShmFrameReader("camera_9", root=str(tmp_path)).latest() is None
