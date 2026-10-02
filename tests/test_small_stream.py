"""The node's small stream: the whole frame reduced on the node, beside the windows.

A real ``CameraNode`` with the synthetic sensor (pixel (r, c) of frame ``seq``
is ``(7r + 3c + seq) & 0xFF``) serves windows and a small stream at once; the
receiver must keep the two apart and the small frames must be the reduced
sensor frame of the stamp they carry.
"""
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localization_contracts import lan_capture  # noqa: E402
from pi_cam import camera_node, lan_protocol as proto  # noqa: E402
from pi_cam.camera_node import CameraNode, NodeConfig  # noqa: E402

WIDTH, HEIGHT = 328, 246
SMALL_W, SMALL_H = 128, 96


def start_node(camera_id="cam_s"):
    cfg = NodeConfig(camera_id=camera_id, port=0, sensor="synthetic",
                     synthetic_width=WIDTH, synthetic_height=HEIGHT, synthetic_fps=100.0,
                     status_period_s=0.2, ptp_enabled=False)
    node = CameraNode(cfg)
    node.start()
    return node


class ScaleLuma(unittest.TestCase):
    def test_flat_frame_stays_flat_with_either_backend(self):
        y = np.full((1232, 1640), 117, np.uint8)
        saved = camera_node._cv2
        try:
            for backend in (saved, None):
                camera_node._cv2 = backend
                for size in ((640, 480), (820, 616)):
                    out = camera_node.scale_luma(y, *size)
                    self.assertEqual(out.shape, size[::-1])
                    self.assertEqual(out.dtype, np.uint8)
                    self.assertTrue((out == 117).all())
        finally:
            camera_node._cv2 = saved

    def test_bayer_cell_becomes_its_mean(self):
        # A BGGR mosaic with B=40, G=100, R=200: every 2x2 cell averages to 110.
        y = np.empty((1232, 1640), np.uint8)
        y[0::2, 0::2], y[0::2, 1::2], y[1::2, 0::2], y[1::2, 1::2] = 40, 100, 100, 200
        out = camera_node.scale_luma(y, 640, 480)
        self.assertLessEqual(int(np.abs(out.astype(int) - 110).max()), 1)

    def test_reduction_is_fast(self):
        y = np.random.default_rng(0).integers(0, 255, (1232, 1640), dtype=np.uint8)
        camera_node.scale_luma(y, 640, 480)
        begin = time.perf_counter()
        for _ in range(10):
            camera_node.scale_luma(y, 640, 480)
        self.assertLess((time.perf_counter() - begin) / 10, 0.015)


class ProtocolFlag(unittest.TestCase):
    def test_scaled_flag_roundtrips_and_defaults_off(self):
        header = proto.FrameHeader("camera_1", 5, 6, 7, 8, 9, 0, 0, 640, 480, proto.FORMAT_Y8,
                                   0, 1, 1640, 1232, 3, 10, flags=proto.FLAG_SCALED)
        back = proto.FrameHeader.unpack(header.pack())
        self.assertTrue(back.scaled)
        self.assertEqual(back, header)
        plain = proto.FrameHeader("camera_1", 5, 6, 7, 8, 9, 0, 0, 640, 480, proto.FORMAT_Y8,
                                  0, 1, 1640, 1232, 3, 10)
        self.assertFalse(proto.FrameHeader.unpack(plain.pack()).scaled)


class NodeSmallStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = start_node()
        cls.source = lan_capture.LanCameraSource([("127.0.0.1", cls.node.port)])
        assert cls.source.wait_connected(5.0) == ["cam_s"]

    @classmethod
    def tearDownClass(cls):
        cls.source.close()
        cls.node.stop()

    def tearDown(self):
        self.source.stream_small("cam_s", hz=0)
        self.source.request_windows("cam_s", [])

    def small_frames(self, seconds):
        frames, deadline = [], time.monotonic() + seconds
        while time.monotonic() < deadline:
            frame = self.source.take_small("cam_s", timeout=0.05)
            if frame is not None:
                frames.append(frame)
        return frames

    def test_small_frames_arrive_at_the_rate_reduced_and_apart_from_windows(self):
        self.source.request_windows("cam_s", [(10, 20, 64, 48)])
        ack = self.source.stream_small("cam_s", SMALL_W, SMALL_H, 25.0)
        self.assertTrue(ack["ok"], ack)
        self.small_frames(0.3)                       # let the stream settle
        frames = self.small_frames(1.2)
        rate = len(frames) / 1.2
        self.assertGreater(rate, 18.0)
        self.assertLess(rate, 32.0)
        for f in frames:
            self.assertTrue(f.scaled)
            self.assertFalse(f.is_full)
            self.assertEqual((f.width, f.height), (SMALL_W, SMALL_H))
            self.assertEqual((f.sensor_width, f.sensor_height), (WIDTH, HEIGHT))
            self.assertEqual(f.window_count, 1)
            rows = np.arange(HEIGHT)[:, None]
            cols = np.arange(WIDTH)[None, :]
            full = ((7 * rows + 3 * cols + f.frame_seq) & 0xFF).astype(np.uint8)
            np.testing.assert_array_equal(f.array, camera_node.scale_luma(full, SMALL_W, SMALL_H))
        # The windows still come as groups of their own, never with a small frame in them.
        group = self.source.take("cam_s", 1.0)
        self.assertIsNotNone(group)
        self.assertTrue(all(not w.scaled and (w.width, w.height) == (64, 48) for w in group))
        status = self.source.request_status("cam_s") and self.wait_status()
        self.assertGreater(status["small_sent"], 0)
        self.assertIsNotNone(status["small_scale_ms"]["p50"])

    def wait_status(self):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            status = self.source.status("cam_s")
            if status and status.get("small_sent"):
                return status
            time.sleep(0.05)
        return self.source.status("cam_s")

    def test_phase_places_frames_on_the_common_grid(self):
        period = 40_000_000
        phase = 13_000_000
        self.source.stream_small("cam_s", SMALL_W, SMALL_H, 1e9 / period, phase)
        self.small_frames(0.3)
        frames = self.small_frames(1.0)
        self.assertGreater(len(frames), 10)
        ticks = [(f.stamp_ns - phase) // period for f in frames]
        # One frame per period, the first sensor frame at or after the phase.
        self.assertEqual(len(set(ticks)), len(ticks))
        offsets = [(f.stamp_ns - phase) % period for f in frames]
        self.assertLess(max(offsets), 10_000_000 + 2_000_000)   # within one 100 fps frame

    def test_bad_requests_are_refused(self):
        too_big = self.source.stream_small("cam_s", WIDTH, HEIGHT, 30.0)
        self.assertFalse(too_big["ok"])
        bad_format = self.source.stream_small("cam_s", 64, 48, 30.0, fmt="bayer8")
        self.assertFalse(bad_format["ok"])
        stop = self.source.stream_small("cam_s", hz=0)
        self.assertTrue(stop["ok"])


if __name__ == "__main__":
    unittest.main()
