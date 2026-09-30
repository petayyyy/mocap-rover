import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from localization_contracts import link_emulation, roi_tracker  # noqa: E402
from localization_contracts.camera_model import CameraModel  # noqa: E402
from localization_contracts.lan_capture import demosaic_bayer_gray  # noqa: E402


class Demosaic(unittest.TestCase):
    def test_a_gain_checkerboard_is_removed(self):
        rng = np.random.default_rng(1)
        flat = np.full((64, 96), 120.0)
        mosaic = np.clip(flat * np.tile([[1.05, 1.0], [1.0, 0.95]], (32, 48))
                         + rng.normal(0, 0.3, flat.shape), 0, 255).astype(np.uint8)
        gray = demosaic_bayer_gray(mosaic)
        self.assertEqual(gray.shape, mosaic.shape)
        inner = gray[4:-4, 4:-4].astype(int)
        self.assertLess(np.abs(np.diff(inner, axis=1)).mean(), 1.0)
        self.assertGreater(np.abs(np.diff(mosaic[4:-4, 4:-4].astype(int), axis=1)).mean(), 4.0)
        self.assertLess(abs(inner.mean() - 120), 3)

    def test_an_odd_offset_keeps_the_phase(self):
        mosaic = (np.tile([[200, 100], [100, 50]], (16, 16))).astype(np.uint8)
        whole = demosaic_bayer_gray(mosaic)
        shifted = demosaic_bayer_gray(mosaic[1:, 1:], row0=1, col0=1)
        right = np.abs(whole[5:-5, 5:-5].astype(int) - shifted[4:-5, 4:-5])
        wrong = np.abs(whole[5:-5, 5:-5].astype(int) - demosaic_bayer_gray(mosaic[1:, 1:])[4:-5, 4:-5])
        self.assertLessEqual(right.max(), 1)
        self.assertGreater(wrong.mean(), 10 * max(right.mean(), 0.1))


class SensorAndJpeg(unittest.TestCase):
    def test_cm5_is_luminance_and_cm4_is_close_to_it(self):
        rng = np.random.default_rng(2)
        rgb = rng.integers(40, 200, (48, 64, 3), dtype=np.uint8)
        rgb[:] = rgb[:1, :1]                       # flat colour patch
        y = link_emulation.SensorPath("cm5")(rgb)
        self.assertEqual(y.shape, rgb.shape)
        self.assertTrue((y[..., 0] == y[..., 1]).all())
        bayer = link_emulation.SensorPath("cm4")(rgb)
        self.assertLess(np.abs(bayer[4:-4, 4:-4, 0].astype(int) - y[4:-4, 4:-4, 0]).max(), 4)

    def test_jpeg_changes_only_the_window_and_reports_its_size(self):
        rng = np.random.default_rng(3)
        gray = rng.integers(0, 255, (120, 160), dtype=np.uint8)
        image = np.repeat(gray[:, :, None], 3, axis=2)
        before = image.copy()
        size, decode_ms = link_emulation.jpeg_roundtrip(image, (20, 30, 64, 48))
        self.assertGreater(size, 0)
        self.assertGreaterEqual(decode_ms, 0.0)
        outside = np.ones(image.shape[:2], bool)
        outside[30:78, 20:84] = False
        np.testing.assert_array_equal(image[outside], before[outside])
        self.assertFalse((image[30:78, 20:84] == before[30:78, 20:84]).all())
        self.assertTrue((image[..., 0] == image[..., 2]).all())


class LinkModelTest(unittest.TestCase):
    def test_node_work_grows_with_window_and_jpeg_and_on_cm4(self):
        raw = link_emulation.node_work_ms("cm5", [(320, 320, "raw")])
        jpeg = link_emulation.node_work_ms("cm5", [(320, 320, "jpeg")])
        big = link_emulation.node_work_ms("cm5", [(640, 640, "jpeg")])
        slow = link_emulation.node_work_ms("cm4", [(320, 320, "jpeg")])
        self.assertLess(raw, jpeg)
        self.assertLess(jpeg, big)
        self.assertGreater(slow, 1.88 * jpeg)
        # The measured table: two 480 JPEG windows on CM5 cost 2 ms over the floor.
        two = link_emulation.node_work_ms("cm5", [(480, 480, "jpeg")] * 2)
        self.assertAlmostEqual(two, 2.0 + 2 * 0.17, places=2)

    def test_one_window_arrives_after_floor_work_and_wire(self):
        link = link_emulation.LinkModel("cm4")
        windows = [(320, 320, "raw", 320 * 320)]
        served = link.serve([("camera_1", 1_000_000_000, windows)])
        ready, arrival = served["camera_1"]
        work = link_emulation.node_work_ms("cm4", [(320, 320, "raw")])
        self.assertEqual(ready, 1_000_000_000 + 22_600_000 + round(work * 1e6))
        wire_ns = 320 * 320 * 8 / ((940 - 22.5) * 1e6) * 1e9
        self.assertAlmostEqual(arrival - ready, wire_ns, delta=2)

    def test_the_port_serialises_cameras_in_ready_order(self):
        link = link_emulation.LinkModel("cm5")
        full = [(1640, 1232, "raw", 1640 * 1232)]
        served = link.serve([("camera_2", 0, full), ("camera_1", 0, full), ("camera_3", 0, [])])
        self.assertNotIn("camera_3", served)
        wire = 1640 * 1232 * 8 / ((940 - 22.5) * 1e6) * 1e9
        self.assertAlmostEqual(served["camera_2"][1] - served["camera_1"][1], wire, delta=2)
        self.assertLess(served["camera_1"][1], served["camera_2"][1])

    def test_the_load_series_counts_bytes_and_the_lidar(self):
        link = link_emulation.LinkModel("cm5")
        for k in range(10):                     # 1 s of one 100 kB window every 100 ms
            link.serve([("camera_1", k * 100_000_000, [(100, 1000, "raw", 100_000)])])
        report = link.report(0, 1_000_000_000)
        self.assertEqual(len(report["series_mbps"]), 10)
        self.assertAlmostEqual(report["mbps_p50"], 8.0 + 22.5, places=3)
        self.assertEqual(report["fraction_above_900_mbps"], 0.0)
        self.assertEqual(report["windows_raw"], 10)


def _camera():
    fx = 734.5
    return {"name": "camera_1", "K": [fx, 0, 820, 0, fx, 616, 0, 0, 1], "D": [0, 0, 0, 0],
            "distortion_model": "fisheye", "image_size": [1640, 1232],
            "R_world_optical": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            "position_world": [6.0, 6.0, 2.9]}


class PlannerFormat(unittest.TestCase):
    def planner(self, **kw):
        cam = _camera()
        return roi_tracker.CameraRoiPlanner(
            CameraModel.from_config(cam), cam["R_world_optical"], cam["position_world"],
            max_roi_px=320, exhausted_full_frame_period_s=0.25, **kw)

    def test_without_a_jpeg_cap_the_window_is_clamped_raw(self):
        plan = self.planner().plan((6.0, 6.0, 0.3), 0)
        self.assertEqual((plan.mode, plan.fmt, plan.roi[2]), ("roi", "raw", 320))

    def test_above_the_raw_cap_the_window_goes_jpeg_then_full_frame(self):
        p = self.planner(jpeg_max_roi_px=640)
        small = p.plan((6.0, 6.0, 0.01), 0)
        self.assertEqual((small.mode, small.fmt), ("roi", "raw"))
        mid = p.plan((6.0, 6.0, 0.15), 0)
        self.assertEqual((mid.mode, mid.fmt), ("roi", "jpeg"))
        self.assertTrue(320 < mid.roi[2] <= 640)
        huge = p.plan((6.0, 6.0, 1.0), 1_000_000_000)
        self.assertEqual((huge.mode, huge.reason), ("acquire", "roi_over_jpeg_cap"))
        between = p.plan((6.0, 6.0, 1.0), 1_100_000_000)
        self.assertEqual((between.mode, between.fmt, between.roi[2]), ("roi", "jpeg", 640))
        again = p.plan((6.0, 6.0, 1.0), 1_250_000_000)
        self.assertEqual(again.mode, "acquire")

    def test_acquire_rate_without_a_track(self):
        p = self.planner(acquire_period_s=0.1)
        modes = [p.plan(None, k * 12_000_000).mode for k in range(20)]
        self.assertEqual(modes.count("acquire"), 3)       # at 0, 108 and 216 ms


class CoverageReport(unittest.TestCase):
    def test_could_and_accepted_are_counted_per_instant(self):
        import coverage_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, replay = root / "dataset", root / "replay"
            dataset.mkdir()
            replay.mkdir()
            cam = _camera()
            (replay / "initial_calibration.json").write_text(json.dumps({"cameras": [cam]}))
            (replay / "status.json").write_text(json.dumps({"cameras": {"camera_1": {}}}))
            stamps = [k * 12_000_000 for k in range(10)]
            (dataset / "camera_1.jsonl").write_text(
                "".join(json.dumps({"index": k, "stamp_ns": s}) + "\n" for k, s in enumerate(stamps)))
            rows = []
            for k in range(-1, 12):
                t = k * 12_000_000
                # tag_rover under the camera, then far off beyond the incidence gate
                rows.append({"object_id": "tag_rover", "stamp_ns": t,
                             "x": 6.0 if k < 5 else 20.0, "y": 6.0, "z": 0.14, "yaw": 0})
                rows.append({"object_id": "opponent", "stamp_ns": t, "x": 7.0, "y": 6.0,
                             "z": 0.14, "yaw": 0})
            (dataset / "truth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            obs = [{"accepted": True, "observation": {"object_id": "tag_rover",
                                                       "camera_id": "camera_1",
                                                       "capture_time_ns": s}} for s in stamps[:3]]
            obs.append({"accepted": False, "observation": {"object_id": "opponent",
                                                            "camera_id": "camera_1",
                                                            "capture_time_ns": stamps[0]}})
            (replay / "observations.jsonl").write_text("".join(json.dumps(o) + "\n" for o in obs))
            report = coverage_report.coverage(replay, dataset)
        self.assertEqual(report["instants"], 10)
        self.assertAlmostEqual(report["marker"]["could"], 0.5)
        self.assertAlmostEqual(report["marker"]["accepted"], 0.3)
        self.assertAlmostEqual(report["marker"]["accepted_when_could"], 0.6)
        self.assertAlmostEqual(report["silhouette"]["could"], 1.0)
        self.assertAlmostEqual(report["silhouette"]["accepted"], 0.0)


if __name__ == "__main__":
    unittest.main()
