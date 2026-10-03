"""ChArUco calibration: fisheye intrinsics at three sizes and the joint camera solve.

Synthetic 1640x1232 views with known answers (tools/charuco_synthetic.py):
a hand board through a fisheye lens with distortion, and the floor strip seen
by the cameras of cameras_ground_truth.json (mount errors included).
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localization_contracts import charuco_calibration as cc  # noqa: E402
from tools.charuco_strip import Placement  # noqa: E402
from tools.charuco_synthetic import HandBoardRenderer, StripRenderer  # noqa: E402

TRUE_K = np.array([[742.0, 0.0, 818.3], [0.0, 741.2, 620.9], [0.0, 0.0, 1.0]])
TRUE_D = np.array([0.045, -0.012, 0.004, -0.0011])
TRUTH = json.loads((ROOT / "config/mocap_arena_imx219/cameras_ground_truth.json").read_text())


class PlacementGeometry(unittest.TestCase):
    def test_pose_matches_the_strip_tool_and_inverts(self):
        target = cc.strip_target()
        border = target.border_m
        for x0, y0, yaw in ((0.0, 7.5, 0.0), (1.2, 4.0, 7.0), (-0.3, 9.1, -12.0)):
            r, t = cc.placement_pose(x0, y0, math.radians(yaw), border)
            ids = np.array([0, 13, 77, 144])
            board_m = target.object_points(ids)
            ours = board_m @ r.T + t
            tool = Placement(x0, y0, yaw, border * 1000).board_to_arena(board_m[:, :2] * 1000)
            np.testing.assert_allclose(ours, tool, atol=1e-9)
            back = cc.placement_from_pose(r, t, border)
            np.testing.assert_allclose(back, (x0, y0, math.radians(yaw)), atol=1e-12)

    def test_scaled_intrinsics_keep_pixel_centres(self):
        k = cc.scale_intrinsics(TRUE_K, (1640, 1232), (820, 616))
        self.assertAlmostEqual(k[0, 0], 371.0)
        self.assertAlmostEqual(k[0, 2], (818.3 + 0.5) / 2 - 0.5)

    def test_field_of_view_of_an_equidistant_lens(self):
        k = cc.initial_fisheye_k((1640, 1232), 160.0)
        fov = cc.field_of_view_deg(k, np.zeros(4), (1640, 1232))
        self.assertAlmostEqual(fov["diagonal"], 160.0, delta=0.2)


class Intrinsics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = cc.CharucoTarget(legacy="auto")
        renderer = HandBoardRenderer(cc.CharucoTarget(legacy=True), TRUE_K, TRUE_D, seed=3)
        cls.images = []
        while len(cls.images) < 24:
            image, _ = renderer.render()
            if cls.target.detect(image).count >= 20:
                cls.images.append(image)
        cls.results = cc.calibrate_intrinsics(cls.images, cls.target)

    def test_auto_layout_settles_on_the_legacy_board(self):
        self.assertIs(self.target.decided, True)

    def test_three_sizes_recover_the_lens(self):
        self.assertEqual([r["size"] for r in self.results], [list(s) for s in cc.INTRINSIC_SIZES])
        for r in self.results:
            truth = cc.scale_intrinsics(TRUE_K, (1640, 1232), r["size"])
            scale = r["size"][0] / 1640
            self.assertLess(r["reprojection_error"], 0.5, r["size"])
            self.assertAlmostEqual(r["K"][0, 0], truth[0, 0], delta=2.0 * scale + 0.5)
            self.assertAlmostEqual(r["K"][0, 2], truth[0, 2], delta=3.0 * scale + 0.5)
            self.assertAlmostEqual(r["K"][1, 2], truth[1, 2], delta=3.0 * scale + 0.5)
            self.assertAlmostEqual(r["D"][0], TRUE_D[0], delta=0.01)
            self.assertIn("K_scaled_from", r) if r["size"] != [1640, 1232] else None

    def test_written_files_load_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = cc.write_intrinsics(tmp, "camera_1", self.results, self.target)
            names = sorted(p.name for p in written)
            self.assertIn("imx219_640x480.yaml", names)
            self.assertIn("camera_1_820x616.json", names)
            k, d = cc.load_intrinsics(tmp, "camera_1", (640, 480))
            np.testing.assert_allclose(k, self.results[2]["K"])
            yaml = (Path(tmp) / "imx219_1640x1232.yaml").read_text()
            self.assertIn("distortion_model: equidistant", yaml)
            self.assertIn("num_calibration_frames: 24", yaml)

    def test_outlier_view_is_dropped(self):
        objp, imgp = [], []
        for image in self.images:            # 24 > 35 // 2: the floor lets views go
            det = self.target.detect(image)
            objp.append(self.target.object_points(det.ids))
            imgp.append(det.points)
        imgp[5] = imgp[5] + np.random.default_rng(0).normal(0, 6.0, imgp[5].shape)
        result = cc.calibrate_fisheye(objp, imgp, (1640, 1232), max_frame_err=2.0)
        self.assertIn(5, result["dropped_outliers"])
        self.assertNotIn(5, result["views_used"])


class Extrinsics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cams = [c for c in TRUTH["cameras"] if c["name"] in ("camera_1", "camera_3", "camera_5")]
        cls.renderer = StripRenderer(cams)
        cls.target = cc.strip_target()
        cls.placements = [cc.StripPlacement(0.0, 7.5, 0.0, True), cc.StripPlacement(0.05, 4.6, 0.4),
                          cc.StripPlacement(-0.05, 10.4, -0.3)]
        cls.observations = []
        for i, p in enumerate(cls.placements):
            for c in cams:
                det = cls.target.detect(cls.renderer.render(c["name"], p, seed=i))
                cls.observations.append(cc.StripObservation(c["name"], i, det.ids, det.points))
        cls.intrinsics = {c["name"]: (cls.renderer.k, cls.renderer.d) for c in cams}
        cls.solution = cc.solve_extrinsics(cls.observations, cls.intrinsics, cls.placements, cls.target)

    def test_cameras_land_on_the_truth(self):
        errors = cc.compare_to_truth(self.solution, TRUTH)
        self.assertEqual(sorted(errors), ["camera_1", "camera_3", "camera_5"])
        for name, e in errors.items():
            self.assertLess(e["position_error_mm"], 5.0, name)
            self.assertLess(e["rotation_error_deg"], 0.2, name)
        self.assertLess(self.solution["p95_px"], 1.5)

    def test_free_placements_are_solved(self):
        for p, truth in zip(self.solution["placements"], self.placements):
            self.assertAlmostEqual(p["x0"], truth.x0, delta=0.005)
            self.assertAlmostEqual(p["y0"], truth.y0, delta=0.005)
            self.assertAlmostEqual(p["yaw_deg"], truth.yaw_deg, delta=0.05)

    def test_a_camera_without_a_shared_placement_is_reported(self):
        lonely = [o for o in self.observations if not (o.camera_id == "camera_5" and o.placement == 0)]
        only_known = [cc.StripPlacement(0.0, 7.5, 0.0, True)]
        kept = [o for o in lonely if o.placement == 0] + [
            cc.StripObservation(o.camera_id, 1, o.ids, o.points) for o in lonely
            if o.camera_id == "camera_5" and o.placement == 2]
        sol = cc.solve_extrinsics(kept, self.intrinsics, only_known + [cc.StripPlacement(0, 0, 0)],
                                  self.target)
        self.assertIn("camera_5", sol["unsolved"])
        self.assertIn("camera_1", sol["cameras"])

    def test_runtime_config_carries_the_poses(self):
        base = json.loads((ROOT / "config/mocap_arena_imx219/runtime_cameras.json").read_text())
        config = cc.runtime_config(base, self.solution, self.intrinsics)
        cam = next(c for c in config["cameras"] if c["name"] == "camera_3")
        self.assertTrue(cam["calibrated"])
        self.assertEqual(cam["position_world"], self.solution["cameras"]["camera_3"]["position_world"])
        self.assertEqual(config["role"], "image_calibrated")
        self.assertTrue(config["calibration_version"].startswith("charuco-"))
        self.assertFalse(next(c for c in config["cameras"] if c["name"] == "camera_2")["calibrated"])

    def test_session_on_disk_solves_the_same(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            entries = []
            for i, p in enumerate(self.placements):
                folder = session / "extrinsics" / f"placement_{i}"
                folder.mkdir(parents=True)
                cams = {}
                for name in self.intrinsics:
                    cv2.imwrite(str(folder / f"{name}.png"), self.renderer.render(name, p, seed=i))
                    cams[name] = {"image": f"extrinsics/placement_{i}/{name}.png"}
                entries.append({"x0": p.x0, "y0": p.y0, "yaw_deg": p.yaw_deg, "known": p.known,
                                "cameras": cams})
            entries.append({"x0": 0, "y0": 0, "yaw_deg": 0, "known": False, "cameras": {},
                            "deleted": True})
            (session / "extrinsics" / "manifest.json").write_text(json.dumps({"placements": entries}))
            base = json.loads((ROOT / "config/mocap_arena_imx219/runtime_cameras.json").read_text())
            sol = cc.solve_session(session, base, self.target, self.intrinsics, truth=TRUTH)
            self.assertTrue((session / "extrinsics" / "runtime_cameras.json").is_file())
            for e in sol["vs_truth"].values():
                self.assertLess(e["position_error_mm"], 5.0)


if __name__ == "__main__":
    unittest.main()
