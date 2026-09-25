import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from localization_contracts.capture import LatestFrames, rgb_array
from localization_contracts.apriltag import Detection, PnpAprilTagObserver, TagConfig
from localization_contracts.fusion import PlanarFusion
from localization_contracts.opponent import Detection2D, MultiCameraAssociator


class LiveRegressions(unittest.TestCase):
    def test_padded_rgb_and_one_protobuf_data_access(self):
        class Message:
            width = 2
            height = 2
            step = 8
            pixel_format_type = 3
            reads = 0

            @property
            def data(self):
                self.reads += 1
                return bytes([1, 2, 3, 4, 5, 6, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0])

        msg = Message()
        image = rgb_array(msg)
        self.assertEqual(msg.reads, 1)
        self.assertEqual(
            image.tolist(), [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        )

    def test_latest_frame_is_bounded(self):
        frames = LatestFrames(["c"])
        for i in range(100):
            frames.push("c", i)
        self.assertEqual(len(frames.pending), 1)
        self.assertEqual(frames.take("c")[0], 99)
        self.assertEqual(frames.dropped["c"], 99)
        frames.close()

    def test_same_observer_uses_distinct_top_and_bottom_geometry(self):
        import cv2

        K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]])
        transforms = {
            0: {"rotation": np.eye(3), "translation": [0, 0, 0.2254]},
            1: {"rotation": np.diag([1.0, -1.0, -1.0]), "translation": [0, 0, 0.0146]},
        }
        obs = PnpAprilTagObserver(
            TagConfig(calibration_version="c"),
            K,
            [0] * 5,
            {"rotation": np.eye(3), "translation": [0, 0, 0]},
            transforms,
        )
        obj = np.array([[-0.2, 0.2, 0], [0.2, 0.2, 0], [0.2, -0.2, 0], [-0.2, -0.2, 0]])
        for ident in (0, 1):
            R = np.array(transforms[ident]["rotation"])
            rv, _ = cv2.Rodrigues(R)
            t = np.array([0.12, -0.08, 2.0]) + transforms[ident]["translation"]
            corners, _ = cv2.projectPoints(obj, rv, t, K, np.zeros(5))
            result = obs.observe(
                Detection(
                    "c", 1, ident, tuple(map(tuple, corners[:, 0])), 100, 110, 120
                )
            )
            np.testing.assert_allclose(result.position_m, [0.12, -0.08, 2.0], atol=1e-5)

    def test_association_recovers_after_gap(self):
        a = MultiCameraAssociator()
        d = Detection2D("c", 1, 0, (0, 0, 10, 10), 0.9, position_m=(1, 1))
        self.assertIsNotNone(a.select([d]))
        self.assertIsNotNone(
            a.select([replace(d, frame_seq=2, stamp_ns=1_000_000_000)])
        )

    def test_status_does_not_manufacture_publications(self):
        f = PlanarFusion()
        for _ in range(100):
            f.status(0)
        self.assertEqual(f.outputs, 0)


if __name__ == "__main__":
    unittest.main()


class EstimatorRegressions(unittest.TestCase):
    def test_future_observation_is_not_published_as_current(self):
        from localization_contracts.apriltag import AprilTagObserver

        obs = AprilTagObserver(
            TagConfig(calibration_version="c"), 500, 500, 320, 240
        ).observe(
            Detection(
                "c",
                1,
                0,
                ((270, 190), (370, 190), (370, 290), (270, 290)),
                100,
                110,
                120,
            )
        )
        f = PlanarFusion()
        f.update(obs)
        self.assertFalse(f.publish(50)["valid"])
        self.assertEqual(f.status(50).tracking_state, "TIME_UNCERTAIN")

    def test_filter_suppresses_static_velocity_noise(self):
        from localization_contracts.contracts import Observation, SCHEMA_VERSION

        rng = np.random.default_rng(19)
        f = PlanarFusion()
        velocities = []
        for i in range(300):
            x, y = rng.normal(0, 0.004, 2)
            t = i * 33_333_333
            obs = Observation(
                SCHEMA_VERSION,
                "c",
                i,
                str(i),
                "tag_rover",
                t,
                "sim",
                0,
                0,
                t,
                t,
                "c",
                "arena",
                (x, y, 0.14),
                (0.000016, 0.000016, 0.0001),
                1.0,
                "test",
                pixel_features={"yaw_rad": 0.0},
            )
            self.assertTrue(f.update(obs))
            if i > 60:
                velocities.append(np.hypot(f.state.vx, f.state.vy))
        self.assertLess(np.percentile(velocities, 95), 0.15)

    def test_cuboid_recovers_metric_center_from_oblique_box(self):
        from localization_contracts.cuboid import localize_box
        from scipy.spatial.transform import Rotation
        import cv2

        R = Rotation.from_euler("xyz", [np.pi, 0.07, 0.02]).as_matrix()
        c = {
            "K": [510, 0, 720, 0, 510, 540, 0, 0, 1],
            "D": [0] * 5,
            "position_world": [9, 6, 2.9],
            "R_world_optical": R.tolist(),
        }
        pts = np.array(
            [[x, y, z] for x in (-0.5, 0.5) for y in (-0.36, 0.36) for z in (0, 0.483)]
        )
        Q = Rotation.from_euler("z", 0.6).as_matrix()
        world = pts @ Q.T + [8.2, 5.3, 0]
        camera = (world - np.array(c["position_world"])) @ R
        uv, _ = cv2.projectPoints(
            camera,
            np.zeros(3),
            np.zeros(3),
            np.array(c["K"], dtype=float).reshape(3, 3),
            np.zeros(5),
        )
        uv = uv[:, 0]
        box = np.r_[uv.min(axis=0), uv.max(axis=0)]
        localized = localize_box(box, c)
        self.assertLess(
            np.linalg.norm(np.array(localized["position_m"][:2]) - [8.2, 5.3]), 0.01
        )
        self.assertFalse(localized["yaw_valid"])

    def test_printed_marker_axes_match_configured_tag_frame(self):
        import xml.etree.ElementTree as ET
        from pathlib import Path
        from localization_contracts.detector import AprilTagImageDetector

        root = ET.parse(Path(__file__).resolve().parents[1] / "worlds/mocap_arena.sdf")
        link = root.find(".//model[@name='tag_rover']/link[@name='base_link']")
        for ident in (0, 1):
            image = np.full((240, 240), 255, np.uint8)
            for visual in link.findall("visual"):
                if not visual.attrib["name"].startswith(f"tag_{ident}_"):
                    continue
                x, y, *_ = map(float, visual.findtext("pose").split())
                if ident == 1:
                    y = -y
                u = round((x + 0.25) * 400) + 20
                v = round((0.25 - y) * 400) + 20
                image[v - 10 : v + 10, u - 10 : u + 10] = 0
            detections = AprilTagImageDetector().detect(image)
            self.assertEqual([d.tag_id for d in detections], [ident])
            corners = np.array(detections[0].corners)
            self.assertLess(corners[0, 0], corners[1, 0])
            self.assertLess(corners[0, 1], corners[3, 1])
