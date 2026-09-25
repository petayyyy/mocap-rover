import math
import unittest

import numpy as np

import arena_render as ar
from localization_contracts.apriltag import (
    AprilTagObserver, Detection, PnpAprilTagObserver, TagConfig,
)
from localization_contracts.camera_model import CameraModel
from localization_contracts.contracts import covariance_matrix

BROKEN_IMX219_D = [-0.3112978153440105, 0.15431516540306534,
                   -0.0014882893582261717, 0.0022164735436313475,
                   -0.05947053168235783]


def observer(camera_xyz, R_world_optical, flipped=False, tag_dz=ar.TAG_DZ, **kwargs):
    options = dict(quality_min=0.05, min_side_px=14.0, image_size=list(ar.IMAGE_SIZE))
    options.update(kwargs)
    R_base_tag = np.diag([1.0, -1.0, -1.0]) if flipped else np.eye(3)
    return PnpAprilTagObserver(
        TagConfig(ids=(0, 1), calibration_version="test"), ar.K, [0.0] * 5,
        {"rotation": R_world_optical, "translation": list(camera_xyz)},
        {"rotation": R_base_tag, "translation": [0, 0, tag_dz]}, **options)


def look(obs, camera_xyz, R, base_xy, yaw=0.0, tag_id=0, **render):
    """Render, detect and observe, the way the runtime chains them."""
    frame = ar.render(camera_xyz, R, base_xy, yaw=yaw, tag_id=tag_id, **render)
    if frame is None:
        return None, None
    hits = ar.visible_corners(frame)
    if not hits:
        return None, frame
    detection = Detection("camera_1", 1, hits[0].tag_id, hits[0].corners, 100, 110, 120)
    return obs.observe(detection), frame


class CornerOnlyObserver(unittest.TestCase):
    def setUp(self):
        self.o = AprilTagObserver(TagConfig(calibration_version="c1"), 500, 500, 320, 240)

    def d(self, tag=0, side=100):
        return Detection("camera_1", 1, tag,
                         ((270, 190), (270 + side, 190),
                          (270 + side, 190 + side), (270, 190 + side)), 100, 110, 120)

    def test_observation_identity_geometry_and_quality(self):
        x = self.o.observe(self.d())
        self.assertEqual(x.marker_id, 0)
        self.assertEqual(x.calibration_version, "c1")
        self.assertGreater(x.position_m[2], 1)
        self.assertFalse(x.pose_6d_valid)
        self.assertEqual(x.attitude_state, "unknown")

    def test_wrong_id_missing_and_tiny_marker_rejected(self):
        self.assertIsNone(self.o.observe(self.d(9)))
        self.assertIsNone(self.o.observe(self.d(0, 1)))


class NadirCamera(unittest.TestCase):
    CAMERA = (9.0, 9.0, 2.9)

    def setUp(self):
        self.R = ar.camera_rotation()
        self.observer = observer(self.CAMERA, self.R)

    def test_recovers_base_pose_from_a_rendered_marker(self):
        result, _ = look(self.observer, self.CAMERA, self.R, (8.5, 9.4), yaw=0.6)
        self.assertIsNotNone(result, self.observer.last_diagnostic)
        self.assertTrue(result.pose_6d_valid)
        self.assertLess(math.hypot(result.position_m[0] - 8.5,
                                   result.position_m[1] - 9.4), 0.02)
        self.assertAlmostEqual(result.position_m[2], ar.BASE_Z, places=6)
        self.assertLess(abs(result.pixel_features["yaw_rad"] - 0.6), math.radians(3))

    def test_both_xy_estimates_are_reported(self):
        result, _ = look(self.observer, self.CAMERA, self.R, (8.0, 9.0))
        features = result.pixel_features
        self.assertIn("pnp_xy", features)
        self.assertIn("ray_xy", features)
        self.assertEqual(features["xy_source"], "ray")
        self.assertLess(features["pnp_ray_disagreement_m"], 0.10)

    def test_published_height_is_the_known_plane_not_the_pnp_range(self):
        result, _ = look(self.observer, self.CAMERA, self.R, (8.0, 9.6))
        self.assertEqual(result.position_m[2], ar.BASE_Z)
        self.assertNotAlmostEqual(result.pixel_features["pnp_base_z_m"], ar.BASE_Z, places=9)

    def test_covariance_is_anisotropic_and_grows_off_axis(self):
        near, _ = look(self.observer, self.CAMERA, self.R, (9.0, 9.0))
        far, _ = look(self.observer, self.CAMERA, self.R, (7.4, 9.0))
        self.assertIsNotNone(far, self.observer.last_diagnostic)
        near_block = covariance_matrix(near.covariance_m2)
        far_block = covariance_matrix(far.covariance_m2)
        self.assertGreater(np.trace(far_block), np.trace(near_block))
        eigenvalues = np.linalg.eigvalsh(far_block)
        self.assertGreater(eigenvalues[1] / eigenvalues[0], 1.3)

    def test_covariance_is_positive_definite(self):
        result, _ = look(self.observer, self.CAMERA, self.R, (8.2, 9.5))
        self.assertTrue((np.linalg.eigvalsh(covariance_matrix(result.covariance_m2)) > 0).all())


class TiltedCentreCamera(unittest.TestCase):
    CAMERA = (5.7, 6.0, 2.9)

    def setUp(self):
        self.R = ar.camera_rotation(40.0, 35.0)
        self.observer = observer(self.CAMERA, self.R)

    def test_oblique_view_still_resolves(self):
        result, _ = look(self.observer, self.CAMERA, self.R, (7.3, 7.2), yaw=-1.2)
        self.assertIsNotNone(result, self.observer.last_diagnostic)
        self.assertLess(math.hypot(result.position_m[0] - 7.3,
                                   result.position_m[1] - 7.2), 0.05)

    def test_covariance_reflects_the_incidence_penalty(self):
        nadir_camera, nadir_R = (9.0, 9.0, 2.9), ar.camera_rotation()
        flat, _ = look(observer(nadir_camera, nadir_R), nadir_camera, nadir_R, (9.0, 9.0))
        far, _ = look(self.observer, self.CAMERA, self.R, (8.4, 7.9))
        self.assertIsNotNone(far, self.observer.last_diagnostic)
        # 52 deg incidence against 0 deg: radial error grows as 1/cos^2.
        self.assertGreater(np.trace(covariance_matrix(far.covariance_m2)),
                           2 * np.trace(covariance_matrix(flat.covariance_m2)))


class Gates(unittest.TestCase):
    CAMERA = (9.0, 9.0, 2.9)

    def setUp(self):
        self.R = ar.camera_rotation()
        self.observer = observer(self.CAMERA, self.R)
        frame = ar.render(self.CAMERA, self.R, (9.0, 9.0))
        self.corners = np.asarray(ar.visible_corners(frame)[0].corners, dtype=float)

    def observe(self, corners, tag_id=0):
        return self.observer.observe(
            Detection("camera_1", 1, tag_id, tuple(map(tuple, corners)), 1, 2, 3))

    def reason(self):
        return self.observer.last_diagnostic["reason"]

    def test_physically_impossible_base_height_is_rejected(self):
        # The recorded run accepted base Z = 1.082 m at 1.47 px reprojection.
        # Shrinking the quad is how such a range error arises.
        centre = self.corners.mean(axis=0)
        self.assertIsNone(self.observe(centre + (self.corners - centre) * 0.45))
        self.assertIn(self.reason(), {"base_height", "pnp_ray_disagreement"})

    def test_marker_below_the_minimum_size_is_rejected(self):
        centre = self.corners.mean(axis=0)
        self.assertIsNone(self.observe(centre + (self.corners - centre) * 0.08))
        self.assertEqual(self.reason(), "too_small")

    def test_squashed_quadrilateral_is_rejected(self):
        self.assertIsNone(self.observe(
            np.array([[600.0, 500.0], [700.0, 500.0], [700.0, 502.0], [600.0, 502.0]])))
        self.assertEqual(self.reason(), "degenerate_quad")

    def test_marker_touching_the_frame_edge_is_rejected(self):
        self.assertIsNone(self.observe(
            np.array([[1.0, 500.0], [41.0, 500.0], [41.0, 540.0], [1.0, 540.0]])))
        self.assertEqual(self.reason(), "frame_edge")

    def test_grazing_incidence_is_rejected(self):
        tight = observer(self.CAMERA, self.R, max_incidence_deg=10.0)
        result, _ = look(tight, self.CAMERA, self.R, (7.4, 9.0))
        self.assertIsNone(result)
        self.assertEqual(tight.last_diagnostic["reason"], "incidence")

    def test_corners_outside_the_model_domain_are_rejected(self):
        # The plumb_bob fit that shipped in the world folds over at r_d=0.833,
        # so a marker further out has no meaningful undistorted ray.
        broken = observer(self.CAMERA, self.R,
                          camera_model=CameraModel(ar.K, BROKEN_IMX219_D, list(ar.IMAGE_SIZE)))
        frame = ar.render(self.CAMERA, self.R, (7.1, 9.0))
        hits = ar.visible_corners(frame)
        self.assertTrue(hits, "marker should still be detectable this far out")
        self.assertIsNone(broken.observe(
            Detection("camera_1", 1, 0, hits[0].corners, 1, 2, 3)))
        self.assertEqual(broken.last_diagnostic["reason"], "outside_model_domain")

    def test_mirrored_corner_order_cannot_be_fitted(self):
        self.assertIsNone(self.observe(self.corners[::-1]))
        self.assertIn(self.reason(), {"no_physical_solution", "nonplanar_pose",
                                      "base_height", "pnp_ray_disagreement"})

    def test_gates_scale_with_the_depth_precision_of_the_view(self):
        far = self.observer.range_sigma_m(22.0, 2.5)
        near = self.observer.range_sigma_m(77.0, 2.5)
        self.assertGreater(far, 3 * near)


class AccuracySweep(unittest.TestCase):
    """The regression this rewrite exists to prevent."""

    RIGS = (
        ((9.0, 9.0, 2.9), ar.camera_rotation()),
        ((3.0, 3.0, 2.9), ar.camera_rotation()),
        ((5.7, 6.0, 2.9), ar.camera_rotation(40.0, 35.0)),
        ((6.3, 6.0, 2.9), ar.camera_rotation(40.0, 215.0)),
    )

    def test_accepted_measurements_are_centimetre_accurate(self):
        rng = np.random.default_rng(3)
        errors, yaw_errors, detected, accepted = [], [], 0, 0
        for camera, R in self.RIGS:
            obs = observer(camera, R)
            for _ in range(60):
                base = rng.uniform(0.6, 11.4, 2)
                yaw = rng.uniform(-math.pi, math.pi)
                frame = ar.render(camera, R, base, yaw=yaw)
                if frame is None:
                    continue
                hits = ar.visible_corners(frame)
                if not hits:
                    continue
                detected += 1
                result = obs.observe(
                    Detection("c", 1, hits[0].tag_id, hits[0].corners, 1, 2, 3))
                if result is None:
                    continue
                accepted += 1
                errors.append(math.hypot(result.position_m[0] - base[0],
                                         result.position_m[1] - base[1]))
                yaw_errors.append(abs(math.degrees(
                    (result.pixel_features["yaw_rad"] - yaw + math.pi) % (2 * math.pi) - math.pi)))
                self.assertEqual(result.position_m[2], ar.BASE_Z)
        self.assertGreater(accepted, 40)
        self.assertGreater(accepted / detected, 0.80)
        self.assertLess(float(np.percentile(errors, 95)), 0.05)
        self.assertLess(max(errors), 0.10)
        self.assertLess(float(np.percentile(yaw_errors, 95)), 5.0)


class TagTransforms(unittest.TestCase):
    """Marker id 1 is mounted with R_base_tag = diag(1,-1,-1), facing down.

    A ceiling camera never sees it, so this is checked from below; inverting
    that transform is what keeps the reported pose a base pose rather than a
    marker pose.
    """

    CAMERA = (3.0, 3.0, -2.5)

    def test_flipped_marker_transform_recovers_the_base(self):
        R = ar.camera_rotation() @ np.diag([1.0, -1.0, -1.0])
        obs = observer(self.CAMERA, R, flipped=True, tag_dz=ar.BOTTOM_TAG_DZ)
        result, _ = look(obs, self.CAMERA, R, (2.8, 3.2), yaw=0.3, tag_id=1,
                         flipped=True, tag_dz=ar.BOTTOM_TAG_DZ)
        self.assertIsNotNone(result, obs.last_diagnostic)
        self.assertLess(math.hypot(result.position_m[0] - 2.8,
                                   result.position_m[1] - 3.2), 0.03)


class XySource(unittest.TestCase):
    CAMERA = (5.7, 6.0, 2.9)

    def test_pnp_source_is_selectable(self):
        R = ar.camera_rotation(40.0, 35.0)
        obs = observer(self.CAMERA, R, xy_source="pnp")
        result, _ = look(obs, self.CAMERA, R, (7.3, 7.2))
        self.assertIsNotNone(result, obs.last_diagnostic)
        self.assertEqual(result.pixel_features["xy_source"], "pnp")

    def test_unknown_source_is_refused(self):
        with self.assertRaises(ValueError):
            observer(self.CAMERA, ar.camera_rotation(), xy_source="homography")


if __name__ == "__main__":
    unittest.main()
