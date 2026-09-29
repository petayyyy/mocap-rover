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


# The rover as the world actually builds it: a marker on top and another,
# printed upside down, underneath.
ARENA_TAGS = {
    0: {"rotation": np.eye(3), "translation": [0, 0, ar.TAG_DZ]},
    1: {"rotation": np.diag([1.0, -1.0, -1.0]),
        "translation": [0, 0, ar.BOTTOM_TAG_DZ]},
}
ARENA_PLACEMENT = {0: "top", 1: "bottom"}


def arena_observer(camera_xyz, R_world_optical, **kwargs):
    """Observer configured with both markers, as the runtime configures it."""
    options = dict(quality_min=0.05, min_side_px=14.0,
                   image_size=list(ar.IMAGE_SIZE),
                   tag_placement=ARENA_PLACEMENT)
    options.update(kwargs)
    return PnpAprilTagObserver(
        TagConfig(ids=(0, 1), calibration_version="test"), ar.K, [0.0] * 5,
        {"rotation": R_world_optical, "translation": list(camera_xyz)},
        ARENA_TAGS, **options)


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


class IdealFisheyeCamera(unittest.TestCase):
    """The IMX219-160 world: r = f*theta, calibrated as D = [0, 0, 0, 0].

    The runtime builds the observer from K and D alone, so the projection
    model has to follow from the four-element vector.  Read as a pinhole, the
    bearing was bent off-axis and the base-height gate refused most views.
    """

    K = [[734.53211682016, 0.0, 820.0], [0.0, 734.53211682016, 616.0], [0.0, 0.0, 1.0]]
    SIZE = (1640, 1232)
    CAMERA = (3.0, 6.0, 2.9)

    def corners(self, base_xy, yaw):
        """Exact marker corners through the fisheye, in detector order."""
        s = 0.2
        c, n = math.cos(yaw), math.sin(yaw)
        R_world_tag = np.array([[c, -n, 0], [n, c, 0], [0, 0, 1.0]])
        origin = np.array([base_xy[0], base_xy[1], ar.BASE_Z + ar.TAG_DZ])
        tag = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
        world = tag @ R_world_tag.T + origin
        R = ar.camera_rotation()
        optical = (world - np.asarray(self.CAMERA)) @ R
        model = CameraModel(self.K, [0.0] * 4, self.SIZE, model="fisheye")
        return tuple(map(tuple, model.project(optical)))

    def test_off_axis_views_are_accepted_and_centimetre_accurate(self):
        obs = PnpAprilTagObserver(
            TagConfig(ids=(0,), calibration_version="test"), self.K, [0.0] * 4,
            {"rotation": ar.camera_rotation(), "translation": list(self.CAMERA)},
            {"rotation": np.eye(3), "translation": [0, 0, ar.TAG_DZ]},
            image_size=list(self.SIZE))
        self.assertEqual(obs.camera_model.model, "fisheye")
        # World X runs along the short side of this frame, Y along the long
        # one; the last view is 50 degrees off-axis.
        offsets = ((0.0, 0.3), (0.8, 0.3), (1.6, 0.3), (2.2, 0.3),
                   (0.3, 1.5), (0.3, 2.3), (1.5, 2.0), (0.3, 3.0))
        for dx, dy in offsets:
            for yaw in (0.0, 0.7, 2.5):
                base = (self.CAMERA[0] + dx, self.CAMERA[1] + dy)
                with self.subTest(dx=dx, dy=dy, yaw=yaw):
                    result = obs.observe(Detection(
                        "c", 1, 0, self.corners(base, yaw), 1, 2, 3))
                    self.assertIsNotNone(result, obs.last_diagnostic)
                    self.assertLess(math.hypot(result.position_m[0] - base[0],
                                               result.position_m[1] - base[1]), 0.005)
                    self.assertLess(abs(obs.last_diagnostic["pnp_base_z_m"] - ar.BASE_Z), 0.02)
                    self.assertLess(abs(math.remainder(
                        result.pixel_features["yaw_rad"] - yaw, 2 * math.pi)), math.radians(1))


class RollingShutter(unittest.TestCase):
    """The marker is stamped when its rows were exposed, not at frame start."""

    def test_the_exposure_time_follows_the_marker_row(self):
        from localization_contracts.rover_filter import measurement_from_observation
        camera, R = (9.0, 9.0, 2.9), ar.camera_rotation()
        results = {}
        for line_time in (0.0, 9500.0):
            obs = observer(camera, R, line_time_ns=line_time)
            result, _ = look(obs, camera, R, (9.4, 8.7))
            self.assertIsNotNone(result, obs.last_diagnostic)
            results[line_time] = result
        row = results[9500.0].pixel_features["exposure_row_px"]
        self.assertGreater(row, 0)
        self.assertEqual(results[0.0].pixel_features["exposure_time_ns"], 100)
        self.assertEqual(results[9500.0].pixel_features["exposure_time_ns"],
                         100 + round(row * 9500.0))
        stamps = {m.stamp_ns for m in measurement_from_observation(results[9500.0])}
        self.assertEqual(stamps, {100 + round(row * 9500.0)})
        # Only the stamp moves; the geometry is the same frame.
        self.assertEqual(results[0.0].position_m, results[9500.0].position_m)


class RollingShutterOnAMovingRover(unittest.TestCase):
    """The row correction pays off exactly when the shutter really rolls.

    A rover drives at 4 m/s under a camera whose rows are read 9.5 us apart
    (IMX219).  Each frame shows the marker where the rover was when the
    marker's rows were exposed, about 6 ms after the frame stamp.  The same
    corners are fed to two filters: one told the line time, one not.
    Gazebo renders with a global shutter, where telling the observer a line
    time it does not have moves every stamp the wrong way instead.
    """

    SPEED = 4.0
    LINE_NS = 9500.0

    def track_error(self, render_line_ns, observer_line_ns):
        from localization_contracts.rover_filter import (
            ImmRoverFilter, measurement_from_observation)
        camera, R = (9.0, 9.0, 2.9), ar.camera_rotation()
        obs = observer(camera, R, line_time_ns=observer_line_ns)
        track = ImmRoverFilter()
        start_ns = 1_000_000_000

        def position(t_ns):
            return (8.0 + self.SPEED * (t_ns - start_ns) / 1e9, 8.6)

        errors = []
        for k in range(50):
            stamp = start_ns + k * 12_000_000
            row = 0.0
            for _ in range(3):   # the row fixes the time, the time moves the row
                seen_at = position(stamp + row * render_line_ns)
                frame = ar.render(camera, R, seen_at)
                hits = ar.visible_corners(frame)
                row = float(np.asarray(hits[0].corners)[:, 1].mean())
            result = obs.observe(Detection("camera_1", k, hits[0].tag_id, hits[0].corners,
                                           stamp, stamp, stamp))
            self.assertIsNotNone(result, obs.last_diagnostic)
            track.apply_group(measurement_from_observation(result))
            if k >= 20:
                now = stamp + 20_000_000
                state = track.publish(now)["state"]
                truth = position(now)
                errors.append(math.hypot(state["x"] - truth[0], state["y"] - truth[1]))
        return float(np.mean(errors))

    def test_the_correction_helps_a_rolling_shutter(self):
        corrected = self.track_error(self.LINE_NS, self.LINE_NS)
        ignored = self.track_error(self.LINE_NS, 0.0)
        self.assertLess(corrected, 0.5 * ignored)
        self.assertGreater(ignored, 0.01)

    def test_the_correction_hurts_a_global_shutter(self):
        honest = self.track_error(0.0, 0.0)
        wrong = self.track_error(0.0, self.LINE_NS)
        self.assertGreater(wrong, 2 * honest)


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


class InvertedRover(unittest.TestCase):
    """A flip is exactly when the underside marker becomes visible.

    At speed the rover turns over, and a tracker that only knows the top
    marker loses it at the moment it most needs to be found.
    """

    CAMERA = (9.0, 9.0, 2.9)
    PLACEMENT = {0: "top", 1: "bottom"}

    def setUp(self):
        self.R = ar.camera_rotation()

    def rig(self, **kwargs):
        return arena_observer(self.CAMERA, self.R, **kwargs)

    def test_placement_states_which_way_up_the_rover_is(self):
        obs = self.rig()
        self.assertEqual(obs.base_orientation(0), 1.0)
        self.assertEqual(obs.base_orientation(1), -1.0)
        self.assertAlmostEqual(obs.nominal_base_z(0), ar.BASE_Z)
        self.assertAlmostEqual(obs.nominal_base_z(1), 0.225)
        # Flipping mirrors the offset through the base.
        self.assertAlmostEqual(obs.tag_plane_z(0), ar.BASE_Z + ar.TAG_DZ)
        self.assertAlmostEqual(obs.tag_plane_z(1), 0.225 - ar.BOTTOM_TAG_DZ)

    def test_the_underside_marker_locates_a_flipped_rover(self):
        obs = self.rig()
        frame = ar.render_inverted(self.CAMERA, self.R, (8.6, 9.3), yaw=0.4)
        hits = ar.visible_corners(frame)
        self.assertTrue(hits, "underside marker should be visible from above")
        result = obs.observe(Detection("camera_1", 1, 1, hits[0].corners, 1, 2, 3))
        self.assertIsNotNone(result, obs.last_diagnostic)
        self.assertLess(math.hypot(result.position_m[0] - 8.6,
                                   result.position_m[1] - 9.3), 0.05)
        self.assertTrue(result.pixel_features["base_inverted"])
        self.assertEqual(result.pixel_features["placement"], "bottom")

    def test_an_upright_rover_still_reads_from_the_top_marker(self):
        obs = self.rig()
        result, _ = look(obs, self.CAMERA, self.R, (8.5, 9.4), yaw=0.6)
        self.assertIsNotNone(result, obs.last_diagnostic)
        self.assertFalse(result.pixel_features["base_inverted"])
        self.assertAlmostEqual(result.position_m[2], ar.BASE_Z, places=6)

    def test_the_placement_map_is_what_makes_the_flip_work(self):
        # Mislabel the underside marker as a top one and the same corners are
        # refused: the gate follows the declared placement rather than having
        # been loosened to accept anything roughly level.
        wrong = arena_observer(self.CAMERA, self.R,
                               tag_placement={0: "top", 1: "top"})
        frame = ar.render_inverted(self.CAMERA, self.R, (8.6, 9.3))
        hits = ar.visible_corners(frame)
        self.assertTrue(hits)
        self.assertIsNone(wrong.observe(
            Detection("camera_1", 1, 1, hits[0].corners, 1, 2, 3)))
        self.assertIn(wrong.last_diagnostic["reason"],
                      {"nonplanar_pose", "base_height", "no_physical_solution",
                       "pnp_ray_disagreement"})

    def test_a_mirrored_branch_is_still_refused_for_either_placement(self):
        # Reversing the corner winding cannot be fitted by any rigid planar
        # pose; declaring a placement must not have opened that door.
        obs = self.rig()
        for tag_id, frame in (
                (0, ar.render(self.CAMERA, self.R, (8.5, 9.4))),
                (1, ar.render_inverted(self.CAMERA, self.R, (8.5, 9.4)))):
            hits = ar.visible_corners(frame)
            self.assertTrue(hits)
            reversed_corners = tuple(reversed(hits[0].corners))
            with self.subTest(tag_id=tag_id):
                self.assertIsNone(obs.observe(
                    Detection("camera_1", 1, tag_id, reversed_corners, 1, 2, 3)))

    def test_default_placement_keeps_the_old_behaviour(self):
        obs = observer(self.CAMERA, self.R)
        self.assertEqual(obs.base_orientation(0), 1.0)
        self.assertEqual(obs.base_orientation(1), 1.0)

    def test_both_markers_place_the_rover_at_the_same_xy(self):
        obs = self.rig()
        upright, _ = look(obs, self.CAMERA, self.R, (8.4, 9.2), yaw=0.2)
        frame = ar.render_inverted(self.CAMERA, self.R, (8.4, 9.2), yaw=0.2)
        hits = ar.visible_corners(frame)
        flipped = obs.observe(Detection("camera_1", 2, 1, hits[0].corners, 1, 2, 3))
        self.assertIsNotNone(upright)
        self.assertIsNotNone(flipped, obs.last_diagnostic)
        self.assertLess(math.hypot(upright.position_m[0] - flipped.position_m[0],
                                   upright.position_m[1] - flipped.position_m[1]), 0.06)
