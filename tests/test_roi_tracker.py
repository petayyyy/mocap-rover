import math
import unittest

import numpy as np

import arena_render as ar
from localization_contracts.camera_model import CameraModel
from localization_contracts.detector import AprilTagImageDetector
from localization_contracts.roi_tracker import (
    ACQUIRE, IDLE, ROI, WATCHDOG, CameraRoiPlanner, KltMarkerTracker,
    project_to_image,
)

MODEL = CameraModel(ar.K, [0.0] * 5, list(ar.IMAGE_SIZE))
NADIR = ar.camera_rotation()
CENTRE = ar.camera_rotation(40.0, 35.0)


def planner(position=(9.0, 9.0, 2.9), rotation=NADIR, **kwargs):
    return CameraRoiPlanner(MODEL, rotation, position, **kwargs)


PLANES = (ar.INVERTED_BASE_Z - ar.BOTTOM_TAG_DZ, ar.BASE_Z + ar.TAG_DZ)


class FlippedRover(unittest.TestCase):
    """A flip moves the visible marker down by the height of the rover.

    The window is planned from the track's XY, which does not change, so the
    only thing that can put the marker outside it is aiming at the wrong
    plane.  A tilted camera turns that plane error into a sideways offset.
    """

    CAMERA = (5.7, 6.0, 2.9)
    BASE_XY = (8.6, 9.3)

    def detected_centre(self):
        frame = ar.render_inverted(self.CAMERA, CENTRE, self.BASE_XY, yaw=0.4)
        hits = AprilTagImageDetector(allowed_ids=(0, 1)).detect(frame)
        self.assertEqual([hit.tag_id for hit in hits], [1])
        return np.asarray(hits[0].corners).mean(axis=0)

    def test_the_window_is_aimed_nearer_the_underside_marker(self):
        """The top plane is not where a flipped marker is.

        At the shipped 160 px floor the padding covers the error anyway, so
        this measures the aim rather than a lost detection: what the second
        plane buys is margin that does not depend on the floor.
        """
        centre = self.detected_centre()
        def miss(plane):
            plan = planner(self.CAMERA, CENTRE, tag_plane_z=plane).plan(
                (*self.BASE_XY, 0.05))
            return math.hypot(plan.projected[0] - centre[0],
                              plan.projected[1] - centre[1])
        self.assertLess(miss(PLANES), miss(ar.BASE_Z + ar.TAG_DZ))

    def test_covering_both_planes_keeps_the_underside_marker_in_the_window(self):
        plan = planner(self.CAMERA, CENTRE, tag_plane_z=PLANES).plan(
            (*self.BASE_XY, 0.05))
        self.assertEqual(plan.mode, ROI)
        x, y, w, h = plan.roi
        u, v = self.detected_centre()
        self.assertTrue(x <= u < x + w and y <= v < y + h,
                        f"marker at {(u, v)} outside roi {plan.roi}")

    def test_an_upright_rover_is_still_covered_by_the_same_window(self):
        plan = planner(self.CAMERA, CENTRE, tag_plane_z=PLANES).plan(
            (*self.BASE_XY, 0.05))
        frame = ar.render(self.CAMERA, CENTRE, self.BASE_XY, yaw=0.4)
        hits = AprilTagImageDetector(allowed_ids=(0, 1)).detect(frame)
        u, v = np.asarray(hits[0].corners).mean(axis=0)
        x, y, w, h = plan.roi
        self.assertTrue(x <= u < x + w and y <= v < y + h)

    def test_a_nadir_camera_pays_nothing_for_the_second_plane(self):
        one = planner(rotation=NADIR, tag_plane_z=PLANES[1]).plan((9.0, 9.0, 0.05))
        both = planner(rotation=NADIR, tag_plane_z=PLANES).plan((9.0, 9.0, 0.05))
        self.assertEqual(one.roi[2], both.roi[2])


class Projection(unittest.TestCase):
    def test_a_point_under_the_camera_lands_on_the_principal_point(self):
        uv = project_to_image((9.0, 9.0, 0.3654), MODEL, NADIR, (9.0, 9.0, 2.9))
        self.assertAlmostEqual(uv[0], ar.K[0, 2], places=3)
        self.assertAlmostEqual(uv[1], ar.K[1, 2], places=3)

    def test_a_point_behind_the_camera_has_no_projection(self):
        self.assertIsNone(
            project_to_image((1.0, 1.0, 0.3654), MODEL, CENTRE, (5.7, 6.0, 2.9)))

    def test_the_projection_matches_where_the_detector_finds_the_marker(self):
        frame = ar.render((9.0, 9.0, 2.9), NADIR, (8.3, 9.4), yaw=0.5)
        hits = AprilTagImageDetector(allowed_ids=(0, 1)).detect(frame)
        centre = np.asarray(hits[0].corners).mean(axis=0)
        uv = project_to_image((8.3, 9.4, ar.BASE_Z + ar.TAG_DZ), MODEL, NADIR,
                              (9.0, 9.0, 2.9))
        self.assertLess(math.hypot(uv[0] - centre[0], uv[1] - centre[1]), 2.0)


class Planning(unittest.TestCase):
    def test_no_track_means_full_frame_acquisition(self):
        self.assertEqual(planner().plan(None).mode, ACQUIRE)

    def test_a_confident_prediction_in_view_gives_a_small_window(self):
        plan = planner().plan((9.0, 9.0, 0.03))
        self.assertEqual(plan.mode, ROI)
        self.assertLessEqual(plan.roi[2], 240)
        self.assertGreaterEqual(plan.roi[2], 160)

    def test_the_window_grows_with_position_uncertainty(self):
        tight = planner().plan((9.0, 9.0, 0.02)).roi[2]
        loose = planner().plan((9.0, 9.0, 0.60)).roi[2]
        self.assertGreater(loose, tight)

    def test_the_window_is_centred_on_the_projected_marker(self):
        plan = planner().plan((8.2, 9.3, 0.05))
        centre = (plan.roi[0] + plan.roi[2] / 2, plan.roi[1] + plan.roi[3] / 2)
        self.assertLess(math.hypot(centre[0] - plan.projected[0],
                                   centre[1] - plan.projected[1]), 1.5)

    def test_the_window_covers_where_the_marker_actually_is(self):
        frame = ar.render((9.0, 9.0, 2.9), NADIR, (8.2, 9.3), yaw=0.4)
        detector = AprilTagImageDetector(allowed_ids=(0, 1))
        plan = planner().plan((8.2, 9.3, 0.05))
        self.assertEqual([d.tag_id for d in detector.detect(frame, roi=plan.roi)], [0])

    def test_a_camera_the_prediction_misses_goes_idle(self):
        plan = planner(position=(5.7, 6.0, 2.9), rotation=CENTRE).plan((1.0, 1.0, 0.05))
        self.assertIn(plan.mode, (IDLE, WATCHDOG))

    def test_an_idle_camera_still_checks_in_on_the_watchdog(self):
        p = planner(position=(5.7, 6.0, 2.9), rotation=CENTRE, watchdog_period_s=1.0)
        self.assertEqual(p.plan((1.0, 1.0, 0.05), now_ns=0).mode, WATCHDOG)
        self.assertEqual(p.plan((1.0, 1.0, 0.05), now_ns=500_000_000).mode, IDLE)
        self.assertEqual(p.plan((1.0, 1.0, 0.05), now_ns=1_500_000_000).mode, WATCHDOG)

    def test_repeated_misses_widen_the_window_then_fall_back_to_the_full_frame(self):
        p = planner(max_misses=4)
        sizes = []
        for _ in range(4):
            sizes.append(p.plan((9.0, 9.0, 0.05)).roi[2])
            p.report(False)
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreater(sizes[-1], sizes[0])
        self.assertEqual(p.plan((9.0, 9.0, 0.05)).mode, ACQUIRE)

    def test_a_hit_resets_the_window(self):
        p = planner()
        for _ in range(3):
            p.report(False)
        widened = p.plan((9.0, 9.0, 0.05)).roi[2]
        p.report(True)
        self.assertLess(p.plan((9.0, 9.0, 0.05)).roi[2], widened)

    def test_the_window_never_exceeds_its_ceiling(self):
        plan = planner(max_roi_px=320).plan((9.0, 9.0, 5.0))
        self.assertLessEqual(plan.roi[2], 320)


class Klt(unittest.TestCase):
    def frames(self, path):
        return [ar.render((9.0, 9.0, 2.9), NADIR, point, yaw=0.3) for point in path]

    def corners(self, frame):
        hits = AprilTagImageDetector(allowed_ids=(0, 1)).detect(frame)
        return np.asarray(hits[0].corners, dtype=np.float32) if hits else None

    def test_flow_follows_the_marker_between_detections(self):
        path = [(9.0, 9.0), (8.95, 9.0), (8.90, 9.0), (8.85, 9.0)]
        frames = self.frames(path)
        tracker = KltMarkerTracker()
        tracker.start(frames[0], self.corners(frames[0]))
        for frame, point in zip(frames[1:], path[1:]):
            tracked = tracker.track(frame)
            self.assertIsNotNone(tracked, "flow gave up on a smooth 5 cm step")
            truth = self.corners(frame)
            self.assertLess(np.abs(tracked - truth).max(), 4.0)

    def test_flow_gives_up_rather_than_inventing_a_pose(self):
        frames = self.frames([(9.0, 9.0)])
        tracker = KltMarkerTracker()
        tracker.start(frames[0], self.corners(frames[0]))
        # The marker leaves; a tracker that keeps answering here is the one
        # that silently walks a track onto the background.
        empty = ar.render((9.0, 9.0, 2.9), NADIR, (1.0, 1.0))
        tracker.track(empty)
        self.assertFalse(tracker.active)

    def test_flow_is_bounded_in_time(self):
        frames = self.frames([(9.0, 9.0)])
        tracker = KltMarkerTracker(max_frames=3)
        tracker.start(frames[0], self.corners(frames[0]))
        for _ in range(3):
            self.assertIsNotNone(tracker.track(frames[0]))
        self.assertIsNone(tracker.track(frames[0]))
        self.assertFalse(tracker.active)

    def test_an_unstarted_tracker_reports_nothing(self):
        self.assertIsNone(KltMarkerTracker().track(self.frames([(9.0, 9.0)])[0]))


if __name__ == "__main__":
    unittest.main()
