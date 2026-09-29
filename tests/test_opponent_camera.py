"""One camera's reading of the rover without a marker."""
import math
import unittest

import cv2
import numpy as np

from localization_contracts.foreground import ClipBackground
from localization_contracts.opponent_camera import FLOW, SILHOUETTE, OpponentCamera
from localization_contracts.roi_tracker import KltFeatureTracker
from test_foreground import CAMERA, MODEL, NADIR, floor, window

SIZE = (0.9, 0.52, 0.483)
TAG = (0.72, 0.52, 0.40)


def textured_body(image, xy, yaw, size=SIZE, seed=0):
    """A cuboid silhouette with a fixed texture, so flow has corners to follow."""
    length, width, height = size
    c, s = math.cos(yaw), math.sin(yaw)
    grid = np.array([[a * length / 2, b * width / 2, z] for a in np.linspace(-1, 1, 9)
                     for b in np.linspace(-1, 1, 7) for z in (0.0, height)])
    grid[:, :2] = grid[:, :2] @ np.array([[c, s], [-s, c]]) + xy
    uv = MODEL.project((grid - CAMERA) @ NADIR)
    out = image.copy()
    cv2.fillConvexPoly(out, cv2.convexHull(np.round(uv).astype(np.int32)), (150, 60, 50))
    rng = np.random.default_rng(seed)
    top = np.array([[a, b, height] for a in np.linspace(-0.4, 0.4, 7)
                    for b in np.linspace(-0.2, 0.2, 5)])
    shade = rng.integers(40, 250, len(top))
    top[:, :2] = top[:, :2] @ np.array([[c, s], [-s, c]]) + xy
    for (u, v), g in zip(MODEL.project((top - CAMERA) @ NADIR), shade):
        cv2.rectangle(out, (int(u) - 4, int(v) - 4), (int(u) + 4, int(v) + 4),
                      (int(g), int(g), int(g)), -1)
    return out


class OpponentView(unittest.TestCase):
    def setUp(self):
        self.empty = floor()
        self.camera = OpponentCamera(
            "c", MODEL, NADIR, CAMERA, ClipBackground.from_frames([self.empty], alpha=0.0),
            size_m=SIZE, tag_size_m=TAG)

    def test_the_silhouette_is_read_near_the_body(self):
        xy = np.array([4.2, 7.1])
        frame = textured_body(self.empty, xy, 0.4)
        reading, diag = self.camera.read(frame, window(xy), xy + 0.05)
        self.assertIsNotNone(reading, diag)
        self.assertEqual(reading.method, SILHOUETTE)
        self.assertLess(math.hypot(reading.x - xy[0], reading.y - xy[1]), 0.06)

    def test_the_tag_rover_is_cut_out_before_the_blob_is_chosen(self):
        opponent, tag = np.array([4.4, 7.0]), np.array([4.4, 6.25])
        frame = textured_body(textured_body(self.empty, tag, 0.0, TAG, seed=1), opponent, 0.0)
        roi = window((4.4, 6.6), half=330)
        reading, diag = self.camera.read(frame, roi, opponent, tag_pose=(*tag, 0.0))
        self.assertIsNotNone(reading, diag)
        self.assertGreater(diag["tag_excluded_px"], 0)
        self.assertLess(math.hypot(reading.x - opponent[0], reading.y - opponent[1]), 0.08)

    def test_flow_bridges_a_frame_whose_blob_is_cut_by_the_window(self):
        xy = np.array([4.2, 7.1])
        first, _ = self.camera.read(textured_body(self.empty, xy, 0.4), window(xy), xy)
        self.assertEqual(first.method, SILHOUETTE)
        moved = xy + [0.04, 0.02]
        frame = textured_body(self.empty, moved, 0.4)
        # A window far too small for the body: the silhouette is refused.
        u, v = (np.array(window(moved)[:2]) + 260).astype(int)
        reading, diag = self.camera.read(frame, (u - 60, v - 60, 120, 120), moved)
        self.assertEqual(diag["silhouette"], "blob_clipped_by_window")
        self.assertIsNotNone(reading, diag)
        self.assertEqual(reading.method, FLOW)
        # Flow moves the last estimate by the body's own motion; the error of
        # that estimate is inherited, the motion is what flow has to get right.
        shift = np.array([reading.x - first.x, reading.y - first.y])
        self.assertLess(np.linalg.norm(shift - (moved - xy)), 0.01)


class FeatureFlow(unittest.TestCase):
    def test_points_follow_a_shift_and_a_blank_frame_ends_the_track(self):
        rng = np.random.default_rng(3)
        base = np.zeros((200, 200), np.uint8)
        for _ in range(40):
            x, y = rng.integers(20, 170, 2)
            cv2.rectangle(base, (int(x), int(y)), (int(x) + 6, int(y) + 6), int(rng.integers(80, 255)), -1)
        tracker = KltFeatureTracker()
        self.assertTrue(tracker.start(base, (100, 50, 200, 200)))
        shifted = np.roll(np.roll(base, 3, axis=1), 2, axis=0)
        old, new = tracker.track(lambda rect: shifted)
        np.testing.assert_allclose(np.median(new - old, axis=0), [3, 2], atol=0.3)
        self.assertIsNone(tracker.track(lambda rect: np.zeros_like(base)))
        self.assertFalse(tracker.active)


if __name__ == "__main__":
    unittest.main()
