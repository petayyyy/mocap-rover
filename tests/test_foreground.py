"""Foreground and silhouette for the rover without a marker."""
import math
import unittest

import cv2
import numpy as np

from localization_contracts.camera_model import CameraModel
from localization_contracts.foreground import (
    ClipBackground, SilhouetteObserver, in_masked_zone,
)

K = [[734.5, 0, 820], [0, 734.5, 616], [0, 0, 1]]
MODEL = CameraModel(K, [0.0] * 4, (1640, 1232))
NADIR = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=float)
CAMERA = np.array([3.0, 6.0, 2.9])
SIZE = (0.9, 0.52, 0.483)


def floor():
    """A textured empty arena: a smooth gradient plus panel seams."""
    yy, xx = np.mgrid[0:1232, 0:1640]
    base = (30 + 20 * xx / 1640 + 10 * yy / 1232).astype(np.uint8)
    base[::97, :] = 45
    base[:, ::131] = 45
    return np.dstack([base, base + 3, base + 6])


def with_body(image, xy, yaw, colour=(160, 40, 40), size=SIZE):
    """Paint the silhouette of a cuboid as the fisheye sees it."""
    length, width, height = size
    c, s = math.cos(yaw), math.sin(yaw)
    ts = np.linspace(-0.5, 0.5, 25)
    edges = []
    for z in (0.0, height):
        for t in ts:
            edges += [(t * length, -width / 2, z), (t * length, width / 2, z),
                      (-length / 2, t * width, z), (length / 2, t * width, z)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            edges += [(sx * length / 2, sy * width / 2, t * height + height / 2) for t in ts]
    pts = np.array(edges)
    pts[:, :2] = pts[:, :2] @ np.array([[c, s], [-s, c]]) + xy
    uv = MODEL.project((pts - CAMERA) @ NADIR)
    out = image.copy()
    cv2.fillConvexPoly(out, cv2.convexHull(np.round(uv).astype(np.int32)), colour)
    return out


def window(xy, half=260):
    optical = (np.array([xy[0], xy[1], 0.24]) - CAMERA) @ NADIR
    u, v = MODEL.project(optical.reshape(1, 3))[0]
    return (int(u - half), int(v - half), 2 * half, 2 * half)


class Background(unittest.TestCase):
    def setUp(self):
        self.empty = floor()
        self.model = ClipBackground.from_frames([self.empty] * 3, alpha=0.5)

    def test_the_empty_arena_has_no_foreground(self):
        roi = window((4.0, 7.0))
        self.assertEqual(int(self.model.foreground(self.empty, roi).sum()), 0)

    def test_a_body_is_foreground_and_the_floor_around_it_is_not(self):
        roi = window((4.0, 7.0))
        frame = with_body(self.empty, (4.0, 7.0), 0.3)
        mask = self.model.foreground(frame, roi)
        truth = (frame != self.empty).any(axis=2)[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        agree = (mask == truth).mean()
        self.assertGreater(agree, 0.995)

    def test_a_global_brightness_change_is_divided_out(self):
        roi = window((4.0, 7.0))
        frame = with_body(self.empty, (4.0, 7.0), 0.3)
        reference = self.model.foreground(frame, roi)
        for gain in (0.6, 1.4):
            dimmed = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            estimate = self.model.gain(dimmed, [roi])
            self.assertAlmostEqual(estimate, gain, delta=0.05)
            mask = self.model.foreground(dimmed, roi, estimate)
            self.assertGreater((mask == reference).mean(), 0.99)
            # Without the gain the whole window would light up or vanish.
            self.assertLess((self.model.foreground(dimmed, roi, 1.0) == reference).mean(), 0.9)

    def test_learning_never_touches_the_track_windows(self):
        roi = window((4.0, 7.0))
        debris = with_body(self.empty, (4.0, 7.0), 0.3, colour=(200, 200, 200))
        before = self.model.mean.copy()
        for _ in range(2 * self.model.height // self.model.band_rows + 2):
            self.model.update(debris, exclude=[roi])
        x, y, w, h = roi
        np.testing.assert_array_equal(self.model.mean[y:y + h, x:x + w],
                                      before[y:y + h, x:x + w])
        # The same debris outside any window is learned: alpha 0.5 needs a
        # few passes over the frame to bring 170 grey levels under 12.
        for _ in range(8 * (self.model.height // self.model.band_rows + 1)):
            self.model.update(debris)
        self.assertLess(int(self.model.foreground(debris, roi).sum()), 50)

    def test_masked_pixels_never_report_foreground(self):
        roi = window((4.0, 7.0))
        mask = np.ones((1232, 1640), dtype=bool)
        model = ClipBackground.from_frames([self.empty], pixel_mask=mask)
        frame = with_body(self.empty, (4.0, 7.0), 0.3)
        self.assertEqual(int(model.foreground(frame, roi).sum()), 0)


class Silhouette(unittest.TestCase):
    def setUp(self):
        self.empty = floor()
        self.model = ClipBackground.from_frames([self.empty], alpha=0.0)

    def read(self, xy, yaw, estimator="extent"):
        observer = SilhouetteObserver(MODEL, NADIR, CAMERA, size_m=SIZE, estimator=estimator)
        roi = window(xy)
        mask = self.model.foreground(with_body(self.empty, xy, yaw), roi)
        return observer.measure(mask, roi, xy), observer

    def test_the_extent_centre_is_accurate_across_the_ray(self):
        for offset, yaw in ((0.0, 0.2), (1.2, 0.7), (2.2, -0.4)):
            xy = np.array([3.0 + offset, 6.0 + 0.5 * offset])
            with self.subTest(offset=offset):
                result, observer = self.read(xy, yaw)
                self.assertIsNotNone(result, observer.last_reason)
                error = np.array([result.x, result.y]) - xy
                radial = xy - CAMERA[:2]
                radial = radial / max(np.linalg.norm(radial), 1e-9) if offset else np.array([1.0, 0.0])
                across = abs(error @ np.array([-radial[1], radial[0]]))
                self.assertLess(across, 0.02)
                self.assertLess(np.linalg.norm(error), 0.06)

    def test_the_covariance_is_long_along_the_camera_ray(self):
        xy = np.array([5.0, 7.0])
        result, _ = self.read(xy, 0.5)
        cov = np.array(result.covariance_xy).reshape(2, 2)
        values, vectors = np.linalg.eigh(cov)
        radial = (xy - CAMERA[:2]) / np.linalg.norm(xy - CAMERA[:2])
        self.assertGreater(values[1], 4 * values[0])
        self.assertGreater(abs(vectors[:, 1] @ radial), 0.99)

    def test_the_density_centroid_is_pulled_to_the_camera(self):
        xy = np.array([5.0, 7.0])
        extent, _ = self.read(xy, 0.5, "extent")
        centroid, _ = self.read(xy, 0.5, "centroid")
        self.assertLess(np.linalg.norm([extent.x - xy[0], extent.y - xy[1]]),
                        np.linalg.norm([centroid.x - xy[0], centroid.y - xy[1]]))

    def test_a_blob_cut_by_the_window_is_refused(self):
        xy = np.array([4.0, 7.0])
        observer = SilhouetteObserver(MODEL, NADIR, CAMERA, size_m=SIZE)
        roi = window(xy, half=60)
        mask = self.model.foreground(with_body(self.empty, xy, 0.3), roi)
        self.assertIsNone(observer.measure(mask, roi, xy))
        self.assertEqual(observer.last_reason, "blob_clipped_by_window")

    def test_a_piece_smaller_than_the_body_is_refused(self):
        # What is left of the opponent after tag_rover's hull is cut out, or
        # its lit top alone on a dark floor: dataset 03, t = 99.0-99.2 s.
        xy = np.array([4.5, 7.0])
        observer = SilhouetteObserver(MODEL, NADIR, CAMERA, size_m=SIZE)
        roi = window(xy)
        piece = with_body(self.empty, xy, 0.3, size=(0.6, 0.12, 0.483))
        self.assertIsNone(observer.measure(self.model.foreground(piece, roi), roi, xy))
        self.assertEqual(observer.last_reason, "blob_smaller_than_body")
        whole = with_body(self.empty, xy, 0.3)
        self.assertIsNotNone(observer.measure(self.model.foreground(whole, roi), roi, xy))

    def test_nothing_near_the_prediction_is_nothing(self):
        observer = SilhouetteObserver(MODEL, NADIR, CAMERA, size_m=SIZE)
        roi = window((4.0, 7.0))
        mask = self.model.foreground(with_body(self.empty, (4.0, 7.0), 0.3), roi)
        self.assertIsNone(observer.measure(mask, roi, (6.0, 9.0), gate_m=0.5))
        self.assertEqual(observer.last_reason, "no_blob_in_gate")


class MaskedZones(unittest.TestCase):
    def test_rect_and_circle_with_margin(self):
        zones = [("rect", 0, 0, 1, 1), ("circle", 5, 5, 0.5)]
        inside = in_masked_zone([0.5, 1.1, 5.4, 5.6, 3.0], [0.5, 0.5, 5.0, 5.0, 3.0], zones, 0.15)
        self.assertEqual(inside.tolist(), [True, True, True, True, False])
        self.assertFalse(in_masked_zone([0.5], [0.5], []).any())


if __name__ == "__main__":
    unittest.main()
