"""ROI detection and the absolute minimum-perimeter threshold."""
import time
import unittest

import cv2
import numpy as np

import arena_render as ar
from localization_contracts.detector import AprilTagImageDetector


def frame():
    return ar.render((9.0, 9.0, 2.9), ar.camera_rotation(), (8.2, 9.3), yaw=0.4)


class RoiDetection(unittest.TestCase):
    def setUp(self):
        self.frame = frame()
        self.detector = AprilTagImageDetector(allowed_ids=(0, 1))
        self.full = self.detector.detect(self.frame)
        self.assertEqual(len(self.full), 1)
        self.centre = np.asarray(self.full[0].corners).mean(axis=0)

    def roi(self, size):
        return (int(self.centre[0] - size // 2), int(self.centre[1] - size // 2),
                size, size)

    def test_corners_come_back_in_full_image_coordinates(self):
        found = self.detector.detect(self.frame, roi=self.roi(240))
        self.assertEqual(len(found), 1)
        np.testing.assert_allclose(np.asarray(found[0].corners),
                                   np.asarray(self.full[0].corners), atol=1.0)

    def test_the_window_and_sweep_profiles_find_the_same_corners(self):
        window = AprilTagImageDetector(allowed_ids=(0, 1), profile="window")
        sweep = AprilTagImageDetector(allowed_ids=(0, 1), profile="sweep")
        for found in (window.detect(self.frame, roi=self.roi(200)), sweep.detect(self.frame)):
            self.assertEqual(len(found), 1)
            np.testing.assert_allclose(np.asarray(found[0].corners),
                                       np.asarray(self.full[0].corners), atol=0.5)

    def test_an_rgb_window_is_cropped_before_it_is_converted(self):
        rgb = np.repeat(self.frame[:, :, None], 3, axis=2)
        found = self.detector.detect(rgb, roi=self.roi(240))
        np.testing.assert_allclose(np.asarray(found[0].corners),
                                   np.asarray(self.full[0].corners), atol=1.0)

    def test_an_unknown_profile_is_refused(self):
        with self.assertRaises(ValueError):
            AprilTagImageDetector(profile="turbo")

    def test_a_roi_that_misses_the_marker_finds_nothing(self):
        self.assertEqual(self.detector.detect(self.frame, roi=(0, 0, 160, 160)), ())

    def test_a_roi_clipped_at_the_frame_edge_is_still_valid(self):
        self.assertIsInstance(self.detector.detect(self.frame, roi=(-50, -50, 200, 200)), tuple)
        self.assertIsInstance(self.detector.detect(self.frame, roi=(1200, 900, 400, 400)), tuple)

    def test_the_minimum_candidate_perimeter_is_held_in_pixels(self):
        # The rate OpenCV stores is relative to the image, so it has to change
        # when the detector looks at a smaller picture or the threshold moves
        # with the ROI size.
        self.detector.detect(self.frame)
        full_rate = self.detector._parameters.minMarkerPerimeterRate
        self.detector.detect(self.frame, roi=self.roi(240))
        roi_rate = self.detector._parameters.minMarkerPerimeterRate
        self.assertAlmostEqual(full_rate * 1280, self.detector.min_perimeter_px, places=6)
        self.assertAlmostEqual(roi_rate * 240, self.detector.min_perimeter_px, places=6)
        self.assertGreater(roi_rate, full_rate)

    def test_roi_detection_is_much_cheaper_than_the_full_frame(self):
        roi = self.roi(240)
        self.detector.detect(self.frame)
        self.detector.detect(self.frame, roi=roi)

        def timed(**kwargs):
            start = time.perf_counter()
            for _ in range(5):
                self.detector.detect(self.frame, **kwargs)
            return (time.perf_counter() - start) / 5

        self.assertLess(timed(roi=roi), 0.5 * timed())


class InvertedMarkers(unittest.TestCase):
    def test_an_inverted_marker_is_not_reported(self):
        # A ceiling rig never sees one, and accepting it yields corners that
        # wind the other way, which no rigid planar pose can fit.
        inverted = 255 - frame()
        self.assertEqual(AprilTagImageDetector(allowed_ids=(0, 1)).detect(inverted), ())


class SmallMarkers(unittest.TestCase):
    def test_a_twenty_pixel_marker_is_still_found(self):
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 20)
        canvas = np.full((480, 640), 255, np.uint8)
        canvas[230:250, 310:330] = marker
        found = AprilTagImageDetector(allowed_ids=(0,)).detect(canvas)
        self.assertEqual([d.tag_id for d in found], [0])

    def test_the_threshold_can_be_raised_to_exclude_them(self):
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 20)
        canvas = np.full((480, 640), 255, np.uint8)
        canvas[230:250, 310:330] = marker
        strict = AprilTagImageDetector(allowed_ids=(0,), min_perimeter_px=200.0)
        self.assertEqual(strict.detect(canvas), ())


if __name__ == "__main__":
    unittest.main()
