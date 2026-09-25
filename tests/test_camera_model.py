import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization_contracts.camera_model import (
    CameraModel, CameraModelError, radial_turnover, infer_model,
)

# The IMX219-160 plumb_bob fit that shipped in the L2 world.  It is the
# regression this module exists to catch.
IMX219_K = [[488.31088823052755, 0, 647.846919881951],
            [0, 493.6294483217684, 513.414889797276],
            [0, 0, 1]]
IMX219_D = [-0.3112978153440105, 0.15431516540306534,
            -0.0014882893582261717, 0.0022164735436313475, -0.05947053168235783]
SIZE = [1280, 960]


class RadialTurnover(unittest.TestCase):
    def test_monotonic_polynomial_has_no_turnover(self):
        self.assertEqual(radial_turnover(0.0, 0.0, 0.0), (math.inf, math.inf))
        self.assertEqual(radial_turnover(0.05, 0.01, 0.0), (math.inf, math.inf))

    def test_strong_barrel_folds_over(self):
        r_u, r_d = radial_turnover(-0.3112978, 0.1543152, -0.0594705)
        self.assertAlmostEqual(r_u, 1.204, places=2)
        self.assertAlmostEqual(r_d, 0.833, places=2)


class InferModel(unittest.TestCase):
    def test_names(self):
        self.assertEqual(infer_model([0, 0, 0, 0, 0]), "pinhole")
        self.assertEqual(infer_model([]), "pinhole")
        self.assertEqual(infer_model([0.1, 0, 0, 0, 0]), "plumb_bob")
        self.assertEqual(infer_model([0.1, 0, 0, 0]), "fisheye")
        self.assertEqual(infer_model([0.1] * 8), "rational")
        with self.assertRaises(CameraModelError):
            infer_model([0.1] * 7)


class ShippedCalibrationIsRejected(unittest.TestCase):
    """The whole point of this module: this calibration must not pass."""

    def setUp(self):
        self.model = CameraModel(IMX219_K, IMX219_D, SIZE, name="camera_1")

    def test_model_cannot_reach_the_image_corners(self):
        self.assertLess(self.model.max_valid_radius, 0.9)
        self.assertGreater(self.model.frame_corner_radius(), 1.6)

    def test_validate_fails(self):
        report = self.model.validate()
        self.assertFalse(report.ok)
        self.assertLess(report.valid_area_fraction, 0.5)
        self.assertGreater(report.round_trip_p95_px, 10.0)
        with self.assertRaises(CameraModelError):
            report.raise_for_status()

    def test_points_outside_the_domain_are_flagged(self):
        centre = [self.model.K[0, 2], self.model.K[1, 2]]
        self.assertTrue(self.model.inside_valid_radius([centre])[0])
        self.assertFalse(self.model.inside_valid_radius([[0.0, 0.0]])[0])

    def test_pixel_radius_limit_matches_the_measured_ellipse(self):
        self.assertAlmostEqual(self.model.pixel_radius_limit(), 406.7, delta=2.0)


class PinholePasses(unittest.TestCase):
    def setUp(self):
        self.model = CameraModel(IMX219_K, [0.0] * 5, SIZE)

    def test_validate_passes(self):
        report = self.model.validate()
        self.assertTrue(report.ok, report.reasons)
        self.assertEqual(report.valid_area_fraction, 1.0)
        self.assertLess(report.round_trip_p95_px, 1e-6)

    def test_round_trip_projection(self):
        pts = np.array([[10.0, 10.0], [1270.0, 950.0], [640.0, 480.0]])
        rays = self.model.undistort(pts)
        back = self.model.project(np.c_[rays, np.ones(len(rays))])
        np.testing.assert_allclose(back, pts, atol=1e-6)


class FisheyePasses(unittest.TestCase):
    """Equidistant is the model a 105 deg lens actually needs."""

    def setUp(self):
        self.model = CameraModel(IMX219_K, [-0.02, 0.004, -0.001, 0.0], SIZE)

    def test_model_name_and_domain(self):
        self.assertEqual(self.model.model, "fisheye")
        self.assertTrue(math.isinf(self.model.max_valid_radius))

    def test_validate_passes_over_the_whole_frame(self):
        report = self.model.validate()
        self.assertTrue(report.ok, report.reasons)

    def test_solve_pnp_distortion_is_zero(self):
        np.testing.assert_allclose(self.model.solve_pnp_distortion(), np.zeros((5, 1)))


class Guards(unittest.TestCase):
    def test_rejects_unknown_model(self):
        with self.assertRaises(CameraModelError):
            CameraModel(IMX219_K, [0.0] * 5, SIZE, model="equirectangular")

    def test_rejects_degenerate_intrinsics(self):
        with self.assertRaises(CameraModelError):
            CameraModel([[0, 0, 1], [0, 0, 1], [0, 0, 1]], [0.0] * 5, SIZE)

    def test_rejects_nonpositive_image_size(self):
        with self.assertRaises(CameraModelError):
            CameraModel(IMX219_K, [0.0] * 5, [0, 960])


if __name__ == "__main__":
    unittest.main()
