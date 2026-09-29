"""The shared pixel -> plane geometry used by marker, silhouette and box paths."""
import math
import unittest

import numpy as np

from localization_contracts.camera_model import CameraModel
from localization_contracts.ray_plane import (
    pixels_to_plane, quad_centre_ray, ray_plane, rays_plane,
)

K = [[734.5, 0, 820], [0, 734.5, 616], [0, 0, 1]]
FISHEYE = CameraModel(K, [0.0] * 4, (1640, 1232))
NADIR = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=float)
CAMERA = np.array([3.0, 6.0, 2.9])


class RayPlane(unittest.TestCase):
    def project(self, world):
        optical = (np.asarray(world, dtype=float) - CAMERA) @ NADIR
        return FISHEYE.project(optical)

    def test_a_pixel_returns_to_the_point_it_came_from_through_a_fisheye(self):
        points = np.array([[3.2, 6.1, 0.483], [5.0, 8.5, 0.483], [1.0, 3.0, 0.483]])
        back, valid = pixels_to_plane(FISHEYE, NADIR, CAMERA, self.project(points), 0.483)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(back, points, atol=1e-6)

    def test_the_scalar_and_vector_forms_agree(self):
        rays = np.array([[0.1, -0.2], [0.8, 0.3]])
        points, valid = rays_plane(rays, NADIR, CAMERA, 0.3654)
        for ray, point in zip(rays, points):
            single, incidence = ray_plane(ray, NADIR, CAMERA, 0.3654)
            np.testing.assert_allclose(single, point, atol=1e-12)
            self.assertLess(incidence, math.pi / 2)
        self.assertTrue(valid.all())

    def test_a_plane_behind_the_camera_is_refused(self):
        self.assertEqual(ray_plane((0.0, 0.0), NADIR, CAMERA, 3.5), (None, None))
        self.assertFalse(rays_plane([[0.0, 0.0]], NADIR, CAMERA, 3.5)[1][0])

    def test_the_diagonal_crossing_is_the_true_centre_off_axis(self):
        s, centre = 0.2, np.array([5.2, 8.0, 0.3654])
        corners = centre + np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
        ray = quad_centre_ray(FISHEYE, self.project(corners))
        point, _ = ray_plane(ray, NADIR, CAMERA, 0.3654)
        self.assertLess(np.linalg.norm(point[:2] - centre[:2]), 1e-6)
        mean_ray = FISHEYE.undistort(self.project(corners).mean(axis=0, keepdims=True))[0]
        biased, _ = ray_plane(mean_ray, NADIR, CAMERA, 0.3654)
        self.assertGreater(np.linalg.norm(biased[:2] - centre[:2]), 0.005)


if __name__ == "__main__":
    unittest.main()
