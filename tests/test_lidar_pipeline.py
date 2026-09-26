"""Unitree L2 pipeline, against ray-cast scans of the arena."""
import math
import unittest
import warnings

import numpy as np

import lidar_render as lr
from localization_contracts.lidar_pipeline import (
    ArenaLidar, LidarCluster, StaticVoxelMap, measurement_from_cluster,
    parse_laser_scan,
)
from localization_contracts.rover_filter import POSITION

warnings.filterwarnings("ignore", category=RuntimeWarning)


def lidar(**kwargs):
    return ArenaLidar(lr.SENSOR_POSITION, lr.SENSOR_ROTATION, **kwargs)


def scan_for(xy, yaw=0.0, extra_boxes=(), **kwargs):
    unit = lidar(**kwargs)
    parsed = lr.render(xy, yaw, extra_boxes)
    return unit, unit.scan_to_arena(parsed)


class FakeStamp:
    sec, nsec = 12, 340_000_000


class FakeScan:
    header = type("H", (), {"stamp": FakeStamp})()
    count = 4
    vertical_count = 2
    angle_min, angle_step = -math.pi, math.pi / 2
    vertical_angle_min, vertical_angle_step = 0.0, 0.1
    range_min, range_max = 0.05, 30.0
    ranges = [1.0] * 8


class Parsing(unittest.TestCase):
    def test_organized_grid_and_stamp(self):
        parsed = parse_laser_scan(FakeScan())
        self.assertEqual(parsed["ranges"].shape, (2, 4))
        self.assertEqual(parsed["stamp_ns"], 12_340_000_000)
        np.testing.assert_allclose(parsed["azimuth"],
                                   [-math.pi, -math.pi / 2, 0, math.pi / 2])

    def test_a_mismatched_range_count_is_refused(self):
        bad = FakeScan()
        bad.ranges = [1.0] * 7
        with self.assertRaises(ValueError):
            parse_laser_scan(bad)


class Conversion(unittest.TestCase):
    def test_the_floor_lands_at_z_zero(self):
        unit, scan = scan_for(None)
        floor = scan.points[np.abs(scan.points[:, 2]) < 0.02]
        self.assertGreater(len(floor), 1000)
        self.assertLess(abs(float(np.median(scan.points[:, 2]))), 0.02)

    def test_the_rover_appears_at_its_own_height(self):
        unit, scan = scan_for((7.0, 6.0))
        band = scan.points[unit.static_mask(scan.points)]
        self.assertGreater(len(band), 50)
        # Base 0.14 + top board 0.2254.
        self.assertAlmostEqual(float(np.max(band[:, 2])), 0.365, delta=0.02)

    def test_the_arena_floor_and_walls_are_filtered_out(self):
        unit, scan = scan_for(None)
        self.assertEqual(int(unit.static_mask(scan.points).sum()), 0)


class Deskew(unittest.TestCase):
    def test_a_gz_scan_is_instantaneous_so_deskew_does_nothing(self):
        unit, scan = scan_for((7.0, 6.0))
        np.testing.assert_array_equal(unit.deskew(scan, (11.0, 0.0, 0.0)),
                                      scan.points)

    def test_a_rotating_sweep_corrects_target_motion(self):
        unit, scan = scan_for((7.0, 6.0), sweep_duration_s=1 / 5.55)
        moved = unit.deskew(scan, (11.11, 0.0, 0.0))
        self.assertFalse(np.array_equal(moved, scan.points))
        # Offsets are centred on the stamp, so the correction is antisymmetric
        # and cannot shift the scan as a whole.
        self.assertLess(abs(float(np.mean(moved[:, 0] - scan.points[:, 0]))), 0.02)
        span = scan.offsets_ns.max() - scan.offsets_ns.min()
        self.assertAlmostEqual(span / 1e9, 1 / 5.55, delta=0.01)


class Detection(unittest.TestCase):
    def test_the_rover_is_found_and_measured(self):
        unit, scan = scan_for((7.5, 6.5))
        cluster = unit.detect(scan.points, (7.5, 6.5), 0.05, 0.5, 99)
        self.assertIsNotNone(cluster, unit.rejections)
        self.assertLess(math.hypot(cluster.x - 7.5, cluster.y - 6.5), 0.05)
        self.assertEqual(cluster.stamp_ns, 99)
        self.assertGreater(cluster.points, 20)

    def test_accuracy_over_the_arena(self):
        errors, sigmas = [], []
        for x in np.arange(1.0, 11.1, 1.0):
            for y in np.arange(1.0, 11.1, 1.0):
                unit, scan = scan_for((x, y))
                cluster = unit.detect(scan.points, (x, y), 0.05, 0.5, 0)
                if cluster is None:
                    continue
                errors.append(math.hypot(cluster.x - x, cluster.y - y))
                sigmas.append(cluster.sigma_m)
        self.assertGreater(len(errors), 80)
        self.assertLess(float(np.percentile(errors, 50)), 0.04)
        self.assertLess(float(np.percentile(errors, 95)), 0.09)
        # The reported sigma must not claim more than the estimator delivers.
        self.assertGreater(float(np.median(sigmas)), float(np.percentile(errors, 50)))

    def test_the_estimate_is_not_pulled_toward_the_sensor(self):
        # A radial fan samples the near side of a flat top far more densely,
        # so a density-weighted centre drifts sensor-ward with range.
        toward = []
        for x in (7.0, 8.0, 9.0):
            unit, scan = scan_for((x, 6.0))
            cluster = unit.detect(scan.points, (x, 6.0), 0.05, 0.5, 0)
            self.assertIsNotNone(cluster, unit.rejections)
            toward.append(x - cluster.x)        # positive means sensor-ward
        self.assertLess(max(toward), 0.05)

    def test_range_is_limited_by_point_count_not_by_a_fixed_radius(self):
        # The SDF fires 288x64 rays; the real L2 delivers about 4.7x fewer
        # points. Decimating to that density must make the same code stop
        # accepting distant clusters, with no threshold retuned.
        for distance, dense_ok, sparse_ok in ((2.0, True, True), (5.0, True, False)):
            with self.subTest(distance=distance):
                xy = (6.0 + distance, 6.0)
                unit, scan = scan_for(xy)
                self.assertEqual(
                    unit.detect(scan.points, xy, 0.05, 0.5, 0) is not None, dense_ok,
                    unit.rejections)
                sparse = lidar()
                parsed = lr.render(xy)
                thinned = parsed["ranges"].copy()
                thinned[::2, :] = np.inf
                thinned[:, ::2] = np.inf          # keep 1 ray in 4, ~ hardware
                parsed["ranges"] = thinned
                points = sparse.scan_to_arena(parsed).points
                self.assertEqual(
                    sparse.detect(points, xy, 0.05, 0.5, 0) is not None, sparse_ok,
                    sparse.rejections)

    def test_a_hard_radius_still_bounds_the_search(self):
        unit = lidar(max_useful_radius_m=2.0)
        _, scan = scan_for((9.0, 6.0))
        self.assertIsNone(unit.detect(scan.points, (9.0, 6.0), 0.05, 0.5, 0))
        self.assertIn("beyond_useful_radius", unit.rejections)

    def test_an_empty_arena_yields_nothing(self):
        unit, scan = scan_for(None)
        self.assertIsNone(unit.detect(scan.points, (7.0, 6.0), 0.05, 0.5, 0))

    def test_a_prediction_far_from_the_rover_finds_nothing(self):
        unit, scan = scan_for((7.5, 6.0))
        self.assertIsNone(unit.detect(scan.points, (4.5, 7.5), 0.05, 0.5, 0))

    def test_an_object_of_the_wrong_size_is_rejected(self):
        wall = ((7.0, 6.0, 0.25), (2.4, 0.3, 0.5))
        unit, scan = scan_for(None, extra_boxes=[wall])
        self.assertIsNone(unit.detect(scan.points, (7.0, 6.0), 0.05, 0.5, 0))
        self.assertIn(set(unit.rejections),
                      [{"merged_cluster"}, {"no_rover_shaped_cluster"}])


class SecondRover(unittest.TestCase):
    """The case the whole identity design exists for."""

    def rival_at(self, distance):
        body = ((7.0 + distance, 6.0, 0.26), (0.72, 0.52, 0.18))
        plate = ((7.0 + distance, 6.0, 0.36), (0.50, 0.50, 0.01))
        return scan_for((7.0, 6.0), extra_boxes=[body, plate])

    def test_a_touching_rival_blocks_the_update(self):
        # At 0.8 m the two bodies leave an 8 cm gap, which is inside the
        # clustering cell, so they arrive as one blob. Its extent gives it
        # away; reporting its centre would put the track 0.1 m off.
        unit, scan = self.rival_at(0.8)
        self.assertIsNone(unit.detect(scan.points, (7.0, 6.0), 0.05, 0.5, 0))
        self.assertIn("merged_cluster", unit.rejections)

    def test_two_separated_rivals_in_a_wide_gate_block_the_update(self):
        # A wide gate is what an uncertain track produces, and it is exactly
        # when picking the nearer of two candidates would swap rovers.
        unit, scan = self.rival_at(1.5)
        self.assertIsNone(unit.detect(scan.points, (7.0, 6.0), 0.5, 0.5, 0))
        self.assertIn("ambiguous_cluster", unit.rejections)

    def test_a_distant_rival_does_not_block_anything(self):
        unit, scan = self.rival_at(2.6)
        cluster = unit.detect(scan.points, (7.0, 6.0), 0.05, 0.5, 0)
        self.assertIsNotNone(cluster, unit.rejections)
        self.assertLess(math.hypot(cluster.x - 7.0, cluster.y - 6.0), 0.06)

    def test_the_gate_does_not_widen_to_half_a_revolution(self):
        # Sizing the gate from half the 180 ms period instead of the sector
        # dwell would add a metre of radius at speed and swallow the rival.
        unit = lidar()
        self.assertLess(unit.gate_radius(0.05, 11.11), 1.0)


class Measurements(unittest.TestCase):
    def test_a_cluster_becomes_a_continuation_only_measurement(self):
        cluster = LidarCluster(3.0, 4.0, 0.36, 40, 0.7, 0.5, 0.02, 0.03, 555, 0)
        measurement = measurement_from_cluster(cluster)
        self.assertEqual(measurement.kind, POSITION)
        self.assertEqual(measurement.source, "lidar")
        self.assertEqual(measurement.stamp_ns, 555)
        self.assertIsNone(measurement.identity)
        self.assertFalse(measurement.confirms_identity)
        np.testing.assert_allclose(measurement.as_arrays()[1],
                                   np.diag([0.03 ** 2, 0.03 ** 2]))


class Background(unittest.TestCase):
    def test_a_learned_map_removes_a_static_obstacle(self):
        crate = ((7.0, 6.0, 0.25), (0.6, 0.6, 0.5))
        empty, empty_scan = scan_for(None, extra_boxes=[crate])
        occupied = empty_scan.points[empty.static_mask(empty_scan.points)]
        self.assertGreater(len(occupied), 20)

        background = StaticVoxelMap(0.08).fit([occupied])
        unit, scan = scan_for((8.2, 6.0), extra_boxes=[crate])
        unit.background = background
        remaining = scan.points[unit.static_mask(scan.points)]
        near_crate = np.linalg.norm(remaining[:, :2] - np.array([7.0, 6.0]), axis=1) < 0.45
        self.assertEqual(int(near_crate.sum()), 0)
        cluster = unit.detect(scan.points, (8.2, 6.0), 0.05, 0.5, 0)
        self.assertIsNotNone(cluster, unit.rejections)

    def test_a_map_survives_a_round_trip(self):
        background = StaticVoxelMap(0.1).fit([np.array([[1.0, 2.0, 0.3]])])
        restored = StaticVoxelMap.from_dict(background.to_dict())
        self.assertEqual(restored.voxels, background.voxels)
        self.assertTrue(restored.contains([[1.0, 2.0, 0.3]])[0])

    def test_an_empty_map_removes_nothing(self):
        self.assertFalse(StaticVoxelMap().contains([[1.0, 2.0, 0.3]])[0])


if __name__ == "__main__":
    unittest.main()
