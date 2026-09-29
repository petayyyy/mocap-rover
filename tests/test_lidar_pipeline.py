"""Unitree L2 pipeline, against ray-cast scans of the arena."""
import json
import math
import unittest
import warnings

import numpy as np

import lidar_render as lr
from localization_contracts.lidar_pipeline import (
    ArenaLidar, LidarCluster, RangeBackground, StaticVoxelMap, fit_floor,
    measurement_from_cluster, parse_laser_scan, sensor_level_points,
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

    def test_a_map_is_json_serialisable(self):
        # np.floor(...).astype(int64) keys survive the set but not json.dumps,
        # which only shows up after a full empty-arena recording.
        background = StaticVoxelMap(0.1).fit([np.array([[1.0, 2.0, 0.3],
                                                        [1.4, 2.2, 0.35]])])
        json.dumps(background.to_dict())

    def test_a_map_survives_a_round_trip(self):
        background = StaticVoxelMap(0.1).fit([np.array([[1.0, 2.0, 0.3]])])
        restored = StaticVoxelMap.from_dict(background.to_dict())
        self.assertEqual(restored.voxels, background.voxels)
        self.assertTrue(restored.contains([[1.0, 2.0, 0.3]])[0])

    def test_an_empty_map_removes_nothing(self):
        self.assertFalse(StaticVoxelMap().contains([[1.0, 2.0, 0.3]])[0])


if __name__ == "__main__":
    unittest.main()


class PerRayBackground(unittest.TestCase):
    """Range per ray, the way the real Airy pipeline subtracts the arena."""

    CRATE = ((7.0, 6.0, 0.25), (0.6, 0.6, 0.5))

    def test_a_learned_background_removes_floor_and_a_static_obstacle(self):
        empty = [lr.render(None, extra_boxes=[self.CRATE])["ranges"] for _ in range(3)]
        background = RangeBackground.build(empty)
        unit = lidar(range_background=background)
        scan = unit.scan_to_arena(lr.render((8.2, 6.0), extra_boxes=[self.CRATE]))
        near_crate = np.linalg.norm(scan.points[:, :2] - np.array([7.0, 6.0]), axis=1) < 0.45
        self.assertEqual(int(near_crate.sum()), 0)
        # Nothing but the rover survives: every return is on it.
        self.assertLess(float(np.max(np.linalg.norm(scan.points[:, :2] - [8.2, 6.0], axis=1))), 0.8)
        cluster = unit.detect(scan.points, (8.2, 6.0), 0.05, 0.5, 0)
        self.assertIsNotNone(cluster, unit.rejections)
        self.assertLess(math.hypot(cluster.x - 8.2, cluster.y - 6.0), 0.06)

    def test_the_margin_follows_the_spread_of_each_ray(self):
        grids = [np.full((2, 3), 5.0), np.full((2, 3), 5.0)]
        grids[1][0, 0] = 4.0                    # a wandering grazing ray
        background = RangeBackground.build(grids, percentile=80, min_hits=1)
        probe = np.full((2, 3), 4.7)
        mask = background.foreground(probe)
        self.assertFalse(mask[0, 0])            # inside that ray's 4 sigma
        self.assertTrue(mask[1, 1])             # 0.3 m nearer than a steady ray

    def test_a_ray_never_seen_empty_reports_nothing(self):
        grids = [np.array([[5.0, np.inf]]), np.array([[5.0, np.inf]])]
        background = RangeBackground.build(grids)
        self.assertEqual(background.foreground(np.array([[1.0, 1.0]])).tolist(), [[True, False]])

    def test_a_background_survives_a_round_trip(self):
        import tempfile
        from pathlib import Path
        background = RangeBackground.build([lr.render(None)["ranges"]] * 2, min_margin_m=0.2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bg.npz"
            background.save(path)
            again = RangeBackground.load(path)
        np.testing.assert_array_equal(again.threshold, background.threshold)
        self.assertEqual(again.min_margin_m, 0.2)


class OpponentOptions(unittest.TestCase):
    """The same detector for a taller rover whose highest part is off-centre."""

    TALL = ((6.0 + 3.0, 6.0, 0.25), (0.9, 0.52, 0.22))        # deck
    CABIN = ((6.0 + 3.0 - 0.12, 6.0, 0.41), (0.38, 0.44, 0.14))  # to 0.48 m

    def scan(self, **kwargs):
        unit = lidar(**kwargs)
        return unit, unit.scan_to_arena(lr.render(None, extra_boxes=[self.TALL, self.CABIN]))

    def test_a_body_taller_than_the_default_height_gate_needs_its_own(self):
        unit, scan = self.scan()
        self.assertIsNone(unit.detect(scan.points, (9.0, 6.0), 0.05))
        cluster = unit.detect(scan.points, (9.0, 6.0), 0.05, max_z_m=0.55)
        self.assertIsNotNone(cluster, unit.rejections)

    def test_a_thicker_top_slab_is_not_pulled_to_the_cabin(self):
        unit, scan = self.scan()
        thin = unit.detect(scan.points, (9.0, 6.0), 0.05, max_z_m=0.55)
        thick = unit.detect(scan.points, (9.0, 6.0), 0.05, max_z_m=0.55, top_slab_m=0.30)
        self.assertGreater(abs(thin.x - 9.0), 0.05)
        self.assertLess(abs(thick.x - 9.0), abs(thin.x - 9.0))

    def test_every_centre_method_is_selectable(self):
        unit, scan = self.scan()
        for method in ArenaLidar.CENTRE_METHODS:
            cluster = unit.detect(scan.points, (9.0, 6.0), 0.05, max_z_m=0.55,
                                  centre_method=method)
            self.assertIsNotNone(cluster, (method, unit.rejections))
        with self.assertRaises(ValueError):
            lidar(centre_method="mean")

    def test_a_masked_zone_hides_what_stands_in_it(self):
        crate = ((7.0, 6.0, 0.25), (0.6, 0.6, 0.5))
        unit = lidar(masked_zones=[("rect", 6.8, 5.8, 7.2, 6.2)])
        scan = unit.scan_to_arena(lr.render((8.4, 6.0), extra_boxes=[crate]))
        kept = scan.points[unit.static_mask(scan.points)]
        self.assertEqual(int((np.linalg.norm(kept[:, :2] - [7.0, 6.0], axis=1) < 0.4).sum()), 0)
        self.assertIsNotNone(unit.detect(scan.points, (8.4, 6.0), 0.05))


class FloorFit(unittest.TestCase):
    """fit_floor checks the tool, not the world: a known plane must come back."""

    @staticmethod
    def plane(tilt_deg, height, azimuth_deg=30.0, n=20000, seed=4):
        rng = np.random.default_rng(seed)
        xy = rng.uniform(-6, 6, (n, 2))
        tilt, azimuth = math.radians(tilt_deg), math.radians(azimuth_deg)
        normal = np.array([math.sin(tilt) * math.cos(azimuth),
                           math.sin(tilt) * math.sin(azimuth), math.cos(tilt)])
        z = -(height + normal[0] * xy[:, 0] + normal[1] * xy[:, 1]) / normal[2]
        points = np.column_stack([xy, z + rng.normal(0, 0.005, n)])
        wall = np.column_stack([np.full(3000, 5.9), rng.uniform(-6, 6, 3000),
                                rng.uniform(-height, -height + 0.4, 3000)])
        return np.vstack([points, wall]), normal

    def test_a_two_degree_tilt_is_recovered(self):
        points, normal = self.plane(2.0, 2.75)
        fit = fit_floor(points, expected_height=2.75)
        self.assertAlmostEqual(fit.tilt_deg, 2.0, delta=0.05)
        self.assertAlmostEqual(fit.height_m, 2.75, delta=0.01)
        self.assertGreater(float(np.dot(fit.normal, normal)), math.cos(math.radians(0.05)))
        self.assertLess(fit.residual_p95_m, 0.015)
        self.assertGreater(min(fit.span_m), 3.0)

    def test_a_level_ray_cast_floor_is_level(self):
        parsed = lr.render(None)
        fit = fit_floor(sensor_level_points(parsed, lr.SENSOR_ROTATION),
                        expected_height=float(lr.SENSOR_POSITION[2]))
        self.assertLess(fit.tilt_deg, 0.1)
        self.assertAlmostEqual(fit.height_m, 2.75, delta=0.01)

    def test_a_plane_steeper_than_the_limit_is_refused(self):
        points, _ = self.plane(15.0, 2.75)
        with self.assertRaises(ValueError):
            fit_floor(points, expected_height=2.75, max_tilt_deg=10.0)


class DefaultsAreNotCopied(unittest.TestCase):
    """A stale second copy of a default cost two Gazebo runs."""

    def test_the_cli_takes_its_defaults_from_the_module(self):
        import ast
        from pathlib import Path

        import localization_contracts.lidar_pipeline as module

        source = Path(__file__).resolve().parents[1] / "scripts" / "run_localization.py"
        tree = ast.parse(source.read_text())
        literals = {}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and str(node.args[0].value).startswith("--lidar-")):
                continue
            flag = node.args[0].value
            for keyword in node.keywords:
                if keyword.arg != "default":
                    continue
                # None means "ask the calibration", which cannot go stale.
                if (isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is not None):
                    literals[flag] = keyword.value.value
        self.assertEqual(literals, {},
                         f"these lidar flags hard-code a default: {sorted(literals)}")

    def test_the_module_default_reaches_the_object(self):
        unit = ArenaLidar(lr.SENSOR_POSITION, lr.SENSOR_ROTATION)
        from localization_contracts.lidar_pipeline import DEFAULT_MAX_USEFUL_RADIUS_M
        self.assertEqual(unit.max_useful_radius_m, DEFAULT_MAX_USEFUL_RADIUS_M)


class ContinuationPolicy(unittest.TestCase):
    """The lidar must be allowed to run in the state a gap actually produces."""

    def test_lost_bars_continuation_and_reacquiring_does_not(self):
        from localization_contracts.lidar_pipeline import may_continue
        self.assertTrue(may_continue("TRACKING"))
        self.assertTrue(may_continue("COASTING"))
        self.assertTrue(may_continue("REACQUIRING"))
        self.assertFalse(may_continue("LOST"))

    def test_the_runtime_uses_that_policy_rather_than_its_own_list(self):
        import ast
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1]
                  / "scripts" / "run_localization.py").read_text()
        self.assertIn("may_continue(tracking)", source)
        self.assertNotIn('tracking in ("LOST", "REACQUIRING")', source)

    def test_lidar_alone_holds_a_track_through_a_camera_gap(self):
        from localization_contracts.rover_filter import (
            ImmRoverFilter, Measurement, POSITION, COASTING, REACQUIRING,
        )
        ms = 1_000_000
        rover = ImmRoverFilter()
        rover.apply_group([Measurement(0, POSITION, (7.0, 6.0),
                                       (1e-5, 0.0, 0.0, 1e-5), "camera_1",
                                       "tag36h11:0", True)])
        # Marker gone. Lidar keeps arriving at 5.55 Hz.
        held, states = 0, []
        for step in range(1, 16):
            stamp = step * 180 * ms
            # Mid-revolution: what a 200 Hz publisher sees between fixes.
            states.append(rover.publish(stamp - 90 * ms)["tracking_state"])
            unit, scan = scan_for((7.0, 6.0))
            cluster = unit.detect(scan.points, (7.0, 6.0), 0.05, 0.0, stamp)
            self.assertIsNotNone(cluster, unit.rejections)
            if not may_continue_state(rover.tracking_state(stamp)):
                break
            rover.apply_group([measurement_from_cluster(cluster)])
            if rover.publish(stamp)["valid"]:
                held += 1
        # identity_max_age_s is 2 s, which at 5.55 Hz is eleven revolutions:
        # the lidar holds the track valid across that and no longer, because
        # a track running on continuation alone stops being this rover.
        self.assertGreaterEqual(held, 10)
        self.assertLessEqual(held, 12)
        self.assertEqual(rover.tracking_state(15 * 180 * ms), REACQUIRING)
        # Between fixes the track coasts rather than claiming a fresh marker.
        self.assertIn(COASTING, states)
        self.assertEqual(states[-1], REACQUIRING)


def may_continue_state(state):
    from localization_contracts.lidar_pipeline import may_continue
    return may_continue(state)


class BackgroundPerformance(unittest.TestCase):
    """A revolution arrives every 180 ms; lookup has to fit inside that."""

    def test_removal_is_fast_enough_for_a_full_revolution(self):
        import time

        rng = np.random.default_rng(0)
        occupied = rng.uniform([0, 0, 0.05], [12, 12, 0.6], size=(1000, 3))
        background = StaticVoxelMap(0.10).fit([occupied])
        cloud = rng.uniform([0, 0, 0.0], [12, 12, 0.6], size=(18432, 3))
        background.contains(cloud)
        start = time.perf_counter()
        for _ in range(5):
            mask = background.contains(cloud)
        elapsed = (time.perf_counter() - start) / 5
        self.assertLess(elapsed, 0.020, f"{elapsed*1000:.0f} ms per revolution")
        self.assertEqual(mask.shape, (18432,))

    def test_the_recorded_arena_background_loads_and_removes_the_barriers(self):
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "config" / "lidar_background.json"
        if not path.exists():
            self.skipTest("run scripts/record_lidar_background.py first")
        background = StaticVoxelMap.from_dict(json.loads(path.read_text()))
        self.assertGreater(len(background.voxels), 100)
        # The barriers sit at x or y = 0.08, 0.16 wide and 0.4 tall.
        barrier = np.array([[0.08, 5.0, 0.30], [11.92, 5.0, 0.30],
                            [5.0, 0.08, 0.30], [5.0, 11.92, 0.30]])
        self.assertTrue(background.contains(barrier).all())
        # The middle of the arena at rover height must stay visible.
        self.assertFalse(background.contains([[6.0, 6.0, 0.30]])[0])
