import math
import unittest

import numpy as np

from localization_contracts.rover_filter import (
    POSITION, VELOCITY, YAW_ONLY, TRACKING, COASTING, REACQUIRING, LOST,
    AsyncObservationBuffer, CoordinatedTurnModel, ImmRoverFilter, Measurement,
    MotionModel, measurement_from_observation, reject_position_outliers, wrap,
)

MS = 1_000_000


def position(t_ms, x, y, source="camera_1", sigma=0.02, identity="tag36h11:0",
             confirms=True):
    return Measurement(int(t_ms * MS), POSITION, (x, y),
                       (sigma ** 2, 0.0, 0.0, sigma ** 2), source, identity, confirms)


def yaw(t_ms, angle, source="camera_1", sigma=math.radians(2)):
    return Measurement(int(t_ms * MS), YAW_ONLY, (angle,), (sigma ** 2,), source,
                       "tag36h11:0", False)


class Buffer(unittest.TestCase):
    """A group must close on a timeout, never on a slow camera reporting."""

    def setUp(self):
        self.buffer = AsyncObservationBuffer(window_ns=12 * MS)

    def test_group_closes_on_timeout_without_the_other_cameras(self):
        self.buffer.push(position(0, 1.0, 1.0, "camera_1"))
        self.assertEqual(self.buffer.drain(5 * MS), [])
        groups = self.buffer.drain(20 * MS)
        self.assertEqual(len(groups), 1)
        self.assertEqual([m.source for m in groups[0]], ["camera_1"])

    def test_near_simultaneous_cameras_land_in_one_group(self):
        for index, source in enumerate(("camera_1", "camera_2", "camera_5")):
            self.buffer.push(position(index * 3, 1.0, 1.0, source))
        groups = self.buffer.drain(40 * MS)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)

    def test_measurements_past_the_window_start_a_new_group(self):
        self.buffer.push(position(0, 1.0, 1.0, "camera_1"))
        self.buffer.push(position(30, 1.1, 1.0, "camera_2"))
        groups = self.buffer.drain(100 * MS)
        self.assertEqual([[m.source for m in g] for g in groups],
                         [["camera_1"], ["camera_2"]])

    def test_one_camera_contributes_one_measurement_per_kind(self):
        self.buffer.push(position(0, 1.0, 1.0, "camera_1", sigma=0.05))
        self.buffer.push(position(1, 1.0, 1.0, "camera_1", sigma=0.01))
        self.buffer.push(yaw(0, 0.2, "camera_1"))
        group = self.buffer.drain(100 * MS)[0]
        self.assertEqual(len(group), 2)
        chosen = next(m for m in group if m.kind == POSITION)
        self.assertAlmostEqual(chosen.covariance[0], 0.01 ** 2)
        self.assertEqual(self.buffer.dropped_duplicate, 1)

    def test_a_zero_window_groups_only_simultaneous_measurements(self):
        # Used to loop forever: no measurement satisfied stamp < oldest + 0.
        buffer = AsyncObservationBuffer(window_ns=0)
        buffer.push(position(0, 1.0, 1.0, "camera_1"))
        buffer.push(position(0, 1.0, 1.0, "camera_2"))
        buffer.push(position(1, 1.0, 1.0, "camera_3"))
        groups = buffer.drain(10 * MS)
        self.assertEqual([[m.source for m in g] for g in groups],
                         [["camera_1", "camera_2"], ["camera_3"]])
        self.assertEqual(buffer.pending, [])

    def test_force_releases_everything(self):
        self.buffer.push(position(0, 1.0, 1.0))
        self.assertEqual(len(self.buffer.drain(0, force=True)), 1)


class OutlierRejection(unittest.TestCase):
    def test_median_test_needs_a_majority(self):
        group = [position(0, 1.0, 1.0, "camera_1"), position(0, 9.0, 9.0, "camera_2")]
        kept, rejected = reject_position_outliers(group)
        self.assertEqual(len(kept), 2)
        self.assertEqual(rejected, [])

    def test_lone_disagreeing_camera_is_dropped(self):
        group = [position(0, 1.00, 1.0, "camera_1"),
                 position(0, 1.02, 1.0, "camera_2"),
                 position(0, 6.00, 1.0, "camera_5")]
        kept, rejected = reject_position_outliers(group)
        self.assertEqual([m.source for m in rejected], ["camera_5"])
        self.assertEqual(len(kept), 2)

    def test_yaw_measurements_survive_the_position_test(self):
        group = [position(0, 1.0, 1.0, "camera_1"),
                 position(0, 1.0, 1.0, "camera_2"),
                 position(0, 6.0, 1.0, "camera_5"),
                 yaw(0, 0.5, "camera_1")]
        kept, _ = reject_position_outliers(group)
        self.assertEqual(sum(m.kind == YAW_ONLY for m in kept), 1)


class Identity(unittest.TestCase):
    def setUp(self):
        self.filter = ImmRoverFilter()

    def test_only_a_marker_can_create_a_track(self):
        lidar = Measurement(0, POSITION, (3.0, 3.0), (0.01, 0.0, 0.0, 0.01),
                            "lidar", None, False)
        self.assertFalse(self.filter.apply_group([lidar]))
        self.assertFalse(self.filter.initialized)
        self.assertTrue(self.filter.apply_group([position(0, 3.0, 3.0)]))
        self.assertTrue(self.filter.initialized)

    def test_lidar_continues_a_track_the_marker_created(self):
        self.filter.apply_group([position(0, 3.0, 3.0)])
        lidar = Measurement(50 * MS, POSITION, (3.05, 3.0), (0.01, 0.0, 0.0, 0.01),
                            "lidar", None, False)
        self.assertTrue(self.filter.apply_group([lidar]))
        self.assertIn("lidar", self.filter.publish(50 * MS)["sources"])

    def test_a_second_marker_id_cannot_take_the_track(self):
        self.filter.apply_group([position(0, 3.0, 3.0, identity="tag36h11:0")])
        rival = position(50, 3.1, 3.0, "camera_2", identity="tag36h11:2")
        self.assertFalse(self.filter.apply_group([rival]))
        self.assertGreater(self.filter.id_rejections, 0)
        self.assertEqual(self.filter.identity, "tag36h11:0")

    def test_the_underside_marker_keeps_the_track_after_a_flip(self):
        """A flip swaps which marker is visible; the rover is still the rover.

        Both ids name one object, so the identity lock must not read the
        underside marker as a rival and reject every fix the only visible
        marker can still give.
        """
        flipping = ImmRoverFilter(identity_aliases={"tag36h11:0", "tag36h11:1"})
        flipping.apply_group([position(0, 6.0, 6.0, identity="tag36h11:0")])
        t = 0
        while t < 3000:
            t += 30
            flipping.apply_group([position(t, 6.0, 6.0, identity="tag36h11:1")])
        self.assertEqual(flipping.id_rejections, 0)
        self.assertEqual(flipping.tracking_state(t * MS), TRACKING)
        self.assertEqual(flipping.identity, "tag36h11:1")

    def test_an_unlisted_marker_still_cannot_take_the_track(self):
        flipping = ImmRoverFilter(identity_aliases={"tag36h11:0", "tag36h11:1"})
        flipping.apply_group([position(0, 3.0, 3.0, identity="tag36h11:0")])
        rival = position(50, 3.1, 3.0, "camera_2", identity="tag36h11:7")
        self.assertFalse(flipping.apply_group([rival]))
        self.assertGreater(flipping.id_rejections, 0)

    def test_a_stale_identity_forces_reacquiring_even_while_lidar_tracks(self):
        self.filter.apply_group([position(0, 3.0, 3.0)])
        t = 0
        while t < 2400:
            t += 100
            lidar = Measurement(t * MS, POSITION, (3.0 + 0.001 * t, 3.0),
                                (0.01, 0.0, 0.0, 0.01), "lidar", None, False)
            self.filter.apply_group([lidar])
        published = self.filter.publish(t * MS)
        self.assertEqual(published["tracking_state"], REACQUIRING)
        self.assertFalse(published["valid"])


class IdentityHold(unittest.TestCase):
    """Past identity_max_age the identity is held only while the lidar confirms."""

    def run_lidar(self, until_ms, hold_s=10.0, revoke_at_ms=None, lidar_until_ms=None):
        f = ImmRoverFilter(identity_max_age_s=2.0, identity_hold_max_s=hold_s)
        f.apply_group([position(0, 3.0, 3.0)])
        t = 0
        while t < until_ms:
            t += 100
            if lidar_until_ms is None or t <= lidar_until_ms:
                f.apply_group([position(t, 3.0, 3.0, source="lidar", identity=None,
                                        confirms=False)])
            if revoke_at_ms is not None and t == revoke_at_ms:
                f.revoke_identity_hold()
        return f, t

    def test_a_lidar_confirmed_track_keeps_its_identity(self):
        f, t = self.run_lidar(5000)
        self.assertEqual(f.tracking_state(t * MS), TRACKING)

    def test_without_the_hold_the_old_limit_stands(self):
        f, t = self.run_lidar(5000, hold_s=0.0)
        self.assertEqual(f.tracking_state(t * MS), REACQUIRING)

    def test_an_encounter_ends_the_hold_until_the_next_marker(self):
        f, t = self.run_lidar(5000, revoke_at_ms=3000)
        self.assertEqual(f.tracking_state(t * MS), REACQUIRING)
        f.apply_group([position(t + 10, 3.0, 3.0)])            # the marker again
        self.assertTrue(f.publish((t + 10) * MS)["valid"])

    def test_the_hold_needs_the_lidar_every_scan(self):
        f, t = self.run_lidar(5000, lidar_until_ms=3000)
        self.assertEqual(f.tracking_state(3200 * MS), COASTING)   # 200 ms < 250 ms
        self.assertEqual(f.tracking_state(3300 * MS), REACQUIRING)

    def test_the_hold_has_a_ceiling(self):
        f, t = self.run_lidar(12500)
        self.assertEqual(f.tracking_state(t * MS), REACQUIRING)


class HeadingFromVelocity(unittest.TestCase):
    def drive(self, **kwargs):
        f = ImmRoverFilter(**kwargs)
        f.apply_group([position(0, 2.0, 3.0), yaw(0, 0.6)])     # marker: yaw 0.6 rad off
        for step in range(1, 120):
            t = step * 12
            f.apply_group([position(t, 2.0 + 2.0 * t / 1000.0, 3.0, source="lidar",
                                    identity=None, confirms=False)])
        return f.publish(119 * 12 * MS)

    def test_a_moving_track_without_a_marker_turns_to_its_velocity(self):
        published = self.drive()
        self.assertLess(abs(published["state"]["yaw"]), math.radians(3))
        self.assertLess(published["yaw_age_ms"], 20)

    def test_disabled_the_yaw_stays_where_the_marker_left_it(self):
        published = self.drive(heading_after_ms=0)
        self.assertGreater(abs(published["state"]["yaw"]), 0.3)
        self.assertGreater(published["yaw_age_ms"], 1000)


class TrackingStates(unittest.TestCase):
    def setUp(self):
        self.filter = ImmRoverFilter(coast_ms=300, lost_ms=1500)
        self.filter.apply_group([position(0, 3.0, 3.0)])

    def test_states_follow_measurement_age_not_a_timeout_knob(self):
        self.assertEqual(self.filter.publish(0)["tracking_state"], TRACKING)
        self.assertEqual(self.filter.publish(100 * MS)["tracking_state"], COASTING)
        self.assertEqual(self.filter.publish(500 * MS)["tracking_state"], REACQUIRING)
        self.assertEqual(self.filter.publish(2000 * MS)["tracking_state"], LOST)

    def test_reacquiring_and_lost_are_not_published_as_valid(self):
        self.assertTrue(self.filter.publish(100 * MS)["valid"])
        self.assertFalse(self.filter.publish(500 * MS)["valid"])
        self.assertFalse(self.filter.publish(2000 * MS)["valid"])

    def test_covariance_grows_while_coasting(self):
        tight = self.filter.publish(0)["covariance"][0][0]
        loose = self.filter.publish(250 * MS)["covariance"][0][0]
        self.assertGreater(loose, tight)

    def test_a_heading_alone_does_not_keep_the_track_valid(self):
        # Dataset 03, t = 100 s: silhouettes kept sending the long axis while
        # every position was gated, and the track stayed "valid" for 2.9 s.
        t = 0
        while t < 500:
            t += 12
            self.filter.apply_group([yaw(t, 0.0, source="camera_4")])
        published = self.filter.publish(t * MS)
        self.assertEqual(published["tracking_state"], REACQUIRING)
        self.assertFalse(published["valid"])
        self.assertGreaterEqual(published["measurement_age_ms"], 500)

    def test_a_heading_does_not_shorten_the_speed_test_for_a_position(self):
        # Position last fixed at t = 0; a heading at 190 ms must not make a
        # 0.5 m correction at 200 ms read as 0.5 m in 10 ms.
        self.filter.apply_group([yaw(190, 0.0, source="camera_4")])
        taken = self.filter.apply_group([position(200, 3.5, 3.0, source="camera_4",
                                                  identity=None, confirms=False)])
        self.assertEqual([m.kind for m in taken], [POSITION])

    def test_publishing_does_not_change_the_filter(self):
        before = (self.filter.x.copy(), self.filter.P.copy(), self.filter.stamp_ns)
        for t in (5, 10, 15, 20):
            self.filter.publish(t * MS)
        np.testing.assert_array_equal(self.filter.x, before[0])
        np.testing.assert_array_equal(self.filter.P, before[1])
        self.assertEqual(self.filter.stamp_ns, before[2])
        # A measurement captured before the last tick is not "late".
        self.filter.apply_group([position(12, 3.01, 3.0)])
        self.assertEqual(self.filter.out_of_sequence, 0)

    def test_an_uninitialised_filter_reports_lost_not_a_pose(self):
        published = ImmRoverFilter().publish(0)
        self.assertEqual(published["tracking_state"], LOST)
        self.assertIsNone(published["state"])


class Estimation(unittest.TestCase):
    def test_constant_velocity_is_recovered(self):
        f = ImmRoverFilter()
        for step in range(40):
            t = step * 10
            f.apply_group([position(t, 1.0 + 2.0 * t / 1000.0, 1.0)])
        state = f.publish(390 * MS)["state"]
        self.assertAlmostEqual(state["vx"], 2.0, delta=0.15)
        self.assertAlmostEqual(state["vy"], 0.0, delta=0.15)

    def test_prediction_advances_the_pose_between_measurements(self):
        f = ImmRoverFilter()
        for step in range(40):
            t = step * 10
            f.apply_group([position(t, 2.0 * t / 1000.0, 0.0)])
        near = f.publish(390 * MS)["state"]["x"]
        later = f.publish(440 * MS)["state"]["x"]
        self.assertAlmostEqual(later - near, 0.1, delta=0.03)

    def test_yaw_is_tracked_separately_from_position(self):
        f = ImmRoverFilter()
        for step in range(30):
            t = step * 10
            f.apply_group([position(t, 1.0, 1.0), yaw(t, wrap(0.5 + 1.0 * t / 1000.0))])
        state = f.publish(290 * MS)["state"]
        self.assertAlmostEqual(state["yaw"], wrap(0.5 + 0.29), delta=0.08)
        self.assertAlmostEqual(state["yaw_rate"], 1.0, delta=0.3)

    def test_speed_is_clamped_to_the_physical_limit(self):
        f = ImmRoverFilter(max_speed_mps=11.2)
        f.apply_group([position(0, 0.5, 0.5)])
        f.apply_group([position(10, 2.0, 0.5)])
        state = f.publish(10 * MS)["state"]
        self.assertLessEqual(math.hypot(state["vx"], state["vy"]), 11.2 + 1e-6)

    def test_two_cameras_beat_one(self):
        def spread(n_cameras):
            f = ImmRoverFilter()
            rng = np.random.default_rng(11)
            for step in range(30):
                t = step * 10
                group = [position(t, 3.0 + rng.normal(0, 0.02),
                                  3.0 + rng.normal(0, 0.02), f"camera_{c}")
                         for c in range(n_cameras)]
                f.apply_group(group)
            return f.publish(290 * MS)["covariance"][0][0]
        self.assertLess(spread(3), spread(1))

    def test_measurements_outside_the_arena_are_refused(self):
        f = ImmRoverFilter()
        f.apply_group([position(0, 3.0, 3.0)])
        self.assertFalse(f.apply_group([position(20, 40.0, 3.0)]))

    def test_a_teleport_is_refused_on_physical_speed(self):
        f = ImmRoverFilter()
        f.apply_group([position(0, 1.0, 1.0)])
        self.assertFalse(f.apply_group([position(10, 11.0, 1.0)]))


class OutOfSequence(unittest.TestCase):
    """A late measurement must be replayed, not dropped and not misapplied."""

    def feed(self, order):
        f = ImmRoverFilter()
        f.apply_group([position(0, 0.0, 1.0)])
        for t in order:
            f.apply_group([position(t, 2.0 * t / 1000.0, 1.0, f"camera_{t}")])
        return f

    def test_order_does_not_change_the_result(self):
        forward = self.feed([10, 20, 30, 40, 50])
        shuffled = self.feed([10, 30, 20, 50, 40])
        a = forward.publish(60 * MS)["state"]
        b = shuffled.publish(60 * MS)["state"]
        self.assertAlmostEqual(a["x"], b["x"], places=6)
        self.assertAlmostEqual(a["y"], b["y"], places=6)
        self.assertAlmostEqual(a["vx"], b["vx"], places=6)
        self.assertGreater(shuffled.out_of_sequence, 0)

    def test_a_late_measurement_is_counted_not_silently_dropped(self):
        f = self.feed([10, 20, 30])
        self.assertTrue(f.apply_group([position(15, 0.03, 1.0, "camera_late")]))
        self.assertEqual(f.out_of_sequence, 1)

    def test_a_measurement_older_than_the_history_is_refused(self):
        f = ImmRoverFilter(history_s=0.05)
        f.apply_group([position(0, 1.0, 1.0)])
        for t in (20, 40, 60, 80, 100, 120):
            f.apply_group([position(t, 1.0, 1.0, f"camera_{t}")])
        self.assertFalse(f.apply_group([position(1, 1.0, 1.0, "ancient")]))
        self.assertEqual(f.too_old, 1)

    def test_state_time_is_restored_after_a_replay(self):
        f = self.feed([10, 20, 30])
        before = f.stamp_ns
        f.apply_group([position(15, 0.03, 1.0, "camera_late")])
        self.assertEqual(f.stamp_ns, before)


class Robustness(unittest.TestCase):
    def test_one_bad_camera_does_not_drag_the_track(self):
        clean, dirty = ImmRoverFilter(), ImmRoverFilter()
        for step in range(30):
            t = step * 10
            good = [position(t, 3.0, 3.0, "camera_1"), position(t, 3.0, 3.0, "camera_2")]
            clean.apply_group(list(good))
            dirty.apply_group(good + [position(t, 3.3, 3.0, "camera_5")])
        a = clean.publish(290 * MS)["state"]
        b = dirty.publish(290 * MS)["state"]
        self.assertLess(abs(b["x"] - a["x"]), 0.05)

    def test_gate_rejects_an_implausible_jump(self):
        f = ImmRoverFilter()
        for step in range(20):
            f.apply_group([position(step * 10, 3.0, 3.0)])
        before = f.rejected
        self.assertFalse(f.apply_group([Measurement(
            200 * MS, POSITION, (3.4, 3.0), (0.0004, 0.0, 0.0, 0.0004),
            "lidar", None, False)]))
        self.assertGreater(f.rejected, before)


class ModelMixing(unittest.TestCase):
    def test_a_turn_shifts_probability_towards_the_turn_model(self):
        # At 10 ms steps a 2 m turn deviates 2 mm from a straight line, which
        # is below the measurement noise, so the models are indistinguishable.
        # Sample at 50 ms where the deviation is real.
        turning = ImmRoverFilter()
        radius, omega = 2.0, 3.0
        for step in range(40):
            t = step * 50
            angle = omega * t / 1000.0
            turning.apply_group([position(t, 6.0 + radius * math.cos(angle),
                                          6.0 + radius * math.sin(angle))])
        straight = ImmRoverFilter()
        for step in range(40):
            t = step * 50
            straight.apply_group([position(t, 1.0 + 2.0 * t / 1000.0, 3.0)])
        turning_mu = turning.publish(1950 * MS)["model_probabilities"]
        straight_mu = straight.publish(1950 * MS)["model_probabilities"]
        self.assertGreater(turning_mu[1], straight_mu[1])

    def test_probabilities_stay_normalised(self):
        f = ImmRoverFilter()
        for step in range(50):
            f.apply_group([position(step * 10, 3.0 + 0.01 * step, 3.0)])
        self.assertAlmostEqual(sum(f.publish(500 * MS)["model_probabilities"]), 1.0, places=9)

    def test_a_single_model_filter_still_works(self):
        f = ImmRoverFilter(models=[MotionModel()])
        f.apply_group([position(0, 3.0, 3.0)])
        f.apply_group([position(10, 3.02, 3.0)])
        self.assertEqual(f.publish(10 * MS)["model_probabilities"], [1.0])


class Adapter(unittest.TestCase):
    def test_observation_becomes_a_position_and_a_yaw_measurement(self):
        from localization_contracts.contracts import Observation, SCHEMA_VERSION
        observation = Observation(
            SCHEMA_VERSION, "camera_4", 3, "camera_4:3:0", "tag_rover",
            1_000, "sim", 0, 0, 1_100, 1_200, "cal", "arena",
            (4.0, 5.0, 0.14), (0.001, 0.0002, 0.0, 0.0002, 0.002, 0.0, 0.0, 0.0, 0.0004),
            0.7, "apriltag36h11_pnp", "tag36h11", 0, pose_6d_valid=True,
            pixel_features={"yaw_rad": 0.4, "yaw_sigma_rad": 0.03},
        )
        measurements = measurement_from_observation(observation)
        self.assertEqual([m.kind for m in measurements], [POSITION, YAW_ONLY])
        self.assertEqual(measurements[0].identity, "tag36h11:0")
        self.assertTrue(measurements[0].confirms_identity)
        self.assertFalse(measurements[1].confirms_identity)
        np.testing.assert_allclose(measurements[0].as_arrays()[1],
                                   [[0.001, 0.0002], [0.0002, 0.002]])


if __name__ == "__main__":
    unittest.main()


class PerSourceOutcome(unittest.TestCase):
    """Accept/reject has to be reported per source, not per group."""

    def test_only_the_measurements_that_were_applied_come_back(self):
        f = ImmRoverFilter()
        f.apply_group([position(0, 3.0, 3.0, "camera_1")])
        good = position(20, 3.01, 3.0, "camera_1")
        # A metre off with a tight covariance: gated out on its own merits.
        rogue = Measurement(20 * MS, POSITION, (4.0, 3.0),
                            (1e-6, 0.0, 0.0, 1e-6), "camera_5", None, False)
        applied = f.apply_group([good, rogue])
        sources = {m.source for m in applied}
        self.assertIn("camera_1", sources)
        self.assertNotIn("camera_5", sources)

    def test_an_empty_result_is_falsy_and_a_full_one_is_truthy(self):
        f = ImmRoverFilter()
        self.assertFalse(f.apply_group([]))
        self.assertTrue(f.apply_group([position(0, 3.0, 3.0)]))

    def test_a_replayed_late_measurement_reports_itself(self):
        f = ImmRoverFilter()
        f.apply_group([position(0, 0.0, 1.0)])
        for t in (10, 20, 30):
            f.apply_group([position(t, 2.0 * t / 1000.0, 1.0, f"camera_{t}")])
        applied = f.apply_group([position(15, 0.03, 1.0, "camera_late")])
        self.assertEqual({m.source for m in applied}, {"camera_late"})
