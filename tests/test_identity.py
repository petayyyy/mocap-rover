"""The marker puts the tracks back after two rovers pass through each other."""
import unittest

import numpy as np

from localization_contracts.identity import TwoRoverIdentity, swap_tracks
from localization_contracts.rover_filter import ImmRoverFilter, Measurement, POSITION

MS = 1_000_000


def fix(t_ms, xy, source, identity=None, confirms=False, sigma=0.03):
    return Measurement(int(t_ms * MS), POSITION, (float(xy[0]), float(xy[1])),
                       (sigma ** 2, 0.0, 0.0, sigma ** 2), source, identity, confirms)


def paths(t):
    """Side by side eastward, 0.3 m apart, then splitting south and north at t = 1 s."""
    split = max(0.0, t - 1.0) * 1.5
    tag = np.array([3.0 + 2.0 * t, 6.0 - split])
    opponent = np.array([3.0 + 2.0 * t, 6.3 + split])
    return tag, opponent


class Crossing(unittest.TestCase):
    def run_crossing(self, confuse):
        tag = ImmRoverFilter(identity_aliases={"tag36h11:0"})
        opponent = ImmRoverFilter(identity_max_age_s=1e6)
        guard = TwoRoverIdentity()
        a, b = paths(0.0)
        tag.apply_group([fix(0, a, "marker", "tag36h11:0", True, 0.01)])
        opponent.apply_group([fix(0, b, "operator", "operator:opponent", True)])
        swaps = []
        for step in range(1, 181):
            t_ms = step * 12.5
            a, b = paths(t_ms / 1000)
            # Continuation sources cannot tell the rovers apart.  After the
            # crossing a confused association feeds each track the other rover.
            if 500 < t_ms <= 1000:
                # Touching: one merged blob, both tracks fed its centre.
                merged = (a + b) / 2
                tag_fix, opponent_fix = merged, merged
            elif confuse and t_ms > 1000:
                # Splitting, the association goes the wrong way.
                tag_fix, opponent_fix = b, a
            else:
                tag_fix, opponent_fix = a, b
            tag.apply_group([fix(t_ms, tag_fix, "blob")])
            opponent.apply_group([fix(t_ms, opponent_fix, "blob")])
            tag.publish(int(t_ms * MS))
            opponent.publish(int(t_ms * MS))
            guard.observe_tracks(t_ms * MS, tag.x[:2], opponent.x[:2])
            if t_ms > 1800 and step % 8 == 0:       # the marker is read again
                if guard.marker_says_swap(t_ms * MS, a, tag.x[:2], opponent.x[:2]):
                    guard.swap(t_ms * MS, tag, opponent, a)
                    swaps.append(t_ms)
                tag.apply_group([fix(t_ms, a, "marker", "tag36h11:0", True, 0.01)])
        return tag, opponent, guard, swaps

    def test_a_marker_on_the_opponent_track_swaps_the_tracks_back(self):
        tag, opponent, guard, swaps = self.run_crossing(confuse=True)
        self.assertEqual(guard.encounters, 1)
        encounter = next(e for e in guard.events if e["event"] == "encounter")
        self.assertLess(encounter["closest_m"], 0.4)
        self.assertEqual(guard.swaps, 1)
        self.assertEqual([e["event"] for e in guard.events], ["encounter", "swap"])
        a, b = paths(180 * 12.5 / 1000)
        # Each estimate ends on the rover it belongs to; roles stayed put.
        self.assertLess(np.linalg.norm(tag.x[:2] - a), 0.1)
        self.assertEqual(tag.identity, "tag36h11:0")
        self.assertEqual(opponent.identity, "operator:opponent")

    def test_a_clean_crossing_swaps_nothing(self):
        tag, opponent, guard, swaps = self.run_crossing(confuse=False)
        self.assertEqual((guard.encounters, guard.swaps, swaps), (1, 0, []))

    def test_no_encounter_means_no_swap(self):
        guard = TwoRoverIdentity()
        guard.observe_tracks(0, (1.0, 1.0), (5.0, 5.0))
        self.assertFalse(guard.marker_says_swap(1, (5.0, 5.0), (1.0, 1.0), (5.0, 5.0)))
        self.assertEqual(guard.encounters, 0)

    def test_swap_moves_state_and_keeps_roles(self):
        first = ImmRoverFilter(identity_aliases={"tag36h11:0"})
        second = ImmRoverFilter(identity_max_age_s=1e6)
        first.apply_group([fix(0, (1.0, 1.0), "marker", "tag36h11:0", True)])
        second.apply_group([fix(0, (4.0, 4.0), "operator", "operator:opponent", True)])
        swap_tracks(first, second)
        np.testing.assert_allclose(first.x[:2], [4.0, 4.0])
        np.testing.assert_allclose(second.x[:2], [1.0, 1.0])
        self.assertEqual(first.identity_aliases, {"tag36h11:0"})
        self.assertEqual(first.identity, "tag36h11:0")


if __name__ == "__main__":
    unittest.main()
