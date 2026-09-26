"""The status snapshot must survive a track that has no state yet.

A localization run that cannot write status.json never reports ready, and the
coverage experiment aborts after 45 s with nothing to analyse.  That is how a
missing `or {}` cost a whole Gazebo run.
"""
import json
import unittest

from localization_contracts.rover_filter import ImmRoverFilter, Measurement, POSITION


def snapshot_track(tracks):
    """The expression run_localization.snapshot() uses for its summary."""
    if "tag_rover" not in tracks:
        return None
    return {**(tracks["tag_rover"]["state"] or {}),
            "tracking_state": tracks["tag_rover"]["tracking_state"]}


class Snapshot(unittest.TestCase):
    def test_an_uninitialised_filter_publishes_a_serialisable_row(self):
        item = ImmRoverFilter().publish(0)
        self.assertIsNone(item["state"])
        row = {**item, "object_id": "tag_rover", "source_mask": list(item["sources"])}
        json.dumps(row)                       # must not raise
        track = snapshot_track({"tag_rover": row})
        self.assertEqual(track["tracking_state"], "LOST")

    def test_a_tracked_filter_still_reports_its_pose(self):
        f = ImmRoverFilter()
        f.apply_group([Measurement(0, POSITION, (3.0, 3.0),
                                   (0.0004, 0.0, 0.0, 0.0004), "camera_1",
                                   "tag36h11:0", True)])
        item = f.publish(0)
        track = snapshot_track({"tag_rover": {**item}})
        self.assertAlmostEqual(track["x"], 3.0, places=6)
        self.assertEqual(track["tracking_state"], "TRACKING")


if __name__ == "__main__":
    unittest.main()
