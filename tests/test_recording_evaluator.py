import json
import tempfile
import unittest
from pathlib import Path
from scripts.evaluate_recording import evaluate
from scripts.record_gazebo_truth import pose_stamp_ns


class RecordingEvaluation(unittest.TestCase):
    def test_nested_pose_timestamp_not_empty_envelope(self):
        from gz.msgs10.pose_v_pb2 import Pose_V

        msg = Pose_V()
        pose = msg.pose.add(name="tag_rover")
        pose.header.stamp.sec = 17
        pose.header.stamp.nsec = 23
        self.assertEqual(msg.header.stamp.sec, 0)
        self.assertEqual(pose_stamp_ns(pose), 17_000_000_023)

    def test_missing_pose_timestamp_rejected(self):
        from gz.msgs10.pose_pb2 import Pose

        with self.assertRaises(ValueError):
            pose_stamp_ns(Pose())

    def test_evaluation_is_serializable_and_matches_by_capture_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            truth = []
            outputs = []
            obs = []
            for name in ("tag_rover", "opponent"):
                for i in range(61):
                    stamp = 1_000_000_000 + i * 33_333_333
                    truth.append(
                        {
                            "object_id": name,
                            "stamp_ns": stamp,
                            "x": 1.0,
                            "y": 2.0,
                            "yaw": 0.0,
                        }
                    )
                    if i % 2 == 0 and name == "opponent" or name == "tag_rover":
                        outputs.append(
                            {
                                "object_id": name,
                                "wall_ns": stamp + 100,
                                "capture_ns": stamp,
                                "valid": True,
                                "measurement_age_ms": 0.0,
                                "state": {
                                    "stamp_ns": stamp,
                                    "x": 1.0,
                                    "y": 2.0,
                                    "yaw": 0.0,
                                    "vx": 0.0,
                                    "vy": 0.0,
                                },
                            }
                        )
                        obs.append(
                            {
                                "accepted": True,
                                "observation": {
                                    "object_id": name,
                                    "capture_time_ns": stamp,
                                },
                                "wall_ns": stamp,
                            }
                        )

            def save(name, rows):
                (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))

            save("truth.jsonl", truth)
            save("odometry.jsonl", outputs)
            save("observations.jsonl", obs)
            report = evaluate(root, root / "truth.jsonl")
            json.dumps(report, allow_nan=False)
            self.assertEqual(report["tracks"]["tag_rover"]["p95_xy_m"], 0.0)
            self.assertTrue(report["tracks"]["opponent"]["pass"])
            self.assertFalse(report["sim_accepted"])
