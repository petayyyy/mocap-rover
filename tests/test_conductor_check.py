import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.conductor_check import check_replay, check_truth, verdict


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


class ConductorCheck(unittest.TestCase):
    def _dataset(self, root, seconds=2.0, hz=200):
        truth = []
        n = int(seconds * hz)
        for i in range(n):
            t = 1_000_000_000 + i * (1_000_000_000 // hz)
            s = i / hz
            truth.append({"object_id": "tag_rover", "stamp_ns": t, "x": 2.0 + 4.0 * s, "y": 6.0, "z": 0.14, "yaw": 0.0})
            truth.append({"object_id": "opponent", "stamp_ns": t, "x": 9.0, "y": 6.0 - 1.0 * s,
                          "z": 0.30 if i < n // 10 else 0.14, "yaw": math.pi / 2})
        write_jsonl(root / "truth.jsonl", truth)
        (root / "meta.json").write_text(json.dumps({
            "truth_rows": len(truth), "image_size_errors": 0, "lidar": {"scans": 20},
            "sim_start_ns": truth[0]["stamp_ns"], "sim_end_ns": truth[-1]["stamp_ns"],
            "cameras": {"camera_1": {"frames": 3}}}))
        write_jsonl(root / "camera_1.jsonl", [{"index": i, "stamp_ns": 1_000_000_000 + i * 12_500_000, "wall_ns": 0} for i in range(3)])
        (root / "runtime_cameras.json").write_text(json.dumps({"cameras": [
            {"name": "camera_1", "position_world": [3, 6, 2.9]}, {"name": "camera_2", "position_world": [9, 6, 2.9]}]}))
        return truth

    def test_truth_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._dataset(root)
            report = check_truth(root)
            tag = report["objects"]["tag_rover"]
            self.assertAlmostEqual(tag["speed_p50"], 4.0, delta=0.05)
            self.assertEqual(tag["xy_jumps_over_limit"], 0)
            self.assertAlmostEqual(report["objects"]["opponent"]["z_above_limit_fraction"], 0.1, delta=0.01)
            self.assertEqual(report["cameras"]["camera_1"]["index_rows"], 3)
            self.assertTrue(report["cameras"]["camera_1"]["monotonic"])

    def test_replay_metrics_and_swap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            truth = self._dataset(root)
            runtime = root / "run"
            runtime.mkdir()
            rows = []
            tag = [r for r in truth if r["object_id"] == "tag_rover"]
            opp = [r for r in truth if r["object_id"] == "opponent"]
            for i, (a, b) in enumerate(zip(tag, opp)):
                t = a["stamp_ns"]
                # 30 outputs (150 ms) sit on the opponent: one swap episode.
                swapped = 100 <= i < 130
                x, y = (b["x"], b["y"]) if swapped else (a["x"] + 0.02, a["y"])
                rows.append({"object_id": "tag_rover", "valid": i % 50 != 0, "tracking_state": "TRACKING",
                             "measurement_age_ms": 5.0, "wall_ns": i,
                             "state": {"stamp_ns": t, "x": x, "y": y, "yaw": 0.01, "vx": 4.0, "vy": 0.0}})
                rows.append({"object_id": "opponent", "valid": True, "tracking_state": "COASTING",
                             "measurement_age_ms": 20.0, "wall_ns": i,
                             "state": {"stamp_ns": t, "x": b["x"] + 0.05, "y": b["y"], "yaw": 0.0, "vx": 0.0, "vy": -1.0}})
            write_jsonl(runtime / "odometry.jsonl", rows)
            write_jsonl(runtime / "camera_frames.jsonl", [
                {"camera_id": "camera_1", "mode": "roi", "latency_ms": 1.0, "detections": 1, "pnp_valid": 1,
                 "fusion_accepted": 1, "pnp_rejections": {}},
                {"camera_id": "camera_1", "mode": "acquire", "latency_ms": 9.0, "detections": 1, "pnp_valid": 0,
                 "fusion_accepted": 0, "pnp_rejections": {"base_height": 1}}])
            write_jsonl(runtime / "observations.jsonl", [
                {"accepted": True, "selection_reason": "fusion_accepted",
                 "observation": {"object_id": "tag_rover", "camera_id": "camera_1", "method": "pnp"}}])
            report = check_replay(runtime, root / "truth.jsonl")
            t = report["objects"]["tag_rover"]
            self.assertEqual(t["swap_episodes_over_limit"], 1)
            self.assertGreater(t["swap_longest_ms"], 100)
            self.assertAlmostEqual(t["valid_fraction_outputs"], 0.98, delta=0.01)
            self.assertLess(t["xy_error_m"]["p50"], 0.03)
            self.assertEqual(report["camera_frames"]["cameras"]["camera_1"]["pnp_rejections"], {"base_height": 1})
            self.assertAlmostEqual(report["camera_frames"]["cameras"]["camera_1"]["mode_share"]["roi"], 0.5)
            o = report["objects"]["opponent"]
            self.assertEqual(o["swap_episodes_over_limit"], 0)
            self.assertAlmostEqual(o["xy_error_m"]["p95"], 0.05, delta=0.005)
            self.assertIn("0-2", o["xy_error_by_range_to_nearest_camera_m"])
            v = verdict(report)
            self.assertFalse(v["tag_rover"]["pass"])
            self.assertTrue(v["opponent"]["pass"])


if __name__ == "__main__":
    unittest.main()
