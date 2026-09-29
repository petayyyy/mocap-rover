"""The offline replay drives the live tracker from a recorded dataset.

A two-camera, ten-frame dataset is written in the recorder's own format --
FFV1 video, per-frame jsonl index, clock and truth rows -- with the marker
rendered where each camera sees it, then replayed end to end.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import arena_render  # noqa: E402

FRAMES = 10
PERIOD_NS = 12_000_000
START_NS = 44_294_000_000          # datasets do not start at zero
CAMERAS = {"camera_1": (3.0, 2.0, 2.9), "camera_2": (4.0, 2.0, 2.9)}


OPPONENT_XY = (3.6, 3.0)
OPPONENT_YAW = 0.3


def rover_xy(i):
    return 3.3 + 0.01 * i, 2.2


def paint_opponent(rgb, position):
    """The opponent as a plain cuboid, 0.9 x 0.52 x 0.483, through the pinhole."""
    from localization_contracts.camera_model import CameraModel
    model = CameraModel(arena_render.K, [0.0] * 5, arena_render.IMAGE_SIZE)
    c, s = np.cos(OPPONENT_YAW), np.sin(OPPONENT_YAW)
    pts = np.array([[a * 0.45, b * 0.26, z] for a in (-1, 1) for b in (-1, 1) for z in (0, 0.483)])
    pts[:, :2] = pts[:, :2] @ np.array([[c, s], [-s, c]]) + OPPONENT_XY
    uv = model.project((pts - np.asarray(position)) @ arena_render.camera_rotation())
    import cv2
    out = rgb.copy()
    cv2.fillConvexPoly(out, cv2.convexHull(np.round(uv).astype(np.int32)), (200, 80, 60))
    return out


def write_video(path, frames, width, height):
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", "80", "-i", "-", "-c:v", "ffv1", str(path)],
        input=b"".join(f.tobytes() for f in frames), check=True)


def write_background(root):
    """The empty arena: same cameras, same size, nothing on the floor."""
    width, height = arena_render.IMAGE_SIZE
    for name in CAMERAS:
        empty = np.full((height, width, 3), 150, np.uint8)
        write_video(root / f"{name}.mkv", [empty] * 4, width, height)
        with (root / f"{name}.jsonl").open("w") as handle:
            for i in range(4):
                handle.write(json.dumps({"camera_id": name, "stamp_ns": i, "wall_ns": i,
                                         "index": i}) + "\n")


def write_dataset(root):
    width, height = arena_render.IMAGE_SIZE
    R = arena_render.camera_rotation().tolist()
    cameras = []
    for name, position in CAMERAS.items():
        cameras.append({
            "name": name, "position_world": list(position), "R_world_optical": R,
            "K": arena_render.K.reshape(-1).tolist(), "D": [0.0] * 5,
            "distortion_model": "pinhole", "image_size": [width, height]})
        frames = []
        for i in range(FRAMES):
            gray = arena_render.render(position, R, rover_xy(i))
            frames.append(paint_opponent(np.repeat(gray[:, :, None], 3, axis=2), position))
        write_video(root / f"{name}.mkv", frames, width, height)
        with (root / f"{name}.jsonl").open("w") as handle:
            for i in range(FRAMES):
                handle.write(json.dumps({"camera_id": name, "stamp_ns": START_NS + i * PERIOD_NS,
                                         "wall_ns": i, "index": i}) + "\n")
    config = {
        "role": "image_calibrated", "calibration_version": "synthetic-replay",
        "cameras": cameras,
        "tags": [{"family": "tag36h11", "id": 0, "size": arena_render.TAG_SIZE,
                  "placement": "top",
                  "T_base_tag_translation": [0.0, 0.0, arena_render.TAG_DZ],
                  "R_base_tag": np.eye(3).tolist()}],
    }
    (root / "runtime_cameras.json").write_text(json.dumps(config))
    with (root / "clock.jsonl").open("w") as handle:
        for i in range(2):
            handle.write(json.dumps({"sim_ns": START_NS + i * 100_000_000, "wall_ns": i}) + "\n")
    with (root / "truth.jsonl").open("w") as handle:
        for i in range(FRAMES * 3 + 10):
            stamp = START_NS + (i - 10) * 5_000_000
            x, y = rover_xy((i - 10) * 5 / 12)
            handle.write(json.dumps({"object_id": "tag_rover", "stamp_ns": stamp, "x": x,
                                     "y": y, "z": 0.14, "yaw": 0.0}) + "\n")
            handle.write(json.dumps({"object_id": "opponent", "stamp_ns": stamp,
                                     "x": OPPONENT_XY[0], "y": OPPONENT_XY[1], "z": 0.14,
                                     "yaw": OPPONENT_YAW}) + "\n")


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is needed to write FFV1")
class ReplayDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.dataset = cls.root / "dataset"
        cls.dataset.mkdir()
        write_dataset(cls.dataset)
        cls.background = cls.root / "background"
        cls.background.mkdir()
        write_background(cls.background)
        import replay_dataset
        cls.replay = replay_dataset
        cls.out = cls.root / "replay"
        with open(cls.root / "stdout", "w") as sink:
            stdout, sys.stdout = sys.stdout, sink
            stderr, sys.stderr = sys.stderr, sink
            try:
                replay_dataset.main([str(cls.dataset), "--output", str(cls.out)])
            finally:
                sys.stdout, sys.stderr = stdout, stderr

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, name):
        return [json.loads(l) for l in (self.out / name).read_text().splitlines()]

    def test_odometry_is_published_on_the_simulated_clock(self):
        rows = self.rows("odometry.jsonl")
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("measurement_age_ms", row)
            self.assertIn("tracking_state", row)
        stamps = [row["stamp_ns"] for row in rows]
        self.assertEqual(stamps[0], START_NS)
        self.assertEqual(set(np.diff(stamps)), {5_000_000})

    def test_the_evaluator_reads_the_output(self):
        import evaluate_recording
        for row in self.rows("odometry.jsonl"):
            self.assertEqual(row["state"]["stamp_ns"], row["stamp_ns"])
            if row["valid"]:
                self.assertIn("x", row["state"])
        report = evaluate_recording.evaluate(
            self.out, self.dataset / "truth.jsonl", ("tag_rover",))
        track = report["tracks"]["tag_rover"]
        self.assertGreater(track["valid_outputs"], 0)
        self.assertLess(track["p95_xy_m"], 0.05)

    def test_the_marker_creates_a_track_where_the_rover_is(self):
        valid = [r for r in self.rows("odometry.jsonl") if r["valid"]]
        self.assertTrue(valid)
        last = valid[-1]["state"]
        x, y = rover_xy(FRAMES - 1)
        self.assertLess(np.hypot(last["x"] - x, last["y"] - y), 0.05)

    def test_a_measurement_is_never_used_before_it_arrives(self):
        transport = 15_000_000
        for row in self.rows("observations.jsonl"):
            obs = row["observation"]
            self.assertEqual(obs["receive_time_ns"] - obs["capture_time_ns"], transport)
            self.assertGreaterEqual(row["wall_ns"], obs["receive_time_ns"])
        first_valid = next(r for r in self.rows("odometry.jsonl") if r["valid"])
        self.assertGreaterEqual(first_valid["stamp_ns"], START_NS + transport)

    def test_every_processed_frame_is_timed(self):
        frames = self.rows("camera_frames.jsonl")
        self.assertEqual(len(frames), 2 * FRAMES)
        self.assertTrue(all(row["latency_ms"] > 0 for row in frames))
        timing = json.loads((self.out / "timing.json").read_text())
        self.assertEqual(set(timing["cameras"]), set(CAMERAS))
        for key in ("mean", "p50", "p95"):
            self.assertIsNotNone(timing["cameras"]["camera_1"]["latency_ms"][key])
        self.assertGreater(timing["fps_total"], 0)
        self.assertGreater(timing["realtime_ratio"], 0)

    def test_the_operator_box_is_the_only_use_of_truth(self):
        params = json.loads((self.out / "runtime_parameters.json").read_text())
        box = params["opponent_operator_box"]
        self.assertIsNotNone(box)
        self.assertEqual(box["stamp_ns"], START_NS)
        x0, y0, x1, y1 = box["boxes_xyxy_px"]["camera_1"]
        self.assertTrue(0 < x0 < x1 < arena_render.IMAGE_SIZE[0])
        self.assertTrue(0 < y0 < y1 < arena_render.IMAGE_SIZE[1])
        self.assertFalse(params["opponent_enabled"])

    def test_the_replay_is_deterministic(self):
        again = self.root / "again"
        with open(self.root / "stdout2", "w") as sink:
            stdout, sys.stdout = sys.stdout, sink
            stderr, sys.stderr = sys.stderr, sink
            try:
                self.replay.main([str(self.dataset), "--output", str(again)])
            finally:
                sys.stdout, sys.stderr = stdout, stderr

        def strip(path):
            out = []
            for line in path.read_text().splitlines():
                row = json.loads(line)
                row.pop("replay_wall_ns", None)
                out.append(row)
            return out
        self.assertEqual(strip(self.out / "odometry.jsonl"), strip(again / "odometry.jsonl"))

    def run_replay(self, name, *extra):
        out = self.root / name
        with open(self.root / f"{name}.log", "w") as sink:
            stdout, sys.stdout = sys.stdout, sink
            stderr, sys.stderr = sys.stderr, sink
            try:
                self.replay.main([str(self.dataset), "--output", str(out),
                                  "--camera-background", str(self.background), *extra])
            finally:
                sys.stdout, sys.stderr = stdout, stderr
        return out

    def test_the_opponent_is_tracked_from_the_operator_box(self):
        for gain in ("1.0", "0.6"):
            with self.subTest(gain=gain):
                out = self.run_replay(f"opponent_{gain}", "--gain", gain)
                rows = [json.loads(l) for l in (out / "odometry.jsonl").read_text().splitlines()]
                valid = [r for r in rows if r["object_id"] == "opponent" and r["valid"]]
                self.assertTrue(valid)
                last = valid[-1]["state"]
                self.assertLess(np.hypot(last["x"] - OPPONENT_XY[0], last["y"] - OPPONENT_XY[1]), 0.08)
                methods = {json.loads(l)["observation"]["method"]
                           for l in (out / "observations.jsonl").read_text().splitlines()
                           if '"opponent"' in l}
                self.assertIn("operator_box", methods)
                self.assertIn("silhouette_extent", methods)
                status = json.loads((out / "status.json").read_text())
                self.assertEqual(status["identity"]["swaps"], 0)
                self.assertTrue(status["opponent_enabled"])

    def test_an_existing_output_is_never_overwritten(self):
        with self.assertRaises(SystemExit):
            self.replay.main([str(self.dataset), "--output", str(self.out)])


if __name__ == "__main__":
    unittest.main()
