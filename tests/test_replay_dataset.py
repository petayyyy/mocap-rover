"""The offline replay drives the live tracker from a recorded dataset.

A two-camera, ten-frame dataset is written in the recorder's own format --
FFV1 video, per-frame jsonl index, clock and truth rows -- with the marker
rendered where each camera sees it, then replayed end to end.
"""
import collections
import importlib.util
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

    def test_threads_processes_and_one_thread_give_the_same_output(self):
        def strip(path):
            rows = []
            for line in path.read_text().splitlines():
                row = json.loads(line)
                row.pop("replay_wall_ns", None)
                rows.append(row)
            return rows
        outputs = {mode: self.run_replay(f"parallel_{mode}", "--parallel", mode)
                   for mode in ("threads", "processes", "inline")}
        base = strip(outputs["threads"] / "odometry.jsonl")
        self.assertTrue(base)
        for mode in ("processes", "inline"):
            with self.subTest(mode=mode):
                self.assertEqual(strip(outputs[mode] / "odometry.jsonl"), base)
        timing = json.loads((outputs["processes"] / "timing.json").read_text())
        self.assertEqual(timing["parallel"], "processes")
        for stage in ("decode_ms", "detect_ms", "pnp_ms", "gain_ms", "background_update_ms",
                      "main_filter_publish_ms"):
            self.assertIn(stage, timing["stages_ms"])

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

    def test_the_operator_box_starts_tag_rover_once(self):
        params = json.loads((self.out / "runtime_parameters.json").read_text())
        box = params["tag_rover_operator_box"]
        self.assertIsNotNone(box)
        self.assertEqual(box["cuboid_m"], list(self.replay.TAG_BODY_M))
        starts = [json.loads(l)["observation"] for l in (self.out / "observations.jsonl").read_text().splitlines()
                  if '"tag_rover"' in l and '"operator_box"' in l]
        self.assertTrue(starts)
        self.assertEqual({o["capture_time_ns"] for o in starts}, {box["stamp_ns"]})
        for obs in starts:
            x, y = rover_xy(0)
            self.assertLess(np.hypot(obs["position_m"][0] - x, obs["position_m"][1] - y), 0.15)
        out = self.run_replay("no_tag_box", "--no-tag-operator-box")
        params = json.loads((out / "runtime_parameters.json").read_text())
        self.assertIsNone(params["tag_rover_operator_box"])
        self.assertNotIn("tag_operator", (out / "observations.jsonl").read_text())

    def test_link_emulation_delays_each_camera_by_the_link_model(self):
        out = self.run_replay("link_cm4", "--link-emulation", "cm4", "--roi-max-px", "320",
                              "--opponent-roi-max-px", "480", "--acquire-full-frame-hz", "10")
        observations = [json.loads(l)["observation"]
                        for l in (out / "observations.jsonl").read_text().splitlines()]
        delays = [o["receive_time_ns"] - o["capture_time_ns"] for o in observations
                  if o["method"] != "operator_box"]
        self.assertTrue(delays)
        self.assertGreaterEqual(min(delays), 22_600_000)
        timing = json.loads((out / "timing.json").read_text())
        link = timing["link"]
        self.assertEqual(link["mode"], "cm4")
        self.assertTrue(link["series_mbps"])
        self.assertGreaterEqual(min(link["series_mbps"]), 22.5)
        self.assertGreater(link["raw_windows"] + link["jpeg_windows"] + link["full_frames"], 0)
        params = json.loads((out / "runtime_parameters.json").read_text())
        self.assertEqual(params["link_emulation"]["roi_jpeg_max_px"], 640)
        valid = [json.loads(l) for l in (out / "odometry.jsonl").read_text().splitlines()]
        self.assertTrue(any(r["valid"] and r["object_id"] == "tag_rover" for r in valid))
        self.assertIsNone(json.loads((self.out / "timing.json").read_text())["link"])

    @unittest.skipUnless(importlib.util.find_spec("av"), "PyAV not installed")
    def test_the_small_stream_through_h264_is_counted_on_the_port(self):
        out = self.run_replay("small_h264", "--link-emulation", "cm4", "--roi-max-px", "320",
                              "--opponent-stream", "320x240@30", "--opponent-stream-codec", "h264",
                              "--opponent-stream-bitrate", "2")
        timing = json.loads((out / "timing.json").read_text())
        small = timing["small_stream"]
        self.assertEqual(small["codec"], "h264")
        self.assertGreater(small["all_cameras"]["frames"], 0)
        self.assertLess(small["all_cameras"]["frame_kb_p50"], 320 * 240 / 1000)
        self.assertGreater(timing["link"]["windows_h264"], 0)

    @unittest.skipUnless(importlib.util.find_spec("av"), "PyAV not installed")
    def test_the_full_frame_stream_keeps_its_grid_and_latency(self):
        out = self.run_replay("full_h264", "--link-emulation", "cm4", "--roi-max-px", "480",
                              "--roi-jpeg-max-px", "480", "--opponent-stream", "320x240@30",
                              "--full-stream-codec", "h264", "--full-stream-fps", "40",
                              "--full-stream-bitrate", "4", "--full-stream-latency-ms", "70",
                              "--full-stream-latency-jitter-ms", "10")
        timing = json.loads((out / "timing.json").read_text())
        self.assertIsNone(timing["link"])
        full = timing["full_stream"]
        # 12 ms frames on a 25 ms grid: every other or third frame exists.
        stamps = collections.defaultdict(list)
        for row in (json.loads(l) for l in (out / "camera_frames.jsonl").read_text().splitlines()):
            stamps[row["camera_id"]].append(row["capture_ns"])
        for values in stamps.values():
            self.assertLess(len(values), FRAMES)
            self.assertGreaterEqual(min(np.diff(sorted(values))), 24_000_000)
        self.assertEqual(full["all_cameras"]["frames"], sum(map(len, stamps.values())))
        self.assertGreaterEqual(full["arrival_ms"]["p50"], 70.0)
        observations = [json.loads(l)["observation"]
                        for l in (out / "observations.jsonl").read_text().splitlines()]
        delays = [o["receive_time_ns"] - o["capture_time_ns"] for o in observations
                  if o["method"] != "operator_box"]
        if delays:
            self.assertGreaterEqual(min(delays), 70_000_000)

    def test_an_existing_output_is_never_overwritten(self):
        with self.assertRaises(SystemExit):
            self.replay.main([str(self.dataset), "--output", str(self.out)])


if __name__ == "__main__":
    unittest.main()
