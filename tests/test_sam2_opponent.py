"""SAM2 as a second opinion on the opponent.

The policy (prompts, mask -> reading, identity guards, bounded memory) is
tested with a stand-in engine, so it runs without torch.  The engine itself
runs only where torch sees CUDA and the weights are in models/sam2/.
"""
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from localization_contracts.camera_model import CameraModel  # noqa: E402
from localization_contracts.camera_worker import scaled_camera  # noqa: E402
from localization_contracts.opponent_camera import body_hull_px  # noqa: E402
from localization_contracts import sam2_opponent as so  # noqa: E402

SIZE = (0.9, 0.52, 0.483)
NADIR = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=float)
CAMERA_XYZ = np.array([6.0, 6.0, 2.9])
FULL = {"name": "c", "K": [734.5, 0, 819.5, 0, 734.5, 615.5, 0, 0, 1], "D": [0.0] * 4,
        "distortion_model": "fisheye", "image_size": [1640, 1232],
        "R_world_optical": NADIR.tolist(), "position_world": CAMERA_XYZ.tolist()}
SMALL, _ = scaled_camera(FULL, 640, 480)
MODEL = CameraModel.from_config(SMALL)
WEIGHTS = ROOT / "models/sam2/sam2.1_hiera_tiny.pt"


def body_mask(xy, yaw, size=SIZE, shape=(480, 640)):
    hull = body_hull_px(MODEL, NADIR, CAMERA_XYZ, xy, yaw, size)
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(hull).astype(np.int32)), 1)
    return mask.astype(bool)


class FakeTracker(so.Sam2Tracker):
    pass


class FakeEngine:
    """Returns the queued (mask, score) per call; records the prompts."""

    window = 4
    config = "fake"
    image_size = 0

    def device_name(self):
        return "fake"

    def reset_peak_memory(self):
        pass

    def peak_memory_mb(self):
        return None

    def __init__(self):
        self.queue = []
        self.prompts = []

    def new_tracker(self, camera_id):
        return FakeTracker(camera_id, self.window)

    def step(self, items):
        out = []
        for tracker, image, box in items:
            if box is not None:
                tracker.reset()
            tracker.frame_idx += 1
            key = "cond_frame_outputs" if box is not None else "non_cond_frame_outputs"
            tracker.output_dict[key][tracker.frame_idx] = {}
            tracker.prompted = True
            tracker.prune()
            self.prompts.append(box)
            mask, score = self.queue.pop(0)
            out.append(so.StepOutput(mask, score, box is not None, 1.0))
        return out


def policy(engine=None, **kwargs):
    engine = engine or FakeEngine()
    return so.Sam2Opponent(engine, {"c": (MODEL, NADIR, CAMERA_XYZ)}, size_m=SIZE,
                           marker=(0.0, 0.0, 0.3654, 0.4), **kwargs), engine


IMAGE = np.zeros((480, 640, 3), np.uint8)


class Sam2Policy(unittest.TestCase):
    def test_nothing_runs_before_a_prompt(self):
        sam2, engine = policy()
        readings, refusals, steps, ms = sam2.process([("c", IMAGE)], 0)
        self.assertEqual((readings, refusals, steps, ms), ([], [], {}, None))
        self.assertEqual(engine.prompts, [])

    def test_the_operator_box_prompts_and_the_mask_becomes_a_reading(self):
        sam2, engine = policy()
        xy = np.array([6.8, 5.6])
        engine.queue.append((body_mask(xy, 0.3), 0.95))
        readings, refusals, _, _ = sam2.process([("c", IMAGE)], 0,
                                                operator_boxes={"c": (10, 20, 30, 40)})
        self.assertEqual(refusals, [])
        self.assertEqual(engine.prompts, [(10.0, 20.0, 30.0, 40.0)])
        self.assertEqual(len(readings), 1)
        reading = readings[0]
        self.assertLess(math.hypot(reading.x - xy[0], reading.y - xy[1]), 0.06)
        self.assertTrue(reading.prompted)
        # Next frame: no prompt, the memory carries the object.
        engine.queue.append((body_mask(xy + 0.05, 0.3), 0.95))
        readings, _, _, _ = sam2.process([("c", IMAGE)], 50_000_000,
                                         opp_prediction=(*xy, 0.3))
        self.assertEqual(engine.prompts[-1], None)
        self.assertEqual(len(readings), 1)

    def test_a_weak_or_empty_mask_is_refused_and_reprompted_from_the_prediction(self):
        sam2, engine = policy()
        xy = np.array([5.5, 6.5])
        engine.queue.append((body_mask(xy, 0.0), 0.9))
        sam2.process([("c", IMAGE)], 0, operator_boxes={"c": (0, 0, 9, 9)})
        engine.queue.append((body_mask(xy, 0.0), 0.2))
        _, refusals, _, _ = sam2.process([("c", IMAGE)], 1, opp_prediction=(*xy, 0.0))
        self.assertEqual(refusals[0].reason, so.LOW_SCORE)
        engine.queue.append((np.zeros((480, 640), bool), 0.9))
        # Not allowed to prompt (track not confirmed by another source): memory kept.
        _, refusals, _, _ = sam2.process([("c", IMAGE)], 2, opp_prediction=(*xy, 0.0))
        self.assertIsNone(engine.prompts[-1])
        self.assertEqual(refusals[0].reason, so.EMPTY_MASK)
        engine.queue.append((body_mask(xy, 0.0), 0.9))
        readings, _, _, _ = sam2.process([("c", IMAGE)], 3, opp_prediction=(*xy, 0.0),
                                         may_prompt=True)
        box = engine.prompts[-1]
        self.assertIsNotNone(box)
        expected = body_hull_px(MODEL, NADIR, CAMERA_XYZ, xy, 0.0, SIZE)
        self.assertLess(box[0], expected[:, 0].min())
        self.assertGreater(box[2], expected[:, 0].max())
        self.assertEqual(len(readings), 1)
        self.assertEqual(sam2.prompts["prediction"], 1)

    def test_a_broken_mask_is_refused(self):
        sam2, engine = policy()
        a = body_mask((5.0, 6.0), 0.0)
        b = body_mask((7.2, 6.0), 0.0)
        engine.queue.append((a | b, 0.9))
        _, refusals, _, _ = sam2.process([("c", IMAGE)], 0, operator_boxes={"c": (0, 0, 9, 9)})
        self.assertEqual(refusals[0].reason, so.FRAGMENTED)

    def test_identity_guards(self):
        opp, tag = np.array([6.6, 6.4]), np.array([5.4, 6.4])
        cases = [
            # mask at the tag_rover while the opponent is predicted elsewhere
            ("gate", body_mask(tag, 0.0), (*opp, 0.0), (*tag, 0.0), so.OUT_OF_GATE, 0.8),
            # mask between them but nearer the tag_rover prediction
            ("nearer", body_mask(tag + [0.35, 0], 0.0), (*opp, 0.0), (*tag, 0.0), so.NEARER_TAG, 2.0),
            # mask covering the tag_rover marker (tag_rover predicted half under it)
            ("marker", body_mask(opp, 0.0), (*opp, 0.0), (*(opp + [0.45, 0]), 0.0),
             so.ON_TAG_MARKER, 2.0),
        ]
        for name, mask, opp_pred, tag_pred, reason, gate in cases:
            with self.subTest(name):
                sam2, engine = policy()
                engine.queue.append((body_mask(opp, 0.0), 0.9))
                sam2.process([("c", IMAGE)], 0, operator_boxes={"c": (0, 0, 9, 9)})
                engine.queue.append((mask, 0.9))
                readings, refusals, _, _ = sam2.process(
                    [("c", IMAGE)], 1, opp_prediction=opp_pred, opp_gate_m=gate,
                    tag_prediction=tag_pred)
                self.assertEqual(readings, [])
                self.assertEqual(refusals[0].reason, reason)

    def test_the_marker_rule_alone(self):
        sam2, _ = policy()
        tag = (5.4, 6.4, 0.0)
        rect = sam2.marker_rect("c", tag)
        mask = np.zeros((480, 640), bool)
        x0, y0, x1, y1 = (int(round(v)) for v in rect)
        mask[y0:y1, x0:x0 + (x1 - x0) // 2] = True
        self.assertAlmostEqual(sam2.rect_covered(mask, rect), 0.5, delta=0.1)
        self.assertEqual(sam2.rect_covered(np.zeros_like(mask), rect), 0.0)

    def test_lost_track_needs_distance_from_tag_rover(self):
        sam2, engine = policy()
        tag = np.array([5.4, 6.4])
        engine.queue.append((body_mask(tag + [0.7, 0], 0.0), 0.9))
        _, refusals, _, _ = sam2.process([("c", IMAGE)], 0, operator_boxes={"c": (0, 0, 9, 9)},
                                         tag_prediction=(*tag, 0.0))
        self.assertEqual(refusals[0].reason, so.NEAR_TAG_LOST)

    def test_memory_stays_bounded_and_a_stale_camera_starts_over(self):
        sam2, engine = policy(stale_ns=100)
        xy = (6.5, 6.0)
        engine.queue.append((body_mask(xy, 0.0), 0.9))
        sam2.process([("c", IMAGE)], 0, operator_boxes={"c": (0, 0, 9, 9)})
        for k in range(1, 30):
            engine.queue.append((body_mask(xy, 0.0), 0.9))
            sam2.process([("c", IMAGE)], k, opp_prediction=(*xy, 0.0))
        self.assertLessEqual(sam2.memory_frames()["c"], engine.window + 1)
        self.assertIn(0, sam2.cameras["c"]["tracker"].output_dict["cond_frame_outputs"])
        readings, _, steps, _ = sam2.process([("c", IMAGE)], 10_000)
        self.assertEqual(steps, {})          # memory dropped, nothing to prompt with

    def test_config_follows_the_checkpoint_name(self):
        self.assertEqual(so.config_for_checkpoint("x/sam2.1_hiera_small.pt"),
                         "configs/sam2.1/sam2.1_hiera_s.yaml")
        self.assertEqual(so.config_for_checkpoint("other.pt"), so.DEFAULT_CONFIG)


def cuda_and_weights():
    if not WEIGHTS.exists() or not so.sam2_available():
        return False
    import torch
    return torch.cuda.is_available()


@unittest.skipUnless(cuda_and_weights(), "torch with CUDA, sam2 and models/sam2 weights needed")
class Sam2EngineOnGpu(unittest.TestCase):
    def test_a_box_prompt_follows_a_moving_body(self):
        engine = so.Sam2Engine(WEIGHTS, image_size=512)
        tracker = engine.new_tracker("c")
        rng = np.random.default_rng(1)
        floor = rng.integers(90, 110, (480, 640, 3)).astype(np.uint8)
        ious = []
        for k in range(25):
            x0, y0 = 200 + 4 * k, 150 + 2 * k
            frame = floor.copy()
            cv2.rectangle(frame, (x0, y0), (x0 + 90, y0 + 50), (200, 60, 40), -1)
            cv2.rectangle(frame, (x0 + 20, y0 + 10), (x0 + 45, y0 + 30), (30, 30, 30), -1)
            truth = np.zeros((480, 640), bool)
            truth[y0:y0 + 51, x0:x0 + 91] = True
            box = (x0 - 4, y0 - 4, x0 + 94, y0 + 54) if k == 0 else None
            out = engine.step([(tracker, frame, box)])[0]
            ious.append((out.mask & truth).sum() / max((out.mask | truth).sum(), 1))
        self.assertGreater(min(ious[1:]), 0.6)
        self.assertLessEqual(tracker.stored_frames, engine.window + 1)
        self.assertGreater(out.score, 0.5)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is needed to write FFV1")
class Sam2InTheReplay(unittest.TestCase):
    """The replay with a stand-in engine that reads the painted opponent's colour."""

    @classmethod
    def setUpClass(cls):
        import test_replay_dataset as trd
        import replay_dataset
        cls.trd, cls.replay = trd, replay_dataset
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.dataset = cls.root / "dataset"
        cls.dataset.mkdir()
        trd.write_dataset(cls.dataset)
        cls.background = cls.root / "background"
        cls.background.mkdir()
        trd.write_background(cls.background)

        class ColourEngine(FakeEngine):
            window = 6

            def step(self, items):
                out = []
                for tracker, image, box in items:
                    if box is not None:
                        tracker.reset()
                    tracker.frame_idx += 1
                    tracker.prompted = True
                    tracker.output_dict["non_cond_frame_outputs"][tracker.frame_idx] = {}
                    tracker.prune()
                    diff = np.abs(image.astype(int) - [200, 80, 60]).sum(axis=2)
                    out.append(so.StepOutput(diff < 40, 0.9, box is not None, 2.0))
                return out

        cls.engine_class = ColourEngine

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_replay(self, name, *extra):
        out = self.root / name
        self.replay.SAM2_ENGINE_FACTORY = self.engine_class
        with open(self.root / f"{name}.log", "w") as sink:
            stdout, sys.stdout = sys.stdout, sink
            stderr, sys.stderr = sys.stderr, sink
            try:
                self.replay.main([str(self.dataset), "--output", str(out), "--camera-background",
                                  str(self.background), "--opponent-stream", "640x480@80",
                                  "--parallel", "inline", "--sam2-hz", "80", *extra])
            finally:
                sys.stdout, sys.stderr = stdout, stderr
                self.replay.SAM2_ENGINE_FACTORY = None
        return out

    def observations(self, out):
        return [json.loads(l) for l in (out / "observations.jsonl").read_text().splitlines()]

    def test_always_feeds_sam2_readings_to_the_opponent(self):
        out = self.run_replay("always", "--sam2-mode", "always", "--sam2-latency-ms", "5")
        sam2 = [r for r in self.observations(out) if r["observation"]["method"] == "sam2"]
        self.assertTrue(sam2)
        accepted = [r for r in sam2 if r["accepted"]]
        self.assertTrue(accepted)
        for r in accepted:
            x, y = r["observation"]["position_m"][:2]
            self.assertLess(math.hypot(x - self.trd.OPPONENT_XY[0], y - self.trd.OPPONENT_XY[1]), 0.1)
            # the reading reaches the filter after the simulated SAM2 latency
            o = r["observation"]
            self.assertEqual(o["processed_time_ns"] - o["receive_time_ns"], 5_000_000)
        timing = json.loads((out / "timing.json").read_text())
        self.assertEqual(timing["sam2"]["mode"], "always")
        self.assertGreater(timing["sam2"]["counts"]["frames"], 0)
        self.assertGreater(timing["sam2"]["prompts"]["operator"], 0)
        self.assertTrue((out / "sam2.jsonl").exists())

    def test_backup_keeps_sam2_in_reserve_while_silhouettes_come(self):
        # The rovers here stand 0.85 m apart: closer than 1 m is an encounter,
        # and SAM2 is fed; at 0.5 m they are not, and SAM2 waits.
        out = self.run_replay("backup_close", "--sam2-mode", "backup")
        counts = json.loads((out / "timing.json").read_text())["sam2"]["counts"]
        self.assertGreater(counts.get("fed:rovers_close", 0), 0)
        out = self.run_replay("backup", "--sam2-mode", "backup", "--identity-close-m", "0.5")
        reasons = {r["selection_reason"] for r in self.observations(out)
                   if r["observation"]["method"] == "sam2"}
        self.assertIn("sam2_standby", reasons)

    def test_blackout_shows_the_empty_arena(self):
        out = self.run_replay("blackout", "--sam2-mode", "always", "--opponent-blackout", "0.03:1")
        rows = self.observations(out)
        late = [r for r in rows if r["observation"]["object_id"] == "opponent"
                and r["observation"]["method"] not in ("operator_box",)
                and r["observation"]["capture_time_ns"] >= self.trd.START_NS + 30_000_000]
        self.assertTrue(late)
        self.assertFalse([r for r in late if r["accepted"]])
        self.assertIn("sam2_reject:empty_mask", {r["selection_reason"] for r in late})
        params = json.loads((out / "runtime_parameters.json").read_text())
        self.assertEqual(params["opponent_blackout_s"], [[0.03, 1.0]])

    def test_sam2_needs_the_small_stream_and_the_opponent(self):
        with self.assertRaises(SystemExit):
            self.replay.parse_args([str(self.dataset), "--output", "x", "--sam2-mode", "backup",
                                    "--camera-background", str(self.background)])
        with self.assertRaises(SystemExit):
            self.replay.parse_args([str(self.dataset), "--output", "x", "--sam2-mode", "backup",
                                    "--opponent-stream", "640x480@30"])
        with self.assertRaises(SystemExit):
            self.replay.parse_args([str(self.dataset), "--output", "x",
                                    "--opponent-blackout", "3:2"])
        a = self.replay.parse_args([str(self.dataset), "--output", "x"])
        self.assertEqual(a.sam2_mode, "off")
        self.assertEqual(a.sam2_max_cameras, 2)
        self.assertEqual(a.sam2_hz, 15.0)


if __name__ == "__main__":
    unittest.main()
