"""Everything one camera does with one frame, and three ways to run six of them.

``CameraWorker`` owns the per-camera state: the frame source, the marker
detector and PnP observer, the opponent's background model, silhouette
observer and KLT flow.  It never reads the filters.  The main thread plans
the frame (marker window, opponent window, the other rover's body to cut out)
from the filters, hands the plan in, and applies what comes back to the
filters in camera order -- so the filters sit behind one owner and the result
does not depend on which pool ran the cameras.

Pools, one interface (``run(jobs) -> results`` in job order):

``InlinePool``   one thread, cameras one after another;
``ThreadPool``   one thread per camera (the GIL serialises the Python parts);
``ProcessPool``  one process per camera, each building its own worker; only
                 the plan and the (small) result cross the pipe, never pixels.

Per-camera state lives in exactly one worker, which sees that camera's frames
in order in every pool, so all three give identical output.
"""
from __future__ import annotations

import collections
import concurrent.futures
import multiprocessing
import time
from pathlib import Path

import numpy as np

from . import roi_tracker
from .apriltag import Detection
from .camera_model import CameraModel
from .cuboid import localize_box
from .foreground import ClipBackground
from .frame_source import DatasetFrameSource
from .image_pipeline import OneCameraImagePipeline
from .opponent_camera import OpponentCamera


def _read_jsonl(path):
    import json
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class CameraWorker:
    """One camera: decode, marker in its window, opponent in its window.

    ``spec`` is a plain, picklable dict (see ``Replay.worker_spec``) so a
    worker can be built inside another process.
    """

    def __init__(self, spec):
        self.spec = spec
        self.camera_id = cid = spec["camera_id"]
        cam = spec["camera"]
        self.camera = cam
        self.model = CameraModel.from_config(cam)
        self.transport_ns = int(spec["transport_ns"])
        self.processing_ns = int(spec["processing_ns"])
        self.pipe = OneCameraImagePipeline(
            cid, np.array(cam["K"]).reshape(3, 3), cam["D"],
            {"rotation": cam["R_world_optical"], "translation": cam["position_world"]},
            spec["tags"], spec["calibration_version"], image_size=cam["image_size"],
            **spec["pipeline"])
        self.source = DatasetFrameSource(
            Path(spec["dataset"]) / f"{cid}.mkv", cid, transport_ns=self.transport_ns,
            line_time_ns=spec["pipeline"].get("line_time_ns", 0.0), gain=spec["gain"])
        self.opponent = None
        self.background_build_s = 0.0
        opponent = spec.get("opponent")
        if opponent is not None:
            started = time.monotonic()
            clip = DatasetFrameSource(Path(opponent["background_dir"]) / f"{cid}.mkv", cid)
            rows = _read_jsonl(Path(opponent["background_dir"]) / f"{cid}.jsonl")
            background = ClipBackground.from_frames(
                (clip.read_image(int(row["index"])) for row in rows[::opponent["stride"]]),
                threshold=opponent["threshold"], alpha=opponent["alpha"])
            clip.close()
            self.opponent = OpponentCamera(
                cid, self.model, cam["R_world_optical"], cam["position_world"], background,
                size_m=opponent["size"], tag_size_m=opponent["tag_size"],
                gate_m=opponent["gate_m"])
            self.background_build_s = time.monotonic() - started

    def process(self, row, plan, job=None):
        """One frame.  ``job`` is the opponent's plan for it, or None."""
        cid = self.camera_id
        busy = plan.mode != roi_tracker.IDLE or (
            job is not None and job["plan"].mode != roi_tracker.IDLE)
        begin = time.perf_counter_ns()
        frame = self.source.read(row, decode=busy)
        decode_ms = (time.perf_counter_ns() - begin) / 1e6
        if not busy:
            return {"idle": True, "decode_ms": decode_ms}
        image = frame.image
        if [image.shape[1], image.shape[0]] != list(self.camera["image_size"]):
            raise ValueError(f"{cid}: image size differs from calibration")
        stamp = frame.stamp_ns
        received = frame.receive_ns
        processed = received + self.processing_ns
        pipe = self.pipe
        begin = time.perf_counter_ns()
        hits = ()
        if plan.mode == roi_tracker.ROI:
            hits = pipe.roi_detector.detect(image, roi=plan.roi)
        elif plan.mode != roi_tracker.IDLE:
            hits = pipe.detector.detect(image)
        detect_ms = (time.perf_counter_ns() - begin) / 1e6
        observations, diagnostics, qualities, reprojection = [], [], [], []
        rejections = collections.Counter()
        for hit in hits:
            obs = pipe.observer.observe(Detection(
                cid, frame.sequence, hit.tag_id, hit.corners, stamp, received, processed))
            diagnostic = pipe.observer.last_diagnostic or {}
            diagnostics.append({"tag_id": int(hit.tag_id), **diagnostic})
            if obs is not None:
                observations.append(obs)
                qualities.append(float(obs.quality))
                reprojection.append(float(obs.pixel_features["reprojection_error_px"]))
            else:
                rejections[str(diagnostic.get("reason", "unknown"))] += 1
        tag_ms = (time.perf_counter_ns() - begin) / 1e6
        result = {"idle": False, "decode_ms": decode_ms, "tag_ms": tag_ms,
                  "stages": {"detect_ms": detect_ms, "pnp_ms": tag_ms - detect_ms},
                  "hits": hits, "observations": observations, "diagnostics": diagnostics,
                  "qualities": qualities, "reprojection": reprojection,
                  "rejections": dict(rejections), "stamp": stamp,
                  "received": received, "processed": processed,
                  "opponent_ms": 0.0, "background_update_ms": 0.0}
        if job is not None:
            result.update(self.process_opponent(image, job, result["stages"]))
        result["latency_ms"] = tag_ms + result["opponent_ms"]
        return result

    def process_opponent(self, image, job, stages):
        camera = self.opponent
        plan = job["plan"]
        out = {"opponent_reading": None, "opponent_diag": None}
        begin = time.perf_counter_ns()
        gain = camera.background.gain(image, job["exclude"])
        stages["gain_ms"] = (time.perf_counter_ns() - begin) / 1e6
        if plan.mode != roi_tracker.IDLE:
            roi = plan.roi if plan.roi is not None else (0, 0, image.shape[1], image.shape[0])
            reading, diag = camera.read(image, roi, job["prediction"], job["tag_pose"], gain,
                                        job.get("gate_m"))
            out["opponent_reading"], out["opponent_diag"] = reading, diag
            stages["foreground_ms"] = diag.get("foreground_ms", 0.0)
            stages["silhouette_ms"] = diag.get("measure_ms", 0.0)
            if reading is None and job.get("operator_box") is not None:
                # The operator's rectangle stands even when the silhouette in
                # it cannot be read: fall back to the box itself, through the
                # camera's own lens.
                out["opponent_box_fit"] = localize_box(
                    job["operator_box"], self.camera, dimensions=self.spec["opponent"]["size"],
                    camera_model=self.model)
        out["opponent_ms"] = (time.perf_counter_ns() - begin) / 1e6
        begin = time.perf_counter_ns()
        camera.background.update(image, job["exclude"], gain)
        out["background_update_ms"] = stages["background_update_ms"] = \
            (time.perf_counter_ns() - begin) / 1e6
        return out

    def close(self):
        self.source.close()


class InlinePool:
    name = "inline"

    def __init__(self, specs):
        self.workers = {spec["camera_id"]: CameraWorker(spec) for spec in specs}

    def background_build_s(self):
        return sum(w.background_build_s for w in self.workers.values())

    def run(self, jobs):
        """``jobs``: list of (camera_id, row, plan, opponent_job)."""
        return [self.workers[cid].process(row, plan, job) for cid, row, plan, job in jobs]

    def close(self):
        for worker in self.workers.values():
            worker.close()


class ThreadPool(InlinePool):
    name = "threads"

    def __init__(self, specs, max_workers=None):
        self.workers = {}
        # Building the workers is itself per camera (six background clips).
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as build:
            for worker in build.map(CameraWorker, specs):
                self.workers[worker.camera_id] = worker
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers or len(specs))

    def run(self, jobs):
        futures = [self.executor.submit(self.workers[cid].process, row, plan, job)
                   for cid, row, plan, job in jobs]
        return [future.result() for future in futures]

    def close(self):
        self.executor.shutdown()
        super().close()


def _serve(connection, spec, threads):
    import cv2
    cv2.setNumThreads(threads)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(threads)
    except ImportError:
        pass
    worker = CameraWorker(spec)
    connection.send(("ready", worker.background_build_s))
    while True:
        message = connection.recv()
        if message is None:
            break
        try:
            connection.send(("ok", worker.process(*message)))
        except Exception as error:          # noqa: BLE001 -- reported to the parent
            connection.send(("error", repr(error)))
    worker.close()
    connection.close()


class ProcessPool:
    """One process per camera.  Frames are decoded where they are processed."""

    name = "processes"

    def __init__(self, specs, threads_per_process=1):
        context = multiprocessing.get_context("spawn")
        self.links = {}
        self.processes = []
        for spec in specs:
            parent, child = context.Pipe()
            process = context.Process(target=_serve, args=(child, spec, threads_per_process),
                                      daemon=True)
            process.start()
            self.links[spec["camera_id"]] = parent
            self.processes.append(process)
        self.build_s = 0.0
        for cid, link in self.links.items():
            status, value = link.recv()
            if status != "ready":
                raise RuntimeError(f"{cid}: worker failed to start: {value}")
            self.build_s += value

    def background_build_s(self):
        return self.build_s

    def run(self, jobs):
        for cid, row, plan, job in jobs:
            self.links[cid].send((row, plan, job))
        results = []
        for cid, _, _, _ in jobs:
            status, value = self.links[cid].recv()
            if status != "ok":
                raise RuntimeError(f"{cid}: {value}")
            results.append(value)
        return results

    def close(self):
        for link in self.links.values():
            link.send(None)
        for process in self.processes:
            process.join(timeout=10)


POOLS = {"inline": InlinePool, "threads": ThreadPool, "processes": ProcessPool}
