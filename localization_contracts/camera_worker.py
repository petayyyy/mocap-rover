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

import cv2
import numpy as np

from . import roi_tracker
from .apriltag import Detection
from .camera_model import CameraModel
from .cuboid import localize_box
from .foreground import ClipBackground
from .frame_source import CameraFrame, DatasetFrameSource
from .image_pipeline import OneCameraImagePipeline
from .link_emulation import SensorPath, clip_roi, jpeg_roundtrip
from .opponent_camera import OpponentCamera
from .ray_plane import pixel_rays, ray_plane

# A lost opponent is looked for over the whole frame at most this often per
# camera.
REACQUIRE_PERIOD_NS = 250_000_000


def scaled_camera(cam, width, height):
    """Calibration of the whole frame resized to ``width`` x ``height``; (camera, (sx, sy))."""
    w0, h0 = cam["image_size"]
    sx, sy = width / w0, height / h0
    K = list(cam["K"])
    K[0] *= sx; K[4] *= sy
    K[2] = (K[2] + 0.5) * sx - 0.5
    K[5] = (K[5] + 0.5) * sy - 0.5
    return {**cam, "K": K, "image_size": [int(width), int(height)]}, (sx, sy)


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
        self.prefetched = None          # (index, image, decode_ms)
        # Link emulation (replay only): what the camera node would deliver.
        link = spec.get("link") or {}
        self.link_mode = link.get("mode", "ideal")
        self.sensor = SensorPath(self.link_mode) if self.link_mode in ("cm4", "cm5") else None
        self.jpeg_quality = int(link.get("jpeg_quality", 90))
        # This camera's windows are planned here, from the track poses the
        # main thread sends: the planners' state (misses, watchdog) is per
        # camera, and so is all the projecting.
        R, C = cam["R_world_optical"], cam["position_world"]
        self.planner = roi_tracker.CameraRoiPlanner(self.model, R, C, **spec["planner"])
        self.opponent_planner = (None if spec.get("opponent_planner") is None else
                                 roi_tracker.CameraRoiPlanner(self.model, R, C,
                                                              **spec["opponent_planner"]))
        self.last_reacquire_ns = -10**18
        self.background_build_s = 0.0
        # Replay only: the opponent reads a second, smaller stream at a lower
        # rate (``opponent_stream`` = (width, height, hz)); the marker keeps
        # the full-resolution frames.  None: one stream for both.
        stream = spec.get("opponent_stream")
        self.opp_size = self.opp_period_ns = self.opp_next_ns = None
        self.opp_phase_ns = 0
        self.opp_camera, self.opp_model, self.opp_scale = cam, self.model, (1.0, 1.0)
        if stream:
            w, h, hz = int(stream[0]), int(stream[1]), float(stream[2])
            self.opp_camera, (sx, sy) = scaled_camera(cam, w, h)
            self.opp_model = CameraModel.from_config(self.opp_camera)
            self.opp_size, self.opp_scale = (w, h), (sx, sy)
            self.opp_period_ns = int(1e9 / hz)
            if len(stream) > 3 and stream[3]:
                index = int("".join(ch for ch in cid if ch.isdigit()) or 1) - 1
                self.opp_phase_ns = index * self.opp_period_ns // int(stream[3])
            self.opponent_planner = (None if spec.get("opponent_planner") is None else
                                     roi_tracker.CameraRoiPlanner(self.opp_model, R, C,
                                                                  **spec["opponent_planner"]))
        # Replay only: SAM2 in the main process asks for this camera's small
        # frame (``sam2_frame``), and an opponent blackout replaces what the
        # opponent sees with the empty arena (``blackout``).
        self.want_sam2 = False
        self.blackout = False
        self.blackout_image = None
        opponent = spec.get("opponent")
        if opponent is not None:
            started = time.monotonic()
            clip = DatasetFrameSource(Path(opponent["background_dir"]) / f"{cid}.mkv", cid)
            rows = _read_jsonl(Path(opponent["background_dir"]) / f"{cid}.jsonl")
            # The empty-arena clip goes through the same sensor path as the
            # live frames, or the model would compare grey with colour.
            convert = self.sensor or (lambda image: image)
            shrink = ((lambda image: cv2.resize(image, self.opp_size, interpolation=cv2.INTER_AREA))
                      if self.opp_size else (lambda image: image))
            background = ClipBackground.from_frames(
                (shrink(convert(clip.read_image(int(row["index"])))) for row in rows[::opponent["stride"]]),
                threshold=opponent["threshold"], alpha=opponent["alpha"])
            clip.close()
            self.opponent = OpponentCamera(
                cid, self.opp_model, cam["R_world_optical"], cam["position_world"], background,
                size_m=opponent["size"], tag_size_m=opponent["tag_size"],
                gate_m=opponent["gate_m"], extent_plane_z=opponent.get("extent_plane_z"),
                along_sigma_scale=opponent.get("along_sigma_scale", 0.10))
            # The empty arena as the opponent's stream sees it, for blackouts.
            self.blackout_image = background.mean_u8.copy()
            self.background_build_s = time.monotonic() - started

    def plan(self, context):
        """(marker plan, opponent job or None) for one frame, from the track poses."""
        now_ns = context["now_ns"]
        if context.get("tag_allowed", True):
            plan = self.planner.plan(context["prediction"], now_ns)
        else:
            plan = roi_tracker.Plan(roi_tracker.IDLE, None, None, "tag_camera_limit")
        if not context["opponent_enabled"]:
            return plan, None
        spec = self.spec["opponent"]
        tag, opp = context["tag"], context["opp"]
        camera = self.opponent
        exclude = [r for r in (
            camera.exclusion_rect(tag[:2], tag[2], spec["tag_size"]) if tag else None,
            camera.exclusion_rect(opp[:2], opp[2], spec["size"]) if opp else None)
            if r is not None]
        box = context.get("operator_box")
        if box is not None:
            sx, sy = self.opp_scale
            x0, y0, x1, y1 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
            pad = max(40.0, 0.3 * max(x1 - x0, y1 - y0))
            roi = (int(x0 - pad), int(y0 - pad), int(x1 - x0 + 2 * pad), int(y1 - y0 + 2 * pad))
            centre = pixel_rays(self.opp_model, [((x0 + x1) / 2, (y0 + y1) / 2)])[0]
            point, _ = ray_plane(centre, self.camera["R_world_optical"],
                                 self.camera["position_world"], spec["size"][2] / 2)
            if point is None:
                return plan, None
            return plan, {"plan": roi_tracker.Plan(roi_tracker.ROI, roi, None, "operator_box"),
                          "prediction": point[:2], "tag_pose": tag and tag[:3],
                          "exclude": exclude, "gate_m": 1.5,
                          "operator_box": (x0, y0, x1, y1)}
        if opp is not None and not context.get("opponent_allowed", True):
            # One of the cameras over the limit on opponent windows this
            # instant: learn the background, look elsewhere.
            return plan, {"plan": roi_tracker.Plan(roi_tracker.IDLE, None, None, "opponent_camera_limit"),
                          "prediction": None, "tag_pose": None, "exclude": exclude}
        if opp is not None:
            opponent_plan = self.opponent_planner.plan(
                None if self.spec["no_roi_tracking"] else (opp[0], opp[1], opp[3]), now_ns)
            return plan, {"plan": opponent_plan, "prediction": opp[:2],
                          "tag_pose": tag and tag[:3], "exclude": exclude,
                          "gate_m": max(spec["gate_m"], 3.0 * opp[3])}
        if (tag is not None and context["operator_done"]
                and now_ns - self.last_reacquire_ns >= REACQUIRE_PERIOD_NS):
            # The opponent track is gone.  The marker can only ever name
            # tag_rover, so the other rover is whatever blob is left well
            # away from it -- looked for over the whole frame, sparingly.
            self.last_reacquire_ns = now_ns
            return plan, {"plan": roi_tracker.Plan(roi_tracker.ACQUIRE, None, None,
                                                   "opponent_reacquire"),
                          "prediction": None, "tag_pose": tag[:3], "exclude": exclude,
                          "reacquire": True}
        if exclude:
            # No opponent track here: still learn the background.
            return plan, {"plan": roi_tracker.Plan(roi_tracker.IDLE, None, None, "no_track"),
                          "prediction": None, "tag_pose": None, "exclude": exclude}
        return plan, None

    def process(self, row, context):
        """Plan, read and report one frame; the plans come back in the result."""
        plan, job = self.plan(context)
        self.opp_tick = False
        self.blackout = bool(context.get("blackout")) and self.blackout_image is not None
        if self.opp_period_ns:
            # The opponent's stream has a frame only on its own, slower grid,
            # and the node sends that frame whether or not anyone reads it.
            stamp = int(row["stamp_ns"])
            if self.opp_next_ns is None and self.opp_phase_ns:
                # Staggered: each camera's small frame leaves at its own phase
                # of the period, so the six never hit the port together.
                self.opp_next_ns = stamp + self.opp_phase_ns
            if self.opp_next_ns is None or stamp >= self.opp_next_ns:
                self.opp_tick = True
                self.opp_next_ns = (self.opp_next_ns or stamp) + self.opp_period_ns
                while self.opp_next_ns <= stamp:
                    self.opp_next_ns += self.opp_period_ns
            if job is not None and not self.opp_tick and context.get("operator_box") is None:
                job = None
        # SAM2 reads the small stream: a frame exists on the stream's grid
        # (and at the operator's instant, when the prompt is drawn).
        self.want_sam2 = bool(context.get("sam2_frame")) and (
            self.opp_tick or context.get("operator_box") is not None)
        result = self.read(row, plan, job)
        result["plan"], result["job"] = plan, job
        if not result["idle"]:
            if plan.mode != roi_tracker.IDLE:
                self.planner.report(bool(result["hits"]))
            if (job is not None and job["plan"].mode != roi_tracker.IDLE
                    and job["plan"].reason not in ("operator_box", "opponent_reacquire")):
                self.opponent_planner.report(result.get("opponent_reading") is not None)
        return result

    def read(self, row, plan, job=None):
        """One frame.  ``job`` is the opponent's plan for it, or None."""
        cid = self.camera_id
        busy = plan.mode != roi_tracker.IDLE or (
            job is not None and job["plan"].mode != roi_tracker.IDLE) or (
            self.sensor is not None and getattr(self, "opp_tick", False)) or self.want_sam2
        begin = time.perf_counter_ns()
        index = int(row["index"])
        if self.prefetched is not None and self.prefetched[0] == index:
            _, image, decode_ms = self.prefetched
            stamp = int(row["stamp_ns"])
            frame = CameraFrame(cid, image if busy else None, stamp, 0, 0,
                                self.source.line_time_ns, 0, stamp + self.transport_ns, index)
        else:
            frame = self.source.read(row, decode=busy)
            decode_ms = (time.perf_counter_ns() - begin) / 1e6
        self.prefetched = None
        if not busy:
            return {"idle": True, "decode_ms": decode_ms}
        image = frame.image
        if [image.shape[1], image.shape[0]] != list(self.camera["image_size"]):
            raise ValueError(f"{cid}: image size differs from calibration")
        link_windows, jpeg_decode_ms = [], 0.0
        opponent_image = image
        if self.sensor is not None:
            image, opponent_image, link_windows, jpeg_decode_ms = self.emulate_link(image, plan, job)
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
        if job is not None or self.want_sam2:
            if self.opp_size:
                opponent_image = cv2.resize(opponent_image, self.opp_size, interpolation=cv2.INTER_AREA)
            if self.blackout:
                opponent_image = self.blackout_image.copy()
        if job is not None:
            result.update(self.process_opponent(opponent_image, job, result["stages"]))
        if self.want_sam2:
            result["sam2_image"] = opponent_image
        # The laptop decodes the JPEG windows; that is part of its frame time.
        result["latency_ms"] = tag_ms + result["opponent_ms"] + jpeg_decode_ms
        result["link_windows"] = link_windows
        result["jpeg_decode_ms"] = jpeg_decode_ms
        if self.sensor is not None:
            result["stages"]["jpeg_decode_ms"] = jpeg_decode_ms
        return result

    def emulate_link(self, rgb, plan, job):
        """The frame as the node would deliver it for this plan.

        Returns (marker image, opponent image, windows, JPEG decode ms);
        windows are (width, height, fmt, bytes) as sent.  A full frame is
        sent once, raw, whichever of the two plans asked for it.
        """
        base = self.sensor(rgb)
        height, width = base.shape[:2]
        windows, decode_ms = [], 0.0
        images = []
        full = False
        for p in (plan, None if (job is None or self.opp_size) else job["plan"]):
            if p is None or p.mode == roi_tracker.IDLE:
                images.append(base)
                continue
            if p.mode != roi_tracker.ROI or p.roi is None:
                full = True
                images.append(base)
                continue
            clipped = clip_roi(p.roi, width, height)
            if clipped is None:
                images.append(base)
                continue
            if p.fmt == "jpeg":
                image = base.copy()
                size, ms = jpeg_roundtrip(image, clipped, self.jpeg_quality)
                decode_ms += ms
                windows.append((clipped[2], clipped[3], "jpeg", size))
                images.append(image)
            else:
                windows.append((clipped[2], clipped[3], "raw", clipped[2] * clipped[3]))
                images.append(base)
        if full:
            windows.append((width, height, "raw", width * height))
        if self.opp_size and getattr(self, "opp_tick", False):
            w, h = self.opp_size
            windows.append((w, h, "raw", w * h))       # the small stream's frame
        return images[0], images[1], windows, decode_ms

    def prefetch(self):
        """Decode the next frame now, while the caller is busy elsewhere.

        Whether a frame will be looked at is only known when its plan comes,
        so this decodes it either way; an idle frame's decode is then wasted
        on this worker, never on the main thread.  The frame is the same one
        ``process`` would have read, so the output does not change.
        """
        index = self.source.next_index
        begin = time.perf_counter_ns()
        try:
            image = self.source.read_image(index)
        except RuntimeError:             # past the end of the video
            return
        self.prefetched = (index, image, (time.perf_counter_ns() - begin) / 1e6)

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
                    job["operator_box"], self.opp_camera, dimensions=self.spec["opponent"]["size"],
                    camera_model=self.opp_model)
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
        """``jobs``: list of (camera_id, row, context)."""
        return [self.workers[cid].process(row, context) for cid, row, context in jobs]

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
        futures = [self.executor.submit(self.workers[cid].process, row, context)
                   for cid, row, context in jobs]
        return [future.result() for future in futures]

    def close(self):
        self.executor.shutdown()
        super().close()


def cpu_plan():
    """(main_cpu, worker_cpus) for a hybrid CPU, or None where it cannot be read.

    The main thread holds both filters and is the serial part of every
    instant, so it gets the fastest core to itself (its hyper-thread sibling
    stays empty); the camera processes share the rest, except the slowest
    cluster (on the Core Ultra 155H two 2.5 GHz low-power cores, where one
    stray worker would hold up every instant).
    """
    import os
    root = Path("/sys/devices/system/cpu")
    try:
        cpus = sorted(os.sched_getaffinity(0))
        top = {c: int((root / f"cpu{c}/cpufreq/cpuinfo_max_freq").read_text()) for c in cpus}
    except (OSError, ValueError, AttributeError):
        return None
    fastest = max(top.values())
    slowest = min(top.values())
    main = min(c for c in cpus if top[c] == fastest)
    try:
        siblings = (root / f"cpu{main}/topology/thread_siblings_list").read_text().strip()
        taken = set()
        for part in siblings.split(","):
            low, _, high = part.partition("-")
            taken.update(range(int(low), int(high or low) + 1))
    except (OSError, ValueError):
        taken = {main}
    workers = [c for c in cpus if c not in taken and (top[c] > slowest or fastest == slowest)]
    return main, workers


def _serve(connection, spec, threads, prefetch=True, cpus=None):
    import os
    if cpus:
        os.sched_setaffinity(0, cpus)
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
            continue
        if prefetch:
            worker.prefetch()
    worker.close()
    connection.close()


class ProcessPool:
    """One process per camera.  Frames are decoded where they are processed."""

    name = "processes"

    def __init__(self, specs, threads_per_process=1, prefetch=True, pin=False):
        import os
        context = multiprocessing.get_context("spawn")
        plan = cpu_plan() if pin else None
        self.cpu_plan = plan
        self.links = {}
        self.processes = []
        for spec in specs:
            parent, child = context.Pipe()
            process = context.Process(
                target=_serve, args=(child, spec, threads_per_process, prefetch,
                                     plan[1] if plan else None), daemon=True)
            process.start()
            self.links[spec["camera_id"]] = parent
            self.processes.append(process)
        self.build_s = 0.0
        for cid, link in self.links.items():
            status, value = link.recv()
            if status != "ready":
                raise RuntimeError(f"{cid}: worker failed to start: {value}")
            self.build_s += value
        if plan:
            os.sched_setaffinity(0, {plan[0]})

    def background_build_s(self):
        return self.build_s

    def run(self, jobs):
        for cid, row, context in jobs:
            self.links[cid].send((row, context))
        results = []
        for cid, _, _ in jobs:
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
