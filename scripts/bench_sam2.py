#!/usr/bin/env python3
"""SAM2 speed on this GPU for the opponent's small stream (prompt 08, item 7).

Real frames of a dataset, resized to the small stream (640x480), go through
``Sam2Engine`` for 1..N cameras:

* ``max``    back to back for ``--seconds``: one call encodes every camera's
             frame of an instant together, then steps each camera's memory.
             Gives calls/s, camera-frames/s, P50/P95 ms per call, VRAM peak.
* ``paced``  the replay's schedule: each camera sends a frame at ``--hz`` on its
             own phase (staggered), one call per frame, in real time.  Gives
             ms per call and the GPU utilisation (nvidia-smi, 100 ms samples).

The prompt is a box around the opponent in each camera's first frame (from
truth: this is a benchmark, not the runtime).  Results go to stdout as JSON
and, with ``--output``, to a file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def load_frames(dataset, cameras, count, size, stride):
    from localization_contracts.camera_model import CameraModel
    from replay_dataset import operator_box_from_truth, read_jsonl
    cfg = json.loads((dataset / "runtime_cameras.json").read_text())
    cams = {c["name"]: c for c in cfg["cameras"]}
    models = {cid: CameraModel.from_config(c) for cid, c in cams.items()}
    stamps = {cid: {int(r["stamp_ns"]): int(r["index"]) for r in read_jsonl(dataset / f"{cid}.jsonl")}
              for cid in cams}
    frames, boxes = {}, {}
    for cid in cameras:
        capture = cv2.VideoCapture(str(dataset / f"{cid}.mkv"))
        images = []
        index = 0
        while len(images) < count:
            ok, bgr = capture.read()
            if not ok:
                break
            if index % stride == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                images.append(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA))
            index += 1
        capture.release()
        frames[cid] = images
        w0, h0 = cams[cid]["image_size"]
        sx, sy = size[0] / w0, size[1] / h0
        # A box around the opponent in the first frame; any camera without
        # one gets the frame centre (timing does not depend on what is seen).
        one = operator_box_from_truth(dataset, {cid: cams[cid]}, {cid: models[cid]},
                                      {cid: stamps[cid]})
        if one is not None and one["frame_index"][cid] < stride:
            b = one["boxes_xyxy_px"][cid]
            boxes[cid] = (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)
        else:
            boxes[cid] = (size[0] * 0.4, size[1] * 0.4, size[0] * 0.6, size[1] * 0.6)
    return frames, boxes


class GpuSampler:
    """nvidia-smi utilisation every 100 ms in a thread."""

    def __init__(self):
        self.samples, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        proc = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                 "--format=csv,noheader,nounits", "-lms", "100"],
                                stdout=subprocess.PIPE, text=True)
        try:
            for line in proc.stdout:
                if self.stop.is_set():
                    break
                try:
                    util, mem = (float(v) for v in line.split(","))
                    self.samples.append((util, mem))
                except ValueError:
                    pass
        finally:
            proc.terminate()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=2)

    def summary(self):
        util = [u for u, _ in self.samples]
        mem = [m for _, m in self.samples]
        if not util:
            return {"util_mean": None, "util_p95": None, "mem_used_mb_max": None}
        return {"util_mean": float(np.mean(util)), "util_p95": float(np.percentile(util, 95)),
                "mem_used_mb_max": float(max(mem)), "samples": len(util)}


def stats(ms):
    ms = np.asarray(ms[5:] if len(ms) > 20 else ms)        # first calls warm up
    return {"n": int(len(ms)), "p50": float(np.percentile(ms, 50)),
            "p95": float(np.percentile(ms, 95)), "max": float(ms.max()),
            "mean": float(ms.mean())}


def run_max(engine, frames, boxes, cameras, seconds):
    trackers = {cid: engine.new_tracker(cid) for cid in cameras}
    engine.reset_peak_memory()
    engine.step([(trackers[cid], frames[cid][0], boxes[cid]) for cid in cameras])
    ms, k = [], 1
    begin = time.perf_counter()
    with GpuSampler() as gpu:
        while time.perf_counter() - begin < seconds:
            items = [(trackers[cid], frames[cid][k % len(frames[cid])], None) for cid in cameras]
            out = engine.step(items)
            ms.append(out[0].gpu_ms)
            k += 1
    wall = time.perf_counter() - begin
    return {"calls_per_s": len(ms) / wall, "camera_frames_per_s": len(ms) * len(cameras) / wall,
            "ms_per_call": stats(ms), "vram_peak_mb": engine.peak_memory_mb(), "gpu": gpu.summary()}


def run_paced(engine, frames, boxes, cameras, seconds, hz):
    trackers = {cid: engine.new_tracker(cid) for cid in cameras}
    engine.reset_peak_memory()
    for cid in cameras:
        engine.step([(trackers[cid], frames[cid][0], boxes[cid])])
    period = 1.0 / hz
    # Staggered phases, as --opponent-stream-stagger.
    schedule = [(i * period / len(cameras), cid) for i, cid in enumerate(cameras)]
    ms, late, k = [], 0, 1
    begin = time.perf_counter()
    with GpuSampler() as gpu:
        n = 0
        while True:
            cycle, slot = divmod(n, len(cameras))
            due = cycle * period + schedule[slot][0]
            if due > seconds:
                break
            now = time.perf_counter() - begin
            if due > now:
                time.sleep(due - now)
            elif now - due > period:
                late += 1
            cid = schedule[slot][1]
            out = engine.step([(trackers[cid], frames[cid][(cycle + 1) % len(frames[cid])], None)])
            ms.append(out[0].gpu_ms)
            n += 1
    return {"hz_per_camera": hz, "ms_per_call": stats(ms), "late_calls": late,
            "vram_peak_mb": engine.peak_memory_mb(), "gpu": gpu.summary()}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=ROOT / "artifacts/dataset_imx219_01")
    p.add_argument("--models", nargs="+", default=["tiny", "small"])
    p.add_argument("--dtypes", nargs="+", default=["bfloat16", "float16"])
    p.add_argument("--cameras", nargs="+", type=int, default=[1, 2, 3, 6])
    p.add_argument("--image-sizes", nargs="+", type=int, default=[512, 1024])
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--paced-hz", nargs="*", type=float, default=[15.0],
                   help="also run each camera count paced at these rates (first dtype only)")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    import torch
    from localization_contracts.sam2_opponent import Sam2Engine
    names = [f"camera_{i}" for i in range(1, 7)]
    frames, boxes = load_frames(a.dataset, names, a.frames, (640, 480), 3)
    machine = {"gpu": torch.cuda.get_device_name(0),
               "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20,
               "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
    try:
        import importlib.metadata as md
        machine["sam2"] = md.version("SAM-2")
    except Exception:
        machine["sam2"] = None
    results = []
    for model in a.models:
        checkpoint = ROOT / f"models/sam2/sam2.1_hiera_{model}.pt"
        for size in a.image_sizes:
            for d, dtype in enumerate(a.dtypes):
                engine = Sam2Engine(checkpoint, dtype=dtype, image_size=size)
                for n in a.cameras:
                    cams = names[:n]
                    row = {"model": model, "image_size": size, "dtype": dtype, "cameras": n,
                           "max": run_max(engine, frames, boxes, cams, a.seconds)}
                    if d == 0:
                        row["paced"] = {str(hz): run_paced(engine, frames, boxes, cams,
                                                           a.seconds, hz) for hz in a.paced_hz}
                    results.append(row)
                    print(json.dumps(row), flush=True)
                del engine
                torch.cuda.empty_cache()
    report = {"machine": machine, "seconds": a.seconds, "frame_size": [640, 480],
              "results": results}
    if a.output:
        a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
