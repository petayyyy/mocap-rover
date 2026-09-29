#!/usr/bin/env python3
"""Record a replayable dataset from the Gazebo arena without ROS or rosbag.

Everything is plain files so the recording can be replayed by any process,
in any language, without a ROS installation or a transport partition:

    <output>/
      meta.json                 world, topics, counts, drops, sim time span
      runtime_cameras.json      calibration handed to the localization runtime
      cameras_ground_truth.json evaluator-only camera truth (copied when present)
      camera_N.mkv              lossless FFV1 video, RGB, one file per camera
      camera_N.jsonl            one row per stored frame: index, sim stamp, wall time
      lidar/scan_NNNNNN.npz     ranges float32 (vertical x horizontal) + angles
      lidar.jsonl               one row per scan: file, sim stamp, wall time
      truth.jsonl               PosePublisher rows for both rovers (evaluation only)
      clock.jsonl               sim clock samples, one per 100 ms

Frames are stored in the order they arrive.  A frame is never dropped for
being late: the writer queues are unbounded and the simulation is far slower
than the disk.  The sim stamp comes from the image header, which Gazebo sets
at render time, so replay tools must read camera_N.jsonl rather than assume a
constant period.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.image_pb2 import Image
from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localization_contracts.capture import rgb_array          # noqa: E402
from localization_contracts.lidar_pipeline import parse_laser_scan  # noqa: E402


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nsec)


class VideoWriter(threading.Thread):
    """Feed raw RGB frames to one ffmpeg process; index rows go to a jsonl."""

    def __init__(self, path: Path, width, height, fps, codec_args):
        super().__init__(daemon=True)
        self.path = path
        self.queue: queue.Queue = queue.Queue()
        self.index = path.with_suffix(".jsonl").open("w")
        self.count = 0
        self.width, self.height = width, height
        self.process = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
             "-i", "-", *codec_args, str(path)],
            stdin=subprocess.PIPE)

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            frame, row = item
            self.process.stdin.write(frame)
            row["index"] = self.count
            self.index.write(json.dumps(row) + "\n")
            self.count += 1
        self.process.stdin.close()
        self.process.wait()
        self.index.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="runtime_cameras.json of the world being recorded")
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=float, default=30.0,
                   help="Simulated seconds to record, measured on /clock")
    p.add_argument("--clock-topic", default="/clock")
    p.add_argument("--no-lidar", action="store_true")
    p.add_argument("--codec", choices=("ffv1", "x264-lossless"), default="ffv1")
    p.add_argument("--wall-timeout", type=float, default=3600.0,
                   help="Give up after this many wall seconds")
    a = p.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    cams = {c["name"]: c for c in cfg["cameras"]}
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; choose a new --output")
    out.mkdir(parents=True, exist_ok=True)
    (out / "lidar").mkdir(exist_ok=True)
    shutil.copy(a.config, out / "runtime_cameras.json")
    truth_path = Path(a.config).with_name("cameras_ground_truth.json")
    if truth_path.exists():
        shutil.copy(truth_path, out / "cameras_ground_truth.json")

    codec = (["-c:v", "ffv1", "-level", "3", "-threads", "2"] if a.codec == "ffv1"
             else ["-c:v", "libx264", "-qp", "0", "-preset", "veryfast",
                   "-pix_fmt", "rgb24", "-threads", "2"])
    writers = {}
    for cid, c in cams.items():
        w, h = c["image_size"]
        writers[cid] = VideoWriter(out / f"{cid}.mkv", w, h,
                                   cfg.get("camera_fps", 80), codec)
        writers[cid].start()

    lock = threading.Lock()
    stop = threading.Event()
    state = {"sim_ns": None, "start_ns": None, "last_clock_wall": None}
    counts = {"frames": dict.fromkeys(cams, 0), "size_errors": 0, "scans": 0, "truth": 0}
    truth_file = (out / "truth.jsonl").open("w")
    clock_file = (out / "clock.jsonl").open("w")
    lidar_index = (out / "lidar.jsonl").open("w")
    lidar_queue: queue.Queue = queue.Queue()

    def image_cb(msg, cid):
        try:
            frame = rgb_array(msg)
        except ValueError:
            with lock:
                counts["size_errors"] += 1
            return
        expected = tuple(cams[cid]["image_size"])
        if (msg.width, msg.height) != expected:
            with lock:
                counts["size_errors"] += 1
            return
        row = {"camera_id": cid, "stamp_ns": stamp_ns(msg.header),
               "wall_ns": time.monotonic_ns()}
        writers[cid].queue.put((np.ascontiguousarray(frame).tobytes(), row))
        with lock:
            counts["frames"][cid] += 1

    def lidar_cb(msg):
        lidar_queue.put((msg, time.monotonic_ns()))

    def lidar_worker():
        while True:
            item = lidar_queue.get()
            if item is None:
                break
            msg, wall = item
            parsed = parse_laser_scan(msg)
            with lock:
                n = counts["scans"]
                counts["scans"] += 1
            name = f"scan_{n:06d}.npz"
            np.savez_compressed(out / "lidar" / name,
                                ranges=parsed["ranges"].astype(np.float32),
                                azimuth=parsed["azimuth"].astype(np.float32),
                                elevation=parsed["elevation"].astype(np.float32),
                                range_min=parsed["range_min"], range_max=parsed["range_max"])
            lidar_index.write(json.dumps({"file": f"lidar/{name}", "stamp_ns": parsed["stamp_ns"],
                                          "wall_ns": wall}) + "\n")

    def pose_cb(msg, name):
        for q in msg.pose:
            if q.name != name:
                continue
            quat = q.orientation
            yaw = np.arctan2(2 * (quat.w * quat.z + quat.x * quat.y),
                             1 - 2 * (quat.y ** 2 + quat.z ** 2))
            row = {"object_id": "opponent" if name == "yolo_rover" else name,
                   "model": name, "stamp_ns": stamp_ns(q.header),
                   "x": q.position.x, "y": q.position.y, "z": q.position.z,
                   "yaw": float(yaw), "qx": quat.x, "qy": quat.y, "qz": quat.z, "qw": quat.w,
                   "wall_ns": time.monotonic_ns()}
            with lock:
                truth_file.write(json.dumps(row) + "\n")
                counts["truth"] += 1

    last_clock_row = [0]

    def clock_cb(msg):
        sim = int(msg.sim.sec) * 1_000_000_000 + int(msg.sim.nsec)
        with lock:
            if state["start_ns"] is None:
                state["start_ns"] = sim
            state["sim_ns"] = sim
            state["last_clock_wall"] = time.monotonic_ns()
            if sim - last_clock_row[0] >= 100_000_000:
                last_clock_row[0] = sim
                clock_file.write(json.dumps({"sim_ns": sim, "wall_ns": time.monotonic_ns()}) + "\n")

    node = Node()
    topics = []
    if not node.subscribe(Clock, a.clock_topic, clock_cb):
        raise SystemExit(f"cannot subscribe {a.clock_topic}")
    topics.append(a.clock_topic)
    for cid in cams:
        topic = f"/cameras/{cid}/image"
        if not node.subscribe(Image, topic, lambda msg, cid=cid: image_cb(msg, cid)):
            raise SystemExit(f"cannot subscribe {topic}")
        topics.append(topic)
    lidar_thread = None
    lidar_topic = (cfg.get("lidar") or {}).get("topic")
    if lidar_topic and not a.no_lidar:
        if not node.subscribe(LaserScan, lidar_topic, lidar_cb):
            raise SystemExit(f"cannot subscribe {lidar_topic}")
        topics.append(lidar_topic)
        lidar_thread = threading.Thread(target=lidar_worker, daemon=True)
        lidar_thread.start()
    for name in ("tag_rover", "yolo_rover"):
        topic = f"/model/{name}/pose"
        node.subscribe(Pose_V, topic, lambda msg, name=name: pose_cb(msg, name))
        topics.append(topic)

    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    wall_start = time.monotonic()
    print(f"recording to {out}; waiting for /clock", flush=True)
    while not stop.wait(1.0):
        with lock:
            sim, start = state["sim_ns"], state["start_ns"]
            frames = dict(counts["frames"])
            scans, truth = counts["scans"], counts["truth"]
        elapsed_sim = 0 if sim is None else (sim - start) / 1e9
        wall = time.monotonic() - wall_start
        if sim is not None:
            print(f"sim {elapsed_sim:6.2f}s wall {wall:6.1f}s rtf {elapsed_sim/max(wall,1e-9):.3f} "
                  f"frames {sum(frames.values())} scans {scans} truth {truth} "
                  f"queued {sum(w.queue.qsize() for w in writers.values())}", flush=True)
        if sim is not None and elapsed_sim >= a.seconds:
            break
        if wall > a.wall_timeout:
            print("wall timeout", flush=True)
            break
    for topic in topics:
        node.unsubscribe(topic)
    for w in writers.values():
        w.queue.put(None)
    for w in writers.values():
        w.join()
    if lidar_thread is not None:
        lidar_queue.put(None)
        lidar_thread.join()
    with lock:
        truth_file.close()
        clock_file.close()
        lidar_index.close()
        meta = {
            "world": cfg.get("world"),
            "calibration_version": cfg.get("calibration_version"),
            "topics": topics,
            "cameras": {cid: {"image_size": cams[cid]["image_size"], "video": f"{cid}.mkv",
                              "index": f"{cid}.jsonl", "frames": writers[cid].count}
                        for cid in cams},
            "lidar": {"topic": lidar_topic, "scans": counts["scans"]} if lidar_thread else None,
            "truth_rows": counts["truth"],
            "image_size_errors": counts["size_errors"],
            "sim_start_ns": state["start_ns"],
            "sim_end_ns": state["sim_ns"],
            "wall_seconds": time.monotonic() - wall_start,
            "codec": a.codec,
            "pixel_format": "rgb24",
        }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
