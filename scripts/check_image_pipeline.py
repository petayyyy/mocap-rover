#!/usr/bin/env python3
"""Run the real simulation image -> detector -> PnP path on Gazebo topics.

This intentionally does not subscribe to world pose topics and does not use
ground truth.  Camera transforms come from the simulation calibration snapshot;
K/D are taken from the live CameraInfo topic, so the result is a runtime
image-path smoke check, not acceptance.
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.msgs10.camera_info_pb2 import CameraInfo
from gz.transport13 import Node

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization_contracts.apriltag import Detection, PnpAprilTagObserver, TagConfig
from localization_contracts.detector import AprilTagImageDetector


def stamp_ns(msg, fallback):
    try:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nsec)
    except (AttributeError, TypeError):
        return fallback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/cameras.json")
    p.add_argument("--camera", default="camera_1")
    p.add_argument("--seconds", type=float, default=10.0)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    camera = next(c for c in config["cameras"] if c["name"] == args.camera)
    observer = None
    detector = AprilTagImageDetector()
    node = Node(); info_ready = threading.Event(); lock = threading.Lock()
    stats = {"frames": 0, "detections": 0, "accepted": 0, "latencies_ms": [], "ids": {}}
    start = time.monotonic()

    def info_callback(msg):
        nonlocal observer
        k = list(msg.intrinsics.k)
        if len(k) != 9:
            return
        observer = PnpAprilTagObserver(
            TagConfig(calibration_version="gazebo-camera-info"), np.asarray(k).reshape(3, 3), list(msg.distortion.k),
            {"rotation": camera["R_world_optical"], "translation": camera["position_world"]},
            {"rotation": config["tag"]["R_base_tag"], "translation": config["tag"]["T_base_tag_translation"]},
        )
        info_ready.set()

    def callback(msg):
        now = time.monotonic_ns()
        if msg.pixel_format_type != 3:
            return
        raw = b"".join(msg.data[r * msg.step:r * msg.step + msg.width * 3] for r in range(msg.height))
        image = np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        hits = detector.detect(image)
        capture = stamp_ns(msg, now)
        with lock:
            stats["frames"] += 1; stats["detections"] += len(hits)
            for hit in hits:
                stats["ids"][str(hit.tag_id)] = stats["ids"].get(str(hit.tag_id), 0) + 1
                detection = Detection(args.camera, stats["frames"], hit.tag_id, hit.corners,
                    capture, now, time.monotonic_ns())
                if observer is not None and observer.observe(detection) is not None:
                    stats["accepted"] += 1
            stats["latencies_ms"].append((time.monotonic_ns() - now) / 1e6)

    info_topic=f"/cameras/{args.camera}/camera_info"; image_topic=f"/cameras/{args.camera}/image"
    if not node.subscribe(CameraInfo, info_topic, info_callback):
        raise RuntimeError("unable to subscribe to camera_info topic")
    if not node.subscribe(Image, image_topic, callback):
        raise RuntimeError("unable to subscribe to image topic")
    if not info_ready.wait(10):
        raise RuntimeError("CameraInfo was not received")
    time.sleep(args.seconds)
    # Detach the callback before interpreter teardown; gz-transport otherwise
    # can destroy pybind objects from a worker thread and abort on GIL checks.
    node.unsubscribe(info_topic); node.unsubscribe(image_topic)
    time.sleep(0.1)
    with lock:
        elapsed = max(time.monotonic() - start, 1e-9)
        lat = sorted(stats["latencies_ms"])
        p95 = lat[min(len(lat) - 1, int(len(lat) * .95))] if lat else None
        print(json.dumps({**stats, "wall_seconds": elapsed, "wall_fps": stats["frames"] / elapsed,
            "detector_p95_ms": p95, "backend": detector.backend,
            "ground_truth_used_by_runtime": False, "hardware_verified": False}, sort_keys=True), flush=True)
    # gz-transport 13 can still tear down a worker-owned pybind object after
    # unsubscribe. Exit after the flushed report to keep this diagnostic's
    # result usable; the Gazebo shutdown issue remains documented separately.
    os._exit(0)


if __name__ == "__main__":
    main()
