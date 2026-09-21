#!/usr/bin/env python3
"""Run the real simulation image -> detector -> PnP path on Gazebo topics.

This intentionally does not subscribe to world pose topics and does not use
ground truth.  Camera transforms and K/D are loaded from the calibration
snapshot, so the result is a runtime image-path smoke check, not acceptance.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from gz.msgs10.image_pb2 import Image
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
    K = np.asarray(camera["K"], dtype=float).reshape(3, 3)
    observer = PnpAprilTagObserver(
        TagConfig(calibration_version="config-cameras-json"), K, camera["D"],
        {"rotation": camera["R_world_optical"], "translation": camera["position_world"]},
        {"rotation": config["tag"]["R_base_tag"], "translation": config["tag"]["T_base_tag_translation"]},
    )
    detector = AprilTagImageDetector()
    node = Node(); done = threading.Event(); lock = threading.Lock()
    stats = {"frames": 0, "detections": 0, "accepted": 0, "latencies_ms": [], "ids": {}}
    start = time.monotonic()

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
                if observer.observe(detection) is not None:
                    stats["accepted"] += 1
            stats["latencies_ms"].append((time.monotonic_ns() - now) / 1e6)

    if not node.subscribe(Image, f"/cameras/{args.camera}/image", callback):
        raise RuntimeError("unable to subscribe to image topic")
    time.sleep(args.seconds)
    # Detach the callback before interpreter teardown; gz-transport otherwise
    # can destroy pybind objects from a worker thread and abort on GIL checks.
    node.unsubscribe(f"/cameras/{args.camera}/image")
    time.sleep(0.1)
    with lock:
        elapsed = max(time.monotonic() - start, 1e-9)
        lat = sorted(stats["latencies_ms"])
        p95 = lat[min(len(lat) - 1, int(len(lat) * .95))] if lat else None
        print(json.dumps({**stats, "wall_seconds": elapsed, "wall_fps": stats["frames"] / elapsed,
            "detector_p95_ms": p95, "backend": detector.backend,
            "ground_truth_used_by_runtime": False, "hardware_verified": False}, sort_keys=True))


if __name__ == "__main__":
    main()
