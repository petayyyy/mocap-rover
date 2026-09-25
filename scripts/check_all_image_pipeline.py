#!/usr/bin/env python3
"""Measure bounded six-camera Gazebo image -> AprilTag -> PnP processing."""
import argparse, json, os, sys, threading, time
from pathlib import Path
import cv2
import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.msgs10.camera_info_pb2 import CameraInfo
from gz.transport13 import Node

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization_contracts.image_pipeline import MultiCameraImagePipeline
from localization_contracts.capture import LatestFrames, rgb_array


def main(seconds=5.0, config_path="config/cameras.json", output=None):
    cv2.setNumThreads(1)
    config = json.loads(Path(config_path).read_text())
    entries = {c["name"]: c for c in config["cameras"]}
    infos = {}
    ready = threading.Event()
    lock = threading.Lock()
    stop = threading.Event()
    stats = {
        cid: {
            "frames": 0,
            "detections": 0,
            "accepted": 0,
            "latency_ms": [],
            "queue_ms": [],
            "stamps": [],
        }
        for cid in entries
    }
    latest = LatestFrames(entries)
    node = Node()

    def info_cb(msg, cid):
        with lock:
            infos[cid] = (list(msg.intrinsics.k), list(msg.distortion.k))
            if len(infos) == len(entries):
                ready.set()

    topics = []
    for cid in entries:
        it = f"/cameras/{cid}/camera_info"
        im = f"/cameras/{cid}/image"
        topics += [it, im]
        if not node.subscribe(CameraInfo, it, lambda msg, cid=cid: info_cb(msg, cid)):
            raise RuntimeError(it)
        if not node.subscribe(Image, im, lambda msg, cid=cid: latest.push(cid, msg)):
            raise RuntimeError(im)
    if not ready.wait(15):
        raise RuntimeError(f"CameraInfo missing: {sorted(set(entries)-set(infos))}")
    specs = {
        cid: {
            "K": np.asarray(infos[cid][0]).reshape(3, 3),
            "D": infos[cid][1],
            "R_world_optical": c["R_world_optical"],
            "position_world": c["position_world"],
        }
        for cid, c in entries.items()
    }
    tags = {
        int(t["id"]): {
            "rotation": t["R_base_tag"],
            "translation": t["T_base_tag_translation"],
        }
        for t in config.get("tags", [dict(config["tag"], id=0)])
    }
    tag_entries = config.get("tags", [dict(config["tag"], id=0)])
    family = tag_entries[0]["family"]
    pipeline = MultiCameraImagePipeline(
        specs, tags, "gazebo-camera-info", detector_scale=1.0,
        family=family, detector_profile="coverage", marker_ids=tuple(tags)
    )
    errors = []

    def worker(cid):
        try:
            while not stop.is_set():
                item = latest.take(cid)
                if item is None:
                    continue
                msg, received, seq = item
                t0 = time.monotonic_ns()
                stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nsec
                p = pipeline.cameras[cid]
                before = p.accepted
                detected_before = p.detections
                pipeline.process(
                    cid, rgb_array(msg), stamp, receive_time_ns=received, frame_seq=seq
                )
                s = stats[cid]
                s["frames"] += 1
                s["accepted"] += p.accepted - before
                s["detections"] += p.detections - detected_before
                s["latency_ms"].append((time.monotonic_ns() - t0) / 1e6)
                s["queue_ms"].append((t0 - received) / 1e6)
                s["stamps"].append(stamp)
        except Exception as exc:
            errors.append(f"{cid}: {exc}")
            stop.set()

    threads = [threading.Thread(target=worker, args=(cid,)) for cid in entries]
    start = time.monotonic()
    for thread in threads:
        thread.start()
    stop.wait(seconds)
    stop.set()
    latest.close()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - start
    for topic in topics:
        node.unsubscribe(topic)
    out = {}
    for cid, s in stats.items():
        stamps = s.pop("stamps")
        sim_seconds = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else 0
        latency = s.pop("latency_ms")
        queue = s.pop("queue_ms")
        out[cid] = {
            **s,
            "received": latest.received[cid],
            "overwritten": latest.dropped[cid],
            "wall_fps": s["frames"] / elapsed,
            "accepted_hz": s["accepted"] / elapsed,
            "sim_fps": (s["frames"] - 1) / sim_seconds if sim_seconds else 0,
            "rtf": sim_seconds / elapsed,
            "latency_ms_p95": float(np.percentile(latency, 95)) if latency else None,
            "queue_ms_p95": float(np.percentile(queue, 95)) if queue else None,
        }
    report = {
        "cameras": out,
        "wall_seconds": elapsed,
        "errors": errors,
        "ground_truth_used_by_runtime": False,
        "hardware_verified": False,
    }
    encoded = json.dumps(report, indent=2)
    if output:
        Path(output).write_text(encoded + "\n")
    print(encoded, flush=True)
    # transport13 Python destructor has a known GIL crash after unsubscribe.
    # All workers and report writes have completed before bypassing destruction.
    os._exit(1 if errors else 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--config", default="config/cameras.json")
    p.add_argument("--output")
    a = p.parse_args()
    main(a.seconds, a.config, a.output)
