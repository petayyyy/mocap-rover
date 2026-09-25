#!/usr/bin/env python3
"""Capture rendered Gazebo RGB with evaluator-only projected cuboid labels."""
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_v_pb2 import Pose_V
from google.protobuf.text_format import MessageToString

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization_contracts.capture import LatestFrames, rgb_array
from localization_contracts.gazebo_services import ServiceClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=120)
    p.add_argument("--seed", type=int, default=73)
    a = p.parse_args()
    cv2.setNumThreads(1)
    # This file is ONLY a label generator. Camera truth is forbidden to runtime.
    cfg = json.loads(
        (Path(a.scenario) / "config/cameras_ground_truth.json").read_text()
    )
    cams = {c["name"]: c for c in cfg["cameras"]}
    root = Path(a.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    services = ServiceClient()
    node = Node()
    latest = LatestFrames(cams)
    topics = []
    rng = np.random.default_rng(a.seed)
    records = []
    for cid in cams:
        topic = f"/cameras/{cid}/image"
        topics.append(topic)
        assert node.subscribe(Image, topic, lambda msg, cid=cid: latest.push(cid, msg))
    try:
        for sample in range(a.samples):
            split = (
                "train"
                if sample < a.samples * 0.7
                else "val" if sample < a.samples * 0.85 else "test"
            )
            poses = []
            command = Pose_V()
            for name in ("tag_rover", "yolo_rover"):
                x, y = rng.uniform(0.7, 11.3, 2)
                yaw = rng.uniform(-math.pi, math.pi)
                q = command.pose.add(name=name)
                q.position.x = x
                q.position.y = y
                q.position.z = 0.14
                q.orientation.z = math.sin(yaw / 2)
                q.orientation.w = math.cos(yaw / 2)
                poses.append((x, y, yaw))
            if not services.request("set_pose_vector", command):
                raise RuntimeError("set_pose_vector")
            time.sleep(0.25)
            for cid, c in cams.items():
                latest.take(cid)
                item = latest.take(cid, 2)
                if item is None:
                    raise RuntimeError("no image " + cid)
                image = rgb_array(item[0])
                h, w = image.shape[:2]
                labels = []
                for klass, (x, y, yaw) in enumerate(poses):
                    # Full vehicle cuboid, including wheels. Label geometry only.
                    half_x = 0.5 if klass else 0.39
                    zmax = 0.483 if klass else 0.366
                    pts = np.array(
                        [
                            [xx, yy, z]
                            for xx in (-half_x, half_x)
                            for yy in (-0.36, 0.36)
                            for z in (0, zmax)
                        ]
                    )
                    R = np.array(
                        [
                            [math.cos(yaw), -math.sin(yaw), 0],
                            [math.sin(yaw), math.cos(yaw), 0],
                            [0, 0, 1],
                        ]
                    )
                    world = pts @ R.T + np.array([x, y, 0])
                    cp = (world - np.array(c["position_world"])) @ np.array(
                        c["R_world_optical"]
                    )
                    uv = cp @ np.array(c["K"]).reshape(3, 3).T
                    uv = uv[:, :2] / uv[:, 2:]
                    lo = uv.min(axis=0)
                    hi = uv.max(axis=0)
                    unclipped = np.prod(hi - lo)
                    lo = np.maximum(lo, [0, 0])
                    hi = np.minimum(hi, [w, h])
                    size = hi - lo
                    if min(size) < 8 or np.prod(size) / max(unclipped, 1) < 0.35:
                        continue
                    center = (lo + hi) / 2
                    labels.append(
                        f"{klass} {center[0]/w:.7f} {center[1]/h:.7f} {size[0]/w:.7f} {size[1]/h:.7f}"
                    )
                stem = f"{a.seed}_{sample:04d}_{cid}"
                imdir = root / "images" / split
                labdir = root / "labels" / split
                imdir.mkdir(parents=True, exist_ok=True)
                labdir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(imdir / (stem + ".jpg")),
                    cv2.cvtColor(cv2.resize(image, (720, 540)), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                (labdir / (stem + ".txt")).write_text("\n".join(labels) + "\n")
                records.append(
                    {
                        "image": f"images/{split}/{stem}.jpg",
                        "split": split,
                        "scene": sample,
                        "camera": cid,
                        "seed": a.seed,
                        "labels": len(labels),
                        "poses_label_only": poses,
                    }
                )
            if sample % 10 == 0:
                print(sample, len(records), flush=True)
        (root / "data.yaml").write_text(
            f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: tag_rover\n  1: opponent\n"
        )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "renderer": "Gazebo Harmonic",
                    "truth_role": "labels_only",
                    "split_unit": "scene_all_cameras",
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        latest.close()
        services.close()
        for topic in topics:
            node.unsubscribe(topic)
    print("saved", root, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
