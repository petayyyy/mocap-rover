#!/usr/bin/env python3
"""Learn the empty arena's lidar background, once, into a voxel map.

The arena is not empty above the floor.  worlds/mocap_arena_l2.sdf puts
0.16 m x 0.4 m barriers along all four edges with tape strips at z = 0.407,
squarely inside the rover height band, and they sit exactly where the cameras
lose the marker.  At 5 to 7 m the clustering cell grows to half a metre and
the rover merges with them, so the lidar refuses every scan in the one place
it was added to help.

A height band cannot separate them because they occupy the same heights, and
an arena inset cannot either without also cutting the working area.  What
distinguishes them is that they never move, which is what this records.

Run it on an empty arena.  An occupied voxel is permanent, so anything parked
in the arena while this runs is invisible to the tracker afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from localization_contracts.lidar_pipeline import (
    ArenaLidar, StaticVoxelMap, parse_laser_scan,
)


def empty_world(source, destination):
    """Copy the world with every non-static model removed."""
    tree = ET.parse(source)
    world = tree.getroot().find("world")
    removed = []
    for model in list(world.findall("model")):
        if (model.findtext("static") or "false").strip().lower() != "true":
            world.remove(model)
            removed.append(model.get("name"))
    ET.indent(tree)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default=str(ROOT / "worlds/mocap_arena_l2.sdf"))
    parser.add_argument("--output", default=str(ROOT / "config/lidar_background.json"))
    parser.add_argument("--scans", type=int, default=40)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--min-fraction", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    from gz.msgs10.laserscan_pb2 import LaserScan
    from gz.transport13 import Node
    from run_tag_coverage_experiment import export_camera_config

    staging = Path(args.output).parent / "_empty_arena.sdf"
    staging.parent.mkdir(parents=True, exist_ok=True)
    removed = empty_world(args.world, staging)
    print(f"removed non-static models: {removed or 'none'}")
    config = export_camera_config(args.world, staging.with_suffix(".cameras.json"))
    lidar_config = config["lidar"]
    if lidar_config is None:
        raise SystemExit("the world declares no gpu_lidar")

    unit = ArenaLidar(lidar_config["position_world"], lidar_config["R_world_sensor"])
    clouds = []
    done = threading.Event()

    def on_scan(message):
        if done.is_set():
            return
        scan = unit.scan_to_arena(parse_laser_scan(message))
        # Store what the tracker would have to reject: everything already
        # inside the height band and the arena, minus nothing.
        band = ((scan.points[:, 2] >= unit.z_band[0])
                & (scan.points[:, 2] <= unit.z_band[1]))
        clouds.append(scan.points[band])
        print(f"  scan {len(clouds)}/{args.scans}: {int(band.sum())} points in band",
              flush=True)
        if len(clouds) >= args.scans:
            done.set()

    environment = dict(os.environ, MOCAP_WORLD=str(staging))
    gazebo = subprocess.Popen(
        [str(ROOT / "scripts/run.sh"), "-s", "--headless-rendering"],
        cwd=ROOT, env=environment, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        node = Node()
        if not node.subscribe(LaserScan, lidar_config["topic"], on_scan):
            raise SystemExit(f"cannot subscribe to {lidar_config['topic']}")
        if not done.wait(args.timeout):
            raise SystemExit(
                f"only {len(clouds)} scans in {args.timeout:.0f} s; is Gazebo running?")
    finally:
        try:
            os.killpg(gazebo.pid, signal.SIGINT)
            gazebo.wait(timeout=15)
        except Exception:
            os.killpg(gazebo.pid, signal.SIGKILL)

    background = StaticVoxelMap(args.voxel_size).fit(clouds, args.min_fraction)
    Path(args.output).write_text(json.dumps(background.to_dict()) + "\n")
    counts = [len(c) for c in clouds]
    print(f"scans {len(clouds)}, in-band points per scan "
          f"min {min(counts)} median {int(np.median(counts))} max {max(counts)}")
    print(f"voxels {len(background.voxels)} at {args.voxel_size} m -> {args.output}")
    staging.unlink(missing_ok=True)
    staging.with_suffix(".cameras.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
