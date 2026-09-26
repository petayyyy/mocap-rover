#!/usr/bin/env python3
"""Raise the world's speed and sampling limits for a high-speed run.

The shipped world cannot reach 40 km/h and cannot measure it if it could:

- DiffDrive caps the rover at ``max_linear_velocity = 1`` m/s, so 11.11 m/s
  is unreachable rather than untested.
- The cameras run at 15 Hz, which at 11.11 m/s is 0.74 m between frames.
- PosePublisher runs at 30 Hz, so ground truth arrives every 0.37 m and the
  error in measuring the error would swamp the error itself.

Each of those is a property of the world, not of the localization code, so
they are changed here, in one reproducible place, instead of by hand.

What this cannot change: Gazebo renders a perfectly sharp frame. At 11.11 m/s
a 1 ms exposure smears the marker 11 mm in the world, and 5 ms smears it
45 mm; on a 3 m slant at f = 245 that is 0.9 px and 4.5 px. Detection recall
measured in this world is therefore optimistic at speed.
"""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def set_text(parent, tag, value):
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    before = (node.text or "").strip()
    node.text = str(value)
    return None if before == str(value) else f"{tag}: {before or 'unset'} -> {value}"


def patch(path: Path, *, camera_hz, truth_hz, max_linear, max_angular,
          lidar_hz, step_size):
    tree = ET.parse(path)
    root = tree.getroot()
    world = root.find("world")
    changes = []

    if step_size is not None:
        physics = world.find("physics")
        if physics is not None:
            note = set_text(physics, "max_step_size", step_size)
            if note:
                changes.append(f"physics {note}")

    for sensor in root.iter("sensor"):
        kind = sensor.get("type")
        if kind == "camera" and camera_hz is not None:
            note = set_text(sensor, "update_rate", camera_hz)
            if note:
                changes.append(f"{sensor.get('name')} {note}")
        elif kind == "gpu_lidar" and lidar_hz is not None:
            note = set_text(sensor, "update_rate", lidar_hz)
            if note:
                changes.append(f"{sensor.get('name')} {note}")

    for plugin in root.iter("plugin"):
        name = plugin.get("name") or ""
        if name.endswith("DiffDrive"):
            owner = "?"
            for model in world.findall("model"):
                if plugin in list(model.iter("plugin")):
                    owner = model.get("name")
                    break
            for tag, value in (("max_linear_velocity", max_linear),
                               ("min_linear_velocity", None if max_linear is None else -max_linear),
                               ("max_angular_velocity", max_angular),
                               ("min_angular_velocity", None if max_angular is None else -max_angular)):
                if value is None:
                    continue
                note = set_text(plugin, tag, value)
                if note:
                    changes.append(f"{owner} {note}")
        elif name.endswith("PosePublisher") and truth_hz is not None:
            note = set_text(plugin, "update_frequency", truth_hz)
            if note:
                changes.append(f"PosePublisher {note}")
    return tree, changes


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("world", type=Path)
    parser.add_argument("--output", type=Path,
                        help="Write here instead of editing in place")
    parser.add_argument("--camera-hz", type=float, default=90.0)
    parser.add_argument("--truth-hz", type=float, default=200.0)
    parser.add_argument("--lidar-hz", type=float, default=None,
                        help="Leave unset to keep the L2's own 5.55 Hz")
    parser.add_argument("--max-linear", type=float, default=12.0,
                        help="m/s; 11.11 is 40 km/h")
    parser.add_argument("--max-angular", type=float, default=3.0)
    parser.add_argument("--step-size", type=float, default=None,
                        help="Physics step; 0.001 leaves 11 mm of travel per "
                             "step at 11.11 m/s")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tree, changes = patch(
        args.world, camera_hz=args.camera_hz, truth_hz=args.truth_hz,
        max_linear=args.max_linear, max_angular=args.max_angular,
        lidar_hz=args.lidar_hz, step_size=args.step_size)
    for line in changes:
        print(line)
    if not changes:
        print("no change needed")
    destination = args.output or args.world
    if not args.dry_run:
        ET.indent(tree)
        tree.write(destination, encoding="utf-8", xml_declaration=True)
        print(f"wrote {destination}")

    if args.camera_hz and args.max_linear:
        travel = args.max_linear / args.camera_hz
        print(f"\nat {args.max_linear:.2f} m/s ({args.max_linear*3.6:.1f} km/h):")
        print(f"  {travel*1000:.0f} mm of travel between camera frames")
        print(f"  {args.max_linear/args.truth_hz*1000:.0f} mm between truth samples")
        cameras = sum(1 for s in ET.parse(args.world).getroot().iter("sensor")
                      if s.get("type") == "camera")
        print(f"  {cameras * args.camera_hz:.0f} rendered frames/s; expect the "
              f"real-time factor to fall well below 1, which hands the runtime "
              f"extra wall time per simulated second and flatters latency -- "
              f"report RTF alongside any timing number from this world")
    return 0


if __name__ == "__main__":
    sys.exit(main())
