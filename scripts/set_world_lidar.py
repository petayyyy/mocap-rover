#!/usr/bin/env python3
"""Rewrite a world's ceiling lidar to a named sensor profile.

The arena world was hand-built around a Unitree L2.  Swapping the physical
sensor changes the sweep rate, the ray pattern and the range, and every one
of those reaches the runtime: the sweep rate sets how often a track can be
corrected, the ray pattern sets how many returns land on a rover, and both
decide whether the recorded static background is still valid.

Profiles carry the numbers from the manufacturer's own user guide, so the
world cannot quietly disagree with the datasheet.  A world rewritten here
needs its lidar background re-recorded: the voxel map was fitted to the old
ray pattern and does not describe the new one.
"""
import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

# Each profile is the sensor's published specification, not a tuning knob.
PROFILES = {
    # Unitree L2, as the world originally modelled it.
    "l2": dict(
        model="arena_unitree_l2", link="l2_lidar_link", prefix="unitree_l2",
        rate_hz=5.55, h_samples=288, v_samples=64,
        v_min_deg=0.0, v_max_deg=90.0,
        range_min=0.05, range_max=30.0, range_resolution=0.0045,
        source="modelled from the Unitree L2 sweep rate",
    ),
    # RoboSense Airy, from the Airy User Guide specification table:
    # 96 channels, 0-360 deg H, 0-90 deg V, 0.4 deg H, 0.947 deg V,
    # 10 Hz frame rate (600 rpm), 856,320 pts/s, 60 m range (30 m @10%),
    # 0.1 m blind zone, 1.5 cm (1 sigma) accuracy.
    "airy": dict(
        model="arena_robosense_airy", link="airy_lidar_link", prefix="robosense_airy",
        rate_hz=10.0, h_samples=900, v_samples=95,
        v_min_deg=0.0, v_max_deg=90.0,
        range_min=0.1, range_max=60.0, range_resolution=0.015,
        source="RoboSense Airy User Guide, Specifications table",
    ),
}


def text(parent, tag, value):
    node = parent.find(tag)
    if node is None:
        raise ValueError(f"lidar sensor has no <{tag}>")
    node.text = str(value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("world", type=Path)
    p.add_argument("--profile", choices=sorted(PROFILES), required=True)
    p.add_argument("--output", type=Path,
                   help="write here instead of rewriting the world in place")
    p.add_argument("--rename", action="store_true",
                   help="also rename the model, link and topics to match the "
                        "profile; leaving them alone keeps existing configs and "
                        "recorded backgrounds pointing at the same topic")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    spec = PROFILES[a.profile]
    tree = ET.parse(a.world)
    world = tree.getroot().find("world")
    holder = None
    for model in world.findall("model"):
        if model.find(".//sensor[@type='gpu_lidar']") is not None:
            holder = model
            break
    if holder is None:
        raise SystemExit("world has no gpu_lidar to rewrite")
    sensor = holder.find(".//sensor[@type='gpu_lidar']")

    changes = []

    def note(what, old, new):
        if str(old) != str(new):
            changes.append(f"{what}: {old} -> {new}")

    note("update_rate", sensor.findtext("update_rate"), spec["rate_hz"])
    text(sensor, "update_rate", spec["rate_hz"])
    scan = sensor.find("ray/scan")
    h, v = scan.find("horizontal"), scan.find("vertical")
    note("horizontal samples", h.findtext("samples"), spec["h_samples"])
    text(h, "samples", spec["h_samples"])
    note("vertical samples", v.findtext("samples"), spec["v_samples"])
    text(v, "samples", spec["v_samples"])
    for node, tag, deg in ((v, "min_angle", spec["v_min_deg"]),
                           (v, "max_angle", spec["v_max_deg"])):
        note(f"vertical {tag}", node.findtext(tag), math.radians(deg))
        text(node, tag, math.radians(deg))
    rng = sensor.find("ray/range")
    for tag, value in (("min", spec["range_min"]), ("max", spec["range_max"]),
                       ("resolution", spec["range_resolution"])):
        note(f"range {tag}", rng.findtext(tag), value)
        text(rng, tag, value)

    if a.rename:
        old_prefix = sensor.findtext("topic").strip("/").split("/")[0]
        note("model", holder.get("name"), spec["model"])
        holder.set("name", spec["model"])
        link = holder.find("link")
        note("link", link.get("name"), spec["link"])
        link.set("name", spec["link"])
        for node in holder.iter():
            if node.tag == "topic" and node.text and old_prefix in node.text:
                new = node.text.replace(old_prefix, spec["prefix"])
                note("topic", node.text, new)
                node.text = new

    h_res = 360.0 / spec["h_samples"]
    v_res = (spec["v_max_deg"] - spec["v_min_deg"]) / spec["v_samples"]
    points = spec["h_samples"] * spec["v_samples"]
    print(f"profile {a.profile} ({spec['source']})")
    for line in changes:
        print("  " + line)
    if not changes:
        print("  world already matches this profile")
    print(f"  -> {h_res:.3f} deg horizontal, {v_res:.3f} deg vertical, "
          f"{points} rays/sweep, {points * spec['rate_hz']:,.0f} pts/s, "
          f"{1000 / spec['rate_hz']:.0f} ms between sweeps")
    if a.dry_run:
        return
    destination = a.output or a.world
    ET.indent(tree)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    print(f"wrote {destination}")
    print("re-record the lidar background: the voxel map was fitted to the "
          "previous ray pattern (scripts/record_lidar_background.py)")


if __name__ == "__main__":
    main()
