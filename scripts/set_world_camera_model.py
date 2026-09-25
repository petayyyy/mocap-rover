#!/usr/bin/env python3
"""Set the projection model of every camera sensor in an SDF world.

The IMX219 plumb_bob coefficients are only a valid fit near the image centre.
Rendering with them and then handing the same coefficients to solvePnP made
57% of every frame unusable (see docs/hybrid_localization_answer.md).  For
simulation the honest choices are:

``pinhole``
    Render an ideal gnomonical camera at the calibrated focal length.  The
    runtime then solves with zero distortion and the model is exact over the
    whole frame.  This is the default: it removes the defect instead of
    modelling it, which is what a simulator should do until the wide-angle
    model itself is validated on hardware.

``equidistant``
    Render a fisheye lens.  Use this only together with a runtime calibration
    that also uses the fisheye model; the plumb_bob path cannot represent it.

The real lens distortion belongs in an offline replay test over recorded
hardware frames, not in the loop that also defines ground truth.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def patch(path: Path, model: str, dry_run=False):
    tree = ET.parse(path)
    root = tree.getroot()
    changed = []
    for sensor in root.iter("sensor"):
        if sensor.get("type") != "camera":
            continue
        camera = sensor.find("camera")
        if camera is None:
            continue
        name = sensor.get("name", "?")
        distortion = camera.find("distortion")
        if model == "pinhole":
            if distortion is not None:
                camera.remove(distortion)
                changed.append(f"{name}: removed <distortion>")
            lens = camera.find("lens")
            if lens is not None:
                kind = lens.find("type")
                if kind is None:
                    kind = ET.SubElement(lens, "type")
                if kind.text != "gnomonical":
                    kind.text = "gnomonical"
                    changed.append(f"{name}: lens type -> gnomonical")
                # scale_to_hfov rescales the rendered image and silently
                # changes the effective focal length away from <intrinsics>.
                scale = lens.find("scale_to_hfov")
                if scale is not None and scale.text != "false":
                    scale.text = "false"
                    changed.append(f"{name}: scale_to_hfov -> false")
        elif model == "equidistant":
            lens = camera.find("lens")
            if lens is None:
                lens = ET.SubElement(camera, "lens")
            kind = lens.find("type")
            if kind is None:
                kind = ET.SubElement(lens, "type")
            kind.text = "equidistant"
            changed.append(f"{name}: lens type -> equidistant")
            if distortion is not None:
                camera.remove(distortion)
                changed.append(f"{name}: removed <distortion>")
        else:
            raise SystemExit(f"unknown model {model!r}")
    if changed and not dry_run:
        ET.indent(tree)
        tree.write(path, encoding="utf-8", xml_declaration=True)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("world", type=Path)
    parser.add_argument("--model", choices=("pinhole", "equidistant"), default="pinhole")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    changes = patch(args.world, args.model, args.dry_run)
    if not changes:
        print("no change needed")
        return 0
    for line in changes:
        print(line)
    print(f"{'would write' if args.dry_run else 'wrote'} {args.world}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
