#!/usr/bin/env python3
"""Fisheye intrinsics of one camera from saved 1640x1232 captures, three resolutions.

    .venv/bin/python scripts/calibrate_intrinsics.py \\
        artifacts/calibration/<session>/intrinsics/camera_1/captures --camera camera_1

Calibrates 1640x1232, 820x616 and 640x480 (the captures reduced the way the
camera node reduces its small stream) and writes next to the captures folder:
``imx219_<w>x<h>.yaml`` (ROS / sverk camera_calibration format),
``<camera>_<w>x<h>.json`` (K, D for runtime_cameras.json) and
``intrinsics_summary.json``.  The board flags are the station's.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from localization_contracts import charuco_calibration as cc  # noqa: E402
from calibration_station import add_board_args, hand_target  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("captures", type=Path, help="folder of full-resolution grey PNG/JPEG frames")
    p.add_argument("--camera", required=True, help="camera id written into the results")
    p.add_argument("--out", type=Path, help="default: the parent of the captures folder")
    p.add_argument("--sensor-name", default="imx219")
    add_board_args(p)
    args = p.parse_args(argv)
    paths = sorted(q for q in args.captures.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg"))
    images = []
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or (gray.shape[1], gray.shape[0]) != cc.SENSOR_SIZE:
            print(f"skip {path.name}: not a {cc.SENSOR_SIZE[0]}x{cc.SENSOR_SIZE[1]} image", file=sys.stderr)
            continue
        images.append(gray)
    if len(images) < 5:
        raise SystemExit(f"{len(images)} usable captures; need at least 5, {cc.RECOMMENDED_VIEWS} recommended")
    target = hand_target(args)
    results = cc.calibrate_intrinsics(images, target, max_frame_err=args.max_frame_err)
    out = args.out or args.captures.parent
    written = cc.write_intrinsics(out, args.camera, results, target, sensor_name=args.sensor_name)
    print(f"{target.describe()}, {len(images)} captures")
    print(f"{'size':>10} {'views':>7} {'rms':>7} {'mean':>7} {'fx':>8} {'fy':>8} {'cx':>8} {'cy':>8} {'HFOV':>6}")
    for r in results:
        size = f"{r['size'][0]}x{r['size'][1]}"
        if "K" not in r:
            print(f"{size:>10}  failed: {r.get('error')}")
            continue
        k = r["K"]
        print(f"{size:>10} {len(r['views_used']):>3}/{r['views_captured']:<3} {r['rms']:7.3f} "
              f"{r['reprojection_error']:7.3f} {k[0, 0]:8.2f} {k[1, 1]:8.2f} {k[0, 2]:8.2f} {k[1, 2]:8.2f} "
              f"{r['fov_deg']['horizontal']:6.1f}")
        print(f"{'':>10} D = {', '.join(f'{v:.5f}' for v in r['D'])}"
              + (f"; dropped {len(r['dropped_outliers'])} outliers, {len(r['dropped_illcond'])} ill-conditioned"
                 if r["dropped_outliers"] or r["dropped_illcond"] else ""))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
