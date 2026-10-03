#!/usr/bin/env python3
"""Mutual poses of all cameras from a calibration session's strip snapshots, offline.

    .venv/bin/python scripts/solve_extrinsics.py artifacts/calibration/<session>

Re-detects the strip in ``<session>/extrinsics/placement_*/camera_N.png``
(placements and their measured / free status from ``manifest.json``) and
solves all camera poses together.  K, D per camera: this session's
``intrinsics/<camera>/<camera>_1640x1232.json``, else ``--intrinsics``, else
the base config (which is NOT a lens calibration -- a warning is printed).
Writes ``<session>/extrinsics/runtime_cameras.json`` and
``calibration_report.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localization_contracts import charuco_calibration as cc  # noqa: E402

DEFAULT_BASE = ROOT / "config" / "mocap_arena_imx219" / "runtime_cameras.json"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("--base-config", type=Path, default=DEFAULT_BASE)
    p.add_argument("--intrinsics", type=Path,
                   help="directory of <camera>_1640x1232.json, or a runtime_cameras.json")
    p.add_argument("--truth", type=Path, help="reference config to compare against (simulation)")
    args = p.parse_args(argv)
    base = json.loads(args.base_config.read_text())
    intrinsics, sources = {}, {}
    for cam in base["cameras"]:
        name = cam["name"]
        found = cc.load_intrinsics(args.session / "intrinsics", name)
        source = "session"
        if found is None and args.intrinsics:
            found, source = cc.load_intrinsics(args.intrinsics, name), str(args.intrinsics)
        if found is None:
            found = (np.asarray(cam["K"], float).reshape(3, 3), np.asarray(cam["D"], float))
            source = "base config (NOT a lens calibration)"
            print(f"warning: {name}: no intrinsics, using the base config's K, D", file=sys.stderr)
        intrinsics[name], sources[name] = found, source
    truth = json.loads(args.truth.read_text()) if args.truth else None
    sol = cc.solve_session(args.session, base, cc.strip_target(), intrinsics, sources, truth)
    print(f"{sol['observations']} views, {sol['corners']} corners, reprojection RMS {sol['rms_px']:.3f} px, "
          f"P95 {sol['p95_px']:.3f} px")
    print(f"{'camera':>9} {'X':>7} {'Y':>7} {'Z':>7} {'tilt':>6} {'head':>7} {'corners':>7} "
          f"{'P95px':>6} {'sigma_mm':>8} {'spread':>7}")
    for name, c in sol["cameras"].items():
        x, y, z = c["position_world"]
        sigma = max(c["sigma_position_mm"]) if c["sigma_position_mm"] else float("nan")
        spread = max(c["single_view_spread_mm"] or [0])
        print(f"{name:>9} {x:7.3f} {y:7.3f} {z:7.3f} {c['tilt_from_nadir_deg']:6.1f} "
              f"{c['image_x_heading_deg']:7.1f} {c['corners']:7d} {c['reproj_p95_px']:6.2f} "
              f"{sigma:8.1f} {spread:7.0f}")
    for name, why in sol["unsolved"].items():
        print(f"{name:>9} NOT SOLVED: {why}")
    for p in sol["placements"]:
        if "x0" in p:
            print(f"placement {p['index']}: x0 {p['x0']:.3f} y0 {p['y0']:.3f} yaw {p['yaw_deg']:.2f} deg"
                  + (" (measured)" if p["known"] else " (solved)"))
    for name, e in (sol.get("vs_truth") or {}).items():
        print(f"{name}: {e['position_error_mm']:.1f} mm, {e['rotation_error_deg']:.3f} deg against truth")
    for path in sol["files"]:
        print(path)


if __name__ == "__main__":
    main()
