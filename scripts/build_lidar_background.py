#!/usr/bin/env python3
"""Build the lidar background of the empty arena from recorded scans.

The same job as ``scripts/record_lidar_background.py``, but from the npz
scans of a ``record_dataset.py`` directory instead of a live Gazebo, so it
needs neither Gazebo nor ROS and gives the same result every time.

Two backgrounds come out of one pass:

* ``RangeBackground`` (``--output``, npz): range and spread per ray, with the
  floor plane fitted by ``fit_floor`` stored alongside.  This is the one the
  replay uses with ``--lidar-range-background``.
* ``StaticVoxelMap`` (``--voxel-output``, json), the older occupancy map, for
  comparison and for ``--lidar-background``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localization_contracts import lidar_pipeline  # noqa: E402


def load_scans(dataset):
    rows = [json.loads(line) for line in (dataset / "lidar.jsonl").read_text().splitlines()
            if line.strip()]
    scans = []
    for row in rows:
        with np.load(dataset / row["file"]) as z:
            scans.append({"ranges": z["ranges"].astype(float),
                          "azimuth": z["azimuth"].astype(float),
                          "elevation": z["elevation"].astype(float),
                          "range_min": float(z["range_min"]),
                          "range_max": float(z["range_max"]),
                          "stamp_ns": int(row["stamp_ns"])})
    if not scans:
        raise SystemExit(f"no scans in {dataset}")
    return scans


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="empty-arena dataset directory")
    p.add_argument("--output", type=Path,
                   help="RangeBackground npz; default <dataset>/lidar_background.npz")
    p.add_argument("--voxel-output", type=Path, help="also write a StaticVoxelMap json here")
    p.add_argument("--percentile", type=float, default=80.0)
    p.add_argument("--min-hits", type=int, default=None,
                   help="default: half the scans")
    p.add_argument("--min-margin", type=float, default=0.12)
    p.add_argument("--sigma-factor", type=float, default=4.0)
    p.add_argument("--voxel-size", type=float, default=0.10)
    p.add_argument("--min-fraction", type=float, default=0.5)
    a = p.parse_args(argv)
    config = json.loads((a.dataset / "runtime_cameras.json").read_text())["lidar"]
    scans = load_scans(a.dataset)

    background = lidar_pipeline.RangeBackground.build(
        [scan["ranges"] for scan in scans], percentile=a.percentile, min_hits=a.min_hits,
        min_margin_m=a.min_margin, sigma_factor=a.sigma_factor)
    rng = np.random.default_rng(1)
    chunks = []
    for scan in scans:
        points = lidar_pipeline.sensor_level_points(scan, config["R_world_sensor"])
        if len(points) > 4000:
            points = points[rng.choice(len(points), 4000, replace=False)]
        chunks.append(points)
    floor = lidar_pipeline.fit_floor(np.concatenate(chunks),
                                     expected_height=float(config["position_world"][2]))
    background.floor = floor
    output = a.output or (a.dataset / "lidar_background.npz")
    background.save(output)
    report = {"scans": len(scans), "range_background": str(output),
              **background.summary(),
              "floor": {"height_m": floor.height_m, "tilt_deg": floor.tilt_deg,
                        "inlier_fraction": floor.inlier_fraction,
                        "residual_p95_m": floor.residual_p95_m, "span_m": list(floor.span_m),
                        "normal": list(floor.normal),
                        "expected_height_m": float(config["position_world"][2])}}

    if a.voxel_output:
        unit = lidar_pipeline.ArenaLidar(config["position_world"], config["R_world_sensor"])
        clouds = []
        for scan in scans:
            points = unit.scan_to_arena(scan).points
            band = (points[:, 2] >= unit.z_band[0]) & (points[:, 2] <= unit.z_band[1])
            clouds.append(points[band])
        voxels = lidar_pipeline.StaticVoxelMap(a.voxel_size).fit(clouds, a.min_fraction)
        a.voxel_output.write_text(json.dumps(voxels.to_dict()) + "\n")
        report["voxel_background"] = {"path": str(a.voxel_output), "voxels": len(voxels.voxels),
                                      "voxel_size_m": a.voxel_size}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
