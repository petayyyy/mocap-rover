#!/usr/bin/env python3
"""Print the acceptance numbers for one tag-coverage run.

Reads only the CSV the run produced, so the same command works on an old run
and on a new one and the two can be put side by side.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path


def pct(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return float("nan")
    index = (len(values) - 1) * p / 100
    low = int(index)
    high = min(low + 1, len(values) - 1)
    alpha = index - low
    return values[low] * (1 - alpha) + values[high] * alpha


def num(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def report(run_dir):
    run_dir = Path(run_dir)
    out = {"run": str(run_dir)}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        out["summary"] = json.loads(summary_path.read_text())

    frames = rows(run_dir / "camera_frames.csv")
    by_stamp = collections.defaultdict(lambda: [False, False])
    for row in frames:
        state = by_stamp[row["capture_ns"]]
        state[0] |= int(row["detections"]) > 0
        state[1] |= int(row["pnp_valid"]) > 0
    moments = max(len(by_stamp), 1)
    out["moments"] = len(by_stamp)
    out["detect_any_pct"] = 100 * sum(v[0] for v in by_stamp.values()) / moments
    out["pnp_any_pct"] = 100 * sum(v[1] for v in by_stamp.values()) / moments
    latency = [num(r, "latency_ms") for r in frames]
    out["frame_latency_p50_ms"] = pct(latency, 50)
    out["frame_latency_p95_ms"] = pct(latency, 95)
    out["frame_modes"] = dict(collections.Counter(r.get("mode", "") for r in frames))

    observations = rows(run_dir / "observations.csv")
    base_z = [num(r, "pnp_base_z_m") for r in observations]
    base_z = [v for v in base_z if v is not None]
    if base_z:
        nominal = 0.14
        out["obs_with_base_z"] = len(base_z)
        out["base_z_p50_m"] = pct(base_z, 50)
        out["base_z_bad_pct"] = 100 * sum(abs(v - nominal) > 0.30 for v in base_z) / len(base_z)
    out["obs_error_by_camera"] = {}
    per_camera = collections.defaultdict(list)
    for row in observations:
        error = num(row, "xy_error_m")
        if error is not None:
            per_camera[row["camera_id"]].append(error)
    for camera, values in sorted(per_camera.items()):
        out["obs_error_by_camera"][camera] = {
            "n": len(values), "p50": pct(values, 50), "p95": pct(values, 95)}
    all_errors = [e for values in per_camera.values() for e in values]
    out["obs_error_p50_m"] = pct(all_errors, 50)
    out["obs_error_p95_m"] = pct(all_errors, 95)

    estimates = rows(run_dir / "estimates.csv")
    out["estimate_samples"] = len(estimates)
    out["valid_samples"] = sum(r["valid"] == "1" for r in estimates)
    out["valid_coverage"] = out["valid_samples"] / max(len(estimates), 1)
    out["tracking_states"] = dict(collections.Counter(r["tracking_state"] for r in estimates))
    ages = [num(r, "measurement_age_ms") for r in estimates if r["valid"] == "1"]
    out["age_p50_ms"] = pct(ages, 50)
    out["age_p95_ms"] = pct(ages, 95)
    errors = [num(r, "xy_error_m") for r in estimates if r["valid"] == "1"]
    errors = [v for v in errors if v is not None]
    out["fused_p50_m"] = pct(errors, 50)
    out["fused_p95_m"] = pct(errors, 95)
    yaw = [abs(num(r, "yaw_error_deg")) for r in estimates
           if r["valid"] == "1" and num(r, "yaw_error_deg") is not None]
    out["fused_yaw_p95_deg"] = pct(yaw, 95)

    lidar = [json.loads(line) for line in
             (run_dir / "runtime" / "lidar.jsonl").read_text().splitlines() if line] \
        if (run_dir / "runtime" / "lidar.jsonl").exists() else []
    if lidar:
        out["lidar_scans"] = len(lidar)
        out["lidar_reasons"] = dict(collections.Counter(r.get("reason") for r in lidar))
        hits = [r for r in lidar if r.get("reason") == "accepted"]
        out["lidar_accepted"] = len(hits)
        if hits:
            out["lidar_points_p50"] = pct([r["points"] for r in hits], 50)
            out["lidar_sigma_p50_m"] = pct([r["sigma_m"] for r in hits], 50)
            out["lidar_residual_p50_m"] = pct([r["residual_m"] for r in hits], 50)
            out["lidar_residual_p95_m"] = pct([r["residual_m"] for r in hits], 95)
            out["lidar_processing_p95_ms"] = pct([r["processing_ms"] for r in hits], 95)

    gaps, current, previous = [], 0, None
    for row in estimates:
        stamp = int(row["stamp_ns"])
        if row["valid"] == "1":
            if current:
                gaps.append(current)
            current = 0
        elif previous is not None:
            current += stamp - previous
        previous = stamp
    if current:
        gaps.append(current)
    out["invalid_gaps"] = len(gaps)
    out["longest_gap_ms"] = max(gaps) / 1e6 if gaps else 0.0
    out["total_gap_s"] = sum(gaps) / 1e9
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="+")
    args = parser.parse_args()
    for run_dir in args.run_dir:
        print(json.dumps(report(run_dir), indent=2, default=float))


if __name__ == "__main__":
    main()
