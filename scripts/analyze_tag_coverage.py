#!/usr/bin/env python3
"""Convert one tag-coverage run into analysis-friendly CSV files."""

import argparse
import bisect
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def angle_error(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * p / 100
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    alpha = index - lo
    return values[lo] * (1 - alpha) + values[hi] * alpha


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_truth(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = {
        "stamp_ns": int,
        "wall_ns": int,
        "x": float,
        "y": float,
        "z": float,
        "yaw": float,
        "vx": float,
        "vy": float,
        "waypoint_index": int,
        "target_x": float,
        "target_y": float,
        "cmd_linear": float,
        "cmd_angular": float,
    }
    for row in rows:
        for key, convert in numeric.items():
            row[key] = convert(row[key])
    rows.sort(key=lambda row: row["stamp_ns"])
    return rows


def truth_at(rows, stamps, stamp_ns, max_gap_ns=250_000_000):
    index = bisect.bisect_left(stamps, stamp_ns)
    if index == 0 or index == len(rows):
        return None
    before, after = rows[index - 1], rows[index]
    span = after["stamp_ns"] - before["stamp_ns"]
    if span <= 0 or span > max_gap_ns:
        return None
    alpha = (stamp_ns - before["stamp_ns"]) / span
    yaw_delta = angle_error(after["yaw"], before["yaw"])
    return {
        "truth_x": before["x"] * (1 - alpha) + after["x"] * alpha,
        "truth_y": before["y"] * (1 - alpha) + after["y"] * alpha,
        "truth_z": before["z"] * (1 - alpha) + after["z"] * alpha,
        "truth_yaw": before["yaw"] + yaw_delta * alpha,
        "truth_vx": before["vx"] * (1 - alpha) + after["vx"] * alpha,
        "truth_vy": before["vy"] * (1 - alpha) + after["vy"] * alpha,
        "waypoint_index": before["waypoint_index"],
    }


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(run_dir):
    run_dir = Path(run_dir)
    runtime = run_dir / "runtime"
    truth = load_truth(run_dir / "truth.csv")
    if len(truth) < 2:
        raise ValueError("truth.csv has too few samples")
    truth_stamps = [row["stamp_ns"] for row in truth]

    observation_rows = []
    observation_by_frame = {}
    accepted_by_frame = set()
    for wrapper in load_jsonl(runtime / "observations.jsonl"):
        obs = wrapper["observation"]
        target = truth_at(truth, truth_stamps, obs["capture_time_ns"])
        px = obs.get("pixel_features", {})
        row = {
            "camera_id": obs["camera_id"],
            "frame_seq": obs["frame_seq"],
            "capture_ns": obs["capture_time_ns"],
            "accepted": int(wrapper["accepted"]),
            "selection_reason": wrapper.get("selection_reason"),
            "marker_id": obs.get("marker_id"),
            "estimate_x": obs["position_m"][0],
            "estimate_y": obs["position_m"][1],
            "estimate_z": obs["position_m"][2],
            "estimate_yaw": px.get("yaw_rad"),
            "quality": obs["quality"],
            "reprojection_px": px.get("reprojection_error_px"),
            "tag_side_px": px.get("side_px"),
            "planar_tilt_deg": px.get("planar_tilt_deg"),
            "xy_sigma_m": px.get("xy_sigma_m"),
            "processing_ms": (obs["processed_time_ns"] - obs["receive_time_ns"]) / 1e6,
        }
        if target:
            row.update(target)
            row["xy_error_m"] = math.hypot(
                row["estimate_x"] - target["truth_x"],
                row["estimate_y"] - target["truth_y"],
            )
            row["z_error_m"] = row["estimate_z"] - target["truth_z"]
            row["yaw_error_deg"] = math.degrees(
                angle_error(row["estimate_yaw"], target["truth_yaw"])
            )
        observation_rows.append(row)
        observation_by_frame[(row["camera_id"], row["capture_ns"])] = row
        if row["accepted"]:
            accepted_by_frame.add((row["camera_id"], row["capture_ns"]))

    frame_rows = []
    bins = defaultdict(lambda: {"frames": 0, "detected": 0, "pnp": 0, "fusion": 0})
    camera_values = defaultdict(lambda: {"frames": 0, "detected": 0, "pnp": 0, "fusion": 0, "latency": [], "reprojection": [], "xy_error": []})
    for frame in load_jsonl(runtime / "camera_frames.jsonl"):
        target = truth_at(truth, truth_stamps, frame["capture_ns"])
        row = dict(frame)
        row["fusion_accepted"] = int(
            (frame["camera_id"], frame["capture_ns"]) in accepted_by_frame
        )
        row["tag_ids"] = ";".join(map(str, frame.get("tag_ids", [])))
        row["pnp_rejections"] = json.dumps(frame.get("pnp_rejections", {}), sort_keys=True)
        row["pnp_diagnostics"] = json.dumps(frame.get("pnp_diagnostics", []), sort_keys=True)
        if target:
            row.update(target)
            cell = (
                frame["camera_id"],
                max(0, min(11, int(target["truth_x"]))),
                max(0, min(11, int(target["truth_y"]))),
            )
            value = bins[cell]
            value["frames"] += 1
            value["detected"] += int(frame["detections"] > 0)
            value["pnp"] += int(frame["pnp_valid"] > 0)
            value["fusion"] += int(row["fusion_accepted"] > 0)
        obs = observation_by_frame.get((frame["camera_id"], frame["capture_ns"]))
        if obs:
            row["xy_error_m"] = obs.get("xy_error_m")
            row["yaw_error_deg"] = obs.get("yaw_error_deg")
        stats = camera_values[frame["camera_id"]]
        stats["frames"] += 1
        stats["detected"] += int(frame["detections"] > 0)
        stats["pnp"] += int(frame["pnp_valid"] > 0)
        stats["fusion"] += int(row["fusion_accepted"] > 0)
        stats["latency"].append(frame["latency_ms"])
        if frame.get("best_reprojection_px") is not None:
            stats["reprojection"].append(frame["best_reprojection_px"])
        if row.get("xy_error_m") is not None:
            stats["xy_error"].append(row["xy_error_m"])
        frame_rows.append(row)

    estimate_rows = []
    for item in load_jsonl(runtime / "odometry.jsonl"):
        state = item["state"]
        target = truth_at(truth, truth_stamps, state["stamp_ns"])
        row = {
            "stamp_ns": state["stamp_ns"],
            "capture_ns": item["capture_ns"],
            "valid": int(item["valid"]),
            "tracking_state": item["tracking_state"],
            "estimate_x": state["x"],
            "estimate_y": state["y"],
            "estimate_yaw": state["yaw"],
            "estimate_vx": state["vx"],
            "estimate_vy": state["vy"],
            "measurement_age_ms": item["measurement_age_ms"],
            "measurement_wall_hz": item.get("measurement_wall_hz"),
            "output_wall_hz": item.get("output_wall_hz"),
        }
        if target:
            row.update(target)
            row["xy_error_m"] = math.hypot(
                state["x"] - target["truth_x"], state["y"] - target["truth_y"]
            )
            row["yaw_error_deg"] = math.degrees(
                angle_error(state["yaw"], target["truth_yaw"])
            )
        estimate_rows.append(row)

    frame_fields = [
        "camera_id", "sequence", "capture_ns", "detections", "tag_ids",
        "pnp_valid", "fusion_accepted", "best_quality",
        "best_reprojection_px", "pnp_rejections", "pnp_diagnostics", "latency_ms", "truth_x", "truth_y",
        "truth_z", "truth_yaw", "truth_vx", "truth_vy", "waypoint_index",
        "xy_error_m", "yaw_error_deg",
    ]
    observation_fields = [
        "camera_id", "frame_seq", "capture_ns", "accepted", "selection_reason", "marker_id",
        "estimate_x", "estimate_y", "estimate_z", "estimate_yaw", "quality",
        "reprojection_px", "tag_side_px", "planar_tilt_deg", "xy_sigma_m", "processing_ms", "truth_x",
        "truth_y", "truth_z", "truth_yaw", "waypoint_index", "xy_error_m",
        "z_error_m", "yaw_error_deg",
    ]
    estimate_fields = [
        "stamp_ns", "capture_ns", "valid", "tracking_state", "estimate_x",
        "estimate_y", "estimate_yaw", "estimate_vx", "estimate_vy",
        "measurement_age_ms", "measurement_wall_hz", "output_wall_hz",
        "truth_x", "truth_y", "truth_z", "truth_yaw", "truth_vx",
        "truth_vy", "waypoint_index", "xy_error_m", "yaw_error_deg",
    ]
    write_csv(run_dir / "camera_frames.csv", frame_rows, frame_fields)
    write_csv(run_dir / "observations.csv", observation_rows, observation_fields)
    write_csv(run_dir / "estimates.csv", estimate_rows, estimate_fields)

    summary_rows = []
    for camera_id, values in sorted(camera_values.items()):
        total = max(values["frames"], 1)
        summary_rows.append({
            "camera_id": camera_id,
            "frames": values["frames"],
            "detection_rate": values["detected"] / total,
            "pnp_rate": values["pnp"] / total,
            "fusion_accept_rate": values["fusion"] / total,
            "p95_latency_ms": percentile(values["latency"], 95),
            "p95_reprojection_px": percentile(values["reprojection"], 95),
            "p50_xy_error_m": percentile(values["xy_error"], 50),
            "p95_xy_error_m": percentile(values["xy_error"], 95),
        })
    write_csv(run_dir / "camera_summary.csv", summary_rows, list(summary_rows[0]) if summary_rows else ["camera_id"])

    bin_rows = []
    for (camera_id, cell_x, cell_y), values in sorted(bins.items()):
        total = values["frames"]
        bin_rows.append({
            "camera_id": camera_id, "cell_x": cell_x, "cell_y": cell_y,
            "frames": total, "detection_rate": values["detected"] / total,
            "pnp_rate": values["pnp"] / total,
            "fusion_accept_rate": values["fusion"] / total,
        })
    write_csv(run_dir / "spatial_bins.csv", bin_rows, list(bin_rows[0]) if bin_rows else ["camera_id", "cell_x", "cell_y"])

    valid_estimates = [row for row in estimate_rows if row.get("xy_error_m") is not None and row["valid"]]
    report = {
        "truth_samples": len(truth),
        "camera_frames": len(frame_rows),
        "observations": len(observation_rows),
        "estimate_samples": len(estimate_rows),
        "valid_estimate_samples": len(valid_estimates),
        "p50_fused_xy_error_m": percentile([row["xy_error_m"] for row in valid_estimates], 50),
        "p95_fused_xy_error_m": percentile([row["xy_error_m"] for row in valid_estimates], 95),
        "p95_fused_yaw_error_deg": percentile([abs(row["yaw_error_deg"]) for row in valid_estimates], 95),
        "files_to_share": [
            "truth.csv", "camera_frames.csv", "observations.csv",
            "estimates.csv", "camera_summary.csv", "spatial_bins.csv",
            "manifest.json", "summary.json",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
