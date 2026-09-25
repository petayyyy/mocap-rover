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
            "incidence_deg": px.get("incidence_deg"),
            "edge_distance_px": px.get("edge_distance_px"),
            "quad_aspect": px.get("quad_aspect"),
            "xy_source": px.get("xy_source"),
            "pnp_x": (px.get("pnp_xy") or [None, None])[0],
            "pnp_y": (px.get("pnp_xy") or [None, None])[1],
            "ray_x": (px.get("ray_xy") or [None, None])[0],
            "ray_y": (px.get("ray_xy") or [None, None])[1],
            "pnp_base_z_m": px.get("pnp_base_z_m"),
            "pnp_ray_disagreement_m": px.get("pnp_ray_disagreement_m"),
            "range_sigma_m": px.get("range_sigma_m"),
            "xy_sigma_m": px.get("xy_sigma_m"),
            "yaw_sigma_rad": px.get("yaw_sigma_rad"),
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
        row["roi"] = ";".join(str(v) for v in (frame.get("roi") or []))
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
        state = item.get("state")
        stamp_ns = item.get("stamp_ns", (state or {}).get("stamp_ns"))
        if state is None:
            # LOST before the first marker: record the gap rather than
            # dropping the sample, or coverage is computed over a shorter run
            # than actually happened.
            estimate_rows.append({
                "stamp_ns": stamp_ns, "capture_ns": item.get("capture_ns"),
                "valid": 0, "tracking_state": item["tracking_state"],
                "measurement_age_ms": item.get("measurement_age_ms"),
            })
            continue
        covariance = item.get("covariance") or []
        target = truth_at(truth, truth_stamps, stamp_ns)
        row = {
            "stamp_ns": stamp_ns,
            "capture_ns": item.get("capture_ns"),
            "valid": int(item["valid"]),
            "tracking_state": item["tracking_state"],
            "estimate_x": state["x"],
            "estimate_y": state["y"],
            "estimate_yaw": state["yaw"],
            "estimate_vx": state["vx"],
            "estimate_vy": state["vy"],
            "estimate_yaw_rate": state.get("yaw_rate", state.get("omega")),
            "measurement_age_ms": item["measurement_age_ms"],
            "identity_age_ms": item.get("identity_age_ms"),
            "measurement_wall_hz": item.get("measurement_wall_hz"),
            "output_wall_hz": item.get("output_wall_hz"),
            "source_mask": ";".join(item.get("source_mask") or item.get("sources") or []),
            "identity": item.get("identity"),
            "out_of_sequence": item.get("out_of_sequence"),
            "dropped_too_old": item.get("dropped_too_old"),
            "id_rejections": item.get("id_rejections"),
            "cov_xx": covariance[0][0] if covariance else None,
            "cov_xy": covariance[0][1] if covariance else None,
            "cov_yy": covariance[1][1] if covariance else None,
            "cov_yawyaw": covariance[4][4] if len(covariance) > 4 else None,
            "model_probabilities": ";".join(
                f"{v:.4f}" for v in item.get("model_probabilities", [])),
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
        "best_reprojection_px", "mode", "roi", "pnp_rejections",
        "pnp_diagnostics", "latency_ms", "truth_x", "truth_y",
        "truth_z", "truth_yaw", "truth_vx", "truth_vy", "waypoint_index",
        "xy_error_m", "yaw_error_deg",
    ]
    observation_fields = [
        "camera_id", "frame_seq", "capture_ns", "accepted", "selection_reason", "marker_id",
        "estimate_x", "estimate_y", "estimate_z", "estimate_yaw", "quality",
        "reprojection_px", "tag_side_px", "planar_tilt_deg", "incidence_deg",
        "edge_distance_px", "quad_aspect", "xy_source", "pnp_x", "pnp_y",
        "ray_x", "ray_y", "pnp_base_z_m", "pnp_ray_disagreement_m",
        "range_sigma_m", "xy_sigma_m", "yaw_sigma_rad", "processing_ms", "truth_x",
        "truth_y", "truth_z", "truth_yaw", "waypoint_index", "xy_error_m",
        "z_error_m", "yaw_error_deg",
    ]
    estimate_fields = [
        "stamp_ns", "capture_ns", "valid", "tracking_state", "estimate_x",
        "estimate_y", "estimate_yaw", "estimate_vx", "estimate_vy",
        "estimate_yaw_rate", "measurement_age_ms", "identity_age_ms",
        "measurement_wall_hz", "output_wall_hz", "source_mask", "identity",
        "out_of_sequence", "dropped_too_old", "id_rejections",
        "cov_xx", "cov_xy", "cov_yy", "cov_yawyaw", "model_probabilities",
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
    ages = [row["measurement_age_ms"] for row in estimate_rows
            if row.get("measurement_age_ms") is not None]
    states = defaultdict(int)
    for row in estimate_rows:
        states[row["tracking_state"]] += 1
    report = {
        "truth_samples": len(truth),
        "camera_frames": len(frame_rows),
        "observations": len(observation_rows),
        "estimate_samples": len(estimate_rows),
        "valid_estimate_samples": len(valid_estimates),
        "valid_coverage": len(valid_estimates) / max(len(estimate_rows), 1),
        "tracking_states": dict(states),
        "p50_measurement_age_ms": percentile(ages, 50),
        "p95_measurement_age_ms": percentile(ages, 95),
        "id_rejections": max((row.get("id_rejections") or 0 for row in estimate_rows),
                             default=0),
        "out_of_sequence": max((row.get("out_of_sequence") or 0 for row in estimate_rows),
                               default=0),
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
