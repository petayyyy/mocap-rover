#!/usr/bin/env python3
"""Conductor-side checks: dataset truth sanity and a full replay breakdown.

Evaluator-side tool, it reads truth on purpose and is never imported by the
runtime.  ``scripts/evaluate_recording.py`` stays the acceptance evaluator;
this adds what the acceptance table in ``docs/dataset_tz/README.md`` asks
for and the evaluator does not print: P50, valid share of simulated time,
identity swaps, error by range to the nearest camera and by speed, PnP
rejection reasons, ROI mode share and per-camera frame times.

    conductor_check.py truth  <dataset>
    conductor_check.py replay <runtime> --truth <dataset>/truth.jsonl [--config ...]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OBJECTS = ("tag_rover", "opponent")
RANGE_BINS = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 99.0))
SPEED_BINS = ((0.0, 3.0), (3.0, 8.0), (8.0, 99.0))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(values, q):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.percentile(values, q)) if values else None


def summary(values):
    return {"n": len(values), "mean": (float(np.mean(values)) if values else None),
            "p50": pct(values, 50), "p95": pct(values, 95), "max": (float(np.max(values)) if values else None)}


# --------------------------------------------------------------------- truth

class Track:
    """Interpolated truth for one object, with a finite-difference speed."""

    def __init__(self, rows):
        rows = sorted(rows, key=lambda r: r["stamp_ns"])
        self.t = np.array([r["stamp_ns"] for r in rows], dtype=np.int64)
        self.xy = np.array([[r["x"], r["y"]] for r in rows], dtype=float)
        self.z = np.array([r.get("z", 0.0) for r in rows], dtype=float)
        self.yaw = np.unwrap(np.array([r["yaw"] for r in rows], dtype=float))
        # Speed over a 50 ms window: 200 Hz rows differenced one apart are
        # dominated by physics jitter.
        step = max(1, int(round(0.05 / max(np.median(np.diff(self.t)) / 1e9, 1e-6)))) if len(self.t) > 1 else 1
        self.speed = np.zeros(len(self.t))
        if len(self.t) > step:
            d = np.linalg.norm(self.xy[step:] - self.xy[:-step], axis=1)
            dt = (self.t[step:] - self.t[:-step]) / 1e9
            v = d / np.maximum(dt, 1e-9)
            self.speed[step // 2: step // 2 + len(v)] = v

    def __len__(self):
        return len(self.t)

    def at(self, stamp_ns):
        """(xy, yaw, speed) or None when outside the record."""
        i = int(np.searchsorted(self.t, stamp_ns))
        if i <= 0 or i >= len(self.t):
            return None
        dt = self.t[i] - self.t[i - 1]
        if dt <= 0 or dt > 200_000_000:
            return None
        a = (stamp_ns - self.t[i - 1]) / dt
        return (self.xy[i - 1] * (1 - a) + self.xy[i] * a,
                self.yaw[i - 1] * (1 - a) + self.yaw[i] * a,
                self.speed[i - 1] * (1 - a) + self.speed[i] * a)


def load_truth(path):
    by_object = defaultdict(list)
    for row in read_jsonl(path):
        by_object[row["object_id"]].append(row)
    return {name: Track(rows) for name, rows in by_object.items()}


def check_truth(dataset, z_limit=0.2, jump_m=0.4):
    dataset = Path(dataset)
    meta = json.loads((dataset / "meta.json").read_text()) if (dataset / "meta.json").exists() else {}
    report = {"dataset": str(dataset), "meta": {
        "truth_rows": meta.get("truth_rows"), "image_size_errors": meta.get("image_size_errors"),
        "scans": (meta.get("lidar") or {}).get("scans"),
        "sim_seconds": ((meta.get("sim_end_ns", 0) - meta.get("sim_start_ns", 0)) / 1e9
                        if meta.get("sim_end_ns") and meta.get("sim_start_ns") else None),
        "frames": {cid: c.get("frames") for cid, c in (meta.get("cameras") or {}).items()},
    }, "cameras": {}, "objects": {}, "files": {}}
    for cid in sorted((meta.get("cameras") or {})):
        rows = read_jsonl(dataset / f"{cid}.jsonl")
        stamps = np.array([r["stamp_ns"] for r in rows], dtype=np.int64)
        video = dataset / f"{cid}.mkv"
        gaps = np.diff(stamps) / 1e6 if len(stamps) > 1 else np.array([])
        report["cameras"][cid] = {
            "index_rows": len(rows), "meta_frames": meta["cameras"][cid].get("frames"),
            "video_bytes": video.stat().st_size if video.exists() else None,
            "monotonic": bool(np.all(gaps > 0)) if len(gaps) else None,
            "fps": (len(stamps) - 1) / max((stamps[-1] - stamps[0]) / 1e9, 1e-9) if len(stamps) > 1 else None,
            "frame_gap_ms_p50": pct(gaps.tolist(), 50), "frame_gap_ms_max": (float(gaps.max()) if len(gaps) else None),
        }
    lidar_rows = read_jsonl(dataset / "lidar.jsonl")
    report["files"]["lidar_index_rows"] = len(lidar_rows)
    report["files"]["lidar_npz_files"] = len(list((dataset / "lidar").glob("*.npz"))) if (dataset / "lidar").exists() else 0
    report["files"]["truth_bytes"] = (dataset / "truth.jsonl").stat().st_size if (dataset / "truth.jsonl").exists() else None
    for name, track in load_truth(dataset / "truth.jsonl").items():
        jumps = np.linalg.norm(np.diff(track.xy, axis=0), axis=1) if len(track) > 1 else np.array([])
        report["objects"][name] = {
            "rows": len(track),
            "sim_seconds": float((track.t[-1] - track.t[0]) / 1e9) if len(track) > 1 else 0.0,
            "z_above_limit_fraction": float(np.mean(track.z > z_limit)) if len(track) else None,
            "z_max": float(track.z.max()) if len(track) else None,
            "xy_jump_max_m": float(jumps.max()) if len(jumps) else None,
            "xy_jumps_over_limit": int(np.sum(jumps > jump_m)) if len(jumps) else 0,
            "speed_max": float(track.speed.max()) if len(track) else None,
            "speed_p90": pct(track.speed.tolist(), 90), "speed_p50": pct(track.speed.tolist(), 50),
            "row_gap_ms_max": float(np.diff(track.t).max() / 1e6) if len(track) > 1 else None,
            "x_range": [float(track.xy[:, 0].min()), float(track.xy[:, 0].max())] if len(track) else None,
            "y_range": [float(track.xy[:, 1].min()), float(track.xy[:, 1].max())] if len(track) else None,
        }
    tag, opp = report["objects"].get("tag_rover"), report["objects"].get("opponent")
    if tag and opp:
        a, b = load_truth(dataset / "truth.jsonl")["tag_rover"], load_truth(dataset / "truth.jsonl")["opponent"]
        close = []
        for t in a.t[::10]:
            q = b.at(int(t))
            p = a.at(int(t))
            if p is not None and q is not None:
                close.append(float(np.linalg.norm(p[0] - q[0])))
        report["rover_distance"] = {"min_m": min(close) if close else None,
                                    "fraction_under_1m": float(np.mean(np.array(close) < 1.0)) if close else None}
    return report


# -------------------------------------------------------------------- replay

def episodes(flags, stamps, min_ns):
    """Count contiguous runs of True longer than ``min_ns``; return (count, longest_ms)."""
    count, longest, start = 0, 0.0, None
    for flag, t in zip(list(flags) + [False], list(stamps) + [stamps[-1] if len(stamps) else 0]):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            length = t - start
            longest = max(longest, length / 1e6)
            if length > min_ns:
                count += 1
            start = None
    return count, longest


def check_replay(runtime, truth_path, config=None, swap_m=0.5, swap_ms=100.0):
    runtime = Path(runtime)
    truth = load_truth(truth_path)
    config = Path(config) if config else Path(truth_path).with_name("runtime_cameras.json")
    cameras = {}
    if config.exists():
        cfg = json.loads(config.read_text())
        cameras = {c["name"]: np.array(c["position_world"][:2], dtype=float) for c in cfg["cameras"]}
    camera_xy = np.array(list(cameras.values())) if cameras else None

    odometry = read_jsonl(runtime / "odometry.jsonl")
    by_object = defaultdict(list)
    for row in odometry:
        by_object[row["object_id"]].append(row)
    report = {"runtime": str(runtime), "objects": {}}
    span = None
    for track in truth.values():
        if len(track) > 1:
            lo, hi = int(track.t[0]), int(track.t[-1])
            span = (lo, hi) if span is None else (max(span[0], lo), min(span[1], hi))

    for name in OBJECTS:
        rows = sorted(by_object.get(name, []), key=lambda r: r["state"]["stamp_ns"] if r.get("state") else r.get("stamp_ns", 0))
        track = truth.get(name)
        other = truth.get("opponent" if name == "tag_rover" else "tag_rover")
        out = {"outputs": len(rows)}
        if not rows or track is None:
            out["reason"] = "no outputs or no truth"
            report["objects"][name] = out
            continue
        stamps, valid_flags, errors, yaw_errors, ages, states = [], [], [], [], [], Counter()
        by_range = defaultdict(list)
        by_speed = defaultdict(list)
        swap_flags = []
        for row in rows:
            stamp = int(row["state"]["stamp_ns"]) if row.get("state") else int(row.get("stamp_ns", 0))
            if span and not (span[0] <= stamp <= span[1]):
                continue
            gt = track.at(stamp)
            if gt is None:
                continue
            stamps.append(stamp)
            states[row.get("tracking_state", "?")] += 1
            valid = bool(row.get("valid"))
            valid_flags.append(valid)
            if not valid:
                swap_flags.append(False)
                continue
            p = np.array([row["state"]["x"], row["state"]["y"]], dtype=float)
            err = float(np.linalg.norm(p - gt[0]))
            errors.append(err)
            ages.append(row.get("measurement_age_ms"))
            if name == "tag_rover" and row["state"].get("yaw") is not None:
                yaw_errors.append(abs((row["state"]["yaw"] - gt[1] + math.pi) % (2 * math.pi) - math.pi) * 180 / math.pi)
            if camera_xy is not None:
                d = float(np.min(np.linalg.norm(camera_xy - gt[0], axis=1)))
                for lo, hi in RANGE_BINS:
                    if lo <= d < hi:
                        by_range[f"{lo:g}-{hi:g}"].append(err)
            for lo, hi in SPEED_BINS:
                if lo <= gt[2] < hi:
                    by_speed[f"{lo:g}-{hi:g}"].append(err)
            swapped = False
            if other is not None:
                og = other.at(stamp)
                if og is not None:
                    swapped = (float(np.linalg.norm(p - og[0])) < swap_m) and (err > swap_m)
            swap_flags.append(swapped)
        stamps_arr = np.array(stamps, dtype=np.int64)
        aligned = len(stamps)
        valid_n = int(sum(valid_flags))
        # Time-weighted: each output covers until the next one.
        time_valid = None
        if aligned > 1:
            intervals = np.diff(stamps_arr)
            time_valid = float(np.sum(intervals[np.array(valid_flags[:-1])]) / max(np.sum(intervals), 1))
        swaps, longest = episodes(swap_flags, stamps, swap_ms * 1e6)
        _, longest_gap = episodes([not v for v in valid_flags], stamps, 0)
        out.update({
            "aligned_outputs": aligned, "valid_outputs": valid_n,
            "valid_fraction_outputs": valid_n / max(aligned, 1),
            "valid_fraction_sim_time": time_valid,
            "output_hz_sim": (aligned - 1) / max((stamps_arr[-1] - stamps_arr[0]) / 1e9, 1e-9) if aligned > 1 else None,
            "longest_invalid_gap_ms": longest_gap,
            "xy_error_m": summary(errors),
            "yaw_error_deg": summary(yaw_errors) if name == "tag_rover" else None,
            "measurement_age_ms": summary([a for a in ages if a is not None]),
            "swap_episodes_over_limit": swaps, "swap_longest_ms": longest,
            "swap_flagged_outputs": int(sum(swap_flags)),
            "tracking_states": dict(states),
            "xy_error_by_range_to_nearest_camera_m": {k: summary(v) for k, v in sorted(by_range.items())},
            "xy_error_by_speed_mps": {k: summary(v) for k, v in sorted(by_speed.items())},
        })
        report["objects"][name] = out

    # Observations: what the observers produced and what the filter did with it.
    observations = read_jsonl(runtime / "observations.jsonl")
    obs_report = defaultdict(lambda: {"total": 0, "accepted": 0, "selection_reason": Counter(),
                                      "by_camera_accepted": Counter(), "method": Counter()})
    for row in observations:
        obs = row.get("observation", {})
        entry = obs_report[obs.get("object_id", "?")]
        entry["total"] += 1
        entry["accepted"] += int(bool(row.get("accepted")))
        entry["selection_reason"][str(row.get("selection_reason"))] += 1
        entry["method"][str(obs.get("method"))] += 1
        if row.get("accepted"):
            entry["by_camera_accepted"][str(obs.get("camera_id"))] += 1
    report["observations"] = {k: {kk: (dict(vv) if isinstance(vv, Counter) else vv) for kk, vv in v.items()}
                              for k, v in obs_report.items()}

    # Camera frames: mode share, rejection reasons and wall time per frame.
    frames = read_jsonl(runtime / "camera_frames.jsonl")
    per_camera = defaultdict(lambda: {"frames": 0, "modes": Counter(), "detections": 0, "pnp_valid": 0,
                                      "fusion_accepted": 0, "pnp_rejections": Counter(), "latency_ms": []})
    for row in frames:
        entry = per_camera[row.get("camera_id", "?")]
        entry["frames"] += 1
        entry["modes"][str(row.get("mode"))] += 1
        entry["detections"] += int(row.get("detections", 0) or 0)
        entry["pnp_valid"] += int(row.get("pnp_valid", 0) or 0)
        entry["fusion_accepted"] += int(row.get("fusion_accepted", 0) or 0)
        for reason, count in (row.get("pnp_rejections") or {}).items():
            entry["pnp_rejections"][reason] += int(count)
        if row.get("latency_ms") is not None:
            entry["latency_ms"].append(float(row["latency_ms"]))
    all_latency = []
    cam_out = {}
    for cid in sorted(per_camera):
        e = per_camera[cid]
        all_latency.extend(e["latency_ms"])
        modes = {m: round(c / max(e["frames"], 1), 3) for m, c in e["modes"].items()}
        by_mode = defaultdict(list)
        for row in frames:
            if row.get("camera_id") == cid and row.get("latency_ms") is not None:
                by_mode[str(row.get("mode"))].append(float(row["latency_ms"]))
        cam_out[cid] = {"frames": e["frames"], "mode_share": modes, "detections": e["detections"],
                        "pnp_valid": e["pnp_valid"], "fusion_accepted": e["fusion_accepted"],
                        "pnp_rejections": dict(e["pnp_rejections"]),
                        "latency_ms": summary(e["latency_ms"]),
                        "latency_ms_by_mode": {m: summary(v) for m, v in sorted(by_mode.items())}}
    report["camera_frames"] = {"total": len(frames), "latency_ms_all": summary(all_latency), "cameras": cam_out}

    lidar = read_jsonl(runtime / "lidar.jsonl")
    report["lidar"] = {"rows": len(lidar), "reasons": dict(Counter(str(r.get("reason")) for r in lidar)),
                       "processing_ms": summary([float(r["processing_ms"]) for r in lidar if r.get("processing_ms") is not None])}
    for extra in ("timing.json", "runtime_parameters.json"):
        path = runtime / extra
        if path.exists():
            try:
                report[extra] = json.loads(path.read_text())
            except json.JSONDecodeError:
                report[extra] = "unreadable"
    return report


def verdict(report, p95_m=0.15, p50_m=0.06, valid_min=0.98, yaw_p95_deg=5.0):
    """Acceptance table from docs/dataset_tz/README.md, accuracy rows only."""
    out = {}
    for name, obj in report["objects"].items():
        if "xy_error_m" not in obj:
            out[name] = {"pass": False, "reason": obj.get("reason", "no data")}
            continue
        checks = {
            "p95_xy": obj["xy_error_m"]["p95"] is not None and obj["xy_error_m"]["p95"] <= p95_m,
            "p50_xy": obj["xy_error_m"]["p50"] is not None and obj["xy_error_m"]["p50"] <= p50_m,
            "valid_sim_time": (obj["valid_fraction_sim_time"] or 0) >= valid_min,
            "swaps": obj["swap_episodes_over_limit"] == 0,
        }
        if name == "tag_rover":
            y = obj.get("yaw_error_deg") or {}
            checks["yaw_p95"] = y.get("p95") is not None and y["p95"] <= yaw_p95_deg
        out[name] = {"checks": checks, "pass": all(checks.values())}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("truth", help="dataset sanity from meta.json, camera indexes and truth.jsonl")
    t.add_argument("dataset", type=Path)
    t.add_argument("--z-limit", type=float, default=0.2)
    t.add_argument("--jump-m", type=float, default=0.4)
    t.add_argument("--output", type=Path)
    r = sub.add_parser("replay", help="full breakdown of a replay directory against truth")
    r.add_argument("runtime", type=Path)
    r.add_argument("--truth", type=Path, required=True)
    r.add_argument("--config", type=Path, help="runtime_cameras.json; defaults to the one beside truth")
    r.add_argument("--swap-m", type=float, default=0.5)
    r.add_argument("--swap-ms", type=float, default=100.0)
    r.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.command == "truth":
        report = check_truth(a.dataset, a.z_limit, a.jump_m)
    else:
        report = check_replay(a.runtime, a.truth, a.config, a.swap_m, a.swap_ms)
        report["verdict"] = verdict(report)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
