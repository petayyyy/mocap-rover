#!/usr/bin/env python3
"""Join recorded image odometry to evaluator-only Gazebo truth by simulation time."""
import argparse, json, math
from pathlib import Path
import numpy as np


def evaluate(runtime, truth_path, objects=("tag_rover", "opponent")):
    runtime = Path(runtime)
    truth = {name: [] for name in objects}
    for line in Path(truth_path).read_text().splitlines():
        row = json.loads(line)
        if row["object_id"] in truth:
            truth[row["object_id"]].append(row)
    records = {name: [] for name in truth}
    accepted = {name: [] for name in truth}
    for line in (runtime / "odometry.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["object_id"] in records:
            records[row["object_id"]].append(row)
    for line in (runtime / "observations.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["accepted"] and row["observation"]["object_id"] in accepted:
            accepted[row["observation"]["object_id"]].append(row)
    report = {}
    for name, rows in records.items():
        gt = sorted(truth[name], key=lambda r: r["stamp_ns"])
        ts = np.array([r["stamp_ns"] for r in gt])
        xy = np.array([[r["x"], r["y"]] for r in gt])
        yaw = np.unwrap([r["yaw"] for r in gt])
        errors = []
        yaw_errors = []
        speed_errors = []
        ages = []
        valid = 0
        aligned = 0
        if not rows or not gt:
            report[name] = {"pass": False, "reason": "missing observations or truth"}
            continue
        for row in rows:
            state = row["state"]
            t = state["stamp_ns"]
            index = np.searchsorted(ts, t)
            if index == 0 or index >= len(ts):
                continue
            lo = index - 1
            hi = index
            dt = (ts[hi] - ts[lo]) / 1e9
            if dt <= 0 or dt > 0.2:
                continue
            aligned += 1
            if not row["valid"]:
                continue
            valid += 1
            a = (t - ts[lo]) / (ts[hi] - ts[lo])
            target = xy[lo] * (1 - a) + xy[hi] * a
            predicted = np.array([state["x"], state["y"]])
            errors.append(float(np.linalg.norm(predicted - target)))
            heading = yaw[lo] * (1 - a) + yaw[hi] * a
            yaw_errors.append(
                abs((state["yaw"] - heading + math.pi) % (2 * math.pi) - math.pi)
                * 180
                / math.pi
            )
            speed_errors.append(
                float(
                    np.linalg.norm(
                        np.array([state["vx"], state["vy"]]) - (xy[hi] - xy[lo]) / dt
                    )
                )
            )
            ages.append(row["measurement_age_ms"])
        wall = np.array([r["wall_ns"] for r in rows])
        interval = np.diff(wall) / 1e6
        duration = (wall[-1] - wall[0]) / 1e9
        observations = accepted[name]
        stamps = [r["observation"]["capture_time_ns"] for r in observations]
        p95 = lambda values: float(np.percentile(values, 95)) if values else None
        m = {
            "outputs": len(rows),
            "aligned_outputs": aligned,
            "valid_outputs": valid,
            "availability": valid / max(aligned, 1),
            "wall_seconds": duration,
            "output_wall_hz": (len(rows) - 1) / max(duration, 1e-9),
            "accepted_measurement_wall_hz": len(set(stamps)) / max(duration, 1e-9),
            "p99_output_interval_ms": (
                float(np.percentile(interval, 99)) if len(interval) else None
            ),
            "p95_xy_m": p95(errors),
            "p95_yaw_deg": p95(yaw_errors) if name == "tag_rover" else None,
            "p95_velocity_error_m_s": p95(speed_errors),
            "p95_age_ms": p95(ages),
        }
        limits = (
            (0.05, 20, 100, 0.15, 66.7)
            if name == "tag_rover"
            else (0.10, 10, 150, 0.25, 133.3)
        )
        checks = {
            "availability": m["availability"] >= 0.99,
            "xy": bool(errors) and m["p95_xy_m"] <= limits[0],
            "fresh_measurement_hz": m["accepted_measurement_wall_hz"] >= limits[1],
            "age": bool(ages) and m["p95_age_ms"] <= limits[2],
            "velocity": bool(speed_errors) and m["p95_velocity_error_m_s"] <= limits[3],
            "publication_jitter": m["p99_output_interval_ms"] is not None
            and m["p99_output_interval_ms"] <= limits[4],
        }
        if name == "tag_rover":
            checks["yaw"] = bool(yaw_errors) and m["p95_yaw_deg"] <= 3
        checks = {key: bool(value) for key, value in checks.items()}
        report[name] = {**m, "checks": checks, "pass": all(checks.values())}
    return {
        "truth_role": "evaluator_only",
        "tracks": report,
        "sim_accepted": False,
        "scope": "this recording only; scenario suite and research gates separate",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tag-only", action="store_true")
    a = p.parse_args()
    result = evaluate(a.runtime, a.truth, ("tag_rover",) if a.tag_only else ("tag_rover", "opponent"))
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
