#!/usr/bin/env python3
"""Evaluator only (reads truth): what SAM2 added to a replay, for report 08.

Never imported by the runtime.  For one replay directory and its dataset:

* the accuracy table of ``conductor_check.py replay`` (both rovers: P50/P95,
  valid by sim time, swaps, measurement age P95);
* the opponent's accepted measurements by source, and SAM2's share;
* SAM2's refusals and stand-bys by reason (``observations.jsonl``);
* the error of every SAM2 reading against truth, whether it reached the
  filter or not;
* contact episodes (truth centre distance below ``--contact-m``) and what
  the opponent track did in each: largest error, swap;
* reacquisition after each ``--opponent-blackout`` interval: time from its
  end to the first valid opponent output within ``--reacquire-m`` of truth,
  and the source of the measurement that brought it there;
* SAM2 GPU time and VRAM from ``timing.json``.

``truth`` alone (``--truth-only``) prints the contact episodes of a dataset.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conductor_check import check_replay, load_truth, read_jsonl  # noqa: E402


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def contact_episodes(truth, contact_m=0.8, merge_ns=200_000_000):
    """[(start_ns, end_ns, min distance)] where the truth centres are closer than contact_m."""
    tag, opp = truth.get("tag_rover"), truth.get("opponent")
    if tag is None or opp is None:
        return []
    stamps = tag.t
    flags, distances = [], []
    for stamp in stamps:
        a, b = tag.at(int(stamp)), opp.at(int(stamp))
        if a is None or b is None:
            flags.append(False)
            distances.append(math.inf)
            continue
        d = float(np.linalg.norm(a[0] - b[0]))
        flags.append(d < contact_m)
        distances.append(d)
    episodes = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        if (not flag or i == len(flags) - 1) and start is not None:
            end = i if flag else i - 1
            episodes.append([int(stamps[start]), int(stamps[end]),
                             float(min(distances[start:end + 1]))])
            start = None
    merged = []
    for episode in episodes:
        if merged and episode[0] - merged[-1][1] <= merge_ns:
            merged[-1][1] = episode[1]
            merged[-1][2] = min(merged[-1][2], episode[2])
        else:
            merged.append(episode)
    return [tuple(e) for e in merged]


def opponent_rows(runtime):
    rows = [r for r in read_jsonl(runtime / "odometry.jsonl") if r.get("object_id") == "opponent"]
    rows.sort(key=lambda r: int(r["state"]["stamp_ns"]))
    return rows


def episode_report(rows, truth, episodes, t0=0, pad_ns=500_000_000, swap_m=0.5, swap_ms=100.0):
    opp, tag = truth["opponent"], truth["tag_rover"]
    out = []
    stamps = np.array([int(r["state"]["stamp_ns"]) for r in rows], dtype=np.int64)
    for start, end, dmin in episodes:
        lo, hi = np.searchsorted(stamps, [start - pad_ns, end + pad_ns])
        errors, swap_run, swap_longest, valid = [], 0, 0, 0
        last = None
        for r in rows[lo:hi]:
            stamp = int(r["state"]["stamp_ns"])
            g, t = opp.at(stamp), tag.at(stamp)
            if not r.get("valid") or g is None:
                swap_run = 0
                continue
            valid += 1
            p = np.array([r["state"]["x"], r["state"]["y"]])
            err = float(np.linalg.norm(p - g[0]))
            errors.append(err)
            swapped = t is not None and float(np.linalg.norm(p - t[0])) < swap_m and err > swap_m
            if swapped:
                swap_run += (stamp - last) if last is not None else 0
                swap_longest = max(swap_longest, swap_run)
            else:
                swap_run = 0
            last = stamp
        out.append({"start_s": round((start - t0) / 1e9, 3), "duration_s": (end - start) / 1e9,
                    "min_distance_m": round(dmin, 3),
                    "opp_err_max_m": max(errors) if errors else None,
                    "opp_err_p95_m": pct(errors, 95),
                    "valid_outputs": valid, "outputs": int(hi - lo),
                    "swap": swap_longest / 1e6 > swap_ms, "swap_longest_ms": swap_longest / 1e6})
    return out


def reacquire_report(runtime, rows, truth, params, reacquire_m=0.15):
    blackouts = params.get("opponent_blackout_s") or []
    t0 = int(params["replay"]["sim_start_ns"])
    if not blackouts:
        return []
    observations = [r for r in read_jsonl(runtime / "observations.jsonl")
                    if r.get("accepted") and r["observation"].get("object_id") == "opponent"]
    observations.sort(key=lambda r: int(r["wall_ns"]))
    accepted_at = np.array([int(r["wall_ns"]) for r in observations], dtype=np.int64)
    opp = truth["opponent"]
    status = json.loads((runtime / "status.json").read_text())
    reacq = (status.get("identity") or {}).get("opponent_reacquisitions") or []
    out = []
    for lo, hi in blackouts:
        end = t0 + int(hi * 1e9)
        start = t0 + int(lo * 1e9)
        first = None
        for r in rows:
            stamp = int(r["state"]["stamp_ns"])
            if stamp < end or not r.get("valid"):
                continue
            g = opp.at(stamp)
            if g is None:
                continue
            if float(np.linalg.norm(np.array([r["state"]["x"], r["state"]["y"]]) - g[0])) < reacquire_m:
                first = stamp
                break
        source = None
        if first is not None:
            i = int(np.searchsorted(accepted_at, first, side="right")) - 1
            if i >= 0:
                source = observations[i]["observation"].get("method")
        during = [r for r in rows if start <= int(r["state"]["stamp_ns"]) < end]
        out.append({"interval_s": [lo, hi],
                    "reacquire_s": None if first is None else (first - end) / 1e9,
                    "source": source,
                    "valid_during_blackout_fraction": (sum(bool(r.get("valid")) for r in during)
                                                       / max(len(during), 1)),
                    "reacquisitions": [x for x in reacq
                                       if end - 2_000_000_000 <= int(x["stamp_ns"]) <= end + 5_000_000_000]})
    return out


def sam2_accuracy(runtime, truth):
    opp, tag = truth["opponent"], truth["tag_rover"]
    errors, by_reason, nearer_tag = [], Counter(), 0
    rows = read_jsonl(runtime / "observations.jsonl")
    for r in rows:
        o = r["observation"]
        if o.get("method") != "sam2" or o.get("position_m") is None:
            continue
        g, t = opp.at(int(o["capture_time_ns"])), tag.at(int(o["capture_time_ns"]))
        if g is None:
            continue
        p = np.array(o["position_m"][:2], dtype=float)
        err = float(np.linalg.norm(p - g[0]))
        reason = str(r.get("selection_reason"))
        by_reason[reason] += 1
        if not reason.startswith("sam2_reject"):
            errors.append(err)
            if t is not None and float(np.linalg.norm(p - t[0])) < err:
                nearer_tag += 1
    return {"readings": len(errors), "err_p50_m": pct(errors, 50), "err_p95_m": pct(errors, 95),
            "err_max_m": max(errors) if errors else None,
            "readings_nearer_tag_truth": nearer_tag}


def evaluate(runtime, dataset, contact_m=0.8):
    runtime, dataset = Path(runtime), Path(dataset)
    truth = load_truth(dataset / "truth.jsonl")
    check = check_replay(runtime, dataset / "truth.jsonl")
    params = json.loads((runtime / "runtime_parameters.json").read_text())
    timing = json.loads((runtime / "timing.json").read_text())
    rows = opponent_rows(runtime)
    out = {"runtime": str(runtime), "dataset": str(dataset), "objects": {}}
    for name, obj in check["objects"].items():
        if "xy_error_m" not in obj:
            continue
        out["objects"][name] = {
            "p50_m": obj["xy_error_m"]["p50"], "p95_m": obj["xy_error_m"]["p95"],
            "max_m": obj["xy_error_m"]["max"], "valid": obj["valid_fraction_sim_time"],
            "swaps": obj["swap_episodes_over_limit"], "age_p95_ms": obj["measurement_age_ms"]["p95"],
            "yaw_p95_deg": (obj["yaw_error_deg"] or {}).get("p95")}
    obs = check["observations"].get("opponent", {})
    methods = Counter()
    reasons = Counter()
    for r in read_jsonl(runtime / "observations.jsonl"):
        o = r["observation"]
        if o.get("object_id") != "opponent":
            continue
        if r.get("accepted"):
            methods[o.get("method")] += 1
        if o.get("method") == "sam2":
            reasons[str(r.get("selection_reason"))] += 1
    total = sum(methods.values())
    out["opponent_accepted_by_method"] = dict(methods)
    out["sam2_share"] = methods.get("sam2", 0) / max(total, 1)
    out["sam2_selection_reasons"] = dict(reasons)
    out["sam2_reading_error"] = sam2_accuracy(runtime, truth)
    episodes = contact_episodes(truth, contact_m)
    t0 = int(params["replay"]["sim_start_ns"])
    if params["replay"].get("seconds"):
        stop = t0 + int(params["replay"]["seconds"] * 1e9)
        episodes = [e for e in episodes if e[0] < stop]
    out["contact_episodes"] = episode_report(rows, truth, episodes, t0)
    out["contact_swaps"] = sum(e["swap"] for e in out["contact_episodes"])
    out["reacquire"] = reacquire_report(runtime, rows, truth, params)
    sam2 = timing.get("sam2")
    out["sam2_timing"] = None if not sam2 else {
        "sam2_ms": sam2["sam2_ms"], "vram_peak_mb": sam2["vram_peak_mb"],
        "counts": sam2["counts"], "prompts": sam2["prompts"]}
    out["realtime_ratio"] = timing.get("realtime_ratio")
    out["observations_total"] = obs.get("total")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runtime", type=Path, nargs="?")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--contact-m", type=float, default=0.8)
    p.add_argument("--truth-only", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.truth_only:
        truth = load_truth(a.dataset / "truth.jsonl")
        episodes = contact_episodes(truth, a.contact_m)
        t0 = min(int(track.t[0]) for track in truth.values())
        report = {"dataset": str(a.dataset), "contact_m": a.contact_m,
                  "episodes": len(episodes),
                  "total_s": sum((e - s) / 1e9 for s, e, _ in episodes),
                  "list": [{"start_s": round((s - t0) / 1e9, 3), "duration_s": round((e - s) / 1e9, 3),
                            "min_distance_m": round(d, 3)} for s, e, d in episodes]}
    else:
        report = evaluate(a.runtime, a.dataset, a.contact_m)
    text = json.dumps(report, indent=2)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
