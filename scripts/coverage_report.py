#!/usr/bin/env python3
"""Coverage of a replay: could each rover be seen, and was it?

Evaluator-side only: reads ``truth.jsonl``, so nothing here may reach the
runtime.  For every render instant of the cameras the replay used:

``marker could``     tag_rover stands (truth z below --upright-max-z) and its
                     top marker centre projects inside at least one camera's
                     frame and valid lens radius, at an incidence of at most
                     --max-incidence-deg from the vertical and with a projected
                     side of at least --min-side-px;
``marker accepted``  at least one camera's marker observation of that instant
                     was accepted by the fusion (``observations.jsonl``);
``silhouette could`` the opponent's top-face centre projects inside a frame
                     at the same incidence bound;
``silhouette accepted`` at least one opponent observation of that instant was
                     accepted.

"could" minus "accepted" is what the tract lost, not what the cameras'
placement cannot see.  Writes ``coverage.json`` into the replay directory
and prints it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localization_contracts.camera_model import CameraModel  # noqa: E402

MARKER_ABOVE_BASE_M = 0.2254
MARKER_SIDE_M = 0.4
OPPONENT_TOP_M = 0.483


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class View:
    def __init__(self, camera):
        self.model = CameraModel.from_config(camera)
        self.R = np.asarray(camera["R_world_optical"], dtype=float)
        self.C = np.asarray(camera["position_world"], dtype=float)

    def pixel(self, point):
        optical = self.R.T @ (np.asarray(point, dtype=float) - self.C)
        if optical[2] <= 1e-6:
            return None
        uv = self.model.project(optical.reshape(1, 3))
        if not (np.isfinite(uv).all() and self.model.inside_valid_radius(uv).all()):
            return None
        u, v = uv[0]
        if not (0 <= u < self.model.width and 0 <= v < self.model.height):
            return None
        return float(u), float(v)

    def incidence_deg(self, point):
        ray = np.asarray(point, dtype=float) - self.C
        return math.degrees(math.acos(min(1.0, abs(ray[2]) / max(np.linalg.norm(ray), 1e-9))))

    def side_px(self, x, y, z):
        half = MARKER_SIDE_M / 2
        sides = []
        for dx, dy in ((half, 0.0), (0.0, half)):
            a = self.pixel((x - dx, y - dy, z))
            b = self.pixel((x + dx, y + dy, z))
            if a is None or b is None:
                return 0.0
            sides.append(math.hypot(b[0] - a[0], b[1] - a[1]))
        return float(np.mean(sides))


def interpolate(rows, stamp):
    """(x, y, z) of a sorted truth list at ``stamp``, or None outside it."""
    ts = [r["stamp_ns"] for r in rows]
    i = int(np.searchsorted(ts, stamp))
    if i <= 0 or i >= len(rows):
        return None
    lo, hi = rows[i - 1], rows[i]
    a = (stamp - lo["stamp_ns"]) / max(hi["stamp_ns"] - lo["stamp_ns"], 1)
    return tuple(lo[k] * (1 - a) + hi[k] * a for k in ("x", "y", "z"))


def coverage(replay, dataset, *, max_incidence_deg=65.0, min_side_px=20.0, upright_max_z=0.2):
    replay, dataset = Path(replay), Path(dataset)
    calibration = json.loads((replay / "initial_calibration.json").read_text())
    status = json.loads((replay / "status.json").read_text())
    used = sorted(status["cameras"])
    views = {c["name"]: View(c) for c in calibration["cameras"] if c["name"] in used}
    stamps = sorted({int(r["stamp_ns"]) for cid in used
                     for r in read_jsonl(dataset / f"{cid}.jsonl")})
    truth = {}
    for row in read_jsonl(dataset / "truth.jsonl"):
        truth.setdefault(row["object_id"], []).append(row)
    for rows in truth.values():
        rows.sort(key=lambda r: r["stamp_ns"])
    accepted = {"tag_rover": set(), "opponent": set()}
    for row in read_jsonl(replay / "observations.jsonl"):
        obs = row["observation"]
        if row["accepted"] and obs["object_id"] in accepted and obs["camera_id"] in views:
            accepted[obs["object_id"]].add(int(obs["capture_time_ns"]))
    counts = {k: 0 for k in ("instants", "marker_could", "marker_accepted", "marker_both",
                             "silhouette_could", "silhouette_accepted", "silhouette_both")}
    for stamp in stamps:
        tag = interpolate(truth.get("tag_rover", []), stamp)
        opp = interpolate(truth.get("opponent", []), stamp)
        if tag is None and opp is None:
            continue
        counts["instants"] += 1
        if tag is not None:
            could = False
            if tag[2] < upright_max_z:
                point = (tag[0], tag[1], tag[2] + MARKER_ABOVE_BASE_M)
                could = any(v.pixel(point) is not None
                            and v.incidence_deg(point) <= max_incidence_deg
                            and v.side_px(*point) >= min_side_px for v in views.values())
            took = stamp in accepted["tag_rover"]
            counts["marker_could"] += could
            counts["marker_accepted"] += took
            counts["marker_both"] += could and took
        if opp is not None:
            point = (opp[0], opp[1], OPPONENT_TOP_M)
            could = any(v.pixel(point) is not None and v.incidence_deg(point) <= max_incidence_deg
                        for v in views.values())
            took = stamp in accepted["opponent"]
            counts["silhouette_could"] += could
            counts["silhouette_accepted"] += took
            counts["silhouette_both"] += could and took
    n = max(counts["instants"], 1)
    report = {
        "replay": str(replay), "dataset": str(dataset), "cameras": used,
        "max_incidence_deg": max_incidence_deg, "min_side_px": min_side_px,
        "instants": counts["instants"],
        "marker": {"could": counts["marker_could"] / n,
                   "accepted": counts["marker_accepted"] / n,
                   "accepted_when_could": counts["marker_both"] / max(counts["marker_could"], 1)},
        "silhouette": {"could": counts["silhouette_could"] / n,
                       "accepted": counts["silhouette_accepted"] / n,
                       "accepted_when_could": counts["silhouette_both"]
                       / max(counts["silhouette_could"], 1)},
        "note": "fractions of render instants; accepted = at least one camera's observation "
                "of that instant accepted by the fusion",
    }
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("replay", type=Path)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--max-incidence-deg", type=float, default=65.0)
    p.add_argument("--min-side-px", type=float, default=20.0)
    p.add_argument("--upright-max-z", type=float, default=0.2)
    a = p.parse_args(argv)
    report = coverage(a.replay, a.dataset, max_incidence_deg=a.max_incidence_deg,
                      min_side_px=a.min_side_px, upright_max_z=a.upright_max_z)
    (a.replay / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
