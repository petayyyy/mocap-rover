#!/usr/bin/env python3
"""Independent sanity check of a recorded dataset against its own truth.

Evaluator-side tool: it reads truth on purpose.  It decodes a sample of
frames from every camera, runs the existing marker detector and PnP observer
with the dataset's calibration, and compares the ray-plane XY with the
interpolated truth.  It also converts a few lidar scans to arena points and
looks for returns near the rovers.  A dataset whose calibration or timestamps
are wrong fails here before anyone spends a day on the tracker.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localization_contracts.apriltag import Detection, PnpAprilTagObserver, TagConfig  # noqa: E402
from localization_contracts.detector import AprilTagImageDetector  # noqa: E402
from localization_contracts.lidar_pipeline import ArenaLidar  # noqa: E402


def load_truth(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        out.setdefault(row["object_id"], []).append(row)
    tracks = {}
    for name, rows in out.items():
        rows.sort(key=lambda r: r["stamp_ns"])
        tracks[name] = (np.array([r["stamp_ns"] for r in rows], dtype=np.int64),
                        np.array([[r["x"], r["y"], r["yaw"]] for r in rows]))
    return tracks


def truth_at(track, stamp_ns):
    ts, xyy = track
    i = np.searchsorted(ts, stamp_ns)
    if i <= 0 or i >= len(ts):
        return None
    a = (stamp_ns - ts[i - 1]) / max(ts[i] - ts[i - 1], 1)
    return xyy[i - 1] * (1 - a) + xyy[i] * a


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--frames", type=int, default=60, help="frames sampled per camera")
    p.add_argument("--scans", type=int, default=5)
    a = p.parse_args()
    ds = a.dataset
    cfg = json.loads((ds / "runtime_cameras.json").read_text())
    truth = load_truth(ds / "truth.jsonl")
    tags = {int(t["id"]): {"rotation": t["R_base_tag"], "translation": t["T_base_tag_translation"]}
            for t in cfg["tags"]}
    family = cfg["tags"][0]["family"]
    size = float(cfg["tags"][0]["size"])
    detector = AprilTagImageDetector(family, allowed_ids=tuple(tags))
    report = {"cameras": {}}
    for cam in cfg["cameras"]:
        cid = cam["name"]
        index = [json.loads(l) for l in (ds / f"{cid}.jsonl").read_text().splitlines()]
        observer = PnpAprilTagObserver(
            TagConfig(family=family, ids=tuple(tags), size_m=size,
                      calibration_version=cfg["calibration_version"]),
            np.array(cam["K"]).reshape(3, 3), cam["D"],
            {"rotation": cam["R_world_optical"], "translation": cam["position_world"]},
            tags, image_size=cam["image_size"],
            tag_placement={int(t["id"]): t.get("placement", "top") for t in cfg["tags"]})
        capture = cv2.VideoCapture(str(ds / f"{cid}.mkv"))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = set(np.linspace(0, min(total, len(index)) - 1, a.frames).astype(int).tolist())
        stats = {"sampled": 0, "detected": 0, "accepted": 0, "errors_m": [], "yaw_err_deg": [],
                 "detect_ms": [], "decode_ms": [], "reject": {}, "video_frames": total,
                 "index_rows": len(index)}
        frame_no = 0
        while frame_no <= max(wanted):
            t0 = time.perf_counter()
            ok, bgr = capture.read()
            if not ok:
                break
            stats["decode_ms"].append((time.perf_counter() - t0) * 1000)
            if frame_no in wanted:
                stats["sampled"] += 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                stamp = int(index[frame_no]["stamp_ns"])
                t0 = time.perf_counter()
                hits = detector.detect(rgb)
                stats["detect_ms"].append((time.perf_counter() - t0) * 1000)
                for hit in hits:
                    stats["detected"] += 1
                    obs = observer.observe(Detection(cid, frame_no, hit.tag_id, hit.corners,
                                                     stamp, stamp, stamp))
                    diag = observer.last_diagnostic or {}
                    if obs is None:
                        reason = str(diag.get("reason"))
                        stats["reject"][reason] = stats["reject"].get(reason, 0) + 1
                        continue
                    stats["accepted"] += 1
                    gt = truth_at(truth["tag_rover"], stamp)
                    if gt is not None:
                        stats["errors_m"].append(float(np.hypot(obs.position_m[0] - gt[0],
                                                                obs.position_m[1] - gt[1])))
                        yaw = obs.pixel_features["yaw_rad"]
                        d = (yaw - gt[2] + np.pi) % (2 * np.pi) - np.pi
                        stats["yaw_err_deg"].append(float(abs(np.degrees(d))))
            frame_no += 1
        capture.release()
        e = np.array(stats["errors_m"]) if stats["errors_m"] else np.array([np.nan])
        y = np.array(stats["yaw_err_deg"]) if stats["yaw_err_deg"] else np.array([np.nan])
        report["cameras"][cid] = {
            "video_frames": total, "index_rows": len(index), "sampled": stats["sampled"],
            "detected": stats["detected"], "accepted": stats["accepted"],
            "xy_error_p50_m": float(np.nanmedian(e)), "xy_error_p95_m": float(np.nanpercentile(e, 95)),
            "xy_error_max_m": float(np.nanmax(e)),
            "yaw_error_p95_deg": float(np.nanpercentile(y, 95)),
            "detect_ms_full_frame_p50": float(np.median(stats["detect_ms"])),
            "decode_ms_p50": float(np.median(stats["decode_ms"])),
            "pnp_rejections": stats["reject"],
        }
        print(cid, json.dumps(report["cameras"][cid]), flush=True)

    lidar_cfg = cfg.get("lidar")
    if lidar_cfg and (ds / "lidar.jsonl").exists():
        rows = [json.loads(l) for l in (ds / "lidar.jsonl").read_text().splitlines()]
        lidar = ArenaLidar(lidar_cfg["position_world"], lidar_cfg["R_world_sensor"])
        near = []
        for row in rows[:: max(1, len(rows) // a.scans)][: a.scans]:
            z = np.load(ds / row["file"])
            parsed = {"ranges": z["ranges"].astype(float), "azimuth": z["azimuth"].astype(float),
                      "elevation": z["elevation"].astype(float), "range_min": float(z["range_min"]),
                      "range_max": float(z["range_max"]), "stamp_ns": int(row["stamp_ns"])}
            scan = lidar.scan_to_arena(parsed)
            pts = scan.points[lidar.static_mask(scan.points)]
            for name in ("tag_rover", "opponent"):
                gt = truth_at(truth[name], parsed["stamp_ns"])
                if gt is None:
                    continue
                d = np.linalg.norm(pts[:, :2] - gt[:2], axis=1)
                near.append({"scan": row["file"], "object": name, "points_within_0.6m": int(np.sum(d < 0.6)),
                             "returns_in_band": int(len(pts))})
        report["lidar"] = near
        for item in near:
            print(json.dumps(item))
    (ds / "check_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {ds / 'check_report.json'}")


if __name__ == "__main__":
    main()
