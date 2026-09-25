#!/usr/bin/env python3
"""Reject a camera calibration whose distortion model is invalid inside the frame.

Run this before any localization run.  A model that cannot represent its own
image corners does not fail loudly: ``cv2.undistortPoints`` diverges, PnP
returns poses with hundreds of pixels of reprojection error, and the pipeline
just stops producing measurements away from the image centre.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization_contracts.camera_model import CameraModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="camera configuration JSON")
    parser.add_argument("--csv", help="write the per-camera report here")
    parser.add_argument("--min-valid-area", type=float, default=0.99)
    parser.add_argument("--max-round-trip-p95-px", type=float, default=0.05)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    rows = []
    for camera in config["cameras"]:
        model = CameraModel.from_config(camera)
        report = model.validate(args.min_valid_area, args.max_round_trip_p95_px)
        rows.append({"camera_id": camera.get("name", "?"), **asdict(report),
                     "reasons": "; ".join(report.reasons)})
        print(
            f"{rows[-1]['camera_id']:10} {report.model:10} "
            f"reach={report.max_valid_radius:.3f} needs={report.frame_corner_radius:.3f} "
            f"valid_area={report.valid_area_fraction*100:5.1f}% "
            f"roundtrip_p95={report.round_trip_p95_px:.3f}px  "
            f"{'PASS' if report.ok else 'FAIL: ' + rows[-1]['reasons']}"
        )
    if args.csv:
        with Path(args.csv).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0 if all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
