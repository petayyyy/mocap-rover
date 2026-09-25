#!/usr/bin/env python3
"""Generate a printable/simulation fiducial texture from OpenCV."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization_contracts.marker_families import normalize_marker_family

parser = argparse.ArgumentParser()
parser.add_argument("--family", default="tag36h11")
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--output")
args = parser.parse_args()
family = normalize_marker_family(args.family)
if not 0 <= args.id < 50:
    parser.error("--id must be in [0, 49] for the supported dictionaries")
root = Path(__file__).resolve().parents[1]
out = Path(args.output) if args.output else root / "models/mocap-rover/materials/textures" / f"{family}_{args.id}.png"
out.parent.mkdir(parents=True, exist_ok=True)
dictionary_id = cv2.aruco.DICT_APRILTAG_36h11 if family == "tag36h11" else cv2.aruco.DICT_4X4_50
dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
marker = np.zeros((1024, 1024), dtype=np.uint8)
(getattr(cv2.aruco, "drawMarker", None) or cv2.aruco.generateImageMarker)(
    dictionary, args.id, 1024, marker, 1
)
texture = 255 * np.ones((1200, 1200), dtype=np.uint8)
texture[88:1112, 88:1112] = marker
if not cv2.imwrite(str(out), texture):
    raise RuntimeError(f"failed to write {out}")
print(out)
