#!/usr/bin/env python3
"""Generate the checked-in simulation AprilTag texture from OpenCV."""
from pathlib import Path
import cv2

out = Path(__file__).resolve().parents[1] / "models/mocap-rover/materials/textures/tag36h11_0.png"
out.parent.mkdir(parents=True, exist_ok=True)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
marker = cv2.aruco.drawMarker(dictionary, 0, 1024, borderBits=1)
if not cv2.imwrite(str(out), marker):
    raise RuntimeError(f"failed to write {out}")
print(out)
