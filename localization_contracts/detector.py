"""Image detector boundary for simulation.

This module uses OpenCV's legacy ArUco API only when the AprilTag dictionary
and detector entry points are present.  It returns pixel corners and IDs; pose
estimation remains in :mod:`apriltag` and no evaluator/ground truth is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PixelDetection:
    tag_id: int
    corners: tuple[tuple[float, float], ...]
    decision_margin: float | None = None


class AprilTagImageDetector:
    def __init__(self, family="tag36h11", border_bits=1, scale=1.0):
        if family != "tag36h11":
            raise ValueError("only tag36h11 is supported by the simulation backend")
        self.family = family
        self.border_bits = int(border_bits)
        if not 0 < scale <= 1: raise ValueError("scale must be in (0,1]")
        self.scale = float(scale)
        try:
            import cv2
            self._cv2 = cv2
            self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("OpenCV AprilTag dictionary backend is unavailable") from exc
        if not hasattr(self._cv2.aruco, "detectMarkers"):
            raise RuntimeError("OpenCV legacy marker detector is unavailable")

    @property
    def backend(self):
        return "opencv-aruco-apriltag-36h11-legacy"

    def detect(self, image) -> tuple[PixelDetection, ...]:
        array = np.asarray(image)
        if array.ndim not in (2, 3) or array.size == 0:
            raise ValueError("image must be a non-empty grayscale or BGR array")
        if array.ndim == 3:
            array = self._cv2.cvtColor(array, self._cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            array = self._cv2.resize(array, None, fx=self.scale, fy=self.scale, interpolation=self._cv2.INTER_AREA)
        candidates=[]
        for candidate in (array, self._cv2.threshold(array, 0, 255, self._cv2.THRESH_BINARY + self._cv2.THRESH_OTSU)[1]):
            corners, ids, _ = self._cv2.aruco.detectMarkers(candidate, self._dictionary)
            if ids is not None:
                candidates.extend(zip(corners, ids.reshape(-1)))
        if not candidates: return ()
        out = []
        seen=set()
        for marker_corners, marker_id in candidates:
            if int(marker_id) in seen: continue
            seen.add(int(marker_id))
            points = marker_corners.reshape(4, 2) / self.scale
            out.append(PixelDetection(int(marker_id), tuple((float(x), float(y)) for x, y in points)))
        return tuple(out)
