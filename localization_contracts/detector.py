"""High-recall fiducial detector boundary for simulation and real capture.

Both AprilTag 36h11 and ArUco DICT_4X4_50 use OpenCV's ArUco API. It returns
pixel corners and IDs; pose estimation remains in :mod:`apriltag` and no
evaluator/ground truth is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .marker_families import normalize_marker_family


@dataclass(frozen=True)
class PixelDetection:
    tag_id: int
    corners: tuple[tuple[float, float], ...]
    decision_margin: float | None = None


class AprilTagImageDetector:
    """Compatibility name for a detector supporting both marker families."""

    def __init__(self, family="tag36h11", border_bits=1, scale=1.0,
                 profile="coverage", allowed_ids=None):
        self.family = normalize_marker_family(family)
        self.border_bits = int(border_bits)
        if profile not in {"coverage", "balanced", "fast"}:
            raise ValueError("profile must be coverage, balanced or fast")
        self.profile = profile
        self.allowed_ids = None if allowed_ids is None else frozenset(map(int, allowed_ids))
        if not 0 < scale <= 1:
            raise ValueError("scale must be in (0,1]")
        self.scale = float(scale)
        try:
            import cv2

            self._cv2 = cv2
            dictionary_id = (
                cv2.aruco.DICT_APRILTAG_36h11
                if self.family == "tag36h11"
                else cv2.aruco.DICT_4X4_50
            )
            self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "OpenCV ArUco/AprilTag dictionary backend is unavailable"
            ) from exc
        aruco = self._cv2.aruco
        self._parameters = (
            aruco.DetectorParameters_create()
            if hasattr(aruco, "DetectorParameters_create")
            else aruco.DetectorParameters()
        )
        self._parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # Coverage profile deliberately admits small/weak candidates. Metric
        # confidence is handled downstream by PnP checks and covariance.
        for name, value in (
            # Below 0.015 noisy 640x480 frames create thousands of tiny quads
            # and destroy effective FPS without recovering useful markers.
            ("minMarkerPerimeterRate", 0.015),
            ("maxMarkerPerimeterRate", 4.0),
            ("polygonalApproxAccuracyRate", 0.05),
            ("minCornerDistanceRate", 0.015),
            ("minDistanceToBorder", 1),
            ("errorCorrectionRate", 0.8 if profile == "coverage" else 0.6),
            ("maxErroneousBitsInBorderRate", 0.5),
            ("detectInvertedMarker", True),
            ("markerBorderBits", self.border_bits),
            ("perspectiveRemovePixelPerCell", 8 if profile == "coverage" else 4),
            ("perspectiveRemoveIgnoredMarginPerCell", 0.13),
            ("minOtsuStdDev", 2.0 if profile == "coverage" else 5.0),
            ("cornerRefinementWinSize", 5),
            ("cornerRefinementMaxIterations", 40),
            ("cornerRefinementMinAccuracy", 0.01),
            ("adaptiveThreshWinSizeMin", 3),
            ("adaptiveThreshWinSizeMax", 61 if profile == "coverage" else 39),
            ("adaptiveThreshWinSizeStep", 4),
            ("aprilTagQuadDecimate", 1.0),
            ("aprilTagQuadSigma", 0.0),
            ("aprilTagMinClusterPixels", 3),
            ("aprilTagMaxNmaxima", 15),
            ("aprilTagMinWhiteBlackDiff", 3),
            ("aprilTagDeglitch", 1 if profile == "coverage" else 0),
            # ArUco3 is faster for large markers but OpenCV's implementation
            # drops the 12--32 px markers that dominate the coverage boundary.
            ("useAruco3Detection", False),
        ):
            if hasattr(self._parameters, name):
                setattr(self._parameters, name, value)
        self._modern = (
            aruco.ArucoDetector(self._dictionary, self._parameters)
            if hasattr(aruco, "ArucoDetector")
            else None
        )

    @property
    def backend(self):
        name = "apriltag-36h11" if self.family == "tag36h11" else "4x4-50"
        return "opencv-aruco-" + name + "-" + ("modern" if self._modern else "legacy")

    def _variants(self, array):
        yield array
        if self.profile == "fast":
            return
        smoothed = self._cv2.GaussianBlur(array, (3, 3), 0)
        if self.profile == "coverage":
            clahe = self._cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(smoothed)
            yield clahe
        yield self._cv2.threshold(
            smoothed, 0, 255, self._cv2.THRESH_BINARY + self._cv2.THRESH_OTSU
        )[1]

    def detect(self, image) -> tuple[PixelDetection, ...]:
        array = np.asarray(image)
        if array.ndim not in (2, 3) or array.size == 0:
            raise ValueError("image must be a non-empty grayscale or BGR array")
        if array.ndim == 3:
            array = self._cv2.cvtColor(array, self._cv2.COLOR_RGB2GRAY)
        original = array
        if self.scale != 1.0:
            array = self._cv2.resize(
                array,
                None,
                fx=self.scale,
                fy=self.scale,
                interpolation=self._cv2.INTER_AREA,
            )
        candidates = []
        for pass_index, candidate in enumerate(self._variants(array)):
            corners, ids, rejected = (
                self._modern.detectMarkers(candidate)
                if self._modern
                else self._cv2.aruco.detectMarkers(
                    candidate, self._dictionary, parameters=self._parameters
                )
            )
            if ids is not None:
                for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
                    marker_id = int(marker_id)
                    if self.allowed_ids is None or marker_id in self.allowed_ids:
                        candidates.append((marker_corners, marker_id))
            # Extra preprocessing is fallback-only. Running every pass after a
            # valid ID reduces temporal coverage without improving its corners.
            if candidates:
                break
            # If the raw image contains no marker-like quads, enhancement only
            # amplifies noise and costs most of a frame period. Reprocess only
            # when OpenCV saw a plausible rejected candidate.
            if pass_index == 0 and not rejected:
                break
        if not candidates:
            return ()
        # Multiple preprocessing passes often recover the same marker. Keep the
        # largest candidate per configured ID and refine it on the original.
        best = {}
        for marker_corners, marker_id in candidates:
            points = marker_corners.reshape(4, 2) / self.scale
            perimeter = float(sum(np.linalg.norm(points[(i + 1) % 4] - points[i]) for i in range(4)))
            if marker_id not in best or perimeter > best[marker_id][0]:
                best[marker_id] = (perimeter, points)
        out = []
        for marker_id, (_, points) in sorted(best.items()):
            try:
                points = self._cv2.cornerSubPix(
                    original,
                    points.astype(np.float32).reshape(-1, 1, 2),
                    (5, 5),
                    (-1, -1),
                    (
                        self._cv2.TERM_CRITERIA_EPS | self._cv2.TERM_CRITERIA_MAX_ITER,
                        40,
                        0.01,
                    ),
                ).reshape(4, 2)
            except self._cv2.error:
                pass
            out.append(
                PixelDetection(
                    marker_id, tuple((float(x), float(y)) for x, y in points)
                )
            )
        return tuple(out)
