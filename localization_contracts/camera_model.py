"""Projection model boundary with an explicit validity domain.

A Brown-Conrady radial polynomial is only usable while it is strictly
increasing.  Past its turning point no ray maps to a pixel any more, so
``cv2.undistortPoints`` diverges and ``solvePnP`` returns poses with hundreds
of pixels of reprojection error instead of failing.  On a 105 deg lens that
boundary can sit well inside the image, which silently deletes most of the
usable frame.  Every model built here therefore knows the largest normalized
radius it can represent and refuses to convert a pixel outside it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

PINHOLE = "pinhole"
PLUMB_BOB = "plumb_bob"
RATIONAL = "rational"
FISHEYE = "fisheye"
MODELS = (PINHOLE, PLUMB_BOB, RATIONAL, FISHEYE)


class CameraModelError(ValueError):
    """The calibration cannot represent the image it claims to describe."""


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment failure
        raise RuntimeError("CameraModel requires OpenCV") from exc
    return cv2


def infer_model(distortion) -> str:
    """Name the model implied by a raw distortion vector."""
    d = np.asarray(distortion, dtype=float).reshape(-1)
    if d.size == 0 or not np.any(d):
        return PINHOLE
    if d.size in (4,):
        return FISHEYE
    if d.size in (5,):
        return PLUMB_BOB
    if d.size in (8, 12, 14):
        return RATIONAL
    raise CameraModelError(f"unsupported distortion length {d.size}")


def radial_turnover(k1, k2, k3, limit=4.0, samples=40000):
    """First maximum of ``r_u (1 + k1 r^2 + k2 r^4 + k3 r^6)``.

    Returns ``(r_u, r_d)`` at the turning point, or ``(inf, inf)`` when the map
    is monotonic over ``limit``.
    """
    r = np.linspace(0.0, float(limit), int(samples))
    rd = r * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    falling = np.flatnonzero(np.diff(rd) < 0)
    if falling.size == 0:
        return math.inf, math.inf
    index = int(falling[0])
    return float(r[index]), float(rd[index])


@dataclass(frozen=True)
class ValidityReport:
    model: str
    max_valid_radius: float          # largest representable normalized radius
    frame_corner_radius: float       # radius the image corners actually need
    valid_area_fraction: float
    round_trip_p50_px: float
    round_trip_p95_px: float
    ok: bool
    reasons: tuple[str, ...]

    def raise_for_status(self):
        if not self.ok:
            raise CameraModelError(
                f"{self.model} calibration is invalid inside the frame: "
                + "; ".join(self.reasons)
            )
        return self


class CameraModel:
    """Intrinsics plus the domain on which they are meaningful."""

    def __init__(self, K, distortion, image_size, model=None, name=""):
        self.name = str(name)
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(distortion, dtype=np.float64).reshape(-1)
        if self.D.size == 0:
            self.D = np.zeros(5)
        self.width, self.height = (int(v) for v in image_size)
        self.model = str(model) if model else infer_model(self.D)
        if self.model not in MODELS:
            raise CameraModelError(f"unknown model {self.model!r}")
        if self.width <= 0 or self.height <= 0:
            raise CameraModelError("image_size must be positive")
        if not np.isfinite(self.K).all() or min(self.K[0, 0], self.K[1, 1]) <= 0:
            raise CameraModelError("invalid K")
        if not np.isfinite(self.D).all():
            raise CameraModelError("invalid distortion")
        self.max_valid_radius = self._max_valid_radius()

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_config(cls, camera):
        """Build from one entry of a cameras.json ``cameras`` list."""
        return cls(
            np.asarray(camera["K"], dtype=float).reshape(3, 3),
            camera.get("D", [0.0] * 5),
            camera["image_size"],
            model=camera.get("distortion_model"),
            name=camera.get("name", ""),
        )

    def _max_valid_radius(self):
        if self.model == PINHOLE:
            return math.inf
        if self.model == FISHEYE:
            # equidistant: theta_d is monotonic for sane coefficients up to the
            # lens half-angle; the projection itself never folds over.
            return math.inf
        if self.model == PLUMB_BOB:
            k1, k2, p1, p2, k3 = self.D[:5]
            return radial_turnover(k1, k2, k3)[1]
        # rational: no closed form, probe numerically in distorted space
        return self._probe_rational_radius()

    def _probe_rational_radius(self, limit=4.0, samples=4000):
        k1, k2, p1, p2, k3, k4, k5, k6 = (list(self.D[:8]) + [0.0] * 8)[:8]
        r = np.linspace(0.0, limit, samples)
        num = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6
        den = 1 + k4 * r**2 + k5 * r**4 + k6 * r**6
        with np.errstate(divide="ignore", invalid="ignore"):
            rd = r * num / den
        bad = ~np.isfinite(rd)
        falling = np.flatnonzero((np.diff(rd) < 0) | bad[1:])
        if falling.size == 0:
            return math.inf
        return float(rd[int(falling[0])])

    # -------------------------------------------------------------- geometry

    def frame_corner_radius(self):
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        corners = (
            (0.0, 0.0),
            (self.width - 1.0, 0.0),
            (0.0, self.height - 1.0),
            (self.width - 1.0, self.height - 1.0),
        )
        return max(math.hypot((u - cx) / fx, (v - cy) / fy) for u, v in corners)

    def pixel_radius_limit(self):
        """Largest pixel radius from the principal point that is representable."""
        if not math.isfinite(self.max_valid_radius):
            return math.inf
        return self.max_valid_radius * min(self.K[0, 0], self.K[1, 1])

    def inside_valid_radius(self, points, margin=1.0):
        """Boolean mask: are these pixels inside ``margin`` x the valid domain?"""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if not math.isfinite(self.max_valid_radius):
            return np.ones(len(pts), dtype=bool)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        r = np.hypot((pts[:, 0] - cx) / fx, (pts[:, 1] - cy) / fy)
        return r <= self.max_valid_radius * float(margin)

    def edge_distance_px(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        return np.minimum.reduce([
            pts[:, 0], pts[:, 1],
            self.width - 1.0 - pts[:, 0], self.height - 1.0 - pts[:, 1],
        ])

    def undistort(self, points):
        """Pixels -> normalized undistorted rays ``(x/z, y/z)``."""
        cv2 = _cv2()
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if self.model == FISHEYE:
            out = cv2.fisheye.undistortPoints(pts, self.K, self.D[:4].reshape(4, 1))
        elif self.model == PINHOLE:
            out = cv2.undistortPoints(pts, self.K, np.zeros(5))
        else:
            out = cv2.undistortPoints(pts, self.K, self.D)
        return np.asarray(out, dtype=float).reshape(-1, 2)

    def project(self, points_camera):
        """Camera-frame XYZ -> pixels."""
        cv2 = _cv2()
        pts = np.asarray(points_camera, dtype=np.float64).reshape(-1, 1, 3)
        zero = np.zeros(3)
        if self.model == FISHEYE:
            out, _ = cv2.fisheye.projectPoints(
                pts, zero, zero, self.K, self.D[:4].reshape(4, 1)
            )
        elif self.model == PINHOLE:
            out, _ = cv2.projectPoints(pts.reshape(-1, 3), zero, zero, self.K, np.zeros(5))
        else:
            out, _ = cv2.projectPoints(pts.reshape(-1, 3), zero, zero, self.K, self.D)
        return np.asarray(out, dtype=float).reshape(-1, 2)

    def solve_pnp_distortion(self):
        """Distortion vector to hand to ``cv2.solvePnP*``.

        The fisheye model is not accepted by ``solvePnP``; callers undistort
        first and then solve with zero distortion.
        """
        if self.model in (PINHOLE, FISHEYE):
            return np.zeros((5, 1))
        return self.D.reshape(-1, 1)

    # -------------------------------------------------------------- validity

    def validate(self, min_valid_area=0.99, max_round_trip_p95_px=0.05, step=16):
        """Measure whether this calibration covers its own image."""
        corner = self.frame_corner_radius()
        area = self._valid_area_fraction(step)
        p50, p95 = self._round_trip(step * 2)
        reasons = []
        if area < min_valid_area:
            reasons.append(
                f"only {area*100:.1f}% of the frame is inside the model domain "
                f"(needs r_d={corner:.3f}, model reaches {self.max_valid_radius:.3f})"
            )
        if p95 > max_round_trip_p95_px:
            reasons.append(f"undistort/project round trip P95 = {p95:.3f} px")
        return ValidityReport(
            self.model, self.max_valid_radius, corner, area, p50, p95,
            not reasons, tuple(reasons),
        )

    def _valid_area_fraction(self, step):
        if not math.isfinite(self.max_valid_radius):
            return 1.0
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        uu, vv = np.meshgrid(
            np.arange(0, self.width, step, dtype=float),
            np.arange(0, self.height, step, dtype=float),
        )
        r = np.hypot((uu - cx) / fx, (vv - cy) / fy)
        return float((r <= self.max_valid_radius).mean())

    def _round_trip(self, step):
        uu, vv = np.meshgrid(
            np.arange(0, self.width, step, dtype=float),
            np.arange(0, self.height, step, dtype=float),
        )
        pts = np.stack([uu.ravel(), vv.ravel()], axis=1)
        rays = self.undistort(pts)
        back = self.project(np.concatenate([rays, np.ones((len(rays), 1))], axis=1))
        error = np.linalg.norm(back - pts, axis=1)
        error = error[np.isfinite(error)]
        if error.size == 0:
            return math.inf, math.inf
        return float(np.percentile(error, 50)), float(np.percentile(error, 95))
