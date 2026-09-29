"""One camera's view of the rover without a marker.

Background in the planned window -> tag_rover's predicted body removed ->
silhouette nearest the opponent's prediction -> position on the body's
half-height plane.  When the blob cannot be read (cut by the window, merged,
too small) and the previous frame was, points inside the last blob are
followed with KLT and the last position is moved by their median motion on
the same plane.  Nothing here reads truth or learns from the rover.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .foreground import SilhouetteObserver
from .ray_plane import pixels_to_plane
from .roi_tracker import KltFeatureTracker

SILHOUETTE = "silhouette_extent"
FLOW = "klt_flow"


@dataclass(frozen=True)
class OpponentReading:
    method: str                 # SILHOUETTE or FLOW
    x: float
    y: float
    covariance_xy: tuple        # 2x2 row major
    yaw_axis: float | None      # long axis modulo pi, silhouettes only
    incidence_rad: float
    pixels: int
    detail: dict


def body_hull_px(camera_model, R, C, xy, yaw, size, margin_m=0.0):
    """Pixel hull of a box ``size = (l, w, h)`` standing at ``xy``; None if behind."""
    length, width, height = (float(v) for v in size)
    c, s = math.cos(yaw), math.sin(yaw)
    corners = np.array([[a * (length / 2 + margin_m), b * (width / 2 + margin_m), z]
                        for a in (-1, 1) for b in (-1, 1) for z in (0.0, height)])
    corners[:, :2] = corners[:, :2] @ np.array([[c, s], [-s, c]]) + np.asarray(xy[:2])
    optical = (corners - C) @ R
    if (optical[:, 2] <= 1e-6).any():
        return None
    return camera_model.project(optical)


class OpponentCamera:
    """Background, silhouette and flow for one camera; one worker thread uses it."""

    def __init__(self, camera_id, camera_model, R_world_camera, camera_position, background,
                 *, size_m=(0.9, 0.52, 0.483), tag_size_m=(0.72, 0.52, 0.40),
                 gate_m=0.8, exclusion_margin_m=0.12, klt_max_frames=8,
                 estimator="extent", zones=None):
        self.camera_id = camera_id
        self.model = camera_model
        self.R = np.asarray(R_world_camera, dtype=float)
        self.C = np.asarray(camera_position, dtype=float)
        self.background = background
        self.size = tuple(float(v) for v in size_m)
        self.tag_size = tuple(float(v) for v in tag_size_m)
        self.gate_m = float(gate_m)
        self.exclusion_margin_m = float(exclusion_margin_m)
        self.observer = SilhouetteObserver(camera_model, self.R, self.C, size_m=size_m,
                                           estimator=estimator, zones=zones)
        self.flow = KltFeatureTracker(max_frames=klt_max_frames)
        self.flow_plane_z = self.size[2] / 2.0
        self.last_xy = None
        # Features are only needed when a later frame fails, so the last good
        # crop and blob are kept and corners are searched for then.
        self.last_good = None

    def exclusion_rect(self, xy, yaw, size, margin_px=20):
        """Pixel rectangle around a predicted body, for background learning."""
        hull = body_hull_px(self.model, self.R, self.C, xy, yaw, size, 0.1)
        if hull is None:
            return None
        x0, y0 = np.floor(hull.min(axis=0)) - margin_px
        x1, y1 = np.ceil(hull.max(axis=0)) + margin_px
        return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))

    def _gray(self, image, rect):
        x, y, w, h = rect
        H, W = image.shape[:2]
        if x < 0 or y < 0 or x + w > W or y + h > H or w <= 0 or h <= 0:
            return None
        return cv2.cvtColor(np.ascontiguousarray(image[y:y + h, x:x + w]), cv2.COLOR_RGB2GRAY)

    def read(self, image, roi, prediction_xy, tag_pose=None, gain=1.0, gate_m=None):
        """One frame.  Returns (OpponentReading | None, diagnostic dict)."""
        begin = time.perf_counter()
        x, y, w, h = (int(v) for v in roi)
        H, W = image.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        roi = (x0, y0, min(W, x + w) - x0, min(H, y + h) - y0)
        mask = self.background.foreground(image, roi, gain)
        excluded = 0
        if tag_pose is not None:
            hull = body_hull_px(self.model, self.R, self.C, tag_pose[:2], tag_pose[2],
                                self.tag_size, self.exclusion_margin_m)
            if hull is not None:
                poly = np.round(cv2.convexHull(hull.astype(np.float32)).reshape(-1, 2)
                                - [roi[0], roi[1]]).astype(np.int32)
                cut = np.zeros(mask.shape, np.uint8)
                cv2.fillConvexPoly(cut, poly, 1)
                excluded = int((mask & cut.astype(bool)).sum())
                mask &= ~cut.astype(bool)
        fg_ms = (time.perf_counter() - begin) * 1e3
        found = self.observer.measure(mask, roi, prediction_xy, gate_m or self.gate_m)
        diag = {"foreground_px": int(mask.sum()), "tag_excluded_px": excluded,
                "gain": float(gain), "foreground_ms": fg_ms,
                "silhouette": self.observer.last_reason}
        if found is not None:
            self.last_xy = np.array([found.x, found.y])
            blob = np.zeros(mask.shape, np.uint8)
            hull_local = np.round(np.asarray(found.hull_px) - [roi[0], roi[1]]).astype(np.int32)
            cv2.fillConvexPoly(blob, hull_local, 1)
            gray = self._gray(image, roi)
            self.flow.stop()
            self.last_good = None if gray is None else (gray, roi, blob & mask.astype(np.uint8))
            diag["measure_ms"] = (time.perf_counter() - begin) * 1e3 - fg_ms
            return OpponentReading(SILHOUETTE, found.x, found.y, found.covariance_xy,
                                   found.yaw, found.incidence_rad, found.pixels,
                                   {"length_m": found.length_m, "width_m": found.width_m}), diag
        if not self.flow.active and self.last_good is not None:
            gray, rect, blob = self.last_good
            self.last_good = None
            self.flow.start(gray, rect, blob)
        if not self.flow.active or self.last_xy is None:
            return None, diag
        moved = self.flow.track(lambda rect: self._gray(image, rect))
        if moved is None:
            diag["flow"] = "lost"
            return None, diag
        old, new = moved
        before, ok0 = pixels_to_plane(self.model, self.R, self.C, old, self.flow_plane_z)
        after, ok1 = pixels_to_plane(self.model, self.R, self.C, new, self.flow_plane_z)
        ok = ok0 & ok1
        if ok.sum() < self.flow.min_points:
            self.flow.stop()
            diag["flow"] = "off_plane"
            return None, diag
        shift = np.median(after[ok, :2] - before[ok, :2], axis=0)
        self.last_xy = self.last_xy + shift
        # Drift grows with every frame flow bridges; start at 4 cm.
        sigma = 0.04 * math.sqrt(self.flow.frames)
        radial = self.last_xy - self.C[:2]
        incidence = math.atan2(float(np.linalg.norm(radial)), self.C[2] - self.flow_plane_z)
        diag["flow"] = f"frame_{self.flow.frames}"
        diag["measure_ms"] = (time.perf_counter() - begin) * 1e3 - fg_ms
        return OpponentReading(FLOW, float(self.last_xy[0]), float(self.last_xy[1]),
                               (sigma ** 2, 0.0, 0.0, sigma ** 2), None, incidence,
                               int(ok.sum()), {"frames": self.flow.frames}), diag
