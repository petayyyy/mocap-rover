"""Camera foreground for a rover without a marker: background model + silhouette.

No learning anywhere.  What is known in advance is the empty arena (a short
clip from every camera), the opponent's measured size, and one rectangle an
operator draws around it before the match.

``BackgroundModel`` is the swappable part: anything that can answer "which
pixels of this window differ from the empty arena" and "learn from this frame
outside these windows".  ``ClipBackground`` is the CPU implementation used
here -- a per-pixel mean of the empty-arena clip, evaluated only inside the
planned windows.  A CUDA MOG2 model would implement the same two methods.

``SilhouetteObserver`` turns a foreground window into a metric position with
the same ray-plane geometry the marker uses: blob pixels are intersected with
the plane of the body top, where the silhouette of a convex body seen from a
point is the convex hull of its projected corners.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .ray_plane import pixels_to_plane


def in_masked_zone(x, y, zones, margin=0.0):
    """Which arena points lie in the masked zones (br_lidar's in_masked_zone).

    Zones are floor areas in arena metres: ``("rect", x0, y0, x1, y1)`` or
    ``("circle", cx, cy, r)``.  Masking by floor area rather than by ray keeps
    a rover standing in front of an obstacle visible.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    for zone in zones or ():
        kind = zone[0]
        if kind == "rect":
            _, x0, y0, x1, y1 = zone
            inside |= ((x >= x0 - margin) & (x <= x1 + margin)
                       & (y >= y0 - margin) & (y <= y1 + margin))
        elif kind == "circle":
            _, cx, cy, radius = zone
            inside |= (x - cx) ** 2 + (y - cy) ** 2 <= (radius + margin) ** 2
        else:
            raise ValueError(f"unknown zone kind {kind!r}")
    return inside


class BackgroundModel:
    """Interface: foreground inside a window, learning outside the tracks."""

    def gain(self, image, exclude=()):
        """Global brightness of ``image`` relative to the model."""
        raise NotImplementedError

    def foreground(self, image, roi, gain=1.0):
        """Boolean mask of the window ``roi = (x, y, w, h)``."""
        raise NotImplementedError

    def update(self, image, exclude=(), gain=1.0):
        """Learn from this frame, never inside the ``exclude`` windows."""
        raise NotImplementedError


class ClipBackground(BackgroundModel):
    """Per-pixel mean of an empty-arena clip, evaluated only inside windows.

    Lighting can change during a match, so the frame is compared after
    dividing out a global gain: the median ratio of frame to model over a
    sparse grid of pixels outside the track windows.  A rover covers a few
    percent of the frame, so the median is the arena's own brightness.

    ``update`` blends a band of rows per call (round robin) into the model,
    skipping the track windows: debris that stops moving becomes background
    in ``1/alpha`` passes, a rover standing inside its own window never does.
    """

    def __init__(self, mean, *, threshold=12.0, alpha=0.02, band_rows=32,
                 gain_step_px=16, pixel_mask=None, open_px=3, close_px=5):
        self.mean = np.asarray(mean, dtype=np.float32)
        if self.mean.ndim != 3 or self.mean.shape[2] != 3:
            raise ValueError("background mean must be HxWx3")
        self.height, self.width = self.mean.shape[:2]
        # uint8 copy for the per-window comparison: cv2.absdiff on bytes is an
        # order of magnitude cheaper than float arithmetic on a 500 px window.
        self.mean_u8 = np.clip(np.rint(self.mean), 0, 255).astype(np.uint8)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.band_rows = int(band_rows)
        self.gain_step_px = int(gain_step_px)
        # Pixels that never report foreground: masked arena zones, or anything
        # outside the arena the operator excluded.  None = every pixel counts.
        self.pixel_mask = None if pixel_mask is None else np.asarray(pixel_mask, dtype=bool)
        self.open_kernel = np.ones((open_px, open_px), np.uint8) if open_px else None
        self.close_kernel = np.ones((close_px, close_px), np.uint8) if close_px else None
        self.next_band = 0
        ys = np.arange(gain_step_px // 2, self.height, gain_step_px)
        xs = np.arange(gain_step_px // 2, self.width, gain_step_px)
        self.grid_y, self.grid_x = np.meshgrid(ys, xs, indexing="ij")
        grid = self.mean[self.grid_y, self.grid_x].mean(axis=2)
        # Ratios off near-black pixels are noise.
        self.grid_valid = grid > 12.0
        if self.pixel_mask is not None:
            self.grid_valid &= ~self.pixel_mask[self.grid_y, self.grid_x]
        self.grid_mean = grid

    @classmethod
    def from_frames(cls, frames, **kwargs):
        total, count = None, 0
        for frame in frames:
            frame = np.asarray(frame, dtype=np.float32)
            total = frame.copy() if total is None else total + frame
            count += 1
        if not count:
            raise ValueError("no frames for the background")
        return cls(total / count, **kwargs)

    @staticmethod
    def _outside(shape_yx, ys, xs, exclude):
        keep = np.ones(ys.shape, dtype=bool)
        for x, y, w, h in exclude or ():
            keep &= ~((xs >= x) & (xs < x + w) & (ys >= y) & (ys < y + h))
        return keep

    def gain(self, image, exclude=()):
        sample = np.asarray(image)[self.grid_y, self.grid_x].astype(np.float32).mean(axis=2)
        keep = self.grid_valid & self._outside(None, self.grid_y, self.grid_x, exclude)
        if keep.sum() < 50:
            return 1.0
        return float(np.clip(np.median(sample[keep] / self.grid_mean[keep]), 0.2, 5.0))

    def foreground(self, image, roi, gain=1.0):
        x, y, w, h = (int(v) for v in roi)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        out = np.zeros((max(h, 0), max(w, 0)), dtype=bool)
        if x1 <= x0 or y1 <= y0:
            return out
        crop = np.ascontiguousarray(np.asarray(image)[y0:y1, x0:x1])
        if abs(gain - 1.0) > 1e-3:
            crop = cv2.convertScaleAbs(crop, alpha=1.0 / max(gain, 1e-3))
        diff = cv2.absdiff(crop, self.mean_u8[y0:y1, x0:x1])
        diff = cv2.max(cv2.max(diff[:, :, 0], diff[:, :, 1]), diff[:, :, 2])
        mask = (diff > self.threshold).astype(np.uint8)
        if self.pixel_mask is not None:
            mask[self.pixel_mask[y0:y1, x0:x1]] = 0
        if self.open_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        if self.close_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        out[y0 - y:y1 - y, x0 - x:x1 - x] = mask.astype(bool)
        return out

    def update(self, image, exclude=(), gain=1.0):
        if self.alpha <= 0:
            return
        r0 = self.next_band
        r1 = min(self.height, r0 + self.band_rows)
        self.next_band = 0 if r1 >= self.height else r1
        band = np.asarray(image)[r0:r1].astype(np.float32) * (1.0 / max(gain, 1e-3))
        ys, xs = np.mgrid[r0:r1, 0:self.width]
        keep = self._outside(None, ys, xs, exclude)
        if self.pixel_mask is not None:
            keep &= ~self.pixel_mask[r0:r1]
        target = self.mean[r0:r1]
        target[keep] += self.alpha * (band[keep] - target[keep])
        self.mean_u8[r0:r1] = np.clip(np.rint(target), 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class Silhouette:
    """One blob read on the plane of the body top, arena metres."""

    x: float
    y: float
    yaw: float                 # long axis, defined modulo pi
    length_m: float
    width_m: float
    sigma_m: float
    residual_m: float
    pixels: int
    incidence_rad: float
    estimator: str
    hull_px: tuple             # blob hull in pixels, for the tracker
    top_corners_px: tuple      # footprint corners on the top plane, for KLT
    covariance_xy: tuple = ()  # 2x2 row major, long axis along the camera ray


class SilhouetteObserver:
    """Foreground blob in a window -> position of a known-size body.

    Three centre estimators, selectable, so they can be compared on data:

    ``centroid``  mean of the blob pixels on the top plane;
    ``median``    br_lidar's robust centre: median, then again without the
                  points farther than ``reject_radius_m`` from it;
    ``fit``       the known footprint fitted to the silhouette: the convex
                  hull of the four top corners and the four ground corners
                  projected along the camera rays onto the top plane;
    ``extent``    the middle of the blob's extent along and across the
                  camera ray, on the plane at half the body height.

    Measured on dataset_imx219_01, ``extent`` is the one to use.  The real
    body is not the operator's cuboid: its deck is at 0.35 m, only a cabin
    reaches 0.483 m, and the wheels make it 0.71 m wide.  Density statistics
    (centroid, median) are pulled to the camera by the sides it sees, and a
    fit of the measured cuboid is pulled sideways by the wheels.  The
    silhouette's near edge is made by low points and its far edge by high
    ones, so on the half-height plane their offsets cancel to first order,
    and the middle of the extent does not care how many pixels a side has.
    """

    ESTIMATORS = ("centroid", "median", "fit", "extent")

    def __init__(self, camera_model, R_world_camera, camera_position, *,
                 size_m=(0.9, 0.52, 0.483), estimator="extent", min_pixels=150,
                 reject_radius_m=0.6, pixel_step=3, zones=None, max_extent_m=1.6,
                 extent_plane_z=None, trim_percentile=2.0):
        if estimator not in self.ESTIMATORS:
            raise ValueError(f"estimator must be one of {self.ESTIMATORS}")
        self.model = camera_model
        self.R = np.asarray(R_world_camera, dtype=float)
        self.C = np.asarray(camera_position, dtype=float)
        self.length, self.width, self.top = (float(v) for v in size_m)
        self.estimator = estimator
        self.min_pixels = int(min_pixels)
        self.reject_radius_m = float(reject_radius_m)
        self.pixel_step = int(pixel_step)
        self.zones = zones
        self.max_extent_m = float(max_extent_m)
        # Ground corners seen along the camera ray land on the top plane
        # scaled towards the point under the camera by this factor.
        self.shrink = (self.C[2] - self.top) / self.C[2]
        self.extent_plane_z = (self.top / 2.0 if extent_plane_z is None
                               else float(extent_plane_z))
        self.trim_percentile = float(trim_percentile)
        self.last_reason = None

    # ---------------------------------------------------------------- model

    def corners(self, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        half = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=float) \
            * [self.length / 2, self.width / 2]
        return half @ np.array([[c, s], [-s, c]]) + [x, y]

    def silhouette(self, x, y, yaw):
        """Model silhouette on the top plane: hull of top and projected ground corners."""
        top = self.corners(x, y, yaw)
        ground = self.C[:2] + (top - self.C[:2]) * self.shrink
        points = np.vstack([top, ground]).astype(np.float32)
        return cv2.convexHull(points).reshape(-1, 2).astype(float)

    @staticmethod
    def _to_polygon(points, polygon):
        """Distance from each point to the boundary of a convex polygon."""
        a = polygon
        b = np.roll(polygon, -1, axis=0)
        ab = b - a
        t = np.clip(((points[:, None, :] - a[None]) * ab[None]).sum(axis=2)
                    / np.maximum((ab * ab).sum(axis=1), 1e-12)[None], 0.0, 1.0)
        nearest = a[None] + t[..., None] * ab[None]
        return np.sqrt(((points[:, None, :] - nearest) ** 2).sum(axis=2)).min(axis=1)

    @staticmethod
    def _resample(polygon, count=32):
        """``count`` points evenly spaced along a closed polygon's perimeter."""
        closed = np.vstack([polygon, polygon[:1]])
        steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        along = np.concatenate([[0.0], np.cumsum(steps)])
        if along[-1] <= 1e-9:
            return np.repeat(polygon[:1], count, axis=0)
        targets = np.linspace(0.0, along[-1], count, endpoint=False)
        return np.column_stack([np.interp(targets, along, closed[:, k]) for k in (0, 1)])

    # -------------------------------------------------------------- measure

    def _reject(self, reason):
        self.last_reason = reason
        return None

    def measure(self, mask, roi, prediction_xy, gate_m=1.0, frame_size=None):
        """Read the blob nearest the prediction; None with ``last_reason`` if not."""
        x0, y0 = int(roi[0]), int(roi[1])
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        best = None
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_pixels:
                continue
            bx, by, bw, bh = (int(v) for v in stats[label, :4])
            ys, xs = np.nonzero(labels[by:by + bh, bx:bx + bw] == label)
            step = self.pixel_step
            pick = (ys % step == 0) & (xs % step == 0)
            pixels = np.column_stack([xs[pick] + bx + x0, ys[pick] + by + y0]).astype(float)
            if len(pixels) < 10:
                continue
            points, valid = pixels_to_plane(self.model, self.R, self.C, pixels, self.top)
            points = points[valid, :2]
            if len(points) < 10:
                continue
            if self.zones:
                points = points[~in_masked_zone(points[:, 0], points[:, 1], self.zones)]
                if len(points) < 10:
                    continue
            centre = np.median(points, axis=0)
            distance = float(np.linalg.norm(centre - np.asarray(prediction_xy, dtype=float)))
            if distance > gate_m:
                continue
            touches = (bx == 0 or by == 0 or bx + bw >= mask.shape[1]
                       or by + bh >= mask.shape[0])
            if best is None or distance < best[0]:
                best = (distance, label, points, (bx, by, bw, bh), touches, area, pixels)
        if best is None:
            return self._reject("no_blob_in_gate")
        _, label, points, (bx, by, bw, bh), touches, area, pixels_all = best
        if touches:
            # The window cut the body: its outline is the window's, not the rover's.
            return self._reject("blob_clipped_by_window")
        contour_mask = (labels[by:by + bh, bx:bx + bw] == label).astype(np.uint8)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        outline = max(contours, key=len).reshape(-1, 2).astype(float) + [bx + x0, by + y0]
        hull_px = cv2.convexHull(outline.astype(np.float32)).reshape(-1, 2)
        edge, valid = pixels_to_plane(self.model, self.R, self.C, outline[::2], self.top)
        edge = edge[valid, :2]
        if len(edge) < 8:
            return self._reject("outline_off_plane")
        observed = cv2.convexHull(edge.astype(np.float32)).reshape(-1, 2).astype(float)

        # Orientation and size from the principal axes of the blob.
        centred = points - points.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov(centred.T))
        major = vectors[:, int(np.argmax(values))]
        yaw = math.atan2(major[1], major[0])
        minor = np.array([-major[1], major[0]])
        length = float(np.ptp(centred @ major))
        width = float(np.ptp(centred @ minor))
        if max(length, width) > self.max_extent_m:
            return self._reject("blob_too_large")

        along_sigma_scale = 0.10
        if self.estimator == "extent":
            flat, ok = pixels_to_plane(self.model, self.R, self.C, pixels_all,
                                       self.extent_plane_z)
            flat = flat[ok, :2]
            guess = np.median(flat, axis=0)
            radial = guess - self.C[:2]
            norm = float(np.linalg.norm(radial))
            radial = radial / norm if norm > 1e-6 else np.array([1.0, 0.0])
            across = np.array([-radial[1], radial[0]])
            low, high = self.trim_percentile, 100.0 - self.trim_percentile
            a = np.percentile(flat @ radial, [low, high])
            b = np.percentile(flat @ across, [low, high])
            centre = radial * a.mean() + across * b.mean()
            residual = 0.0
        elif self.estimator == "centroid":
            centre = points.mean(axis=0)
            residual = 0.0
        elif self.estimator == "median":
            centre = np.median(points, axis=0)
            far = np.linalg.norm(points - centre, axis=1) > self.reject_radius_m
            if far.any() and (~far).sum() >= 10:
                centre = np.median(points[~far], axis=0)
            residual = 0.0
        else:
            start = np.median(points, axis=0)

            samples = self._resample(observed)

            def cost(p):
                model = self.silhouette(p[0], p[1], p[2])
                return np.concatenate([self._to_polygon(samples, model),
                                       self._to_polygon(self._resample(model), observed)])
            fits = [least_squares(cost, [start[0], start[1], yaw + k * math.pi / 2],
                                  loss="soft_l1", f_scale=0.03, max_nfev=60)
                    for k in (0, 1)]
            fit = min(fits, key=lambda f: float(np.sum(f.fun ** 2)))
            centre = fit.x[:2]
            yaw = float(fit.x[2])
            residual = float(np.sqrt(np.mean(fit.fun ** 2)))
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
        ray = centre - self.C[:2]
        incidence = math.atan2(float(np.linalg.norm(ray)), self.C[2] - self.top)
        # The body top is not one plane (cabin 0.48 m, deck lower), so the
        # error along the viewing ray grows with tan(incidence); across it the
        # silhouette is sharp.  A 0.1 m height spread gives the along-ray term.
        # Across the ray the outline is sharp; along it the unknown height
        # of whatever forms the near and far edges moves the estimate by
        # about height_spread * tan(incidence).
        sigma_across = 0.02 + residual
        sigma_along = math.hypot(sigma_across, along_sigma_scale * math.tan(incidence))
        sigma = sigma_along
        ray_dir = centre - self.C[:2]
        ray_norm = float(np.linalg.norm(ray_dir))
        u = ray_dir / ray_norm if ray_norm > 1e-6 else np.array([1.0, 0.0])
        v = np.array([-u[1], u[0]])
        cov = sigma_along ** 2 * np.outer(u, u) + sigma_across ** 2 * np.outer(v, v)
        top = self.corners(centre[0], centre[1], yaw)
        top_px = None
        optical = (np.column_stack([top, np.full(4, self.top)]) - self.C) @ self.R
        if (optical[:, 2] > 1e-6).all():
            top_px = tuple(map(tuple, self.model.project(optical)))
        self.last_reason = "accepted"
        return Silhouette(float(centre[0]), float(centre[1]), float(yaw), length, width,
                          float(sigma), residual, int(area), float(incidence),
                          self.estimator, tuple(map(tuple, hull_px)), top_px,
                          tuple(float(c) for c in cov.reshape(-1)))
