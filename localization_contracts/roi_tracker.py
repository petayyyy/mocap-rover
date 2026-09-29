"""Two-mode per-camera detection: full-frame acquisition, then a small ROI.

The recorded runtime ran the full detector on every frame of all six cameras
whether or not the rover could possibly be in view, which cost 65 ms per frame
and dropped 35% of the frames outright.  Once a track exists its position is
known to a few centimetres, so the marker can only be in a small part of each
image, and the detector only has to look there.

A camera the prediction does not reach at all drops to a slow watchdog rate
rather than being switched off: a track can be wrong, and a camera that never
looks can never say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ACQUIRE = "acquire"       # full frame, looking for the marker
ROI = "roi"               # small window around the prediction
IDLE = "idle"             # prediction is not in this camera's view
WATCHDOG = "watchdog"     # occasional full frame while idle


@dataclass(frozen=True)
class Plan:
    mode: str
    roi: tuple | None = None          # (x, y, w, h) in full-image pixels
    projected: tuple | None = None    # predicted marker centre, pixels
    reason: str = ""


def project_to_image(point_world, camera_model, R_world_optical, position_world):
    """Return pixel coordinates, or None when the point is behind the camera."""
    point = np.asarray(point_world, dtype=float).reshape(3)
    R = np.asarray(R_world_optical, dtype=float).reshape(3, 3)
    optical = R.T @ (point - np.asarray(position_world, dtype=float).reshape(3))
    if optical[2] <= 1e-6:
        return None
    return tuple(camera_model.project(optical.reshape(1, 3))[0])


class CameraRoiPlanner:
    """Decide what one camera should look at for the next frame."""

    def __init__(self, camera_model, R_world_optical, position_world, *,
                 min_roi_px=160, max_roi_px=480, sigma_multiplier=3.0,
                 marker_size_m=0.40, growth=1.5, max_misses=8,
                 valid_radius_margin=0.85, watchdog_period_s=2.0,
                 tag_plane_z=0.3654, max_incidence_deg=None, min_marker_px=None,
                 exhausted_full_frame_period_s=None):
        self.camera_model = camera_model
        self.R = np.asarray(R_world_optical, dtype=float).reshape(3, 3)
        self.position = np.asarray(position_world, dtype=float).reshape(3)
        self.min_roi_px = int(min_roi_px)
        self.max_roi_px = int(max_roi_px)
        self.sigma_multiplier = float(sigma_multiplier)
        self.marker_size_m = float(marker_size_m)
        self.growth = float(growth)
        self.max_misses = int(max_misses)
        self.valid_radius_margin = float(valid_radius_margin)
        self.watchdog_period_ns = int(watchdog_period_s * 1e9)
        # A wide lens puts almost the whole arena "in view", including where
        # the marker is too oblique or too small for the detector to read.
        # There a window only collects misses, exhausts, and every later frame
        # pays for a full-frame search that cannot succeed: on the IMX219
        # corner cameras that was 50-70 % of all frames, nearly all with the
        # rover beyond 55 degrees of incidence.  Both bounds are the
        # observer's own acceptance gates, so nothing it could accept is
        # skipped.  None keeps the old behaviour.
        self.max_incidence_rad = (None if max_incidence_deg is None
                                  else math.radians(float(max_incidence_deg)))
        self.min_marker_px = None if min_marker_px is None else float(min_marker_px)
        # After the window is exhausted, search the full frame at most this
        # often and keep the widest window in between, instead of a full
        # frame on every frame.  None: full frame every frame, as before.
        self.exhausted_period_ns = (None if exhausted_full_frame_period_s is None
                                    else int(exhausted_full_frame_period_s * 1e9))
        self.misses = 0
        self.last_full_frame_ns = None
        # Scalar, or (low, high) when the marker can sit on more than one
        # plane.  See _plane_bounds.
        self.tag_plane_z = tag_plane_z

    def report(self, found):
        """Feed back whether the last planned look succeeded."""
        self.misses = 0 if found else self.misses + 1

    def _plane_bounds(self):
        """The lowest and highest plane the marker may lie on.

        The rover carries a marker on top and another underneath, so flipping
        it moves the visible marker down by the height of the rover.  Aiming
        at the top plane while the underside marker is showing displaces the
        window by the plane error times the tangent of the incidence: up to
        50 px over this arena, against a worst-case margin of 17 px at the
        default 160 px floor.  That survives on padding alone -- lower the
        floor to 96 px with a confident track and 15% of the sampled views
        clip the marker instead.  Covering both planes takes the worst margin
        to 37 px and makes it independent of the floor.  It is close to free:
        for a camera looking straight down the two projections coincide and
        the window is unchanged.
        """
        value = self.tag_plane_z
        if np.isscalar(value):
            return float(value), float(value)
        low, high = (float(v) for v in value)
        return (low, high) if low <= high else (high, low)

    def _marker_size_px(self, optical_z, incidence):
        focal = float(min(self.camera_model.K[0, 0], self.camera_model.K[1, 1]))
        return focal * self.marker_size_m * max(math.cos(incidence), 0.2) / max(optical_z, 1e-6)

    def plan(self, prediction, now_ns=None):
        """``prediction`` is (x, y, sigma_m) in arena metres, or None."""
        if prediction is None:
            self.last_full_frame_ns = now_ns
            return Plan(ACQUIRE, None, None, "no_track")
        x, y, sigma = prediction
        corners = []
        for tag_z in set(self._plane_bounds()):
            uv = project_to_image((x, y, tag_z), self.camera_model, self.R, self.position)
            if uv is None:
                return self._idle_or_watchdog(now_ns, "behind_camera")
            corners.append((tag_z, uv))
        width, height = self.camera_model.width, self.camera_model.height

        def visible(uv):
            return (self.camera_model.inside_valid_radius(
                        [uv], self.valid_radius_margin)[0]
                    and 0 <= uv[0] < width and 0 <= uv[1] < height)

        # One plane leaving the frame is not a reason to stop looking: the
        # marker may be on the other one.
        if not any(visible(uv) for _, uv in corners):
            return self._idle_or_watchdog(now_ns, "outside_frame")
        if self.max_incidence_rad is not None:
            # Incidence at the camera-nearest edge of the uncertainty disc,
            # so a track that may be closer than predicted is still looked at.
            reach = max(0.0, math.hypot(x - self.position[0], y - self.position[1])
                        - 2.0 * float(sigma))
            if all(math.atan2(reach, abs(self.position[2] - tag_z)) > self.max_incidence_rad
                   for tag_z, _ in corners):
                return self._idle_or_watchdog(now_ns, "beyond_incidence")
        if self.min_marker_px is not None:
            if max(self._projected_marker_px(x, y, tag_z) for tag_z, _ in corners) \
                    < self.min_marker_px:
                return self._idle_or_watchdog(now_ns, "marker_too_small")

        us = [uv[0] for _, uv in corners]
        vs = [uv[1] for _, uv in corners]
        uv = ((min(us) + max(us)) / 2.0, (min(vs) + max(vs)) / 2.0)
        # Half the span between the two projections, so a window centred
        # between them still reaches either one.
        spread = max(max(us) - min(us), max(vs) - min(vs)) / 2.0

        focal = float(min(self.camera_model.K[0, 0], self.camera_model.K[1, 1]))
        half = 0.0
        for tag_z, _ in corners:
            optical = self.R.T @ (np.array([x, y, tag_z]) - self.position)
            ray = np.array([x, y, tag_z]) - self.position
            incidence = math.acos(min(1.0, abs(ray[2]) / max(np.linalg.norm(ray), 1e-9)))
            marker_px = self._marker_size_px(optical[2], incidence)
            # Position uncertainty projects to pixels through the same focal
            # length; the marker itself has to fit inside the window as well.
            sigma_px = focal * float(sigma) / max(optical[2], 1e-6)
            # The nearer plane makes the marker larger, so it sets the window.
            half = max(half, self.sigma_multiplier * sigma_px + marker_px)
        base = half + spread
        half = base * self.growth ** min(self.misses, 6)
        size = int(min(max(2 * half, self.min_roi_px), self.max_roi_px))
        size += size % 2
        if self.misses >= self.max_misses:
            if (self.exhausted_period_ns is None or now_ns is None
                    or self.last_full_frame_ns is None
                    or now_ns - self.last_full_frame_ns >= self.exhausted_period_ns):
                self.last_full_frame_ns = now_ns
                return Plan(ACQUIRE, None, uv, "roi_exhausted")
            # Growing further has already failed eight times; between the
            # periodic full-frame searches keep the window the prediction
            # alone implies, which costs a fraction of the widest one and
            # still catches the marker once it becomes readable again.
            size = int(min(max(2 * base, self.min_roi_px), self.max_roi_px))
            size += size % 2
            roi = (int(round(uv[0] - size / 2)), int(round(uv[1] - size / 2)), size, size)
            return Plan(ROI, roi, uv, "roi_exhausted_base")
        roi = (int(round(uv[0] - size / 2)), int(round(uv[1] - size / 2)), size, size)
        return Plan(ROI, roi, uv, "prediction_in_view")

    def _projected_marker_px(self, x, y, tag_z):
        """Apparent marker side through the actual lens, mean of both axes."""
        half = self.marker_size_m / 2.0
        sides = []
        for dx, dy in ((half, 0.0), (0.0, half)):
            a = project_to_image((x - dx, y - dy, tag_z), self.camera_model, self.R, self.position)
            b = project_to_image((x + dx, y + dy, tag_z), self.camera_model, self.R, self.position)
            if a is None or b is None:
                return 0.0
            sides.append(math.hypot(b[0] - a[0], b[1] - a[1]))
        return float(np.mean(sides))

    def _idle_or_watchdog(self, now_ns, reason):
        """A camera the prediction misses still checks in occasionally."""
        if now_ns is None:
            return Plan(IDLE, None, None, reason)
        if (self.last_full_frame_ns is None
                or now_ns - self.last_full_frame_ns >= self.watchdog_period_ns):
            self.last_full_frame_ns = now_ns
            return Plan(WATCHDOG, None, None, reason + "_watchdog")
        return Plan(IDLE, None, None, reason)


class KltMarkerTracker:
    """Follow the marker corners between detections with pyramidal LK flow.

    Flow drifts, so it is deliberately short-lived: it only bridges the gap to
    the next detection, and after ``max_frames`` without one it gives up and
    lets the ROI grow instead of quietly reporting a pose it invented.
    """

    def __init__(self, max_frames=8, max_reverse_error_px=0.5,
                 max_shape_change=0.15, window=(15, 15), levels=3):
        import cv2
        self.cv2 = cv2
        self.max_frames = int(max_frames)
        self.max_reverse_error_px = float(max_reverse_error_px)
        self.max_shape_change = float(max_shape_change)
        self.window = tuple(window)
        self.levels = int(levels)
        self.previous = None
        self.corners = None
        self.reference_sides = None
        self.frames = 0

    def start(self, image, corners):
        self.previous = np.asarray(image)
        self.corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        self.reference_sides = self._sides(self.corners)
        self.frames = 0

    def stop(self):
        self.previous = self.corners = self.reference_sides = None
        self.frames = 0

    @property
    def active(self):
        return self.corners is not None

    @staticmethod
    def _sides(corners):
        return np.array([np.linalg.norm(corners[(i + 1) % 4] - corners[i])
                         for i in range(4)])

    def track(self, image):
        """Return the corners in this frame, or None when flow is untrustworthy."""
        if self.corners is None:
            return None
        if self.frames >= self.max_frames:
            self.stop()
            return None
        cv2 = self.cv2
        image = np.asarray(image)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous, image, self.corners.reshape(-1, 1, 2), None,
            winSize=self.window, maxLevel=self.levels, criteria=criteria)
        if forward is None or status is None or int(status.sum()) < 4:
            self.stop()
            return None
        backward, _, _ = cv2.calcOpticalFlowPyrLK(
            image, self.previous, forward, None,
            winSize=self.window, maxLevel=self.levels, criteria=criteria)
        if backward is None:
            self.stop()
            return None
        reverse_error = np.linalg.norm(
            backward.reshape(4, 2) - self.corners, axis=1).max()
        if reverse_error > self.max_reverse_error_px:
            self.stop()
            return None
        corners = forward.reshape(4, 2)
        sides = self._sides(corners)
        change = np.abs(sides - self.reference_sides) / np.maximum(self.reference_sides, 1e-6)
        if change.max() > self.max_shape_change or not self._convex(corners):
            self.stop()
            return None
        self.previous = image
        self.corners = corners
        self.frames += 1
        return corners

    @staticmethod
    def _convex(corners):
        signs = []
        for i in range(4):
            a = corners[(i + 1) % 4] - corners[i]
            b = corners[(i + 2) % 4] - corners[(i + 1) % 4]
            signs.append(np.sign(a[0] * b[1] - a[1] * b[0]))
        return len(set(signs)) == 1 and signs[0] != 0
