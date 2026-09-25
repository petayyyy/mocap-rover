"""Identity-safe marker/lidar/camera tracking for a fast rover.

The marker is the only source allowed to create (or re-create) a track.  Once
created, metric lidar clusters and camera motion detections may keep it alive,
provided they agree with the timestamped prediction.  This prevents a moving
opponent or a shadow from silently becoming the tagged rover.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RoverMeasurement:
    stamp_ns: int
    source: str                    # marker, lidar, or camera_motion
    x: float
    y: float
    covariance_m2: tuple = (0.04, 0.04)
    confidence: float = 1.0
    identity: str | None = None


@dataclass(frozen=True)
class HybridRoverState:
    stamp_ns: int
    x: float
    y: float
    vx: float
    vy: float
    covariance_m2: tuple
    mode: str                      # tracking, coasting, lost
    last_source: str
    marker_age_s: float


class HybridRoverTracker:
    """Timestamp-driven constant-velocity Kalman tracker.

    ``marker`` measurements establish identity.  Lidar and camera motion are
    continuation-only observations and are rejected after a track is lost.
    """

    SOURCES = {"marker", "lidar", "camera_motion"}

    def __init__(self, acceleration_sigma=30.0, coast_ms=240,
                 min_confidence=0.15, max_speed_mps=13.5):
        self.acceleration_sigma = float(acceleration_sigma)
        self.coast_ns = int(coast_ms * 1_000_000)
        self.min_confidence = float(min_confidence)
        self.max_speed_mps = float(max_speed_mps)
        self.x = None
        self.P = None
        self.stamp_ns = None
        self.last_measurement_ns = None
        self.last_marker_ns = None
        self.last_source = "none"
        self.identity = None
        self.accepted = {s: 0 for s in self.SOURCES}
        self.rejected = {s: 0 for s in self.SOURCES}

    def _predict_in_place(self, stamp_ns):
        if self.x is None:
            return
        dt = (int(stamp_ns) - self.stamp_ns) / 1e9
        if dt < 0:
            raise ValueError("out-of-order timestamp")
        if dt == 0:
            return
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                      [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        q = self.acceleration_sigma ** 2
        Q = q * np.array([
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.stamp_ns = int(stamp_ns)

    def _is_lost(self, stamp_ns):
        return (self.last_measurement_ns is None or
                int(stamp_ns) - self.last_measurement_ns > self.coast_ns)

    def update(self, measurement):
        m = measurement
        if m.source not in self.SOURCES:
            raise ValueError(f"unsupported source: {m.source}")
        if (not np.isfinite([m.x, m.y, *m.covariance_m2]).all() or
                min(m.covariance_m2) <= 0 or m.confidence < self.min_confidence):
            self.rejected[m.source] += 1
            return False

        is_marker = m.source == "marker"
        if is_marker and self.identity is not None and m.identity is not None and m.identity != self.identity:
            self.rejected[m.source] += 1
            return False
        if self.x is None or self._is_lost(m.stamp_ns):
            if not is_marker:
                self.rejected[m.source] += 1
                return False
            self.x = np.array([m.x, m.y, 0.0, 0.0])
            self.P = np.diag([*m.covariance_m2, 9.0, 9.0]).astype(float)
            self.stamp_ns = self.last_measurement_ns = self.last_marker_ns = int(m.stamp_ns)
            self.last_source = m.source
            self.identity = m.identity or self.identity or "tagged_rover"
            self.accepted[m.source] += 1
            return True

        if int(m.stamp_ns) < self.stamp_ns:
            self.rejected[m.source] += 1
            return False
        self._predict_in_place(m.stamp_ns)

        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = np.diag(np.asarray(m.covariance_m2, dtype=float) /
                    max(float(m.confidence), 0.1))
        innovation = np.array([m.x, m.y]) - H @ self.x
        S = H @ self.P @ H.T + R
        mahalanobis2 = float(innovation @ np.linalg.solve(S, innovation))
        # Marker reacquisition is deliberately more permissive. Continuation
        # sources may only update the predicted rover-sized gate.
        gate2 = 25.0 if is_marker else 11.83  # 5 sigma vs 99.73% in 2-D
        if mahalanobis2 > gate2:
            self.rejected[m.source] += 1
            return False

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I = np.eye(4)
        # Joseph form remains positive semi-definite under floating point.
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        speed = float(np.linalg.norm(self.x[2:]))
        if speed > self.max_speed_mps:
            self.x[2:] *= self.max_speed_mps / speed
        self.last_measurement_ns = int(m.stamp_ns)
        self.last_source = m.source
        if is_marker:
            self.last_marker_ns = int(m.stamp_ns)
            self.identity = m.identity or self.identity or "tagged_rover"
        self.accepted[m.source] += 1
        return True

    def predict(self, stamp_ns):
        if self.x is None:
            return None
        self._predict_in_place(stamp_ns)
        age_ns = int(stamp_ns) - self.last_measurement_ns
        mode = "lost" if age_ns > self.coast_ns else ("coasting" if age_ns > 0 else "tracking")
        marker_age = ((int(stamp_ns) - self.last_marker_ns) / 1e9
                      if self.last_marker_ns is not None else math.inf)
        return HybridRoverState(
            int(stamp_ns), *map(float, self.x),
            (float(self.P[0, 0]), float(self.P[1, 1])), mode,
            self.last_source, marker_age,
        )


@dataclass(frozen=True)
class LidarCluster:
    x: float
    y: float
    z: float
    point_count: int
    distance_to_prediction_m: float
    covariance_m2: tuple


class LidarRoverDetector:
    """Find a rover-sized point cluster close to the timestamped prediction."""

    def __init__(self, z_limits=(0.04, 0.85), cluster_radius_m=0.20,
                 min_points=6, max_extent_m=1.6):
        self.z_limits = tuple(map(float, z_limits))
        self.cluster_radius_m = float(cluster_radius_m)
        self.min_points = int(min_points)
        self.max_extent_m = float(max_extent_m)

    def detect(self, points_world, predicted_xy, gate_radius_m):
        pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
        finite = np.isfinite(pts).all(axis=1)
        dz = (pts[:, 2] >= self.z_limits[0]) & (pts[:, 2] <= self.z_limits[1])
        dist = np.linalg.norm(pts[:, :2] - np.asarray(predicted_xy), axis=1)
        pts = pts[finite & dz & (dist <= float(gate_radius_m))]
        if len(pts) < self.min_points:
            return None

        # Connected components in a spatial hash. Unlike an NxN distance matrix,
        # memory stays linear when a complete L2 scan reaches this function.
        cell = self.cluster_radius_m
        cell_keys = np.floor(pts[:, :2] / cell).astype(np.int64)
        buckets = {}
        for index, key in enumerate(map(tuple, cell_keys)):
            buckets.setdefault(key, []).append(index)
        unseen = set(range(len(pts)))
        components = []
        while unseen:
            seed = unseen.pop()
            component = {seed}
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                cx, cy = cell_keys[current]
                nearby = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nearby.extend(buckets.get((cx + dx, cy + dy), ()))
                neighbours = {
                    i for i in nearby if i in unseen and
                    np.linalg.norm(pts[i, :2] - pts[current, :2]) <= cell
                }
                unseen -= neighbours
                component |= neighbours
                frontier.extend(neighbours)
            if len(component) >= self.min_points:
                components.append(pts[sorted(component)])

        candidates = []
        for cluster in components:
            extent = np.ptp(cluster, axis=0)
            if max(extent[0], extent[1]) > self.max_extent_m:
                continue
            center = np.median(cluster, axis=0)
            residual = float(np.linalg.norm(center[:2] - predicted_xy))
            spread = np.maximum(np.var(cluster[:, :2], axis=0) / len(cluster), 0.0025)
            candidates.append((residual, -len(cluster), center, spread, len(cluster)))
        if not candidates:
            return None
        residual, _, center, spread, count = min(candidates, key=lambda item: item[:2])
        return LidarCluster(*map(float, center), count, residual,
                            (float(spread[0]), float(spread[1])))


class StaticVoxelBackground:
    """Immutable lidar background learned from an empty-arena calibration.

    Do not learn this map while a rover is standing still in the arena: an
    occupied voxel is intentionally treated as permanent background.
    """

    def __init__(self, voxel_size_m=0.08, neighbour_cells=1):
        self.voxel_size_m = float(voxel_size_m)
        self.neighbour_cells = int(neighbour_cells)
        self.voxels = set()

    def fit(self, point_clouds, min_frame_fraction=0.6):
        clouds = list(point_clouds)
        if not clouds:
            raise ValueError("at least one empty-arena cloud is required")
        counts = {}
        for cloud in clouds:
            points = np.asarray(cloud, dtype=float).reshape(-1, 3)
            points = points[np.isfinite(points).all(axis=1)]
            keys = {tuple(key) for key in np.floor(points / self.voxel_size_m).astype(np.int64)}
            for key in keys:
                counts[key] = counts.get(key, 0) + 1
        threshold = max(1, math.ceil(len(clouds) * float(min_frame_fraction)))
        self.voxels = {key for key, count in counts.items() if count >= threshold}
        return self

    def remove(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not self.voxels:
            return points.copy()
        keys = np.floor(points / self.voxel_size_m).astype(np.int64)
        keep = []
        n = self.neighbour_cells
        for key in keys:
            static = any(
                tuple(key + (dx, dy, dz)) in self.voxels
                for dx in range(-n, n + 1)
                for dy in range(-n, n + 1)
                for dz in range(-n, n + 1)
            )
            keep.append(not static)
        return points[np.asarray(keep, dtype=bool)]


def lidar_gate_radius(state, scan_period_s=1 / 5.55, timestamp_sigma_s=0.01,
                      base_radius_m=0.35, max_speed_mps=11.12):
    """Conservative spatial gate including scan and timestamp motion."""
    sigma = math.sqrt(max(state.covariance_m2))
    motion = max_speed_mps * (float(scan_period_s) / 2 + float(timestamp_sigma_s))
    return float(min(2.5, max(base_radius_m, 3 * sigma + motion)))


def scan_grid_to_points(ranges, horizontal_angles, vertical_angles,
                        sensor_to_world=None):
    """Convert an organized spherical lidar scan into world XYZ points."""
    r = np.asarray(ranges, dtype=float).reshape(len(vertical_angles), len(horizontal_angles))
    az, el = np.meshgrid(np.asarray(horizontal_angles), np.asarray(vertical_angles))
    xyz = np.c_[
        (r * np.cos(el) * np.cos(az)).ravel(),
        (r * np.cos(el) * np.sin(az)).ravel(),
        (r * np.sin(el)).ravel(),
    ]
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if sensor_to_world is not None:
        T = np.asarray(sensor_to_world, dtype=float).reshape(4, 4)
        xyz = np.c_[xyz, np.ones(len(xyz))] @ T.T
        xyz = xyz[:, :3]
    return xyz


def project_world_to_image(point_world, camera):
    """Project an arena XYZ point using the calibrated optical camera pose."""
    import cv2
    point = np.asarray(point_world, dtype=float).reshape(3)
    origin = np.asarray(camera["position_world"], dtype=float).reshape(3)
    R_world_optical = np.asarray(camera["R_world_optical"], dtype=float).reshape(3, 3)
    point_optical = R_world_optical.T @ (point - origin)
    if point_optical[2] <= 0:
        return None
    K = np.asarray(camera["K"], dtype=float).reshape(3, 3)
    D = np.asarray(camera.get("D", [0] * 5), dtype=float)
    uv, _ = cv2.projectPoints(point_optical.reshape(1, 3), np.zeros(3),
                              np.zeros(3), K, D)
    return tuple(map(float, uv.reshape(2)))


@dataclass(frozen=True)
class MotionDetection:
    bbox: tuple
    confidence: float
    foreground_pixels: int


class CameraMotionFollower:
    """Fixed-camera MOG2 fallback, constrained by a predicted image ROI.

    It deliberately returns nothing without a prediction: background motion is
    allowed to continue a tagged track, never to assign rover identity.
    """

    def __init__(self, history=180, var_threshold=12, min_area_px=20):
        import cv2
        self.cv2 = cv2
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=int(history), varThreshold=float(var_threshold), detectShadows=False
        )
        self.min_area_px = int(min_area_px)
        self.kernel = np.ones((3, 3), np.uint8)

    def update(self, image, predicted_uv=None, roi_radius_px=100, learning_rate=-1):
        cv2 = self.cv2
        mask = self.subtractor.apply(image, learningRate=float(learning_rate))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        if predicted_uv is None:
            return None
        height, width = mask.shape[:2]
        u, v = map(float, predicted_uv)
        radius = int(roi_radius_px)
        x0, y0 = max(0, int(u) - radius), max(0, int(v) - radius)
        x1, y1 = min(width, int(u) + radius + 1), min(height, int(v) + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        n, _, stats, centers = cv2.connectedComponentsWithStats(mask[y0:y1, x0:x1])
        candidates = []
        for idx in range(1, n):
            x, y, w, h, area = map(int, stats[idx])
            if area < self.min_area_px:
                continue
            cx, cy = centers[idx] + [x0, y0]
            pixel_error = math.hypot(cx - u, cy - v)
            fill = area / max(w * h, 1)
            confidence = max(0.05, min(0.8, fill * math.exp(-pixel_error / max(radius, 1))))
            candidates.append((pixel_error, -area,
                               MotionDetection((x + x0, y + y0, x + x0 + w, y + y0 + h),
                                               confidence, area)))
        return min(candidates, key=lambda item: item[:2])[2] if candidates else None
