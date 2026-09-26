"""Unitree L2 ceiling lidar: scan -> arena points -> one rover cluster.

The lidar is a continuation source, never an identity source.  It can keep a
track alive when the marker is hidden, out of every camera's reach, or simply
stationary, but it may not create a track and it may not restore one: a second
rover and a shadow both look like a cluster of the right size.

Geometry that shapes everything here, from the SDF and the arena:

- 288 azimuth x 64 elevation per revolution at 5.55 Hz, so the angular pitch
  is 1.25 deg by 1.43 deg and the footprint on the rover's top plate grows
  from 1 x 6 cm directly below to 11 x 31 cm at 5 m horizontal range.  Cluster
  point counts fall from a few hundred to about ten over the same span, which
  is why the useful radius is a parameter and not an assumption.
- The rover subtends 8 to 40 deg of azimuth, so a real L2 sweeps across it in
  4 to 20 ms rather than over the full 180 ms revolution.  The measurement is
  therefore nearly instantaneous at a well-defined instant, and stamping it at
  the end of the revolution would throw away up to 90 ms for nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rover_filter import Measurement, POSITION

# Single source of truth for the tunables a CLI also exposes.  Repeating a
# default in argparse is how --lidar-max-radius came to sit at 3.5 m in three
# places, two of them stale, and cost two Gazebo runs in which every scan was
# rejected.  Anything with a flag is named here and referenced, never copied.
DEFAULT_Z_BAND = (0.05, 0.60)
DEFAULT_MAX_USEFUL_RADIUS_M = 8.0
DEFAULT_MAX_EXTENT_M = 1.10
DEFAULT_MIN_POINTS = 5
DEFAULT_SWEEP_DURATION_S = 0.0


@dataclass(frozen=True)
class LidarCluster:
    """One rover-shaped group of returns, in arena coordinates."""

    x: float
    y: float
    z_max: float
    points: int
    extent_x: float
    extent_y: float
    residual_m: float          # distance from the prediction that selected it
    sigma_m: float             # 1-sigma position uncertainty
    stamp_ns: int              # midpoint of the sector that saw it
    rejected_candidates: int   # other clusters inside the gate


@dataclass(frozen=True)
class Scan:
    """One organized revolution, already in arena coordinates."""

    points: np.ndarray         # (N, 3)
    offsets_ns: np.ndarray     # (N,) time of each point relative to stamp_ns
    stamp_ns: int
    rays: int
    returns: int


def parse_laser_scan(message):
    """Pull the organized grid out of a gz.msgs.LaserScan."""
    vertical = max(int(getattr(message, "vertical_count", 1)), 1)
    horizontal = int(message.count)
    ranges = np.asarray(message.ranges, dtype=float)
    if ranges.size != vertical * horizontal:
        raise ValueError(
            f"laser scan has {ranges.size} ranges for {vertical}x{horizontal}")
    azimuth = message.angle_min + np.arange(horizontal) * message.angle_step
    if vertical > 1:
        elevation = (message.vertical_angle_min
                     + np.arange(vertical) * message.vertical_angle_step)
    else:
        elevation = np.zeros(1)
    stamp = message.header.stamp
    return {
        "ranges": ranges.reshape(vertical, horizontal),
        "azimuth": azimuth,
        "elevation": elevation,
        "range_min": float(message.range_min),
        "range_max": float(message.range_max),
        "stamp_ns": int(stamp.sec) * 1_000_000_000 + int(stamp.nsec),
    }


class ArenaLidar:
    """Turn L2 revolutions into at most one rover position per revolution."""

    def __init__(self, sensor_position, sensor_rotation, *,
                 z_band=DEFAULT_Z_BAND, arena_bounds=(0.15, 11.85),
                 sweep_duration_s=DEFAULT_SWEEP_DURATION_S, cluster_cell_m=0.20,
                 min_points=DEFAULT_MIN_POINTS, min_points_far=3, far_range_m=2.5,
                 max_extent_m=DEFAULT_MAX_EXTENT_M, min_extent_m=0.10, max_z_m=0.45,
                 sigma_range_m=0.02, sigma_shape_m=0.01,
                 top_slab_m=0.03, trim_percentile=2.0, trim_min_points=20,
                 max_useful_radius_m=DEFAULT_MAX_USEFUL_RADIUS_M, background=None):
        self.position = np.asarray(sensor_position, dtype=float).reshape(3)
        self.rotation = np.asarray(sensor_rotation, dtype=float).reshape(3, 3)
        self.z_band = (float(z_band[0]), float(z_band[1]))
        self.arena_bounds = (float(arena_bounds[0]), float(arena_bounds[1]))
        # A gz gpu_lidar renders the whole grid at one instant, so the sweep is
        # zero here and deskew is a no-op by construction.  Hardware sets this
        # to the revolution period and the same code path corrects the smear.
        self.sweep_duration_s = float(sweep_duration_s)
        self.cluster_cell_m = float(cluster_cell_m)
        self.min_points = int(min_points)
        self.min_points_far = int(min_points_far)
        self.far_range_m = float(far_range_m)
        self.max_extent_m = float(max_extent_m)
        self.min_extent_m = float(min_extent_m)
        self.max_z_m = float(max_z_m)
        self.sigma_range_m = float(sigma_range_m)
        self.sigma_shape_m = float(sigma_shape_m)
        self.top_slab_m = float(top_slab_m)
        self.trim_percentile = float(trim_percentile)
        self.trim_min_points = int(trim_min_points)
        # A sanity bound, not the working limit.  How far the lidar is useful
        # is set by how many returns land on the rover, and the point-count
        # gate measures that directly instead of guessing it from a radius.
        # This matters because the SDF fires 288x64 rays per revolution while
        # the real L2 delivers about 3900 points, a 4.7x lower density: the
        # same thresholds then stop accepting clusters around 3.5 to 4 m
        # without anything needing to be retuned.
        self.max_useful_radius_m = float(max_useful_radius_m)
        self.background = background
        self.scans = 0
        self.detections = 0
        self.rejections = {}

    # ------------------------------------------------------------ conversion

    def scan_to_arena(self, parsed):
        """Organized ranges -> arena points, with a per-point time offset."""
        ranges = parsed["ranges"]
        vertical, horizontal = ranges.shape
        azimuth = parsed["azimuth"]
        elevation = parsed["elevation"]
        valid = (np.isfinite(ranges)
                 & (ranges > parsed["range_min"])
                 & (ranges < parsed["range_max"]))
        rows, columns = np.nonzero(valid)
        if rows.size == 0:
            return Scan(np.empty((0, 3)), np.empty(0, dtype=np.int64),
                        parsed["stamp_ns"], vertical * horizontal, 0)
        r = ranges[rows, columns]
        az = azimuth[columns]
        el = elevation[rows]
        local = np.column_stack([
            r * np.cos(el) * np.cos(az),
            r * np.cos(el) * np.sin(az),
            r * np.sin(el),
        ])
        points = local @ self.rotation.T + self.position
        # Azimuth index is the sweep phase, so it is also the time within the
        # revolution.  Centre the offsets so the scan stamp stays the mean.
        phase = columns / max(horizontal, 1) - 0.5
        offsets = (phase * self.sweep_duration_s * 1e9).astype(np.int64)
        return Scan(points, offsets, parsed["stamp_ns"],
                    vertical * horizontal, int(rows.size))

    def deskew(self, scan, velocity):
        """Undo target motion within the revolution.

        The sensor is bolted to the ceiling, so this corrects the rover moving
        while the beam crosses it, not the sensor moving.  At 11.11 m/s and a
        20 ms dwell that is 22 cm of smear, enough to inflate the cluster past
        its own size gate.
        """
        if self.sweep_duration_s <= 0 or scan.points.size == 0:
            return scan.points
        shift = np.asarray(velocity, dtype=float).reshape(1, 3)
        return scan.points - shift * (scan.offsets_ns.reshape(-1, 1) / 1e9)

    # -------------------------------------------------------------- filtering

    def static_mask(self, points):
        """Keep only returns that could be a rover."""
        low, high = self.arena_bounds
        keep = ((points[:, 2] >= self.z_band[0]) & (points[:, 2] <= self.z_band[1])
                & (points[:, 0] >= low) & (points[:, 0] <= high)
                & (points[:, 1] >= low) & (points[:, 1] <= high))
        if self.background is not None:
            keep &= ~self.background.contains(points)
        return keep

    def gate_radius(self, sigma_m, speed_mps, dwell_s=0.02):
        """Spatial gate: prediction uncertainty plus motion during the dwell.

        The rover crosses only a sector of the revolution, so the motion term
        uses that dwell and not half the 180 ms period; the wider window the
        period implies would pull a passing second rover into the gate.
        """
        motion = float(speed_mps) * float(dwell_s)
        return float(min(2.5, max(0.35, 3.0 * float(sigma_m) + motion + 0.4)))

    # ------------------------------------------------------------- clustering

    ELEVATION_STEP_RAD = math.radians(1.43)
    AZIMUTH_STEP_RAD = math.radians(1.25)

    def footprint_m(self, distance, surface_z=0.35):
        """Largest gap between adjacent returns on the rover's top surface.

        On a horizontal surface the elevation step lands radially as
        ``h/cos^2(theta) * d_el``, the same 1/cos^2 that stretches a camera's
        radial error, while the azimuth step lands tangentially as
        ``slant * d_az``.  Using the slant range for both understates the
        radial gap threefold at 5 m, which is enough for one rover to arrive
        as two clusters and read as two rovers.
        """
        height = max(self.position[2] - float(surface_z), 1e-3)
        slant = math.hypot(distance, height)
        cos_theta = max(height / slant, 1e-3)
        radial = height / cos_theta ** 2 * self.ELEVATION_STEP_RAD
        tangential = slant * self.AZIMUTH_STEP_RAD
        return float(max(radial, tangential))

    def _clusters(self, points, cell=None):
        """Connected components over an occupancy grid.

        Grid union-find rather than pairwise distances: the cell count is what
        grows with the cloud, not the square of the point count.

        The cell has to be at least the return spacing at this range, or a
        sparse distant rover fragments into pieces that each look like a small
        cluster, and two of them inside one gate read as two rovers.
        """
        if len(points) == 0:
            return []
        cell = self.cluster_cell_m if cell is None else float(cell)
        keys = np.floor(points[:, :2] / cell).astype(np.int64)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        index = {tuple(key): i for i, key in enumerate(unique)}
        parent = list(range(len(unique)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            a, b = find(i), find(j)
            if a != b:
                parent[b] = a

        for i, key in enumerate(unique):
            for dx in (0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy <= 0:
                        continue
                    neighbour = index.get((key[0] + dx, key[1] + dy))
                    if neighbour is not None:
                        union(i, neighbour)
        roots = np.array([find(i) for i in range(len(unique))])
        labels = roots[inverse]
        return [points[labels == root] for root in np.unique(labels)]

    def _min_points(self, distance):
        return self.min_points if distance <= self.far_range_m else self.min_points_far

    def centre_of(self, cluster):
        """Centre of the rover's top surface, in arena XY.

        Not the centroid: a radial fan samples the near side of a horizontal
        surface far more densely than the far side, so any density-weighted
        statistic is pulled toward the sensor.  Measured on rendered scans the
        median sits 55 to 145 mm sensor-ward while the centre of the sampled
        extent stays within 37 mm, because the extent is set by the geometric
        edges rather than by how many rays landed where.

        Only the top slab is used.  The body's vertical sides are visible on
        the near face and occluded on the far one, which is a real asymmetry
        rather than a sampling artefact, and no choice of statistic fixes it.
        """
        top = cluster[cluster[:, 2] >= float(np.max(cluster[:, 2])) - self.top_slab_m]
        if len(top) < 3:
            top = cluster
        if len(top) >= self.trim_min_points:
            low = np.percentile(top[:, :2], self.trim_percentile, axis=0)
            high = np.percentile(top[:, :2], 100 - self.trim_percentile, axis=0)
        else:
            low = top[:, :2].min(axis=0)
            high = top[:, :2].max(axis=0)
        return (low + high) / 2, top

    def _sigma(self, top, distance):
        """1-sigma of the extent centre.

        Each edge is located to about half the elevation footprint, and the
        centre averages two of them, so the footprint term carries a 1/(2*sqrt2).
        """
        footprint = self.footprint_m(distance, float(np.median(top[:, 2])))
        spread = max(self.sigma_range_m, footprint / (2 * math.sqrt(2)))
        return float(math.hypot(spread, self.sigma_shape_m))

    # ---------------------------------------------------------------- detect

    def detect(self, points, prediction, sigma_m, speed_mps=0.0, stamp_ns=0):
        """Find the rover near ``prediction``; None when it is not unambiguous.

        Clustering runs over the whole in-band cloud and the prediction gate is
        applied to finished clusters, not to the points.  Cropping first cuts
        a neighbouring object in half, and the truncated remains merge with the
        rover into one blob that still passes a size gate: two rovers 0.8 m
        apart then read as a single cluster 0.10 m off toward the intruder,
        with nothing to warn that it happened.
        """
        self.scans += 1
        prediction = np.asarray(prediction, dtype=float).reshape(2)
        distance = float(np.linalg.norm(prediction - self.position[:2]))
        if distance > self.max_useful_radius_m:
            return self._reject("beyond_useful_radius")
        candidates = points[self.static_mask(points)]
        if len(candidates) < self.min_points_far:
            return self._reject("no_returns_in_band")
        radius = self.gate_radius(sigma_m, speed_mps)

        cell = max(self.cluster_cell_m, 1.6 * self.footprint_m(distance))
        accepted = []
        oversized = 0
        for cluster in self._clusters(candidates, cell):
            if len(cluster) < self._min_points(distance):
                continue
            centre, top = self.centre_of(cluster)
            residual = float(np.linalg.norm(centre - prediction))
            if residual > radius:
                continue
            extent = np.ptp(cluster, axis=0)
            # One footprint of slack per side: the sampled extent overshoots
            # the box by about the spacing between returns.
            slack = min(2 * self.footprint_m(distance), 0.40)
            if max(extent[0], extent[1]) > self.max_extent_m + slack:
                oversized += 1
                continue
            if len(cluster) >= 8 and max(extent[0], extent[1]) < self.min_extent_m:
                continue
            if float(np.max(cluster[:, 2])) > self.max_z_m:
                continue
            accepted.append((residual, cluster, centre, extent, top))
        if oversized and not accepted:
            # Something rover-shaped is in the gate but merged with a
            # neighbour. Reporting the merged centre would move the track.
            return self._reject("merged_cluster")
        if not accepted:
            return self._reject("no_rover_shaped_cluster")
        accepted.sort(key=lambda item: item[0])
        # Two rover-shaped clusters inside one gate is exactly the situation
        # where picking the nearer one silently swaps rovers.
        if len(accepted) > 1:
            return self._reject("ambiguous_cluster")
        residual, cluster, centre, extent, top = accepted[0]
        self.detections += 1
        return LidarCluster(
            float(centre[0]), float(centre[1]), float(np.max(cluster[:, 2])),
            len(cluster), float(extent[0]), float(extent[1]), residual,
            self._sigma(top, distance), int(stamp_ns), len(accepted) - 1,
        )

    def _reject(self, reason):
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        return None


def may_continue(tracking_state):
    """May a continuation source update a track in this state?

    Only LOST bars it.  LOST means the identity is gone and a marker has to
    establish it again, which is the rule that stops a second rover inheriting
    the track.  REACQUIRING means the marker is merely stale, and barring
    continuation there is circular: the state reaches REACQUIRING 300 ms into
    a gap precisely because nothing is updating the track, so a lidar allowed
    to run only while the track is healthy can never fill a gap.  Measured:
    with REACQUIRING barred, all eleven coverage gaps logged
    no_confirmed_track for their whole length and coverage did not move.

    A track fed only by continuation still goes REACQUIRING once the marker
    ages out, so it stops being published as valid.  What continuation buys is
    a position good enough that the marker, when it returns, reacquires
    instantly and the identity check means something.
    """
    return tracking_state not in ("LOST", "INITIALIZING")


def measurement_from_cluster(cluster, source="lidar"):
    """Adapt a cluster into a continuation-only filter measurement."""
    variance = cluster.sigma_m ** 2
    return Measurement(
        int(cluster.stamp_ns), POSITION, (cluster.x, cluster.y),
        (variance, 0.0, 0.0, variance), source,
        identity=None, confirms_identity=False,
        quality=min(1.0, cluster.points / 40.0),
    )


class StaticVoxelMap:
    """Occupied voxels of the empty arena, learned once and never updated.

    An occupied voxel is treated as permanent, so a rover standing still while
    this is built becomes part of the background for the rest of the session.
    Build it on an empty arena, from a file, and not from the live stream.
    """

    _SPAN = 2048          # voxel indices per axis, enough for a 12 m arena
    _OFFSET = 1024

    def __init__(self, voxel_size_m=0.06, dilate=1):
        self.voxel_size_m = float(voxel_size_m)
        self.dilate = int(dilate)
        self.voxels = set()
        self._encoded = np.empty(0, dtype=np.int64)

    @classmethod
    def _encode(cls, keys):
        """Pack integer voxel indices into one sortable int64 per voxel."""
        keys = np.asarray(keys, dtype=np.int64).reshape(-1, 3) + cls._OFFSET
        if keys.size and (keys.min() < 0 or keys.max() >= cls._SPAN):
            raise ValueError("voxel index outside the encodable range")
        return (keys[:, 0] * cls._SPAN + keys[:, 1]) * cls._SPAN + keys[:, 2]

    def _rebuild(self):
        """Bake the dilation in once so lookup is a vectorised membership test.

        The obvious loop -- 27 dict probes per point -- costs half a million
        lookups per revolution and does not fit in the 180 ms budget.
        """
        if not self.voxels:
            self._encoded = np.empty(0, dtype=np.int64)
            return
        base = np.asarray(sorted(self.voxels), dtype=np.int64).reshape(-1, 3)
        span = range(-self.dilate, self.dilate + 1)
        shifts = np.array([(dx, dy, dz) for dx in span for dy in span for dz in span],
                          dtype=np.int64)
        expanded = (base[:, None, :] + shifts[None, :, :]).reshape(-1, 3)
        self._encoded = np.unique(self._encode(expanded))

    def fit(self, clouds, min_fraction=0.6):
        clouds = [np.asarray(c, dtype=float).reshape(-1, 3) for c in clouds]
        if not clouds:
            raise ValueError("at least one empty-arena cloud is required")
        counts = {}
        for cloud in clouds:
            cloud = cloud[np.isfinite(cloud).all(axis=1)]
            keys = np.unique(np.floor(cloud / self.voxel_size_m).astype(np.int64), axis=0)
            for key in map(tuple, keys):
                counts[key] = counts.get(key, 0) + 1
        threshold = max(1, math.ceil(len(clouds) * float(min_fraction)))
        self.voxels = {tuple(int(v) for v in key)
                       for key, count in counts.items() if count >= threshold}
        self._rebuild()
        return self

    def contains(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if self._encoded.size == 0 or len(points) == 0:
            return np.zeros(len(points), dtype=bool)
        keys = np.floor(points / self.voxel_size_m).astype(np.int64) + self._OFFSET
        inside = ((keys >= 0) & (keys < self._SPAN)).all(axis=1)
        out = np.zeros(len(points), dtype=bool)
        if not inside.any():
            return out
        packed = (keys[inside, 0] * self._SPAN + keys[inside, 1]) * self._SPAN \
            + keys[inside, 2]
        index = np.searchsorted(self._encoded, packed)
        index[index >= self._encoded.size] = 0
        out[inside] = self._encoded[index] == packed
        return out

    def to_dict(self):
        # numpy integers survive the set but not json.dumps.
        return {"voxel_size_m": self.voxel_size_m, "dilate": self.dilate,
                "voxels": sorted(tuple(int(v) for v in key) for key in self.voxels)}

    @classmethod
    def from_dict(cls, data):
        out = cls(data["voxel_size_m"], data.get("dilate", 1))
        out.voxels = {tuple(int(v) for v in key) for key in data["voxels"]}
        out._rebuild()
        return out
