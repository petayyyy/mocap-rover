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
DEFAULT_VIEW_POINTS = 1500        # cap on points shipped to the dashboard


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
                 max_useful_radius_m=DEFAULT_MAX_USEFUL_RADIUS_M, background=None,
                 range_background=None, masked_zones=None, mask_margin_m=0.2,
                 centre_method="top_extent", reject_radius_m=0.6):
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
        # Per-ray empty-arena ranges (RangeBackground).  Applied to the
        # organized grid before it becomes points, so floor and walls leave
        # at the ray where they are cheapest to recognise.
        self.range_background = range_background
        # Floor areas never to be read as a rover (obstacles, a pit): same
        # zones and margin semantics as br_lidar.  Empty in Gazebo.
        self.masked_zones = list(masked_zones or ())
        self.mask_margin_m = float(mask_margin_m)
        if centre_method not in self.CENTRE_METHODS:
            raise ValueError(f"centre_method must be one of {self.CENTRE_METHODS}")
        self.centre_method = centre_method
        self.reject_radius_m = float(reject_radius_m)
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
        # The footprint model needs the sensor's real angular pitch: the
        # constants below are the L2's, and the Airy is three times finer in
        # azimuth.  An organized grid states its own pitch.
        if len(azimuth) > 1:
            self.azimuth_step_rad = float(abs(np.median(np.diff(azimuth))))
        if len(elevation) > 1:
            self.elevation_step_rad = float(abs(np.median(np.diff(elevation))))
        if self.range_background is not None:
            valid &= self.range_background.foreground(ranges)
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
        if self.masked_zones:
            from .foreground import in_masked_zone
            keep &= ~in_masked_zone(points[:, 0], points[:, 1], self.masked_zones,
                                    self.mask_margin_m)
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
        elevation_step = getattr(self, "elevation_step_rad", self.ELEVATION_STEP_RAD)
        azimuth_step = getattr(self, "azimuth_step_rad", self.AZIMUTH_STEP_RAD)
        radial = height / cos_theta ** 2 * elevation_step
        tangential = slant * azimuth_step
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

    CENTRE_METHODS = ("top_extent", "extent", "median")

    def centre_of(self, cluster, method=None, top_slab_m=None):
        """Centre of the cluster in arena XY by the chosen method.

        ``top_extent`` (default) is the trimmed extent of the top slab, below.
        ``extent`` is the trimmed extent of the whole cluster: for a body whose
        highest part is off-centre -- the opponent's cabin sits 0.12 m aft --
        the top slab is not the body.  ``median`` is br_lidar's robust centre,
        the median again after dropping points beyond ``reject_radius_m``.
        """
        method = method or self.centre_method
        if method == "top_extent":
            return self._top_extent_centre(cluster, top_slab_m)
        xy = cluster[:, :2]
        if method == "median":
            centre = np.median(xy, axis=0)
            far = np.linalg.norm(xy - centre, axis=1) > self.reject_radius_m
            if far.any() and (~far).sum() >= self.min_points_far:
                centre = np.median(xy[~far], axis=0)
            return centre, cluster
        if len(cluster) >= self.trim_min_points:
            low = np.percentile(xy, self.trim_percentile, axis=0)
            high = np.percentile(xy, 100 - self.trim_percentile, axis=0)
        else:
            low, high = xy.min(axis=0), xy.max(axis=0)
        return (low + high) / 2, cluster

    def _top_extent_centre(self, cluster, top_slab_m=None):
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
        slab = self.top_slab_m if top_slab_m is None else float(top_slab_m)
        top = cluster[cluster[:, 2] >= float(np.max(cluster[:, 2])) - slab]
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

    def detect(self, points, prediction, sigma_m, speed_mps=0.0, stamp_ns=0,
               max_z_m=None, centre_method=None, top_slab_m=None):
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
            centre, top = self.centre_of(cluster, centre_method, top_slab_m)
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
            if float(np.max(cluster[:, 2])) > (self.max_z_m if max_z_m is None else max_z_m):
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


class RangeBackground:
    """Empty-arena range and spread for every ray of the organized grid.

    Subtracting occupancy seen from above cannot work for a ceiling lidar:
    the empty floor occupies every cell, and the rover would be erased with
    it.  What changes when a rover arrives is the range along each ray, so
    that is what is stored -- one distance and one spread per (channel,
    azimuth) ray.  A return is foreground when it comes back closer than the
    background by ``max(min_margin_m, sigma_factor * sigma)``: grazing rays
    wander more from revolution to revolution and get a wider margin.

    Built from a high percentile, not the mean, so a person crossing during
    the capture does not pull the floor closer.  Rays seen fewer than
    ``min_hits`` times have no background and never report foreground.
    The same principle runs on the real Airy in br_lidar/airy_py/background.py.
    """

    def __init__(self, distance, sigma, seen, *, min_margin_m=0.12, sigma_factor=4.0,
                 floor=None):
        self.distance = np.asarray(distance, dtype=np.float32)
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self.seen = np.asarray(seen, dtype=bool)
        if not (self.distance.shape == self.sigma.shape == self.seen.shape):
            raise ValueError("distance, sigma and seen must share one grid shape")
        self.min_margin_m = float(min_margin_m)
        self.sigma_factor = float(sigma_factor)
        self.floor = floor
        self.threshold = np.where(
            self.seen,
            self.distance - np.maximum(self.min_margin_m, self.sigma_factor * self.sigma),
            0.0).astype(np.float32)

    @classmethod
    def build(cls, grids, percentile=80.0, min_hits=None, **kwargs):
        """From organized range grids of the empty arena; 0/inf/nan = no return."""
        data = np.stack([np.asarray(g, dtype=np.float32) for g in grids])
        if data.ndim != 3 or len(data) == 0:
            raise ValueError("need at least one 2-D range grid")
        valid = np.isfinite(data) & (data > 0)
        hits = valid.sum(axis=0)
        if min_hits is None:
            min_hits = max(1, int(math.ceil(0.5 * len(data))))
        masked = np.where(valid, data, np.nan)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                distance = np.nanpercentile(masked, float(percentile), axis=0)
                sigma = np.nanstd(masked, axis=0)
        seen = hits >= int(min_hits)
        distance = np.where(seen, np.nan_to_num(distance), 0.0)
        sigma = np.where(seen, np.nan_to_num(sigma), 0.0)
        return cls(distance, sigma, seen, **kwargs)

    def foreground(self, ranges):
        """Boolean grid: which returns are closer than the empty arena."""
        ranges = np.asarray(ranges)
        if ranges.shape != self.threshold.shape:
            raise ValueError(f"scan grid {ranges.shape} does not match background "
                             f"{self.threshold.shape}")
        with np.errstate(invalid="ignore"):
            return (self.threshold > 0) & np.isfinite(ranges) & (ranges < self.threshold)

    def save(self, path):
        data = dict(distance=self.distance, sigma=self.sigma, seen=self.seen,
                    min_margin_m=self.min_margin_m, sigma_factor=self.sigma_factor)
        if self.floor is not None:
            data.update(self.floor.to_arrays())
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path, **overrides):
        with np.load(path, allow_pickle=False) as d:
            floor = FloorFit.from_arrays(d) if "floor_normal" in d.files else None
            kwargs = dict(min_margin_m=float(d["min_margin_m"]),
                          sigma_factor=float(d["sigma_factor"]))
            kwargs.update(overrides)
            return cls(d["distance"], d["sigma"], d["seen"], floor=floor, **kwargs)

    def summary(self):
        d, s = self.distance[self.seen], self.sigma[self.seen]
        return {"rays": int(self.seen.size), "rays_with_background": int(self.seen.sum()),
                "distance_m": [float(d.min()), float(d.max())] if d.size else None,
                "sigma_p50_mm": float(np.median(s) * 1000) if s.size else None,
                "sigma_p95_mm": float(np.percentile(s, 95) * 1000) if s.size else None,
                "min_margin_m": self.min_margin_m, "sigma_factor": self.sigma_factor}


@dataclass(frozen=True)
class FloorFit:
    """Floor plane in the lidar's level frame: ``normal . p + height = 0``."""

    normal: tuple
    height_m: float
    tilt_deg: float
    inlier_fraction: float
    residual_p95_m: float
    span_m: tuple

    def to_arrays(self):
        return dict(floor_normal=np.asarray(self.normal, dtype=float),
                    floor_height_m=np.float64(self.height_m),
                    floor_tilt_deg=np.float64(self.tilt_deg),
                    floor_inlier_fraction=np.float64(self.inlier_fraction),
                    floor_residual_p95_m=np.float64(self.residual_p95_m),
                    floor_span_m=np.asarray(self.span_m, dtype=float))

    @classmethod
    def from_arrays(cls, d):
        return cls(tuple(float(v) for v in d["floor_normal"]), float(d["floor_height_m"]),
                   float(d["floor_tilt_deg"]), float(d["floor_inlier_fraction"]),
                   float(d["floor_residual_p95_m"]),
                   tuple(float(v) for v in d["floor_span_m"]))


def fit_floor(points, expected_height, *, max_tilt_deg=10.0, height_tolerance=0.5,
              distance_threshold=0.03, min_span=3.0, iterations=400, seed=0):
    """Floor plane under a ceiling lidar: constrained RANSAC, then SVD.

    ``points`` are in the sensor's level frame -- sensor at the origin, +Z up
    -- so the floor is expected at ``z = -expected_height``.  Candidate
    planes steeper than ``max_tilt_deg`` or further than ``height_tolerance``
    from the expected height are refused before they are scored, so a wall
    can never win.  Dense returns under the sensor are thinned to one per
    10 cm cell first, or the patch below it would decide the tilt alone.
    Raises ValueError when no plane has the required support and span.
    Same procedure as br_lidar/airy_py/floor_calibration.py.
    """
    settings = (expected_height, max_tilt_deg, height_tolerance, distance_threshold, min_span)
    if (not np.all(np.isfinite(settings)) or expected_height <= 0
            or not 0 < max_tilt_deg < 45 or height_tolerance <= 0
            or distance_threshold <= 0 or min_span <= 0):
        raise ValueError("invalid floor fit settings")
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    radius = np.hypot(points[:, 0], points[:, 1])
    band = height_tolerance + math.tan(math.radians(max_tilt_deg)) * radius
    points = points[np.abs(points[:, 2] + expected_height) <= band]
    if len(points) < 100:
        raise ValueError("too few returns near the expected floor height")
    _, first = np.unique(np.floor(points / 0.10).astype(np.int64), axis=0, return_index=True)
    points = points[np.sort(first)]
    rng = np.random.default_rng(seed)
    if len(points) > 12000:
        points = points[rng.choice(len(points), 12000, replace=False)]
    if len(points) < 100:
        raise ValueError("too little spatially distinct floor data")
    cosine = math.cos(math.radians(max_tilt_deg))
    best_mask, best_count = None, 0
    for _ in range(int(iterations)):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal
        height = -float(normal @ sample[0])
        if normal[2] < cosine or abs(height - expected_height) > height_tolerance:
            continue
        mask = np.abs(points @ normal + height) <= distance_threshold
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
    required = max(100, int(math.ceil(0.35 * len(points))))
    if best_count < required:
        raise ValueError("no floor plane with enough support")
    mask = best_mask
    for _ in range(3):
        centre = points[mask].mean(axis=0)
        _, _, axes = np.linalg.svd(points[mask] - centre, full_matrices=False)
        normal = axes[-1] if axes[-1][2] >= 0 else -axes[-1]
        height = -float(normal @ centre)
        mask = np.abs(points @ normal + height) <= distance_threshold
        if int(mask.sum()) < required:
            raise ValueError("floor refinement lost support")
    if normal[2] < cosine or height <= 0 or abs(height - expected_height) > height_tolerance:
        raise ValueError("floor tilt or height outside the allowed range")
    inliers = points[mask]
    xy = inliers[:, :2] - inliers[:, :2].mean(axis=0)
    _, axes = np.linalg.eigh(np.cov(xy.T))
    spans = np.diff(np.percentile(xy @ axes, [5, 95], axis=0), axis=0)[0]
    if float(np.min(spans)) < min_span:
        raise ValueError(f"floor coverage {spans[0]:.2f} x {spans[1]:.2f} m is below {min_span} m")
    residual = float(np.percentile(np.abs(inliers @ normal + height), 95))
    return FloorFit(tuple(float(v) for v in normal), float(height),
                    float(math.degrees(math.acos(min(1.0, float(normal[2]))))),
                    float(mask.mean()), residual, (float(spans[0]), float(spans[1])))


def sensor_level_points(parsed, rotation):
    """Organized ranges -> points in the sensor's level frame (origin, +Z up).

    Rotated by the nominal mounting but not translated, which is the frame
    ``fit_floor`` checks the mounting height and tilt in.
    """
    ranges = np.asarray(parsed["ranges"], dtype=float)
    valid = (np.isfinite(ranges) & (ranges > parsed["range_min"])
             & (ranges < parsed["range_max"]))
    rows, columns = np.nonzero(valid)
    r = ranges[rows, columns]
    az = np.asarray(parsed["azimuth"], dtype=float)[columns]
    el = np.asarray(parsed["elevation"], dtype=float)[rows]
    local = np.column_stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az),
                             r * np.sin(el)])
    return local @ np.asarray(rotation, dtype=float).reshape(3, 3).T
