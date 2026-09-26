"""Synthesise a Unitree L2 revolution of the arena, the way the SDF defines it.

288 azimuth x 64 elevation over the lower hemisphere at 5.55 Hz, the sensor
bolted to the ceiling at (6, 6, 2.75) with roll = pi.  Ray-casts against the
arena floor and axis-aligned boxes so a test can ask what the lidar would
actually return for a given rover pose.
"""
from __future__ import annotations

import math

import numpy as np

SENSOR_POSITION = np.array([6.0, 6.0, 2.75])
SENSOR_ROTATION = np.diag([1.0, -1.0, -1.0])      # roll = pi
AZIMUTH_SAMPLES = 288
ELEVATION_SAMPLES = 64
RANGE_MIN, RANGE_MAX = 0.05, 30.0
REVOLUTION_S = 1 / 5.55

# tag_rover in worlds/mocap_arena_l2.sdf, relative to the base link at z=0.14.
ROVER_BOXES = (
    ((0.72, 0.52, 0.18), 0.12),        # body, centre 0.12 above the base
    ((0.50, 0.50, 0.01), 0.22),        # top marker board
)
FLOOR_Z = 0.0
ARENA = (0.0, 12.0)


def ray_grid():
    azimuth = -math.pi + np.arange(AZIMUTH_SAMPLES) * (2 * math.pi / AZIMUTH_SAMPLES)
    elevation = np.arange(ELEVATION_SAMPLES) * (math.pi / 2) / (ELEVATION_SAMPLES - 1)
    el, az = np.meshgrid(elevation, azimuth, indexing="ij")
    local = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)
    world = local.reshape(-1, 3) @ SENSOR_ROTATION.T
    return azimuth, elevation, world.reshape(ELEVATION_SAMPLES, AZIMUTH_SAMPLES, 3)


_AZIMUTH, _ELEVATION, _DIRECTIONS = ray_grid()


def _slab(origin, direction, low, high, axis):
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (low - origin[axis]) / direction[..., axis]
        t1 = (high - origin[axis]) / direction[..., axis]
    return np.minimum(t0, t1), np.maximum(t0, t1)


def _box_range(centre, size):
    """Distance to an axis-aligned box, inf where the ray misses it."""
    lower = np.asarray(centre) - np.asarray(size) / 2
    upper = np.asarray(centre) + np.asarray(size) / 2
    near = np.full(_DIRECTIONS.shape[:2], -np.inf)
    far = np.full(_DIRECTIONS.shape[:2], np.inf)
    for axis in range(3):
        low, high = _slab(SENSOR_POSITION, _DIRECTIONS, lower[axis], upper[axis], axis)
        near = np.maximum(near, np.nan_to_num(low, nan=-np.inf))
        far = np.minimum(far, np.nan_to_num(high, nan=np.inf))
    hit = (near <= far) & (far > 0)
    return np.where(hit, np.maximum(near, 0.0), np.inf)


def render(rover_xy=None, rover_yaw=0.0, extra_boxes=(), base_z=0.14):
    """Return a parsed-scan dict, the same shape parse_laser_scan produces.

    Rover yaw is approximated by the axis-aligned bounding box of the rotated
    footprint; the clustering under test cares about extent and point count,
    not about which corner is which.
    """
    ranges = np.full((ELEVATION_SAMPLES, AZIMUTH_SAMPLES), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        floor = (FLOOR_Z - SENSOR_POSITION[2]) / _DIRECTIONS[..., 2]
    floor = np.where(floor > 0, floor, np.inf)
    with np.errstate(invalid="ignore"):
        hit = SENSOR_POSITION + _DIRECTIONS * floor[..., None]
    inside = ((hit[..., 0] >= ARENA[0]) & (hit[..., 0] <= ARENA[1])
              & (hit[..., 1] >= ARENA[0]) & (hit[..., 1] <= ARENA[1]))
    ranges = np.minimum(ranges, np.where(inside, floor, np.inf))

    boxes = list(extra_boxes)
    if rover_xy is not None:
        c, s = abs(math.cos(rover_yaw)), abs(math.sin(rover_yaw))
        for (sx, sy, sz), dz in ROVER_BOXES:
            boxes.append(((rover_xy[0], rover_xy[1], base_z + dz),
                          (sx * c + sy * s, sx * s + sy * c, sz)))
    for centre, size in boxes:
        ranges = np.minimum(ranges, _box_range(centre, size))

    ranges = np.where(np.isfinite(ranges) & (ranges < RANGE_MAX), ranges, np.inf)
    return {"ranges": ranges, "azimuth": _AZIMUTH, "elevation": _ELEVATION,
            "range_min": RANGE_MIN, "range_max": RANGE_MAX, "stamp_ns": 0}
