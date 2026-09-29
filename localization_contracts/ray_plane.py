"""Pixel bearings intersected with a known horizontal plane.

Shared by every image measurement that knows the height of what it sees: the
marker observer (marker plane), the opponent silhouette (top of the body) and
the bounding-box fallback.  All of them go through ``CameraModel.undistort``,
so a fisheye lens is handled once, here, instead of per caller.
"""
from __future__ import annotations

import math

import numpy as np


def pixel_rays(camera_model, pixels):
    """Pixels -> normalized undistorted rays ``(x/z, y/z)``, shape (N, 2)."""
    return camera_model.undistort(np.asarray(pixels, dtype=float).reshape(-1, 2))


def quad_centre_ray(camera_model, corners_px):
    """Normalized bearing of a planar quadrilateral's centre.

    The centre of a planar square is where its diagonals cross, and a
    perspective projection keeps that true, so the crossing is taken on the
    undistorted rays.  The pixel mean of the corners is not the centre:
    perspective and, far more, a fisheye lens pull it towards the image
    centre -- 1.2 cm of XY at 40 degrees on the IMX219 lens.
    """
    rays = pixel_rays(camera_model, corners_px)
    h = np.hstack([rays, np.ones((4, 1))])
    crossing = np.cross(np.cross(h[0], h[2]), np.cross(h[1], h[3]))
    if abs(crossing[2]) < 1e-12 or not np.isfinite(crossing).all():
        return rays.mean(axis=0)
    return crossing[:2] / crossing[2]


def ray_plane(ray, R_world_camera, camera_position, plane_z):
    """Intersect one normalized bearing with ``z = plane_z``.

    Returns ``(point, incidence_rad)`` or ``(None, None)`` when the ray is
    parallel to the plane or the plane is behind the camera.
    """
    direction = np.asarray(R_world_camera, dtype=float) @ np.array([ray[0], ray[1], 1.0])
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None, None
    direction = direction / norm
    origin = np.asarray(camera_position, dtype=float)
    drop = plane_z - origin[2]
    if abs(direction[2]) < 1e-3 or drop / direction[2] <= 0:
        return None, None
    point = origin + direction * (drop / direction[2])
    return point, math.acos(min(1.0, abs(direction[2])))


def rays_plane(rays, R_world_camera, camera_position, plane_z):
    """Vectorised ``ray_plane``: (N, 2) rays -> (N, 3) points and a validity mask."""
    rays = np.asarray(rays, dtype=float).reshape(-1, 2)
    directions = np.column_stack([rays, np.ones(len(rays))]) @ np.asarray(
        R_world_camera, dtype=float).T
    origin = np.asarray(camera_position, dtype=float)
    drop = plane_z - origin[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = drop / directions[:, 2]
    valid = np.isfinite(scale) & (scale > 0) & (np.abs(directions[:, 2]) > 1e-6)
    points = origin + directions * np.where(valid, scale, 0.0)[:, None]
    return points, valid


def pixels_to_plane(camera_model, R_world_camera, camera_position, pixels, plane_z):
    """Pixels straight to points on ``z = plane_z``; (N, 3) and a validity mask."""
    return rays_plane(pixel_rays(camera_model, pixels), R_world_camera,
                      camera_position, plane_z)
