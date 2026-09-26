"""Render the arena marker into a synthetic ceiling-camera frame.

Corner ordering is a contract between three parties: the ArUco detector, the
object points handed to IPPE, and the tag-to-base transform.  Hand-written
corner arrays silently pick one convention and prove nothing about the other
two, so these tests render an actual marker and detect it.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# Matches worlds/mocap_arena_l2.sdf and the runtime calibration derived from it.
K = np.array([[488.31088823052755, 0.0, 647.846919881951],
              [0.0, 493.6294483217684, 513.414889797276],
              [0.0, 0.0, 1.0]])
IMAGE_SIZE = (1280, 960)
BASE_Z = 0.14
TAG_DZ = 0.2254          # top marker surface above the base link
BOTTOM_TAG_DZ = 0.0146   # bottom marker, mounted with R_base_tag = diag(1,-1,-1)
TAG_SIZE = 0.40          # outer black square
BOARD_SIZE = 0.50        # white plate the marker is printed on

# scripts/run_tag_coverage_experiment.py: R_world_optical = Rz*Ry*Rx @ this.
GAZEBO_FROM_OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)


def rotation(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def camera_rotation(tilt_deg=0.0, yaw_deg=0.0):
    """R_world_optical for a ceiling camera, in the world's own convention.

    A Gazebo camera looks along its link +X, so nadir is pitch = +pi/2.
    """
    return rotation(0.0, math.pi / 2 - math.radians(tilt_deg), math.radians(yaw_deg)) @ GAZEBO_FROM_OPTICAL


def marker_image(tag_id=0, family="tag36h11", modules_px=40):
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11 if family == "tag36h11" else cv2.aruco.DICT_4X4_50
    )
    side = modules_px * (8 if family == "tag36h11" else 6)
    return cv2.aruco.generateImageMarker(dictionary, int(tag_id), side)


def render(camera_xyz, R_world_optical, base_xy, yaw=0.0, tag_id=0,
           family="tag36h11", base_z=BASE_Z, tag_dz=TAG_DZ,
           flipped=False, background=150, blur=0.0):
    """Return an 8-bit frame with the marker drawn where the camera sees it.

    ``flipped`` mounts the marker with R_base_tag = diag(1,-1,-1), the
    convention marker id 1 uses.
    """
    width, height = IMAGE_SIZE
    frame = np.full((height, width), background, np.uint8)
    marker = marker_image(tag_id, family)
    pad = int(round(marker.shape[0] * (BOARD_SIZE - TAG_SIZE) / (2 * TAG_SIZE)))
    board = cv2.copyMakeBorder(marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    n = board.shape[0]

    c, s = math.cos(yaw), math.sin(yaw)
    R_world_base = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    R_base_tag = np.diag([1.0, -1.0, -1.0]) if flipped else np.eye(3)
    R_world_tag = R_world_base @ R_base_tag
    origin = np.array([base_xy[0], base_xy[1], base_z]) + R_world_base @ np.array([0, 0, tag_dz])

    half = BOARD_SIZE / 2
    # Board corners in tag coordinates, in the same order as the board image
    # pixels below: top-left, top-right, bottom-right, bottom-left with the
    # tag frame's +x right and +y up.
    tag_corners = np.array([[-half, half, 0], [half, half, 0],
                            [half, -half, 0], [-half, -half, 0]])
    world = tag_corners @ R_world_tag.T + origin
    optical = (world - np.asarray(camera_xyz, float)) @ np.asarray(R_world_optical, float)
    if (optical[:, 2] <= 1e-6).any():
        return None
    uv = (optical[:, :2] / optical[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    if not np.isfinite(uv).all() or np.abs(uv).max() > 20000:
        return None
    source = np.array([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, uv.astype(np.float32))
    warped = cv2.warpPerspective(board, homography, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpPerspective(np.full_like(board, 255), homography, (width, height),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    frame[mask > 127] = warped[mask > 127]
    if blur:
        frame = cv2.GaussianBlur(frame, (0, 0), float(blur))
    return frame


INVERTED_BASE_Z = 0.225      # upside down, resting on the top plate


def render_inverted(camera_xyz, R_world_optical, base_xy, yaw=0.0, **kwargs):
    """The rover on its back: the underside marker now faces the ceiling.

    Two rotations cancel. The marker is printed with R_base_tag =
    diag(1,-1,-1) so it faces down, and turning the rover over applies the
    same rotation again, so what the ceiling sees is an unmirrored marker --
    which is the whole reason it can be decoded at all. What flips is the
    offset: +0.0146 in base coordinates ends up below the base in the world,
    and the base rests a top-plate height above the floor.
    """
    return render(camera_xyz, R_world_optical, base_xy, yaw=yaw, tag_id=1,
                  flipped=False, base_z=INVERTED_BASE_Z,
                  tag_dz=-BOTTOM_TAG_DZ, **kwargs)


def visible_corners(frame, family="tag36h11", ids=(0, 1)):
    """Detect the marker the way the runtime does."""
    from localization_contracts.detector import AprilTagImageDetector
    detector = AprilTagImageDetector(family, allowed_ids=ids)
    return detector.detect(frame)
