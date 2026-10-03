"""Synthetic 1640x1232 fisheye views for testing and demonstrating the calibration.

* ``strip_views``: the 12x3 m floor strip seen by the cameras of a runtime
  config (equidistant lens, D = 0, as ``tools.charuco_strip.render_view``).
* ``HandBoardRenderer``: a hand-held ChArUco board in random poses through an
  OpenCV fisheye model with a known K, D -- the answer the intrinsic
  calibration has to find.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from tools.charuco_strip import Board, Lens, Placement, render_raster, render_view

ROOT = Path(__file__).resolve().parents[1]
STRIP_BOARD_JSON = ROOT / "assets" / "charuco_strip" / "charuco_strip_12x3_board.json"
SIZE = (1640, 1232)


def strip_board():
    spec = json.loads(STRIP_BOARD_JSON.read_text())["board"]
    return Board.from_dict(spec)


def equidistant_intrinsics(size=SIZE, dfov_deg=160.0):
    lens = Lens("fisheye", size[0], size[1], dfov_deg=dfov_deg)
    k = np.array([[lens.f, 0, lens.cx], [0, lens.f, lens.cy], [0, 0, 1]], float)
    return lens, k, np.zeros(4)


class StripRenderer:
    def __init__(self, cameras, size=SIZE, px_per_mm=0.5, dfov_deg=160.0):
        self.board = strip_board()
        self.texture = render_raster(self.board, px_per_mm)
        self.px_per_mm = px_per_mm
        self.cameras = {c["name"]: c for c in cameras}
        self.lens, self.k, self.d = equidistant_intrinsics(size, dfov_deg)

    def render(self, camera_id, placement, seed=0):
        place = Placement(placement.x0, placement.y0, placement.yaw_deg, 0.0)
        return render_view(self.cameras[camera_id], self.lens, self.texture, self.px_per_mm, place,
                           supersample=1, blur_px=0.6, noise=2.0, floor_gray=40,
                           rng=np.random.default_rng(seed))


class HandBoardRenderer:
    """An A4 ChArUco board (``target``) in front of a fisheye camera with known K, D."""

    def __init__(self, target, k, d, size=SIZE, px_per_mm=8.0, seed=1):
        self.size, self.k, self.d = tuple(size), np.asarray(k, float), np.asarray(d, float).reshape(4)
        self.rng = np.random.default_rng(seed)
        board = target._detectors[target.decided if target.decided is not None
                                  else next(iter(target._detectors))][0]
        sx, sy = target.squares
        self.board_w_m, self.board_h_m = sx * target.square_m, sy * target.square_m
        self.margin_m = 0.012
        self.px_per_m = px_per_mm * 1000
        margin_px = int(round(self.margin_m * self.px_per_m))
        w = int(round(self.board_w_m * self.px_per_m)) + 2 * margin_px
        h = int(round(self.board_h_m * self.px_per_m)) + 2 * margin_px
        self.texture = board.generateImage((w, h), marginSize=margin_px)
        u, v = np.meshgrid(np.arange(size[0], dtype=np.float64), np.arange(size[1], dtype=np.float64))
        pts = np.stack([u.ravel(), v.ravel()], -1).reshape(-1, 1, 2)
        norm = cv2.fisheye.undistortPoints(pts, self.k, self.d.reshape(4, 1)).reshape(-1, 2)
        # Rays beyond ~85 deg come back unreliable; they look at the room, not the board.
        r = np.hypot(norm[:, 0], norm[:, 1])
        self.valid = r < math.tan(math.radians(85))
        self.rays = np.column_stack([norm, np.ones(len(norm))])

    def random_pose(self):
        """Board -> camera pose: 0.22-0.45 m away, anywhere in the field, tilted."""
        theta = math.radians(self.rng.uniform(0, 62))
        phi = self.rng.uniform(-math.pi, math.pi)
        dist = self.rng.uniform(0.22, 0.45)
        centre = dist * np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi),
                                  math.cos(theta)])
        # Face roughly toward the camera, then tilt.
        z = centre / np.linalg.norm(centre)
        tilt = cv2.Rodrigues(self.rng.uniform(-0.6, 0.6, 3))[0]
        zb = tilt @ z
        xb = np.cross([0.0, 1.0, 0.0], zb)
        xb /= np.linalg.norm(xb)
        yb = np.cross(zb, xb)
        roll = cv2.Rodrigues(zb * self.rng.uniform(-math.pi, math.pi))[0]
        r = roll @ np.column_stack([xb, yb, zb])
        t = centre - r @ np.array([self.board_w_m / 2, self.board_h_m / 2, 0.0])
        return r, t

    def render(self, r=None, t=None, noise=2.0, blur=0.6):
        if r is None:
            r, t = self.random_pose()
        n = r[:, 2]
        denom = self.rays @ n
        s = np.where(np.abs(denom) > 1e-9, (n @ t) / np.where(np.abs(denom) > 1e-9, denom, 1), -1)
        p = self.rays * s[:, None]
        local = (p - t) @ r                       # board frame metres
        bx = (local[:, 0] + self.margin_m) * self.px_per_m
        by = (local[:, 1] + self.margin_m) * self.px_per_m
        th, tw = self.texture.shape
        inside = self.valid & (s > 0) & (bx >= 0) & (bx < tw - 1) & (by >= 0) & (by < th - 1)
        shape = (self.size[1], self.size[0])
        map_x = np.where(inside, bx, -10).astype(np.float32).reshape(shape)
        map_y = np.where(inside, by, -10).astype(np.float32).reshape(shape)
        img = cv2.remap(self.texture, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=90).astype(np.float64)
        if blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), blur)
        img += self.rng.normal(0, noise, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8), (r, t)
