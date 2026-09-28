#!/usr/bin/env python3
"""ChArUco strip for the extrinsic calibration of the arena.

One long ChArUco sheet (by default 12 x 3 m: a 30 x 6 chessboard of 375 mm
squares with a white border one square wide on every side, laid on the
floor) that every ceiling camera can see at once.  The chessboard corners of the strip are the
common points from which the relative poses of the cameras, the arena origin
and the lidar-to-camera transform are solved later.  Intrinsics (K, D) are
calibrated separately with a small hand-held board; this file is only about
the floor target.

The layout is exactly OpenCV's ``cv2.aruco.CharucoBoard`` (non-legacy
pattern): square (0,0) in the top-left corner of the print is black, markers
sit in the white squares and are numbered row by row from the top-left, the
board frame has x to the right and y *down* the print, origin at the top-left
corner.  ``selftest`` renders our own raster next to OpenCV's and asserts they
are identical pixel for pixel, so the printed sheet, the editor project and
the detector always agree.

Subcommands
-----------
generate   write the print files (SVG, PDF, PNG), the OpenCV description
           (board.json), a project for aruco_map_editor.html, the drone-map
           .txt and a Gazebo model with the strip as a textured mesh.
evaluate   render the strip through each camera of a world file with the real
           lens (160 deg diagonal fisheye) at a chosen resolution, run the
           ChArUco detector on the synthetic frame and report what it found.
           This is what sizes the squares.
place      put the strip model into a world file at a chosen position.
detect     run the detector on real photos of the printed strip.
selftest   check the layout against OpenCV.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DICT_NAMES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]

# Waveshare IMX219-160: 160 deg diagonal field of view on the full 4:3 sensor.
DEFAULT_DFOV_DEG = 160.0


# --------------------------------------------------------------------------
# Board geometry
# --------------------------------------------------------------------------
class Board:
    """ChArUco strip geometry in millimetres, board frame = print frame."""

    def __init__(self, squares_x, squares_y, square_mm, marker_mm, dict_name, id_offset=0, border_mm=None):
        if dict_name not in DICT_NAMES:
            raise SystemExit(f"unknown dictionary {dict_name}")
        self.squares_x = int(squares_x)
        self.squares_y = int(squares_y)
        self.square_mm = float(square_mm)
        self.marker_mm = float(marker_mm)
        self.dict_name = dict_name
        self.id_offset = int(id_offset)
        # White border of the sheet around the chessboard, one square by default.
        self.border_mm = self.square_mm if border_mm is None else float(border_mm)
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        self.n = int(self.dictionary.markerSize)
        self.cells = self.n + 2  # code plus one border cell each side
        self.cell_mm = self.marker_mm / self.cells
        self.margin_mm = (self.square_mm - self.marker_mm) / 2
        self.width_mm = self.squares_x * self.square_mm
        self.height_mm = self.squares_y * self.square_mm
        self.sheet_width_mm = self.width_mm + 2 * self.border_mm
        self.sheet_height_mm = self.height_mm + 2 * self.border_mm
        self.white = [(i, j) for j in range(self.squares_y) for i in range(self.squares_x) if (i + j) % 2 == 1]
        self.ids = [self.id_offset + k for k in range(len(self.white))]
        n_codes = int(self.dictionary.bytesList.shape[0])
        if self.ids[-1] >= n_codes:
            raise SystemExit(f"{len(self.white)} markers with offset {self.id_offset} need IDs up to "
                             f"{self.ids[-1]}, but {dict_name} has only {n_codes}")
        if self.margin_mm < self.cell_mm:
            print(f"warning: white margin around a marker ({self.margin_mm:.1f} mm) is smaller than one "
                  f"code cell ({self.cell_mm:.1f} mm); detection suffers", file=sys.stderr)

    # -- markers ------------------------------------------------------------
    def marker_rect_mm(self, k):
        i, j = self.white[k]
        x = i * self.square_mm + self.margin_mm
        y = j * self.square_mm + self.margin_mm
        return x, y, self.marker_mm, self.marker_mm

    def marker_corners_mm(self, k):
        x, y, w, h = self.marker_rect_mm(k)
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)

    def marker_bits(self, marker_id):
        """(n+2) x (n+2) array, 1 = white cell, exactly as OpenCV draws it."""
        if hasattr(self.dictionary, "generateImageMarker"):
            img = self.dictionary.generateImageMarker(marker_id, self.cells, borderBits=1)
        else:  # OpenCV 4.6 and older
            img = cv2.aruco.drawMarker(self.dictionary, marker_id, self.cells, borderBits=1)
        return (img > 127).astype(np.uint8)

    # -- chessboard corners -------------------------------------------------
    def chess_corners_mm(self):
        out = []
        cid = 0
        for j in range(1, self.squares_y):
            for i in range(1, self.squares_x):
                out.append((cid, i * self.square_mm, j * self.square_mm))
                cid += 1
        return out

    def black_squares(self):
        return [(i, j) for j in range(self.squares_y) for i in range(self.squares_x) if (i + j) % 2 == 0]

    def opencv_board(self):
        ids = np.array(self.ids, dtype=np.int32)
        if hasattr(cv2.aruco, "CharucoBoard_create"):  # OpenCV 4.6 and older
            board = cv2.aruco.CharucoBoard_create(
                self.squares_x, self.squares_y, self.square_mm / 1000.0,
                self.marker_mm / 1000.0, self.dictionary)
            if not np.array_equal(board.ids.ravel(), ids):
                board.setIds(ids)
            return board
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_mm / 1000.0,
                                      self.marker_mm / 1000.0, self.dictionary, ids)

    # -- rectangles that make up the print, in mm, image frame -----------------
    def black_rects_mm(self, margin_mm=None):
        margin_mm = self.border_mm if margin_mm is None else margin_mm
        rects = []
        s = self.square_mm
        for i, j in self.black_squares():
            rects.append((margin_mm + i * s, margin_mm + j * s, s, s))
        for k, mid in enumerate(self.ids):
            x0, y0, _, _ = self.marker_rect_mm(k)
            bits = self.marker_bits(mid)
            c = self.cell_mm
            for r in range(self.cells):
                cc = 0
                while cc < self.cells:
                    if bits[r, cc] == 0:
                        c1 = cc
                        while c1 < self.cells and bits[r, c1] == 0:
                            c1 += 1
                        rects.append((margin_mm + x0 + cc * c, margin_mm + y0 + r * c, (c1 - cc) * c, c))
                        cc = c1
                    else:
                        cc += 1
        return rects

    def describe(self):
        return (f"{self.squares_x}x{self.squares_y} squares of {self.square_mm:g} mm, marker {self.marker_mm:g} mm "
                f"({self.marker_mm / self.square_mm:.2f} of square), {self.dict_name}, cell {self.cell_mm:.1f} mm, "
                f"{len(self.ids)} markers (IDs {self.ids[0]}..{self.ids[-1]}), "
                f"{(self.squares_x - 1) * (self.squares_y - 1)} chessboard corners, "
                f"board {self.width_mm / 1000:g} x {self.height_mm / 1000:g} m, white border {self.border_mm:g} mm, "
                f"sheet {self.sheet_width_mm / 1000:g} x {self.sheet_height_mm / 1000:g} m")

    def to_dict(self):
        return dict(squares_x=self.squares_x, squares_y=self.squares_y, square_mm=self.square_mm,
                    marker_mm=self.marker_mm, dictionary=self.dict_name, id_offset=self.id_offset,
                    border_mm=self.border_mm)

    @staticmethod
    def from_dict(d):
        return Board(d["squares_x"], d["squares_y"], d["square_mm"], d["marker_mm"], d["dictionary"],
                     d.get("id_offset", 0), d.get("border_mm", 0.0))


def opencv_board_image(board, size, margin_size=0, border_bits=1):
    """Render a board with either the old or the new OpenCV ChArUco API."""
    if hasattr(board, "generateImage"):
        return board.generateImage(size, marginSize=margin_size, borderBits=border_bits)
    return board.draw(size, marginSize=margin_size, borderBits=border_bits)


def detect_charuco(board: Board, image):
    """Return (ChArUco corners/ids, marker corners/ids) on OpenCV 4.6+."""
    cv_board = board.opencv_board()
    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(
            cv_board, cv2.aruco.CharucoParameters(), detector_params())
        return detector.detectBoard(image)
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        image, board.dictionary, parameters=detector_params())
    if marker_ids is None:
        return None, None, marker_corners, marker_ids
    # Old aruco detectors can report a smaller nested quad with the same ID.
    # A ChArUco board contains every ID once, so retain the largest candidate.
    ids = marker_ids.ravel()
    if len(np.unique(ids)) != len(ids):
        keep = []
        for marker_id in np.unique(ids):
            candidates = np.flatnonzero(ids == marker_id)
            keep.append(max(candidates, key=lambda i: abs(cv2.contourArea(
                np.asarray(marker_corners[i]).reshape(-1, 2).astype(np.float32)))))
        keep.sort()
        marker_corners = [marker_corners[i] for i in keep]
        marker_ids = marker_ids[keep]
    _, corners, corner_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, image, cv_board)
    return corners, corner_ids, marker_corners, marker_ids


def render_raster(board: Board, px_per_mm: float, margin_mm: float = None) -> np.ndarray:
    """Grayscale raster of the whole sheet (border included); edges land on
    integer pixels when square, marker, cell and border sizes are multiples
    of 1/px_per_mm."""
    margin_mm = board.border_mm if margin_mm is None else margin_mm
    w = int(round((board.width_mm + 2 * margin_mm) * px_per_mm))
    h = int(round((board.height_mm + 2 * margin_mm) * px_per_mm))
    img = np.full((h, w), 255, np.uint8)
    for x, y, rw, rh in board.black_rects_mm(margin_mm):
        x0, y0 = int(round(x * px_per_mm)), int(round(y * px_per_mm))
        x1, y1 = int(round((x + rw) * px_per_mm)), int(round((y + rh) * px_per_mm))
        img[y0:y1, x0:x1] = 0
    return img


# --------------------------------------------------------------------------
# Print files
# --------------------------------------------------------------------------
def write_svg(path: Path, board: Board):
    margin_mm = board.border_mm
    W, H = board.sheet_width_mm, board.sheet_height_mm
    d = []
    for x, y, w, h in board.black_rects_mm(margin_mm):
        d.append(f"M{x:g} {y:g}h{w:g}v{h:g}h{-w:g}z")
    svg = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:g}mm" height="{H:g}mm" '
           f'viewBox="0 0 {W:g} {H:g}" shape-rendering="crispEdges">\n'
           f'<title>ChArUco strip {board.describe()}</title>\n'
           f'<rect x="0" y="0" width="{W:g}" height="{H:g}" fill="#ffffff"/>\n'
           f'<path fill="#000000" d="{"".join(d)}"/>\n'
           f'</svg>\n')
    path.write_text(svg, encoding="utf-8")


def write_pdf(path: Path, board: Board, scale: float = 1.0):
    """Vector PDF written by hand.  scale=1 uses /UserUnit so the 12 m page is
    legal (PDF 1.6); scale=0.1 is a plain 1:10 page for viewers that ignore it."""
    MM = 72 / 25.4
    margin_mm = board.border_mm
    W, H = board.sheet_width_mm, board.sheet_height_mm
    if scale == 1.0:
        media = f"[0 0 {W:.4f} {H:.4f}]"
        userunit = f" /UserUnit {MM:.6f}"
        cm = ""
    else:
        media = f"[0 0 {W * scale * MM:.4f} {H * scale * MM:.4f}]"
        userunit = ""
        cm = f"{scale * MM:.6f} 0 0 {scale * MM:.6f} 0 0 cm\n"
    ops = [cm, "0 g\n"]
    for x, y, w, h in board.black_rects_mm(margin_mm):
        ops.append(f"{x:.3f} {H - y - h:.3f} {w:.3f} {h:.3f} re\n")
    ops.append("f\n")
    content = zlib.compress("".join(ops).encode("ascii"))
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox {media}{userunit} /Contents 4 0 R >>").encode(),
        b"<< /Length " + str(len(content)).encode() + b" /Filter /FlateDecode >>\nstream\n" + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.6\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def write_png(path: Path, board: Board, px_per_mm: float):
    img = render_raster(board, px_per_mm)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_BILEVEL, 1])
    return img.shape


# --------------------------------------------------------------------------
# aruco_map_editor.html project and the drone-map .txt
# --------------------------------------------------------------------------
def editor_project(board: Board, name: str):
    """Project JSON for aruco_map_editor.html: black squares as filled rects,
    markers as marker objects.  The editor's sheet frame has y up, so the print
    top edge sits at y = sheet height."""
    margin_mm = board.border_mm
    W, H = board.sheet_width_mm, board.sheet_height_mm
    objs = []
    uid = 1
    s = board.square_mm
    for i, j in board.black_squares():
        objs.append(dict(id=uid, type="shape", shape="rect", name=None,
                         x_mm=margin_mm + (i + 0.5) * s, y_mm=H - (margin_mm + (j + 0.5) * s),
                         w_mm=s, h_mm=s, rot_deg=0, z=0, fill="#000000", stroke="#000000",
                         strokeWidth_mm=0, hidden=False, locked=True))
        uid += 1
    for k, mid in enumerate(board.ids):
        x, y, w, h = board.marker_rect_mm(k)
        objs.append(dict(id=uid, type="marker", name=None, markerId=mid, dictionary=board.dict_name,
                         size_mm=board.marker_mm, x_mm=margin_mm + x + w / 2, y_mm=H - (margin_mm + y + h / 2),
                         rot_deg=0, z=1, border_bits=1, invert=False, quiet_zone_mm=board.margin_mm,
                         intentionalNested=False, hidden=False, locked=False))
        uid += 1
    project = dict(name=name, sheet_w_mm=W, sheet_h_mm=H, gridStepMm=s, defaultDict=board.dict_name,
                   originMode="corner", originObjectId=None, originCornerOffset=dict(x=0, y=0),
                   unitsDisplay="mm", minMarkerSizeMm=20, theme="dark")
    return dict(version=1, project=project, objects=objs, guides=[], uidCounter=uid)


def drone_map_txt(board: Board, name: str):
    margin_mm = board.border_mm
    W, H = board.sheet_width_mm, board.sheet_height_mm
    lines = [f"# Карта ArUco: {name}",
             "# Сгенерировано tools/charuco_strip.py; система координат: x вправо, y вверх, ноль в левом нижнем углу листа",
             f"# sheet_size_m: {W / 1000:g} {H / 1000:g}",
             "# id  length  x  y  z  rot_z  rot_y  rot_x"]
    for k, mid in enumerate(board.ids):
        x, y, w, h = board.marker_rect_mm(k)
        cx = (margin_mm + x + w / 2) / 1000
        cy = (H - (margin_mm + y + h / 2)) / 1000
        lines.append(f"{mid} {board.marker_mm / 1000:g} {cx:g} {cy:g} 0 0 0 0")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Gazebo model: the strip as a textured quad
# --------------------------------------------------------------------------
def write_gazebo_model(model_dir: Path, board: Board, texture: np.ndarray):
    """Static model whose origin is the top-left corner of the sheet (white
    border included); the sheet x axis is the model +X, the sheet y axis (down
    the print) is the model -Y.  Include it with the corner pose and the strip
    lies flat on the floor."""
    (model_dir / "meshes").mkdir(parents=True, exist_ok=True)
    (model_dir / "materials" / "textures").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(model_dir / "materials" / "textures" / "charuco_strip.png"), texture)
    W = board.sheet_width_mm / 1000
    H = board.sheet_height_mm / 1000
    x0 = 0.0
    y0 = 0.0
    obj = ("# ChArUco strip; quad with explicit UVs so the print orientation is unambiguous\n"
           "mtllib charuco_strip.mtl\nusemtl charuco_strip\n"
           f"v {x0:.4f} {y0:.4f} 0\nv {x0 + W:.4f} {y0:.4f} 0\nv {x0 + W:.4f} {y0 - H:.4f} 0\nv {x0:.4f} {y0 - H:.4f} 0\n"
           "vt 0 1\nvt 1 1\nvt 1 0\nvt 0 0\nvn 0 0 1\n"
           "f 4/4/1 3/3/1 2/2/1 1/1/1\n")
    (model_dir / "meshes" / "charuco_strip.obj").write_text(obj)
    (model_dir / "meshes" / "charuco_strip.mtl").write_text(
        "newmtl charuco_strip\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd ../materials/textures/charuco_strip.png\n")
    (model_dir / "model.config").write_text(
        '<?xml version="1.0"?>\n<model>\n  <name>charuco_strip</name>\n  <version>1.0</version>\n'
        '  <sdf version="1.9">model.sdf</sdf>\n  <author><name>mocap-rover</name></author>\n'
        f'  <description>ChArUco calibration strip, {board.describe()}</description>\n</model>\n')
    (model_dir / "model.sdf").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<sdf version="1.9">\n'
        '  <model name="charuco_strip">\n    <static>true</static>\n    <link name="link">\n'
        '      <visual name="print">\n        <cast_shadows>false</cast_shadows>\n'
        '        <geometry><mesh><uri>model://charuco_strip/meshes/charuco_strip.obj</uri></mesh></geometry>\n'
        '        <material>\n          <double_sided>true</double_sided>\n'
        '          <diffuse>1 1 1 1</diffuse><specular>0.05 0.05 0.05 1</specular>\n'
        '          <pbr><metal>\n'
        '            <albedo_map>model://charuco_strip/materials/textures/charuco_strip.png</albedo_map>\n'
        '            <roughness>0.9</roughness><metalness>0</metalness>\n'
        '          </metal></pbr>\n        </material>\n      </visual>\n    </link>\n  </model>\n</sdf>\n')


# --------------------------------------------------------------------------
# Cameras from a world file
# --------------------------------------------------------------------------
def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


# Gazebo camera sensors look along their +X with +Z up; the optical frame is
# x right, y down, z forward.
R_SENSOR_OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


def _pose(text):
    m = re.search(r"<pose[^>]*>([^<]+)</pose>", text)
    v = [float(t) for t in m.group(1).split()] if m else [0] * 6
    return np.array(v[:3]), rot_rpy(*v[3:6])


def cameras_from_world(path: Path):
    text = path.read_text(encoding="utf-8")
    cams = []
    for m in re.finditer(r'<model name="(camera_\d+)">(.*?)</model>', text, re.S):
        name, block = m.group(1), m.group(2)
        model_t, model_R = _pose(block.split("<link", 1)[0])
        link = block.split("<link", 1)[1]
        sensor_m = re.search(r"<sensor\b.*?</sensor>", link, re.S)
        sensor = sensor_m.group(0)
        link_wo = re.sub(r"<(visual|collision|sensor)\b.*?</\1>", "", link, flags=re.S)
        link_t, link_R = _pose(link_wo)
        sensor_t, sensor_R = _pose(sensor.split("<camera", 1)[0])
        R = model_R @ link_R @ sensor_R
        t = model_t + model_R @ (link_t + link_R @ sensor_t)
        hfov = float(re.search(r"<horizontal_fov>([^<]+)", sensor).group(1))
        w = int(re.search(r"<width>(\d+)", sensor).group(1))
        h = int(re.search(r"<height>(\d+)", sensor).group(1))
        cams.append(dict(name=name, position_world=t, R_world_optical=R @ R_SENSOR_OPTICAL,
                         sim_hfov=hfov, sim_size=(w, h)))
    if not cams:
        raise SystemExit(f"no camera models found in {path}")
    return cams


# --------------------------------------------------------------------------
# Lens models
# --------------------------------------------------------------------------
class Lens:
    """Central projection with either the real fisheye (equidistant, given
    diagonal FOV) or the pinhole Gazebo uses.  Pixel centres are at integers."""

    def __init__(self, kind, width, height, dfov_deg=None, hfov_rad=None):
        self.kind, self.w, self.h = kind, int(width), int(height)
        self.cx, self.cy = (self.w - 1) / 2, (self.h - 1) / 2
        if kind == "fisheye":
            self.f = math.hypot(self.w / 2, self.h / 2) / math.radians(dfov_deg / 2)
        else:
            self.f = (self.w / 2) / math.tan(hfov_rad / 2)

    def describe(self):
        if self.kind == "fisheye":
            hf = math.degrees(2 * (self.w / 2) / self.f)
            return f"fisheye {self.w}x{self.h}, f={self.f:.1f} px/rad, HFOV~{hf:.0f} deg"
        return f"pinhole {self.w}x{self.h}, f={self.f:.1f} px, HFOV {math.degrees(2 * math.atan(self.w / 2 / self.f)):.1f} deg"

    def rays(self, u, v):
        """Unit direction in the optical frame for pixel coordinates (arrays)."""
        x, y = (u - self.cx) / self.f, (v - self.cy) / self.f
        if self.kind == "fisheye":
            th = np.hypot(x, y)
            ok = th < math.radians(89.5)
            s = np.where(th > 1e-9, np.sin(th) / np.maximum(th, 1e-9), 1.0)
            d = np.stack([x * s, y * s, np.cos(th)], -1)
        else:
            d = np.stack([x, y, np.ones_like(x)], -1)
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            ok = np.ones(x.shape, bool)
        return d, ok

    def project(self, p_opt):
        """Optical-frame points (N,3) -> pixels (N,2) and off-axis angle (N,)."""
        x, y, z = p_opt[:, 0], p_opt[:, 1], p_opt[:, 2]
        rr = np.hypot(x, y)
        th = np.arctan2(rr, z)
        if self.kind == "fisheye":
            r = self.f * th
        else:
            r = self.f * np.tan(np.minimum(th, math.radians(89)))
        scale = np.where(rr > 1e-12, r / np.maximum(rr, 1e-12), 0)
        return np.stack([self.cx + x * scale, self.cy + y * scale], -1), th


# --------------------------------------------------------------------------
# Synthetic view of the strip on the floor
# --------------------------------------------------------------------------
class Placement:
    """Strip on the floor: sheet top-left corner (white border included) at
    arena (x0, y0).  With yaw 0 the sheet x runs along arena +X and the sheet
    y (down the print) along arena -Y; yaw rotates the sheet about that
    corner.  border_mm is where the chessboard starts inside the sheet."""

    def __init__(self, x0, y0, yaw_deg=0.0, border_mm=0.0):
        self.x0, self.y0, self.yaw = float(x0), float(y0), math.radians(yaw_deg)
        self.c, self.s = math.cos(self.yaw), math.sin(self.yaw)
        self.border_mm = float(border_mm)

    def board_to_arena(self, xy_mm):
        """Board-frame mm (origin at the chessboard's top-left corner) -> arena metres."""
        xy = (np.asarray(xy_mm, float).reshape(-1, 2) + self.border_mm) / 1000.0
        bx, by = xy[:, 0], -xy[:, 1]
        return np.stack([self.x0 + self.c * bx - self.s * by, self.y0 + self.s * bx + self.c * by,
                         np.zeros(len(xy))], -1)

    def arena_to_sheet_mm(self, px, py):
        """Arena metres -> sheet mm (origin at the sheet's top-left corner)."""
        dx, dy = px - self.x0, py - self.y0
        bx = self.c * dx + self.s * dy
        by = -self.s * dx + self.c * dy
        return bx * 1000, -by * 1000


def render_view(cam, lens: Lens, texture: np.ndarray, px_per_mm: float, place: Placement,
                supersample=3, blur_px=0.7, noise=3.0, floor_gray=35, rng=None):
    ss = supersample
    W, H = lens.w, lens.h
    sub = (np.arange(ss) + 0.5) / ss - 0.5
    u = (np.arange(W)[:, None] + sub[None, :]).ravel()
    v = (np.arange(H)[:, None] + sub[None, :]).ravel()
    U, V = np.meshgrid(u, v)
    d, ok = lens.rays(U, V)
    R = np.asarray(cam["R_world_optical"], float)
    C = np.asarray(cam["position_world"], float)
    dw = d @ R.T
    dz = dw[..., 2]
    ok &= dz < -1e-6
    t = np.where(ok, -C[2] / np.where(ok, dz, -1), 0)
    px = C[0] + t * dw[..., 0]
    py = C[1] + t * dw[..., 1]
    bx, by = place.arena_to_sheet_mm(px, py)
    bx, by = bx * px_per_mm, by * px_per_mm
    th, tw = texture.shape
    inside = ok & (bx >= 0) & (bx < tw) & (by >= 0) & (by < th)
    val = np.full(U.shape, float(floor_gray))
    val[inside] = texture[by[inside].astype(int), bx[inside].astype(int)]
    val[~ok] = 0
    img = val.reshape(H, ss, W, ss).mean(axis=(1, 3))
    if blur_px > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_px)
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        img = img + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def detector_params():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        p = cv2.aruco.DetectorParameters_create()
    else:
        p = cv2.aruco.DetectorParameters()
    p.minMarkerPerimeterRate = 0.01
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 63
    p.adaptiveThreshWinSizeStep = 10
    p.polygonalApproxAccuracyRate = 0.05
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 3
    return p


def evaluate_view(board: Board, cam, lens: Lens, img, place: Placement, detector=None):
    """Detect the strip in one frame and compare with where it really is."""
    if detector is None:
        ch_corners, ch_ids, m_corners, m_ids = detect_charuco(board, img)
    else:
        ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(img)
    R = np.asarray(cam["R_world_optical"], float)
    C = np.asarray(cam["position_world"], float)

    def to_pixels(xy_mm):
        p_opt = (place.board_to_arena(xy_mm) - C) @ R
        return lens.project(p_opt)

    res = dict(markers=0, markers_wrong=0, corners=0, corners_wrong=0, corner_rms_px=float("nan"),
               theta_max_deg=float("nan"), cell_px_min=float("nan"), cell_px_med=float("nan"))
    if m_ids is not None and len(m_ids):
        cells, thetas = [], []
        for c, mid in zip(m_corners, m_ids.ravel()):
            c = c.reshape(4, 2)
            k = board.ids.index(int(mid)) if int(mid) in board.ids else None
            per = np.sum(np.linalg.norm(np.roll(c, -1, 0) - c, axis=1))
            cells.append(per / 4 / board.cells)
            if k is None:
                res["markers_wrong"] += 1
                continue
            true_px, th = to_pixels(board.marker_corners_mm(k))
            if np.max(np.linalg.norm(true_px - c, axis=1)) > max(4.0, per / 4 * 0.5):
                res["markers_wrong"] += 1
                continue
            res["markers"] += 1
            thetas.append(math.degrees(th.max()))
        if thetas:
            res["theta_max_deg"] = max(thetas)
        res["cell_px_min"] = float(np.min(cells))
        res["cell_px_med"] = float(np.median(cells))
    if ch_ids is not None and len(ch_ids):
        cc = {cid: (x, y) for cid, x, y in board.chess_corners_mm()}
        true_px, _ = to_pixels([cc[int(i)] for i in ch_ids.ravel()])
        err = np.linalg.norm(true_px - ch_corners.reshape(-1, 2), axis=1)
        good = err < 3.0
        res["corners"] = int(good.sum())
        res["corners_wrong"] = int((~good).sum())
        if good.any():
            res["corner_rms_px"] = float(np.sqrt(np.mean(err[good] ** 2)))
    return res, (ch_corners, ch_ids, m_corners, m_ids)


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------
def board_from_args(a):
    return Board(a.squares_x, a.squares_y, a.square_mm, a.marker_mm, a.dictionary, a.id_offset, a.border_mm)


def add_board_args(p):
    p.add_argument("--squares-x", type=int, default=30)
    p.add_argument("--squares-y", type=int, default=6)
    p.add_argument("--square-mm", type=float, default=375)
    p.add_argument("--marker-mm", type=float, default=280)
    p.add_argument("--dictionary", default="DICT_5X5_250", choices=DICT_NAMES)
    p.add_argument("--id-offset", type=int, default=0, help="first marker ID")
    p.add_argument("--border-mm", type=float, default=None,
                   help="white border of the sheet on every side; default one square")


def cmd_selftest(a):
    board = board_from_args(a)
    ppm = 0.4  # 2.5 mm per pixel keeps every edge of the default board on a pixel
    ours = render_raster(board, ppm)
    ref = opencv_board_image(board.opencv_board(), (ours.shape[1], ours.shape[0]),
                             margin_size=int(round(board.border_mm * ppm)), border_bits=1)
    diff = float(np.mean((ours > 127) != (ref > 127)))
    print(board.describe())
    print(f"raster {ours.shape[1]}x{ours.shape[0]} px, pixels different from OpenCV: {diff * 100:.2f}% "
          f"(0 when every edge lies on the {1 / ppm:g} mm raster; a few percent is edge rounding)")
    cc, cid, mc, mid = detect_charuco(board, ours)
    n_c = 0 if cid is None else len(cid)
    n_m = 0 if mid is None else len(mid)
    print(f"detector on our raster: {n_m}/{len(board.ids)} markers, {n_c}/{(board.squares_x - 1) * (board.squares_y - 1)} corners")
    # corner ids must land on our chessboard corner positions
    truth = {i: ((x + board.border_mm) * ppm, (y + board.border_mm) * ppm) for i, x, y in board.chess_corners_mm()}
    err = max(np.hypot(*(np.array(truth[int(i)]) - c.ravel())) for i, c in zip(cid.ravel(), cc))
    print(f"max corner position error vs our layout: {err:.2f} px")
    ok = diff < 0.03 and n_m == len(board.ids) and n_c == (board.squares_x - 1) * (board.squares_y - 1) and err < 1.0
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


def cmd_generate(a):
    board = board_from_args(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.name
    print(board.describe())
    write_svg(out / f"{name}.svg", board)
    write_pdf(out / f"{name}_full_scale.pdf", board, 1.0)
    write_pdf(out / f"{name}_1to10.pdf", board, 0.1)
    shape = write_png(out / f"{name}_{a.png_px_per_mm:g}px_per_mm.png", board, a.png_px_per_mm)
    print(f"print PNG {shape[1]}x{shape[0]} px at {a.png_px_per_mm:g} px/mm")
    (out / f"{name}_editor_project.json").write_text(
        json.dumps(editor_project(board, name), ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"{name}_map.txt").write_text(drone_map_txt(board, name), encoding="utf-8")

    corners = [dict(id=i, x_mm=x, y_mm=y) for i, x, y in board.chess_corners_mm()]
    markers = [dict(id=mid, corners_mm=board.marker_corners_mm(k).tolist()) for k, mid in enumerate(board.ids)]
    b = board.border_mm
    desc = dict(
        board=board.to_dict(), description=board.describe(),
        sheet_mm=[board.sheet_width_mm, board.sheet_height_mm], border_mm=b,
        opencv=dict(call=f"cv2.aruco.CharucoBoard(({board.squares_x}, {board.squares_y}), {board.square_mm / 1000:g}, "
                         f"{board.marker_mm / 1000:g}, cv2.aruco.getPredefinedDictionary(cv2.aruco.{board.dict_name}), "
                         f"ids=np.arange({board.ids[0]}, {board.ids[-1] + 1}))",
                    legacy_pattern=False),
        frame="board frame: origin at the top-left corner of the chessboard (not of the sheet), x right along the "
              "long side, y down the print, z out of the print; OpenCV object points use this frame in metres. "
              f"sheet = board + {b:g} mm on every side, so sheet_mm = board_mm + {b:g}",
        arena_placement="lay the sheet with its long side along arena +X and the top edge of the print (marker ID "
                        f"{board.ids[0]} side) toward arena +Y; with (x0, y0) the arena position of the sheet's "
                        f"top-left corner: arena = (x0 + (x_board + {b:g}) / 1000, y0 - (y_board + {b:g}) / 1000, 0)",
        chess_corners_mm=corners, markers_mm=markers)
    (out / f"{name}_board.json").write_text(json.dumps(desc, indent=1), encoding="utf-8")

    tex = render_raster(board, a.texture_px_per_mm)
    model_dir = Path(a.gazebo_model_dir) if a.gazebo_model_dir else ROOT / "models" / "charuco_strip"
    write_gazebo_model(model_dir, board, tex)
    print(f"Gazebo model in {model_dir} (texture {tex.shape[1]}x{tex.shape[0]} px)")
    print(f"files in {out}")
    return 0


def cmd_evaluate(a):
    board = board_from_args(a)
    cams = cameras_from_world(Path(a.world))
    if a.cameras:
        cams = [c for c in cams if c["name"] in a.cameras]
    ppm = a.texture_px_per_mm
    texture = render_raster(board, ppm)
    W, H = a.resolution
    if a.lens == "fisheye":
        lens = Lens("fisheye", W, H, dfov_deg=a.dfov_deg)
    elif a.lens == "sim":
        lens = Lens("pinhole", cams[0]["sim_size"][0], cams[0]["sim_size"][1], hfov_rad=cams[0]["sim_hfov"])
    else:
        lens = Lens("pinhole", W, H, hfov_rad=math.radians(a.hfov_deg))
    print(board.describe())
    print(f"lens: {lens.describe()}; world {a.world}; {len(cams)} cameras")
    save = Path(a.save_dir) if a.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    n_m, n_c = len(board.ids), (board.squares_x - 1) * (board.squares_y - 1)
    print(f"{'strip_y':>7} {'camera':>9} {'markers':>9} {'wrong':>5} {'corners':>9} {'rms_px':>6} "
          f"{'theta_max':>9} {'cell_px(min/med)':>16}")
    rows = []
    for yc in a.strip_y:
        x0, y0, yaw = strip_pose(board, a.strip_axis, yc, a.strip_x0)
        place = Placement(x0, y0, math.degrees(yaw), board.border_mm)
        for cam in cams:
            img = render_view(cam, lens, texture, ppm, place, supersample=a.supersample, blur_px=a.blur_px,
                              noise=a.noise)
            res, det = evaluate_view(board, cam, lens, img, place)
            rows.append(dict(strip_y=yc, camera=cam["name"], **res))
            print(f"{yc:7.2f} {cam['name']:>9} {res['markers']:5d}/{n_m:<3d} {res['markers_wrong']:5d} "
                  f"{res['corners']:5d}/{n_c:<3d} {res['corner_rms_px']:6.2f} {res['theta_max_deg']:9.1f} "
                  f"{res['cell_px_min']:7.1f}/{res['cell_px_med']:<7.1f}")
            if save:
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cc, cid, mc, mid = det
                if mid is not None and len(mid):
                    cv2.aruco.drawDetectedMarkers(vis, mc, mid)
                if cid is not None and len(cid):
                    cv2.aruco.drawDetectedCornersCharuco(vis, cc, cid, (0, 0, 255))
                cv2.imwrite(str(save / f"{cam['name']}_y{yc:g}.png"), vis)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(dict(board=board.to_dict(), lens=lens.describe(), world=a.world,
                                                    rows=rows), indent=1))
    if a.csv_out:
        cols = ["strip_y", "camera", "markers", "markers_wrong", "corners", "corners_wrong", "corner_rms_px",
                "theta_max_deg", "cell_px_min", "cell_px_med"]
        with open(a.csv_out, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + ",markers_total,corners_total,lens\n")
            lens_txt = '"' + lens.describe().replace('"', "'") + '"'
            for r in rows:
                vals = [f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
                f.write(",".join(vals) + f",{n_m},{n_c},{lens_txt}\n")
    if a.truth_csv:
        x0, y0, yaw = strip_pose(board, a.strip_axis, a.strip_y[0], a.strip_x0)
        Path(a.truth_csv).write_text(ground_truth_csv(board, Placement(x0, y0, math.degrees(yaw), board.border_mm)),
                                     encoding="utf-8")
    return 0


INCLUDE_RE = re.compile(r"\n[ \t]*<include>\s*<name>charuco_strip</name>.*?</include>[ \t]*", re.S)


def strip_pose(board: Board, axis: str, centre: float, start: float):
    """Gazebo pose of the model origin (sheet top-left corner) for a sheet
    whose long side runs along arena X (axis x, centre line Y=centre) or
    along arena Y (axis y, centre line X=centre); start is the near end."""
    half = board.sheet_height_mm / 2000
    if axis == "x":
        return (start, centre + half, 0.0)
    return (centre - half, start, math.pi / 2)


def place_in_world(text: str, board: Board, axis: str, centre: float, start: float, z: float = 0.001):
    x0, y0, yaw = strip_pose(board, axis, centre, start)
    text = INCLUDE_RE.sub("", text)
    include = (f"\n    <include>\n      <name>charuco_strip</name>\n      <uri>model://charuco_strip</uri>\n"
               f"      <pose>{x0:g} {y0:g} {z:g} 0 0 {yaw:g}</pose>\n    </include>")
    idx = text.rfind("</world>")
    if idx < 0:
        raise SystemExit("no </world> in the world file")
    head = text[:idx].rstrip()
    return head + include + "\n  " + text[idx:], (x0, y0, yaw)


SENSOR_RE = re.compile(r'<sensor name="(camera_\d+)" type="(?:camera|wideanglecamera)">(.*?)</sensor>', re.S)


def wide_angle_world(text: str, width: int, height: int, dfov_deg: float, env_texture_size: int):
    """Rewrite every camera sensor of a world as a Gazebo wide-angle camera
    with an equidistant lens covering dfov_deg on the sensor diagonal, at the
    given resolution.  Returns the new text and the lens used."""
    lens = Lens("fisheye", width, height, dfov_deg=dfov_deg)
    hfov = width / lens.f  # equidistant: the image width spans this angle
    cutoff = math.radians(dfov_deg / 2) + 0.02
    lens_xml = (f"<lens>\n              <!-- IMX219-160: {dfov_deg:g} deg on the sensor diagonal, r = f*theta, "
                f"f = {lens.f:.2f} px/rad -->\n"
                f"              <type>equidistant</type>\n              <scale_to_hfov>true</scale_to_hfov>\n"
                f"              <cutoff_angle>{cutoff:.4f}</cutoff_angle>\n"
                f"              <env_texture_size>{env_texture_size}</env_texture_size>\n            </lens>")
    names = []

    def rewrite(m):
        name, body = m.group(1), m.group(2)
        names.append(name)
        body = re.sub(r"<horizontal_fov>[^<]*</horizontal_fov>", f"<horizontal_fov>{hfov:.6f}</horizontal_fov>", body)
        body = re.sub(r"<width>\d+</width>", f"<width>{width}</width>", body)
        body = re.sub(r"<height>\d+</height>", f"<height>{height}</height>", body)
        body = re.sub(r"\s*<lens>.*?</lens>", "", body, flags=re.S)
        body = body.replace("</camera>", f"  {lens_xml}\n          </camera>")
        return f'<sensor name="{name}" type="wideanglecamera">{body}</sensor>'

    new = SENSOR_RE.sub(rewrite, text)
    if not names:
        raise SystemExit("no camera sensors found in the world")
    return new, lens, names


def cmd_world(a):
    board = board_from_args(a)
    src = Path(a.source)
    text = src.read_text(encoding="utf-8")
    W, H = a.resolution
    text, lens, names = wide_angle_world(text, W, H, a.dfov_deg, a.env_texture_size)
    text, (x0, y0, yaw) = place_in_world(text, board, a.strip_axis, a.strip_y, a.strip_x0)
    note = (f"<!-- Generated by tools/charuco_strip.py world from {src.name}: cameras are wide-angle "
            f"({a.dfov_deg:g} deg diagonal, equidistant, {W}x{H}), the ChArUco sheet lies on the floor. -->\n")
    text = re.sub(r"(<\?xml[^>]*\?>\n)", r"\1" + note, text, count=1) if text.startswith("<?xml") else note + text
    out = Path(a.out)
    out.write_text(text, encoding="utf-8")
    cams = cameras_from_world(out)
    cfg = dict(world=(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out).as_posix(),
               source=str(src.name), camera_model="equidistant fisheye (Gazebo wideanglecamera)",
               image_size=[W, H], diagonal_fov_deg=a.dfov_deg, focal_px_per_rad=lens.f,
               horizontal_fov_rad=W / lens.f, vertical_fov_rad=H / lens.f,
               principal_point=[W / 2, H / 2],
               note="pixel = c + f*theta*(x,y)/|xy| for the optical-frame direction (x right, y down, z forward)",
               strip=dict(board=board.to_dict(), sheet_top_left_arena=[x0, y0], yaw_rad=yaw,
                          arena_from_board="arena = (x0 + (x_board + border)/1000, y0 - (y_board + border)/1000, 0)"),
               cameras=[dict(name=c["name"], position_world=np.round(c["position_world"], 6).tolist(),
                             R_world_optical=np.round(c["R_world_optical"], 9).tolist(),
                             image_topic=f"/cameras/{c['name']}/image") for c in cams])
    cfg_path = Path(a.config_out) if a.config_out else ROOT / "config" / (out.stem + "_cameras.json")
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"{len(names)} cameras -> wideanglecamera {W}x{H}, {lens.describe()}")
    print(f"sheet top-left at arena ({x0:g}, {y0:g}), yaw {math.degrees(yaw):g} deg")
    print(f"wrote {out} and {cfg_path}")
    return 0


def ground_truth_csv(board: Board, place: Placement):
    """Every chessboard corner and marker corner of the sheet in arena metres."""
    lines = ["kind,id,corner,x_board_mm,y_board_mm,x_arena_m,y_arena_m,z_arena_m"]
    for cid, x, y in board.chess_corners_mm():
        p = place.board_to_arena([[x, y]])[0]
        lines.append(f"chess_corner,{cid},,{x:g},{y:g},{p[0]:.4f},{p[1]:.4f},0")
    for k, mid in enumerate(board.ids):
        for ci, (x, y) in enumerate(board.marker_corners_mm(k)):
            p = place.board_to_arena([[x, y]])[0]
            lines.append(f"marker_corner,{mid},{ci},{x:g},{y:g},{p[0]:.4f},{p[1]:.4f},0")
    return "\n".join(lines) + "\n"


def cmd_place(a):
    board = board_from_args(a)
    path = Path(a.world)
    text = path.read_text(encoding="utf-8")
    if a.remove:
        new = INCLUDE_RE.sub("", text)
        print("strip removed" if new != text else "no strip in the world")
    else:
        new, (x0, y0, yaw) = place_in_world(text, board, a.strip_axis, a.strip_y, a.strip_x0)
        print(f"sheet {board.sheet_width_mm / 1000:g} x {board.sheet_height_mm / 1000:g} m along {a.strip_axis.upper()}, "
              f"centre line {a.strip_y:g} m: sheet top-left corner at arena ({x0:g}, {y0:g}), yaw {math.degrees(yaw):g} deg")
        print("run.sh already puts models/ on GZ_SIM_RESOURCE_PATH; marker ID "
              f"{board.ids[0]} must appear at the top-left corner of the print in the GUI")
    if a.dry_run:
        print("(dry run, world not written)")
    else:
        path.write_text(new, encoding="utf-8")
        print(f"wrote {path}")
    return 0


def cmd_detect(a):
    spec = json.loads(Path(a.board).read_text())
    board = Board.from_dict(spec["board"])
    out = Path(a.save_dir) if a.save_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    for path in a.images:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"{path}: cannot read")
            continue
        cc, cid, mc, mid = detect_charuco(board, img)
        n_m = 0 if mid is None else len(mid)
        n_c = 0 if cid is None else len(cid)
        print(f"{path}: {n_m} markers, {n_c} chessboard corners")
        if out:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if n_m:
                cv2.aruco.drawDetectedMarkers(vis, mc, mid)
            if n_c:
                cv2.aruco.drawDetectedCornersCharuco(vis, cc, cid, (0, 0, 255))
            cv2.imwrite(str(out / (Path(path).stem + "_detected.png")), vis)
        if a.json_out and n_c:
            rec = dict(image=path, corner_ids=cid.ravel().tolist(), corners_px=cc.reshape(-1, 2).tolist(),
                       marker_ids=[] if not n_m else mid.ravel().tolist())
            with open(a.json_out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="compare our raster with OpenCV's generateImage")
    add_board_args(s)
    s.set_defaults(func=cmd_selftest)

    g = sub.add_parser("generate", help="write print files, editor project, board.json and Gazebo model")
    add_board_args(g)
    g.add_argument("--out", default=str(ROOT / "assets" / "charuco_strip"))
    g.add_argument("--name", default="charuco_strip_12x3")
    g.add_argument("--png-px-per-mm", type=float, default=2.0, help="print raster resolution")
    g.add_argument("--texture-px-per-mm", type=float, default=0.4, help="Gazebo texture resolution")
    g.add_argument("--gazebo-model-dir", default=None)
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("evaluate", help="synthetic detection through the world's cameras")
    add_board_args(e)
    e.add_argument("--world", default=str(ROOT / "worlds" / "mocap_arena_l2.sdf"))
    e.add_argument("--cameras", nargs="*", default=None)
    e.add_argument("--lens", choices=["fisheye", "pinhole", "sim"], default="fisheye",
                   help="fisheye = real IMX219-160 (equidistant, --dfov-deg); sim = the world's pinhole")
    e.add_argument("--dfov-deg", type=float, default=DEFAULT_DFOV_DEG)
    e.add_argument("--hfov-deg", type=float, default=105.3)
    e.add_argument("--resolution", type=int, nargs=2, default=[1280, 960], metavar=("W", "H"))
    e.add_argument("--strip-y", type=float, nargs="+", default=[4.0, 8.0],
                   help="arena Y of the strip centre line for each placement")
    e.add_argument("--strip-x0", type=float, default=0.0, help="arena coordinate of the strip's near end")
    e.add_argument("--strip-axis", choices=["x", "y"], default="x",
                   help="long side along arena X (default) or Y; --strip-y then gives the centre-line X")
    e.add_argument("--texture-px-per-mm", type=float, default=0.4)
    e.add_argument("--supersample", type=int, default=3)
    e.add_argument("--blur-px", type=float, default=0.7)
    e.add_argument("--noise", type=float, default=3.0)
    e.add_argument("--save-dir", default=None, help="write the synthetic frames with detections")
    e.add_argument("--json-out", default=None)
    e.add_argument("--csv-out", default=None, help="one row per camera and placement")
    e.add_argument("--truth-csv", default=None,
                   help="arena coordinates of every corner for the first --strip-y placement")
    e.set_defaults(func=cmd_evaluate)

    wo = sub.add_parser("world", help="derive a world with wide-angle cameras and the sheet on the floor")
    add_board_args(wo)
    wo.add_argument("--source", default=str(ROOT / "worlds" / "mocap_arena_l2.sdf"))
    wo.add_argument("--out", default=str(ROOT / "worlds" / "mocap_arena_charuco.sdf"))
    wo.add_argument("--config-out", default=None, help="camera/lens/sheet description JSON")
    wo.add_argument("--resolution", type=int, nargs=2, default=[1640, 1232], metavar=("W", "H"))
    wo.add_argument("--dfov-deg", type=float, default=DEFAULT_DFOV_DEG)
    wo.add_argument("--env-texture-size", type=int, default=2048, help="cube-map face size of the wide-angle camera")
    wo.add_argument("--strip-y", type=float, default=6.0)
    wo.add_argument("--strip-x0", type=float, default=0.0)
    wo.add_argument("--strip-axis", choices=["x", "y"], default="x")
    wo.set_defaults(func=cmd_world)

    w = sub.add_parser("place", help="put the strip model into a world file (or take it out)")
    add_board_args(w)
    w.add_argument("--world", default=str(ROOT / "worlds" / "mocap_arena_l2.sdf"))
    w.add_argument("--strip-y", type=float, default=6.0, help="arena coordinate of the strip centre line")
    w.add_argument("--strip-x0", type=float, default=0.0, help="arena coordinate of the strip's near end")
    w.add_argument("--strip-axis", choices=["x", "y"], default="x")
    w.add_argument("--remove", action="store_true")
    w.add_argument("--dry-run", action="store_true")
    w.set_defaults(func=cmd_place)

    d = sub.add_parser("detect", help="run the detector on photos of the printed strip")
    d.add_argument("--board", required=True, help="*_board.json from generate")
    d.add_argument("images", nargs="+")
    d.add_argument("--save-dir", default=None)
    d.add_argument("--json-out", default=None, help="append one JSON line per image")
    d.set_defaults(func=cmd_detect)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
