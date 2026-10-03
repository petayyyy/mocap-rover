"""ChArUco calibration of the real stand: fisheye intrinsics and the poses of all cameras.

Two procedures, both on full 1640x1232 sensor frames:

* **Intrinsics, one camera at a time.**  A hand-held ChArUco board (by default
  the A4 board of sverk-ros2 ``camera_calibration``: 11x8 squares of 22 mm,
  16 mm markers, DICT_4X4_50) in ~35 poses.  The same captures give three
  calibrations: the sensor resolution 1640x1232 and the frames reduced to
  820x616 and 640x480 exactly as the camera node reduces its small stream
  (``pi_cam.camera_node.scale_luma``: 2x2 average, then area resize).  Each is
  a full fisheye (equidistant, OpenCV ``cv2.fisheye``) calibration of the
  reduced images, with the iterative outlier rejection of sverk's node: drop
  ill-conditioned views, then the worst view while it is above
  ``max_frame_err`` px.

* **Extrinsics, all cameras together.**  The 12x3 m floor strip
  (``assets/charuco_strip``) in one or more placements.  The reference
  placement is measured on the floor and defines the arena frame; the other
  placements are free (x, y, yaw) on the floor plane and are solved together
  with the six camera poses by minimising the fisheye reprojection error of
  every detected corner (robust Huber loss).  A camera that never shares a
  placement with an already solved camera cannot be placed and is reported.

Arena frame as everywhere in the repository: X, Y on the floor, Z up;
``p_world = R_world_optical @ p_optical + position_world``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pi_cam.camera_node import scale_luma
from tools.charuco_strip import detector_params

ROOT = Path(__file__).resolve().parents[1]
STRIP_BOARD_JSON = ROOT / "assets" / "charuco_strip" / "charuco_strip_12x3_board.json"
SENSOR_SIZE = (1640, 1232)
INTRINSIC_SIZES = ((1640, 1232), (820, 616), (640, 480))
IMX219_160_DFOV_DEG = 160.0
RECOMMENDED_VIEWS = 35


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

@dataclass
class Detection:
    ids: np.ndarray            # (N,) chessboard corner ids
    points: np.ndarray         # (N, 2) pixels, float64
    markers: int               # ArUco markers found
    marker_corners: list = field(default_factory=list)

    @property
    def count(self):
        return int(len(self.ids))


def _make_board(squares, square_m, marker_m, dictionary, ids, legacy):
    size = (int(squares[0]), int(squares[1]))
    if ids is None:
        board = cv2.aruco.CharucoBoard(size, float(square_m), float(marker_m), dictionary)
    else:
        board = cv2.aruco.CharucoBoard(size, float(square_m), float(marker_m), dictionary,
                                       np.asarray(ids, np.int32))
    if legacy and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board


class CharucoTarget:
    """One ChArUco board and its detector.

    ``legacy``: True / False, or ``"auto"`` -- boards printed by calib.io and
    by OpenCV < 4.6 with an even number of rows use the old corner layout.
    In auto mode both layouts are tried until one frame decides it.
    """

    def __init__(self, squares=(11, 8), square_m=0.022, marker_m=0.016,
                 dictionary="DICT_4X4_50", ids=None, legacy="auto", min_corners=12):
        self.squares = (int(squares[0]), int(squares[1]))
        self.square_m, self.marker_m = float(square_m), float(marker_m)
        self.dictionary_name = dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        self.min_corners = int(min_corners)
        self.ids = None if ids is None else [int(i) for i in ids]
        self.legacy = legacy
        self._detectors = {}
        for flag in ((False, True) if legacy == "auto" else (bool(legacy),)):
            board = _make_board(self.squares, square_m, marker_m, self.dictionary, ids, flag)
            detector = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(),
                                                 detector_params())
            self._detectors[flag] = (board, detector)
        self.decided = None if legacy == "auto" else bool(legacy)

    @property
    def corner_count(self):
        return (self.squares[0] - 1) * (self.squares[1] - 1)

    def corners_m(self, legacy=None):
        board = self._detectors[self.decided if legacy is None else legacy][0]
        return np.asarray(board.getChessboardCorners(), np.float64)

    def object_points(self, ids):
        """Board-frame metres of the given corner ids: x right, y down the print, z 0."""
        flag = self.decided if self.decided is not None else next(iter(self._detectors))
        return self.corners_m(flag)[np.asarray(ids, int)]

    def _detect_with(self, flag, gray):
        corners, ids, marker_corners, marker_ids = self._detectors[flag][1].detectBoard(gray)
        markers = 0 if marker_ids is None else len(marker_ids)
        if ids is None or corners is None or len(ids) == 0:
            return Detection(np.zeros(0, int), np.zeros((0, 2)), markers, list(marker_corners or []))
        return Detection(ids.reshape(-1).astype(int), corners.reshape(-1, 2).astype(np.float64),
                         markers, list(marker_corners or []))

    def detect(self, gray) -> Detection:
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if self.decided is not None:
            return self._detect_with(self.decided, gray)
        results = {flag: self._detect_with(flag, gray) for flag in self._detectors}
        best = max(results, key=lambda f: results[f].count)
        other = results[not best].count
        if results[best].count >= self.min_corners and results[best].count > other:
            self.decided = best
        return results[best]

    def describe(self):
        layout = {None: "auto", True: "legacy", False: "new"}[self.decided]
        return (f"ChArUco {self.squares[0]}x{self.squares[1]}, square {self.square_m * 1000:g} mm, "
                f"marker {self.marker_m * 1000:g} mm, {self.dictionary_name}, layout {layout}")

    def to_dict(self):
        return {"squares": list(self.squares), "square_m": self.square_m, "marker_m": self.marker_m,
                "dictionary": self.dictionary_name, "legacy": self.decided if self.decided is not None
                else self.legacy, "min_corners": self.min_corners}


def strip_target(board_json=STRIP_BOARD_JSON, min_corners=8):
    """The 12x3 m floor strip (30x6 squares of 375 mm, DICT_5X5_250, new layout)."""
    spec = json.loads(Path(board_json).read_text())["board"]
    squares = (spec["squares_x"], spec["squares_y"])
    n_markers = squares[0] * squares[1] // 2
    target = CharucoTarget(squares, spec["square_mm"] / 1000.0, spec["marker_mm"] / 1000.0,
                           spec["dictionary"], ids=range(spec.get("id_offset", 0),
                                                         spec.get("id_offset", 0) + n_markers),
                           legacy=False, min_corners=min_corners)
    target.border_m = spec.get("border_mm", spec["square_mm"]) / 1000.0
    return target


def resize_like_node(gray, size):
    """A full sensor frame reduced to ``size`` the way the camera node does it."""
    w, h = int(size[0]), int(size[1])
    if (gray.shape[1], gray.shape[0]) == (w, h):
        return gray
    return scale_luma(gray, w, h)


# --------------------------------------------------------------------------
# Fisheye intrinsics
# --------------------------------------------------------------------------

def initial_fisheye_k(size, dfov_deg=IMX219_160_DFOV_DEG):
    """Equidistant guess: the image diagonal spans ``dfov_deg``."""
    w, h = size
    f = math.hypot(w / 2.0, h / 2.0) / math.radians(dfov_deg / 2.0)
    return np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]])


def fisheye_theta(theta_d, d):
    """Invert theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8) by Newton."""
    k1, k2, k3, k4 = (float(v) for v in np.asarray(d).ravel()[:4])
    theta = float(theta_d)
    for _ in range(30):
        t2 = theta * theta
        f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
        df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        step = f / df
        theta -= step
        if abs(step) < 1e-12:
            break
    return theta


def field_of_view_deg(k, d, size):
    w, h = size
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    hfov = fisheye_theta(cx / fx, d) + fisheye_theta((w - 1 - cx) / fx, d)
    vfov = fisheye_theta(cy / fy, d) + fisheye_theta((h - 1 - cy) / fy, d)
    rd = math.hypot(cx / fx, cy / fy)
    return {"horizontal": math.degrees(hfov), "vertical": math.degrees(vfov),
            "diagonal": math.degrees(2 * fisheye_theta(rd, d))}


def _per_view_errors(objp, imgp, k, d, rvecs, tvecs):
    errors = []
    for o, i, rv, tv in zip(objp, imgp, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(o, rv, tv, k, d)
        errors.append(float(np.linalg.norm((i - proj).reshape(-1, 2), axis=1).mean()))
    return errors


def _fisheye_fit(objp, imgp, size):
    """One fisheye calibration; drops ill-conditioned views (positions returned)."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    base = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
            | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
    keep = list(range(len(objp)))
    dropped = []
    while True:
        if len(keep) < 3:
            raise RuntimeError("too few usable views (ill-conditioned)")
        o = [objp[j] for j in keep]
        i = [imgp[j] for j in keep]
        k = initial_fisheye_k(size)
        d = np.zeros((4, 1))
        try:
            rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
                o, i, size, k, d, None, None, flags=base | cv2.fisheye.CALIB_CHECK_COND,
                criteria=criteria)
            break
        except cv2.error as exc:
            m = re.search(r"input array (\d+)", str(exc))
            if m and int(m.group(1)) < len(keep):
                dropped.append(keep.pop(int(m.group(1))))
                continue
            k = initial_fisheye_k(size)
            d = np.zeros((4, 1))
            rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
                o, i, size, k, d, None, None, flags=base, criteria=criteria)
            break
    errors = _per_view_errors([objp[j] for j in keep], [imgp[j] for j in keep], k, d, rvecs, tvecs)
    return float(rms), k, d, errors, keep, dropped


def calibrate_fisheye(object_points, image_points, size, max_frame_err=2.0, min_frames=None):
    """sverk-style fisheye calibration with iterative rejection of bad views.

    ``object_points``/``image_points``: one (N, 3) / (N, 2) array per view.
    Returns a dict with K, D, rms, mean per-view error and what was dropped;
    view indices refer to the input order.
    """
    n = len(object_points)
    objp = [np.asarray(o, np.float64).reshape(-1, 1, 3) for o in object_points]
    imgp = [np.asarray(i, np.float64).reshape(-1, 1, 2) for i in image_points]
    min_frames = RECOMMENDED_VIEWS if min_frames is None else int(min_frames)
    floor = max(3, min(min_frames // 2, n))
    active = list(range(n))
    illcond, outliers = [], []
    while True:
        if len(active) < 3:
            raise RuntimeError(f"too few usable views: {len(active)} of {n}")
        rms, k, d, errors, kept, dropped = _fisheye_fit([objp[j] for j in active],
                                                        [imgp[j] for j in active], size)
        if dropped:
            gone = {active[p] for p in dropped}
            illcond.extend(sorted(gone))
            active = [j for j in active if j not in gone]
            continue
        if max_frame_err > 0 and len(active) > floor:
            worst = int(np.argmax(errors))
            if errors[worst] > max_frame_err:
                outliers.append(active.pop(worst))
                continue
        break
    return {"K": k, "D": d.reshape(4), "rms": rms, "reprojection_error": float(np.mean(errors)),
            "per_view_error": dict(zip(active, errors)), "views_used": active,
            "views_total": n, "dropped_outliers": outliers, "dropped_illcond": illcond,
            "image_size": list(size), "fov_deg": field_of_view_deg(k, d, size)}


def scale_intrinsics(k, from_size, to_size):
    """K of a reduced image (pixel centres preserved), for comparison."""
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    out = np.array(k, float)
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = (k[0, 2] + 0.5) * sx - 0.5
    out[1, 2] = (k[1, 2] + 0.5) * sy - 0.5
    return out


def calibrate_intrinsics(images, target: CharucoTarget, sizes=INTRINSIC_SIZES, max_frame_err=2.0,
                         progress=None):
    """Full-resolution grey frames of one camera -> one calibration per size.

    Every size is calibrated from its own reduced images (corners detected
    there), not by scaling K; the K of the full size scaled down is reported
    next to it as a cross-check.
    """
    results = []
    for size in sizes:
        objp, imgp, used_images = [], [], []
        for index, image in enumerate(images):
            small = resize_like_node(image, size)
            det = target.detect(small)
            if progress:
                progress(f"{size[0]}x{size[1]}: view {index + 1}/{len(images)}, {det.count} corners")
            if det.count >= target.min_corners:
                objp.append(target.object_points(det.ids))
                imgp.append(det.points)
                used_images.append(index)
        entry = {"size": list(size), "views_detected": len(objp), "views_captured": len(images)}
        try:
            result = calibrate_fisheye(objp, imgp, tuple(size), max_frame_err)
        except (RuntimeError, cv2.error) as exc:
            entry["error"] = str(exc)
            results.append(entry)
            continue
        result["views_used"] = [used_images[j] for j in result["views_used"]]
        result["dropped_outliers"] = [used_images[j] for j in result["dropped_outliers"]]
        result["dropped_illcond"] = [used_images[j] for j in result["dropped_illcond"]]
        result["per_view_error"] = {used_images[j]: e for j, e in result["per_view_error"].items()}
        entry.update(result)
        results.append(entry)
    full = next((r for r in results if "K" in r), None)
    if full is not None:
        for r in results:
            if "K" in r and r is not full:
                ks = scale_intrinsics(full["K"], full["size"], r["size"])
                r["K_scaled_from"] = {"size": full["size"], "K": ks.tolist(),
                                      "delta_fx_px": float(r["K"][0, 0] - ks[0, 0]),
                                      "delta_cx_px": float(r["K"][0, 2] - ks[0, 2]),
                                      "delta_cy_px": float(r["K"][1, 2] - ks[1, 2])}
    return results


def _yaml_matrix(name, rows, cols, data):
    values = ", ".join(f"{float(v):.10g}" for v in np.asarray(data).ravel())
    return f"{name}:\n  rows: {rows}\n  cols: {cols}\n  data: [{values}]\n"


def write_intrinsics(out_dir, camera_id, results, target: CharucoTarget, sensor_name="imx219"):
    """ROS/sverk YAML ``<sensor>_<w>x<h>.yaml`` plus JSON per size; returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    summary = {"camera_id": camera_id, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "target": target.to_dict(), "results": []}
    for r in results:
        w, h = r["size"]
        if "K" not in r:
            summary["results"].append({k: v for k, v in r.items()})
            continue
        k, d = np.asarray(r["K"]), np.asarray(r["D"]).reshape(4)
        p = np.hstack([k, np.zeros((3, 1))])
        yaml = (f"calibration_date: \"{summary['created']}\"\n"
                f"camera_name: {camera_id}\n"
                f"image_width: {w}\nimage_height: {h}\n"
                f"distortion_model: equidistant\ncamera_model: fisheye\n"
                f"board_type: charuco\nboard_width: {target.squares[0]}\nboard_height: {target.squares[1]}\n"
                f"square_size: {target.square_m:g}\nmarker_size: {target.marker_m:g}\n"
                f"aruco_dict: {target.dictionary_name}\n"
                f"num_calibration_frames: {len(r['views_used'])}\n"
                f"num_captured_frames: {r['views_captured']}\n"
                f"dropped_outliers: {len(r['dropped_outliers'])}\n"
                f"dropped_illcond: {len(r['dropped_illcond'])}\n"
                f"reprojection_error: {r['reprojection_error']:.6f}\n"
                f"rms: {r['rms']:.6f}\n"
                + _yaml_matrix("camera_matrix", 3, 3, k)
                + _yaml_matrix("distortion_coefficients", 1, 4, d)
                + _yaml_matrix("rectification_matrix", 3, 3, np.eye(3))
                + _yaml_matrix("projection_matrix", 3, 4, p))
        yaml_path = out_dir / f"{sensor_name}_{w}x{h}.yaml"
        yaml_path.write_text(yaml)
        entry = {"camera_id": camera_id, "image_size": [w, h], "distortion_model": "fisheye",
                 "K": k.ravel().tolist(), "D": d.tolist(), "rms": r["rms"],
                 "reprojection_error": r["reprojection_error"], "fov_deg": r["fov_deg"],
                 "views_used": len(r["views_used"]), "views_captured": r["views_captured"],
                 "views_detected": r["views_detected"],
                 "dropped_outliers": r["dropped_outliers"], "dropped_illcond": r["dropped_illcond"],
                 "K_scaled_from": r.get("K_scaled_from")}
        json_path = out_dir / f"{camera_id}_{w}x{h}.json"
        json_path.write_text(json.dumps(entry, indent=2))
        summary["results"].append(entry)
        written += [yaml_path, json_path]
    (out_dir / "intrinsics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return written


def load_intrinsics(source, camera_id, size=SENSOR_SIZE):
    """K (3x3), D (4,) for ``camera_id`` at ``size``.

    ``source``: a JSON written by ``write_intrinsics``, a directory holding
    ``<camera_id>_<w>x<h>.json`` (directly or under ``<camera_id>/``), or a
    runtime_cameras.json (K/D of that camera).
    """
    source = Path(source)
    name = f"{camera_id}_{size[0]}x{size[1]}.json"
    candidates = [source] if source.is_file() else [source / name, source / camera_id / name]
    for path in candidates:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        if "cameras" in data:
            cam = next((c for c in data["cameras"] if c["name"] == camera_id), None)
            if cam is None:
                continue
            if tuple(cam["image_size"]) != tuple(size):
                raise ValueError(f"{path}: {camera_id} is {cam['image_size']}, need {size}")
            return np.asarray(cam["K"], float).reshape(3, 3), np.asarray(cam["D"], float).reshape(4)
        if tuple(data["image_size"]) != tuple(size):
            continue
        return np.asarray(data["K"], float).reshape(3, 3), np.asarray(data["D"], float).reshape(4)
    return None


# --------------------------------------------------------------------------
# Extrinsics: all cameras on the floor strip
# --------------------------------------------------------------------------

@dataclass
class StripPlacement:
    """The strip's sheet top-left corner (marker ID 0 side, white border included)
    at arena (x0, y0) m, long side along arena +X rotated by ``yaw_deg``; the
    print's top edge toward +Y.  ``known``: measured, fixes the arena frame."""
    x0: float
    y0: float
    yaw_deg: float = 0.0
    known: bool = False


@dataclass
class StripObservation:
    camera_id: str
    placement: int
    ids: np.ndarray
    points: np.ndarray


def placement_pose(x0, y0, yaw, border_m):
    """Board frame (x right, y down the print, z into the floor) -> arena: R, t."""
    c, s = math.cos(yaw), math.sin(yaw)
    r = np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])
    t = np.array([x0 + (c + s) * border_m, y0 + (s - c) * border_m, 0.0])
    return r, t


def placement_from_pose(r, t, border_m):
    yaw = math.atan2(r[1, 0], r[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    return float(t[0] - (c + s) * border_m), float(t[1] - (s - c) * border_m), yaw


def _undistorted(points, k, d):
    return cv2.fisheye.undistortPoints(np.asarray(points, np.float64).reshape(-1, 1, 2),
                                       k, np.asarray(d, np.float64).reshape(4, 1)).reshape(-1, 2)


def single_view_pose(object_points, image_points, k, d):
    """Camera <- board pose from one planar view through the fisheye model."""
    norm = _undistorted(image_points, k, d)
    obj = np.asarray(object_points, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("planar PnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(obj, norm, np.eye(3), None, rvec, tvec)
    r, _ = cv2.Rodrigues(rvec)
    return r, tvec.reshape(3)


def _invert(r, t):
    return r.T, -r.T @ t


def _compose(ra, ta, rb, tb):
    return ra @ rb, ra @ tb + ta


def solve_extrinsics(observations, intrinsics, placements, target: CharucoTarget,
                     min_corners=8, loss_scale_px=1.0):
    """All camera poses (and the free placements) from strip observations.

    ``intrinsics``: {camera_id: (K, D)} at the image size of the observations.
    Returns a dict: per camera R_world_optical, position_world, residuals,
    covariance-based sigmas, single-view spread; per placement its pose;
    unsolved cameras with the reason.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    border = getattr(target, "border_m", 0.0)
    obs = [o for o in observations if len(o.ids) >= min_corners and o.camera_id in intrinsics]
    if not any(p.known for p in placements):
        raise ValueError("at least one placement must be measured (known)")
    board_pts = {id(o): target.object_points(o.ids) for o in obs}
    single = {}
    for o in obs:
        k, d = intrinsics[o.camera_id]
        try:
            single[id(o)] = single_view_pose(board_pts[id(o)], o.points, k, d)
        except (RuntimeError, cv2.error):
            pass
    obs = [o for o in obs if id(o) in single]

    plc_pose = {i: placement_pose(p.x0, p.y0, math.radians(p.yaw_deg), border)
                for i, p in enumerate(placements) if p.known}
    cam_pose = {}                                         # camera -> (R_cw, t_cw)
    changed = True
    while changed:
        changed = False
        for o in sorted(obs, key=lambda o: -len(o.ids)):
            r_cb, t_cb = single[id(o)]
            if o.placement in plc_pose and o.camera_id not in cam_pose:
                r_ab, t_ab = plc_pose[o.placement]
                cam_pose[o.camera_id] = _compose(r_cb, t_cb, *_invert(r_ab, t_ab))
                changed = True
            elif o.camera_id in cam_pose and o.placement not in plc_pose:
                r_ac, t_ac = _invert(*cam_pose[o.camera_id])
                r_ab, t_ab = _compose(r_ac, t_ac, r_cb, t_cb)
                x0, y0, yaw = placement_from_pose(r_ab, t_ab, border)
                plc_pose[o.placement] = placement_pose(x0, y0, yaw, border)
                changed = True
    cams = sorted(cam_pose)
    free = sorted(i for i in plc_pose if not placements[i].known)
    obs = [o for o in obs if o.camera_id in cam_pose and o.placement in plc_pose]
    if not cams:
        raise RuntimeError("no camera sees a measured placement")

    x0 = []
    for c in cams:
        rvec, _ = cv2.Rodrigues(cam_pose[c][0])
        x0 += list(rvec.ravel()) + list(cam_pose[c][1])
    for i in free:
        x0 += list(placement_from_pose(*plc_pose[i], border))
    x0 = np.asarray(x0, float)
    cam_index = {c: 6 * j for j, c in enumerate(cams)}
    free_index = {i: 6 * len(cams) + 3 * j for j, i in enumerate(free)}

    def unpack_placement(x, i):
        if i in free_index:
            a = free_index[i]
            return placement_pose(x[a], x[a + 1], x[a + 2], border)
        p = placements[i]
        return placement_pose(p.x0, p.y0, math.radians(p.yaw_deg), border)

    def residuals(x):
        out = []
        for o in obs:
            a = cam_index[o.camera_id]
            r_ab, t_ab = unpack_placement(x, o.placement)
            world = board_pts[id(o)] @ r_ab.T + t_ab
            k, d = intrinsics[o.camera_id]
            proj, _ = cv2.fisheye.projectPoints(world.reshape(-1, 1, 3), x[a:a + 3], x[a + 3:a + 6],
                                                k, np.asarray(d, float).reshape(4, 1))
            out.append((proj.reshape(-1, 2) - o.points).ravel())
        return np.concatenate(out)

    rows = sum(2 * len(o.ids) for o in obs)
    sparsity = lil_matrix((rows, len(x0)), dtype=int)
    row = 0
    for o in obs:
        n = 2 * len(o.ids)
        a = cam_index[o.camera_id]
        sparsity[row:row + n, a:a + 6] = 1
        if o.placement in free_index:
            b = free_index[o.placement]
            sparsity[row:row + n, b:b + 3] = 1
        row += n
    fit = least_squares(residuals, x0, jac_sparsity=sparsity, loss="huber",
                        f_scale=loss_scale_px, x_scale="jac", method="trf")
    res = fit.fun.reshape(-1, 2)
    err = np.linalg.norm(res, axis=1)
    dof = max(1, len(fit.fun) - len(x0))
    sigma2 = float(np.sum(fit.fun ** 2) / dof)
    jac = fit.jac.toarray() if hasattr(fit.jac, "toarray") else fit.jac
    try:
        cov = np.linalg.pinv(jac.T @ jac) * sigma2
    except np.linalg.LinAlgError:
        cov = None

    result = {"cameras": {}, "placements": [], "unsolved": {}, "loss": "huber",
              "loss_scale_px": loss_scale_px, "observations": len(obs),
              "corners": int(len(err)), "rms_px": float(np.sqrt(np.mean(err ** 2))),
              "p95_px": float(np.percentile(err, 95)), "success": bool(fit.success)}
    offsets = {}
    row = 0
    for o in obs:
        offsets[id(o)] = (row, row + len(o.ids))
        row += len(o.ids)
    for c in cams:
        a = cam_index[c]
        r_cw, _ = cv2.Rodrigues(fit.x[a:a + 3])
        t_cw = fit.x[a + 3:a + 6]
        r_wo, pos = _invert(r_cw, t_cw)
        mine = [o for o in obs if o.camera_id == c]
        e = np.concatenate([err[offsets[id(o)][0]:offsets[id(o)][1]] for o in mine])
        sigma_mm = sigma_deg = None
        if cov is not None:
            block = cov[a:a + 6, a:a + 6]
            jac_pos = np.zeros((3, 6))
            for q in range(6):
                dx = np.zeros(6)
                dx[q] = 1e-6
                rp, _ = cv2.Rodrigues(fit.x[a:a + 3] + dx[:3])
                jac_pos[:, q] = ((-rp.T @ (t_cw + dx[3:])) - pos) / 1e-6
            sigma_mm = (np.sqrt(np.maximum(np.diag(jac_pos @ block @ jac_pos.T), 0)) * 1000).tolist()
            sigma_deg = float(np.degrees(np.sqrt(max(np.trace(block[:3, :3]), 0.0))))
        spread = []
        for o in mine:
            r_cb, t_cb = single[id(o)]
            r_ab, t_ab = unpack_placement(fit.x, o.placement)
            r_ac, t_ac = _compose(r_ab, t_ab, *_invert(r_cb, t_cb))
            spread.append(float(np.linalg.norm(t_ac - pos) * 1000))
        axis = r_wo[:, 2]
        result["cameras"][c] = {
            "position_world": pos.tolist(), "R_world_optical": r_wo.tolist(),
            "tilt_from_nadir_deg": float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))),
            "image_x_heading_deg": float(np.degrees(math.atan2(r_wo[1, 0], r_wo[0, 0]))),
            "corners": int(len(e)), "views": {str(o.placement): int(len(o.ids)) for o in mine},
            "reproj_p50_px": float(np.percentile(e, 50)), "reproj_p95_px": float(np.percentile(e, 95)),
            "reproj_max_px": float(e.max()), "sigma_position_mm": sigma_mm,
            "sigma_rotation_deg": sigma_deg, "single_view_spread_mm": spread}
    for i, p in enumerate(placements):
        if i in plc_pose:
            r_ab, t_ab = unpack_placement(fit.x, i)
            x, y, yaw = placement_from_pose(r_ab, t_ab, border)
            result["placements"].append({"index": i, "x0": x, "y0": y, "yaw_deg": math.degrees(yaw),
                                         "known": p.known})
        else:
            result["placements"].append({"index": i, "solved": False, "known": p.known})
    for c in intrinsics:
        if c not in cam_pose:
            seen = [o for o in observations if o.camera_id == c]
            result["unsolved"][c] = ("no strip corners" if not seen else
                                     "never shares a placement with a solved camera"
                                     if any(len(o.ids) >= min_corners for o in seen)
                                     else f"fewer than {min_corners} corners in every placement")
    names = sorted(result["cameras"])
    result["distances_m"] = {
        a: {b: float(np.linalg.norm(np.subtract(result["cameras"][a]["position_world"],
                                                result["cameras"][b]["position_world"])))
            for b in names} for a in names}
    return result


def runtime_config(base_config, solution, intrinsics, size=SENSOR_SIZE, note="charuco strip"):
    """``base_config`` (runtime_cameras.json) with the solved poses and the intrinsics."""
    config = json.loads(json.dumps(base_config))
    for cam in config["cameras"]:
        name = cam["name"]
        if name in solution["cameras"]:
            s = solution["cameras"][name]
            cam["position_world"] = s["position_world"]
            cam["R_world_optical"] = s["R_world_optical"]
            cam["calibrated"] = True
        else:
            cam["calibrated"] = False
        if name in intrinsics:
            k, d = intrinsics[name]
            cam["K"] = np.asarray(k, float).ravel().tolist()
            cam["D"] = np.asarray(d, float).ravel().tolist()
            cam["distortion_model"] = "fisheye"
            cam["image_size"] = list(size)
    config["role"] = "image_calibrated"
    digest = hashlib.sha1(json.dumps(config["cameras"], sort_keys=True).encode()).hexdigest()[:12]
    config["calibration_version"] = f"charuco-{digest}"
    config["calibration_note"] = f"{note}, {time.strftime('%Y-%m-%d %H:%M:%S')}"
    return config


def compare_to_truth(solution, truth_config):
    """Position / rotation error of each solved camera against a reference config."""
    truth = {c["name"]: c for c in truth_config["cameras"]}
    out = {}
    for name, s in solution["cameras"].items():
        if name not in truth:
            continue
        dp = np.subtract(s["position_world"], truth[name]["position_world"])
        dr = np.asarray(s["R_world_optical"]) @ np.asarray(truth[name]["R_world_optical"]).T
        angle = math.degrees(math.acos(np.clip((np.trace(dr) - 1) / 2, -1, 1)))
        out[name] = {"position_error_mm": float(np.linalg.norm(dp) * 1000),
                     "rotation_error_deg": angle}
    return out


def solve_session(session, base_config, strip, intrinsics, sources=None, truth=None):
    """Re-detect the saved placement snapshots and solve; writes the results."""
    session = Path(session)
    manifest = json.loads((session / "extrinsics" / "manifest.json").read_text())
    entries = [p for p in manifest["placements"] if not p.get("deleted")]
    placements, observations = [], []
    for new_index, p in enumerate(entries):
        placements.append(StripPlacement(p["x0"], p["y0"], p["yaw_deg"], p["known"]))
        for cid, info in p["cameras"].items():
            gray = cv2.imread(str(session / info["image"]), cv2.IMREAD_GRAYSCALE)
            det = strip.detect(gray)
            observations.append(StripObservation(cid, new_index, det.ids, det.points))
    solution = solve_extrinsics(observations, intrinsics, placements, strip)
    solution["intrinsics_source"] = sources or {}
    if truth is not None:
        solution["vs_truth"] = compare_to_truth(solution, truth)
    config = runtime_config(base_config, solution, intrinsics)
    out = session / "extrinsics"
    (out / "runtime_cameras.json").write_text(json.dumps(config, indent=2))
    (out / "calibration_report.json").write_text(json.dumps(solution, indent=2))
    solution["files"] = [str(out / "runtime_cameras.json"), str(out / "calibration_report.json")]
    return solution
