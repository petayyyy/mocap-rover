#!/usr/bin/env python3
"""Check ChArUco detection and camera poses in the fisheye arena.

By default the script subscribes to all camera topics in a running Gazebo
world. ``--camera-source`` accepts real /dev/video, RTSP and HTTP/MJPEG
streams; ``--images-dir`` reads camera_N snapshots. ``--live-3d`` estimates
the poses once and opens a mouse-rotatable 3D view of that fixed result.
The pose is estimated from detected ChArUco corners; the camera pose stored in
the world config is used only as the initial branch for planar PnP and for the
reported error, not as the result.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.charuco_strip import Board, detect_charuco  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "mocap_arena_charuco_cameras.json"
DEFAULT_BOARD = ROOT / "tools" / "charuco_strip_editor" / "charuco_strip_12x3" / "charuco_strip_12x3_board.json"
DEFAULT_MAP = ROOT / "tools" / "charuco_strip_editor" / "charuco_strip_12x3" / "charuco_strip_12x3_map.txt"


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_error_deg(a, b):
    value = (np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(value, -1.0, 1.0))))


def matrix_to_rpy_deg(r):
    """Intrinsic XYZ roll, pitch, yaw, for readable diagnostics only."""
    pitch = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = math.atan2(-r[1, 2], r[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def load_marker_map(path, board):
    sheet_size = None
    markers = {}
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith("# sheet_size_m:"):
            sheet_size = tuple(float(v) for v in line.split(":", 1)[1].split())
        elif line and not line.startswith("#"):
            fields = line.split()
            if len(fields) != 8:
                raise ValueError(f"bad marker-map row: {line}")
            marker_id = int(fields[0])
            markers[marker_id] = {
                "length": float(fields[1]),
                "center": [float(fields[2]), float(fields[3]), float(fields[4])],
                "rotation_zyx": [float(v) for v in fields[5:8]],
            }
    expected_size = (board.sheet_width_mm / 1000.0, board.sheet_height_mm / 1000.0)
    if sheet_size is None or not np.allclose(sheet_size, expected_size):
        raise ValueError(f"map sheet_size {sheet_size} != board {expected_size}")
    if set(markers) != set(board.ids):
        raise ValueError("marker IDs in the map and ChArUco board differ")
    for k, marker_id in enumerate(board.ids):
        corners = board.marker_corners_mm(k)
        expected = np.array([
            board.border_mm / 1000.0 + corners[:, 0].mean() / 1000.0,
            sheet_size[1] - board.border_mm / 1000.0 - corners[:, 1].mean() / 1000.0,
            0.0,
        ])
        if not np.allclose(markers[marker_id]["center"], expected, atol=1e-6):
            raise ValueError(f"marker {marker_id} has inconsistent coordinates")
    return np.asarray(sheet_size, float), markers


def board_frames(config, sheet_size):
    """Board-centred +X/+Y/+Z frame to arena transform."""
    strip = config["strip"]
    x0, y0 = strip["sheet_top_left_arena"]
    yaw = float(strip.get("yaw_rad", 0.0))
    r_world_board = rotation_z(yaw)
    centre_world = np.array([x0, y0, 0.0]) + r_world_board @ np.array(
        [sheet_size[0] / 2.0, -sheet_size[1] / 2.0, 0.0])
    return centre_world, r_world_board


def charuco_object_points(cv_board, corner_ids, board, sheet_size):
    corners = np.asarray(cv_board.chessboardCorners, float)[corner_ids]
    # OpenCV board: origin at chessboard top-left, Y down the print.
    # Our result frame: origin at sheet centre, X right, Y up, Z up.
    return np.column_stack((
        corners[:, 0] + board.border_mm / 1000.0 - sheet_size[0] / 2.0,
        sheet_size[1] / 2.0 - (corners[:, 1] + board.border_mm / 1000.0),
        corners[:, 2],
    ))


def equidistant_to_pinhole(points, focal, principal):
    points = np.asarray(points, float).reshape(-1, 2)
    principal = np.asarray(principal, float)
    delta = points - principal
    radius = np.linalg.norm(delta, axis=1)
    theta = radius / focal
    scale = np.tan(theta) / np.maximum(radius, 1e-12)
    normalized = delta * scale[:, None]
    return principal + focal * normalized


def equidistant_project(points_camera, focal, principal):
    points = np.asarray(points_camera, float)
    xy = points[:, :2]
    radius_3d = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(radius_3d, points[:, 2])
    scale = focal * theta / np.maximum(radius_3d, 1e-12)
    return np.asarray(principal, float) + xy * scale[:, None]


def scaled_lens(config, image_size):
    native = np.asarray(config["image_size"], float)
    scale = np.asarray(image_size, float) / native
    if not np.isclose(scale[0], scale[1], rtol=0.01):
        raise ValueError(f"image aspect ratio {tuple(image_size)} differs from calibrated {tuple(native.astype(int))}")
    return (float(config["focal_px_per_rad"]) * float(scale.mean()),
            np.asarray(config["principal_point"], float) * scale)


def estimate_camera_pose(image_points, object_points, camera, config,
                         board_center_world, r_world_board, image_size=None):
    focal, principal = scaled_lens(config, image_size or config["image_size"])
    pinhole_points = equidistant_to_pinhole(image_points, focal, principal)
    k = np.array([[focal, 0.0, principal[0]],
                  [0.0, focal, principal[1]],
                  [0.0, 0.0, 1.0]])

    nominal_position_world = np.asarray(camera["position_world"], float)
    nominal_r_world_optical = np.asarray(camera["R_world_optical"], float)
    nominal_position_board = r_world_board.T @ (nominal_position_world - board_center_world)
    nominal_r_board_optical = r_world_board.T @ nominal_r_world_optical
    r_camera_board = nominal_r_board_optical.T
    rvec, _ = cv2.Rodrigues(r_camera_board)
    tvec = (-r_camera_board @ nominal_position_board).reshape(3, 1)

    ok, rvec, tvec = cv2.solvePnP(
        object_points.astype(np.float64), pinhole_points.astype(np.float64),
        k, None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    r_camera_board, _ = cv2.Rodrigues(rvec)
    position_board = -r_camera_board.T @ tvec.reshape(3)
    r_board_optical = r_camera_board.T
    camera_points = object_points @ r_camera_board.T + tvec.reshape(3)
    projected = equidistant_project(camera_points, focal, principal)
    errors = np.linalg.norm(projected - image_points, axis=1)
    return {
        "position_board": position_board,
        "R_board_optical": r_board_optical,
        "rms_px": float(np.sqrt(np.mean(errors ** 2))),
        "p95_px": float(np.percentile(errors, 95)),
        "nominal_position_board": nominal_position_board,
        "nominal_R_board_optical": nominal_r_board_optical,
    }


def acquire_live(cameras, timeout):
    try:
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        from localization_contracts.capture import LatestFrames, rgb_array
    except ImportError as exc:
        raise RuntimeError(
            "Gazebo Python modules are unavailable; run with /usr/bin/python3 or use --images-dir") from exc
    ids = [c["name"] for c in cameras]
    latest = LatestFrames(ids)
    node = Node()
    callbacks = []
    for camera in cameras:
        camera_id = camera["name"]
        callback = lambda msg, camera_id=camera_id: latest.push(camera_id, msg)
        callbacks.append(callback)
        if not node.subscribe(Image, camera["image_topic"], callback):
            raise RuntimeError(f"cannot subscribe to {camera['image_topic']}")
    deadline = time.monotonic() + timeout
    images = {}
    while len(images) != len(cameras) and time.monotonic() < deadline:
        for camera_id in ids:
            if camera_id in images:
                continue
            item = latest.take(camera_id, min(0.2, max(0.0, deadline - time.monotonic())))
            if item is not None:
                images[camera_id] = rgb_array(item[0]).copy()
    latest.close()
    if len(images) != len(cameras):
        missing = sorted(set(ids) - set(images))
        raise RuntimeError(f"no frames from: {', '.join(missing)}")
    return images


def acquire_files(cameras, directory):
    directory = Path(directory)
    images = {}
    for camera in cameras:
        camera_id = camera["name"]
        matches = []
        for suffix in (".ppm", ".png", ".jpg", ".jpeg"):
            matches.extend(directory.glob(camera_id + suffix))
        if not matches:
            raise RuntimeError(f"{directory}: no image for {camera_id}")
        bgr = cv2.imread(str(matches[0]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot read {matches[0]}")
        images[camera_id] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return images


class GazeboFrames:
    """Persistent Gazebo subscriptions used by the interactive mode."""
    def __init__(self, cameras):
        try:
            from gz.transport13 import Node
            from gz.msgs10.image_pb2 import Image
            from localization_contracts.capture import LatestFrames, rgb_array
        except ImportError as exc:
            raise RuntimeError(
                "Gazebo Python modules are unavailable; use /usr/bin/python3 or --camera-source") from exc
        self.cameras = cameras
        self.ids = [c["name"] for c in cameras]
        self.latest = LatestFrames(self.ids)
        self.rgb_array = rgb_array
        self.node = Node()
        self.callbacks = []
        for camera in cameras:
            camera_id = camera["name"]
            callback = lambda msg, camera_id=camera_id: self.latest.push(camera_id, msg)
            self.callbacks.append(callback)
            if not self.node.subscribe(Image, camera["image_topic"], callback):
                raise RuntimeError(f"cannot subscribe to {camera['image_topic']}")

    def get(self, timeout):
        deadline = time.monotonic() + timeout
        images = {}
        while len(images) != len(self.ids) and time.monotonic() < deadline:
            for camera_id in self.ids:
                if camera_id in images:
                    continue
                item = self.latest.take(camera_id, min(0.1, max(0.0, deadline - time.monotonic())))
                if item is not None:
                    images[camera_id] = self.rgb_array(item[0]).copy()
        if len(images) != len(self.ids):
            missing = sorted(set(self.ids) - set(images))
            raise RuntimeError(f"no fresh frames from: {', '.join(missing)}")
        return images

    def close(self):
        self.latest.close()
        for camera in self.cameras:
            self.node.unsubscribe(camera["image_topic"])


class ImageDirectoryFrames:
    """Re-read camera_N images; useful when another process updates snapshots."""
    def __init__(self, cameras, directory):
        self.cameras, self.directory = cameras, directory

    def get(self, timeout):
        del timeout
        return acquire_files(self.cameras, self.directory)

    def close(self):
        pass


def parse_camera_sources(values, cameras):
    result = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"camera source must be CAMERA=SOURCE, got {value!r}")
        camera_id, source = value.split("=", 1)
        if camera_id in result:
            raise ValueError(f"duplicate source for {camera_id}")
        result[camera_id] = int(source) if source.isdecimal() else source
    expected = {c["name"] for c in cameras}
    if result and set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise ValueError(f"camera sources must cover every camera; missing={missing}, extra={extra}")
    return result


class OpenCVCameraFrames:
    """Concurrent /dev/video, RTSP, HTTP or MJPEG acquisition via OpenCV."""
    def __init__(self, sources):
        self.sources = sources
        self.condition = threading.Condition()
        self.frames = {}
        self.sequence = dict.fromkeys(sources, 0)
        self.errors = {}
        self.stopping = False
        self.captures = {}
        self.threads = []
        for camera_id, source in sources.items():
            thread = threading.Thread(target=self._read, args=(camera_id, source), daemon=True)
            self.threads.append(thread)
            thread.start()

    def _read(self, camera_id, source):
        capture = cv2.VideoCapture(source)
        self.captures[camera_id] = capture
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            with self.condition:
                self.errors[camera_id] = f"cannot open {source}"
                self.condition.notify_all()
            return
        while not self.stopping:
            ok, bgr = capture.read()
            if not ok:
                time.sleep(0.1)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self.condition:
                self.frames[camera_id] = rgb
                self.sequence[camera_id] += 1
                self.errors.pop(camera_id, None)
                self.condition.notify_all()
        capture.release()

    def get(self, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            baseline = dict(self.sequence)
            self.condition.wait_for(
                lambda: (all(camera_id in self.frames and self.sequence[camera_id] > baseline[camera_id]
                             for camera_id in self.sources) or self.errors or self.stopping), timeout)
            if self.errors:
                raise RuntimeError("; ".join(f"{k}: {v}" for k, v in sorted(self.errors.items())))
            if not all(camera_id in self.frames for camera_id in self.sources):
                missing = sorted(set(self.sources) - set(self.frames))
                raise RuntimeError(f"no frames from: {', '.join(missing)}")
            return {camera_id: self.frames[camera_id].copy() for camera_id in self.sources}

    def close(self):
        self.stopping = True
        for capture in list(self.captures.values()):
            capture.release()
        with self.condition:
            self.condition.notify_all()
        for thread in self.threads:
            thread.join(timeout=1.0)


def make_frame_source(cameras, images_dir=None, camera_sources=None):
    if camera_sources:
        return OpenCVCameraFrames(camera_sources)
    if images_dir:
        return ImageDirectoryFrames(cameras, images_dir)
    return GazeboFrames(cameras)


def rounded(values, digits=6):
    return np.round(np.asarray(values, float), digits).tolist()


def analyze_camera(camera, rgb, config, board, cv_board, sheet_size,
                   board_center_world, field_center_world, r_world_board,
                   min_markers=4, min_corners=6, max_rms_px=3.0,
                   max_position_error_m=0.15, max_angle_error_deg=3.0):
    camera_id = camera["name"]
    reasons = []
    try:
        scaled_lens(config, (rgb.shape[1], rgb.shape[0]))
    except ValueError as exc:
        reasons.append(str(exc))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, corner_ids, marker_corners, marker_ids = detect_charuco(board, gray)
    marker_count = 0 if marker_ids is None else len(marker_ids)
    corner_count = 0 if corner_ids is None else len(corner_ids)
    if marker_count < min_markers:
        reasons.append(f"only {marker_count} markers")
    if corner_count < min_corners:
        reasons.append(f"only {corner_count} ChArUco corners")

    item = {"detected_markers": marker_count, "detected_charuco_corners": corner_count}
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if marker_count:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    if corner_count:
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, corner_ids, (0, 0, 255))

    if corner_count >= 4 and not any("aspect ratio" in reason for reason in reasons):
        ids = corner_ids.ravel().astype(int)
        image_points = corners.reshape(-1, 2).astype(float)
        object_points = charuco_object_points(cv_board, ids, board, sheet_size)
        try:
            pose = estimate_camera_pose(
                image_points, object_points, camera, config, board_center_world,
                r_world_board, (rgb.shape[1], rgb.shape[0]))
            position_board = pose["position_board"]
            position_world = board_center_world + r_world_board @ position_board
            position_field = position_world - field_center_world
            r_world_optical = r_world_board @ pose["R_board_optical"]
            position_error = float(np.linalg.norm(position_world - np.asarray(camera["position_world"])))
            angle_error = rotation_error_deg(r_world_optical, np.asarray(camera["R_world_optical"]))
            vector_to_board = -position_board
            distance_to_board = float(np.linalg.norm(vector_to_board))
            pointing_error = math.degrees(math.acos(float(np.clip(
                np.dot(pose["R_board_optical"][:, 2], vector_to_board) /
                max(distance_to_board, 1e-12), -1.0, 1.0))))
            item["pose"] = {
                "camera_from_charuco_center_m": rounded(position_board),
                "camera_from_field_center_m": rounded(position_field),
                "camera_world_m": rounded(position_world),
                "distance_to_charuco_center_m": round(distance_to_board, 6),
                "optical_axis_in_charuco_frame": rounded(pose["R_board_optical"][:, 2]),
                "R_charuco_optical": rounded(pose["R_board_optical"]),
                "R_world_optical": rounded(r_world_optical),
                "optical_rpy_world_deg": rounded(matrix_to_rpy_deg(r_world_optical), 3),
                "pointing_error_to_charuco_center_deg": round(pointing_error, 3),
                "reprojection_rms_px": round(pose["rms_px"], 4),
                "reprojection_p95_px": round(pose["p95_px"], 4),
                "nominal_position_error_m": round(position_error, 6),
                "nominal_angle_error_deg": round(angle_error, 4),
            }
            if pose["rms_px"] > max_rms_px:
                reasons.append(f"reprojection RMS {pose['rms_px']:.2f}px")
            if position_error > max_position_error_m:
                reasons.append(f"position error {position_error:.3f}m")
            if angle_error > max_angle_error_deg:
                reasons.append(f"angle error {angle_error:.2f}deg")
        except (cv2.error, RuntimeError, ValueError) as exc:
            reasons.append(f"pose failed: {exc}")

    item["status"] = "PASS" if not reasons else "FAIL"
    item["reasons"] = reasons
    label = (f"{item['status']}  markers={marker_count} corners={corner_count}" +
             (f" rms={item['pose']['reprojection_rms_px']:.2f}px" if "pose" in item else ""))
    cv2.putText(vis, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 200, 0) if not reasons else (0, 0, 255), 2, cv2.LINE_AA)
    return item, vis


def analyze_images(images, cameras, config, board, sheet_size, marker_map,
                   board_center_world, field_center_world, r_world_board,
                   thresholds, map_path):
    report = {
        "status": "PASS",
        "timestamp_ns": time.time_ns(),
        "coordinate_frames": {
            "charuco_center_world_m": rounded(board_center_world),
            "field_center_world_m": rounded(field_center_world),
            "charuco_axes": "+X along the strip, +Y toward the marker-0/top side, +Z up",
        },
        "map": {"path": str(Path(map_path)), "markers": len(marker_map), "sheet_size_m": rounded(sheet_size)},
        "board": board.to_dict(),
        "cameras": {},
    }
    annotated = {}
    cv_board = board.opencv_board()
    for camera in cameras:
        item, vis = analyze_camera(
            camera, images[camera["name"]], config, board, cv_board, sheet_size,
            board_center_world, field_center_world, r_world_board, **thresholds)
        report["cameras"][camera["name"]] = item
        annotated[camera["name"]] = vis
        if item["status"] != "PASS":
            report["status"] = "FAIL"
    return report, annotated


def save_scene_3d(report, output_dir, field_size, sheet_size, board_center_world,
                  r_world_board, marker_map, elev=28.0, azim=-58.0):
    """Save a metric 3D overview of the field, ChArUco sheet and cameras."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output_dir = Path(output_dir)
    field_center = np.asarray(report["coordinate_frames"]["field_center_world_m"], float)
    field_w, field_h = map(float, field_size)
    field = np.array([
        field_center + [-field_w / 2, -field_h / 2, 0],
        field_center + [field_w / 2, -field_h / 2, 0],
        field_center + [field_w / 2, field_h / 2, 0],
        field_center + [-field_w / 2, field_h / 2, 0],
    ])

    # Sheet coordinates use +Y toward its top/marker-0 side.
    sw, sh = map(float, sheet_size)
    sheet_local = np.array([
        [-sw / 2, -sh / 2, 0], [sw / 2, -sh / 2, 0],
        [sw / 2, sh / 2, 0], [-sw / 2, sh / 2, 0],
    ])
    sheet = board_center_world + sheet_local @ r_world_board.T

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection([field], facecolors="#dce8d5", edgecolors="#52634b",
                                         alpha=0.28, linewidths=1.5))
    ax.add_collection3d(Poly3DCollection([sheet], facecolors="#f3f3f3", edgecolors="#101010",
                                         alpha=0.92, linewidths=2.5))

    # ChArUco chess grid, including the white sheet border.
    board_spec = report["board"]
    border = float(board_spec["border_mm"]) / 1000.0
    square = float(board_spec["square_mm"]) / 1000.0
    x_start, y_top = -sw / 2 + border, sh / 2 - border
    for i in range(int(board_spec["squares_x"]) + 1):
        x = x_start + i * square
        local = np.array([[x, y_top, 0.006], [x, y_top - int(board_spec["squares_y"]) * square, 0.006]])
        world = board_center_world + local @ r_world_board.T
        ax.plot(world[:, 0], world[:, 1], world[:, 2], color="#777777", linewidth=0.45, alpha=0.8)
    for j in range(int(board_spec["squares_y"]) + 1):
        y = y_top - j * square
        local = np.array([[x_start, y, 0.006], [x_start + int(board_spec["squares_x"]) * square, y, 0.006]])
        world = board_center_world + local @ r_world_board.T
        ax.plot(world[:, 0], world[:, 1], world[:, 2], color="#777777", linewidth=0.45, alpha=0.8)

    marker_local = np.array([
        [m["center"][0] - sw / 2, m["center"][1] - sh / 2, 0.012]
        for m in marker_map.values()
    ])
    marker_world = board_center_world + marker_local @ r_world_board.T
    ax.scatter(marker_world[:, 0], marker_world[:, 1], marker_world[:, 2],
               color="black", marker="s", s=5, alpha=0.75, label="ArUco markers")

    colours = plt.cm.tab10(np.linspace(0, 1, max(6, len(report["cameras"]))))
    plotted_cameras = []
    for colour, (camera_id, item) in zip(colours, report["cameras"].items()):
        pose = item.get("pose")
        if pose is None:
            continue
        centre = np.asarray(pose["camera_world_m"], float)
        rotation = np.asarray(pose["R_world_optical"], float)
        plotted_cameras.append(centre)
        ax.scatter(*centre, color=colour, s=55, depthshade=False)
        ax.text(*(centre + [0, 0, 0.14]), camera_id, color=colour, fontsize=9,
                ha="center", va="bottom", weight="bold")

        # A small four-sided camera frustum. Optical +Z points into the scene.
        depth, half_w, half_h = 0.75, 0.34, 0.25
        rays_optical = np.array([
            [-half_w, -half_h, depth], [half_w, -half_h, depth],
            [half_w, half_h, depth], [-half_w, half_h, depth],
        ])
        corners = centre + rays_optical @ rotation.T
        for corner in corners:
            ax.plot([centre[0], corner[0]], [centre[1], corner[1]], [centre[2], corner[2]],
                    color=colour, linewidth=1.4)
        closed = np.vstack([corners, corners[0]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=colour, linewidth=1.4)
        optical_end = centre + rotation[:, 2] * 1.15
        ax.plot([centre[0], optical_end[0]], [centre[1], optical_end[1]],
                [centre[2], optical_end[2]], color=colour, linewidth=2.2)
        ax.plot([centre[0], board_center_world[0]], [centre[1], board_center_world[1]],
                [centre[2], board_center_world[2]], color=colour, linestyle=":", linewidth=0.8, alpha=0.7)

    ax.scatter(*field_center, color="#1769aa", marker="x", s=90, linewidths=2.5, label="Field centre")
    ax.scatter(*board_center_world, color="#d32f2f", marker="+", s=110, linewidths=2.5,
               label="ChArUco centre")
    all_points = np.vstack([field, sheet] + ([np.asarray(plotted_cameras)] if plotted_cameras else []))
    mins, maxs = all_points.min(axis=0), all_points.max(axis=0)
    centre = (mins + maxs) / 2
    radius = max((maxs - mins).max() / 2, 1.0) * 1.08
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(0, max(3.5, maxs[2] + 0.5))
    try:
        ax.set_box_aspect((2 * radius, 2 * radius, max(3.5, maxs[2] + 0.5)))
    except AttributeError:
        pass
    ax.set_xlabel("Arena X, m")
    ax.set_ylabel("Arena Y, m")
    ax.set_zlabel("Z, m")
    ax.set_title("Estimated camera poses and ChArUco field")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    png = output_dir / "scene_3d.png"
    svg = output_dir / "scene_3d.svg"
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def draw_live_scene(ax, report, field_size, sheet_size, board_center_world,
                    r_world_board, elev, azim):
    """Redraw the interactive axes while preserving the user's view angle."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.clear()
    field_center = np.asarray(report["coordinate_frames"]["field_center_world_m"], float)
    fw, fh = map(float, field_size)
    field = np.array([
        field_center + [-fw / 2, -fh / 2, 0], field_center + [fw / 2, -fh / 2, 0],
        field_center + [fw / 2, fh / 2, 0], field_center + [-fw / 2, fh / 2, 0],
    ])
    sw, sh = map(float, sheet_size)
    sheet_local = np.array([
        [-sw / 2, -sh / 2, 0], [sw / 2, -sh / 2, 0],
        [sw / 2, sh / 2, 0], [-sw / 2, sh / 2, 0],
    ])
    sheet = board_center_world + sheet_local @ r_world_board.T
    ax.add_collection3d(Poly3DCollection([field], facecolors="#dce8d5", edgecolors="#52634b",
                                         alpha=0.25, linewidths=1.3))
    ax.add_collection3d(Poly3DCollection([sheet], facecolors="#eeeeee", edgecolors="#111111",
                                         alpha=0.9, linewidths=2.2))

    board_spec = report["board"]
    border = float(board_spec["border_mm"]) / 1000.0
    square = float(board_spec["square_mm"]) / 1000.0
    x_start, y_top = -sw / 2 + border, sh / 2 - border
    for i in range(int(board_spec["squares_x"]) + 1):
        x = x_start + i * square
        local = np.array([[x, y_top, 0.006], [x, y_top - int(board_spec["squares_y"]) * square, 0.006]])
        world = board_center_world + local @ r_world_board.T
        ax.plot(world[:, 0], world[:, 1], world[:, 2], color="#777", linewidth=0.4)
    for j in range(int(board_spec["squares_y"]) + 1):
        y = y_top - j * square
        local = np.array([[x_start, y, 0.006], [x_start + int(board_spec["squares_x"]) * square, y, 0.006]])
        world = board_center_world + local @ r_world_board.T
        ax.plot(world[:, 0], world[:, 1], world[:, 2], color="#777", linewidth=0.4)

    colours = __import__("matplotlib.pyplot", fromlist=["cm"]).cm.tab10(
        np.linspace(0, 1, max(6, len(report["cameras"]))))
    status_lines = []
    for colour, (camera_id, item) in zip(colours, report["cameras"].items()):
        pose = item.get("pose")
        if pose is None:
            status_lines.append(f"{camera_id}: NO POSE ({item['detected_charuco_corners']} corners)")
            continue
        centre = np.asarray(pose["camera_world_m"], float)
        rotation = np.asarray(pose["R_world_optical"], float)
        display_colour = colour if item["status"] == "PASS" else "#d32f2f"
        ax.scatter(*centre, color=display_colour, s=65, depthshade=False)
        relative = pose["camera_from_charuco_center_m"]
        ax.text(*(centre + [0, 0, 0.13]),
                f"{camera_id}\n({relative[0]:+.2f}, {relative[1]:+.2f}, {relative[2]:+.2f}) m",
                color=display_colour, fontsize=8, ha="center", va="bottom", weight="bold")
        depth, half_w, half_h = 0.7, 0.31, 0.23
        rays = np.array([[-half_w, -half_h, depth], [half_w, -half_h, depth],
                         [half_w, half_h, depth], [-half_w, half_h, depth]])
        corners = centre + rays @ rotation.T
        for corner in corners:
            ax.plot([centre[0], corner[0]], [centre[1], corner[1]], [centre[2], corner[2]],
                    color=display_colour, linewidth=1.3)
        closed = np.vstack([corners, corners[0]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=display_colour, linewidth=1.3)
        optical_end = centre + rotation[:, 2]
        ax.plot([centre[0], optical_end[0]], [centre[1], optical_end[1]],
                [centre[2], optical_end[2]], color=display_colour, linewidth=2.0)
        status_lines.append(
            f"{camera_id}: {item['status']}, M={item['detected_markers']}, "
            f"C={item['detected_charuco_corners']}, RMS={pose['reprojection_rms_px']:.2f}px")

    ax.scatter(*field_center, color="#1769aa", marker="x", s=85, linewidths=2.3)
    ax.scatter(*board_center_world, color="#d32f2f", marker="+", s=105, linewidths=2.3)
    radius = max(fw, fh, sw) * 0.54
    ax.set_xlim(field_center[0] - radius, field_center[0] + radius)
    ax.set_ylim(field_center[1] - radius, field_center[1] + radius)
    ax.set_zlim(0, 3.7)
    try:
        ax.set_box_aspect((2 * radius, 2 * radius, 3.7))
    except AttributeError:
        pass
    ax.set_xlabel("Arena X, m")
    ax.set_ylabel("Arena Y, m")
    ax.set_zlabel("Z, m")
    ax.set_title(f"Camera poses from one ChArUco snapshot — {report['status']}\nDrag to rotate, scroll to zoom")
    ax.text2D(0.01, 0.99, "\n".join(status_lines), transform=ax.transAxes, va="top",
              family="monospace", fontsize=8,
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.78))
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.3)


def matplotlib_backend_is_interactive(matplotlib_module):
    backend = str(matplotlib_module.get_backend()).lower()
    interactive = {str(name).lower() for name in matplotlib_module.rcsetup.interactive_bk}
    return backend in interactive or backend.startswith("module://") and "inline" not in backend


def run_live_3d(source, cameras, config, board, sheet_size, marker_map,
                board_center_world, field_center_world, r_world_board,
                thresholds, args, output_dir):
    import matplotlib
    import matplotlib.pyplot as plt

    if not matplotlib_backend_is_interactive(matplotlib):
        source.close()
        raise RuntimeError(
            f"--live-3d needs a graphical Matplotlib backend, got {matplotlib.get_backend()!r}; "
            "run from a desktop session or set MPLBACKEND=TkAgg")
    json_path = Path(args.json_out) if args.json_out else output_dir / "report.json"
    try:
        images = source.get(args.timeout)
    finally:
        source.close()

    report, annotated = analyze_images(
        images, cameras, config, board, sheet_size, marker_map,
        board_center_world, field_center_world, r_world_board,
        thresholds, args.map_path)
    for camera_id, vis in annotated.items():
        cv2.imwrite(str(output_dir / f"{camera_id}_detected.png"), vis)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    draw_live_scene(ax, report, args.field_size, sheet_size, board_center_world,
                    r_world_board, args.view_elev, args.view_azim)
    print("Poses calculated once; camera streams are closed. The 3D view will not be recalculated.")
    print(f"Report: {json_path}")
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        plt.close(fig)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--board", default=str(DEFAULT_BOARD))
    parser.add_argument("--map", dest="map_path", default=str(DEFAULT_MAP))
    parser.add_argument("--images-dir", help="offline camera_1.ppm ... camera_6.ppm directory")
    parser.add_argument(
        "--camera-source", action="append", default=[], metavar="CAMERA=SOURCE",
        help="real source for every camera: /dev/video, numeric index, RTSP or HTTP/MJPEG URL")
    parser.add_argument("--live-3d", action="store_true",
                        help="estimate once, then open a fixed interactive rotatable 3D window")
    parser.add_argument("--refresh-hz", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default="/tmp/mocap-charuco-check")
    parser.add_argument("--json-out", help="default: OUTPUT_DIR/report.json")
    parser.add_argument("--field-center", type=float, nargs=3, default=[6.0, 6.0, 0.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--field-size", type=float, nargs=2, default=[12.0, 12.0], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--view-elev", type=float, default=28.0, help="3D view elevation in degrees")
    parser.add_argument("--view-azim", type=float, default=-58.0, help="3D view azimuth in degrees")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-markers", type=int, default=4)
    parser.add_argument("--min-corners", type=int, default=6)
    parser.add_argument("--max-rms-px", type=float, default=3.0)
    parser.add_argument("--max-position-error-m", type=float, default=0.15)
    parser.add_argument("--max-angle-error-deg", type=float, default=3.0)
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    board_spec = json.loads(Path(args.board).read_text(encoding="utf-8"))
    board = Board.from_dict(board_spec["board"])
    sheet_size, marker_map = load_marker_map(args.map_path, board)
    cameras = config["cameras"]
    board_center_world, r_world_board = board_frames(config, sheet_size)
    field_center_world = np.asarray(args.field_center, float)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_sources = parse_camera_sources(args.camera_source, cameras)
    if args.images_dir and camera_sources:
        parser.error("use either --images-dir or --camera-source")
    thresholds = dict(
        min_markers=args.min_markers, min_corners=args.min_corners,
        max_rms_px=args.max_rms_px, max_position_error_m=args.max_position_error_m,
        max_angle_error_deg=args.max_angle_error_deg)
    if args.live_3d:
        source = make_frame_source(cameras, args.images_dir, camera_sources)
        return run_live_3d(
            source, cameras, config, board, sheet_size, marker_map,
            board_center_world, field_center_world, r_world_board,
            thresholds, args, output_dir)

    if camera_sources:
        source = OpenCVCameraFrames(camera_sources)
        try:
            images = source.get(args.timeout)
        finally:
            source.close()
    else:
        images = acquire_files(cameras, args.images_dir) if args.images_dir else acquire_live(cameras, args.timeout)
    report, annotated = analyze_images(
        images, cameras, config, board, sheet_size, marker_map,
        board_center_world, field_center_world, r_world_board,
        thresholds, args.map_path)
    for camera_id, vis in annotated.items():
        cv2.imwrite(str(output_dir / f"{camera_id}_detected.png"), vis)

    json_path = Path(args.json_out) if args.json_out else output_dir / "report.json"
    scene_png, scene_svg = save_scene_3d(
        report, output_dir, args.field_size, sheet_size, board_center_world,
        r_world_board, marker_map, args.view_elev, args.view_azim)
    report["visualization"] = {"png": str(scene_png), "svg": str(scene_svg)}
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ChArUco centre in field: {rounded(board_center_world)} m; field centre: {rounded(field_center_world)} m")
    print(f"{'camera':>9} {'result':>6} {'markers':>8} {'corners':>8} {'camera from ChArUco centre, m':>34} {'rms px':>8} {'pos err':>8}")
    for camera_id, item in report["cameras"].items():
        pose = item.get("pose", {})
        print(f"{camera_id:>9} {item['status']:>6} {item['detected_markers']:8d} "
              f"{item['detected_charuco_corners']:8d} "
              f"{str(pose.get('camera_from_charuco_center_m', '-')):>34} "
              f"{pose.get('reprojection_rms_px', float('nan')):8.2f} "
              f"{pose.get('nominal_position_error_m', float('nan')):8.3f}")
        for reason in item["reasons"]:
            print(f"          - {reason}")
    print(f"3D scene: {scene_png} and {scene_svg}")
    print(f"{report['status']}: report {json_path}; annotated images {output_dir}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
