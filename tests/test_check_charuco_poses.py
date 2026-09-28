import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_charuco_poses", ROOT / "scripts" / "check_charuco_poses.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def test_map_and_fisheye_pose_recover_every_camera():
    config = json.loads((ROOT / "config" / "mocap_arena_charuco_cameras.json").read_text())
    board_spec = json.loads((ROOT / "tools" / "charuco_strip_editor" / "charuco_strip_12x3" /
                             "charuco_strip_12x3_board.json").read_text())
    board = check.Board.from_dict(board_spec["board"])
    sheet_size, markers = check.load_marker_map(
        ROOT / "tools" / "charuco_strip_editor" / "charuco_strip_12x3" /
        "charuco_strip_12x3_map.txt", board)
    assert len(markers) == 90
    centre_world, r_world_board = check.board_frames(config, sheet_size)
    cv_board = board.opencv_board()
    ids = np.arange(len(cv_board.chessboardCorners), dtype=int)
    object_points = check.charuco_object_points(cv_board, ids, board, sheet_size)

    for camera in config["cameras"]:
        position_board = r_world_board.T @ (np.asarray(camera["position_world"]) - centre_world)
        r_board_optical = r_world_board.T @ np.asarray(camera["R_world_optical"])
        r_camera_board = r_board_optical.T
        camera_points = object_points @ r_camera_board.T - r_camera_board @ position_board
        image_points = check.equidistant_project(
            camera_points, config["focal_px_per_rad"], config["principal_point"])
        visible = ((image_points[:, 0] >= 0) & (image_points[:, 0] < config["image_size"][0]) &
                   (image_points[:, 1] >= 0) & (image_points[:, 1] < config["image_size"][1]) &
                   (camera_points[:, 2] > 0))
        pose = check.estimate_camera_pose(
            image_points[visible], object_points[visible], camera, config,
            centre_world, r_world_board)
        assert np.allclose(pose["position_board"], position_board, atol=1e-6)
        assert check.rotation_error_deg(pose["R_board_optical"], r_board_optical) < 1e-5
        assert pose["rms_px"] < 1e-6


def test_scene_3d_is_written(tmp_path):
    board_spec = json.loads((ROOT / "tools" / "charuco_strip_editor" / "charuco_strip_12x3" /
                             "charuco_strip_12x3_board.json").read_text())
    board = check.Board.from_dict(board_spec["board"])
    sheet_size, markers = check.load_marker_map(
        ROOT / "tools" / "charuco_strip_editor" / "charuco_strip_12x3" /
        "charuco_strip_12x3_map.txt", board)
    report = {
        "status": "PASS",
        "coordinate_frames": {"field_center_world_m": [6, 6, 0]},
        "board": board.to_dict(),
        "cameras": {
            "camera_1": {
                "status": "PASS", "detected_markers": 20, "detected_charuco_corners": 18,
                "pose": {
                    "camera_world_m": [9, 9, 2.88],
                    "camera_from_charuco_center_m": [3, 3, 2.88],
                    "R_world_optical": [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
                    "reprojection_rms_px": 0.4,
                },
            },
        },
    }
    png, svg = check.save_scene_3d(
        report, tmp_path, [12, 12], sheet_size, np.array([6, 6, 0]),
        np.eye(3), markers)
    assert png.read_bytes().startswith(b"\x89PNG")
    assert "<svg" in svg.read_text(encoding="utf-8")

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    check.draw_live_scene(ax, report, [12, 12], sheet_size, np.array([6, 6, 0]),
                          np.eye(3), 28, -58)
    fig.canvas.draw()
    plt.close(fig)


def test_scaled_fisheye_lens_accepts_half_size_preview():
    config = json.loads((ROOT / "config" / "mocap_arena_charuco_cameras.json").read_text())
    focal, principal = check.scaled_lens(config, [820, 616])
    assert np.isclose(focal, config["focal_px_per_rad"] / 2)
    assert np.allclose(principal, np.asarray(config["principal_point"]) / 2)


def test_real_camera_sources_require_all_camera_ids():
    cameras = [{"name": f"camera_{i}"} for i in range(1, 7)]
    values = [f"camera_{i}=http://cam{i}:8080/video_feed" for i in range(1, 7)]
    parsed = check.parse_camera_sources(values, cameras)
    assert parsed["camera_1"].endswith("/video_feed")
    try:
        check.parse_camera_sources(values[:-1], cameras)
    except ValueError as exc:
        assert "camera_6" in str(exc)
    else:
        raise AssertionError("incomplete source mapping was accepted")


def test_tkagg_is_not_mistaken_for_headless_agg():
    class MatplotlibStub:
        class rcsetup:
            interactive_bk = ["GTK3Agg", "QtAgg", "TkAgg"]

        @staticmethod
        def get_backend():
            return "TkAgg"

    assert check.matplotlib_backend_is_interactive(MatplotlibStub)
    MatplotlibStub.get_backend = staticmethod(lambda: "agg")
    assert not check.matplotlib_backend_is_interactive(MatplotlibStub)
