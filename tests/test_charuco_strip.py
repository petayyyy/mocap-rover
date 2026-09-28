"""The ChArUco strip must be the board OpenCV thinks it is, wherever it goes:
the print, the editor project, the Gazebo world and the synthetic check."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
cv2 = pytest.importorskip("cv2")
pytest.importorskip("cv2.aruco")

spec = importlib.util.spec_from_file_location("charuco_strip", ROOT / "tools" / "charuco_strip.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)

PPM = 0.4  # 2.5 mm per pixel: every edge of the default sheet lands on a pixel
BORDER = 375.0


@pytest.fixture(scope="module")
def board():
    return cs.Board(30, 6, 375, 280, "DICT_5X5_250")


def test_sheet_is_12_by_3_metres_with_one_square_of_white_border(board):
    assert board.border_mm == 375
    assert (board.sheet_width_mm, board.sheet_height_mm) == (12000, 3000)
    assert (board.width_mm, board.height_mm) == (11250, 2250)
    raster = cs.render_raster(board, PPM)
    assert raster.shape == (1200, 4800)
    b = int(BORDER * PPM)
    assert raster[:b, :].min() == 255 and raster[-b:, :].min() == 255
    assert raster[:, :b].min() == 255 and raster[:, -b:].min() == 255
    assert raster[b, b] == 0  # first chessboard square is black


def test_raster_is_opencv_board_pixel_for_pixel(board):
    ours = cs.render_raster(board, PPM)
    ref = cs.opencv_board_image(board.opencv_board(), (ours.shape[1], ours.shape[0]),
                                margin_size=int(BORDER * PPM), border_bits=1)
    assert ours.shape == ref.shape
    different = np.mean((ours > 127) != (ref > 127))
    # OpenCV 4.6's legacy draw() rounds some module edges differently;
    # generateImage() in newer OpenCV is pixel-identical.
    assert different < (0.005 if not hasattr(board.opencv_board(), "generateImage") else 1e-12)


def test_detector_finds_every_marker_and_corner_in_our_raster(board):
    img = cs.render_raster(board, PPM)
    corners, corner_ids, _, marker_ids = cs.detect_charuco(board, img)
    assert len(marker_ids) == 90
    assert len(corner_ids) == 29 * 5
    truth = {i: ((x + BORDER) * PPM, (y + BORDER) * PPM) for i, x, y in board.chess_corners_mm()}
    err = [np.hypot(*(np.array(truth[int(i)]) - c.ravel())) for i, c in zip(corner_ids.ravel(), corners)]
    assert max(err) < 1.0


def test_editor_project_puts_marker_0_in_the_top_left_white_square(board):
    proj = cs.editor_project(board, "t")
    assert proj["project"]["sheet_w_mm"] == 12000 and proj["project"]["sheet_h_mm"] == 3000
    markers = [o for o in proj["objects"] if o["type"] == "marker"]
    squares = [o for o in proj["objects"] if o["type"] == "shape"]
    assert len(markers) == 90 and len(squares) == 90
    m0 = markers[0]
    assert m0["markerId"] == 0 and m0["dictionary"] == "DICT_5X5_250"
    # second square of the top row, after the border; editor y grows upward
    assert (m0["x_mm"], m0["y_mm"]) == (375 + 562.5, 3000 - 375 - 187.5)
    assert squares[0]["fill"] == "#000000"
    assert (squares[0]["x_mm"], squares[0]["y_mm"]) == (375 + 187.5, 3000 - 375 - 187.5)


def test_drone_map_lists_every_marker_in_sheet_metres(board):
    lines = [l for l in cs.drone_map_txt(board, "t").splitlines() if l and not l.startswith("#")]
    assert len(lines) == 90
    assert lines[0].split()[:4] == ["0", "0.28", "0.9375", "2.4375"]


def test_world_cameras_agree_with_cameras_json():
    cams = cs.cameras_from_world(ROOT / "worlds" / "mocap_arena.sdf")
    ref = {c["name"]: c for c in json.loads((ROOT / "config" / "cameras.json").read_text())["cameras"]}
    assert len(cams) == 6
    for cam in cams:
        r = ref[cam["name"]]
        # nominal config vs installed pose: within the generator's error bounds
        assert np.allclose(cam["position_world"], r["position_world"], atol=0.06)
        assert np.allclose(cam["R_world_optical"], r["R_world_optical"], atol=0.12)


def test_synthetic_fisheye_view_is_detected_without_false_ids(board):
    cams = cs.cameras_from_world(ROOT / "worlds" / "mocap_arena_l2.sdf")
    cam = next(c for c in cams if c["name"] == "camera_3")
    lens = cs.Lens("fisheye", 1280, 960, dfov_deg=160)
    texture = cs.render_raster(board, PPM)
    x0, y0, yaw = cs.strip_pose(board, "x", 4.0, 0.0)
    place = cs.Placement(x0, y0, 0.0, board.border_mm)
    img = cs.render_view(cam, lens, texture, PPM, place, supersample=2)
    res, _ = cs.evaluate_view(board, cam, lens, img, place)
    assert res["markers"] >= 30 and res["markers_wrong"] == 0
    assert res["corners"] >= 50 and res["corners_wrong"] == 0
    assert res["corner_rms_px"] < 1.5


def test_placement_maps_board_corner_through_the_border():
    place = cs.Placement(1.0, 7.0, 0.0, BORDER)
    assert np.allclose(place.board_to_arena([[0, 0]])[0], [1.375, 6.625, 0])
    rotated = cs.Placement(1.0, 7.0, 90.0, 0.0)
    p = rotated.board_to_arena([[12000, 3000]])[0]
    bx, by = rotated.arena_to_sheet_mm(np.array([p[0]]), np.array([p[1]]))
    assert np.allclose([bx[0], by[0]], [12000, 3000])


def test_wide_angle_world_rewrites_every_camera(board):
    text = (ROOT / "worlds" / "mocap_arena_l2.sdf").read_text(encoding="utf-8")
    new, lens, names = cs.wide_angle_world(text, 1640, 1232, 160.0, 2048)
    assert len(names) == 6
    assert new.count('type="wideanglecamera"') == 6 and 'type="camera"' not in new
    assert new.count("<type>equidistant</type>") == 6
    assert new.count("<width>1640</width>") == 6 and new.count("<height>1232</height>") == 6
    assert "<fx>" not in new  # the old pinhole intrinsics are gone
    # the image diagonal spans 160 degrees: f = half diagonal / 80 deg
    assert abs(lens.f - np.hypot(820, 616) / np.radians(80)) < 1e-6
    hfov = float(new.split("<horizontal_fov>")[1].split("<")[0])
    assert abs(hfov - 1640 / lens.f) < 1e-5
    # The optical centres sit ahead of the camera mesh.  With a 160-degree
    # fisheye this prevents the lens barrel from appearing in its own image.
    assert new.count("<pose>0.02 0 0 0 0 0</pose>") == 6
    placed, _ = cs.place_in_world(new, board, "x", 6.0, 0.0)
    assert "<uri>model://charuco_strip</uri>" in placed


def test_ground_truth_csv_lists_all_corners(board):
    place = cs.Placement(0.0, 7.5, 0.0, board.border_mm)
    lines = cs.ground_truth_csv(board, place).splitlines()
    assert len(lines) == 1 + 145 + 90 * 4
    first = lines[1].split(",")
    assert first[:2] == ["chess_corner", "0"]
    assert (float(first[5]), float(first[6])) == (0.75, 6.75)


def test_place_and_remove_in_world(board):
    text = (ROOT / "worlds" / "mocap_arena_l2.sdf").read_text(encoding="utf-8")
    placed, (x0, y0, yaw) = cs.place_in_world(text, board, "x", 6.0, 0.0)
    assert (x0, y0, yaw) == (0.0, 7.5, 0.0)
    assert placed.count("<name>charuco_strip</name>") == 1
    assert "<pose>0 7.5 0.001 0 0 0</pose>" in placed
    again, _ = cs.place_in_world(placed, board, "y", 4.0, 0.0)
    assert again.count("<name>charuco_strip</name>") == 1
    assert "<pose>2.5 0 0.001 0 0 1.5708</pose>" in again
    assert cs.INCLUDE_RE.sub("", again) == text
