import numpy as np

from localization_contracts.hybrid_rover import (
    CameraMotionFollower,
    HybridRoverTracker,
    LidarRoverDetector,
    RoverMeasurement,
    StaticVoxelBackground,
    lidar_gate_radius,
    project_world_to_image,
    scan_grid_to_points,
)


def measurement(t, source, x, y, variance=.01, confidence=1.0):
    return RoverMeasurement(int(t * 1e9), source, x, y,
                            (variance, variance), confidence, "rover-7")


def test_marker_initializes_and_secondary_sources_continue_track():
    tracker = HybridRoverTracker()
    assert not tracker.update(measurement(0, "lidar", 0, 0))
    assert tracker.update(measurement(0, "marker", 0, 0))
    assert tracker.update(measurement(.09, "camera_motion", 1, 0, .04))
    assert tracker.update(measurement(.18, "lidar", 2, 0, .01))
    state = tracker.predict(int(.20e9))
    assert state.mode == "coasting"
    assert state.x > 1.5
    assert state.vx > 0


def test_lost_track_requires_marker_to_reacquire():
    tracker = HybridRoverTracker(coast_ms=200)
    tracker.update(measurement(0, "marker", 1, 1))
    assert not tracker.update(measurement(.30, "lidar", 1.2, 1))
    assert tracker.update(measurement(.30, "marker", 1.2, 1))


def test_different_marker_identity_cannot_steal_track():
    tracker = HybridRoverTracker()
    assert tracker.update(measurement(0, "marker", 0, 0))
    other = RoverMeasurement(10_000_000, "marker", .1, 0, (.01, .01), 1.0, "opponent")
    assert not tracker.update(other)
    assert tracker.identity == "rover-7"


def test_lidar_cluster_is_selected_near_prediction():
    rng = np.random.default_rng(4)
    rover = rng.normal([3.0, 2.0, .3], [.12, .10, .08], (40, 3))
    distractor = rng.normal([4.3, 2.0, .3], [.1, .1, .08], (50, 3))
    floor = rng.normal([3.0, 2.0, 0.0], [.2, .2, .005], (50, 3))
    hit = LidarRoverDetector().detect(np.vstack([rover, distractor, floor]), (3, 2), 2.0)
    assert hit is not None
    assert hit.point_count >= 30
    assert abs(hit.x - 3) < .1 and abs(hit.y - 2) < .1


def test_lidar_gate_accounts_for_two_metres_per_scan_at_40_kmh():
    tracker = HybridRoverTracker()
    tracker.update(measurement(0, "marker", 0, 0))
    state = tracker.predict(0)
    assert lidar_gate_radius(state) >= 1.1


def test_spherical_scan_conversion():
    points = scan_grid_to_points([1, 1], [0, np.pi / 2], [0])
    assert np.allclose(points, [[1, 0, 0], [0, 1, 0]], atol=1e-7)


def test_static_lidar_background_removes_wall_but_not_rover():
    wall = np.array([[0, y, .3] for y in np.linspace(0, 1, 20)])
    background = StaticVoxelBackground(.05, neighbour_cells=0).fit([wall, wall])
    rover = np.array([[1.0, .5, .3], [1.05, .5, .3]])
    foreground = background.remove(np.vstack([wall, rover]))
    assert len(foreground) == 2
    assert np.allclose(foreground, rover)


def test_world_point_projection_uses_optical_extrinsic():
    camera = {
        "position_world": [0, 0, 0],
        "R_world_optical": np.eye(3).tolist(),
        "K": [100, 0, 50, 0, 100, 40, 0, 0, 1],
        "D": [0, 0, 0, 0, 0],
    }
    assert np.allclose(project_world_to_image([1, 0, 2], camera), [100, 40])
    assert project_world_to_image([0, 0, -1], camera) is None


def test_motion_follower_only_operates_inside_predicted_roi():
    follower = CameraMotionFollower(history=20, min_area_px=25)
    blank = np.zeros((160, 200, 3), np.uint8)
    for _ in range(15):
        assert follower.update(blank, None, learning_rate=.5) is None
    frame = blank.copy()
    frame[65:95, 85:125] = 255
    assert follower.update(frame, (105, 80), 50, learning_rate=0) is not None
    assert follower.update(frame, (15, 15), 20, learning_rate=0) is None
