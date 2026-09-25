#!/usr/bin/env python3
"""Run a full-field AprilTag coverage experiment against Gazebo ground truth.

The script creates a per-run world without yolo_rover, starts Gazebo and the
existing six-camera localization runtime, drives tag_rover along a lawnmower
path, publishes evaluator-only truth on /mocap/tag_rover/ground_truth, and
generates CSV files suitable for camera-layout and detector analysis.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import rclpy
    from nav_msgs.msg import Odometry
except ImportError as exc:
    raise SystemExit(
        "ROS 2 environment is missing. Run: source /opt/ros/jazzy/setup.bash"
    ) from exc

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_tag_coverage import analyze

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization_contracts.camera_model import CameraModel, infer_model


ROOT = Path(__file__).resolve().parents[1]
WORLD_SIZE = 12.0


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rotation(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
    ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
    return matmul(matmul(rz, ry), rx)


def make_experiment_world(source, destination):
    tree = ET.parse(source)
    world = tree.getroot().find("world")
    if world is None:
        raise ValueError("SDF has no world")
    opponents = [m for m in world.findall("model") if m.get("name") == "yolo_rover"]
    if len(opponents) != 1:
        raise ValueError(f"expected one yolo_rover, found {len(opponents)}")
    world.remove(opponents[0])
    ET.indent(tree)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def text_float(parent, path):
    node = parent.find(path)
    if node is None or node.text is None:
        raise ValueError(f"missing SDF field {path}")
    return float(node.text)


def export_camera_config(world_path, destination):
    tree = ET.parse(world_path)
    world = tree.getroot().find("world")
    base = json.loads((ROOT / "config/cameras.json").read_text())
    cameras = []
    # OpenCV optical axes (right, down, forward) expressed in a Gazebo camera
    # link (forward +X, left +Y, up +Z).
    gazebo_from_optical = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    for number in range(1, 7):
        name = f"camera_{number}"
        model = world.find(f"model[@name='{name}']")
        if model is None:
            raise ValueError(f"{name} missing from experiment world")
        pose = [float(value) for value in model.findtext("pose").split()]
        sensor = model.find(".//sensor[@type='camera']")
        camera = sensor.find("camera")
        intrinsics = camera.find("lens/intrinsics")
        distortion = camera.find("distortion")
        image = camera.find("image")
        k = [
            text_float(intrinsics, "fx"), 0.0, text_float(intrinsics, "cx"),
            0.0, text_float(intrinsics, "fy"), text_float(intrinsics, "cy"),
            0.0, 0.0, 1.0,
        ]
        # A camera with no <distortion> renders an ideal gnomonical projection,
        # so the runtime must solve with zero distortion.  Copying non-zero
        # coefficients out of the world is only safe when the renderer and
        # OpenCV agree on the model, which validate() below checks.
        if distortion is None:
            d = [0.0] * 5
        else:
            d = [
                text_float(distortion, "k1"), text_float(distortion, "k2"),
                text_float(distortion, "p1"), text_float(distortion, "p2"),
                text_float(distortion, "k3"),
            ]
        cameras.append({
            "name": name,
            "position_world": pose[:3],
            "R_world_optical": matmul(rotation(*pose[3:]), gazebo_from_optical),
            "K": k,
            "D": d,
            "distortion_model": infer_model(d),
            "image_size": [int(image.findtext("width")), int(image.findtext("height"))],
            "horizontal_fov": text_float(camera, "horizontal_fov"),
            "image_topic": sensor.findtext("topic"),
            "camera_info_topic": camera.findtext("camera_info_topic"),
        })
    # Refuse to start a run on a projection model that cannot reach its own
    # image corners: PnP would fail silently over most of every frame.
    for entry in cameras:
        CameraModel.from_config(entry).validate().raise_for_status()
    digest = hashlib.sha256(Path(world_path).read_bytes()).hexdigest()[:12]
    config = {
        "world": world.get("name"),
        "tag": base["tag"],
        "tags": base["tags"],
        "tag_frame_convention": base["tag_frame_convention"],
        "cameras": cameras,
        "role": "image_calibrated",
        "calibration_version": f"gazebo-l2-layout@{digest}",
        "camera_model_validated": True,
    }
    Path(destination).write_text(json.dumps(config, indent=2) + "\n")
    return config


def lawnmower_waypoints(margin, spacing):
    high = WORLD_SIZE - margin
    rows = []
    y = margin
    while y < high - 1e-9:
        rows.append(y)
        y += spacing
    if not rows or high - rows[-1] > 0.2:
        rows.append(high)
    points = []
    for index, y in enumerate(rows):
        endpoints = [(margin, y), (high, y)] if index % 2 == 0 else [(high, y), (margin, y)]
        for point in endpoints:
            if not points or point != points[-1]:
                points.append(point)
    return points


def pose_stamp_ns(pose):
    return pose.header.stamp.sec * 1_000_000_000 + pose.header.stamp.nsec


def yaw_from_quaternion(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def stop_process(process, timeout=12):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="New output directory")
    parser.add_argument("--world", default=str(ROOT / "worlds/mocap_arena_l2.sdf"))
    parser.add_argument("--speed", type=float, default=0.55)
    parser.add_argument("--lane-spacing", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.8)
    parser.add_argument("--waypoint-tolerance", type=float, default=0.22)
    parser.add_argument("--max-seconds", type=float, default=480)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--detector-scale", type=float, default=1.0)
    parser.add_argument("--detector-profile", choices=("coverage", "balanced", "fast"), default="coverage")
    parser.add_argument("--tag-quality-min", type=float, default=0.07)
    parser.add_argument("--tag-max-reprojection-px", type=float, default=2.0)
    parser.add_argument("--tag-max-planar-tilt-deg", type=float, default=40.0)
    parser.add_argument("--tag-min-side-px", type=float, default=20.0)
    # Forwarded to run_localization.py so a run can be reproduced from its
    # manifest alone; every one of these changes what the CSV measures.
    parser.add_argument("--base-z-nominal", type=float, default=0.14)
    parser.add_argument("--base-z-tolerance", type=float, default=0.25)
    parser.add_argument("--max-incidence-deg", type=float, default=65.0)
    parser.add_argument("--pnp-ray-disagreement", type=float, default=0.35)
    parser.add_argument("--min-edge-distance-px", type=float, default=8.0)
    parser.add_argument("--xy-source", choices=("ray", "pnp"), default="ray")
    parser.add_argument("--publish-hz", type=float, default=200.0)
    parser.add_argument("--group-window-ms", type=float, default=12.0)
    parser.add_argument("--coast-ms", type=float, default=300.0)
    parser.add_argument("--identity-max-age-s", type=float, default=2.0)
    parser.add_argument("--lost-ms", type=float, default=1500.0)
    parser.add_argument("--roi-min-px", type=int, default=160)
    parser.add_argument("--roi-max-px", type=int, default=480)
    parser.add_argument("--no-roi-tracking", action="store_true")
    args = parser.parse_args()
    if not 0.1 <= args.speed <= 0.9:
        parser.error("--speed must be in [0.1, 0.9]")
    if not 0.7 <= args.margin <= 2.0:
        parser.error("--margin must be in [0.7, 2.0]")
    if not 0.5 <= args.lane_spacing <= 2.0:
        parser.error("--lane-spacing must be in [0.5, 2.0]")

    run_dir = Path(args.output or ROOT / "artifacts" / ("tag_coverage_" + time.strftime("%Y%m%d_%H%M%S"))).resolve()
    if run_dir.exists():
        raise SystemExit(f"output already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    runtime_dir = run_dir / "runtime"
    world_path = run_dir / "experiment_world.sdf"
    config_path = run_dir / "camera_config.json"
    truth_path = run_dir / "truth.csv"
    make_experiment_world(Path(args.world), world_path)
    config = export_camera_config(world_path, config_path)
    waypoints = lawnmower_waypoints(args.margin, args.lane_spacing)

    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        revision = None
    # A revision alone lies whenever the hot path is edited without a commit,
    # which is exactly how the 20260925 runs became incomparable.  Record the
    # dirty flag and hash every file that decides what the run measures.
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        dirty = None
    code_digest = {}
    for relative in sorted(
        [p.relative_to(ROOT) for p in (ROOT / "localization_contracts").glob("*.py")]
        + [Path("scripts/run_localization.py"), Path("scripts/run_tag_coverage_experiment.py"),
           Path("scripts/analyze_tag_coverage.py")]
    ):
        candidate = ROOT / relative
        if candidate.exists():
            code_digest[str(relative)] = hashlib.sha256(
                candidate.read_bytes()).hexdigest()[:12]
    manifest = {
        "schema": "tag-coverage-experiment-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_world": str(Path(args.world).resolve()),
        "experiment_world": str(world_path),
        "opponent_removed": True,
        "truth_topic": "/mocap/tag_rover/ground_truth",
        "estimate_topics": ["/mocap/tag_rover/odom", "/mocap/tag_rover/status"],
        "frame_id": "arena",
        "speed_m_s": args.speed,
        "margin_m": args.margin,
        "lane_spacing_m": args.lane_spacing,
        "waypoints": waypoints,
        "camera_calibration_version": config["calibration_version"],
        "tag_pipeline": {
            "detector_scale": args.detector_scale,
            "detector_profile": args.detector_profile,
            "quality_min": args.tag_quality_min,
            "max_reprojection_px": args.tag_max_reprojection_px,
            "max_planar_tilt_deg": args.tag_max_planar_tilt_deg,
            "min_side_px": args.tag_min_side_px,
            "base_z_nominal_m": args.base_z_nominal,
            "base_z_tolerance_m": args.base_z_tolerance,
            "max_incidence_deg": args.max_incidence_deg,
            "pnp_ray_disagreement_m": args.pnp_ray_disagreement,
            "min_edge_distance_px": args.min_edge_distance_px,
            "xy_source": args.xy_source,
        },
        "scheduling": {
            "publish_hz": args.publish_hz,
            "group_window_ms": args.group_window_ms,
            "coast_ms": args.coast_ms,
            "identity_max_age_s": args.identity_max_age_s,
            "lost_ms": args.lost_ms,
            "roi_tracking": not args.no_roi_tracking,
            "roi_min_px": args.roi_min_px,
            "roi_max_px": args.roi_max_px,
        },
        "git_revision": revision,
        "git_dirty": dirty,
        "code_digest": code_digest,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rclpy.init()
    ros_node = rclpy.create_node("tag_coverage_ground_truth")
    truth_pub = ros_node.create_publisher(Odometry, "/mocap/tag_rover/ground_truth", 30)
    gz_node = Node()
    cmd_pub = gz_node.advertise("/model/tag_rover/cmd_vel", Twist)
    truth_ready = threading.Event()
    state_lock = threading.Lock()
    rows = queue.SimpleQueue()
    navigation = {"index": 0, "target": waypoints[0], "linear": 0.0, "angular": 0.0}
    latest = {}
    previous = None

    def pose_callback(message):
        nonlocal previous
        for pose in message.pose:
            if pose.name != "tag_rover":
                continue
            stamp = pose_stamp_ns(pose)
            yaw = yaw_from_quaternion(pose.orientation)
            vx = vy = 0.0
            if previous and stamp > previous[0]:
                dt = (stamp - previous[0]) / 1e9
                vx = (pose.position.x - previous[1]) / dt
                vy = (pose.position.y - previous[2]) / dt
            previous = (stamp, pose.position.x, pose.position.y)
            with state_lock:
                nav = dict(navigation)
                latest.update(stamp_ns=stamp, x=pose.position.x, y=pose.position.y, z=pose.position.z, yaw=yaw, vx=vx, vy=vy)
            msg = Odometry()
            msg.header.stamp.sec = stamp // 1_000_000_000
            msg.header.stamp.nanosec = stamp % 1_000_000_000
            msg.header.frame_id = "arena"
            msg.child_frame_id = "tag_rover/ground_truth"
            msg.pose.pose.position.x = pose.position.x
            msg.pose.pose.position.y = pose.position.y
            msg.pose.pose.position.z = pose.position.z
            msg.pose.pose.orientation.x = pose.orientation.x
            msg.pose.pose.orientation.y = pose.orientation.y
            msg.pose.pose.orientation.z = pose.orientation.z
            msg.pose.pose.orientation.w = pose.orientation.w
            msg.twist.twist.linear.x = math.cos(yaw) * vx + math.sin(yaw) * vy
            msg.twist.twist.linear.y = -math.sin(yaw) * vx + math.cos(yaw) * vy
            # SIGINT may shut the ROS context down while a Gazebo transport
            # callback is still in flight. Avoid publishing into an invalid
            # rcl context during orderly experiment teardown.
            if rclpy.ok():
                truth_pub.publish(msg)
            rows.put({
                "stamp_ns": stamp, "wall_ns": time.monotonic_ns(),
                "x": pose.position.x, "y": pose.position.y, "z": pose.position.z,
                "yaw": yaw, "vx": vx, "vy": vy,
                "waypoint_index": nav["index"], "target_x": nav["target"][0],
                "target_y": nav["target"][1], "cmd_linear": nav["linear"],
                "cmd_angular": nav["angular"],
            })
            truth_ready.set()

    pose_topic = "/model/tag_rover/pose"
    if not gz_node.subscribe(Pose_V, pose_topic, pose_callback):
        raise RuntimeError(f"cannot subscribe {pose_topic}")

    gazebo = localization = None
    gazebo_log = (run_dir / "gazebo.log").open("w")
    localization_log = (run_dir / "localization.log").open("w")
    fields = ["stamp_ns", "wall_ns", "x", "y", "z", "yaw", "vx", "vy", "waypoint_index", "target_x", "target_y", "cmd_linear", "cmd_angular"]
    truth_stream = truth_path.open("w", newline="")
    truth_writer = csv.DictWriter(truth_stream, fieldnames=fields)
    truth_writer.writeheader()

    def drain_rows():
        while True:
            try:
                truth_writer.writerow(rows.get_nowait())
            except queue.Empty:
                break
        truth_stream.flush()

    exit_reason = "unknown"
    failure = None
    start = time.monotonic()
    try:
        env = os.environ.copy()
        env["MOCAP_WORLD"] = str(world_path)
        gazebo_command = [str(ROOT / "scripts/run.sh")]
        if args.headless:
            gazebo_command += ["-s", "--headless-rendering"]
        gazebo = subprocess.Popen(gazebo_command, cwd=ROOT, env=env, stdout=gazebo_log, stderr=subprocess.STDOUT, start_new_session=True)
        if not truth_ready.wait(45):
            raise RuntimeError("Gazebo did not publish tag_rover pose within 45 seconds")

        localization = subprocess.Popen([
            str(ROOT / "scripts/run_localization.sh"), "--config", str(config_path),
            "--tag-only", "--no-ui", "--clock-topic",
            "/world/mocap_arena/clock", "--output", str(runtime_dir),
            "--detector-scale", str(args.detector_scale),
            "--detector-profile", args.detector_profile,
            "--tag-quality-min", str(args.tag_quality_min),
            "--tag-max-reprojection-px", str(args.tag_max_reprojection_px),
            "--tag-max-planar-tilt-deg", str(args.tag_max_planar_tilt_deg),
            "--tag-min-side-px", str(args.tag_min_side_px),
            "--base-z-nominal", str(args.base_z_nominal),
            "--base-z-tolerance", str(args.base_z_tolerance),
            "--max-incidence-deg", str(args.max_incidence_deg),
            "--pnp-ray-disagreement", str(args.pnp_ray_disagreement),
            "--min-edge-distance-px", str(args.min_edge_distance_px),
            "--xy-source", args.xy_source,
            "--publish-hz", str(args.publish_hz),
            "--group-window-ms", str(args.group_window_ms),
            "--coast-ms", str(args.coast_ms),
            "--identity-max-age-s", str(args.identity_max_age_s),
            "--lost-ms", str(args.lost_ms),
            "--roi-min-px", str(args.roi_min_px),
            "--roi-max-px", str(args.roi_max_px),
        ] + (["--no-roi-tracking"] if args.no_roi_tracking else []), cwd=ROOT, stdout=localization_log, stderr=subprocess.STDOUT, start_new_session=True)
        ready_deadline = time.monotonic() + 45
        while time.monotonic() < ready_deadline:
            drain_rows()
            rclpy.spin_once(ros_node, timeout_sec=0.05)
            if localization.poll() is not None:
                raise RuntimeError("localization exited during startup; inspect localization.log")
            if (runtime_dir / "status.json").exists():
                break
        else:
            raise RuntimeError("localization did not become ready within 45 seconds")

        print(f"Experiment directory: {run_dir}")
        print(f"Ground truth ROS topic: /mocap/tag_rover/ground_truth")
        print(f"Waypoints: {len(waypoints)}; estimated limit: {args.max_seconds:.0f} s")
        warmup_until = time.monotonic() + 3
        while time.monotonic() < warmup_until:
            drain_rows()
            rclpy.spin_once(ros_node, timeout_sec=0.05)

        while navigation["index"] < len(waypoints):
            if time.monotonic() - start > args.max_seconds:
                exit_reason = "timeout"
                break
            if gazebo.poll() is not None:
                raise RuntimeError("Gazebo exited unexpectedly; inspect gazebo.log")
            if localization.poll() is not None:
                raise RuntimeError("localization exited unexpectedly; inspect localization.log")
            with state_lock:
                pose = dict(latest)
                index = navigation["index"]
            target_x, target_y = waypoints[index]
            dx, dy = target_x - pose["x"], target_y - pose["y"]
            distance = math.hypot(dx, dy)
            if distance < args.waypoint_tolerance:
                with state_lock:
                    navigation["index"] += 1
                    if navigation["index"] < len(waypoints):
                        navigation["target"] = waypoints[navigation["index"]]
                    navigation["linear"] = navigation["angular"] = 0.0
                cmd_pub.publish(Twist())
                continue
            heading_error = (math.atan2(dy, dx) - pose["yaw"] + math.pi) % (2 * math.pi) - math.pi
            angular = max(-1.5, min(1.5, 2.4 * heading_error))
            alignment = max(0.0, 1.0 - abs(heading_error) / 0.7)
            linear = min(args.speed, distance) * alignment
            command = Twist()
            command.linear.x = linear
            command.angular.z = angular
            cmd_pub.publish(command)
            with state_lock:
                navigation["linear"] = linear
                navigation["angular"] = angular
            drain_rows()
            rclpy.spin_once(ros_node, timeout_sec=0.05)
        else:
            exit_reason = "completed"

        cmd_pub.publish(Twist())
        settle_until = time.monotonic() + 2
        while time.monotonic() < settle_until:
            drain_rows()
            rclpy.spin_once(ros_node, timeout_sec=0.05)
    except KeyboardInterrupt:
        exit_reason = "interrupted"
    except Exception as exc:
        exit_reason = "error"
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        cmd_pub.publish(Twist())
        stop_process(localization)
        stop_process(gazebo)
        drain_rows()
        truth_stream.close()
        gazebo_log.close()
        localization_log.close()
        gz_node.unsubscribe(pose_topic)
        ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    manifest["exit_reason"] = exit_reason
    manifest["wall_seconds"] = time.monotonic() - start
    manifest["completed_waypoints"] = navigation["index"]
    manifest["failure"] = failure
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if (runtime_dir / "camera_frames.jsonl").exists():
        report = analyze(run_dir)
        print(json.dumps(report, indent=2))
    else:
        print("No runtime camera log was produced; inspect the *.log files.", file=sys.stderr)
    print(f"Done: {run_dir}")
    if failure:
        raise SystemExit(failure)


if __name__ == "__main__":
    main()
