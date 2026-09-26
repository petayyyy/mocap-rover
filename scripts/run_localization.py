#!/usr/bin/env python3
"""Live six-camera image localization, independent publication, UI and recording.

Runtime subscribes only RGB and /clock. It never imports truth or evaluator data.
Run with the simulation venv; --ros requires sourcing /opt/ros/jazzy/setup.bash.
"""
import fcntl
import hashlib
import tempfile
import argparse, collections, dataclasses, json, math, os, signal, sys, threading, time
from pathlib import Path
import cv2
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.laserscan_pb2 import LaserScan

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization_contracts import lidar_pipeline, roi_tracker
from localization_contracts.camera_model import CameraModel
from localization_contracts.capture import LatestFrames, rgb_array
from localization_contracts.image_pipeline import OneCameraImagePipeline
from localization_contracts.rover_filter import (
    AsyncObservationBuffer, ImmRoverFilter, Measurement, POSITION,
    measurement_from_observation,
)
from localization_contracts.contracts import Observation, SCHEMA_VERSION
from localization_contracts.cuboid import localize_box
from localization_contracts.dashboard import DashboardHandler, serve
from localization_contracts.marker_families import normalize_marker_family


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--weights")
    p.add_argument("--tag-only", action="store_true", help="Disable opponent inference and publication")
    p.add_argument(
        "--output",
        help="New recording directory; existing sessions are never overwritten",
    )
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--ros", action="store_true")
    p.add_argument("--no-ui", action="store_true")
    p.add_argument("--detector-scale", type=float, default=1.0)
    p.add_argument("--detector-profile", choices=("coverage", "balanced", "fast"),
                   default="coverage")
    p.add_argument("--tag-quality-min", type=float, default=0.07)
    p.add_argument("--tag-max-reprojection-px", type=float, default=2.0)
    p.add_argument("--tag-max-planar-tilt-deg", type=float, default=40.0)
    p.add_argument("--tag-min-side-px", type=float, default=20.0)
    # Physical gates. The rover stands on a known plane, so its base height,
    # the marker plane and the arena bounds are measured constants rather than
    # free parameters, and a solution that contradicts them is wrong however
    # small its reprojection error.
    p.add_argument("--base-z-nominal", type=float, default=0.14,
                   help="Height of the rover base link above the arena floor")
    p.add_argument("--base-z-tolerance", type=float, default=0.25,
                   help="Gross-error bound on the PnP base height")
    p.add_argument("--max-incidence-deg", type=float, default=65.0)
    p.add_argument("--pnp-ray-disagreement", type=float, default=0.35,
                   help="Largest accepted gap between the PnP and ray XY estimates")
    p.add_argument("--min-edge-distance-px", type=float, default=8.0)
    p.add_argument("--xy-source", choices=("ray", "pnp"), default="ray",
                   help="ray intersects the marker bearing with its known plane")
    # Scheduling. At 11.11 m/s every millisecond is 11 mm, so the group window
    # is a latency budget, not a synchronisation device.
    p.add_argument("--publish-hz", type=float, default=200.0)
    p.add_argument("--group-window-ms", type=float, default=12.0)
    p.add_argument("--coast-ms", type=float, default=300.0,
                   help="Measurement age at which the track stops being valid")
    p.add_argument("--identity-max-age-s", type=float, default=2.0,
                   help="How long a track may run on continuation sources alone")
    p.add_argument("--lost-ms", type=float, default=1500.0)
    p.add_argument("--max-speed-mps", type=float, default=13.0)
    # Two-mode detection. A confident track puts the marker inside a small
    # window, and a 240x240 window costs 0.38 ms against 3.0 ms for the frame.
    p.add_argument("--roi-min-px", type=int, default=160)
    p.add_argument("--roi-max-px", type=int, default=480)
    p.add_argument("--watchdog-period-s", type=float, default=2.0,
                   help="How often a camera the prediction misses still looks")
    p.add_argument("--no-roi-tracking", action="store_true",
                   help="Always run the full-frame detector")
    # Unitree L2. Continuation only: it can keep a track alive where no camera
    # reaches, but it may not create one, because a second rover and a shadow
    # both look like a cluster of the right size.
    p.add_argument("--no-lidar", action="store_true")
    p.add_argument("--lidar-topic", default=None,
                   help="Defaults to the topic recorded in the calibration")
    p.add_argument("--lidar-z-band", type=float, nargs=2,
                   default=lidar_pipeline.DEFAULT_Z_BAND)
    p.add_argument("--lidar-max-radius", type=float,
                   default=lidar_pipeline.DEFAULT_MAX_USEFUL_RADIUS_M,
                   help="Sanity bound only; point count sets the working range")
    p.add_argument("--lidar-max-extent", type=float,
                   default=lidar_pipeline.DEFAULT_MAX_EXTENT_M)
    p.add_argument("--lidar-min-points", type=int,
                   default=lidar_pipeline.DEFAULT_MIN_POINTS)
    p.add_argument("--lidar-sweep-s", type=float,
                   default=lidar_pipeline.DEFAULT_SWEEP_DURATION_S,
                   help="Revolution time for deskew; a gz gpu_lidar renders "
                        "the whole grid at one instant, so 0 is correct there")
    p.add_argument("--lidar-background",
                   help="JSON static voxel map from record_lidar_background.py")
    p.add_argument(
        "--clock-topic",
        default="/clock",
        help="Gazebo clock topic (for example /world/mocap_arena/clock)",
    )
    a = p.parse_args()
    if not a.tag_only and not a.weights:
        p.error("--weights is required unless --tag-only")
    if a.seconds < 0:
        p.error("--seconds must be nonnegative")
    if not 0 < a.detector_scale <= 1:
        p.error("--detector-scale must be in (0, 1]")
    if not 0 <= a.tag_quality_min <= 1:
        p.error("--tag-quality-min must be in [0, 1]")
    if a.tag_min_side_px < 4 or a.tag_max_reprojection_px <= 0:
        p.error("invalid marker geometry thresholds")
    runtime_lock = Path("/tmp/mocap-localization.lock").open("w")
    try:
        fcntl.flock(runtime_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("localization runtime already running")
    # Bind before starting transport or workers: a busy port must not leave
    # inference threads running after an initialization exception.
    server = serve("127.0.0.1", a.port) if not a.no_ui else None
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    cv2.setNumThreads(1)
    if not a.tag_only:
        import torch
        torch.set_num_threads(1)
        from ultralytics import YOLO
        model = YOLO(a.weights)
        model.predict(np.zeros((540, 720, 3), np.uint8), imgsz=640, device=0, verbose=False)
    cfg = json.loads(Path(a.config).read_text())
    if cfg.get("role") != "image_calibrated":
        raise ValueError(
            "run image calibration first; nominal/truth files are not runtime calibration"
        )
    cams = {c["name"]: c for c in cfg["cameras"]}
    # A calibration whose distortion model cannot reach its own image corners
    # deletes most of every frame from PnP without reporting an error.
    for camera in cfg["cameras"]:
        CameraModel.from_config(camera).validate().raise_for_status()
    out = Path(a.output or ("artifacts/live_" + time.strftime("%Y%m%d_%H%M%S")))
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("observations.jsonl", "odometry.jsonl")):
        raise ValueError("recording already exists; choose a new --output directory")
    (out / "initial_calibration.json").write_text(json.dumps(cfg, indent=2) + "\n")
    history = collections.deque(maxlen=5)
    version = cfg["calibration_version"]
    tag_entries = cfg["tags"]
    marker_families = {normalize_marker_family(t["family"]) for t in tag_entries}
    marker_sizes = {float(t["size"]) for t in tag_entries}
    if len(marker_families) != 1 or len(marker_sizes) != 1:
        raise ValueError("one runtime track must use one marker family and one marker size")
    marker_family = marker_families.pop()
    tag_size_m = marker_sizes.pop()
    tags = {
        int(t["id"]): {
            "rotation": t["R_base_tag"],
            "translation": t["T_base_tag_translation"],
        }
        for t in tag_entries
    }
    if not tags:
        raise ValueError("at least one configured marker is required")
    (out / "runtime_parameters.json").write_text(json.dumps({
        "marker_family": marker_family,
        "marker_ids": sorted(tags),
        "tag_size_m": tag_size_m,
        "detector_scale": a.detector_scale,
        "detector_profile": a.detector_profile,
        "tag_quality_min": a.tag_quality_min,
        "tag_max_reprojection_px": a.tag_max_reprojection_px,
        "tag_max_planar_tilt_deg": a.tag_max_planar_tilt_deg,
        "tag_min_side_px": a.tag_min_side_px,
        "base_z_nominal_m": a.base_z_nominal,
        "base_z_tolerance_m": a.base_z_tolerance,
        "max_incidence_deg": a.max_incidence_deg,
        "pnp_ray_disagreement_m": a.pnp_ray_disagreement,
        "min_edge_distance_px": a.min_edge_distance_px,
        "xy_source": a.xy_source,
        "publish_hz": a.publish_hz,
        "group_window_ms": a.group_window_ms,
        "coast_ms": a.coast_ms,
        "identity_max_age_s": a.identity_max_age_s,
        "lost_ms": a.lost_ms,
        "roi_min_px": a.roi_min_px,
        "roi_max_px": a.roi_max_px,
        "roi_tracking": not a.no_roi_tracking,
        "watchdog_period_s": a.watchdog_period_s,
        "lidar_enabled": bool(not a.no_lidar and cfg.get("lidar")),
        "lidar_topic": (cfg.get("lidar") or {}).get("topic") if not a.no_lidar else None,
        "lidar_max_radius_m": a.lidar_max_radius,
        "lidar_sweep_s": a.lidar_sweep_s,
        "camera_policy": "asynchronous_group_window",
        "covariance_model": "ray_plane_anisotropic_v2",
    }, indent=2) + "\n")
    pose_gates = dict(
        base_z_nominal_m=a.base_z_nominal,
        base_z_tolerance_m=a.base_z_tolerance,
        max_incidence_deg=a.max_incidence_deg,
        pnp_ray_disagreement_m=a.pnp_ray_disagreement,
        min_edge_distance_px=a.min_edge_distance_px,
        xy_source=a.xy_source,
    )
    pipes = {
        cid: OneCameraImagePipeline(
            cid,
            np.array(c["K"]).reshape(3, 3),
            c["D"],
            {"rotation": c["R_world_optical"], "translation": c["position_world"]},
            tags,
            version,
            family=marker_family,
            tag_size_m=tag_size_m,
            detector_scale=a.detector_scale,
            detector_profile=a.detector_profile,
            marker_ids=tuple(tags),
            quality_min=a.tag_quality_min,
            max_reprojection_px=a.tag_max_reprojection_px,
            max_planar_tilt_deg=a.tag_max_planar_tilt_deg,
            min_side_px=a.tag_min_side_px,
            image_size=c["image_size"],
            **pose_gates,
        )
        for cid, c in cams.items()
    }
    capture = LatestFrames(cams)
    yolo_capture = LatestFrames(cams)
    node = Node()
    stop = threading.Event()
    lock = threading.RLock()
    # No barrier. Each camera publishes its own observations the moment it has
    # them; the buffer closes a group on a timeout measured from capture, so a
    # fast camera never waits for a slow one.
    def make_filter():
        return ImmRoverFilter(
            coast_ms=a.coast_ms,
            identity_max_age_s=a.identity_max_age_s,
            lost_ms=a.lost_ms,
            max_speed_mps=a.max_speed_mps,
        )

    filters = {"tag_rover": make_filter(), "opponent": make_filter()}
    buffers = {name: AsyncObservationBuffer(int(a.group_window_ms * 1e6))
               for name in filters}
    if a.tag_only:
        del filters["opponent"]
        del buffers["opponent"]
    metrics = {
        cid: {
            "processed": 0,
            "tag_hits": 0,
            "tag_accepted": 0,
            "yolo_hits": 0,
            "idle_frames": 0,
            "last_capture_ns": 0,
            "latency_ms": 0,
        }
        for cid in cams
    }
    planners = {
        cid: roi_tracker.CameraRoiPlanner(
            CameraModel.from_config(c), c["R_world_optical"], c["position_world"],
            min_roi_px=a.roi_min_px, max_roi_px=a.roi_max_px,
            marker_size_m=tag_size_m, watchdog_period_s=a.watchdog_period_s,
        )
        for cid, c in cams.items()
    }
    for planner in planners.values():
        planner.tag_plane_z = a.base_z_nominal + max(
            float(t["T_base_tag_translation"][2]) for t in tag_entries)

    lidar_config = cfg.get("lidar")
    lidar = None
    if not a.no_lidar and lidar_config:
        background = None
        if a.lidar_background:
            background = lidar_pipeline.StaticVoxelMap.from_dict(
                json.loads(Path(a.lidar_background).read_text()))
        lidar = lidar_pipeline.ArenaLidar(
            lidar_config["position_world"], lidar_config["R_world_sensor"],
            z_band=tuple(a.lidar_z_band),
            sweep_duration_s=a.lidar_sweep_s,
            max_useful_radius_m=a.lidar_max_radius,
            max_extent_m=a.lidar_max_extent,
            min_points=a.lidar_min_points,
            background=background,
        )
    lidar_topic = a.lidar_topic or (lidar_config or {}).get("topic")
    lidar_rows = collections.deque(maxlen=8192)
    lidar_latest = {"scan": None}
    lidar_event = threading.Event()

    def track_prediction():
        """(x, y, sigma) of the tag_rover track, or None when there is none."""
        with lock:
            f = filters.get("tag_rover")
            if f is None or not f.initialized:
                return None
            if f.tracking_state(clock["sim"]) == "LOST":
                return None
            state, covariance = f.x, f.P
        return (float(state[0]), float(state[1]),
                float(math.sqrt(max(covariance[0, 0], covariance[1, 1]))))
    clock = {"sim": 0, "wall": time.monotonic_ns(), "resets": 0}
    tracks = {}
    previews = {}
    errors = []
    start = time.monotonic()
    topics = []
    threads = []
    observations = collections.deque(maxlen=4096)
    pending_observations = {name: [] for name in filters}
    publication = collections.deque(maxlen=4096)
    camera_frames = collections.deque(maxlen=262144)
    rec_drops = 0
    record_done = threading.Event()
    accepted_wall = {name: collections.deque(maxlen=300) for name in filters}
    published_wall = {name: collections.deque(maxlen=300) for name in filters}

    def halt(*_):
        stop.set()

    signal.signal(signal.SIGINT, halt)
    signal.signal(signal.SIGTERM, halt)

    def rate(stamps):
        return (
            (len(stamps) - 1) * 1e9 / (stamps[-1] - stamps[0])
            if len(stamps) > 1 and stamps[-1] > stamps[0]
            else 0.0
        )

    rospy = None
    publishers = {}
    if a.ros:
        import rclpy
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String

        rclpy.init()
        rospy = rclpy.create_node("mocap_image_localization")
        publishers = {
            name: (
                rospy.create_publisher(Odometry, f"/mocap/{name}/odom", 10),
                rospy.create_publisher(String, f"/mocap/{name}/status", 10),
            )
            for name in filters
        }

    def clock_cb(msg):
        stamp = msg.sim.sec * 1_000_000_000 + msg.sim.nsec
        with lock:
            if stamp < clock["sim"]:
                for f in filters.values():
                    f.reset()
                clock["resets"] += 1
                tracks.clear()
                for values in accepted_wall.values():
                    values.clear()
                for values in published_wall.values():
                    values.clear()
            if stamp != clock["sim"]:
                clock.update(sim=stamp, wall=time.monotonic_ns())

    if not node.subscribe(Clock, a.clock_topic, clock_cb):
        raise RuntimeError(a.clock_topic)
    topics.append(a.clock_topic)

    def callback(msg, cid):
        capture.push(cid, msg)
        if not a.tag_only:
            yolo_capture.push(cid, msg)

    if lidar is not None:
        def lidar_cb(msg):
            with lock:
                lidar_latest["scan"] = msg
            lidar_event.set()

        if not node.subscribe(LaserScan, lidar_topic, lidar_cb):
            raise RuntimeError(lidar_topic)
        topics.append(lidar_topic)

    for cid in cams:
        topic = f"/cameras/{cid}/image"
        topics.append(topic)
        if not node.subscribe(Image, topic, lambda msg, cid=cid: callback(msg, cid)):
            raise RuntimeError(topic)

    def record_observation(obs, accepted, reason):
        nonlocal rec_drops
        with lock:
            if len(observations) == observations.maxlen:
                rec_drops += 1
            observations.append(
                {
                    "observation": dataclasses.asdict(obs),
                    "accepted": bool(accepted),
                    "selection_reason": reason,
                    "wall_ns": time.monotonic_ns(),
                }
            )

    def enqueue(obs):
        """Hand one observation to its track's buffer; never blocks a worker."""
        with lock:
            if (
                obs.calibration_version != version
                or obs.capture_time_ns > clock["sim"] + 100_000_000
            ):
                record_observation(obs, False, "calibration_or_future_stamp")
                return False
            buffer = buffers.get(obs.object_id)
            if buffer is None:
                return False
            for measurement in measurement_from_observation(obs):
                buffer.push(measurement)
            pending_observations[obs.object_id].append(obs)
            return True

    def drain_buffers(now_ns, force=False):
        """Close ready groups and apply them; called from the publisher tick."""
        with lock:
            for name, buffer in buffers.items():
                for group in buffer.drain(now_ns, force):
                    applied = filters[name].apply_group(group)
                    if applied:
                        accepted_wall[name].append(time.monotonic_ns())
                    window = {m.stamp_ns for m in group}
                    remaining = []
                    for obs in pending_observations[name]:
                        if obs.capture_time_ns in window:
                            record_observation(
                                obs, applied,
                                "fusion_accepted" if applied else "fusion_gate")
                            if applied:
                                mark_frame_accepted(obs)
                        else:
                            remaining.append(obs)
                    pending_observations[name] = remaining

    def mark_frame_accepted(obs):
        with lock:
            metrics[obs.camera_id]["tag_accepted"] += 1
            for row in reversed(camera_frames):
                if row["camera_id"] == obs.camera_id and row["capture_ns"] == obs.capture_time_ns:
                    row["fusion_accepted"] += 1
                    break

    def tag_worker(cid):
        try:
            preview_at = 0
            while not stop.is_set():
                item = capture.take(cid)
                if item is None:
                    continue
                msg, received, seq = item
                begin = time.monotonic_ns()
                image = rgb_array(msg)
                stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nsec
                # Detector + PnP; the global filter receives the raw observations.
                with lock:
                    pipe = pipes[cid]
                    cam = cams[cid]
                if [msg.width, msg.height] != cam["image_size"]:
                    raise ValueError("live image size differs from calibration")
                # Look where the track says the marker is, not everywhere.
                plan = planners[cid].plan(
                    None if a.no_roi_tracking else track_prediction(), begin)
                if plan.mode == roi_tracker.IDLE:
                    with lock:
                        metrics[cid]["idle_frames"] += 1
                    continue
                hits = pipe.detector.detect(
                    image, roi=plan.roi if plan.mode == roi_tracker.ROI else None)
                planners[cid].report(bool(hits))
                pnp_count = 0
                qualities = []
                reprojection_errors = []
                frame_observations = []
                pnp_rejections = collections.Counter()
                pnp_diagnostics = []
                from localization_contracts.apriltag import Detection

                for hit in hits:
                    obs = pipe.observer.observe(
                        Detection(
                            cid,
                            seq,
                            hit.tag_id,
                            hit.corners,
                            stamp,
                            received,
                            time.monotonic_ns(),
                        )
                    )
                    diagnostic = pipe.observer.last_diagnostic or {}
                    pnp_diagnostics.append({"tag_id": int(hit.tag_id), **diagnostic})
                    if obs is not None:
                        pnp_count += 1
                        qualities.append(float(obs.quality))
                        reprojection_errors.append(
                            float(obs.pixel_features["reprojection_error_px"])
                        )
                        # The observer now derives an anisotropic covariance
                        # from corner noise, incidence and the known marker
                        # plane.  The per-camera fudge factor that used to live
                        # here could not express that the error is elongated
                        # along the viewing ray, so it was simultaneously
                        # overconfident across the ray and loose along it.
                        frame_observations.append(obs)
                    else:
                        pnp_rejections[str(diagnostic.get("reason", "unknown"))] += 1
                if not a.no_ui and time.monotonic() - preview_at > 0.5:
                    preview_at = time.monotonic()
                    small = cv2.resize(image, (360, 270))
                    ok, jpg = cv2.imencode(
                        ".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
                    )
                    if ok:
                        tmp = out / f"{cid}.tmp"
                        tmp.write_bytes(jpg.tobytes())
                        tmp.replace(out / f"{cid}.jpg")
                with lock:
                    m = metrics[cid]
                    m["processed"] += 1
                    m["tag_hits"] += len(hits)
                    m["last_capture_ns"] = stamp
                    m["latency_ms"] = (time.monotonic_ns() - begin) / 1e6
                    camera_frames.append(
                        {
                            "camera_id": cid,
                            "sequence": seq,
                            "capture_ns": stamp,
                            "received_wall_ns": received,
                            "processed_wall_ns": time.monotonic_ns(),
                            "detections": len(hits),
                            "tag_ids": [int(hit.tag_id) for hit in hits],
                            "pnp_valid": pnp_count,
                            "fusion_accepted": 0,
                            "best_quality": max(qualities) if qualities else None,
                            "best_reprojection_px": (
                                min(reprojection_errors)
                                if reprojection_errors
                                else None
                            ),
                            "latency_ms": m["latency_ms"],
                            "mode": plan.mode,
                            "roi": list(plan.roi) if plan.roi else None,
                            "pnp_rejections": dict(pnp_rejections),
                            "pnp_diagnostics": pnp_diagnostics,
                        }
                    )
                for observation in frame_observations:
                    enqueue(observation)
        except Exception as exc:
            errors.append(f"{cid}: {exc}")
            stop.set()

    def lidar_worker():
        """One position per revolution, only for a track a marker created."""
        while not stop.is_set():
            if not lidar_event.wait(0.2):
                continue
            lidar_event.clear()
            with lock:
                message = lidar_latest["scan"]
                lidar_latest["scan"] = None
            if message is None:
                continue
            begin = time.monotonic_ns()
            parsed = lidar_pipeline.parse_laser_scan(message)
            with lock:
                track = filters.get("tag_rover")
                state = track.x if track is not None and track.initialized else None
                covariance = track.P if state is not None else None
                tracking = (track.tracking_state(parsed["stamp_ns"])
                            if track is not None else "LOST")
            row = {"stamp_ns": parsed["stamp_ns"], "wall_ns": time.monotonic_ns(),
                   "tracking_state": tracking}
            # The lidar continues a track a marker created; it never starts
            # one. LOST is the bar, not REACQUIRING -- see may_continue.
            if state is None or not lidar_pipeline.may_continue(tracking):
                row["reason"] = "no_confirmed_track"
                with lock:
                    lidar_rows.append(row)
                continue
            scan = lidar.scan_to_arena(parsed)
            velocity = (float(state[2]), float(state[3]), 0.0)
            points = lidar.deskew(scan, velocity)
            sigma = math.sqrt(max(float(covariance[0, 0]), float(covariance[1, 1])))
            speed = math.hypot(velocity[0], velocity[1])
            before = dict(lidar.rejections)
            cluster = lidar.detect(points, (float(state[0]), float(state[1])),
                                   sigma, speed, parsed["stamp_ns"])
            row.update(returns=scan.returns, rays=scan.rays,
                       prediction=[float(state[0]), float(state[1])],
                       prediction_sigma_m=sigma,
                       processing_ms=(time.monotonic_ns() - begin) / 1e6)
            if cluster is None:
                new = [k for k, v in lidar.rejections.items() if v != before.get(k)]
                row["reason"] = new[0] if new else "unknown"
            else:
                row.update(reason="accepted", x=cluster.x, y=cluster.y,
                           z_max=cluster.z_max, points=cluster.points,
                           extent_x=cluster.extent_x, extent_y=cluster.extent_y,
                           residual_m=cluster.residual_m, sigma_m=cluster.sigma_m)
                with lock:
                    buffers["tag_rover"].push(
                        lidar_pipeline.measurement_from_cluster(cluster))
            with lock:
                lidar_rows.append(row)

    def yolo_worker():
        try:
            next_inference = time.monotonic()
            while not stop.is_set():
                stop.wait(max(0.0, next_inference - time.monotonic()))
                if stop.is_set():
                    break
                next_inference = time.monotonic() + 1 / 15
                batch = []
                ids = []
                for cid in cams:
                    item = yolo_capture.take(cid, 0)
                    if item is not None:
                        msg, received, seq = item
                        image = rgb_array(msg)
                        batch.append(
                            cv2.cvtColor(
                                cv2.resize(image, (720, 540)), cv2.COLOR_RGB2BGR
                            )
                        )
                        ids.append((cid, msg, received, seq))
                if not batch:
                    continue
                with lock:
                    batch_cams = cams
                    batch_version = version
                results = model.predict(
                    batch, imgsz=640, conf=0.3, device=0, verbose=False
                )
                candidates = []
                for result, (cid, msg, received, seq) in zip(results, ids):
                    stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nsec
                    for box in result.boxes:
                        if int(box.cls.item()) != 1:
                            continue
                        bbox = box.xyxy[0].cpu().numpy() * np.array(
                            [
                                msg.width / 720,
                                msg.height / 540,
                                msg.width / 720,
                                msg.height / 540,
                            ]
                        )
                        if (
                            min(bbox[:2]) <= 2
                            or bbox[2] >= msg.width - 2
                            or bbox[3] >= msg.height - 2
                        ):
                            continue
                        localized = localize_box(bbox, batch_cams[cid])
                        conf = float(box.conf.item())
                        if localized["box_fit_rms_px"] > 8:
                            continue
                        obs = Observation(
                            SCHEMA_VERSION,
                            cid,
                            seq,
                            f"{cid}:{seq}:yolo",
                            "opponent",
                            stamp,
                            "sim",
                            0,
                            0,
                            received,
                            time.monotonic_ns(),
                            batch_version,
                            "arena",
                            localized["position_m"],
                            (0.0025, 0.0025, 0.01),
                            conf,
                            "yolo_cuboid",
                            pixel_features={"box": bbox.tolist(), "yaw_valid": False},
                        )
                        candidates.append(obs)
                        with lock:
                            metrics[cid]["yolo_hits"] += 1
                # Select the most confident view per capture timestamp.
                selected = {}
                for obs in candidates:
                    old = selected.get(obs.capture_time_ns)
                    if old is None or obs.quality > old.quality:
                        selected[obs.capture_time_ns] = obs
                for stamp in sorted(selected):
                    enqueue(selected[stamp])
        except Exception as exc:
            errors.append("yolo: " + str(exc))
            stop.set()

    def record_worker():
        with (out / "observations.jsonl").open("w") as fo, (
            out / "odometry.jsonl"
        ).open("w") as fp, (out / "camera_frames.jsonl").open("w") as fc, (
            out / "lidar.jsonl"
        ).open("w") as fl:
            while (
                not record_done.is_set()
                or observations
                or publication
                or camera_frames
                or lidar_rows
            ):
                with lock:
                    ob = list(observations)
                    observations.clear()
                    pub = list(publication)
                    publication.clear()
                    frames = list(camera_frames)
                    camera_frames.clear()
                    scans = list(lidar_rows)
                    lidar_rows.clear()
                for row in scans:
                    fl.write(json.dumps(row) + "\n")
                fl.flush()
                for row in ob:
                    fo.write(json.dumps(row) + "\n")
                for row in pub:
                    fp.write(json.dumps(row) + "\n")
                for row in frames:
                    fc.write(json.dumps(row) + "\n")
                fo.flush()
                fp.flush()
                fc.flush()
                record_done.wait(0.1)

    def publisher_worker():
        """Close ready groups and publish the propagated state at --publish-hz.

        Output rate is decoupled from measurement rate: the filter propagates
        to the current clock every tick, and measurement_age_ms reports how
        stale the newest measurement behind that state is.  Raising the rate
        never makes a stale pose look fresh.
        """
        nonlocal rec_drops
        period = 1.0 / a.publish_hz
        next_tick = time.monotonic()
        tick = 0
        while not stop.is_set():
            stop.wait(max(0, next_tick - time.monotonic()))
            next_tick += period
            if stop.is_set():
                break
            if time.monotonic() - next_tick > 0.1:
                next_tick = time.monotonic() + period
            tick += 1
            with lock:
                stamp = clock["sim"]
            drain_buffers(stamp)
            with lock:
                for name, f in filters.items():
                    item = f.publish(stamp)
                    state = item["state"]
                    row = {
                        **item,
                        "object_id": name,
                        "capture_ns": f.last_measurement_ns,
                        "wall_ns": time.monotonic_ns(),
                        "yaw_valid": name == "tag_rover" and state is not None,
                        "source_mask": list(item["sources"]),
                    }
                    now_ns = time.monotonic_ns()
                    published_wall[name].append(now_ns)
                    row.update(
                        measurement_wall_hz=rate(accepted_wall[name]),
                        output_wall_hz=rate(published_wall[name]),
                        session=clock["resets"],
                        out_of_sequence=f.out_of_sequence,
                        dropped_too_old=f.too_old,
                        id_rejections=f.id_rejections,
                    )
                    if now_ns - clock["wall"] > 200_000_000:
                        row.update(valid=False, tracking_state="TIME_UNCERTAIN")
                    tracks[name] = row
                    if len(publication) == publication.maxlen:
                        rec_drops += 1
                    publication.append(row)
                    if rospy is None or state is None:
                        continue
                    odom_pub, status_pub = publishers[name]
                    status = String()
                    status.data = json.dumps(
                        {k: v for k, v in row.items() if k != "covariance"}
                    )
                    status_pub.publish(status)
                    if not row["valid"]:
                        continue
                    covariance = item["covariance"]
                    msg = Odometry()
                    msg.header.frame_id = "arena"
                    msg.child_frame_id = name + "/base_link"
                    msg.header.stamp.sec = stamp // 1_000_000_000
                    msg.header.stamp.nanosec = stamp % 1_000_000_000
                    msg.pose.pose.position.x = state["x"]
                    msg.pose.pose.position.y = state["y"]
                    msg.pose.pose.position.z = a.base_z_nominal
                    yaw = state["yaw"] if name == "tag_rover" else 0.0
                    msg.pose.pose.orientation.z = math.sin(yaw / 2)
                    msg.pose.pose.orientation.w = math.cos(yaw / 2)
                    msg.twist.twist.linear.x = (
                        math.cos(yaw) * state["vx"] + math.sin(yaw) * state["vy"])
                    msg.twist.twist.linear.y = (
                        -math.sin(yaw) * state["vx"] + math.cos(yaw) * state["vy"])
                    msg.twist.twist.angular.z = (
                        state["yaw_rate"] if name == "tag_rover" else 0.0)
                    # Publish the filter's own 2-D block rather than a scalar:
                    # marker error is elongated along the viewing ray and a
                    # consumer that plans around it needs the shape.
                    msg.pose.covariance[0] = covariance[0][0]
                    msg.pose.covariance[1] = covariance[0][1]
                    msg.pose.covariance[6] = covariance[1][0]
                    msg.pose.covariance[7] = covariance[1][1]
                    msg.pose.covariance[35] = (
                        covariance[4][4] if name == "tag_rover" else 1e6)
                    for index in (14, 21, 28):
                        msg.pose.covariance[index] = 1e6
                    msg.twist.covariance[0] = covariance[2][2]
                    msg.twist.covariance[7] = covariance[3][3]
                    msg.twist.covariance[35] = (
                        covariance[5][5] if name == "tag_rover" else 1e6)
                    for index in (14, 21, 28):
                        msg.twist.covariance[index] = 1e6
                    odom_pub.publish(msg)

    def snapshot():
        with lock:
            return {
                "phase": "SIMULATION",
                "hardware_verified": False,
                "opponent_enabled": not a.tag_only,
                "wall_seconds": time.monotonic() - start,
                "calibration_version": version,
                "marker_family": marker_family,
                "marker_ids": sorted(tags),
                "tracks": dict(tracks),
                # The filter reports state=None until a marker creates the
                # track, so the dashboard summary has to survive that rather
                # than take the runtime down on its first status write.
                "track": (
                    {
                        **(tracks["tag_rover"]["state"] or {}),
                        "tracking_state": tracks["tag_rover"]["tracking_state"],
                    }
                    if "tag_rover" in tracks
                    else None
                ),
                "cameras": {k: dict(v) for k, v in metrics.items()},
                "received": dict(capture.received),
                "overwritten": dict(capture.dropped),
                "record_drops": rec_drops,
                "filters": {
                    name: {
                        "accepted": f.accepted,
                        "rejected": f.rejected,
                        "out_of_sequence": f.out_of_sequence,
                        "dropped_too_old": f.too_old,
                        "id_rejections": f.id_rejections,
                        "identity": f.identity,
                        "model_probabilities": [float(v) for v in f.mu],
                    }
                    for name, f in filters.items()
                },
                "buffer_pending": {name: len(b.pending) for name, b in buffers.items()},
                "lidar": ({"scans": lidar.scans, "detections": lidar.detections,
                           "rejections": dict(lidar.rejections),
                           "topic": lidar_topic}
                          if lidar is not None else None),
                "clock": dict(clock),
                "errors": list(errors),
            }

    def guarded(target, args):
        def run():
            try:
                target(*args)
            except Exception as exc:
                with lock:
                    errors.append(str(exc))
                stop.set()

        return run

    jobs = [(tag_worker, (cid,)) for cid in cams] + [
        (publisher_worker, ()),
        (record_worker, ()),
    ]
    if lidar is not None:
        jobs.insert(0, (lidar_worker, ()))
    if not a.tag_only:
        jobs.insert(0, (yolo_worker, ()))
    threads = [threading.Thread(target=guarded(target, args)) for target, args in jobs]
    for thread in threads:
        thread.start()

    def apply_settings(candidate):
        nonlocal cams, pipes, cfg, version, tags, marker_family, tag_size_m
        rollback = candidate.get("action") == "rollback"
        if rollback:
            with lock:
                if not history:
                    raise ValueError("no previous calibration")
                candidate = history[-1]
        if candidate.get("role") != "image_calibrated":
            raise ValueError("image-derived calibration required")
        replacement = {c["name"]: c for c in candidate["cameras"]}
        if set(replacement) != set(cams):
            raise ValueError("camera identities must remain stable")
        candidate = json.loads(json.dumps(candidate, allow_nan=False))
        digest = hashlib.sha256(
            json.dumps(candidate, sort_keys=True).encode()
        ).hexdigest()[:12]
        new_version = candidate["calibration_version"].split("@")[0] + "@" + digest
        candidate["calibration_version"] = new_version
        new_tag_entries = candidate["tags"]
        new_families = {normalize_marker_family(t["family"]) for t in new_tag_entries}
        new_sizes = {float(t["size"]) for t in new_tag_entries}
        if len(new_families) != 1 or len(new_sizes) != 1:
            raise ValueError("one runtime track must use one marker family and one marker size")
        new_family = new_families.pop()
        new_tag_size_m = new_sizes.pop()
        new_tags = {
            int(t["id"]): {
                "rotation": t["R_base_tag"],
                "translation": t["T_base_tag_translation"],
            }
            for t in new_tag_entries
        }
        if not new_tags:
            raise ValueError("at least one configured marker is required")
        for transform in new_tags.values():
            R = np.asarray(transform["rotation"], dtype=float)
            t = np.asarray(transform["translation"], dtype=float)
            if (
                R.shape != (3, 3)
                or t.shape != (3,)
                or not np.isfinite(t).all()
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(R), 1)
            ):
                raise ValueError("invalid tag transform")
        new_pipes = {}
        for cid, c in replacement.items():
            K = np.array(c["K"], dtype=float).reshape(3, 3)
            R = np.array(c["R_world_optical"], dtype=float)
            if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
                raise ValueError("invalid K")
            if not np.isfinite(c["position_world"]).all():
                raise ValueError("invalid position")
            if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(
                np.linalg.det(R), 1
            ):
                raise ValueError("invalid rotation")
            if c["image_size"] != cams[cid]["image_size"]:
                raise ValueError("image size must match live Gazebo profile")
            if not np.isfinite(c["D"]).all() or len(c["D"]) not in (4, 5, 8):
                raise ValueError("invalid pinhole distortion")
            new_pipes[cid] = OneCameraImagePipeline(
                cid,
                K,
                c["D"],
                {"rotation": R, "translation": c["position_world"]},
                new_tags,
                new_version,
                family=new_family,
                tag_size_m=new_tag_size_m,
                detector_scale=a.detector_scale,
                detector_profile=a.detector_profile,
                marker_ids=tuple(new_tags),
                quality_min=a.tag_quality_min,
                max_reprojection_px=a.tag_max_reprojection_px,
                max_planar_tilt_deg=a.tag_max_planar_tilt_deg,
                min_side_px=a.tag_min_side_px,
                image_size=c["image_size"],
                **pose_gates,
            )
        with tempfile.NamedTemporaryFile(mode="w", dir=out, delete=False) as staged:
            staged.write(json.dumps(candidate, indent=2) + "\n")
            temporary = Path(staged.name)
        with lock:
            temporary.replace(out / "active_calibration.json")
            if rollback:
                history.pop()
            else:
                history.append(cfg)
            cams = replacement
            pipes = new_pipes
            cfg = candidate
            version = new_version
            tags = new_tags
            marker_family = new_family
            tag_size_m = new_tag_size_m
            for f in filters.values():
                f.reset()
            tracks.clear()
            for values in accepted_wall.values():
                values.clear()
            for values in published_wall.values():
                values.clear()
        return {"ok": True, "calibration_version": version, "tracking_reset": True}

    DashboardHandler.settings_provider = staticmethod(lambda: cfg)
    DashboardHandler.settings_apply = staticmethod(apply_settings)
    if not a.no_ui:
        DashboardHandler.status_provider = staticmethod(snapshot)
        DashboardHandler.preview_files_provider = staticmethod(
            lambda: {cid: str(out / f"{cid}.jpg") for cid in cams}
        )
        DashboardHandler.cameras_provider = staticmethod(
            lambda: [{"camera_id": cid, "channels": metrics[cid]} for cid in cams]
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def halt(*_):
        stop.set()

    signal.signal(signal.SIGINT, halt)
    signal.signal(signal.SIGTERM, halt)
    print(f"live runtime ready; http://127.0.0.1:{a.port}", flush=True)
    while not stop.wait(1):
        if a.seconds and time.monotonic() - start >= a.seconds:
            stop.set()
        if time.monotonic() - start > 15 and not any(capture.received.values()):
            errors.append("no camera frames received for 15 seconds")
            stop.set()
        (out / "status.json").write_text(json.dumps(snapshot(), indent=2) + "\n")
    capture.close()
    yolo_capture.close()
    for topic in topics:
        node.unsubscribe(topic)
    for thread in threads[:-1]:
        thread.join(timeout=10)
    drain_buffers(clock["sim"], force=True)
    record_done.set()
    threads[-1].join(timeout=10)
    if server:
        server.shutdown()
    if rospy is not None:
        rospy.destroy_node()
        rclpy.shutdown()
    (out / "status.json").write_text(json.dumps(snapshot(), indent=2) + "\n")
    print(json.dumps(snapshot()), flush=True)
    os._exit(1 if errors or any(t.is_alive() for t in threads) else 0)


if __name__ == "__main__":
    main()
