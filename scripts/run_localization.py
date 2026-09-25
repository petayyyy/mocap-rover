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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization_contracts.capture import LatestFrames, rgb_array
from localization_contracts.image_pipeline import OneCameraImagePipeline
from localization_contracts.fusion import PlanarFusion, SynchronousObservationArbiter
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
    p.add_argument("--tag-max-reprojection-px", type=float, default=5.0)
    p.add_argument("--tag-max-planar-tilt-deg", type=float, default=40.0)
    p.add_argument("--tag-min-side-px", type=float, default=8.0)
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
        "synchronous_camera_policy": "best_quality_then_reprojection_then_size",
        "covariance_model": "coverage_20260925_empirical_v1",
    }, indent=2) + "\n")
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
        )
        for cid, c in cams.items()
    }
    capture = LatestFrames(cams)
    yolo_capture = LatestFrames(cams)
    node = Node()
    stop = threading.Event()
    lock = threading.RLock()
    filters = {
        "tag_rover": PlanarFusion(),
        "opponent": PlanarFusion("opponent", 15, 300),
    }
    if a.tag_only:
        del filters["opponent"]
    tag_arbiter = SynchronousObservationArbiter(cams)
    metrics = {
        cid: {
            "processed": 0,
            "tag_hits": 0,
            "tag_accepted": 0,
            "yolo_hits": 0,
            "last_capture_ns": 0,
            "latency_ms": 0,
        }
        for cid in cams
    }
    clock = {"sim": 0, "wall": time.monotonic_ns(), "resets": 0}
    tracks = {}
    previews = {}
    errors = []
    start = time.monotonic()
    topics = []
    threads = []
    observations = collections.deque(maxlen=4096)
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

    def accept(obs):
        with lock:
            f = filters[obs.object_id]
            if (
                obs.calibration_version != version
                or obs.capture_time_ns > clock["sim"] + 100_000_000
            ):
                record_observation(obs, False, "calibration_or_future_stamp")
                return False
            ok = f.update(obs)
            if ok:
                accepted_wall[obs.object_id].append(time.monotonic_ns())
            record_observation(obs, ok, "fusion_accepted" if ok else "fusion_gate")
            return ok

    def mark_frame_accepted(obs):
        with lock:
            metrics[obs.camera_id]["tag_accepted"] += 1
            for row in reversed(camera_frames):
                if row["camera_id"] == obs.camera_id and row["capture_ns"] == obs.capture_time_ns:
                    row["fusion_accepted"] += 1
                    break

    def consume_tag_batches(batches):
        for _, selected, discarded in batches:
            for obs in discarded:
                record_observation(obs, False, "lower_quality_synchronous_view")
            if selected is not None and accept(selected):
                mark_frame_accepted(selected)

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
                hits = pipe.detector.detect(image)
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
                        # Empirical model from the coverage CSV. The old 3--6 mm
                        # covariance was overconfident versus 8--17 cm observed
                        # errors and made the innovation gate reject good handoffs.
                        view_factor = 1.6 if cid in ("camera_5", "camera_6") else 1.0
                        sigma = max(0.025, 0.018 * view_factor / max(obs.quality, 0.12))
                        yaw_sigma = math.radians(
                            min(20.0, max(2.0, 4.0 / max(obs.quality, 0.20)))
                        )
                        obs = dataclasses.replace(
                            obs,
                            covariance_m2=(
                                sigma * sigma,
                                sigma * sigma,
                                4 * sigma * sigma,
                            ),
                            pixel_features={
                                **obs.pixel_features,
                                "yaw_sigma_rad": yaw_sigma,
                                "xy_sigma_m": sigma,
                            },
                        )
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
                            "pnp_rejections": dict(pnp_rejections),
                            "pnp_diagnostics": pnp_diagnostics,
                        }
                    )
                with lock:
                    batches = tag_arbiter.report(cid, stamp, frame_observations)
                consume_tag_batches(batches)
        except Exception as exc:
            errors.append(f"{cid}: {exc}")
            stop.set()

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
                    accept(selected[stamp])
        except Exception as exc:
            errors.append("yolo: " + str(exc))
            stop.set()

    def record_worker():
        with (out / "observations.jsonl").open("w") as fo, (
            out / "odometry.jsonl"
        ).open("w") as fp, (out / "camera_frames.jsonl").open("w") as fc:
            while (
                not record_done.is_set()
                or observations
                or publication
                or camera_frames
            ):
                with lock:
                    ob = list(observations)
                    observations.clear()
                    pub = list(publication)
                    publication.clear()
                    frames = list(camera_frames)
                    camera_frames.clear()
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
        nonlocal rec_drops
        next_tick = time.monotonic()
        tick = 0
        while not stop.is_set():
            now = time.monotonic()
            stop.wait(max(0, next_tick - now))
            next_tick += 1 / 30
            if stop.is_set():
                break
            if time.monotonic() - next_tick > 0.1:
                next_tick = time.monotonic() + 1 / 30
            tick += 1
            with lock:
                stamp = clock["sim"]
                for name, f in filters.items():
                    if name == "opponent" and tick % 2:
                        continue
                    item = f.publish(stamp)
                    if item is None:
                        continue
                    row = {
                        **item,
                        "state": dataclasses.asdict(item["state"]),
                        "object_id": name,
                        "capture_ns": f.last_measurement.capture_time_ns,
                        "wall_ns": time.monotonic_ns(),
                        "yaw_valid": name == "tag_rover",
                    }
                    now_ns = time.monotonic_ns()
                    published_wall[name].append(now_ns)
                    row.update(
                        measurement_sim_hz=row["measurement_hz"],
                        output_sim_hz=row["output_hz"],
                        measurement_wall_hz=rate(accepted_wall[name]),
                        output_wall_hz=rate(published_wall[name]),
                        session=clock["resets"],
                    )
                    if now_ns - clock["wall"] > 200_000_000:
                        row.update(valid=False, tracking_state="TIME_UNCERTAIN")
                    tracks[name] = row
                    if len(publication) == publication.maxlen:
                        rec_drops += 1
                    publication.append(row)
                    if rospy is not None:
                        odom_pub, status_pub = publishers[name]
                        status = String()
                        status.data = json.dumps(
                            {k: v for k, v in row.items() if k != "state"}
                        )
                        status_pub.publish(status)
                        if not row["valid"]:
                            continue
                        s = item["state"]
                        msg = Odometry()
                        msg.header.frame_id = "arena"
                        msg.child_frame_id = name + "/base_link"
                        msg.header.stamp.sec = stamp // 1_000_000_000
                        msg.header.stamp.nanosec = stamp % 1_000_000_000
                        msg.pose.pose.position.x = s.x
                        msg.pose.pose.position.y = s.y
                        msg.pose.pose.position.z = float(
                            f.last_measurement.position_m[2]
                        )
                        msg.pose.pose.orientation.z = (
                            math.sin(s.yaw / 2) if name == "tag_rover" else 0.0
                        )
                        msg.pose.pose.orientation.w = (
                            math.cos(s.yaw / 2) if name == "tag_rover" else 1.0
                        )
                        msg.twist.twist.linear.x = (
                            math.cos(s.yaw) * s.vx + math.sin(s.yaw) * s.vy
                            if name == "tag_rover"
                            else s.vx
                        )
                        msg.twist.twist.linear.y = (
                            -math.sin(s.yaw) * s.vx + math.cos(s.yaw) * s.vy
                            if name == "tag_rover"
                            else s.vy
                        )
                        msg.twist.twist.angular.z = (
                            s.omega if name == "tag_rover" else 0.0
                        )
                        msg.pose.covariance[0] = s.covariance
                        msg.pose.covariance[7] = s.covariance
                        msg.pose.covariance[35] = 0.002 if name == "tag_rover" else 1e6
                        for index in (14, 21, 28):
                            msg.pose.covariance[index] = 1e6
                        for index in (0, 7, 14, 21, 28, 35):
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
                "track": (
                    {
                        **tracks["tag_rover"]["state"],
                        "tracking_state": tracks["tag_rover"]["tracking_state"],
                    }
                    if "tag_rover" in tracks
                    else None
                ),
                "cameras": {k: dict(v) for k, v in metrics.items()},
                "received": dict(capture.received),
                "overwritten": dict(capture.dropped),
                "record_drops": rec_drops,
                "tag_arbiter_late_reports": tag_arbiter.late_reports,
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
            tag_arbiter.reset()
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
    with lock:
        remaining_tag_batches = tag_arbiter.flush()
    consume_tag_batches(remaining_tag_batches)
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
