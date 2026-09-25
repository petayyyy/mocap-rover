#!/usr/bin/env python3
"""Evaluator/test-driver only; never imported by image localization runtime."""
import argparse, json, math, os, signal, threading, time
from pathlib import Path
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist


def pose_stamp_ns(pose):
    if not pose.HasField("header") or not pose.header.HasField("stamp"):
        raise ValueError("Gazebo Pose has no capture timestamp")
    return pose.header.stamp.sec * 1_000_000_000 + pose.header.stamp.nsec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--drive", action="store_true")
    p.add_argument("--tag-only", action="store_true")
    p.add_argument("--speed", type=float, default=0.7)
    a = p.parse_args()
    if not 0 < a.speed <= 2:
        p.error("--speed must be in (0, 2] m/s")
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    node = Node()
    stop = threading.Event()
    lock = threading.Lock()
    latest = {}
    topics = []
    pubs = {}
    rows = []

    def callback(msg, name):
        for q in msg.pose:
            if q.name == name:
                # PosePublisher stamps each nested Pose, not the Pose_V envelope.
                stamp = pose_stamp_ns(q)
                quat = q.orientation
                yaw = math.atan2(
                    2 * (quat.w * quat.z + quat.x * quat.y),
                    1 - 2 * (quat.y**2 + quat.z**2),
                )
                row = {
                    "object_id": "opponent" if name == "yolo_rover" else name,
                    "stamp_ns": stamp,
                    "x": q.position.x,
                    "y": q.position.y,
                    "z": q.position.z,
                    "yaw": yaw,
                    "wall_ns": time.monotonic_ns(),
                }
                with lock:
                    latest[name] = row
                    rows.append(row)

    for name in (("tag_rover",) if a.tag_only else ("tag_rover", "yolo_rover")):
        topic = f"/model/{name}/pose"
        topics.append(topic)
        node.subscribe(Pose_V, topic, lambda msg, name=name: callback(msg, name))
        pubs[name] = node.advertise(f"/model/{name}/cmd_vel", Twist)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    waypoints = {
        "tag_rover": [(2, 2), (10, 2), (10, 6), (2, 6), (2, 10), (10, 10), (6, 6)],
        "yolo_rover": [(9, 9), (3, 9), (3, 3), (9, 3), (6, 8)],
    }
    indices = dict.fromkeys(pubs, 0)
    start = time.monotonic()
    with out.open("w") as f:
        while not stop.wait(0.05) and time.monotonic() - start < a.seconds:
            with lock:
                batch = list(rows)
                rows.clear()
                poses = dict(latest)
            for row in batch:
                f.write(json.dumps(row) + "\n")
            f.flush()
            if a.drive:
                for name, pose in poses.items():
                    x, y = waypoints[name][indices[name]]
                    dx = x - pose["x"]
                    dy = y - pose["y"]
                    error = (math.atan2(dy, dx) - pose["yaw"] + math.pi) % (
                        2 * math.pi
                    ) - math.pi
                    if math.hypot(dx, dy) < 0.25:
                        indices[name] = (indices[name] + 1) % len(waypoints[name])
                    cmd = Twist()
                    cmd.linear.x = min(a.speed, math.hypot(dx, dy)) * max(
                        0.0, math.cos(error)
                    )
                    cmd.angular.z = max(-1.5, min(1.5, error * 2))
                    pubs[name].publish(cmd)
    for pub in pubs.values():
        pub.publish(Twist())
    for topic in topics:
        node.unsubscribe(topic)
    os._exit(0)


if __name__ == "__main__":
    main()
