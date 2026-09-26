#!/usr/bin/env python3
"""Evaluator/test-driver only; never imported by the localization runtime.

Drives both rovers to random waypoints so a run covers the arena instead of
one rehearsed loop.  ``record_gazebo_truth.py`` walks a fixed polygon, which
is reproducible but visits the same handful of headings and never puts the
two rovers near each other -- exactly the cases worth exercising.

The seed is printed and accepted, so a run that shows something can be
replayed.
"""
import argparse
import json
import math
import os
import random
import signal
import sys
import threading
import time

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node

ROVERS = ("tag_rover", "yolo_rover")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--speed", type=float, default=5.0,
                   help="upper bound on commanded linear speed, m/s")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--margin", type=float, default=1.6,
                   help="keep waypoints this far from the arena walls")
    p.add_argument("--arena", type=float, default=12.0)
    p.add_argument("--only", choices=ROVERS, default=None)
    a = p.parse_args()
    seed = random.randrange(1 << 30) if a.seed is None else a.seed
    rng = random.Random(seed)
    print(f"drive_random seed {seed}, {a.speed} m/s cap", flush=True)

    names = (a.only,) if a.only else ROVERS
    node = Node()
    stop = threading.Event()
    lock = threading.Lock()
    poses = {}
    pubs = {}

    def callback(msg, name):
        for q in msg.pose:
            if q.name != name:
                continue
            quat = q.orientation
            with lock:
                poses[name] = (
                    q.position.x, q.position.y,
                    math.atan2(2 * (quat.w * quat.z + quat.x * quat.y),
                               1 - 2 * (quat.y ** 2 + quat.z ** 2)),
                )

    for name in names:
        node.subscribe(Pose_V, f"/model/{name}/pose",
                       lambda msg, name=name: callback(msg, name))
        pubs[name] = node.advertise(f"/model/{name}/cmd_vel", Twist)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def waypoint():
        lo, hi = a.margin, a.arena - a.margin
        return rng.uniform(lo, hi), rng.uniform(lo, hi)

    targets = {name: waypoint() for name in names}
    reached = dict.fromkeys(names, 0)
    start = time.monotonic()
    while not stop.wait(0.05) and time.monotonic() - start < a.seconds:
        with lock:
            snapshot = dict(poses)
        for name, (x, y, yaw) in snapshot.items():
            tx, ty = targets[name]
            dx, dy = tx - x, ty - y
            distance = math.hypot(dx, dy)
            if distance < 0.5:
                targets[name] = waypoint()
                reached[name] += 1
                continue
            error = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
            cmd = Twist()
            # Ease off well before the waypoint: at 5 m/s a diff-drive rover
            # that aims at full throttle simply orbits the point it wants.
            cmd.linear.x = (min(a.speed, 1.2 * distance)
                            * max(0.0, math.cos(error)) ** 2)
            cmd.angular.z = max(-3.0, min(3.0, 2.5 * error))
            pubs[name].publish(cmd)
    for pub in pubs.values():
        pub.publish(Twist())
    print(json.dumps({"seed": seed, "waypoints_reached": reached}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
