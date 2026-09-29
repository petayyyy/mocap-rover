#!/usr/bin/env python3
"""Evaluator/test-driver only; never imported by the localization runtime.

Drives both rovers along the longest lines the arena allows so a recording
actually contains 11 m/s.  ``drive_random.py`` picks waypoints a few metres
apart and eases off near each one, which measured out at 5.7 m/s peak on a
40 s run; a rover only reaches its cap when it has ten metres of straight
ahead of it.

Each rover cycles through a fixed list of far-apart targets: full throttle
while the target is ahead, a braking ramp over the last ``--brake`` metres,
then a pivot toward the next one.  The two lists are offset so the rovers
cross each other's paths, which is the case worth testing for identity.
"""
import argparse
import json
import math
import os
import signal
import sys
import threading
import time

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from localization_contracts.gazebo_services import ServiceClient  # noqa: E402

ROUTES = {
    "tag_rover": [(2.2, 2.2), (9.8, 9.8), (2.2, 9.8), (9.8, 2.2), (2.2, 6.0), (9.8, 6.0)],
    "yolo_rover": [(9.8, 2.2), (2.2, 9.8), (9.8, 9.8), (2.2, 2.2), (6.0, 9.8), (6.0, 2.2)],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=900.0, help="wall seconds")
    p.add_argument("--speed", type=float, default=11.0)
    p.add_argument("--brake", type=float, default=7.0,
                   help="start slowing this far from the target")
    p.add_argument("--reach", type=float, default=1.0)
    p.add_argument("--stuck-s", type=float, default=2.0,
                   help="wall seconds without motion, or upside down, before "
                        "the rover is set back on its wheels at its own XY")
    p.add_argument("--only", choices=tuple(ROUTES), default=None)
    a = p.parse_args()

    names = (a.only,) if a.only else tuple(ROUTES)
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
            # Body Z axis in world: negative means the rover is on its back.
            up = 1 - 2 * (quat.x ** 2 + quat.y ** 2)
            with lock:
                poses[name] = (
                    q.position.x, q.position.y,
                    math.atan2(2 * (quat.w * quat.z + quat.x * quat.y),
                               1 - 2 * (quat.y ** 2 + quat.z ** 2)),
                    q.position.z, up,
                )

    for name in names:
        node.subscribe(Pose_V, f"/model/{name}/pose",
                       lambda msg, name=name: callback(msg, name))
        pubs[name] = node.advertise(f"/model/{name}/cmd_vel", Twist)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    index = dict.fromkeys(names, 0)
    legs = dict.fromkeys(names, 0)
    resets = dict.fromkeys(names, 0)
    last_move = {name: (None, time.monotonic()) for name in names}
    services = ServiceClient()

    def upright(name, x, y, yaw):
        command = Pose_V()
        q = command.pose.add(name=name)
        q.position.x, q.position.y, q.position.z = x, y, 0.3
        q.orientation.z, q.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        services.request("set_pose_vector", command)
        resets[name] += 1

    start = time.monotonic()
    while not stop.wait(0.02) and time.monotonic() - start < a.seconds:
        with lock:
            snapshot = dict(poses)
        for name, (x, y, yaw, z, up) in snapshot.items():
            previous, since = last_move[name]
            if previous is None or math.hypot(x - previous[0], y - previous[1]) > 0.05:
                last_move[name] = ((x, y), time.monotonic())
            elif up < 0.3 or time.monotonic() - since > a.stuck_s:
                # Flipped, wedged against a barrier or piled on the other
                # rover: put it back on its wheels where it is and move on.
                x = min(max(x, 1.0), 11.0)
                y = min(max(y, 1.0), 11.0)
                upright(name, x, y, yaw)
                last_move[name] = (None, time.monotonic())
                index[name] = (index[name] + 1) % len(ROUTES[name])
                continue
            tx, ty = ROUTES[name][index[name]]
            dx, dy = tx - x, ty - y
            distance = math.hypot(dx, dy)
            if distance < a.reach:
                index[name] = (index[name] + 1) % len(ROUTES[name])
                legs[name] += 1
                continue
            error = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
            cmd = Twist()
            # No pivot dead-band: a four-wheel skid-steer model turns slowly,
            # and alternating "pivot" with "drive" around a threshold measured
            # out at 1 m/s.  Throttle fades with the heading error instead.
            ramp = min(1.0, distance / a.brake)
            cmd.linear.x = a.speed * ramp * max(0.0, math.cos(error)) ** 2
            # Less steering authority the faster it goes: a skid-steer box
            # that yaws hard at 8 m/s rolls onto its side.
            turn_cap = 3.0 if cmd.linear.x < 4.0 else 1.2
            cmd.angular.z = max(-turn_cap, min(turn_cap, 1.5 * error))
            pubs[name].publish(cmd)
    for pub in pubs.values():
        pub.publish(Twist())
    services.close()
    print(json.dumps({"legs": legs, "resets": resets}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
