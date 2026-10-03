#!/usr/bin/env python3
"""Evaluator/test-driver only; never imported by the localization runtime.

Drives the two rovers into each other on purpose, for the contact dataset of
prompt 08 (``dataset_mounts5b_contact_01``).

* ``tag_rover`` drives to random waypoints, like ``drive_random.py``.
* ``yolo_rover`` hunts it: CHASE (aims at a contact point on the tag_rover:
  its side, its nose or its centre), PUSH for 1-3 s once the hulls touch
  (centre distance below ``--contact-m``), then BACKOFF (reverse) and
  ESCAPE (drive to a random point away from the tag_rover) before the next
  chase.  For a head-on encounter the tag_rover turns to face the hunter
  during the last metres.

All durations are in simulated time, read from the pose stamps, so the
script behaves the same at any real-time factor.  Speeds stay at or below
``--speed`` (3 m/s by default) and contacts are made slower
(``--contact-speed``) so the hulls do not climb onto each other.  A rover
that has flipped (roll/pitch over 45 deg) or a clinch (no progress for
``--stuck-s``) ends the encounter with a backoff.

The seed is printed and accepted; the events (contact start/end, kind) go
to stdout as JSON lines so a run can be matched to its truth.
"""
import argparse
import json
import math
import os
import random
import signal
import threading
import time

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node

TAG, HUNTER = "tag_rover", "yolo_rover"
KINDS = ("side", "head_on", "push")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def steer(pose, target, speed, reverse=False):
    """Twist that drives a diff-drive rover at ``pose`` to ``target``."""
    x, y, yaw = pose[:3]
    dx, dy = target[0] - x, target[1] - y
    distance = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)
    if reverse:
        heading = wrap(heading + math.pi)
    error = wrap(heading - yaw)
    cmd = Twist()
    v = min(speed, 1.5 * distance + 0.3) * max(0.0, math.cos(error)) ** 2
    cmd.linear.x = -v if reverse else v
    cmd.angular.z = max(-3.0, min(3.0, 2.5 * error))
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=1200.0,
                   help="wall-clock limit of the run")
    p.add_argument("--speed", type=float, default=3.0, help="m/s cap")
    p.add_argument("--contact-speed", type=float, default=2.2,
                   help="m/s cap for the last metre and the push")
    p.add_argument("--contact-m", type=float, default=0.95,
                   help="centre distance that counts as touching")
    p.add_argument("--push-min-s", type=float, default=1.0)
    p.add_argument("--push-max-s", type=float, default=2.0)
    p.add_argument("--backoff-s", type=float, default=0.5)
    p.add_argument("--escape-s", type=float, default=0.8)
    p.add_argument("--chase-max-s", type=float, default=5.0)
    p.add_argument("--stuck-s", type=float, default=2.5)
    p.add_argument("--tag-speed-fraction", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--margin", type=float, default=1.8)
    p.add_argument("--arena", type=float, default=12.0)
    a = p.parse_args()
    seed = random.randrange(1 << 30) if a.seed is None else a.seed
    rng = random.Random(seed)
    print(json.dumps({"drive_encounters_seed": seed, "speed": a.speed}), flush=True)

    node = Node()
    stop = threading.Event()
    lock = threading.Lock()
    poses = {}
    sim = {"ns": None}

    def callback(msg, name):
        for q in msg.pose:
            if q.name != name:
                continue
            st = q.header.stamp if q.HasField("header") else msg.header.stamp
            stamp = st.sec * 1_000_000_000 + st.nsec
            o = q.orientation
            roll = math.atan2(2 * (o.w * o.x + o.y * o.z), 1 - 2 * (o.x ** 2 + o.y ** 2))
            pitch = math.asin(max(-1.0, min(1.0, 2 * (o.w * o.y - o.z * o.x))))
            yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y ** 2 + o.z ** 2))
            with lock:
                poses[name] = (q.position.x, q.position.y, yaw, roll, pitch)
                if stamp > 0:
                    sim["ns"] = stamp

    pubs = {}
    for name in (TAG, HUNTER):
        node.subscribe(Pose_V, f"/model/{name}/pose",
                       lambda msg, name=name: callback(msg, name))
        pubs[name] = node.advertise(f"/model/{name}/cmd_vel", Twist)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    lo, hi = a.margin, a.arena - a.margin

    def waypoint():
        return rng.uniform(lo, hi), rng.uniform(lo, hi)

    def clamp(pt):
        return min(hi, max(lo, pt[0])), min(hi, max(lo, pt[1]))

    tag_target = waypoint()
    state = {"mode": "escape", "since": None, "kind": "push",
             "goal": waypoint(), "push_s": 2.0, "progress": None}
    contacts = 0
    events = {k: 0 for k in KINDS}

    def enter(mode, now, **extra):
        state.update(mode=mode, since=now, progress=None, **extra)

    start = time.monotonic()
    last_status = [0]
    while not stop.wait(0.02) and time.monotonic() - start < a.seconds:
        with lock:
            snap = dict(poses)
            now = sim["ns"]
        if TAG not in snap or HUNTER not in snap or now is None:
            continue
        if state["since"] is None:
            state["since"] = now
        t_pose, h_pose = snap[TAG], snap[HUNTER]
        elapsed = (now - state["since"]) / 1e9
        dist = math.hypot(t_pose[0] - h_pose[0], t_pose[1] - h_pose[1])
        flipped = any(abs(v) > math.radians(45) for v in (*t_pose[3:], *h_pose[3:]))

        # tag_rover: random waypoints; faces the hunter for a head-on.
        if math.hypot(tag_target[0] - t_pose[0], tag_target[1] - t_pose[1]) < 0.5:
            tag_target = waypoint()
        # The target is slower than the hunter, so every chase ends.
        tag_cmd = steer(t_pose, tag_target, a.tag_speed_fraction * a.speed)
        if state["mode"] == "chase" and state["kind"] == "head_on" and dist < 3.0:
            tag_cmd = steer(t_pose, h_pose, a.contact_speed)
        if state["mode"] == "push":
            tag_cmd = steer(t_pose, tag_target, 0.5 * a.contact_speed)

        mode = state["mode"]
        if mode == "chase":
            # Contact point on the tag_rover hull, by the encounter kind.
            tx, ty, tyaw = t_pose[:3]
            if state["kind"] == "side":
                side = state.get("side", 1.0)
                aim = (tx - side * 0.3 * math.sin(tyaw), ty + side * 0.3 * math.cos(tyaw))
            elif state["kind"] == "head_on":
                aim = (tx + 0.35 * math.cos(tyaw), ty + 0.35 * math.sin(tyaw))
            else:
                aim = (tx, ty)
            speed = a.speed if dist > 1.0 else a.contact_speed
            hunter_cmd = steer(h_pose, aim, speed)
            if dist < a.contact_m:
                contacts += 1
                events[state["kind"]] += 1
                print(json.dumps({"event": "contact", "n": contacts, "kind": state["kind"],
                                  "sim_s": now / 1e9, "dist_m": round(dist, 3)}), flush=True)
                enter("push", now, push_s=rng.uniform(a.push_min_s, a.push_max_s))
            elif elapsed > a.chase_max_s or flipped:
                print(json.dumps({"event": "chase_timeout", "sim_s": now / 1e9,
                                  "kind": state["kind"], "dist_m": round(dist, 3)}), flush=True)
                enter("backoff", now)
        elif mode == "push":
            hunter_cmd = steer(h_pose, t_pose, a.contact_speed)
            if elapsed > state["push_s"] or flipped:
                print(json.dumps({"event": "release", "n": contacts, "sim_s": now / 1e9,
                                  "push_s": round(elapsed, 2), "flipped": flipped}), flush=True)
                enter("backoff", now)
        elif mode == "backoff":
            hunter_cmd = steer(h_pose, t_pose, a.contact_speed, reverse=True)
            if elapsed > a.backoff_s:
                away = math.atan2(h_pose[1] - t_pose[1], h_pose[0] - t_pose[0])
                away += rng.uniform(-0.8, 0.8)
                goal = clamp((h_pose[0] + 2.5 * math.cos(away), h_pose[1] + 2.5 * math.sin(away)))
                enter("escape", now, goal=goal)
        else:  # escape
            hunter_cmd = steer(h_pose, state["goal"], a.speed)
            if elapsed > a.escape_s:
                enter("chase", now, kind=rng.choice(KINDS), side=rng.choice((-1.0, 1.0)))

        # Clinch guard: the hunter has not moved 0.15 m for --stuck-s.
        if state["mode"] in ("chase", "escape"):
            pr = state["progress"]
            if pr is None or math.hypot(h_pose[0] - pr[1], h_pose[1] - pr[2]) > 0.15:
                state["progress"] = (now, h_pose[0], h_pose[1])
            elif (now - pr[0]) / 1e9 > a.stuck_s:
                print(json.dumps({"event": "stuck", "sim_s": now / 1e9, "mode": state["mode"],
                                  "dist_m": round(dist, 2),
                                  "hunter": [round(v, 2) for v in h_pose[:3]]}), flush=True)
                enter("backoff", now)
        if a.verbose and now - last_status[0] > 2_000_000_000:
            last_status[0] = now
            print(json.dumps({"status": state["mode"], "sim_s": now / 1e9, "dist_m": round(dist, 2),
                              "tag": [round(v, 2) for v in t_pose],
                              "hunter": [round(v, 2) for v in h_pose]}), flush=True)
        pubs[TAG].publish(tag_cmd)
        pubs[HUNTER].publish(hunter_cmd)
    for pub in pubs.values():
        pub.publish(Twist())
    print(json.dumps({"seed": seed, "contacts": contacts, "kinds": events}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
