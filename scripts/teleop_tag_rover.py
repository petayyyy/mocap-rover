#!/usr/bin/env python3
"""WASD driving console for the Gazebo rover, with a live odometry lag panel.

The point is not to drive; it is to watch whether the localization keeps up.
At 11.11 m/s the rover covers 11 mm per millisecond, so the console shows,
side by side, where the simulator says the rover is and where the runtime
thinks it is, and how far apart that is in metres.

Ground truth here comes straight from Gazebo and is a driving aid only. The
localization runtime never sees it: the estimate is read back over the
dashboard's HTTP API, the same JSON a browser gets.

The stock world caps the rover at 1 m/s. Run scripts/set_world_rates.py first
or nothing above walking pace is reachable.
"""
from __future__ import annotations

import argparse
import curses
import json
import math
import threading
import time
import urllib.error
import urllib.request

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node

HELP = [
    "W / S   throttle forward / back      A / D   steer left / right",
    "SPACE   brake            X   stop instantly          Q   quit",
    "1..9    top speed 1..9 m/s           0   12 m/s (43 km/h)",
    "R       reset rover to home, upright   T   upright in place",
    "C       clear trip peaks",
]


class Truth:
    """Simulator pose, for the driving display only."""

    def __init__(self):
        self.lock = threading.Lock()
        self.x = self.y = self.yaw = 0.0
        self.vx = self.vy = 0.0
        self.z = 0.14
        self.up = 1.0
        self.stamp_ns = 0
        self.updates = 0

    def on_pose(self, message, model):
        for pose in message.pose:
            if pose.name != model:
                continue
            stamp = pose.header.stamp.sec * 1_000_000_000 + pose.header.stamp.nsec
            q = pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
            # Third column of the rotation: where the rover's own up axis
            # points. Negative means it is on its back, which is when the
            # underside marker is the one facing the ceiling.
            up = 1 - 2 * (q.x * q.x + q.y * q.y)
            with self.lock:
                dt = (stamp - self.stamp_ns) / 1e9
                if 0 < dt < 0.5:
                    self.vx = (pose.position.x - self.x) / dt
                    self.vy = (pose.position.y - self.y) / dt
                self.x, self.y, self.yaw = pose.position.x, pose.position.y, yaw
                self.z, self.up = pose.position.z, up
                self.stamp_ns = stamp
                self.updates += 1
            return

    def read(self):
        with self.lock:
            return (self.x, self.y, self.yaw, self.vx, self.vy,
                    self.stamp_ns, self.updates, self.z, self.up)


class Estimate:
    """What the runtime publishes, polled from its dashboard."""

    def __init__(self, url, period=0.1):
        self.url = url
        self.period = period
        self.lock = threading.Lock()
        self.row = None
        self.error = "не подключено"
        self.stop = threading.Event()

    def poll(self):
        while not self.stop.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=0.5) as response:
                    status = json.loads(response.read())
                track = (status.get("tracks") or {}).get("tag_rover")
                with self.lock:
                    self.row, self.error = track, None if track else "трек не создан"
            except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
                with self.lock:
                    self.row, self.error = None, f"{type(exc).__name__}"
            self.stop.wait(self.period)

    def read(self):
        with self.lock:
            return self.row, self.error


class Placer:
    """Teleport the rover upright, for when it ends up on its back.

    A reset is a teleport, so the filter's physical-speed gate refuses it and
    the track drops to LOST until a marker re-establishes it. That is the
    designed behaviour and worth watching rather than hiding.
    """

    def __init__(self, node, world, model):
        self.node = node
        self.service = f"/world/{world}/set_pose"
        self.model = model
        self.last = ""

    def place(self, x, y, yaw, z):
        request = Pose()
        request.name = self.model
        request.position.x, request.position.y, request.position.z = x, y, z
        request.orientation.z = math.sin(yaw / 2)
        request.orientation.w = math.cos(yaw / 2)
        try:
            ok, reply = self.node.request(self.service, request, Pose, Boolean, 2000)
        except Exception as exc:                      # transport is best effort
            self.last = f"сброс не прошёл: {type(exc).__name__}"
            return False
        ok = bool(ok) and bool(getattr(reply, "data", False))
        self.last = (f"сброс в ({x:.2f}, {y:.2f}) курс {math.degrees(yaw):.0f}°"
                     if ok else f"сброс отклонён сервисом {self.service}")
        return ok


def bar(value, limit, width=28):
    filled = int(round(width * min(abs(value) / max(limit, 1e-6), 1.0)))
    return ("#" * filled).ljust(width)


def run(screen, args):
    screen.nodelay(True)
    screen.keypad(True)
    curses.curs_set(0)
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        for index, colour in enumerate(
                (curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED,
                 curses.COLOR_CYAN), start=1):
            curses.init_pair(index, colour, -1)

    truth = Truth()
    node = Node()
    publisher = node.advertise(args.topic, Twist)
    model = args.topic.split("/")[2]
    node.subscribe(Pose_V, f"/model/{model}/pose",
                   lambda m: truth.on_pose(m, model))
    estimate = Estimate(args.status_url)
    threading.Thread(target=estimate.poll, daemon=True).start()
    placer = Placer(node, args.world, model)
    resets = 0

    top_speed = args.linear
    throttle = steering = 0.0
    last_key = time.monotonic()
    peak_speed = peak_lag = peak_age = 0.0
    period = 1.0 / args.rate

    try:
        while True:
            now = time.monotonic()
            key = screen.getch()
            while key != -1:                     # drain the key-repeat burst
                if key in (ord("q"), ord("Q")):
                    return
                if key in (ord("w"), ord("W")):
                    throttle, last_key = 1.0, now
                elif key in (ord("s"), ord("S")):
                    throttle, last_key = -1.0, now
                elif key in (ord("a"), ord("A")):
                    steering, last_key = 1.0, now
                elif key in (ord("d"), ord("D")):
                    steering, last_key = -1.0, now
                elif key == ord(" "):
                    throttle *= 0.4
                    steering = 0.0
                    last_key = now
                elif key in (ord("x"), ord("X")):
                    throttle = steering = 0.0
                    last_key = now
                elif key in (ord("c"), ord("C")):
                    peak_speed = peak_lag = peak_age = 0.0
                elif key in (ord("r"), ord("R"), ord("t"), ord("T")):
                    throttle = steering = 0.0
                    publisher.publish(Twist())
                    here = truth.read()
                    if key in (ord("t"), ord("T")):
                        target = (here[0], here[1], here[2])
                    else:
                        target = (args.home[0], args.home[1],
                                  math.radians(args.home[2]))
                    if placer.place(target[0], target[1], target[2], args.reset_z):
                        resets += 1
                    last_key = now
                elif ord("0") <= key <= ord("9"):
                    top_speed = 12.0 if key == ord("0") else float(key - ord("0"))
                key = screen.getch()

            # Release is invisible over a terminal, so a key that stops
            # repeating decays instead of latching the rover at full throttle.
            if now - last_key > args.release:
                decay = math.exp(-(now - last_key - args.release) / args.coast)
                throttle *= decay
                steering *= decay
                if abs(throttle) < 0.02:
                    throttle = 0.0
                if abs(steering) < 0.02:
                    steering = 0.0

            linear = throttle * top_speed
            angular = steering * args.angular
            command = Twist()
            command.linear.x = linear
            command.angular.z = angular
            publisher.publish(command)

            tx, ty, tyaw, tvx, tvy, tstamp, updates, tz, up = truth.read()
            row, error = estimate.read()
            speed = math.hypot(tvx, tvy)
            peak_speed = max(peak_speed, speed)

            state = row.get("state") if row else None
            lag = float("nan")
            if state and tstamp:
                lag = math.hypot(state["x"] - tx, state["y"] - ty)
                peak_lag = max(peak_lag, lag)
            age = (row or {}).get("measurement_age_ms")
            if age is not None:
                peak_age = max(peak_age, age)

            screen.erase()
            line = 0

            def put(text, pair=0):
                nonlocal line
                try:
                    screen.addstr(line, 0, text[:screen.getmaxyx()[1] - 1],
                                  curses.color_pair(pair))
                except curses.error:
                    pass
                line += 1

            put(f"  {model}  —  максимум {top_speed:.0f} м/с "
                f"({top_speed*3.6:.0f} км/ч)", 4)
            put("")
            put(f"  тяга    [{bar(throttle, 1.0)}] {linear:+6.2f} м/с")
            put(f"  поворот [{bar(steering, 1.0)}] {angular:+6.2f} рад/с")
            put("")
            flipped = up < 0.0
            attitude = ("ПЕРЕВЁРНУТ — виден нижний маркер" if flipped
                        else "на колёсах" if up > 0.7 else "на боку")
            put(f"  СИМУЛЯТОР   x={tx:7.3f}  y={ty:7.3f}  yaw={math.degrees(tyaw):+7.1f}°"
                f"   {speed:5.2f} м/с = {speed*3.6:5.1f} км/ч", 4)
            put(f"  положение   z={tz:6.3f} м   {attitude}"
                f"   сбросов {resets}", 3 if flipped else 0)
            if state:
                estimate_speed = math.hypot(state["vx"], state["vy"])
                tracking = row.get("tracking_state", "?")
                pair = 1 if tracking in ("TRACKING", "COASTING") else 3
                put(f"  ЛОКАЛИЗАЦИЯ x={state['x']:7.3f}  y={state['y']:7.3f}  "
                    f"yaw={math.degrees(state['yaw']):+7.1f}°"
                    f"   {estimate_speed:5.2f} м/с = {estimate_speed*3.6:5.1f} км/ч", pair)
                put("")
                lag_pair = 1 if lag < 0.10 else 2 if lag < 0.30 else 3
                put(f"  ОТСТАВАНИЕ  {lag*1000:7.0f} мм"
                    f"        пик {peak_lag*1000:.0f} мм", lag_pair)
                age_pair = 1 if (age or 0) < 60 else 2 if (age or 0) < 150 else 3
                put(f"  возраст измерения {age or 0:6.0f} мс   пик {peak_age:.0f} мс"
                    f"   состояние {tracking}", age_pair)
                put(f"  выход {row.get('output_wall_hz') or 0:6.1f} Гц"
                    f"   измерения {row.get('measurement_wall_hz') or 0:5.1f} Гц"
                    f"   источники {','.join(row.get('source_mask') or []) or '—'}")
                if speed > 0.2 and not math.isnan(lag):
                    put(f"  отставание во времени ≈ {lag/max(speed,1e-6)*1000:6.0f} мс "
                        f"пути при текущей скорости")
            else:
                put(f"  ЛОКАЛИЗАЦИЯ  {error or 'нет данных'}   ({args.status_url})", 3)
            put("")
            put(f"  пик скорости {peak_speed:5.2f} м/с = {peak_speed*3.6:5.1f} км/ч"
                f"   отсчётов истины {updates}")
            if placer.last:
                put("  " + placer.last, 2)
            put("")
            for text in HELP:
                put("  " + text)
            screen.refresh()
            time.sleep(period)
    finally:
        estimate.stop.set()
        publisher.publish(Twist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/model/tag_rover/cmd_vel")
    parser.add_argument("--linear", type=float, default=4.0,
                        help="Initial top speed, m/s; 1..9 and 0 change it live")
    parser.add_argument("--angular", type=float, default=2.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--release", type=float, default=0.12,
                        help="Grace period before an unrepeated key decays")
    parser.add_argument("--coast", type=float, default=0.25,
                        help="Decay time constant after release")
    parser.add_argument("--status-url",
                        default="http://127.0.0.1:8081/api/status")
    parser.add_argument("--world", default="mocap_arena",
                        help="World name, for the set_pose service")
    parser.add_argument("--home", type=float, nargs=3, default=(3.0, 2.0, 0.0),
                        metavar=("X", "Y", "YAW_DEG"),
                        help="Where R puts the rover back")
    parser.add_argument("--reset-z", type=float, default=0.30,
                        help="Drop height for a reset; it settles onto its wheels")
    args = parser.parse_args()
    curses.wrapper(run, args)


if __name__ == "__main__":
    main()
