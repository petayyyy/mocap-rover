#!/usr/bin/env python3
"""Keyboard teleoperation for the Gazebo tag rover.

W/S drive forward/backward, A/D turn left/right, Space/X stop, Q exits.
Commands time out automatically when keyboard input stops.
"""

import argparse
import curses
import time

from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node


def command(linear: float, angular: float) -> Twist:
    msg = Twist()
    msg.linear.x = linear
    msg.angular.z = angular
    return msg


def run(screen, topic: str, linear_speed: float, angular_speed: float, timeout: float):
    screen.nodelay(True)
    screen.keypad(True)
    curses.curs_set(0)

    node = Node()
    publisher = node.advertise(topic, Twist)
    linear = angular = 0.0
    last_key = 0.0
    period = 1.0 / 20.0

    try:
        while True:
            key = screen.getch()
            now = time.monotonic()
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("w"), ord("W")):
                linear, angular, last_key = linear_speed, 0.0, now
            elif key in (ord("s"), ord("S")):
                linear, angular, last_key = -linear_speed, 0.0, now
            elif key in (ord("a"), ord("A")):
                linear, angular, last_key = 0.0, angular_speed, now
            elif key in (ord("d"), ord("D")):
                linear, angular, last_key = 0.0, -angular_speed, now
            elif key in (ord("x"), ord("X"), ord(" ")):
                linear = angular = 0.0
                last_key = now

            if (linear or angular) and now - last_key > timeout:
                linear = angular = 0.0

            publisher.publish(command(linear, angular))
            screen.erase()
            screen.addstr(0, 0, "tag_rover: W/S forward/back, A/D turn, Space/X stop, Q quit")
            screen.addstr(2, 0, f"linear.x={linear:+.2f} m/s   angular.z={angular:+.2f} rad/s")
            screen.addstr(4, 0, "Hold a key (terminal key-repeat) to keep moving.")
            screen.refresh()
            time.sleep(period)
    finally:
        publisher.publish(Twist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/model/tag_rover/cmd_vel")
    parser.add_argument("--linear", type=float, default=0.6, help="Linear speed, m/s")
    parser.add_argument("--angular", type=float, default=1.2, help="Angular speed, rad/s")
    parser.add_argument("--timeout", type=float, default=0.35, help="Stop after this many seconds without a key")
    args = parser.parse_args()
    curses.wrapper(run, args.topic, args.linear, args.angular, args.timeout)


if __name__ == "__main__":
    main()
