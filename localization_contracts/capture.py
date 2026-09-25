"""Bounded Gazebo RGB acquisition; transport callbacks never run inference."""

import threading
import time
import numpy as np


def rgb_array(msg):
    if msg.pixel_format_type != 3:
        raise ValueError("expected RGB_INT8")
    # Protobuf copies bytes on property access: read ONCE, never once per row.
    data = msg.data
    if msg.step < msg.width * 3 or len(data) < msg.step * msg.height:
        raise ValueError("truncated RGB image")
    return np.ndarray(
        (msg.height, msg.width, 3),
        dtype=np.uint8,
        buffer=data,
        strides=(msg.step, 3, 1),
    )


class LatestFrames:
    def __init__(self, camera_ids):
        self.condition = threading.Condition()
        self.pending = {}
        self.received = dict.fromkeys(camera_ids, 0)
        self.dropped = dict.fromkeys(camera_ids, 0)
        self.closed = False

    def push(self, camera_id, msg):
        with self.condition:
            if self.closed:
                return
            self.received[camera_id] += 1
            if camera_id in self.pending:
                self.dropped[camera_id] += 1
            self.pending[camera_id] = (
                msg,
                time.monotonic_ns(),
                self.received[camera_id],
            )
            self.condition.notify_all()

    def take(self, camera_id, timeout=0.2):
        with self.condition:
            self.condition.wait_for(
                lambda: camera_id in self.pending or self.closed, timeout
            )
            return self.pending.pop(camera_id, None)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
