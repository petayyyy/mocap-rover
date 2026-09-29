"""One frame contract for every camera source: the dataset replay and the LAN nodes.

The tracking side needs, per frame, the pixels and the times that locate
them: when the first row of the window was exposed (``stamp_ns``), which
sensor row and column the window starts at (``row0``, ``col0``), how long one
row takes to read out (``line_time_ns``, 0 for a global shutter), how long
each row was exposed, and when the frame reached the laptop.  That is the
iteration contract of ``lan_capture.LanFrame``; ``CameraFrame`` carries the
same fields so a recorded dataset and a live node look the same to
``camera_worker.CameraWorker``.

Differences that are not hidden here, by design:

* a dataset frame is RGB 1640x1232, a LAN frame is a gray ("y8") window.
  ``channels`` says which; the camera background model is 3-channel today.
* a dataset frame is always the whole sensor, so ``row0 = col0 = 0``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class CameraFrame:
    camera_id: str
    image: np.ndarray | None      # None when the frame was skipped undecoded
    stamp_ns: int                 # exposure start of sensor row ``row0``
    row0: int = 0
    col0: int = 0
    line_time_ns: float = 0.0
    exposure_ns: int = 0
    receive_ns: int = 0           # arrival on the laptop, same time scale
    sequence: int = 0
    channels: str = "rgb"         # "rgb" or "gray"

    def row_stamp_ns(self, row, exposure_centre=True):
        """Exposure time of window row ``row`` (0 = first row of the window)."""
        stamp = self.stamp_ns + (self.row0 + row) * self.line_time_ns
        return int(stamp + (self.exposure_ns / 2 if exposure_centre else 0))


class DatasetFrameSource:
    """Frames of one camera of a recorded dataset, in order, never all in memory.

    ``read(row)`` takes a row of ``camera_N.jsonl``.  Frames that nobody
    looks at are skipped with ``decode=False`` (the container still has to be
    advanced).  ``gain`` multiplies every decoded frame, the lighting check
    of prompt 03.  Frames stay in the decoder's BGR order unless ``rgb``: the
    conversion is a full-frame pass for nothing when every consumer can be
    built on BGR.
    """

    def __init__(self, path, camera_id, *, transport_ns=0, line_time_ns=0.0, gain=1.0,
                 rgb=True):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.camera_id = camera_id
        self.transport_ns = int(transport_ns)
        self.line_time_ns = float(line_time_ns)
        self.gain = float(gain)
        self.rgb = bool(rgb)
        self.next_index = 0

    def _advance(self, index):
        if index < self.next_index:
            raise ValueError(f"frame {index} requested after {self.next_index - 1}")
        while self.next_index < index:
            if not self.capture.grab():
                raise RuntimeError(f"video ends before frame {index}")
            self.next_index += 1
        self.next_index += 1

    def read_image(self, index, decode=True):
        self._advance(index)
        if not decode:
            if not self.capture.grab():
                raise RuntimeError(f"video ends at frame {index}")
            return None
        ok, image = self.capture.read()
        if not ok:
            raise RuntimeError(f"video ends at frame {index}")
        if self.rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.gain != 1.0:
            image = cv2.convertScaleAbs(image, alpha=self.gain)
        return image

    def read(self, row, decode=True):
        stamp = int(row["stamp_ns"])
        return CameraFrame(self.camera_id, self.read_image(int(row["index"]), decode),
                           stamp, 0, 0, self.line_time_ns, 0, stamp + self.transport_ns,
                           int(row["index"]), "rgb" if self.rgb else "bgr")

    def close(self):
        self.capture.release()


def from_lan_frame(frame):
    """``lan_capture.LanFrame`` -> ``CameraFrame``, nothing recomputed."""
    return CameraFrame(frame.camera_id, frame.array, int(frame.stamp_ns), int(frame.row0),
                       int(frame.col0), float(frame.line_time_ns), int(frame.exposure_ns),
                       int(frame.receive_ns), int(frame.frame_seq),
                       "gray" if frame.array.ndim == 2 else "rgb")
