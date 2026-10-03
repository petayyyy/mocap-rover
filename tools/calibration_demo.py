"""Six in-process camera nodes with rendered views, for the calibration station's --demo.

Real ``pi_cam.camera_node.CameraNode`` instances on localhost ports, each with
a sensor that serves a rendered 1640x1232 image as a raw mosaic (so the node's
grey/JPEG path runs as on a CM4).  The cameras are those of
``config/mocap_arena_imx219/cameras_ground_truth.json`` (mount errors
included); the station only gets the nominal ``runtime_cameras.json`` and has
to find the true poses.  The hand board is shown to the selected camera
through the same lens (equidistant, D = 0), a new pose every capture or 2.5 s.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from localization_contracts import charuco_calibration as cc
from pi_cam.camera_node import CameraNode, Frame, NodeConfig, boottime_ns
from tools.charuco_synthetic import HandBoardRenderer, StripRenderer

ROOT = Path(__file__).resolve().parents[1]
NS = 1_000_000_000
DEMO_PLACEMENTS = [cc.StripPlacement(0.0, 7.5, 0.0, True),
                   cc.StripPlacement(0.05, 4.6, 0.4, False),
                   cc.StripPlacement(-0.05, 10.4, -0.3, False)]


class DemoSensor:
    def __init__(self, world, camera_id, fps=30.0):
        self.world, self.camera_id, self.fps = world, camera_id, fps
        self.width, self.height = cc.SENSOR_SIZE
        self.stream_name, self.wire_format = "raw", "bayer8"
        self.mode = {"size": [self.width, self.height], "bit_depth": 8, "format": "demo",
                     "stream": "raw", "wire_format": "bayer8"}
        self.sequence = 0
        self.next_due = None
        self.exposure_ns = 800_000

    def start(self):
        self.next_due = boottime_ns()

    def stop(self):
        pass

    def set_controls(self, exposure_us=None, gain=None, fps=None):
        return {"exposure_us": exposure_us, "gain": gain, "fps": fps}

    def capture(self):
        period = int(NS / self.fps)
        stamp = self.next_due
        now = boottime_ns()
        if stamp + period > now:
            time.sleep((stamp + period - now) / NS)
        self.next_due += period
        if self.next_due < boottime_ns() - 5 * period:
            self.next_due = boottime_ns() - period
        seq = self.sequence
        self.sequence += 1
        return Frame(self.world.image(self.camera_id), stamp, self.exposure_ns, period, 1.0, seq)


class DemoWorld:
    def __init__(self, base_config_path):
        self.base_config_path = Path(base_config_path)
        self.truth = json.loads((self.base_config_path.parent / "cameras_ground_truth.json").read_text())
        self.camera_ids = [c["name"] for c in self.truth["cameras"]]
        self.strip = StripRenderer(self.truth["cameras"])
        self.hand_target = cc.CharucoTarget(legacy=True)
        self.hand = None
        self.placement = 0
        self.mode, self.selected = "extrinsic", None
        self.lock = threading.Lock()
        self.strip_images = {}                 # (placement, camera) -> image
        self.hand_image = None
        self.hand_shown_at = 0.0
        blank = np.full((cc.SENSOR_SIZE[1], cc.SENSOR_SIZE[0]), 40, np.uint8)
        self.blank = blank
        self.nodes = []
        self.stop_event = threading.Event()
        self.wake = threading.Event()

    @property
    def addresses(self):
        return [f"127.0.0.1:{n.port}" for n in self.nodes]

    def start(self):
        for cid in self.camera_ids:
            cfg = NodeConfig(camera_id=cid, port=0, sensor="synthetic", stream="raw",
                             ptp_enabled=False, status_period_s=1.0, send_queue=3)
            node = CameraNode(cfg, sensor=DemoSensor(self, cid))
            node.start()
            self.nodes.append(node)
        threading.Thread(target=self._render_loop, daemon=True).start()

    def stop(self):
        self.stop_event.set()
        self.wake.set()
        for node in self.nodes:
            node.stop()

    # rendering happens off the capture threads
    def _render_loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                placement, mode, selected = self.placement, self.mode, self.selected
            # Like a person: the board moves on after a while even without a capture.
            if mode == "intrinsic" and self.hand_image is not None and time.monotonic() - self.hand_shown_at > 2.5:
                self.hand_image = None
            if mode == "intrinsic" and selected and self.hand_image is None:
                if self.hand is None:
                    # The same lens as the strip views, so a demo intrinsic calibration
                    # is what the demo extrinsic solve needs.
                    self.hand = HandBoardRenderer(self.hand_target, self.strip.k, self.strip.d, seed=7)
                image = None
                while image is None:
                    candidate, _ = self.hand.render()
                    if self.hand_target.detect(candidate).count >= 20:
                        image = candidate
                self.hand_image = image
                self.hand_shown_at = time.monotonic()
                continue
            missing = [c for c in self.camera_ids if (placement, c) not in self.strip_images]
            if placement < len(DEMO_PLACEMENTS) and missing:
                cid = missing[0]
                image = self.strip.render(cid, DEMO_PLACEMENTS[placement], seed=placement)
                self.strip_images[(placement, cid)] = image
                continue
            self.wake.wait(0.2)
            self.wake.clear()

    def image(self, camera_id):
        if self.mode == "intrinsic" and camera_id == self.selected:
            return self.hand_image if self.hand_image is not None else self.blank
        if self.placement >= len(DEMO_PLACEMENTS):
            return self.blank                  # the strip has been taken away
        return self.strip_images.get((self.placement, camera_id), self.blank)

    # station hooks
    def set_mode(self, mode, selected):
        with self.lock:
            self.mode, self.selected = mode, selected
            self.hand_image = None
        self.wake.set()

    def next_hand_view(self):
        self.hand_image = None
        self.wake.set()

    def next_placement(self):
        with self.lock:
            self.placement += 1
        self.wake.set()

    def suggested_placement(self):
        if self.placement >= len(DEMO_PLACEMENTS):
            return None
        p = DEMO_PLACEMENTS[self.placement]
        ready = sum((self.placement, c) in self.strip_images for c in self.camera_ids)
        return {"index": self.placement, "x0": p.x0, "y0": p.y0, "yaw_deg": p.yaw_deg,
                "known": p.known, "rendered": ready, "cameras": len(self.camera_ids),
                "note": "демо: опорное положение известно, остальные решаются"}

    def strip_intrinsics(self):
        return self.strip.k, self.strip.d
