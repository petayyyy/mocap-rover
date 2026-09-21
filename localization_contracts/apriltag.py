"""Deterministic AprilTag observation geometry boundary for simulation/replay."""
from __future__ import annotations
from dataclasses import dataclass
import math
from .contracts import Observation, SCHEMA_VERSION, FRAME_ARENA

@dataclass(frozen=True)
class TagConfig:
    family: str = "tag36h11"
    ids: tuple = (0, 1)
    size_m: float = .40
    calibration_version: str = ""

@dataclass(frozen=True)
class Detection:
    camera_id: str; frame_seq: int; tag_id: int; corners: tuple
    capture_time_ns: int; receive_time_ns: int; processed_time_ns: int
    timestamp_uncertainty_ns: int = 0; exposure_duration_ns: int = 0

class AprilTagObserver:
    def __init__(self, config: TagConfig, fx, fy, cx, cy, quality_min=.2):
        if config.family != "tag36h11" or config.size_m <= 0: raise ValueError("invalid tag configuration")
        self.config, self.fx, self.fy, self.cx, self.cy, self.quality_min = config, fx, fy, cx, cy, quality_min
    def observe(self, d: Detection):
        if d.tag_id not in self.config.ids or len(d.corners) != 4: return None
        pts = d.corners
        if any(len(p) != 2 or not all(math.isfinite(x) for x in p) for p in pts): return None
        side = sum(math.hypot(pts[(i+1)%4][0]-pts[i][0], pts[(i+1)%4][1]-pts[i][1]) for i in range(4))/4
        if side <= 1: return None
        z = self.fx * self.config.size_m / side
        x = (sum(p[0] for p in pts)/4-self.cx)*z/self.fx
        y = (sum(p[1] for p in pts)/4-self.cy)*z/self.fy
        yaw = math.atan2(pts[1][1]-pts[0][1], pts[1][0]-pts[0][0])
        quality = min(1., side/100.)
        if quality < self.quality_min or z <= 0: return None
        return Observation(SCHEMA_VERSION, d.camera_id, d.frame_seq, f"{d.camera_id}:{d.frame_seq}:{d.tag_id}",
            "tag_rover", d.capture_time_ns, "sim", d.timestamp_uncertainty_ns, d.exposure_duration_ns,
            d.receive_time_ns, d.processed_time_ns, self.config.calibration_version, FRAME_ARENA,
            (x,y,z), (0.01,0.01,0.04), quality, "apriltag36h11", self.config.family, d.tag_id,
            pose_6d_valid=True, attitude_state="valid", pixel_features={"yaw_rad": yaw, "side_px": side}).validate()
