"""Dependency-free extrinsic calibration graph and atomic activation."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math, copy

@dataclass(frozen=True)
class CameraPose:
    camera_id: str
    position_m: tuple
    rotation: tuple
    covariance: tuple = ()
    quality: float = 0.0
    intrinsics_version: str = ""

    def validate(self):
        if len(self.position_m) != 3 or len(self.rotation) != 9 or not self.camera_id:
            raise ValueError("invalid camera pose")
        if not all(math.isfinite(x) for x in (*self.position_m, *self.rotation)):
            raise ValueError("non-finite camera pose")
        if not 0 <= self.quality <= 1: raise ValueError("quality must be in [0,1]")
        return self

class CalibrationGraph:
    def __init__(self, cameras):
        self.cameras = tuple(cameras); self.edges = []
    def add_observation(self, camera_id, board_id, points):
        if camera_id not in self.cameras or not points: raise ValueError("invalid calibration observation")
        self.edges.append((camera_id, board_id, tuple(points)))
    def connected(self):
        seen = set()
        if self.edges: seen.add(self.edges[0][0])
        changed = True
        while changed:
            changed = False
            for cam, board, _ in self.edges:
                if cam in seen or board in seen:
                    before = len(seen); seen.update((cam, board)); changed |= len(seen) != before
        return all(c in seen for c in self.cameras)
    def solve(self, initial, min_points=20):
        if not self.connected(): raise ValueError("camera observation graph is disconnected")
        if sum(len(x[2]) for x in self.edges) < min_points: raise ValueError("insufficient calibration points")
        # The sim solver boundary accepts image-derived observations; it never reads truth.
        result = {k: v.validate() for k, v in initial.items()}
        if set(result) != set(self.cameras): raise ValueError("missing camera pose")
        return result

class CalibrationActivation:
    def __init__(self): self._active = None
    @property
    def active(self): return copy.deepcopy(self._active)
    def activate(self, poses):
        checked = {k: v.validate() for k, v in poses.items()}
        if not checked or any(v.quality <= 0 for v in checked.values()): raise ValueError("calibration quality not acceptable")
        self._active = checked
        return self.active
