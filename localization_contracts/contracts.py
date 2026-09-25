from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping
import math

SCHEMA_VERSION = "1.0"
FRAME_ARENA = "arena"
CLOCK_SIM = "sim"

def _finite(x: float) -> bool: return isinstance(x, (int, float)) and math.isfinite(x)
def _vec(v, n, name):
    if not isinstance(v, (list, tuple)) or len(v) != n or not all(_finite(x) for x in v): raise ValueError(f"{name} must contain {n} finite values")

@dataclass(frozen=True)
class Observation:
    schema_version: str
    camera_id: str
    frame_seq: int
    measurement_id: str
    object_id: str
    capture_time_ns: int
    clock_domain: str
    timestamp_uncertainty_ns: int
    exposure_duration_ns: int
    receive_time_ns: int
    processed_time_ns: int
    calibration_version: str
    frame_id: str
    position_m: tuple[float, float, float]
    covariance_m2: tuple[float, ...]
    quality: float
    method: str
    marker_family: str | None = None
    marker_id: int | None = None
    bundle_id: str | None = None
    pose_6d_valid: bool = False
    attitude_state: str = "unknown"
    pixel_features: Mapping[str, Any] = field(default_factory=dict)

    def validate(self):
        if self.schema_version != SCHEMA_VERSION: raise ValueError("incompatible observation schema")
        if not self.camera_id or self.frame_seq < 0 or not self.measurement_id or not self.object_id: raise ValueError("invalid observation identity")
        if self.clock_domain not in {CLOCK_SIM, "wall", "hardware"}: raise ValueError("unknown clock domain")
        if min(self.capture_time_ns, self.receive_time_ns, self.processed_time_ns, self.timestamp_uncertainty_ns, self.exposure_duration_ns) < 0: raise ValueError("timestamps/durations must be non-negative")
        _vec(self.position_m, 3, "position_m")
        if len(self.covariance_m2) not in (3, 6, 9, 36): raise ValueError("unsupported covariance shape")
        if not all(_finite(x) for x in self.covariance_m2): raise ValueError("covariance must be finite")
        # A dense block carries cross terms, which are signed; only the
        # variances on the diagonal have to be non-negative.
        stride = {3: 1, 6: 1, 9: 3, 36: 6}[len(self.covariance_m2)]
        diagonal = (self.covariance_m2 if stride == 1
                    else [self.covariance_m2[i * stride + i] for i in range(stride)])
        if any(x < 0 for x in diagonal): raise ValueError("covariance variances must be non-negative")
        if not 0 <= self.quality <= 1: raise ValueError("quality must be in [0,1]")
        return self

@dataclass(frozen=True)
class TrackStatus:
    schema_version: str; object_id: str; valid: bool; tracking_state: str
    last_measurement_stamp_ns: int | None; measurement_age_ms: float | None
    sources: tuple[str, ...]; measurement_hz: float; output_hz: float
    orientation_valid: bool; calibration_version: str; session_id: str; reset_counter: int
    pose_frame: str = FRAME_ARENA; twist_frame: str = "base_link"

    def validate(self):
        if self.schema_version != SCHEMA_VERSION or self.pose_frame != FRAME_ARENA: raise ValueError("invalid track schema/frame")
        if self.tracking_state not in {"INITIALIZING", "TRACKING", "COASTING", "LOST", "CALIBRATION_INVALID", "TIME_UNCERTAIN", "DEAD_RECKONING"}: raise ValueError("unknown tracking state")
        if self.valid and self.tracking_state in {"LOST", "CALIBRATION_INVALID", "TIME_UNCERTAIN"}: raise ValueError("invalid track cannot be valid")
        return self

@dataclass(frozen=True)
class CameraStatus:
    schema_version: str; camera_id: str; online: bool; interface: str
    firmware: str | None; mode: str; width: int; height: int; measured_fps: float
    drops: int; calibration_version: str | None; session_id: str
    def validate(self):
        if self.schema_version != SCHEMA_VERSION or self.width <= 0 or self.height <= 0 or self.measured_fps < 0 or self.drops < 0: raise ValueError("invalid camera status")
        return self

@dataclass(frozen=True)
class CalibrationSet:
    schema_version: str; version: str; content_hash: str; frame_id: str
    cameras: Mapping[str, Mapping[str, Any]]; created_at_ns: int
    def validate(self):
        if self.schema_version != SCHEMA_VERSION or self.frame_id != FRAME_ARENA or not self.version or not self.content_hash or not self.cameras: raise ValueError("invalid calibration set")
        for cid, c in self.cameras.items():
            if not isinstance(cid, str) or "K" not in c or "position_world" not in c: raise ValueError(f"incomplete calibration: {cid}")
            if len(c["K"]) != 3 or any(len(row) != 3 for row in c["K"]): raise ValueError(f"invalid K: {cid}")
        return self

def covariance_matrix(covariance, size=2):
    """Read a stored covariance as a dense ``size x size`` matrix.

    Observations carry 3 diagonal terms, a 6-element upper triangle, a full 9
    or a 36-element pose block.  Marker geometry is strongly anisotropic --
    range error along the viewing ray and bearing error across it differ by
    ``1/cos(incidence)`` -- so the cross term has to survive the trip from the
    observer to the filter.
    """
    import numpy as np
    values = np.asarray(covariance, dtype=float).reshape(-1)
    size = int(size)
    if values.size == 3:
        dense = np.diag(values)
    elif values.size == 6:
        dense = np.zeros((3, 3))
        dense[np.triu_indices(3)] = values
        dense = dense + dense.T - np.diag(np.diag(dense))
    elif values.size == 9:
        dense = values.reshape(3, 3)
    elif values.size == 36:
        dense = values.reshape(6, 6)[:3, :3]
    else:
        raise ValueError("unsupported covariance shape")
    return dense[:size, :size]


def to_dict(obj): return asdict(obj)
