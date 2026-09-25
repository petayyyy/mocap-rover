"""Small asynchronous planar fusion filter for simulation/replay."""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from collections import deque
from .contracts import TrackStatus, SCHEMA_VERSION


def angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class State:
    stamp_ns: int
    x: float
    y: float
    yaw: float
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    covariance: float = 1.0


class PlanarFusion:
    def __init__(
        self,
        object_id="tag_rover",
        publish_hz=30,
        dropout_horizon_ms=200,
        gate_sigma=5.0,
        yaw_gate_deg=60.0,
    ):
        self.object_id = object_id
        self.publish_hz = publish_hz
        self.horizon = dropout_horizon_ms * 1_000_000
        self.gate_sigma = gate_sigma
        self.yaw_gate_rad = math.radians(yaw_gate_deg)
        self.state = None
        self.last_measurement = None
        self.measurements = 0
        self.outputs = 0
        self.rejected = 0
        self.session = 0
        self.calibration_version = None
        self.session_start_ns = None
        self.measurement_times = deque(maxlen=7200)
        self.output_times = deque(maxlen=7200)
        self._P = None

    def reset(self):
        self.state = None
        self.last_measurement = None
        self.measurements = 0
        self.outputs = 0
        self.rejected = 0
        self.session += 1
        self.session_start_ns = None
        self.measurement_times = deque(maxlen=7200)
        self.output_times = deque(maxlen=7200)
        self._P = None

    def update(self, obs):
        obs.validate()
        if (
            self.calibration_version is not None
            and obs.calibration_version != self.calibration_version
        ):
            self.reset()
        self.calibration_version = obs.calibration_version
        if (
            self.last_measurement
            and obs.measurement_id == self.last_measurement.measurement_id
        ):
            return False
        if (
            self.last_measurement
            and obs.capture_time_ns < self.last_measurement.capture_time_ns
        ):
            return False
        z = obs.position_m
        yaw = float(obs.pixel_features.get("yaw_rad", 0))
        if self.session_start_ns is None:
            self.session_start_ns = obs.capture_time_ns
        measurement_var = max(
            float(obs.covariance_m2[0]), float(obs.covariance_m2[1]), 1e-6
        )
        yaw_sigma = float(obs.pixel_features.get("yaw_sigma_rad", math.radians(8)))
        yaw_sigma = min(max(yaw_sigma, math.radians(1)), math.radians(30))
        R = np.diag([measurement_var, measurement_var, yaw_sigma ** 2])
        if (
            self.state is None
            or obs.capture_time_ns - self.state.stamp_ns > self.horizon
        ):
            self.state = State(
                obs.capture_time_ns, z[0], z[1], yaw, covariance=measurement_var
            )
            self._P = np.diag(
                [measurement_var, measurement_var, R[2, 2], 1.0, 1.0, 1.0]
            )
        else:
            dt = (obs.capture_time_ns - self.state.stamp_ns) / 1e9
            F = np.eye(6)
            F[:3, 3:] = np.eye(3) * dt
            G = np.vstack([np.eye(3) * dt * dt / 2, np.eye(3) * dt])
            Q = G @ np.diag([4.0, 4.0, 16.0]) @ G.T
            state = np.array(
                [
                    self.state.x,
                    self.state.y,
                    self.state.yaw,
                    self.state.vx,
                    self.state.vy,
                    self.state.omega,
                ]
            )
            predicted = F @ state
            P = F @ self._P @ F.T + Q
            innovation = np.array(
                [
                    z[0] - predicted[0],
                    z[1] - predicted[1],
                    angle_diff(yaw, predicted[2]),
                ]
            )
            S = P[:3, :3] + R
            if (
                innovation[:2] @ np.linalg.solve(S[:2, :2], innovation[:2])
                > self.gate_sigma**2
            ):
                self.rejected += 1
                return False
            yaw_limit = max(self.yaw_gate_rad, 4.0 * math.sqrt(float(S[2, 2])))
            if abs(innovation[2]) > yaw_limit:
                self.rejected += 1
                return False
            gain = np.linalg.solve(S, P[:3, :]).T
            updated = predicted + gain @ innovation
            H = np.zeros((3, 6))
            H[:3, :3] = np.eye(3)
            residual = np.eye(6) - gain @ H
            self._P = residual @ P @ residual.T + gain @ R @ gain.T
            self.state = State(
                obs.capture_time_ns,
                *updated,
                covariance=float(max(self._P[0, 0], self._P[1, 1])),
            )
        self.last_measurement = obs
        self.measurements += 1
        self.measurement_times.append(obs.capture_time_ns)
        return True

    def publish(self, stamp_ns, *, count_output=True):
        if count_output:
            self.outputs += 1
            self.output_times.append(stamp_ns)
        if self.state is None:
            return None
        dt = max(0, (stamp_ns - self.state.stamp_ns) / 1e9)
        age = stamp_ns - self.state.stamp_ns
        if age < 0:
            valid = False
            tracking = "TIME_UNCERTAIN"
        elif age > self.horizon:
            valid = False
            tracking = "LOST"
        elif age:
            valid = True
            tracking = "COASTING"
        else:
            valid = True
            tracking = "TRACKING"
        start = self.session_start_ns if self.session_start_ns is not None else stamp_ns
        elapsed = max((stamp_ns - start) / 1e9, 1e-9)
        dt = min(dt, self.horizon / 1e9)
        return {
            "state": State(
                stamp_ns,
                self.state.x + self.state.vx * dt,
                self.state.y + self.state.vy * dt,
                self.state.yaw + self.state.omega * dt,
                self.state.vx,
                self.state.vy,
                self.state.omega,
                self.state.covariance + max(0, age / 1e9) * 0.1,
            ),
            "valid": valid,
            "tracking_state": tracking,
            "measurement_hz": self.measurements / elapsed,
            "output_hz": self.outputs / elapsed,
            "measurement_age_ms": age / 1e6,
        }

    def status(self, stamp_ns):
        item = self.publish(stamp_ns, count_output=False)
        if item is None:
            state = "INITIALIZING"
            valid = False
            age = None
            last = None
        else:
            state = item["tracking_state"]
            valid = item["valid"]
            age = item["measurement_age_ms"]
            last = self.last_measurement.capture_time_ns
        return TrackStatus(
            SCHEMA_VERSION,
            self.object_id,
            valid,
            state,
            last,
            age,
            tuple([self.last_measurement.camera_id] if self.last_measurement else []),
            item["measurement_hz"] if item else 0.0,
            item["output_hz"] if item else 0.0,
            bool(item and item["state"].yaw is not None),
            self.calibration_version or "unset",
            str(self.session),
            self.session,
        ).validate()


class ObservationSelector:
    """Select one camera observation with quality hysteresis."""

    def __init__(self, switch_margin=0.15):
        self.switch_margin = switch_margin
        self.active_camera = None

    def select(self, observations):
        valid = [o for o in observations if o.quality >= 0]
        if not valid:
            return None
        best = max(valid, key=lambda o: o.quality)
        current = next((o for o in valid if o.camera_id == self.active_camera), None)
        if current is None or best.quality >= current.quality + self.switch_margin:
            self.active_camera = best.camera_id
            return best
        return current


class SynchronousObservationArbiter:
    """Deterministically retain the best camera for each capture timestamp.

    Camera workers must report every processed frame, including frames with no
    observation.  A timestamp is finalized once every camera has advanced to
    it, so thread scheduling cannot decide which synchronous measurement wins.
    """

    def __init__(self, camera_ids, max_pending=4):
        self.camera_ids = frozenset(camera_ids)
        if not self.camera_ids:
            raise ValueError("at least one camera is required")
        self.max_pending = int(max_pending)
        self.latest = {}
        self.pending = {}
        self.last_finalized = None
        self.late_reports = 0

    @staticmethod
    def _rank(obs):
        features = obs.pixel_features
        reproj = float(features.get("reprojection_error_px", float("inf")))
        side = float(features.get("side_px", 0.0))
        # Stable camera-id tie break makes replay independent of worker order.
        return (float(obs.quality), -reproj, side, str(obs.camera_id))

    def report(self, camera_id, capture_time_ns, observations=()):
        if camera_id not in self.camera_ids:
            raise KeyError(camera_id)
        stamp = int(capture_time_ns)
        if self.last_finalized is not None and stamp <= self.last_finalized:
            self.late_reports += 1
            return []
        previous = self.latest.get(camera_id)
        if previous is not None and stamp < previous:
            raise ValueError("camera timestamps must be monotonic")
        checked = []
        for obs in observations:
            if obs.camera_id != camera_id or obs.capture_time_ns != stamp:
                raise ValueError("observation does not match reported frame")
            checked.append(obs)
        self.latest[camera_id] = stamp
        self.pending.setdefault(stamp, {})[camera_id] = tuple(checked)
        return self._drain(False)

    def flush(self):
        return self._drain(True)

    def reset(self):
        self.latest.clear()
        self.pending.clear()
        self.last_finalized = None
        self.late_reports = 0

    def _drain(self, force):
        if not self.pending:
            return []
        watermark = (
            max(self.pending) if force
            else min(self.latest.values()) if self.camera_ids <= self.latest.keys()
            else None
        )
        if watermark is None and len(self.pending) <= self.max_pending:
            return []
        if len(self.pending) > self.max_pending:
            backlog_watermark = sorted(self.pending)[-self.max_pending]
            watermark = backlog_watermark if watermark is None else max(watermark, backlog_watermark)
        ready = [stamp for stamp in sorted(self.pending) if force or stamp <= watermark]
        finalized = []
        for stamp in ready:
            reports = self.pending.pop(stamp)
            candidates = [obs for values in reports.values() for obs in values]
            selected = max(candidates, key=self._rank) if candidates else None
            discarded = tuple(obs for obs in candidates if obs is not selected)
            finalized.append((stamp, selected, discarded))
            self.last_finalized = stamp
        return finalized
