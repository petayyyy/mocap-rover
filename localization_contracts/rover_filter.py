"""Asynchronous multi-source rover filter with an honest tracking state.

The previous runtime held every camera at a barrier: a timestamp was only
finalized once all six cameras had advanced past it, and then a single best
observation survived.  Measured on the recorded run that barrier held each
observation for 226 ms (P50) on top of a 65 ms detector, so measurements
reached the filter 276 ms after capture -- past the 200 ms validity horizon,
which is why 21333 of 21361 published samples were LOST.  At 11.11 m/s that
delay is 3.1 m of travel, so no horizon setting can repair it.

Nothing here waits for a camera.  A group closes on a timeout measured from
capture, every camera in the group is applied, and a late measurement is
replayed into the state history instead of being dropped.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from .contracts import covariance_matrix

# State layout: [x, y, vx, vy, yaw, omega]
X, Y, VX, VY, YAW, OMEGA = range(6)
STATE_SIZE = 6

TRACKING = "TRACKING"
COASTING = "COASTING"
REACQUIRING = "REACQUIRING"
LOST = "LOST"

POSITION = "position"
YAW_ONLY = "yaw"
VELOCITY = "velocity"


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Measurement:
    """One source's contribution at one capture time."""

    stamp_ns: int
    kind: str                       # POSITION, YAW_ONLY or VELOCITY
    value: tuple                    # (x, y) | (yaw,) | (vx, vy)
    covariance: tuple               # matching square block, row major
    source: str                     # camera id, "lidar", "flow"
    identity: str | None = None     # marker id when the source can confirm it
    confirms_identity: bool = False
    quality: float = 1.0

    def as_arrays(self):
        z = np.asarray(self.value, dtype=float).reshape(-1)
        R = np.asarray(self.covariance, dtype=float)
        R = R.reshape(z.size, z.size) if R.size == z.size ** 2 else np.diag(R.reshape(-1))
        return z, R


def measurement_from_observation(observation, source=None):
    """Adapt a marker Observation into a position + yaw pair."""
    block = covariance_matrix(observation.covariance_m2, 2)
    camera = source or observation.camera_id
    identity = (None if observation.marker_id is None
                else f"{observation.marker_family}:{observation.marker_id}")
    out = [Measurement(
        int(observation.capture_time_ns), POSITION,
        (float(observation.position_m[0]), float(observation.position_m[1])),
        tuple(block.reshape(-1)), camera, identity, True,
        float(observation.quality),
    )]
    features = observation.pixel_features or {}
    if "yaw_rad" in features and observation.pose_6d_valid:
        sigma = float(features.get("yaw_sigma_rad", math.radians(8.0)))
        out.append(Measurement(
            int(observation.capture_time_ns), YAW_ONLY,
            (float(features["yaw_rad"]),), (sigma ** 2,), camera,
            identity, False, float(observation.quality),
        ))
    return out


class AsyncObservationBuffer:
    """Group near-simultaneous measurements without waiting for any source.

    A group closes ``window_ns`` after the oldest capture time it holds, not
    when the slowest camera reports.  One source contributes at most one
    measurement of each kind per group, so a camera that produced both a
    marker detection and a tracker continuation cannot be counted twice.
    """

    def __init__(self, window_ns=12_000_000):
        self.window_ns = int(window_ns)
        self.pending = []
        self.released = 0
        self.dropped_duplicate = 0

    def push(self, measurement):
        self.pending.append(measurement)

    def drain(self, now_ns, force=False):
        """Return closed groups, oldest first."""
        if not self.pending:
            return []
        self.pending.sort(key=lambda m: m.stamp_ns)
        groups = []
        while self.pending:
            oldest = self.pending[0].stamp_ns
            if not force and int(now_ns) - oldest < self.window_ns:
                break
            limit = oldest + self.window_ns
            group, rest = [], []
            for measurement in self.pending:
                (group if measurement.stamp_ns < limit else rest).append(measurement)
            self.pending = rest
            groups.append(self._deduplicate(group))
            self.released += 1
        return groups

    def _deduplicate(self, group):
        best = {}
        for measurement in group:
            key = (measurement.source, measurement.kind)
            current = best.get(key)
            if current is None:
                best[key] = measurement
                continue
            self.dropped_duplicate += 1
            if np.trace(measurement.as_arrays()[1]) < np.trace(current.as_arrays()[1]):
                best[key] = measurement
        return sorted(best.values(), key=lambda m: (m.stamp_ns, m.source, m.kind))


def reject_position_outliers(group, max_sigma=3.0):
    """Drop position measurements far from the group median.

    With one or two cameras there is no majority to appeal to, so the median
    test only runs from three.
    """
    positions = [m for m in group if m.kind == POSITION]
    if len(positions) < 3:
        return group, []
    centre = np.median([m.value for m in positions], axis=0)
    kept, rejected = [], []
    for measurement in positions:
        sigma = math.sqrt(max(np.trace(measurement.as_arrays()[1]) / 2, 1e-9))
        if np.linalg.norm(np.asarray(measurement.value) - centre) > max_sigma * sigma:
            rejected.append(measurement)
        else:
            kept.append(measurement)
    if not kept:                      # never discard the whole group
        return group, []
    others = [m for m in group if m.kind != POSITION]
    return kept + others, rejected


@dataclass
class Snapshot:
    stamp_ns: int
    x: list                          # one state per motion model
    P: list
    mu: np.ndarray
    last_measurement_ns: int | None
    last_identity_ns: int | None
    sources: frozenset = field(default_factory=frozenset)


class MotionModel:
    """Constant velocity with an independent heading rate."""

    name = "cv"

    def __init__(self, sigma_a=8.0, sigma_alpha=20.0):
        self.sigma_a = float(sigma_a)
        self.sigma_alpha = float(sigma_alpha)

    def predict(self, x, P, dt):
        F = np.eye(STATE_SIZE)
        F[X, VX] = F[Y, VY] = dt
        F[YAW, OMEGA] = dt
        G = np.zeros((STATE_SIZE, 3))
        G[X, 0] = G[Y, 1] = dt * dt / 2
        G[VX, 0] = G[VY, 1] = dt
        G[YAW, 2] = dt * dt / 2
        G[OMEGA, 2] = dt
        Q = G @ np.diag([self.sigma_a ** 2, self.sigma_a ** 2,
                         self.sigma_alpha ** 2]) @ G.T
        out = F @ x
        out[YAW] = wrap(out[YAW])
        return out, F @ P @ F.T + Q


class CoordinatedTurnModel(MotionModel):
    """Constant turn rate and velocity: the rover carving a turn.

    The process noise deliberately matches the constant-velocity model.  If
    the two differed in how much noise they admit, the IMM likelihood would
    mostly measure which Q is tighter rather than which motion actually
    happened, and the mode probabilities would follow the tuning instead of
    the rover.
    """

    name = "ctrv"

    @staticmethod
    def _propagate(x, dt):
        out = x.copy()
        omega = float(x[OMEGA])
        speed = math.hypot(x[VX], x[VY])
        heading = math.atan2(x[VY], x[VX]) if speed > 1e-9 else float(x[YAW])
        turned = heading + omega * dt
        if abs(omega) > 1e-4:
            out[X] = x[X] + speed / omega * (math.sin(turned) - math.sin(heading))
            out[Y] = x[Y] + speed / omega * (-math.cos(turned) + math.cos(heading))
        else:
            # Second-order limit of the same expression.  Short-circuiting to
            # straight-line motion here would zero d(position)/d(omega), and a
            # turn rate that starts near zero could then never be observed
            # from position curvature -- the filter would be stuck in CV.
            out[X] = x[X] + speed * (math.cos(heading) * dt
                                     - math.sin(heading) * omega * dt * dt / 2)
            out[Y] = x[Y] + speed * (math.sin(heading) * dt
                                     + math.cos(heading) * omega * dt * dt / 2)
        out[VX] = speed * math.cos(turned)
        out[VY] = speed * math.sin(turned)
        out[YAW] = wrap(x[YAW] + omega * dt)
        return out

    def predict(self, x, P, dt):
        out = self._propagate(x, dt)
        # Numerical Jacobian: the closed form adds nothing at these step sizes
        # and is one more place for a sign error to hide.
        F = np.eye(STATE_SIZE)
        step = 1e-6
        for index in range(STATE_SIZE):
            perturbed = x.copy()
            perturbed[index] += step
            shifted = self._propagate(perturbed, dt)
            delta = shifted - out
            delta[YAW] = wrap(delta[YAW])
            F[:, index] = delta / step
        G = np.zeros((STATE_SIZE, 3))
        G[X, 0] = G[Y, 1] = dt * dt / 2
        G[VX, 0] = G[VY, 1] = dt
        G[YAW, 2] = dt * dt / 2
        G[OMEGA, 2] = dt
        Q = G @ np.diag([self.sigma_a ** 2, self.sigma_a ** 2,
                         self.sigma_alpha ** 2]) @ G.T
        return out, F @ P @ F.T + Q


class ImmRoverFilter:
    """Interacting-multiple-model filter over [x, y, vx, vy, yaw, omega].

    Two models run in parallel: constant velocity for a rover under way, and a
    coordinated turn for one pivoting under power.  A single model has to
    choose between a process noise large enough for the turn -- which widens
    every gate and lets a continuation source drag the track -- and one tight
    enough for the straight, which loses the turn.

    Only a source that can read the marker may create a track or restore its
    identity.  Lidar and image trackers continue a track that already exists,
    which is what stops a second rover or a shadow from inheriting the id.
    """

    def __init__(self, *, models=None, transition=None, history_s=1.5,
                 coast_ms=300, identity_max_age_s=2.0, lost_ms=1500,
                 max_speed_mps=13.0, gate_chi2=9.21, marker_gate_chi2=16.3,
                 huber_delta=1.5, arena_bounds=(-0.3, 12.3),
                 initial_speed_sigma=6.0):
        self.models = list(models or (MotionModel(), CoordinatedTurnModel()))
        n = len(self.models)
        if transition is None:
            transition = np.full((n, n), 0.03 / max(n - 1, 1))
            np.fill_diagonal(transition, 0.97)
        self.transition = np.asarray(transition, dtype=float).reshape(n, n)
        self.mu = np.full(n, 1.0 / n)
        self.xs = None                 # one state per model
        self.Ps = None
        self.stamp_ns = None
        self.last_measurement_ns = None
        self.last_identity_ns = None
        self.identity = None
        self.sources = frozenset()
        self.history = deque(maxlen=max(8, int(history_s * 400)))
        self.applied = deque(maxlen=4096)
        self.history_ns = int(history_s * 1e9)
        self.coast_ns = int(coast_ms * 1e6)
        self.identity_max_age_ns = int(identity_max_age_s * 1e9)
        self.lost_ns = int(lost_ms * 1e6)
        self.max_speed_mps = float(max_speed_mps)
        self.gate_chi2 = float(gate_chi2)
        self.marker_gate_chi2 = float(marker_gate_chi2)
        self.huber_delta = float(huber_delta)
        self.arena_bounds = tuple(arena_bounds)
        self.initial_speed_sigma = float(initial_speed_sigma)
        self.accepted = 0
        self.rejected = 0
        self.out_of_sequence = 0
        self.repropagated_steps = 0
        self.too_old = 0
        self.id_rejections = 0
        self.session = 0

    # ------------------------------------------------------------- estimate

    @property
    def initialized(self):
        return self.xs is not None

    @property
    def x(self):
        """Probability-weighted combination of the model states."""
        if self.xs is None:
            return None
        combined = sum(w * xi for w, xi in zip(self.mu, self.xs))
        combined[YAW] = wrap(combined[YAW])
        return combined

    @property
    def P(self):
        if self.Ps is None:
            return None
        mean = self.x
        return sum(w * (Pi + np.outer(xi - mean, xi - mean))
                   for w, xi, Pi in zip(self.mu, self.xs, self.Ps))

    def reset(self):
        self.xs = self.Ps = self.stamp_ns = None
        self.last_measurement_ns = self.last_identity_ns = None
        self.identity = None
        self.sources = frozenset()
        self.history.clear()
        self.applied.clear()
        self.mu = np.full(len(self.models), 1.0 / len(self.models))
        self.session += 1

    def _push_history(self):
        self.history.append(Snapshot(
            self.stamp_ns, [xi.copy() for xi in self.xs],
            [Pi.copy() for Pi in self.Ps], self.mu.copy(),
            self.last_measurement_ns, self.last_identity_ns, self.sources))
        while len(self.history) > 1 and self.stamp_ns - self.history[0].stamp_ns > self.history_ns:
            self.history.popleft()

    def _restore(self, snapshot):
        self.xs = [xi.copy() for xi in snapshot.x]
        self.Ps = [Pi.copy() for Pi in snapshot.P]
        self.mu = snapshot.mu.copy()
        self.stamp_ns = snapshot.stamp_ns
        self.last_measurement_ns = snapshot.last_measurement_ns
        self.last_identity_ns = snapshot.last_identity_ns
        self.sources = snapshot.sources

    # ---------------------------------------------------------------- predict

    def _mix(self):
        n = len(self.models)
        if n == 1:
            return [self.xs[0].copy()], [self.Ps[0].copy()]
        normalizer = np.maximum(self.transition.T @ self.mu, 1e-12)
        mixed_x, mixed_P = [], []
        for j in range(n):
            weights = self.transition[:, j] * self.mu / normalizer[j]
            xj = sum(w * xi for w, xi in zip(weights, self.xs))
            xj[YAW] = wrap(xj[YAW])
            Pj = sum(w * (Pi + np.outer(xi - xj, xi - xj))
                     for w, xi, Pi in zip(weights, self.xs, self.Ps))
            mixed_x.append(xj)
            mixed_P.append(Pj)
        return mixed_x, mixed_P

    def predict_to(self, stamp_ns):
        if self.xs is None:
            return
        stamp_ns = int(stamp_ns)
        dt = (stamp_ns - self.stamp_ns) / 1e9
        if dt <= 0:
            return
        mixed_x, mixed_P = self._mix()
        self.xs, self.Ps = [], []
        for model, xi, Pi in zip(self.models, mixed_x, mixed_P):
            xj, Pj = model.predict(xi, Pi, dt)
            self.xs.append(self._clamp_speed(xj))
            self.Ps.append(Pj)
        self.stamp_ns = stamp_ns

    def _clamp_speed(self, state):
        speed = math.hypot(state[VX], state[VY])
        if speed > self.max_speed_mps:
            state[VX] *= self.max_speed_mps / speed
            state[VY] *= self.max_speed_mps / speed
        return state

    # ----------------------------------------------------------------- update

    @staticmethod
    def _design(kind):
        if kind == YAW_ONLY:
            H = np.zeros((1, STATE_SIZE))
            H[0, YAW] = 1.0
            return H
        H = np.zeros((2, STATE_SIZE))
        if kind == POSITION:
            H[0, X] = H[1, Y] = 1.0
        else:
            H[0, VX] = H[1, VY] = 1.0
        return H

    @staticmethod
    def _residual(kind, z, H, state):
        residual = z - H @ state
        if kind == YAW_ONLY:
            residual[0] = wrap(residual[0])
        return residual

    def _innovation(self, measurement, state=None, covariance=None):
        z, R = measurement.as_arrays()
        H = self._design(measurement.kind)
        state = self.x if state is None else state
        covariance = self.P if covariance is None else covariance
        residual = self._residual(measurement.kind, z, H, state)
        return residual, H, R, H @ covariance @ H.T + R

    def _initialise(self, measurement):
        if measurement.kind != POSITION or not measurement.confirms_identity:
            self.rejected += 1
            return False
        z, R = measurement.as_arrays()
        state = np.zeros(STATE_SIZE)
        state[X], state[Y] = z
        P = np.diag([R[0, 0], R[1, 1],
                     self.initial_speed_sigma ** 2, self.initial_speed_sigma ** 2,
                     (math.pi / 2) ** 2, 4.0])
        P[X, Y] = P[Y, X] = R[0, 1]
        self.xs = [state.copy() for _ in self.models]
        self.Ps = [P.copy() for _ in self.models]
        self.mu = np.full(len(self.models), 1.0 / len(self.models))
        self.stamp_ns = measurement.stamp_ns
        self.last_measurement_ns = measurement.stamp_ns
        self.last_identity_ns = measurement.stamp_ns
        self.identity = measurement.identity or self.identity
        self.sources = frozenset([measurement.source])
        self.accepted += 1
        self._push_history()
        return True

    def _apply_one(self, measurement, weight=1.0):
        """Sequential update of every model; independent R makes this exact."""
        gate = self.marker_gate_chi2 if measurement.confirms_identity else self.gate_chi2
        residual, H, R, S = self._innovation(measurement)
        try:
            distance = float(residual @ np.linalg.solve(S, residual))
        except np.linalg.LinAlgError:
            self.rejected += 1
            return False, 1e-12
        if not math.isfinite(distance) or distance > gate:
            self.rejected += 1
            return False, 1e-12
        z, _ = measurement.as_arrays()
        scaled = R / max(weight, 1e-3)
        likelihoods = []
        for index, (state, covariance) in enumerate(zip(self.xs, self.Ps)):
            innovation = self._residual(measurement.kind, z, H, state)
            Si = H @ covariance @ H.T + scaled
            try:
                gain = covariance @ H.T @ np.linalg.inv(Si)
                mahalanobis = float(innovation @ np.linalg.solve(Si, innovation))
                determinant = float(np.linalg.det(2 * math.pi * Si))
            except np.linalg.LinAlgError:
                likelihoods.append(1e-12)
                continue
            updated = state + gain @ innovation
            updated[YAW] = wrap(updated[YAW])
            I = np.eye(STATE_SIZE)
            self.xs[index] = self._clamp_speed(updated)
            self.Ps[index] = ((I - gain @ H) @ covariance @ (I - gain @ H).T
                              + gain @ scaled @ gain.T)
            likelihoods.append(max(
                math.exp(-0.5 * min(mahalanobis, 700.0)) / math.sqrt(max(determinant, 1e-300)),
                1e-12))
        return True, np.asarray(likelihoods)

    def apply_group(self, group, robust=True):
        """Apply one group of near-simultaneous measurements.

        Returns the measurements that were actually applied, so a caller can
        report per-source outcomes.  A group-level flag would mark a gated
        camera as accepted whenever any other source in the same 12 ms window
        got through, which silently inflates every accept rate downstream.
        Empty is falsy, so ``if filter.apply_group(...)`` still reads as before.

        Sequential updates with independent R are equivalent to one block
        update, and they let a single outlier be down-weighted without
        discarding the rest of the group.
        """
        group = [m for m in group if self._plausible(m)]
        if not group:
            return []
        group, _ = reject_position_outliers(group)
        stamp = min(m.stamp_ns for m in group)
        applied_measurements = []
        if self.xs is None:
            seed = next((m for m in group
                         if m.kind == POSITION and m.confirms_identity), None)
            if seed is None or not self._initialise(seed):
                return []
            applied_measurements.append(seed)
            group = [m for m in group if m is not seed]
            if not group:
                return applied_measurements
            stamp = max(stamp, self.stamp_ns)
        elif stamp < self.stamp_ns:
            return self._apply_out_of_sequence(group)

        self.predict_to(stamp)
        applied = False
        likelihood = np.ones(len(self.models))
        for measurement in group:
            weight = self._huber_weight(measurement) if robust else 1.0
            ok, model_likelihood = self._apply_one(measurement, weight)
            if not ok:
                continue
            applied = True
            applied_measurements.append(measurement)
            self.accepted += 1
            if measurement.kind == POSITION:
                likelihood = likelihood * model_likelihood
            self.last_measurement_ns = max(self.last_measurement_ns or 0,
                                           measurement.stamp_ns)
            if measurement.confirms_identity and self._identity_matches(measurement):
                self.last_identity_ns = measurement.stamp_ns
                self.identity = measurement.identity or self.identity
            self.sources = self.sources | {measurement.source}
        if applied:
            if len(self.models) > 1:
                posterior = (self.transition.T @ self.mu) * likelihood
                total = posterior.sum()
                self.mu = (posterior / total if total > 1e-300
                           else np.full(len(self.models), 1.0 / len(self.models)))
            self.applied.append(tuple(group))
            self._push_history()
        return applied_measurements

    def _huber_weight(self, measurement):
        residual, H, R, S = self._innovation(measurement)
        try:
            distance = math.sqrt(max(float(residual @ np.linalg.solve(S, residual)), 0.0))
        except np.linalg.LinAlgError:
            return 1.0
        return 1.0 if distance <= self.huber_delta else self.huber_delta / max(distance, 1e-9)

    def _identity_matches(self, measurement):
        if self.identity is None or measurement.identity is None:
            return True
        if measurement.identity == self.identity:
            return True
        self.id_rejections += 1
        return False

    def _plausible(self, measurement):
        if measurement.kind != POSITION:
            return True
        low, high = self.arena_bounds
        x, y = measurement.value
        if not (low <= x <= high and low <= y <= high):
            self.rejected += 1
            return False
        if measurement.confirms_identity and not self._identity_matches(measurement):
            return False
        if self.xs is not None and self.last_measurement_ns is not None:
            dt = (measurement.stamp_ns - self.last_measurement_ns) / 1e9
            if dt > 0:
                state = self.x
                if math.hypot(x - state[X], y - state[Y]) / dt > 1.5 * self.max_speed_mps:
                    self.rejected += 1
                    return False
        return True

    def _apply_out_of_sequence(self, group):
        """Rewind to before the measurement, replay what followed, catch up.

        Dropping a late measurement throws away information the filter needs;
        applying it at the current time corrupts the state with a stale
        position.  Neither is acceptable at 11 m/s, where 50 ms is 0.56 m.
        """
        stamp = min(m.stamp_ns for m in group)
        if not self.history or stamp < self.history[0].stamp_ns:
            self.too_old += 1
            return []
        stamps = [snapshot.stamp_ns for snapshot in self.history]
        index = max(bisect_right(stamps, stamp) - 1, 0)
        resume_from = self.stamp_ns
        self._restore(self.history[index])
        while len(self.history) > index + 1:
            self.history.pop()
        replay = [batch for batch in self.applied
                  if min(m.stamp_ns for m in batch) > stamp]
        self.applied = deque(
            (batch for batch in self.applied if min(m.stamp_ns for m in batch) <= stamp),
            maxlen=self.applied.maxlen)
        self.out_of_sequence += 1
        ordered = sorted([tuple(group)] + replay, key=lambda b: min(m.stamp_ns for m in b))
        late = set(id(m) for m in group)
        applied_measurements = []
        for batch in ordered:
            self.repropagated_steps += 1
            for measurement in self.apply_group(list(batch), robust=False):
                if id(measurement) in late:
                    applied_measurements.append(measurement)
        if resume_from is not None:
            self.predict_to(resume_from)
        return applied_measurements

    # ---------------------------------------------------------------- publish

    def tracking_state(self, stamp_ns):
        if self.xs is None:
            return LOST
        age = int(stamp_ns) - (self.last_measurement_ns or 0)
        identity_age = int(stamp_ns) - (self.last_identity_ns or 0)
        if age > self.lost_ns:
            return LOST
        if age > self.coast_ns or identity_age > self.identity_max_age_ns:
            return REACQUIRING
        if age > 0:
            return COASTING
        return TRACKING

    def publish(self, stamp_ns):
        """Propagate to ``stamp_ns`` and report; never invents a measurement."""
        stamp_ns = int(stamp_ns)
        if self.xs is None:
            return {"valid": False, "tracking_state": LOST, "stamp_ns": stamp_ns,
                    "state": None, "covariance": None, "measurement_age_ms": None,
                    "identity_age_ms": None, "sources": (), "identity": None,
                    "model_probabilities": [float(v) for v in self.mu]}
        self.predict_to(stamp_ns)
        state = self.tracking_state(stamp_ns)
        combined = self.x
        age = stamp_ns - (self.last_measurement_ns or stamp_ns)
        return {
            "valid": state in (TRACKING, COASTING),
            "tracking_state": state,
            "stamp_ns": stamp_ns,
            "state": {
                "x": float(combined[X]), "y": float(combined[Y]),
                "vx": float(combined[VX]), "vy": float(combined[VY]),
                "yaw": float(combined[YAW]), "yaw_rate": float(combined[OMEGA]),
            },
            "covariance": [[float(v) for v in row] for row in self.P],
            "measurement_age_ms": age / 1e6,
            "identity_age_ms": (stamp_ns - self.last_identity_ns) / 1e6
            if self.last_identity_ns is not None else None,
            "sources": tuple(sorted(self.sources)),
            "identity": self.identity,
            "model_probabilities": [float(v) for v in self.mu],
        }
