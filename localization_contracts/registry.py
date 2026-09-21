"""Simulation/replay camera registry.

The registry deliberately models source identity and channel health without
opening devices.  A future capture adapter can implement the same source
contract; this module does not claim hardware verification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any, Mapping

SIMULATION = "simulation"
REPLAY = "replay"


@dataclass(frozen=True)
class Capabilities:
    domain: str = SIMULATION
    image_stream: bool = True
    marker_observations: bool = True
    hardware_verified: bool = False
    virtual_edge_node: str = "gazebo"

    def validate(self):
        if self.domain not in {SIMULATION, REPLAY}:
            raise ValueError("capabilities domain must be simulation or replay")
        if self.hardware_verified:
            raise ValueError("S04 cannot declare hardware_verified")
        if not self.virtual_edge_node:
            raise ValueError("virtual_edge_node is required")
        return self


@dataclass(frozen=True)
class PreviewMetadata:
    width: int
    height: int
    pixel_format: str = "R8G8B8"
    frame_seq: int | None = None
    capture_time_ns: int | None = None
    available: bool = False

    def validate(self):
        if self.width <= 0 or self.height <= 0 or not self.pixel_format:
            raise ValueError("invalid preview metadata")
        return self


@dataclass(frozen=True)
class SourceBinding:
    runtime_id: str
    image_topic: str
    stream_id: str
    marker_stream_id: str
    calibration_version: str
    capabilities: Capabilities = field(default_factory=Capabilities)

    def validate(self):
        if not all((self.runtime_id, self.image_topic, self.stream_id,
                    self.marker_stream_id, self.calibration_version)):
            raise ValueError("source binding fields are required")
        self.capabilities.validate()
        return self


@dataclass
class _Channel:
    enabled: bool = True
    online: bool = True
    queue: deque = field(default_factory=deque)
    reconnect_attempts: int = 0
    next_retry_ns: int = 0


class CameraRegistry:
    """Registry of stable logical positions camera_1 through camera_6."""

    def __init__(self, bindings: Mapping[str, SourceBinding], queue_size: int = 3,
                 backoff_initial_ms: int = 100, backoff_max_ms: int = 2000):
        self.queue_size = queue_size
        self.backoff_initial_ms = backoff_initial_ms
        self.backoff_max_ms = backoff_max_ms
        self.bindings = dict(bindings)
        self._channels = {cid: {"image": _Channel(), "markers": _Channel()}
                          for cid in self.bindings}
        self._validate_bindings()

    @classmethod
    def virtual_default(cls, calibration_version: str = "cal-sim-1"):
        return cls({f"camera_{i}": SourceBinding(
            runtime_id=f"gz-camera-{i}", image_topic=f"/camera_{i}/image",
            stream_id=f"sim-image-{i}", marker_stream_id=f"sim-markers-{i}",
            calibration_version=calibration_version) for i in range(1, 7)})

    def _validate_bindings(self):
        expected = {f"camera_{i}" for i in range(1, 7)}
        if set(self.bindings) != expected:
            raise ValueError("registry must contain exactly camera_1..camera_6")
        for b in self.bindings.values(): b.validate()
        for name in ("runtime_id", "image_topic", "stream_id", "marker_stream_id"):
            values = [getattr(b, name) for b in self.bindings.values()]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {name}")

    def binding(self, camera_id):
        if camera_id not in self.bindings: raise KeyError(camera_id)
        return self.bindings[camera_id]

    def preview(self, camera_id, metadata: PreviewMetadata):
        metadata.validate(); return {"camera_id": camera_id, **metadata.__dict__}

    def set_channel(self, camera_id, channel, enabled):
        self._channel(camera_id, channel).enabled = enabled

    def _channel(self, camera_id, channel):
        if channel not in {"image", "markers"}: raise ValueError("unknown channel")
        if camera_id not in self._channels: raise KeyError(camera_id)
        return self._channels[camera_id][channel]

    def push(self, camera_id, channel, item):
        c = self._channel(camera_id, channel)
        if not c.enabled or not c.online: return False
        if len(c.queue) >= self.queue_size: c.queue.popleft()
        c.queue.append(item); return True

    def pop(self, camera_id, channel):
        c = self._channel(camera_id, channel)
        return c.queue.popleft() if c.queue else None

    def disconnect(self, camera_id, channel, now_ns=0):
        c = self._channel(camera_id, channel); c.online = False
        c.reconnect_attempts += 1
        delay = min(self.backoff_initial_ms * 2 ** (c.reconnect_attempts - 1), self.backoff_max_ms)
        c.next_retry_ns = now_ns + delay * 1_000_000

    def reconnect(self, camera_id, channel, now_ns):
        c = self._channel(camera_id, channel)
        if not c.enabled or now_ns < c.next_retry_ns: return False
        c.online = True; c.reconnect_attempts = 0; c.next_retry_ns = 0; return True

    def replace_source(self, camera_id, binding: SourceBinding):
        binding.validate()
        old = self.bindings[camera_id]
        if binding.calibration_version != old.calibration_version:
            raise ValueError("incompatible calibration version for source replacement")
        candidate = dict(self.bindings)
        candidate[camera_id] = binding
        old_bindings = self.bindings
        self.bindings = candidate
        try:
            self._validate_bindings()
        except Exception:
            self.bindings = old_bindings
            raise

    def status(self, camera_id):
        b = self.binding(camera_id)
        return {"camera_id": camera_id, "runtime_id": b.runtime_id,
                "calibration_version": b.calibration_version,
                "capabilities": b.capabilities.__dict__.copy(),
                "channels": {k: {"enabled": v.enabled, "online": v.online,
                                  "queue_depth": len(v.queue), "reconnect_attempts": v.reconnect_attempts}
                             for k, v in self._channels[camera_id].items()}}
