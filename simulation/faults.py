"""Deterministic transport faults for replay/evaluation, never for truth data."""
from dataclasses import dataclass
import random

@dataclass(frozen=True)
class FaultConfig:
    seed: int = 42
    drop_probability: float = 0.0
    max_delay_ms: float = 0.0
    reorder_probability: float = 0.0
    clock_offset_ms: float = 0.0
    clock_drift_ppm: float = 0.0

class FaultInjector:
    def __init__(self, config: FaultConfig):
        if not 0 <= config.drop_probability <= 1 or not 0 <= config.reorder_probability <= 1:
            raise ValueError("fault probabilities must be in [0, 1]")
        if config.max_delay_ms < 0: raise ValueError("max_delay_ms must be non-negative")
        self.config, self.rng = config, random.Random(config.seed)

    def apply(self, frames):
        """Return frame dicts with transport fields; capture_time is immutable."""
        out = []
        for frame in frames:
            if self.rng.random() < self.config.drop_probability: continue
            f = dict(frame)
            f["receive_time_ns"] = int(f["capture_time_ns"] + self.config.clock_offset_ms * 1e6 +
                self.rng.uniform(0, self.config.max_delay_ms) * 1e6)
            f["clock_time_ns"] = int(f["capture_time_ns"] + self.config.clock_offset_ms * 1e6 +
                f["capture_time_ns"] * self.config.clock_drift_ppm / 1e6)
            out.append(f)
        if self.config.reorder_probability and len(out) > 1:
            for i in range(len(out) - 1):
                if self.rng.random() < self.config.reorder_probability: out[i], out[i+1] = out[i+1], out[i]
        return out
