"""Deterministic simulation/replay timing primitives (no hardware clock I/O)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json

@dataclass(frozen=True)
class TimingMetadata:
    sequence: int
    capture_time_ns: int
    receive_time_ns: int
    processed_time_ns: int | None
    clock_domain: str
    timestamp_uncertainty_ns: int
    exposure_duration_ns: int
    exposure_reference: str = "mid"
    trigger_counter: int | None = None

    def validate(self):
        if self.sequence < 0 or min(self.capture_time_ns, self.receive_time_ns,
            self.timestamp_uncertainty_ns, self.exposure_duration_ns) < 0:
            raise ValueError("invalid timing metadata")
        if self.exposure_reference not in {"start", "mid", "end", "delivery"}:
            raise ValueError("unknown exposure reference")
        if self.clock_domain not in {"sim", "wall", "replay"}:
            raise ValueError("unknown clock domain")
        return self

@dataclass(frozen=True)
class ClockModel:
    offset_ns: int = 0
    drift_ppm: float = 0.0
    uncertainty_ns: int = 0
    mode: str = "free_running"

    def correct(self, timestamp_ns: int) -> int:
        return int(timestamp_ns - self.offset_ns - timestamp_ns * self.drift_ppm / 1_000_000)

    def validate(self):
        if self.mode not in {"free_running", "ideal_sync"} or self.uncertainty_ns < 0:
            raise ValueError("invalid clock model")
        return self

@dataclass(frozen=True)
class TimeQuality:
    samples: int
    mean_offset_ns: float
    max_skew_ns: int
    uncertainty_ns: int
    synchronized: bool
    reason: str

def assess_quality(samples, uncertainty_limit_ns=5_000_000, skew_limit_ns=5_000_000):
    offsets = [int(x) for x in samples]
    if not offsets:
        return TimeQuality(0, 0.0, 0, uncertainty_limit_ns, False, "no_samples")
    mean = sum(offsets) / len(offsets)
    skew = max(offsets) - min(offsets)
    ok = max(abs(x) for x in offsets) <= skew_limit_ns and skew <= skew_limit_ns
    return TimeQuality(len(offsets), mean, skew, uncertainty_limit_ns, ok,
                       "measured_offsets" if ok else "offset_or_skew_exceeds_limit")

class ReplayLog:
    def __init__(self, metadata, config_hash="", calibration_hash="", model_hash=""):
        self.header = {"format": "replay-1", "config_hash": config_hash,
                       "calibration_hash": calibration_hash, "model_hash": model_hash}
        self.frames = list(metadata)

    def dump(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"header": self.header, "frames": [asdict(x) for x in self.frames]}, f, sort_keys=True)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f: data = json.load(f)
        frames = [TimingMetadata(**x).validate() for x in data["frames"]]
        obj = cls(frames, **{k: data["header"].get(k, "") for k in ("config_hash", "calibration_hash", "model_hash")})
        obj.header = data["header"]; return obj

    def verify_headers(self, config_hash, calibration_hash, model_hash):
        expected={'config_hash':config_hash,'calibration_hash':calibration_hash,'model_hash':model_hash}
        mismatches=[k for k,v in expected.items() if self.header.get(k)!=v]
        if mismatches: raise ValueError('replay provenance mismatch: '+','.join(mismatches))
        return True

class ReplayScheduler:
    def __init__(self, log): self.log = log
    def schedule(self, arrival_offset_ns=0, reset=False):
        if reset: return [("new_session", None)] + self.schedule(arrival_offset_ns)
        return [("frame", x.receive_time_ns + arrival_offset_ns, x) for x in self.log.frames]
