"""Adapter boundary: production code may provide sim, replay, or hardware frames.
No ground-truth field exists in the measurement interface."""
from dataclasses import dataclass
from typing import Protocol, Iterator, Mapping, Any

@dataclass(frozen=True)
class Frame:
    camera_id: str; sequence: int; capture_time_ns: int; clock_domain: str; payload: Any

class CaptureAdapter(Protocol):
    def frames(self) -> Iterator[Frame]: ...
    def status(self) -> Mapping[str, Any]: ...

class ReplayAdapter:
    def __init__(self, frames): self._frames = tuple(frames)
    def frames(self): return iter(self._frames)
    def status(self): return {"kind": "replay", "count": len(self._frames)}
