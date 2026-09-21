"""Explicit YOLO runtime capability boundary for simulation S10/S11."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class YoloRuntimeStatus:
    available:bool; backend:str; weights:str|None; reason:str; hardware_verified:bool=False

def runtime_status(weights=None):
    path=str(weights) if weights is not None else None
    if path and not Path(path).is_file():
        return YoloRuntimeStatus(False,"unavailable",path,"weights file does not exist")
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return YoloRuntimeStatus(False,"unavailable",path,"ultralytics runtime is not installed")
    if not path:
        return YoloRuntimeStatus(False,"unavailable",None,"no trained weights were supplied")
    return YoloRuntimeStatus(True,"ultralytics",path,"runtime and weights are present")

class YoloDetector:
    """Load only explicit trained weights; never synthesizes bbox detections."""
    def __init__(self, weights):
        status=runtime_status(weights)
        if not status.available: raise RuntimeError(status.reason)
        from ultralytics import YOLO
        self.model=YOLO(str(weights)); self.status=status

    def detect(self, image):
        result=self.model.predict(source=image, verbose=False)
        return result
