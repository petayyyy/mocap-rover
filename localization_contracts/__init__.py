"""Shared sim/replay/hardware contracts for the localization pipeline."""

from .contracts import Observation, TrackStatus, CameraStatus, CalibrationSet
from .config import ConfigStore, ConfigError, load_config

__all__ = ["Observation", "TrackStatus", "CameraStatus", "CalibrationSet", "ConfigStore", "ConfigError", "load_config"]
from .registry import CameraRegistry, Capabilities, PreviewMetadata, SourceBinding

__all__ = ["CameraRegistry", "Capabilities", "PreviewMetadata", "SourceBinding"]
