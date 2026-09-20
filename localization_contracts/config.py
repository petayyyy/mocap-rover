from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
from .contracts import SCHEMA_VERSION

class ConfigError(ValueError): pass

def validate_config(c):
    if not isinstance(c, dict) or c.get("schema_version") != SCHEMA_VERSION: raise ConfigError("unsupported or missing schema_version")
    for key in ("arena", "cameras", "capture", "timing", "detectors", "filters", "recording"):
        if key not in c: raise ConfigError(f"missing config section: {key}")
    if c["arena"].get("frame_id") != "arena": raise ConfigError("arena.frame_id must be arena")
    if c["capture"].get("width", 0) <= 0 or c["capture"].get("height", 0) <= 0: raise ConfigError("invalid capture size")
    for cam in c["cameras"]:
        if not cam.get("camera_id") or not cam.get("interface"): raise ConfigError("camera identity/interface required")
    return c

def load_config(path):
    with open(path, encoding="utf-8") as f: c = json.load(f)
    return validate_config(c)

class ConfigStore:
    def __init__(self, config): self._active = copy.deepcopy(validate_config(config)); self._staged = None; self._history = []
    @property
    def active(self): return copy.deepcopy(self._active)
    @property
    def staged(self): return copy.deepcopy(self._staged)
    def stage(self, patch):
        candidate = copy.deepcopy(self._active); self._merge(candidate, patch); validate_config(candidate); self._staged = candidate; return copy.deepcopy(candidate)
    def apply(self):
        if self._staged is None: raise ConfigError("nothing staged")
        validate_config(self._staged); self._history.append(self._active); self._active = self._staged; self._staged = None; return self.active
    def rollback(self):
        if not self._history: raise ConfigError("no previous config")
        self._active = self._history.pop(); self._staged = None; return self.active
    @staticmethod
    def _merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict): ConfigStore._merge(dst[k], v)
            else: dst[k] = copy.deepcopy(v)
    @staticmethod
    def digest(config): return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
