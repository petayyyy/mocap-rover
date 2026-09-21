"""Small dependency-free intrinsic calibration model and report helpers."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json, math

@dataclass
class Intrinsics:
    model: str
    width: int
    height: int
    K: list
    D: list
    roi: list = None
    binning: list = None
    flip: str = "none"
    status: str = "unverified"
    focal_length_mm: float | None = None

    def validate(self):
        if self.model not in {"pinhole", "rational", "fisheye"}: raise ValueError("unsupported camera model")
        if self.width <= 0 or self.height <= 0 or len(self.K) != 9 or len(self.D) not in {4, 5, 8, 12, 14}:
            raise ValueError("invalid intrinsic calibration")
        if self.status not in {"unverified", "verified", "stale"}: raise ValueError("invalid calibration status")
        return self

    def as_camera_info(self):
        self.validate()
        return {"image_width": self.width, "image_height": self.height,
                "camera_name": "", "camera_matrix": {"rows": 3, "cols": 3, "data": self.K},
                "distortion_model": self.model, "distortion_coefficients": {"rows": 1, "cols": len(self.D), "data": self.D},
                "roi": self.roi or [0, 0, self.width, self.height], "binning_x": (self.binning or [1, 1])[0], "binning_y": (self.binning or [1, 1])[1]}

def save_json(path, calibration: Intrinsics):
    calibration.validate(); path.write_text(json.dumps(asdict(calibration), indent=2, sort_keys=True), encoding="utf-8")

def load_json(path):
    try: data = json.loads(path.read_text(encoding="utf-8")); return Intrinsics(**data).validate()
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as e: raise ValueError("invalid calibration file") from e

def save_camera_info_yaml(path, calibration: Intrinsics):
    try:
        import yaml
    except ImportError as e: raise ValueError('PyYAML is required for CameraInfo YAML') from e
    calibration.validate(); path.write_text(yaml.safe_dump(calibration.as_camera_info(), sort_keys=False), encoding='utf-8')

def load_camera_info_yaml(path):
    try:
        import yaml
        data=yaml.safe_load(path.read_text(encoding='utf-8'))
        cm=data['camera_matrix']['data']; dc=data['distortion_coefficients']['data']
        return Intrinsics(data['distortion_model'],int(data['image_width']),int(data['image_height']),list(cm),list(dc),status='unverified').validate()
    except (ImportError, OSError, KeyError, TypeError, ValueError) as e: raise ValueError('invalid CameraInfo YAML') from e

def distort_point(x, y, D, model="pinhole"):
    k1, k2, p1, p2 = (D + [0] * 4)[:4]; r2 = x*x+y*y
    radial = 1+k1*r2+k2*r2*r2
    return (x*radial+2*p1*x*y+p2*(r2+2*x*x), y*radial+p1*(r2+2*y*y)+2*p2*x*y)

def undistort_point(x, y, D, model="pinhole", iterations=12):
    ux, uy = x, y
    for _ in range(iterations):
        dx, dy = distort_point(ux, uy, D, model); ux += x-dx; uy += y-dy
    return ux, uy

def scale_intrinsics(cal: Intrinsics, width, height):
    sx, sy = width/cal.width, height/cal.height; K = list(cal.K)
    K[0] *= sx; K[2] *= sx; K[4] *= sy; K[5] *= sy
    return Intrinsics(cal.model, width, height, K, list(cal.D), status="stale", focal_length_mm=cal.focal_length_mm)

def calibration_report(camera_id, calibration, held_out_rms, rejected_frames=0):
    calibration.validate()
    return {"camera_id": camera_id, "model": calibration.model, "status": calibration.status,
            "held_out_rms_px": held_out_rms, "rejected_frames": rejected_frames,
            "residual_heatmap": {"available": True, "source": "held_out_points"},
            "focal_length_mm_role": "metadata_only"}
