"""Deterministic AprilTag observation geometry boundary for simulation/replay."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .contracts import Observation, SCHEMA_VERSION, FRAME_ARENA
from .marker_families import normalize_marker_family, marker_method

@dataclass(frozen=True)
class TagConfig:
    family: str = "tag36h11"
    ids: tuple = (0, 1)
    size_m: float = .40
    calibration_version: str = ""

@dataclass(frozen=True)
class Detection:
    camera_id: str; frame_seq: int; tag_id: int; corners: tuple
    capture_time_ns: int; receive_time_ns: int; processed_time_ns: int
    timestamp_uncertainty_ns: int = 0; exposure_duration_ns: int = 0

class AprilTagObserver:
    def __init__(self, config: TagConfig, fx, fy, cx, cy, quality_min=.07):
        family = normalize_marker_family(config.family)
        if config.size_m <= 0: raise ValueError("invalid tag configuration")
        self.config = TagConfig(family, tuple(config.ids), config.size_m, config.calibration_version)
        self.fx, self.fy, self.cx, self.cy, self.quality_min = fx, fy, cx, cy, quality_min
    def observe(self, d: Detection):
        if d.tag_id not in self.config.ids or len(d.corners) != 4: return None
        pts = d.corners
        if any(len(p) != 2 or not all(math.isfinite(x) for x in p) for p in pts): return None
        side = sum(math.hypot(pts[(i+1)%4][0]-pts[i][0], pts[(i+1)%4][1]-pts[i][1]) for i in range(4))/4
        if side <= 1: return None
        z = self.fx * self.config.size_m / side
        x = (sum(p[0] for p in pts)/4-self.cx)*z/self.fx
        y = (sum(p[1] for p in pts)/4-self.cy)*z/self.fy
        yaw = math.atan2(pts[1][1]-pts[0][1], pts[1][0]-pts[0][0])
        quality = min(1., side/100.)
        if quality < self.quality_min or z <= 0: return None
        return Observation(SCHEMA_VERSION, d.camera_id, d.frame_seq, f"{d.camera_id}:{d.frame_seq}:{d.tag_id}",
            "tag_rover", d.capture_time_ns, "sim", d.timestamp_uncertainty_ns, d.exposure_duration_ns,
            d.receive_time_ns, d.processed_time_ns, self.config.calibration_version, FRAME_ARENA,
            (x,y,z), (0.01,0.01,0.04), quality, marker_method(self.config.family), self.config.family, d.tag_id,
            # This boundary only estimates a planar centre/range from corners.  It
            # does not solve PnP, so advertising a 6-D pose here is unsafe.
            pose_6d_valid=False, attitude_state="unknown", pixel_features={"yaw_rad": yaw, "side_px": side}).validate()


class PnpAprilTagObserver(AprilTagObserver):
    """AprilTag corner-to-pose boundary using OpenCV's IPPE square solver.

    The detector remains an input to this class: corners are never synthesized
    here.  ``camera_pose`` is the calibrated ``(R_arena_camera, t_arena_camera)``
    transform, and ``base_tag`` maps tag coordinates into base coordinates.
    Both transforms are explicit so top/bottom tags do not require an ad-hoc
    yaw sign or image mirroring.
    """
    def __init__(self, config, K, D, camera_pose, base_tag=None, quality_min=.07,
                 max_reprojection_px=5.0, max_planar_tilt_deg=40.0,
                 min_side_px=8.0):
        super().__init__(config, float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2]), quality_min)
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(D, dtype=np.float64).reshape(-1, 1)
        self.max_reprojection_px = float(max_reprojection_px)
        self.max_planar_tilt_rad = math.radians(float(max_planar_tilt_deg))
        self.min_side_px = float(min_side_px)
        self.last_diagnostic = None
        self.R_arena_camera, self.t_arena_camera = self._transform(camera_pose)
        if isinstance(base_tag, dict) and 'rotation' not in base_tag:
            self.tag_transforms = {int(k): self._matrix(v) for k, v in base_tag.items()}
            self.T_base_tag = np.eye(4)
        else:
            self.T_base_tag = self._matrix(base_tag) if base_tag is not None else np.eye(4)
            self.tag_transforms = {i: self.T_base_tag for i in config.ids}

    @staticmethod
    def _matrix(value):
        if value is None: return np.eye(4)
        if isinstance(value, dict):
            R = np.asarray(value.get("rotation"), dtype=np.float64).reshape(3, 3)
            t = np.asarray(value.get("translation"), dtype=np.float64).reshape(3)
        else:
            a = np.asarray(value, dtype=np.float64)
            if a.shape == (4, 4): return a
            raise ValueError("transform must be 4x4 or {rotation,translation}")
        out = np.eye(4); out[:3, :3] = R; out[:3, 3] = t; return out

    @classmethod
    def _transform(cls, value):
        T = cls._matrix(value)
        R, t = T[:3, :3], T[:3, 3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1, atol=1e-6):
            raise ValueError("camera rotation must be proper orthonormal")
        return R, t

    def observe(self, d: Detection):
        self.last_diagnostic = {"accepted": False, "reason": "invalid_detection"}
        if d.tag_id not in self.config.ids or len(d.corners) != 4: return None
        if d.tag_id not in self.tag_transforms: return None
        image = np.asarray(d.corners, dtype=np.float64).reshape(4, 2)
        if not np.isfinite(image).all(): return None
        sides = np.asarray([np.linalg.norm(image[(i + 1) % 4] - image[i]) for i in range(4)])
        side = float(np.mean(sides))
        if side < self.min_side_px or float(np.min(sides) / max(np.max(sides), 1e-9)) < 0.18:
            self.last_diagnostic = {"accepted": False, "reason": "degenerate_quad", "side_px": side}
            return None
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("PnpAprilTagObserver requires OpenCV") from exc
        s = self.config.size_m / 2.0
        # tag frame: x right, y up, z outward; image order is top-left first.
        object_points = np.asarray([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
        try:
            result = cv2.solvePnPGeneric(
                object_points, image, self.K, self.D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not result[0]:
                self.last_diagnostic = {"accepted": False, "reason": "pnp_failed", "side_px": side}
                return None
            candidates = []
            T_arena_camera = np.eye(4)
            T_arena_camera[:3, :3] = self.R_arena_camera
            T_arena_camera[:3, 3] = self.t_arena_camera
            for initial_rvec, initial_tvec in zip(result[1], result[2]):
                rvec, tvec = cv2.solvePnPRefineLM(
                    object_points, image, self.K, self.D, initial_rvec, initial_tvec
                )
                if not np.isfinite(tvec).all() or float(tvec[2, 0]) <= 0:
                    continue
                projected, _ = cv2.projectPoints(object_points, rvec, tvec, self.K, self.D)
                reproj = float(np.sqrt(np.mean(np.sum(
                    (projected.reshape(4, 2) - image) ** 2, axis=1
                ))))
                R_camera_tag, _ = cv2.Rodrigues(rvec)
                T_camera_tag = np.eye(4)
                T_camera_tag[:3, :3] = R_camera_tag
                T_camera_tag[:3, 3] = tvec.reshape(3)
                T_arena_base = (
                    T_arena_camera @ T_camera_tag
                    @ np.linalg.inv(self.tag_transforms[d.tag_id])
                )
                # The arena rover is planar. IPPE returns two valid planar-pose
                # branches with very similar reprojection errors; select the one
                # whose reconstructed base Z axis agrees with arena Z.
                up_z = float(np.clip(T_arena_base[2, 2], -1.0, 1.0))
                # Some synthetic/unit-test camera frames use the opposite
                # world-Z convention. Planarity is axis alignment; tag ID and
                # T_base_tag still determine the directed base orientation.
                tilt = math.acos(abs(up_z))
                candidates.append((tilt, reproj, T_arena_base))
        except cv2.error:
            self.last_diagnostic = {"accepted": False, "reason": "opencv_error", "side_px": side}
            return None
        if not candidates:
            self.last_diagnostic = {"accepted": False, "reason": "no_physical_solution", "side_px": side}
            return None
        tilt, reproj, T_arena_base = min(candidates, key=lambda item: (item[0], item[1]))
        self.last_diagnostic = {
            "accepted": False,
            "reason": "candidate",
            "side_px": side,
            "reprojection_error_px": reproj,
            "planar_tilt_deg": math.degrees(tilt),
            "candidate_count": len(candidates),
        }
        if reproj > self.max_reprojection_px:
            self.last_diagnostic["reason"] = "reprojection"
            return None
        if tilt > self.max_planar_tilt_rad:
            self.last_diagnostic["reason"] = "nonplanar_pose"
            return None
        yaw = math.atan2(T_arena_base[1, 0], T_arena_base[0, 0])
        size_score = min(1.0, side / 80.0)
        reprojection_score = 1.0 / (1.0 + (reproj / 1.5) ** 2)
        planarity_score = math.exp(-0.5 * (tilt / math.radians(12.0)) ** 2)
        quality = max(0., min(1., size_score * reprojection_score * planarity_score))
        self.last_diagnostic["quality"] = quality
        if quality < self.quality_min:
            self.last_diagnostic["reason"] = "quality"
            return None
        self.last_diagnostic.update(accepted=True, reason="accepted")
        return Observation(SCHEMA_VERSION, d.camera_id, d.frame_seq, f"{d.camera_id}:{d.frame_seq}:{d.tag_id}",
            "tag_rover", d.capture_time_ns, "sim", d.timestamp_uncertainty_ns, d.exposure_duration_ns,
            d.receive_time_ns, d.processed_time_ns, self.config.calibration_version, FRAME_ARENA,
            tuple(float(x) for x in T_arena_base[:3, 3]), (0.01, 0.01, 0.04), quality,
            marker_method(self.config.family, pnp=True), self.config.family, d.tag_id, pose_6d_valid=True,
            attitude_state="valid", pixel_features={"yaw_rad": yaw, "reprojection_error_px": reproj,
                "side_px": side, "planar_tilt_deg": math.degrees(tilt),
                "pnp_candidate_count": len(candidates),
                "rotation_arena_base": T_arena_base[:3,:3].tolist()}).validate()
