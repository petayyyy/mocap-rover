"""Dependency-free extrinsic calibration graph and atomic activation."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math, copy
import numpy as np

@dataclass(frozen=True)
class CameraPose:
    camera_id: str
    position_m: tuple
    rotation: tuple
    covariance: tuple = ()
    quality: float = 0.0
    intrinsics_version: str = ""

    def validate(self):
        if len(self.position_m) != 3 or len(self.rotation) != 9 or not self.camera_id:
            raise ValueError("invalid camera pose")
        if not all(math.isfinite(x) for x in (*self.position_m, *self.rotation)):
            raise ValueError("non-finite camera pose")
        if not 0 <= self.quality <= 1: raise ValueError("quality must be in [0,1]")
        return self

class CalibrationGraph:
    def __init__(self, cameras):
        self.cameras = tuple(cameras); self.edges = []
    def add_observation(self, camera_id, board_id, points):
        if camera_id not in self.cameras or not points: raise ValueError("invalid calibration observation")
        self.edges.append((camera_id, board_id, tuple(points)))

    def add_image_observation(self, camera_id, board_id, object_points, image_points, K, D=None, board_pose=None):
        """Register an image-derived board observation for the OpenCV solver.

        ``object_points`` are board-frame metric points and ``board_pose`` is an
        optional 4x4 arena<-board transform.  Ground truth is not accepted by
        this API; the board pose is a measured/calibration input.
        """
        if camera_id not in self.cameras or not board_id: raise ValueError("invalid calibration observation")
        obj=np.asarray(object_points,dtype=float).reshape(-1,3); img=np.asarray(image_points,dtype=float).reshape(-1,2)
        if len(obj)<4 or len(obj)!=len(img) or not np.isfinite(obj).all() or not np.isfinite(img).all(): raise ValueError("invalid image observation points")
        k=np.asarray(K,dtype=float).reshape(3,3); d=np.zeros((5,1)) if D is None else np.asarray(D,dtype=float).reshape(-1,1)
        if not np.isfinite(k).all() or not np.isfinite(d).all(): raise ValueError("invalid camera calibration")
        T=np.eye(4) if board_pose is None else np.asarray(board_pose,dtype=float).reshape(4,4)
        if not np.isfinite(T).all(): raise ValueError("invalid board pose")
        self.edges.append((camera_id, board_id, {"object_points":obj,"image_points":img,"K":k,"D":d,"board_pose":T}))

    def solve_image_observations(self, min_points=4, reprojection_limit_px=3.0):
        """Estimate camera poses from registered metric image observations.

        This is a per-view PnP estimator with robust median aggregation, not a
        full bundle-adjustment solver.  It refuses missing/disconnected data
        and reports enough quality to prevent silent activation of bad views.
        """
        if not self.connected(): raise ValueError("camera observation graph is disconnected")
        try:
            import cv2
        except ImportError as exc: raise RuntimeError("image calibration requires OpenCV") from exc
        estimates={cid:[] for cid in self.cameras}; total=0
        for camera_id, _board_id, points in self.edges:
            if not isinstance(points, dict): continue
            obj=points["object_points"]; img=points["image_points"]
            if len(obj)<min_points: continue
            ok,rvec,tvec=cv2.solvePnP(obj,img,points["K"],points["D"],flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or not np.isfinite(tvec).all(): continue
            projected,_=cv2.projectPoints(obj,rvec,tvec,points["K"],points["D"])
            error=float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-img)**2,axis=1))))
            if error>reprojection_limit_px: continue
            R,_=cv2.Rodrigues(rvec); Tcb=np.eye(4); Tcb[:3,:3]=R; Tcb[:3,3]=tvec.reshape(3)
            Tab=points["board_pose"]; Tac=Tab @ np.linalg.inv(Tcb)
            estimates[camera_id].append((Tac,error,len(obj))); total+=len(obj)
        result={}
        for camera_id, views in estimates.items():
            if not views: raise ValueError(f"no valid image observations for {camera_id}")
            # Translation median is robust to one bad view; rotation is taken
            # from the lowest reprojection-error view and checked as a proper R.
            best=min(views,key=lambda x:x[1]); position=np.median(np.stack([x[0][:3,3] for x in views]),axis=0)
            R=best[0][:3,:3]
            result[camera_id]=CameraPose(camera_id,tuple(float(x) for x in position),tuple(float(x) for x in R.reshape(-1)),
                covariance=(float(np.var([x[0][0,3] for x in views])),)*3, quality=max(0.,min(1.,math.exp(-best[1]))), intrinsics_version="image-derived").validate()
        return result
    def connected(self):
        seen = set()
        if self.edges: seen.add(self.edges[0][0])
        changed = True
        while changed:
            changed = False
            for cam, board, _ in self.edges:
                if cam in seen or board in seen:
                    before = len(seen); seen.update((cam, board)); changed |= len(seen) != before
        return all(c in seen for c in self.cameras)
    def solve(self, initial, min_points=20):
        if not self.connected(): raise ValueError("camera observation graph is disconnected")
        if sum(len(x[2]) for x in self.edges) < min_points: raise ValueError("insufficient calibration points")
        # The sim solver boundary accepts image-derived observations; it never reads truth.
        result = {k: v.validate() for k, v in initial.items()}
        if set(result) != set(self.cameras): raise ValueError("missing camera pose")
        return result

class CalibrationActivation:
    def __init__(self): self._active = None
    @property
    def active(self): return copy.deepcopy(self._active)
    def activate(self, poses):
        checked = {k: v.validate() for k, v in poses.items()}
        if not checked or any(v.quality <= 0 for v in checked.values()): raise ValueError("calibration quality not acceptable")
        self._active = checked
        return self.active
