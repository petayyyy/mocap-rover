"""One-camera simulation image pipeline using shared localization contracts."""
from __future__ import annotations

import time
import numpy as np

from .apriltag import Detection, PnpAprilTagObserver, TagConfig
from .detector import AprilTagImageDetector
from .fusion import PlanarFusion
from .fusion import ObservationSelector


class OneCameraImagePipeline:
    """Detector -> PnP -> metric fusion boundary.

    CameraInfo is supplied at construction; evaluator/world pose data is not
    accepted by this runtime object.  ``process`` is capture-driven while
    ``publish`` is deliberately a separate caller/timer boundary.
    """
    def __init__(self, camera_id, K, D, camera_pose, base_tag, calibration_version,
                 family="tag36h11", tag_size_m=.4, detector_scale=1.0,
                 quality_min=.07, max_reprojection_px=5.0,
                 max_planar_tilt_deg=40.0, min_side_px=8.0,
                 detector_profile="coverage", marker_ids=(0, 1)):
        self.camera_id=camera_id
        self.family=family; self.marker_ids=tuple(marker_ids); self.tag_size_m=tag_size_m
        self.detector=AprilTagImageDetector(family,scale=detector_scale,
            profile=detector_profile,allowed_ids=self.marker_ids)
        self._observer_options=dict(quality_min=quality_min,
            max_reprojection_px=max_reprojection_px,
            max_planar_tilt_deg=max_planar_tilt_deg,min_side_px=min_side_px)
        self.observer=PnpAprilTagObserver(
            TagConfig(family=family,ids=self.marker_ids,size_m=tag_size_m,calibration_version=calibration_version),
            K,D,camera_pose,base_tag,quality_min=quality_min,
            max_reprojection_px=max_reprojection_px,
            max_planar_tilt_deg=max_planar_tilt_deg,min_side_px=min_side_px)
        self.fusion=PlanarFusion()
        self.frames=0; self.detections=0; self.accepted=0; self.last_latency_ms=None; self.last_observation=None

    def process(self, image, capture_time_ns, receive_time_ns=None, frame_seq=None):
        receive_time_ns=time.monotonic_ns() if receive_time_ns is None else receive_time_ns
        started=time.monotonic_ns(); self.frames+=1
        hits=self.detector.detect(np.asarray(image)); self.detections+=len(hits)
        accepted=[]
        for hit in hits:
            d=Detection(self.camera_id,self.frames if frame_seq is None else frame_seq,hit.tag_id,hit.corners,
                int(capture_time_ns),int(receive_time_ns),time.monotonic_ns())
            obs=self.observer.observe(d)
            if obs is not None and self.fusion.update(obs):
                accepted.append(obs); self.last_observation=obs; self.accepted+=1
        self.last_latency_ms=(time.monotonic_ns()-started)/1e6
        return accepted

    def reconfigure_calibration(self, K, D, version, camera_pose, base_tag):
        """Atomically replace observer calibration and reset the fusion session."""
        replacement=PnpAprilTagObserver(TagConfig(family=self.family,ids=self.marker_ids,
            size_m=self.tag_size_m,calibration_version=version),K,D,
            camera_pose,base_tag,**self._observer_options)
        self.observer=replacement; self.fusion.reset(); self.last_observation=None

    def publish(self, stamp_ns):
        item=self.fusion.publish(int(stamp_ns))
        status=self.fusion.status(int(stamp_ns))
        return item, status


class MultiCameraImagePipeline:
    """Independent six-camera image pipelines with conservative source choice."""
    def __init__(self, cameras, base_tag, calibration_version, detector_scale=1.0,
                 family="tag36h11", detector_profile="coverage", marker_ids=(0,1)):
        self.cameras={}
        for camera_id, spec in cameras.items():
            self.cameras[camera_id]=OneCameraImagePipeline(camera_id,spec["K"],spec.get("D",[0]*5),
                {"rotation":spec["R_world_optical"],"translation":spec["position_world"]},base_tag,calibration_version,
                detector_scale=detector_scale,family=family,detector_profile=detector_profile,
                marker_ids=marker_ids)
        self.selector=ObservationSelector(); self.disabled=set()

    def disable(self, camera_id):
        if camera_id not in self.cameras: raise KeyError(camera_id)
        self.disabled.add(camera_id)

    def enable(self, camera_id):
        if camera_id not in self.cameras: raise KeyError(camera_id)
        self.disabled.discard(camera_id)

    def reconfigure_camera(self, camera_id, K, D, version, camera_pose, base_tag):
        if camera_id not in self.cameras: raise KeyError(camera_id)
        self.cameras[camera_id].reconfigure_calibration(K,D,version,camera_pose,base_tag)

    def process(self, camera_id, image, capture_time_ns, receive_time_ns=None, frame_seq=None):
        if camera_id in self.disabled: return []
        return self.cameras[camera_id].process(image,capture_time_ns,receive_time_ns,frame_seq)

    def publish(self, stamp_ns):
        results={cid:self.cameras[cid].publish(stamp_ns) for cid in sorted(self.cameras) if cid not in self.disabled}
        observations=[self.cameras[cid].last_observation for cid in results if self.cameras[cid].last_observation is not None and results[cid][1].valid]
        selected=self.selector.select(observations) if observations else None
        return {"cameras":results,"selected_camera":selected.camera_id if selected else None,
                "hardware_verified":False,"disabled":sorted(self.disabled)}
