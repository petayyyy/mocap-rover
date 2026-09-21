"""One-camera simulation image pipeline using shared localization contracts."""
from __future__ import annotations

import time
import numpy as np

from .apriltag import Detection, PnpAprilTagObserver, TagConfig
from .detector import AprilTagImageDetector
from .fusion import PlanarFusion


class OneCameraImagePipeline:
    """Detector -> PnP -> metric fusion boundary.

    CameraInfo is supplied at construction; evaluator/world pose data is not
    accepted by this runtime object.  ``process`` is capture-driven while
    ``publish`` is deliberately a separate caller/timer boundary.
    """
    def __init__(self, camera_id, K, D, camera_pose, base_tag, calibration_version,
                 family="tag36h11", tag_size_m=.4):
        self.camera_id=camera_id
        self.detector=AprilTagImageDetector(family)
        self.observer=PnpAprilTagObserver(TagConfig(family=family,size_m=tag_size_m,calibration_version=calibration_version),K,D,camera_pose,base_tag)
        self.fusion=PlanarFusion()
        self.frames=0; self.detections=0; self.accepted=0; self.last_latency_ms=None

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
                accepted.append(obs); self.accepted+=1
        self.last_latency_ms=(time.monotonic_ns()-started)/1e6
        return accepted

    def publish(self, stamp_ns):
        item=self.fusion.publish(int(stamp_ns))
        status=self.fusion.status(int(stamp_ns))
        return item, status
