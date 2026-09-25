"""Simulation/replay opponent observation and identity-aware tracker boundary."""
from dataclasses import dataclass
import math
def project_contact_point(det, fx, fy, cx, cy, camera_origin, rotation, ground_z=0.0, height_uncertainty_m=0.0):
    """Project bbox bottom-center to horizontal plane; returns point and uncertainty."""
    x1,y1,x2,y2=det.bbox; u=(x1+x2)/2; v=y2
    ray=((u-cx)/fx,(v-cy)/fy,1.0)
    direction=tuple(sum(rotation[r][k]*ray[k] for k in range(3)) for r in range(3))
    if abs(direction[2])<1e-9: raise ValueError('contact ray parallel to ground')
    scale=(ground_z-camera_origin[2])/direction[2]
    if scale<=0: raise ValueError('contact plane is behind camera')
    return {'position_m':tuple(camera_origin[k]+scale*direction[k] for k in range(3)),'height_uncertainty_m':height_uncertainty_m,'source':'bbox_contact_point'}
@dataclass(frozen=True)
class Detection2D:
    camera_id:str; frame_seq:int; stamp_ns:int; bbox:tuple; confidence:float; height_m:float|None=None; position_m:tuple|None=None; identity:str|None=None
@dataclass(frozen=True)
class OpponentState:
    stamp_ns:int; x:float; y:float; vx:float; vy:float; confirmed:bool; yaw_valid:bool=False; lost:bool=False
class OpponentTracker:
    def __init__(self, publish_hz=15, timeout_ms=300, min_confidence=.3): self.timeout=timeout_ms*1_000_000; self.min_confidence=min_confidence; self.state=None; self.updates=0; self.id_switches=0; self._identity="opponent"; self._last_stamp=-1
    def update(self,d:Detection2D):
        if d.confidence < self.min_confidence or len(d.bbox)!=4 or d.stamp_ns < self._last_stamp: return False
        if d.position_m is None:
            # Compatibility fallback for unit/replay fixtures; runtime should
            # provide a projected metric contact point.
            x1,y1,x2,y2=d.bbox; x=(x1+x2)/2; y=y2
        else:
            if len(d.position_m) < 2: return False
            x,y=float(d.position_m[0]),float(d.position_m[1])
        self._last_stamp=d.stamp_ns
        if self.state is not None:
            dt=(d.stamp_ns-self.state.stamp_ns)/1e9
            if dt <= 0: return False
            self.state=OpponentState(d.stamp_ns,x,y,(x-self.state.x)/dt,(y-self.state.y)/dt,True)
        else: self.state=OpponentState(d.stamp_ns,x,y,0.,0.,False)
        self.updates+=1; return True
    def publish(self,stamp_ns):
        if self.state is None: return None
        lost=stamp_ns-self.state.stamp_ns>self.timeout
        return OpponentState(stamp_ns,self.state.x,self.state.y,self.state.vx,self.state.vy,self.state.confirmed,lost=lost)

class MultiCameraAssociator:
    """Associate one known opponent across cameras without creating duplicate tracks."""
    def __init__(self, max_time_skew_ns=100_000_000, max_pixel_jump=160, max_metric_jump_m=2.0):
        self.max_time_skew_ns=max_time_skew_ns; self.max_pixel_jump=max_pixel_jump; self.max_metric_jump_m=max_metric_jump_m; self.last=None; self.id_switches=0
    def select(self, detections):
        valid=[d for d in detections if d.confidence >= .3 and len(d.bbox)==4]
        if not valid: return None
        valid.sort(key=lambda d:(d.stamp_ns,-d.confidence))
        newest=valid[-1]
        if self.last is not None:
            if newest.stamp_ns < self.last.stamp_ns: return None
            if newest.position_m is not None and self.last.position_m is not None:
                jump=math.hypot(newest.position_m[0]-self.last.position_m[0],newest.position_m[1]-self.last.position_m[1])
                if jump>self.max_metric_jump_m: return None
            elif newest.camera_id == self.last.camera_id:
                cx=lambda d:(d.bbox[0]+d.bbox[2])/2
                if abs(cx(newest)-cx(self.last))>self.max_pixel_jump: return None
        # Highest confidence among detections near the newest timestamp.
        candidates=[d for d in valid if abs(d.stamp_ns-newest.stamp_ns)<=self.max_time_skew_ns]
        chosen=max(candidates,key=lambda d:d.confidence)
        if self.last is not None and chosen.identity and self.last.identity and chosen.identity != self.last.identity: self.id_switches += 1
        self.last=chosen
        return chosen
