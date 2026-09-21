"""Simulation/replay opponent observation and identity-aware tracker boundary."""
from dataclasses import dataclass
import math
@dataclass(frozen=True)
class Detection2D:
    camera_id:str; frame_seq:int; stamp_ns:int; bbox:tuple; confidence:float; height_m:float|None=None
@dataclass(frozen=True)
class OpponentState:
    stamp_ns:int; x:float; y:float; vx:float; vy:float; confirmed:bool; yaw_valid:bool=False; lost:bool=False
class OpponentTracker:
    def __init__(self, publish_hz=15, timeout_ms=300, min_confidence=.3): self.timeout=timeout_ms*1_000_000; self.min_confidence=min_confidence; self.state=None; self.updates=0; self.id_switches=0; self._identity="opponent"
    def update(self,d:Detection2D):
        if d.confidence < self.min_confidence or len(d.bbox)!=4: return False
        x1,y1,x2,y2=d.bbox; x=(x1+x2)/2; y=y2
        if self.state is not None:
            dt=max((d.stamp_ns-self.state.stamp_ns)/1e9,1e-6)
            self.state=OpponentState(d.stamp_ns,x,y,(x-self.state.x)/dt,(y-self.state.y)/dt,True)
        else: self.state=OpponentState(d.stamp_ns,x,y,0.,0.,False)
        self.updates+=1; return True
    def publish(self,stamp_ns):
        if self.state is None: return None
        lost=stamp_ns-self.state.stamp_ns>self.timeout
        return OpponentState(stamp_ns,self.state.x,self.state.y,self.state.vx,self.state.vy,self.state.confirmed,lost=lost)

class MultiCameraAssociator:
    """Associate one known opponent across cameras without creating duplicate tracks."""
    def __init__(self, max_time_skew_ns=100_000_000, max_pixel_jump=160):
        self.max_time_skew_ns=max_time_skew_ns; self.max_pixel_jump=max_pixel_jump; self.last=None; self.id_switches=0
    def select(self, detections):
        valid=[d for d in detections if d.confidence >= .3 and len(d.bbox)==4]
        if not valid: return None
        valid.sort(key=lambda d:(d.stamp_ns,-d.confidence))
        newest=valid[-1]
        if self.last is not None:
            if newest.stamp_ns-self.last.stamp_ns > self.max_time_skew_ns: return None
            cx=lambda d:(d.bbox[0]+d.bbox[2])/2
            if abs(cx(newest)-cx(self.last))>self.max_pixel_jump: return None
        # Highest confidence among detections near the newest timestamp.
        candidates=[d for d in valid if abs(d.stamp_ns-newest.stamp_ns)<=self.max_time_skew_ns]
        chosen=max(candidates,key=lambda d:d.confidence)
        if self.last is not None and chosen.camera_id != self.last.camera_id: self.id_switches += 1
        self.last=chosen
        return chosen
