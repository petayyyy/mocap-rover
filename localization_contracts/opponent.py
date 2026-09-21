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
