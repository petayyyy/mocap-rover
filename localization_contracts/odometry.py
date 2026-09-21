"""ROS-independent odometry publication boundary for simulation/replay."""
from __future__ import annotations
from dataclasses import dataclass
from .contracts import FRAME_ARENA

@dataclass(frozen=True)
class OdometrySample:
    stamp_ns:int; frame_id:str; child_frame_id:str; state:object; valid:bool; tracking_state:str
    measurement_age_ms:float|None; measurement_hz:float; output_hz:float

class TimedOdometryPublisher:
    """Drive fusion publication from an independent fixed-rate simulation clock.

    A ROS adapter can map ``OdometrySample`` to nav_msgs/Odometry later; this
    class deliberately has no rclpy dependency and never changes capture time.
    """
    def __init__(self, fusion, publish_hz=30, frame_id=FRAME_ARENA, child_frame_id="base_link"):
        if publish_hz<=0: raise ValueError("publish_hz must be positive")
        self.fusion=fusion; self.period_ns=int(1e9/publish_hz); self.frame_id=frame_id; self.child_frame_id=child_frame_id
        self.next_stamp_ns=None; self.samples=[]

    def step(self, stamp_ns):
        stamp_ns=int(stamp_ns)
        if self.next_stamp_ns is None: self.next_stamp_ns=stamp_ns
        if stamp_ns < self.next_stamp_ns: return None
        latest=None
        while self.next_stamp_ns <= stamp_ns:
            tick=self.next_stamp_ns; item=self.fusion.publish(tick); self.next_stamp_ns += self.period_ns
            if item is None: continue
            latest=OdometrySample(tick,self.frame_id,self.child_frame_id,item["state"],item["valid"],item["tracking_state"],item["measurement_age_ms"],item["measurement_hz"],item["output_hz"])
            self.samples.append(latest)
        return latest
