"""Small asynchronous planar fusion filter for simulation/replay."""
from __future__ import annotations
from dataclasses import dataclass
import math
from .contracts import TrackStatus, SCHEMA_VERSION

def angle_diff(a,b): return (a-b+math.pi)%(2*math.pi)-math.pi
@dataclass(frozen=True)
class State:
    stamp_ns:int; x:float; y:float; yaw:float; vx:float=0.; vy:float=0.; omega:float=0.; covariance:float=1.

class PlanarFusion:
    def __init__(self, object_id="tag_rover", publish_hz=30, dropout_horizon_ms=200, gate_sigma=5.0):
        self.object_id=object_id; self.publish_hz=publish_hz; self.horizon=dropout_horizon_ms*1_000_000; self.gate_sigma=gate_sigma; self.state=None; self.last_measurement=None; self.measurements=0; self.outputs=0; self.rejected=0; self.session=0; self.calibration_version=None; self.session_start_ns=None; self.measurement_times=[]; self.output_times=[]
    def reset(self): self.state=None; self.last_measurement=None; self.measurements=0; self.outputs=0; self.rejected=0; self.session+=1; self.session_start_ns=None; self.measurement_times=[]; self.output_times=[]
    def update(self, obs):
        obs.validate()
        if self.calibration_version is not None and obs.calibration_version != self.calibration_version:
            self.reset()
        self.calibration_version=obs.calibration_version
        if self.last_measurement and obs.measurement_id == self.last_measurement.measurement_id: return False
        if self.last_measurement and obs.capture_time_ns < self.last_measurement.capture_time_ns: return False
        z=obs.position_m; yaw=float(obs.pixel_features.get("yaw_rad",0))
        if self.session_start_ns is None: self.session_start_ns=obs.capture_time_ns
        if self.state is None: self.state=State(obs.capture_time_ns,z[0],z[1],yaw,covariance=.04)
        else:
            delta_ns=obs.capture_time_ns-self.state.stamp_ns
            dt=max(delta_ns/1e9, 0.)
            predicted_x=self.state.x+self.state.vx*dt; predicted_y=self.state.y+self.state.vy*dt
            measurement_var=max(float(obs.covariance_m2[0]), float(obs.covariance_m2[1]) if len(obs.covariance_m2)>1 else float(obs.covariance_m2[0]), 1e-9)
            innovation=math.hypot(z[0]-predicted_x, z[1]-predicted_y)
            threshold=self.gate_sigma*math.sqrt(self.state.covariance+measurement_var+dt*.1)
            if innovation > threshold:
                self.rejected += 1
                return False
            # Simultaneous observations from different cameras are valid
            # measurements, but must not manufacture an enormous velocity.
            if delta_ns <= 0:
                self.state=State(self.state.stamp_ns,z[0],z[1],yaw,self.state.vx,self.state.vy,self.state.omega,measurement_var)
            else:
                dt=delta_ns/1e9
                self.state=State(obs.capture_time_ns,z[0],z[1],yaw,(z[0]-self.state.x)/dt,(z[1]-self.state.y)/dt,angle_diff(yaw,self.state.yaw)/dt,measurement_var)
        self.last_measurement=obs; self.measurements+=1; self.measurement_times.append(obs.capture_time_ns); return True
    def publish(self, stamp_ns):
        self.outputs+=1; self.output_times.append(stamp_ns)
        if self.state is None: return None
        dt=max(0,(stamp_ns-self.state.stamp_ns)/1e9); age=stamp_ns-self.state.stamp_ns
        if age>self.horizon: valid=False; tracking="LOST"
        elif age: valid=True; tracking="COASTING"
        else: valid=True; tracking="TRACKING"
        start=self.session_start_ns if self.session_start_ns is not None else stamp_ns
        elapsed=max((stamp_ns-start)/1e9, 1e-9)
        return {"state": State(stamp_ns,self.state.x+self.state.vx*dt,self.state.y+self.state.vy*dt,self.state.yaw+self.state.omega*dt,self.state.vx,self.state.vy,self.state.omega,self.state.covariance+dt*.1), "valid":valid,"tracking_state":tracking,"measurement_hz":self.measurements/elapsed,"output_hz":self.outputs/elapsed,"measurement_age_ms":age/1e6}

    def status(self, stamp_ns):
        item=self.publish(stamp_ns)
        if item is None:
            state='INITIALIZING'; valid=False; age=None; last=None
        else:
            state=item['tracking_state']; valid=item['valid']; age=item['measurement_age_ms']; last=self.last_measurement.capture_time_ns
        return TrackStatus(SCHEMA_VERSION,self.object_id,valid,state,last,age,tuple([self.last_measurement.camera_id] if self.last_measurement else []),float(self.measurements),float(self.outputs),bool(item and item['state'].yaw is not None),self.calibration_version or 'unset',str(self.session),self.session).validate()

class ObservationSelector:
    """Select one camera observation with quality hysteresis."""
    def __init__(self, switch_margin=.15): self.switch_margin=switch_margin; self.active_camera=None
    def select(self, observations):
        valid=[o for o in observations if o.quality >= 0]
        if not valid:return None
        best=max(valid,key=lambda o:o.quality)
        current=next((o for o in valid if o.camera_id==self.active_camera),None)
        if current is None or best.quality >= current.quality+self.switch_margin: self.active_camera=best.camera_id; return best
        return current
