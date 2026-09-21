"""Deterministic synthetic end-to-end sim/replay pipeline for acceptance regressions."""
from localization_contracts.apriltag import AprilTagObserver, TagConfig, Detection
from localization_contracts.fusion import PlanarFusion
from localization_contracts.opponent import OpponentTracker, Detection2D
from simulation.faults import FaultConfig, FaultInjector
from localization_contracts.registry import CameraRegistry

def run(duration_s=2.0, seed=42, drop_probability=0.0, return_samples=False):
    steps=int(duration_s*30); registry=CameraRegistry.virtual_default(); observer=AprilTagObserver(TagConfig(calibration_version="sim-cal-1"),500,500,320,240)
    friendly=PlanarFusion(dropout_horizon_ms=200); opponent=OpponentTracker(); accepted=0; opponent_updates=0; outputs=0; lost=0
    frames=[{"capture_time_ns":i*33_333_333,"sequence":i} for i in range(steps)]
    delivered=FaultInjector(FaultConfig(seed=seed,drop_probability=drop_probability)).apply(frames)
    by_seq={f["sequence"]:f for f in delivered}; sample_outputs=[]; camera_frames={cid:0 for cid in registry.bindings}; camera_drops={cid:0 for cid in registry.bindings}
    for i in range(steps):
        stamp=i*33_333_333
        if i in by_seq:
            for cid in registry.bindings:
                # Per-camera delivery is independent; camera_6 has a deterministic outage window.
                if cid == "camera_6" and 20 <= i < 30: camera_drops[cid]+=1; continue
                registry.push(cid,"image",(i,stamp)); camera_frames[cid]+=1
            f=by_seq[i]; x=300+i*.4
            d=Detection("camera_1",i,0,((x,190),(x+100,190),(x+100,290),(x,290)),f["capture_time_ns"],f["receive_time_ns"],f["receive_time_ns"]+1)
            o=observer.observe(d)
            if o and friendly.update(o): accepted+=1
            if i%2==0:
                if opponent.update(Detection2D("camera_1",i,stamp,(x+20,200,x+80,280),.9)): opponent_updates+=1
        p=friendly.publish(stamp); sample_outputs.append((stamp,p))
        if p: outputs+=1
        if p and p["tracking_state"]=="LOST": lost+=1
    result={"duration_sim_s":duration_s,"friendly_measurements":accepted,"friendly_output":outputs,"opponent_measurements":opponent_updates,"opponent_hz":opponent_updates/max(duration_s,1e-9),"friendly_hz":accepted/max(duration_s,1e-9),"lost_publishes":lost,"dropped_frames":steps-len(delivered),"camera_frames":camera_frames,"camera_drops":camera_drops,"registered_cameras":sorted(registry.bindings),"seed":seed,"ground_truth_used_by_runtime":False}
    if return_samples: result["samples"]=sample_outputs
    return result

def evaluate_samples(result):
    errors=[]; ages=[]
    for stamp, item in result.get("samples",[]):
        if not item: continue
        state=item["state"]; truth_x=.12+(stamp/33_333_333)*.0016; truth_y=0.0
        errors.append(((state.x-truth_x)**2+(state.y-truth_y)**2)**.5); ages.append(item["measurement_age_ms"])
    errors.sort(); ages.sort()
    at=lambda a,p: a[min(len(a)-1,int((len(a)-1)*p))] if a else None
    return {"matched":len(errors),"xy_p95_m":at(errors,.95),"age_p95_ms":at(ages,.95),"truth_used_only_here":True}
