"""Deterministic synthetic end-to-end sim/replay pipeline for acceptance regressions."""
from localization_contracts.apriltag import AprilTagObserver, TagConfig, Detection
from localization_contracts.fusion import PlanarFusion
from localization_contracts.opponent import OpponentTracker, Detection2D
from simulation.faults import FaultConfig, FaultInjector

def run(duration_s=2.0, seed=42, drop_probability=0.0):
    steps=int(duration_s*30); observer=AprilTagObserver(TagConfig(calibration_version="sim-cal-1"),500,500,320,240)
    friendly=PlanarFusion(dropout_horizon_ms=200); opponent=OpponentTracker(); accepted=0; opponent_updates=0; outputs=0; lost=0
    frames=[{"capture_time_ns":i*33_333_333,"sequence":i} for i in range(steps)]
    delivered=FaultInjector(FaultConfig(seed=seed,drop_probability=drop_probability)).apply(frames)
    by_seq={f["sequence"]:f for f in delivered}
    for i in range(steps):
        stamp=i*33_333_333
        if i in by_seq:
            f=by_seq[i]; x=300+i*.4
            d=Detection("camera_1",i,0,((x,190),(x+100,190),(x+100,290),(x,290)),f["capture_time_ns"],f["receive_time_ns"],f["receive_time_ns"]+1)
            o=observer.observe(d)
            if o and friendly.update(o): accepted+=1
            if i%2==0:
                if opponent.update(Detection2D("camera_1",i,stamp,(x+20,200,x+80,280),.9)): opponent_updates+=1
        if friendly.publish(stamp): outputs+=1
        p=friendly.publish(stamp)
        if p and p["tracking_state"]=="LOST": lost+=1
    return {"duration_sim_s":duration_s,"friendly_measurements":accepted,"friendly_output":outputs,"opponent_measurements":opponent_updates,"opponent_hz":opponent_updates/max(duration_s,1e-9),"friendly_hz":accepted/max(duration_s,1e-9),"lost_publishes":lost,"dropped_frames":steps-len(delivered),"seed":seed,"ground_truth_used_by_runtime":False}
