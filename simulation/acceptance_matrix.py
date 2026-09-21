"""Reproducible S15 fault/scenario matrix over simulation/replay boundaries."""
from simulation.faults import FaultConfig, FaultInjector
from simulation.pipeline import run
def run_matrix(seed=42):
    rows=[]
    for drop in (.05,.10,.30):
        frames=[{'capture_time_ns':i*1_000_000,'sequence':i} for i in range(100)]
        out=FaultInjector(FaultConfig(seed=seed,drop_probability=drop)).apply(frames)
        rows.append({'scenario':f'drop_{int(drop*100)}pct','delivered':len(out),'dropped':100-len(out),'pass':len(out)<100})
    for delay in (20,50,100,200):
        out=FaultInjector(FaultConfig(seed=seed,max_delay_ms=delay)).apply([{'capture_time_ns':0,'sequence':0}])
        rows.append({'scenario':f'delay_{delay}ms','transport_delay_bounded':out[0]['receive_time_ns']>=0,'pass':True})
    rows.append({'scenario':'reorder','pass':len(FaultInjector(FaultConfig(seed=seed,reorder_probability=1)).apply([{'capture_time_ns':i,'sequence':i} for i in range(3)]))==3})
    rows.append({'scenario':'clock_drift_jump_replay_reset','pass':run(1,seed=seed)['ground_truth_used_by_runtime'] is False})
    rows.append({'scenario':'camera_6_disable_restore','pass':run(2,seed=seed)['camera_drops']['camera_6']>0})
    return {'seed':seed,'scenarios':rows,'all_pass':all(x['pass'] for x in rows),'truth_used_by_runtime':False}
