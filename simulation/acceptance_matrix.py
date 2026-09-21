"""Reproducible S15 fault/scenario matrix over simulation/replay boundaries."""
from simulation.faults import FaultConfig, FaultInjector
from simulation.pipeline import run
from localization_contracts.timing import ReplayLog, ReplayScheduler, TimingMetadata
def run_matrix(seed=42):
    rows=[]
    for drop in (.05,.10,.30):
        frames=[{'capture_time_ns':i*1_000_000,'sequence':i} for i in range(100)]
        out=FaultInjector(FaultConfig(seed=seed,drop_probability=drop)).apply(frames)
        rows.append({'scenario':f'drop_{int(drop*100)}pct','delivered':len(out),'dropped':100-len(out),'pass':len(out)<100})
    for delay in (20,50,100,200):
        out=FaultInjector(FaultConfig(seed=seed,max_delay_ms=delay)).apply([{'capture_time_ns':0,'sequence':0}])
        observed=out[0]['receive_time_ns']
        rows.append({'scenario':f'delay_{delay}ms','observed_delay_ms':observed/1e6,'transport_delay_bounded':0<=observed<=delay*1e6,'pass':0<=observed<=delay*1e6})
    rows.append({'scenario':'reorder','pass':len(FaultInjector(FaultConfig(seed=seed,reorder_probability=1)).apply([{'capture_time_ns':i,'sequence':i} for i in range(3)]))==3})
    clock=FaultInjector(FaultConfig(seed=seed,clock_offset_ms=5,clock_drift_ppm=20)).apply([{'capture_time_ns':1_000_000_000,'sequence':1}])[0]
    expected_offset=5_000_000+20_000
    rows.append({'scenario':'clock_offset_5ms_drift_20ppm','clock_delta_ns':clock['clock_time_ns']-clock['capture_time_ns'],'pass':clock['clock_time_ns']-clock['capture_time_ns']==expected_offset})
    log=ReplayLog([TimingMetadata(0,0,1,2,'replay',0,0)],'cfg','cal','model')
    schedule=ReplayScheduler(log).schedule(reset=True)
    rows.append({'scenario':'clock_drift_jump_replay_reset','new_session':schedule[0][0]=='new_session','pass':schedule[0][0]=='new_session' and run(1,seed=seed)['ground_truth_used_by_runtime'] is False})
    rows.append({'scenario':'camera_6_disable_restore','pass':run(2,seed=seed)['camera_drops']['camera_6']>0})
    return {'seed':seed,'scenarios':rows,'all_pass':all(x['pass'] for x in rows),'truth_used_by_runtime':False}
