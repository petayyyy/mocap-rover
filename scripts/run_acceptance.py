#!/usr/bin/env python3
"""Write reproducible simulation acceptance evidence (no physical devices)."""
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localization_contracts.config import ConfigStore, load_config
from simulation.acceptance_matrix import run_matrix
from simulation.pipeline import evaluate_samples, run
from localization_contracts.timing import ReplayLog, TimingMetadata
def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='/tmp/mocap-s15-acceptance.json'); p.add_argument('--recording',default=''); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    config=load_config('config/contracts.json'); pipe=run(2,a.seed,return_samples=True)
    calibration_path=Path('config/cameras.json'); calibration_hash=hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    world_path=Path('worlds/mocap_arena.sdf'); world_hash=hashlib.sha256(world_path.read_bytes()).hexdigest()
    try: git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception: git_revision='unknown'
    recording=a.recording or str(Path(a.output).with_suffix('.replay.json'))
    frames=[TimingMetadata(i, i*33_333_333, i*33_333_333, i*33_333_333+1, 'sim', 0, 0).validate() for i in range(0, min(60,pipe['friendly_output']))]
    ReplayLog(frames,ConfigStore.digest(config),calibration_hash,"untrained-sim-model").dump(recording)
    sweep=[]
    for sweep_seed in (42,7,123):
        s=run(2,sweep_seed); sweep.append({'seed':sweep_seed,'friendly_hz':s['friendly_hz'],'opponent_hz':s['opponent_hz'],'camera_6_drops':s['camera_drops']['camera_6']})
    report={'schema':'sim-acceptance-1','seed':a.seed,'seed_sweep':sweep,'python':sys.version.split()[0],'platform':platform.platform(),'git_revision':git_revision,'world_path':str(world_path),'world_digest':world_hash,'config_digest':ConfigStore.digest(config),'calibration_path':str(calibration_path),'calibration_digest':calibration_hash,'recording':recording,'recording_frames':len(frames),'hardware_verified':False,'simulation_only':True,'pipeline':{k:v for k,v in pipe.items() if k!='samples'},'evaluator':evaluate_samples(pipe),'fault_matrix':run_matrix(a.seed),'sim_accepted':False}
    Path(a.output).write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8'); print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__': main()
