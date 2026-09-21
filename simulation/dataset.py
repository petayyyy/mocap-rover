"""Reproducible simulation dataset manifest; truth is label/evaluation only."""
import hashlib, json
from pathlib import Path
def make_manifest(records, seed=42):
    ordered=sorted(records,key=lambda r:(r['session'],r['frame']))
    sessions=sorted({r['session'] for r in ordered}); split={s:('test' if i%5==0 else 'val' if i%5==1 else 'train') for i,s in enumerate(sessions)}
    return {'schema':'sim-dataset-1','seed':seed,'splits':[{**r,'split':split[r['session']]} for r in ordered], 'truth_role':'labels_and_evaluation_only'}
def save_manifest(path, data): path.write_text(json.dumps(data,sort_keys=True,indent=2),encoding='utf-8')
def digest(data): return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
