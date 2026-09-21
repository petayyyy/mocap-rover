"""Reproducible simulation dataset manifest; truth is label/evaluation only."""
import hashlib, json
from pathlib import Path
import numpy as np
def make_manifest(records, seed=42):
    ordered=sorted(records,key=lambda r:(r['session'],r['frame']))
    sessions=sorted({r['session'] for r in ordered}); split={s:('test' if i%5==0 else 'val' if i%5==1 else 'train') for i,s in enumerate(sessions)}
    return {'schema':'sim-dataset-1','seed':seed,'splits':[{**r,'split':split[r['session']]} for r in ordered], 'truth_role':'labels_and_evaluation_only'}
def save_manifest(path, data): path.write_text(json.dumps(data,sort_keys=True,indent=2),encoding='utf-8')
def digest(data): return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

def build_image_dataset(root, sessions, width=640, height=480, frames_per_session=4):
    """Create deterministic PNG/YOLO-label simulation samples.

    The renderer is deliberately simple and dependency-light; its boxes are
    labels, never runtime detections.  A whole session is assigned one split
    to prevent adjacent-frame leakage.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("image dataset generation requires OpenCV") from exc
    root=Path(root); records=[]
    for session in sorted(sessions, key=lambda x:(x['seed'],x['trajectory'])):
        key=f"{session['seed']}-{session['trajectory']}"
        digest_key=hashlib.sha256(key.encode()).hexdigest()[:10]
        split=('test' if int(session['seed']) % 5 == 0 else 'val' if int(session['seed']) % 5 == 1 else 'train')
        for frame in range(int(frames_per_session)):
            image=np.full((height,width,3), 24, dtype=np.uint8)
            x=int((frame+1)*width/(frames_per_session+1)); y=int(height*.55)
            cv2.rectangle(image,(x-35,y-22),(x+35,y+22),(60,130,210),-1)
            rel=f"images/{split}/{digest_key}_{frame:06d}.png"; label=rel.replace('images/','labels/').rsplit('.',1)[0]+'.txt'
            (root/rel).parent.mkdir(parents=True,exist_ok=True); (root/label).parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(root/rel),image): raise RuntimeError(f"failed to write {rel}")
            (root/label).write_text(f"0 {x/width:.8f} {y/height:.8f} {70/width:.8f} {44/height:.8f}\n",encoding='utf-8')
            records.append({'session':key,'seed':session['seed'],'trajectory':session['trajectory'],'frame':frame,'image':rel,'label':label,'split':split,'truth_role':'label_only'})
    manifest={'schema':'sim-image-dataset-1','width':width,'height':height,'records':records,'truth_role':'labels_and_evaluation_only','weights':None,'training_status':'not_run'}
    manifest['digest']=digest(manifest)
    save_manifest(root/'manifest.json',manifest)
    return manifest
