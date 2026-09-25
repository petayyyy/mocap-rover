#!/usr/bin/env bash
# Stop this repository's workers and Gazebo process tree, including stranded children.
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec /usr/bin/python3 - "$root_dir" "${1:-}" <<'PY'
import os, signal, sys, time
from pathlib import Path
root=Path(sys.argv[1]); dry=sys.argv[2]=='--dry-run'; own=os.getpid()
processes={}
for path in Path('/proc').iterdir():
    if not path.name.isdigit() or int(path.name)==own: continue
    try:
        if path.stat().st_uid!=os.getuid(): continue
        args=path.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
        status=path.joinpath('stat').read_text().rsplit(')',1)[1].split()
        cwd=path.joinpath('cwd').resolve()
        processes[int(path.name)]=(int(status[1]),args,cwd)
    except (OSError,ValueError): continue
selected=set()
workers={'run_localization.py','record_gazebo_truth.py','calibrate_gazebo.py','capture_yolo_dataset.py','serve_live_dashboard.py','serve_dashboard.py','check_all_image_pipeline.py','check_image_pipeline.py','check_sim.py','evaluate_live_pose.py'}
for pid,(_,args,cwd) in processes.items():
    command=' '.join(args).strip()
    python=bool(args and 'python' in Path(args[0]).name)
    script=any(Path(arg).name in workers for arg in args[1:3])
    local_venv=bool(args and (str(root/'.venv') in args[0] or (cwd==root and args[0].startswith('.venv/'))))
    gazebo=command.startswith('gz sim') and (str(root) in command or cwd==root)
    if gazebo or python and cwd==root and (script or local_venv): selected.add(pid)
while True:
    children={pid for pid,(parent,_,_) in processes.items() if parent in selected}
    if children<=selected: break
    selected|=children
for pid in sorted(selected): print(pid,' '.join(processes[pid][1])[:200])
if dry: raise SystemExit(0)
for pid in selected:
    try: os.kill(pid,signal.SIGTERM)
    except ProcessLookupError: pass
for _ in range(50):
    if not any(Path(f'/proc/{pid}').exists() for pid in selected): break
    time.sleep(.1)
for pid in selected:
    try: os.kill(pid,signal.SIGKILL)
    except ProcessLookupError: pass
print('Stopped project processes:',len(selected))
PY
