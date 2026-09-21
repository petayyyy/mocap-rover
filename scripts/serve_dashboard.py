#!/usr/bin/env python3
"""Serve the simulation-only localhost dashboard over checked snapshot files."""
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localization_contracts.dashboard import DashboardHandler, serve
from localization_contracts.registry import CameraRegistry, PreviewMetadata
def main():
 p=argparse.ArgumentParser(); p.add_argument('--snapshot-dir',type=Path,default=Path('/tmp/mocap-camera-check')); p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=8080); a=p.parse_args()
 r=CameraRegistry.virtual_default(); DashboardHandler.status_provider=staticmethod(lambda:{'phase':'SIMULATION','hardware_verified':False,'snapshot_dir':str(a.snapshot_dir)})
 DashboardHandler.cameras_provider=staticmethod(lambda:[r.status(cid) for cid in sorted(r.bindings)])
 DashboardHandler.previews_provider=staticmethod(lambda:[r.preview(cid,PreviewMetadata(1600,1200,available=(a.snapshot_dir/f'{cid}.ppm').is_file())) for cid in sorted(r.bindings)])
 DashboardHandler.preview_files_provider=staticmethod(lambda:{cid:str(a.snapshot_dir/f'{cid}.ppm') for cid in r.bindings})
 print(f'http://{a.host}:{a.port}/'); serve(a.host,a.port).serve_forever()
if __name__=='__main__': main()
