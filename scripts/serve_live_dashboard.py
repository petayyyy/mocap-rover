#!/usr/bin/env python3
"""Serve the simulation dashboard from live Gazebo image topics."""
import argparse, sys, threading, time
from pathlib import Path
import json
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localization_contracts.dashboard import DashboardHandler, serve
from localization_contracts.registry import CameraRegistry, PreviewMetadata

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('/tmp/mocap-live-dashboard')); p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=8080); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    registry=CameraRegistry.virtual_default(); node=Node(); lock=threading.Lock(); counts={cid:0 for cid in registry.bindings}; started=time.monotonic()
    def callback(msg,cid):
        if msg.pixel_format_type!=3: return
        raw=b''.join(msg.data[r*msg.step:r*msg.step+msg.width*3] for r in range(msg.height)); path=a.output/f'{cid}.ppm'; path.write_bytes(f'P6\n{msg.width} {msg.height}\n255\n'.encode()+raw)
        with lock: counts[cid]+=1
        registry.push(cid,'image',(counts[cid],time.monotonic_ns()))
    for cid in sorted(registry.bindings):
        if not node.subscribe(Image,f'/cameras/{cid}/image',lambda msg,cid=cid:callback(msg,cid)): raise RuntimeError(cid)
    DashboardHandler.status_provider=staticmethod(lambda:{'phase':'SIMULATION','hardware_verified':False,'live':True,'wall_seconds':time.monotonic()-started,'frames':dict(counts)})
    DashboardHandler.cameras_provider=staticmethod(lambda:[registry.status(cid) for cid in sorted(registry.bindings)])
    DashboardHandler.previews_provider=staticmethod(lambda:[registry.preview(cid,PreviewMetadata(1600,1200,available=(a.output/f'{cid}.ppm').is_file())) for cid in sorted(registry.bindings)])
    DashboardHandler.preview_files_provider=staticmethod(lambda:{cid:str(a.output/f'{cid}.ppm') for cid in registry.bindings})
    print(f'http://{a.host}:{a.port}/',flush=True); serve(a.host,a.port).serve_forever()
if __name__=='__main__': main()
