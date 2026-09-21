#!/usr/bin/env python3
"""Live six-camera Gazebo image/CameraInfo -> detector -> PnP smoke check."""
import argparse, json, os, sys, threading, time
from pathlib import Path
import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.msgs10.camera_info_pb2 import CameraInfo
from gz.transport13 import Node
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localization_contracts.image_pipeline import MultiCameraImagePipeline

def main(seconds=5.0, config_path='config/cameras.json'):
    config=json.loads(Path(config_path).read_text()); entries={c['name']:c for c in config['cameras']}
    pipeline=None; infos={}; ready=threading.Event(); lock=threading.Lock(); stats={cid:{'frames':0,'detections':0,'accepted':0,'latency_ms':[]} for cid in entries}
    node=Node(); start=time.monotonic()
    def info_cb(msg,cid):
        with lock:
            infos[cid]=(list(msg.intrinsics.k),list(msg.distortion.k))
            if len(infos)==len(entries): ready.set()
    def image_cb(msg,cid):
        nonlocal pipeline
        if msg.pixel_format_type != 3 or pipeline is None: return
        t0=time.monotonic_ns(); raw=b''.join(msg.data[r*msg.step:r*msg.step+msg.width*3] for r in range(msg.height)); image=np.frombuffer(raw,dtype=np.uint8).reshape(msg.height,msg.width,3)
        stamp=int(msg.header.stamp.sec)*1_000_000_000+int(msg.header.stamp.nsec)
        with lock:
            stats[cid]['frames']+=1
            before=pipeline.cameras[cid].accepted; detected_before=pipeline.cameras[cid].detections
            pipeline.process(cid,image,stamp,frame_seq=stats[cid]['frames'])
            stats[cid]['accepted']+=pipeline.cameras[cid].accepted-before; stats[cid]['detections']+=pipeline.cameras[cid].detections-detected_before; stats[cid]['latency_ms'].append((time.monotonic_ns()-t0)/1e6)
    topics=[]
    for cid in entries:
        it=f'/cameras/{cid}/camera_info'; im=f'/cameras/{cid}/image'; topics += [it,im]
        if not node.subscribe(CameraInfo,it,lambda msg,cid=cid:info_cb(msg,cid)): raise RuntimeError(it)
        if not node.subscribe(Image,im,lambda msg,cid=cid:image_cb(msg,cid)): raise RuntimeError(im)
    if not ready.wait(15): raise RuntimeError(f'CameraInfo missing: {sorted(set(entries)-set(infos))}')
    specs={}
    for cid,c in entries.items(): specs[cid]={'K':np.asarray(infos[cid][0]).reshape(3,3),'D':infos[cid][1],'R_world_optical':c['R_world_optical'],'position_world':c['position_world']}
    pipeline=MultiCameraImagePipeline(specs,{'rotation':config['tag']['R_base_tag'],'translation':config['tag']['T_base_tag_translation']},'gazebo-camera-info',detector_scale=.5)
    time.sleep(seconds)
    for topic in topics: node.unsubscribe(topic)
    elapsed=max(time.monotonic()-start,1e-9); out={}
    for cid,s in stats.items():
        lat=sorted(s['latency_ms']); out[cid]={**s,'wall_fps':s['frames']/elapsed,'accepted_hz':s['accepted']/elapsed,'latency_ms_p95':lat[min(len(lat)-1,int(len(lat)*.95))] if lat else None}
        del out[cid]['latency_ms']
    print(json.dumps({'cameras':out,'wall_seconds':elapsed,'ground_truth_used_by_runtime':False,'hardware_verified':False},sort_keys=True),flush=True); os._exit(0)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--seconds',type=float,default=5.0); parser.add_argument('--config',default='config/cameras.json'); args=parser.parse_args(); main(args.seconds,args.config)
