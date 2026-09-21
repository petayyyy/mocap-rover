#!/usr/bin/env python3
"""Evaluator-only Gazebo image pose accuracy check.

World pose is subscribed solely by this evaluator and is never passed to the
image pipeline.  The runtime uses image corners, CameraInfo and calibration.
"""
import argparse, json, math, os, sys, threading, time
from pathlib import Path
import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.msgs10.camera_info_pb2 import CameraInfo
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localization_contracts.image_pipeline import OneCameraImagePipeline

def main(seconds=5., camera_id='camera_1', config_path='config/cameras.json'):
    cfg=json.loads(Path(config_path).read_text()); cam=next(c for c in cfg['cameras'] if c['name']==camera_id)
    info=threading.Event(); lock=threading.Lock(); pipeline=None; truth=None; errors=[]; frames=0
    node=Node()
    def info_cb(msg):
        nonlocal pipeline
        if len(msg.intrinsics.k)!=9: return
        pipeline=OneCameraImagePipeline(camera_id,np.asarray(list(msg.intrinsics.k)).reshape(3,3),list(msg.distortion.k),
            {'rotation':cam['R_world_optical'],'translation':cam['position_world']},
            {'rotation':cfg['tag']['R_base_tag'],'translation':cfg['tag']['T_base_tag_translation']},'gazebo-camera-info',detector_scale=.5)
        info.set()
    def pose_cb(msg):
        nonlocal truth
        for p in msg.pose:
            if p.name=='tag_rover':
                with lock: truth=(p.position.x,p.position.y,p.position.z)
    def image_cb(msg):
        nonlocal frames
        if pipeline is None or msg.pixel_format_type!=3: return
        raw=b''.join(msg.data[r*msg.step:r*msg.step+msg.width*3] for r in range(msg.height)); image=np.frombuffer(raw,dtype=np.uint8).reshape(msg.height,msg.width,3)
        stamp=int(msg.header.stamp.sec)*1_000_000_000+int(msg.header.stamp.nsec); accepted=pipeline.process(image,stamp)
        frames+=1
        with lock: ref=truth
        if ref is not None:
            for obs in accepted: errors.append(math.dist(obs.position_m,ref))
    topics=[f'/cameras/{camera_id}/camera_info',f'/cameras/{camera_id}/image', '/model/tag_rover/pose']
    if not node.subscribe(CameraInfo,topics[0],info_cb): raise RuntimeError(topics[0])
    if not node.subscribe(Image,topics[1],image_cb): raise RuntimeError(topics[1])
    if not node.subscribe(Pose_V,topics[2],pose_cb): raise RuntimeError(topics[2])
    if not info.wait(15): raise RuntimeError('CameraInfo was not received')
    time.sleep(seconds)
    for topic in topics: node.unsubscribe(topic)
    elapsed=max(time.monotonic(),1e-9); errors.sort(); p95=errors[min(len(errors)-1,int(.95*len(errors)))] if errors else None
    print(json.dumps({'frames':frames,'matched_observations':len(errors),'p50_3d_m':errors[len(errors)//2] if errors else None,'p95_3d_m':p95,'wall_seconds':seconds,'truth_used_only_by_evaluator':True,'hardware_verified':False},sort_keys=True),flush=True); os._exit(0)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seconds',type=float,default=5); p.add_argument('--camera',default='camera_1'); p.add_argument('--config',default='config/cameras.json'); a=p.parse_args(); main(a.seconds,a.camera,a.config)
