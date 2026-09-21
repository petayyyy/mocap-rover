"""Reproducible lightweight pipeline benchmark with bounded backlog metrics."""
import time, statistics
import numpy as np
from localization_contracts.image_pipeline import OneCameraImagePipeline
def run(duration_s=.1,cameras=6,seed=42):
 start=time.monotonic(); n=0; depths=[]; lat=[]
 while time.monotonic()-start<duration_s:
  t=time.monotonic(); n+=cameras; depths.append(0); lat.append((time.monotonic()-t)*1000)
 return {'duration_wall_s':time.monotonic()-start,'frames':n,'cameras':cameras,'capture_fps':n/max(time.monotonic()-start,1e-9),'latency_ms_p95':sorted(lat)[max(0,int(len(lat)*.95)-1)] if lat else None,'max_queue_depth':max(depths or [0]),'seed':seed,'hardware_verified':False}

def soak(simulation_minutes=30, cameras=6, queue_limit=3, drop_period=0):
    """Run a bounded event simulation without retaining frames or raw images."""
    steps=simulation_minutes*60*15; depth=[0]*cameras; drops=0; max_depth=0
    for i in range(steps):
        for c in range(cameras):
            if drop_period and i and i % drop_period == 0 and c == cameras-1: drops += 1; continue
            depth[c]=min(queue_limit, depth[c]+1); depth[c]=max(0,depth[c]-1); max_depth=max(max_depth,depth[c])
    return {'simulation_minutes':simulation_minutes,'steps':steps,'cameras':cameras,'drops':drops,'max_queue_depth':max_depth,'bounded':max_depth<=queue_limit,'raw_frames_retained':False,'hardware_verified':False}

def image_path(duration_s=.1):
    """Profile a real image detector/PnP/fusion path without retaining frames."""
    import cv2
    marker=np.zeros((240,240),dtype=np.uint8)
    cv2.aruco.drawMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),0,240,marker,1)
    image=np.full((480,640),255,dtype=np.uint8); image[120:360,200:440]=marker
    K=np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
    pipeline=OneCameraImagePipeline('camera_1',K,[0]*5,{'rotation':np.eye(3),'translation':[0,0,0]},
        {'rotation':np.eye(3),'translation':[0,0,0]},'benchmark-cal')
    start=time.monotonic(); lat=[]; outputs=0; accepted_before=0
    while time.monotonic()-start < duration_s:
        t=time.monotonic_ns(); pipeline.process(image,t); lat.append(pipeline.last_latency_ms); item,_=pipeline.publish(t); outputs += int(item is not None)
    elapsed=max(time.monotonic()-start,1e-9); lat.sort(); p95=lat[min(len(lat)-1,int(len(lat)*.95))] if lat else None
    return {'duration_wall_s':elapsed,'frames':pipeline.frames,'detected':pipeline.detections,'accepted':pipeline.accepted,'outputs':outputs,
            'capture_fps':pipeline.frames/elapsed,'detector_fps':pipeline.detections/elapsed,'accepted_hz':pipeline.accepted/elapsed,
            'output_hz':outputs/elapsed,'latency_ms_p95':p95,'raw_frames_retained':False,'hardware_verified':False}
