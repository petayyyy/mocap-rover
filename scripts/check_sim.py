#!/usr/bin/env python3
"""Live smoke check. Run with /usr/bin/python3 while the simulation is running."""
import argparse
import threading
import time
from pathlib import Path
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--move',action='store_true',help='Drive each rover briefly; use a fresh world with clear space')
parser.add_argument('--output',default='/tmp/mocap-camera-check')
parser.add_argument('--width',type=int,default=1440)
parser.add_argument('--height',type=int,default=1080)
parser.add_argument('--measure-seconds',type=float,default=0,help='measure wall FPS for each image topic')
args=parser.parse_args()
node=Node(); images={}; poses={}; ready=threading.Event(); callbacks=[]
def check():
    if len(images)==6 and len(poses)==2: ready.set()
for i in range(1,7):
    def callback(msg,i=i):
        images[i]=msg; check()
    callbacks.append(callback)
    assert node.subscribe(Image,f'/cameras/camera_{i}/image',callback)
for name in ('tag_rover','yolo_rover'):
    def callback(msg,name=name):
        for p in msg.pose:
            if p.name==name:
                poses[name]=(p.position.x,p.position.y,p.position.z); check()
    callbacks.append(callback)
    assert node.subscribe(Pose_V,f'/model/{name}/pose',callback)
assert ready.wait(60),f'Missing streams: images={list(images)}, poses={poses}'
out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
for i,msg in images.copy().items():
    assert (msg.width,msg.height)==(args.width,args.height), (msg.width,msg.height)
    # Gazebo RGB_INT8 = 3, raw RGB rows may contain padding.
    assert msg.pixel_format_type==3, msg.pixel_format_type
    data=msg.data; raw=data if msg.step==msg.width*3 else b''.join(data[r*msg.step:r*msg.step+msg.width*3] for r in range(msg.height))
    (out/f'camera_{i}.ppm').write_bytes(f'P6\n{msg.width} {msg.height}\n255\n'.encode()+raw)
print('Six RGB streams OK; world poses:',poses)
print('Snapshots:',out)
if args.measure_seconds:
    counts={i:0 for i in range(1,7)}; first=time.monotonic(); deadline=first+args.measure_seconds
    def count_callback(msg,i): counts[i]+=1
    measure_callbacks=[]
    for i in range(1,7):
        cb=lambda msg,i=i: count_callback(msg,i); measure_callbacks.append(cb); assert node.subscribe(Image,f'/cameras/camera_{i}/image',cb)
    while time.monotonic()<deadline: time.sleep(.02)
    elapsed=time.monotonic()-first
    print('wall_fps=',{i:round(n/elapsed,3) for i,n in counts.items()},'wall_seconds=',round(elapsed,3),'simulation_fps_requires_clock_log=true')
if args.move:
    for name in ('tag_rover','yolo_rover'):
        pub=node.advertise(f'/model/{name}/cmd_vel',Twist)
        deadline=time.monotonic()+5
        while not pub.has_connections() and time.monotonic()<deadline:
            time.sleep(0.05)
        assert pub.has_connections()
        before=poses[name]; cmd=Twist(); cmd.linear.x=0.3
        try:
            moved=threading.Event()
            for _ in range(100):
                # Retry while Gazebo transport finishes connecting a new publisher.
                pub.publish(cmd)
                moved.wait(0.2)
                if ((poses[name][0]-before[0])**2+(poses[name][1]-before[1])**2)**0.5>0.1: break
            else: raise AssertionError(f'{name} did not move')
        finally: pub.publish(Twist())
        print(name,'moved from',before,'to',poses[name])
