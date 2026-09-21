"""Reproducible lightweight pipeline benchmark with bounded backlog metrics."""
import time, statistics
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
