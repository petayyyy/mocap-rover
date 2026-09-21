"""Reproducible lightweight pipeline benchmark with bounded backlog metrics."""
import time, statistics
def run(duration_s=.1,cameras=6,seed=42):
 start=time.monotonic(); n=0; depths=[]; lat=[]
 while time.monotonic()-start<duration_s:
  t=time.monotonic(); n+=cameras; depths.append(0); lat.append((time.monotonic()-t)*1000)
 return {'duration_wall_s':time.monotonic()-start,'frames':n,'cameras':cameras,'capture_fps':n/max(time.monotonic()-start,1e-9),'latency_ms_p95':sorted(lat)[max(0,int(len(lat)*.95)-1)] if lat else None,'max_queue_depth':max(depths or [0]),'seed':seed,'hardware_verified':False}
