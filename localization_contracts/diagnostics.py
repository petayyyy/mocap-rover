"""Structured diagnostics for UI/debug consumers; no raw video transport."""
import statistics
def percentile(values,p):
 if not values:return None
 x=sorted(values); return x[min(len(x)-1,int((len(x)-1)*p))]
class Diagnostics:
 def __init__(self): self.events=[]; self.channels={}
 def record(self,kind,stamp_ns,**data): self.events.append({'kind':kind,'stamp_ns':stamp_ns,'source_state':kind,**data})
 def channel(self,camera_id,depth,drops,age_ms): self.channels[camera_id]={'queue_depth':depth,'drops':drops,'age_ms':age_ms}
 def report(self):
  l=[e['latency_ms'] for e in self.events if 'latency_ms' in e]
  return {'events':len(self.events),'latency_ms':{'p50':percentile(l,.5),'p95':percentile(l,.95),'p99':percentile(l,.99)},'channels':dict(self.channels),'truth_visible':False}
