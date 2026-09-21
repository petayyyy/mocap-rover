"""Deterministic replay runner for image pipeline sessions."""
from __future__ import annotations
from dataclasses import dataclass
from localization_contracts.image_pipeline import OneCameraImagePipeline

@dataclass(frozen=True)
class ReplayImageFrame:
    image:object; capture_time_ns:int; receive_time_ns:int; sequence:int

class ReplayImageRunner:
    def __init__(self, pipeline:OneCameraImagePipeline):
        self.pipeline=pipeline; self.session=0; self.processed=0; self.outputs=[]

    def reset(self):
        self.pipeline.fusion.reset(); self.pipeline.last_observation=None; self.session+=1

    def run(self, events):
        for event in events:
            if event[0]=='new_session': self.reset(); continue
            if event[0] != 'frame': raise ValueError('unknown replay event')
            frame=event[2]
            if not isinstance(frame,ReplayImageFrame): raise TypeError('replay frame required')
            self.pipeline.process(frame.image,frame.capture_time_ns,frame.receive_time_ns,frame.sequence)
            self.outputs.append(self.pipeline.publish(frame.capture_time_ns)); self.processed+=1
        return {'session':self.session,'processed':self.processed,'outputs':len(self.outputs),'hardware_verified':False}
