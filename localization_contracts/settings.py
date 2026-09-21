"""Capability-aware settings backend boundary with revision and acknowledgements."""
from dataclasses import dataclass
from .config import ConfigStore, ConfigError
@dataclass(frozen=True)
class Ack: revision:int; ok:bool; error:str|None=None
class SettingsBackend:
 def __init__(self,config,capabilities=None): self.store=ConfigStore(config); self.revision=0; self.capabilities=capabilities or {}; self.last_ack=None
 def stage(self,patch):
  if patch.get('timing',{}).get('trigger') and not self.capabilities.get('trigger',False): raise ConfigError('hardware trigger unavailable in simulation')
  return self.store.stage(patch)
 def apply(self,expected_revision=None):
  if expected_revision is not None and expected_revision != self.revision:
   self.last_ack=Ack(self.revision,False,'stale config revision'); return self.last_ack
  try: self.store.apply(); self.revision+=1; self.last_ack=Ack(self.revision,True); return self.last_ack
  except ConfigError as e: self.last_ack=Ack(self.revision,False,str(e)); return self.last_ack
 def rollback(self): self.store.rollback(); self.revision+=1; self.last_ack=Ack(self.revision,True); return self.last_ack
