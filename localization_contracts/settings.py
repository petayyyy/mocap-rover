"""Capability-aware settings backend boundary with revision and acknowledgements."""
from dataclasses import dataclass
from .config import ConfigStore, ConfigError
@dataclass(frozen=True)
class Ack: revision:int; ok:bool; error:str|None=None
class SettingsBackend:
 def __init__(self,config,capabilities=None): self.store=ConfigStore(config); self.revision=0; self.capabilities=capabilities or {}; self.last_ack=None; self._runtime_apply=None; self._runtime_rollback=None
 def bind_runtime(self, apply_callback, rollback_callback=None):
  if not callable(apply_callback): raise TypeError('apply_callback must be callable')
  self._runtime_apply=apply_callback; self._runtime_rollback=rollback_callback; return self
 def stage(self,patch):
  if patch.get('timing',{}).get('trigger') and not self.capabilities.get('trigger',False): raise ConfigError('hardware trigger unavailable in simulation')
  return self.store.stage(patch)
 def apply(self,expected_revision=None):
  if expected_revision is not None and expected_revision != self.revision:
   self.last_ack=Ack(self.revision,False,'stale config revision'); return self.last_ack
  try:
   candidate=self.store.staged
   if candidate is None: raise ConfigError('nothing staged')
   if self._runtime_apply is not None: self._runtime_apply(candidate)
   applied=self.store.apply()
   self.revision+=1; self.last_ack=Ack(self.revision,True); return self.last_ack
  except Exception as e: self.last_ack=Ack(self.revision,False,str(e)); return self.last_ack
 def rollback(self):
  self.store.rollback()
  if self._runtime_rollback is not None: self._runtime_rollback(self.store.active)
  self.revision+=1; self.last_ack=Ack(self.revision,True); return self.last_ack
