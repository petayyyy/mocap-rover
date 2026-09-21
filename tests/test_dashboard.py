import json, threading, unittest
from urllib.request import urlopen
from localization_contracts.dashboard import DashboardHandler, serve
from localization_contracts.registry import CameraRegistry
class SimulationDashboard(unittest.TestCase):
 def test_localhost_status_and_html(self):
  DashboardHandler.status_provider=staticmethod(lambda:{"hardware_verified":False,"events":0}); s=serve("127.0.0.1",0); threading.Thread(target=s.serve_forever,daemon=True).start(); base=f"http://127.0.0.1:{s.server_port}"; self.assertFalse(json.load(urlopen(base+"/api/status"))["hardware_verified"]); self.assertIn(b"Localization dashboard",urlopen(base+"/").read()); s.shutdown(); s.server_close()
 def test_camera_registry_endpoint_exposes_six_virtual_cameras(self):
  r=CameraRegistry.virtual_default(); DashboardHandler.cameras_provider=staticmethod(lambda:[r.status(cid) for cid in sorted(r.bindings)]); s=serve("127.0.0.1",0); threading.Thread(target=s.serve_forever,daemon=True).start(); data=json.load(urlopen(f"http://127.0.0.1:{s.server_port}/api/cameras")); self.assertEqual(len(data),6); self.assertFalse(data[0]['capabilities']['hardware_verified']); s.shutdown(); s.server_close()
