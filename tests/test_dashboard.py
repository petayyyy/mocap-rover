import json, threading, unittest
from urllib.request import urlopen
from localization_contracts.dashboard import DashboardHandler, serve
class SimulationDashboard(unittest.TestCase):
 def test_localhost_status_and_html(self):
  DashboardHandler.status_provider=staticmethod(lambda:{"hardware_verified":False,"events":0}); s=serve("127.0.0.1",0); threading.Thread(target=s.serve_forever,daemon=True).start(); base=f"http://127.0.0.1:{s.server_port}"; self.assertFalse(json.load(urlopen(base+"/api/status"))["hardware_verified"]); self.assertIn(b"Localization dashboard",urlopen(base+"/").read()); s.shutdown(); s.server_close()
