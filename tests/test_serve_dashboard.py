import subprocess,sys,unittest
class ServeDashboardScript(unittest.TestCase):
 def test_script_help_is_available_without_devices(self):
  p=subprocess.run([sys.executable,'scripts/serve_dashboard.py','--help'],capture_output=True,text=True,check=True); self.assertIn('--snapshot-dir',p.stdout)
