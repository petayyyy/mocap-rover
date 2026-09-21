"""Localhost-only dashboard endpoint for simulation diagnostics/settings."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
class DashboardHandler(BaseHTTPRequestHandler):
    status_provider = staticmethod(lambda: {"phase":"SIMULATION","hardware_verified":False})
    cameras_provider = staticmethod(lambda: [])
    previews_provider = staticmethod(lambda: [])
    preview_files_provider = staticmethod(lambda: {})
    def do_GET(self):
        if self.path == "/api/status": self._json(self.status_provider()); return
        if self.path == "/api/cameras": self._json(self.cameras_provider()); return
        if self.path == "/api/previews": self._json(self.previews_provider()); return
        if self.path.startswith("/preview/"):
            camera_id=self.path[len("/preview/"):]
            path=self.preview_files_provider().get(camera_id)
            if not path or not Path(path).is_file(): self.send_error(404); return
            body=Path(path).read_bytes(); self.send_response(200); self.send_header("Content-Type","image/x-portable-pixmap"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path == "/":
            body=b"<html><title>mocap-rover simulation</title><body><h1>Localization dashboard</h1><pre id='status'></pre><script>fetch('/api/status').then(r=>r.json()).then(x=>status.textContent=JSON.stringify(x,null,2))</script></body></html>"
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body); return
        self.send_error(404)
    def _json(self,value):
        body=json.dumps(value,sort_keys=True).encode(); self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self,*args): pass
def serve(host="127.0.0.1",port=8080): return ThreadingHTTPServer((host,port),DashboardHandler)
