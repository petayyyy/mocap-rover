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
            slots="".join(f"<figure><figcaption>camera_{i}</figcaption><img src='/preview/camera_{i}' alt='camera_{i} preview' width='320'><small id='camera_{i}-state'>metadata pending</small></figure>" for i in range(1,7))
            body=("<html><title>mocap-rover simulation</title><body><h1>Localization dashboard</h1>"
                  "<section><h2>XY map (runtime status)</h2><svg id='map' viewBox='0 0 120 120' width='360' height='360' style='background:#20242b'><path d='M0 0H120V120H0Z' fill='none' stroke='#777'/><circle id='track' cx='0' cy='0' r='2' fill='#4fd'/></svg></section>"
                  "<pre id='status'>loading</pre><section id='previews'>"+slots+"</section>"
                  "<script>async function refresh(){let x=await fetch('/api/status').then(r=>r.json());status.textContent=JSON.stringify(x,null,2);let s=x.track||x.state;if(s&&s.x!==undefined){track.setAttribute('cx',s.x*10);track.setAttribute('cy',(12-s.y)*10);track.setAttribute('fill',s.tracking_state==='LOST'?'#f44':'#4fd')}let xs=await fetch('/api/cameras').then(r=>r.json());xs.forEach(x=>{let e=document.getElementById(x.camera_id+'-state');if(e)e.textContent=JSON.stringify(x.channels)});} refresh();setInterval(refresh,500);</script>"
                  "</body></html>").encode()
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body); return
        self.send_error(404)
    def _json(self,value):
        body=json.dumps(value,sort_keys=True).encode(); self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self,*args): pass
def serve(host="127.0.0.1",port=8080): return ThreadingHTTPServer((host,port),DashboardHandler)
