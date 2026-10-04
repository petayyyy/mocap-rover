#!/usr/bin/env python3
"""All cameras on one page, from the C++ receiver's shared memory.

Run ``mocap_rx --shm`` (or ``run_hardware.sh rx``) first, then

    .venv/bin/python scripts/hardware/camera_wall.py          # http://localhost:8090

Each tile is the newest decoded frame of one camera (reduced for the
browser) with its rate and the age of the frame (now minus the start of its
exposure, on this host's clock -- the same number the tract would see).
Settings of a camera are on its own page, http://<node>:8080.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from localization_contracts.shm_frames import ShmFrameReader  # noqa: E402

PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Camera Wall</title>
<style>
:root{--bg:#f5f6f8;--fg:#14171c;--mut:#5d6673;--card:#fff;--line:#d8dce2;--warn:#c2410c}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--fg:#e7e9ec;--mut:#98a1ad;--card:#181c22;--line:#2a3038;--warn:#fb923c}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
main{display:grid;gap:12px;padding:16px;grid-template-columns:repeat(auto-fill,minmax(360px,1fr))}
.t{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px}
.t img{width:100%;display:block;border-radius:6px;background:#000;aspect-ratio:4/3;object-fit:contain}
.h{display:flex;justify-content:space-between;margin:6px 2px 0;font-variant-numeric:tabular-nums}
.m{color:var(--mut);font-size:12px}.w{color:var(--warn)}a{color:inherit}
</style></head><body><main id="wall"></main><script>
const wall=document.getElementById('wall');const tiles={};
function tile(id){const d=document.createElement('div');d.className='t';
 d.innerHTML=`<img alt="${id}"><div class="h"><b>${id}</b><span class="m"></span></div>`;wall.appendChild(d);
 const img=d.querySelector('img');const next=()=>{img.src='/cam/'+id+'.jpg?'+Date.now()};img.onload=()=>setTimeout(next,100);img.onerror=()=>setTimeout(next,1000);next();
 return d}
async function poll(){try{const s=await (await fetch('/stats')).json();
 for(const c of s){const t=tiles[c.id]||(tiles[c.id]=tile(c.id));const m=t.querySelector('.m');
  m.textContent=`${c.width}×${c.height} ${c.color?'цвет':'серое'} · ${c.fps.toFixed(1)} к/с · возраст ${c.age_ms.toFixed(0)} мс`;
  m.className='m'+(c.fps<45||c.age_ms>80?' w':'')}}catch(e){} setTimeout(poll,1000)}
poll();
</script></body></html>"""


class Wall:
    def __init__(self, width):
        self.width = width
        self.readers = {}
        self.lock = threading.Lock()
        self.rate = {}           # id -> (count, t0, last_seq, fps)

    def cameras(self):
        for p in sorted(Path("/dev/shm").glob("mocap_*")):
            cid = p.name[len("mocap_"):]
            if cid not in self.readers:
                self.readers[cid] = ShmFrameReader(cid)
        return self.readers

    def stats(self):
        out = []
        now = time.time_ns()
        with self.lock:
            for cid, r in self.cameras().items():
                f = r.latest()
                if f is None:
                    continue
                cnt, t0, last, fps = self.rate.get(cid, (0, now, None, 0.0))
                if f.frame_seq != last:
                    # frames since the last poll from the sequence numbers
                    cnt += 1 if last is None else max(0, f.frame_seq - last)
                if now - t0 > 2e9:
                    fps, cnt, t0 = cnt / ((now - t0) / 1e9), 0, now
                self.rate[cid] = (cnt, t0, f.frame_seq, fps)
                out.append({"id": cid, "width": f.width, "height": f.height, "color": f.color,
                            "fps": fps, "age_ms": (now - f.stamp_ns) / 1e6})
        return out

    def jpeg(self, cid):
        with self.lock:
            r = self.cameras().get(cid)
            f = r.latest() if r else None
        if f is None:
            return None
        img = f.bgr() if f.color else f.y
        k = self.width / img.shape[1]
        img = cv2.resize(img, (self.width, int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--width", type=int, default=640, help="tile width in pixels")
    a = p.parse_args()
    wall = Wall(a.width)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self.send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/stats":
                self.send(json.dumps(wall.stats()).encode(), "application/json")
            elif path.startswith("/cam/") and path.endswith(".jpg"):
                body = wall.jpeg(path[5:-4])
                self.send(body or b"", "image/jpeg", 200 if body else 404)
            else:
                self.send(b"not found", "text/plain", 404)

    print(f"camera wall on http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
