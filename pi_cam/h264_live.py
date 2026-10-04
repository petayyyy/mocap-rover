#!/usr/bin/env python3
"""Live H.264 bench: the camera in a browser, a button, and the encoder measured
while someone moves in front of the lens.

Runs on the node with the camera free (``camera_node`` stopped) and serves a
page on ``--port``.  The page shows the camera (the grey picture the stream
carries, a few frames a second) and, live, the encoded bitrate, frame sizes and
the encoder's latency.  ``Start`` counts down and then runs each case from
``--cases`` for ``--seconds`` -- the same cases and numbers as
``h264_probe.py`` -- so a run "still" and a run "moving a hand" can be put side
by side.  Results are kept on the page and written to ``--output-dir``.

    sudo systemctl stop camera_node
    PYTHONPATH=/opt/mocap-rover python3 h264_live.py --config /etc/mocap-rover/node_config.json
    # open http://<node>:8091
"""
from __future__ import annotations

import argparse
import collections
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_cam.camera_node import NodeConfig, boottime_ns, make_sensor  # noqa: E402
from pi_cam.h264_probe import (  # noqa: E402
    FULL_H, FULL_W, CpuMeter, V4L2H264Encoder, find_encoder_device, parse_case, prepare,
    soc_state, summary)

try:
    import cv2 as _cv2
except ImportError:      # pragma: no cover
    _cv2 = None
try:
    import simplejpeg as _simplejpeg
except ImportError:      # pragma: no cover
    _simplejpeg = None

IDLE_CASE = "640x480@all"
PREVIEW_PERIOD_S = 0.15


def jpeg(gray):
    gray = np.ascontiguousarray(gray)
    if _simplejpeg is not None:
        return _simplejpeg.encode_jpeg(gray[:, :, None], quality=70, colorspace="GRAY")
    ok, buf = _cv2.imencode(".jpg", gray, [int(_cv2.IMWRITE_JPEG_QUALITY), 70])
    return buf.tobytes()


class CaseStats:
    """Everything one measured case collects."""

    def __init__(self, case):
        self.case = case
        self.start_ns = boottime_ns()
        self.enc, self.total, self.p_bytes, self.k_bytes = [], [], [], []
        self.outputs = []           # (done_ns, bytes)
        self.seqs = []
        self.skipped_busy = self.skipped_prep = 0
        self.cpu = CpuMeter()

    def result(self):
        end = boottime_ns()
        seconds = (end - self.start_ns) / 1e9
        per_s = collections.Counter()
        for done, nbytes in self.outputs:
            per_s[int((done - self.start_ns) // 1_000_000_000)] += nbytes
        series = [round(per_s.get(i, 0) * 8 / 1e6, 2) for i in range(int(seconds))]
        gaps = np.diff(np.asarray(self.seqs, dtype=np.int64)) if len(self.seqs) > 1 else []
        frames = len(self.outputs)
        return {
            "case": self.case, "seconds": round(seconds, 1),
            "encoded_fps": round(frames / max(seconds, 1e-9), 2),
            "sensor_missed": int(np.sum(np.clip(np.asarray(gaps) - 1, 0, None))),
            "skipped_encoder_busy": self.skipped_busy, "skipped_prep_busy": self.skipped_prep,
            "encoder_ms": summary(self.enc), "total_ms": summary(self.total),
            "p_frame_kb": summary(self.p_bytes, 1e3), "keyframe_kb": summary(self.k_bytes, 1e3),
            "mbit_s": round(sum(b for _, b in self.outputs) * 8 / max(seconds, 1e-9) / 1e6, 2),
            "mbit_s_per_second": series,
            "mbit_s_per_second_max": max(series) if series else None,
            "cpu": self.cpu.percent(), "soc": soc_state(),
        }


class LiveBench:
    def __init__(self, a):
        self.a = a
        self.cfg = NodeConfig.load(a.config)
        self.raw = self.cfg.stream == "raw"
        self.device = a.device or find_encoder_device()
        if self.device is None:
            raise SystemExit("no V4L2 H.264 encoder found; pass --device")
        self.sensor = make_sensor(self.cfg)
        self.lock = threading.Lock()
        self.case = None
        self.encoder = None
        self.pending = {}           # frame id -> (queued_ns, exposure_start_ns)
        self.next_id = 1
        self.recent = collections.deque()   # (done_ns, bytes, keyframe, enc_ms)
        self.measuring = None       # CaseStats while a case is measured
        self.state = {"phase": "idle", "label": "", "case": IDLE_CASE, "left_s": 0}
        self.results = []
        self.clips = []
        self.preview = None
        self.last_preview = 0.0
        self.handoff = collections.deque(maxlen=1)
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.recording = None       # list of full grey frames while a motion clip is recorded
        self.out_dir = Path(a.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ encoder
    def switch(self, case):
        width, height, hz = parse_case(case)
        rate = hz if hz > 0 else self.cfg.fps
        bitrate = (self.a.bitrate_full if (width, height) == (FULL_W, FULL_H) else self.a.bitrate_small)
        with self.lock:
            old, self.encoder = self.encoder, None
        if old is not None:
            old.close()
        encoder = V4L2H264Encoder(self.device, width, height, rate, int(bitrate * 1e6),
                                  max(1, int(round(rate * self.a.gop_s))),
                                  on_output=self.on_output)
        with self.lock:
            self.pending.clear()
            self.recent.clear()
            self.case = (case, width, height, hz)
            self.encoder = encoder
            self.state["case"] = case

    def on_output(self, frame_id, done_ns, nbytes, keyframe, _data):
        with self.lock:
            info = self.pending.pop(frame_id, None)
            if info is None:
                return
            queued, exp_start = info
            enc_ms = (done_ns - queued) / 1e6
            self.recent.append((done_ns, nbytes, keyframe, enc_ms))
            m = self.measuring
            if m is not None and queued >= m.start_ns:
                m.enc.append(done_ns - queued)
                m.total.append(done_ns - exp_start)
                (m.k_bytes if keyframe else m.p_bytes).append(nbytes)
                m.outputs.append((done_ns, nbytes))

    # ------------------------------------------------------------ threads
    def capture_loop(self):
        last_tick = None
        while not self.stop.is_set():
            frame = self.sensor.capture()
            in_python = boottime_ns()
            with self.lock:
                case = self.case
                m = self.measuring
                if m is not None:
                    m.seqs.append(frame.sequence)
            if case is None:
                frame.release()
                continue
            hz = case[3]
            if hz > 0:
                tick = frame.sensor_stamp_ns // int(1e9 / hz)
                if tick == last_tick:
                    frame.release()
                    continue
                last_tick = tick
            raw_copy = np.array(frame.y, copy=True)
            exp_start = frame.sensor_stamp_ns - frame.exposure_ns
            frame.release()
            if self.handoff and m is not None:
                m.skipped_prep += 1
            self.handoff.append((raw_copy, exp_start, in_python, case))
            self.ready.set()

    def worker_loop(self):
        while not self.stop.is_set():
            if not self.ready.wait(0.1):
                continue
            self.ready.clear()
            try:
                raw_copy, exp_start, _in_python, case = self.handoff.popleft()
            except IndexError:
                continue
            _, width, height, _ = case
            gray = prepare(raw_copy, width, height, self.raw)
            rec = self.recording
            if rec is not None and gray.shape == (FULL_H, FULL_W):
                rec.append(np.array(gray, copy=True))
            now = time.monotonic()
            if now - self.last_preview >= PREVIEW_PERIOD_S:
                self.last_preview = now
                small = gray if gray.shape[1] <= 640 else _cv2.resize(gray, (640, 480),
                                                                       interpolation=_cv2.INTER_AREA)
                self.preview = jpeg(small)
            with self.lock:
                encoder = self.encoder
                if encoder is None or self.case != case:
                    continue
                frame_id = self.next_id
                self.next_id += 1
                self.pending[frame_id] = (boottime_ns(), exp_start)
            queued = encoder.submit(gray, frame_id)
            with self.lock:
                if queued is None:
                    self.pending.pop(frame_id, None)
                    if self.measuring is not None:
                        self.measuring.skipped_busy += 1
                elif frame_id in self.pending:
                    self.pending[frame_id] = (queued, exp_start)

    def run_sequence(self, label):
        try:
            for left in (3, 2, 1):
                self.state.update(phase="countdown", label=label, left_s=left)
                time.sleep(1.0)
            batch = {"label": label, "time": time.strftime("%H:%M:%S"), "cases": []}
            for case in self.a.cases:
                self.state.update(phase="switching", case=case, left_s=self.a.seconds)
                self.switch(case)
                time.sleep(0.5)                 # the new encoder's first keyframe
                with self.lock:
                    self.measuring = CaseStats(case)
                self.state["phase"] = "measuring"
                end = time.monotonic() + self.a.seconds
                while time.monotonic() < end:
                    self.state["left_s"] = round(end - time.monotonic(), 1)
                    time.sleep(0.2)
                with self.lock:
                    m, self.measuring = self.measuring, None
                batch["cases"].append(m.result())
            self.results.append(batch)
            name = f"live_{len(self.results):02d}_{label}.json"
            (self.out_dir / name).write_text(json.dumps(batch, indent=2))
            self.switch(IDLE_CASE)
            self.state.update(phase="idle", label="", left_s=0)
        except Exception as exc:          # keep serving the page, show the error there
            self.state.update(phase="error", label=f"{type(exc).__name__}: {exc}")
            try:
                self.switch(IDLE_CASE)
            except Exception:
                pass

    def run_clip(self, seconds, bitrates):
        """Record full grey frames at 60 Hz while the person moves, then put the
        very same frames through the hardware encoder at each bitrate, paced at
        60 fps, and keep the lossless frames as PNG for the comparison."""
        try:
            for left in (3, 2, 1):
                self.state.update(phase="countdown", label="clip", left_s=left)
                time.sleep(1.0)
            self.state.update(phase="switching", case="full@60", left_s=seconds)
            self.switch("full@60")
            time.sleep(0.3)
            self.recording = []
            self.state["phase"] = "recording"
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                self.state["left_s"] = round(end - time.monotonic(), 1)
                time.sleep(0.1)
            frames, self.recording = self.recording, None
            self.switch(IDLE_CASE)
            clip = self.out_dir / f"clip_{time.strftime('%H%M%S')}"
            clip.mkdir(parents=True, exist_ok=True)
            summary_rows = []
            for n, br in enumerate(bitrates):
                self.state.update(phase="encoding", label=f"{br} Мбит/с", left_s=len(bitrates) - n)
                sizes, lock = [], threading.Lock()
                out = open(clip / f"h264_{br:g}mbit.h264", "wb")

                def keep(_fid, _t, nbytes, _k, data):
                    with lock:
                        out.write(data)
                        sizes.append(nbytes)
                enc = V4L2H264Encoder(self.device, FULL_W, FULL_H, 60.0, int(br * 1e6), 12,
                                      keep_data=True, on_output=keep)
                t0 = time.monotonic()
                for i, g in enumerate(frames):
                    while time.monotonic() < t0 + i / 60:
                        time.sleep(0.0005)
                    while enc.submit(g, i + 1) is None:
                        time.sleep(0.001)
                time.sleep(0.5)
                enc.close()
                out.close()
                summary_rows.append({"bitrate_target": br, "frames": len(sizes),
                                     "mbit_s": round(sum(sizes) * 8 / (len(frames) / 60) / 1e6, 2)})
            self.state.update(phase="encoding", label="PNG без сжатия", left_s=0)
            for i, g in enumerate(frames):
                _cv2.imwrite(str(clip / f"ref_{i:03d}.png"), g)
            info = {"clip": str(clip), "frames": len(frames), "encoded": summary_rows}
            (clip / "clip.json").write_text(json.dumps(info, indent=2))
            self.clips.append(info)
            self.state.update(phase="idle", label="", left_s=0)
        except Exception as exc:
            self.recording = None
            self.state.update(phase="error", label=f"{type(exc).__name__}: {exc}")
            try:
                self.switch(IDLE_CASE)
            except Exception:
                pass

    def start_clip(self):
        if self.state["phase"] not in ("idle", "error"):
            return False
        self.state.update(phase="countdown", label="clip", left_s=3)
        threading.Thread(target=self.run_clip, args=(self.a.clip_seconds, self.a.clip_bitrates),
                         daemon=True).start()
        return True

    def start_sequence(self, label):
        if self.state["phase"] not in ("idle", "error"):
            return False
        self.state.update(phase="countdown", label=label, left_s=3)
        threading.Thread(target=self.run_sequence, args=(label,), daemon=True).start()
        return True

    def live_stats(self):
        now = boottime_ns()
        with self.lock:
            while self.recent and now - self.recent[0][0] > 1_000_000_000:
                self.recent.popleft()
            recent = list(self.recent)
        sizes = [b for _, b, k, _ in recent if not k]
        enc = [e for *_, e in recent]
        return {
            "state": dict(self.state),
            "fps": len(recent),
            "mbit_s": round(sum(b for _, b, _, _ in recent) * 8 / 1e6, 2),
            "p_frame_kb": round(float(np.median(sizes)) / 1e3, 2) if sizes else None,
            "enc_ms_p50": round(float(np.median(enc)), 2) if enc else None,
            "enc_ms_max": round(max(enc), 2) if enc else None,
            "soc": soc_state() if int(time.time()) % 3 == 0 else None,
            "results": self.results,
            "clips": self.clips,
        }

    def serve(self):
        self.sensor.start()
        self.sensor.set_controls(exposure_us=self.a.exposure_us, gain=self.a.gain)
        self.switch(IDLE_CASE)
        for target in (self.capture_loop, self.worker_loop):
            threading.Thread(target=target, daemon=True).start()
        bench = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/preview.jpg" and bench.preview is not None:
                    self._send(200, bench.preview, "image/jpeg")
                elif path == "/stats":
                    self._send(200, json.dumps(bench.live_stats()).encode(), "application/json")
                elif path == "/results.json":
                    self._send(200, json.dumps(bench.results, indent=2).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                path, _, query = self.path.partition("?")
                if path == "/clip":
                    ok = bench.start_clip()
                    self._send(200 if ok else 409, json.dumps({"ok": ok}).encode(),
                               "application/json")
                elif path == "/start":
                    label = "motion" if "motion=1" in query else "still"
                    ok = bench.start_sequence(label)
                    self._send(200 if ok else 409, json.dumps({"ok": ok}).encode(),
                               "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

        server = ThreadingHTTPServer(("0.0.0.0", self.a.port), Handler)
        print(f"serving on :{self.a.port}, encoder {self.device}", flush=True)
        try:
            server.serve_forever()
        finally:
            self.stop.set()
            with self.lock:
                encoder, self.encoder = self.encoder, None
            if encoder is not None:
                encoder.close()
            self.sensor.stop()


PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H.264 Bench</title>
<style>
:root{--bg:#f6f7f9;--fg:#15181d;--mut:#5b6472;--card:#fff;--line:#d9dde3;--acc:#2563eb;--ok:#15803d;--warn:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--fg:#e8eaed;--mut:#9aa3ae;--card:#1a1e24;--line:#2c323a;--acc:#60a5fa;--ok:#4ade80;--warn:#fbbf24}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px;display:grid;gap:16px;grid-template-columns:minmax(0,640px) minmax(0,1fr)}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
img{width:100%;aspect-ratio:4/3;background:#000;border-radius:6px;display:block}
h1{font-size:16px;margin:0 0 8px}h2{font-size:14px;margin:0 0 8px;color:var(--mut)}
.big{font-size:28px;font-weight:600;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.k{color:var(--mut);font-size:12px}
button{font:inherit;padding:10px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
button:disabled{opacity:.5;cursor:default}
#phase{font-weight:600}
canvas{width:100%;height:120px;display:block}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{text-align:right;padding:4px 6px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
.wide{grid-column:1/-1}
</style></head><body><main>
<section class="card"><h1>Камера (то, что уходит в кодер)</h1><img id="cam" alt="камера">
<p class="k">Кейс сейчас: <span id="case">—</span></p></section>
<section class="card"><h1>Кодер H.264 сейчас</h1>
<div class="grid">
<div><div class="k">Мбит/с</div><div class="big" id="mbit">—</div></div>
<div><div class="k">кадров/с</div><div class="big" id="fps">—</div></div>
<div><div class="k">P-кадр, КБ</div><div class="big" id="pkb">—</div></div>
<div><div class="k">кодер, мс (медиана / макс)</div><div class="big" id="enc">—</div></div>
</div>
<h2 style="margin-top:12px">Мбит/с, последние 30 с</h2><canvas id="chart" width="600" height="120"></canvas>
<p><span id="phase">готов</span> <span id="label" class="k"></span></p>
<p style="display:flex;gap:8px;flex-wrap:wrap">
<button id="still">Замер: без движения</button>
<button id="motion">Замер: двигаю рукой</button>
<button id="clip" class="primary">Запись 1640×1232 с движением</button></p>
<p class="k" id="cliphint">Запись: после отсчёта 3 с — 5 с полного кадра 60 к/с. Водите доской по кадру: сначала медленно, потом быстрее, держите её целиком в кадре. Потом кодирование на 5 битрейтах (~30 с).</p>
<p class="k" id="clips"></p>
<p class="k">После нажатия — отсчёт 3 с, затем каждый кейс по очереди. Для «двигаю рукой» двигайте всё время замера, быстро и по всему кадру.</p>
</section>
<section class="card wide"><h1>Результаты</h1><div id="results" class="k">Пока нет.</div></section>
</main><script>
const $=id=>document.getElementById(id);let hist=[];
function draw(){const c=$('chart'),g=c.getContext('2d'),w=c.width,h=c.height;g.clearRect(0,0,w,h);
const css=getComputedStyle(document.documentElement);const max=Math.max(1,...hist)*1.15;
g.strokeStyle=css.getPropertyValue('--line');g.beginPath();g.moveTo(0,h-1);g.lineTo(w,h-1);g.stroke();
g.fillStyle=css.getPropertyValue('--mut');g.font='11px system-ui';g.fillText(max.toFixed(1)+' Мбит/с',4,12);
g.strokeStyle=css.getPropertyValue('--acc');g.lineWidth=2;g.beginPath();
hist.forEach((v,i)=>{const x=i*(w/59),y=h-2-(v/max)*(h-16);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()}
function fmt(s){return s&&s.p50!=null?`${s.p50} / ${s.p95}`:'—'}
function table(results){if(!results.length)return'Пока нет.';let r='<table><tr><th>замер</th><th>кейс</th><th>к/с</th><th>Мбит/с</th><th>макс за 1 с</th><th>P-кадр КБ P50/P95</th><th>опорный КБ</th><th>кодер мс P50/P95</th><th>кодер макс</th><th>пропуски</th><th>CPU %</th><th>°C</th></tr>';
results.forEach(b=>b.cases.forEach(c=>{r+=`<tr><td>${b.label==='motion'?'рука':'без движения'} ${b.time}</td><td>${c.case}</td><td>${c.encoded_fps}</td><td>${c.mbit_s}</td><td>${c.mbit_s_per_second_max??'—'}</td><td>${fmt(c.p_frame_kb)}</td><td>${c.keyframe_kb.p50??'—'}</td><td>${fmt(c.encoder_ms)}</td><td>${c.encoder_ms.max??'—'}</td><td>${c.sensor_missed}/${c.skipped_prep_busy}/${c.skipped_encoder_busy}</td><td>${c.cpu.board_percent??'—'}</td><td>${c.soc.temp_c?.toFixed?.(0)??'—'}</td></tr>`}));
return r+'</table><p class="k">Пропуски: сенсор / подготовка не успела / кодер занят. JSON: <a href="/results.json">results.json</a></p>'}
const names={idle:'готов',countdown:'отсчёт',switching:'переключаю кейс',measuring:'замер',recording:'ЗАПИСЬ — двигайте доску!',encoding:'кодирую',error:'ошибка'};
async function tick(){try{const s=await (await fetch('/stats')).json();$('mbit').textContent=s.mbit_s;$('fps').textContent=s.fps;
$('pkb').textContent=s.p_frame_kb??'—';$('enc').textContent=s.enc_ms_p50!=null?`${s.enc_ms_p50} / ${s.enc_ms_max}`:'—';
$('case').textContent=s.state.case;const st=s.state;$('phase').textContent=names[st.phase]||st.phase;
if(st.label==='clip'||st.phase==='recording'||st.phase==='encoding'){$('label').textContent=st.phase==='countdown'?`приготовьтесь двигать доску: ${st.left_s}`:st.phase==='recording'?`осталось ${st.left_s} с`:st.phase==='encoding'?`${st.label}`:'';}else
$('label').textContent=st.phase==='countdown'?`${st.label==='motion'?'приготовьтесь двигать рукой':'не двигайтесь'}: ${st.left_s}`:st.phase==='measuring'?`${st.label==='motion'?'двигайте рукой!':'без движения'} осталось ${st.left_s} с`:st.phase==='error'?st.label:'';
const busy=!['idle','error'].includes(st.phase);$('still').disabled=busy;$('motion').disabled=busy;$('clip').disabled=busy;
$('phase').style.color=st.phase==='recording'?'#dc2626':'';$('clips').textContent=(s.clips||[]).map(c=>`записано: ${c.clip.split('/').pop()}, ${c.frames} кадров; `+c.encoded.map(e=>`${e.bitrate_target}→${e.mbit_s} Мбит/с`).join(', ')).join(' | ');
hist.push(s.mbit_s);if(hist.length>60)hist.shift();draw();$('results').innerHTML=table(s.results)}catch(e){$('phase').textContent='нет связи с CM4'}}
setInterval(tick,500);tick();
setInterval(()=>{$('cam').src='/preview.jpg?'+Date.now()},150);
$('clip').onclick=()=>fetch('/clip',{method:'POST'});
$('still').onclick=()=>fetch('/start',{method:'POST'});$('motion').onclick=()=>fetch('/start?motion=1',{method:'POST'});
</script></body></html>
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="node_config.json (stream: raw on CM4)")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--cases", nargs="+", default=["640x480@all", "640x480@30", "full@30"])
    p.add_argument("--seconds", type=float, default=15.0, help="per case")
    p.add_argument("--exposure-us", type=int, default=3000)
    p.add_argument("--gain", type=float, default=4.0)
    p.add_argument("--bitrate-small", type=float, default=10.0)
    p.add_argument("--bitrate-full", type=float, default=25.0)
    p.add_argument("--gop-s", type=float, default=0.2)
    p.add_argument("--clip-seconds", type=float, default=5.0)
    p.add_argument("--clip-bitrates", type=float, nargs="+", default=[2, 4, 8, 15, 25])
    p.add_argument("--device")
    p.add_argument("--output-dir", default="h264_live")
    return p.parse_args(argv)


if __name__ == "__main__":
    LiveBench(parse_args()).serve()
