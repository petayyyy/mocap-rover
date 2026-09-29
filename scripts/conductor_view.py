#!/usr/bin/env python3
"""Conductor-side page: what the replay detected, against the truth.

Evaluator tool, reads truth on purpose.  Builds one self-contained HTML file
from a replay directory: a top-view animation of both rovers (estimate vs
truth), error over time, per-camera window modes, and sample frames with the
planned windows and the projected truth drawn on them.

    conductor_view.py <replay> --truth <dataset>/truth.jsonl --dataset <dataset> --output page.html
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localization_contracts.camera_model import CameraModel  # noqa: E402
from scripts.conductor_check import check_replay, load_truth, read_jsonl, verdict  # noqa: E402

TAG_PLANE_Z = 0.3654
OPP_TOP_Z = 0.483


def project(cam, model, point):
    R = np.asarray(cam["R_world_optical"], dtype=float).reshape(3, 3)
    C = np.asarray(cam["position_world"], dtype=float)
    optical = R.T @ (np.asarray(point, dtype=float) - C)
    if optical[2] <= 1e-6:
        return None
    uv = model.project(optical.reshape(1, 3))[0]
    return float(uv[0]), float(uv[1])


def track_series(rows, truth, step):
    """Downsampled (t_s, est_xy, truth_xy, err, valid, state) for one object."""
    out = []
    for row in rows[::step]:
        state = row.get("state") or {}
        stamp = int(state.get("stamp_ns", row.get("stamp_ns", 0)))
        gt = truth.at(stamp)
        if gt is None:
            continue
        valid = bool(row.get("valid"))
        if valid and "x" in state:
            est = [round(state["x"], 3), round(state["y"], 3)]
            err = round(float(np.hypot(est[0] - gt[0][0], est[1] - gt[0][1])), 4)
        else:
            est, err = None, None
        out.append([round(stamp / 1e9, 3), est, [round(float(gt[0][0]), 3), round(float(gt[0][1]), 3)],
                    err, valid, row.get("tracking_state"), round(float(gt[1]), 3),
                    round(state.get("yaw", 0.0), 3) if est else None])
    return out


def sample_frames(runtime, dataset, cams, truth, odometry_by_object, count_per_camera=1, at=()):
    """One annotated frame per camera where both windows were planned.

    ``at`` is a list of ``(camera_id, t_s)``: those frames are added first,
    so a reviewer can point the page at the moments that went wrong.
    """
    frames = read_jsonl(runtime / "camera_frames.jsonl")
    forced = {}
    for cid, t in at:
        rows = [r for r in frames if r["camera_id"] == cid]
        if rows:
            forced.setdefault(cid, []).append(min(rows, key=lambda r: abs(r["capture_ns"] / 1e9 - t)))
    by_camera = {}
    for row in frames:
        if row.get("roi") and row.get("opponent_roi") and row.get("fusion_accepted"):
            by_camera.setdefault(row["camera_id"], []).append(row)
    images = []
    for cid in sorted(cams):
        candidates = by_camera.get(cid) or [r for r in frames if r["camera_id"] == cid and r.get("roi")]
        if not candidates:
            continue
        cam = cams[cid]
        model = CameraModel.from_config(cam)
        index = read_jsonl(dataset / f"{cid}.jsonl")
        stamp_to_index = {int(r["stamp_ns"]): int(r["index"]) for r in index}
        picks = [candidates[len(candidates) // 2]] if count_per_camera == 1 else candidates[:: max(1, len(candidates) // count_per_camera)][:count_per_camera]
        picks = forced.get(cid, []) + picks
        capture = cv2.VideoCapture(str(dataset / f"{cid}.mkv"))
        for row in picks:
            stamp = int(row["capture_ns"])
            frame_index = stamp_to_index.get(stamp)
            if frame_index is None:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = capture.read()
            if not ok:
                continue
            draw = bgr.copy()
            for key, color in (("roi", (80, 200, 255)), ("opponent_roi", (255, 140, 60))):
                rect = row.get(key)
                if rect:
                    x, y, w, h = (int(v) for v in rect)
                    cv2.rectangle(draw, (x, y), (x + w, y + h), color, 3)
            labels = []
            for name, plane_z, color in (("tag_rover", TAG_PLANE_Z, (80, 200, 255)),
                                         ("opponent", OPP_TOP_Z, (255, 140, 60))):
                gt = truth[name].at(stamp) if name in truth else None
                if gt is not None:
                    uv = project(cam, model, (gt[0][0], gt[0][1], plane_z))
                    if uv is not None:
                        u, v = int(round(uv[0])), int(round(uv[1]))
                        cv2.drawMarker(draw, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 40, 4)
                        cv2.drawMarker(draw, (u, v), color, cv2.MARKER_CROSS, 36, 2)
                rows = odometry_by_object.get(name, [])
                est = None
                for r in rows:
                    s = r.get("state") or {}
                    if r.get("valid") and "x" in s and abs(int(s["stamp_ns"]) - stamp) <= 5_000_000:
                        est = (s["x"], s["y"])
                        break
                if est is not None:
                    uv = project(cam, model, (est[0], est[1], plane_z))
                    if uv is not None:
                        cv2.circle(draw, (int(round(uv[0])), int(round(uv[1]))), 18, color, 3)
                        gt_xy = truth[name].at(stamp)
                        if gt_xy is not None:
                            labels.append(f"{name}: {np.hypot(est[0]-gt_xy[0][0], est[1]-gt_xy[0][1])*100:.1f} cm")
            small = cv2.resize(draw, (820, 616), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 78])
            images.append({
                "camera": cid, "t": round(stamp / 1e9, 3), "mode": row.get("mode"),
                "opponent_mode": row.get("opponent_mode"), "opponent_method": row.get("opponent_method"),
                "detections": row.get("detections"), "labels": labels,
                "jpeg": base64.b64encode(jpg.tobytes()).decode("ascii"),
            })
        capture.release()
    return images


def build(runtime, truth_path, dataset, output, title, at=()):
    runtime, dataset = Path(runtime), Path(dataset)
    cfg = json.loads((dataset / "runtime_cameras.json").read_text())
    cams = {c["name"]: c for c in cfg["cameras"]}
    truth = load_truth(truth_path)
    report = check_replay(runtime, truth_path, dataset / "runtime_cameras.json")
    report["verdict"] = verdict(report)
    odometry = read_jsonl(runtime / "odometry.jsonl")
    by_object = {}
    for row in odometry:
        by_object.setdefault(row["object_id"], []).append(row)
    for rows in by_object.values():
        rows.sort(key=lambda r: (r.get("state") or {}).get("stamp_ns", r.get("stamp_ns", 0)))
    series = {name: track_series(rows, truth[name], 2) for name, rows in by_object.items() if name in truth}
    images = sample_frames(runtime, dataset, cams, truth, by_object, at=at)
    lidar = cfg.get("lidar") or {}
    data = {
        "title": title,
        "runtime": str(runtime), "dataset": str(dataset),
        "cameras": [{"id": cid, "xy": c["position_world"][:2]} for cid, c in cams.items()],
        "lidar_xy": (lidar.get("position_world") or [None, None])[:2],
        "series": series,
        "summary": {name: {
            "p50_cm": round(o["xy_error_m"]["p50"] * 100, 2) if o.get("xy_error_m") else None,
            "p95_cm": round(o["xy_error_m"]["p95"] * 100, 2) if o.get("xy_error_m") else None,
            "max_cm": round(o["xy_error_m"]["max"] * 100, 1) if o.get("xy_error_m") else None,
            "valid_pct": round(100 * (o.get("valid_fraction_sim_time") or 0), 3),
            "yaw_p95": round(o["yaw_error_deg"]["p95"], 2) if o.get("yaw_error_deg") else None,
            "age_p95": o["measurement_age_ms"]["p95"] if o.get("measurement_age_ms") else None,
            "swaps": o.get("swap_episodes_over_limit"),
            "pass": report["verdict"].get(name, {}).get("pass"),
            "checks": report["verdict"].get(name, {}).get("checks"),
        } for name, o in report["objects"].items()},
        "observations": report.get("observations", {}),
        "camera_frames": {cid: {"mode_share": v["mode_share"], "latency_p95": v["latency_ms"]["p95"],
                                "frames": v["frames"], "pnp_rejections": v["pnp_rejections"]}
                          for cid, v in report["camera_frames"]["cameras"].items()},
        "lidar": report.get("lidar", {}).get("reasons", {}),
        "timing": {k: v for k, v in (report.get("timing.json") or {}).items()
                   if k in ("realtime_ratio", "fps_total", "latency_ms_all_cameras")},
        "images": images,
    }
    html = TEMPLATE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data, ensure_ascii=False))
    Path(output).write_text(html)
    print(f"wrote {output} ({Path(output).stat().st_size/1e6:.1f} MB), {len(images)} frames")


TEMPLATE = r"""<title>__TITLE__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: summary strip, arena canvas beside error chart, camera table, frame gallery */
:root{--bg:#f6f7f4;--panel:#ffffff;--fg:#1c2420;--muted:#5d6a63;--line:#d9ded9;--tag:#1f7fc2;--opp:#d9672a;--truth:#2a3a32;--ok:#2e8b57;--bad:#c0392b;
 --sans:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14181a;--panel:#1d2326;--fg:#e6ebe7;--muted:#98a59d;--line:#2f383b;--tag:#5db4ee;--opp:#f0915a;--truth:#c9d3cc;--ok:#5fbf85;--bad:#e6685a;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#14181a;--panel:#1d2326;--fg:#e6ebe7;--muted:#98a59d;--line:#2f383b;--tag:#5db4ee;--opp:#f0915a;--truth:#c9d3cc;--ok:#5fbf85;--bad:#e6685a;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:15px;line-height:1.45}
.wrap{max-width:1180px;margin:0 auto;padding-block:20px 40px;padding-inline:16px;display:grid;gap:22px}
h1{font-size:1.45rem;font-weight:600;margin:0;text-wrap:balance}h2{font-size:1.05rem;font-weight:600;margin:0 0 8px}
.sub{color:var(--muted);font-family:var(--mono);font-size:.8rem;overflow-wrap:anywhere}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px;min-width:0}
.tile h2{display:flex;justify-content:space-between;align-items:center}
.pill{font-family:var(--mono);font-size:.72rem;padding:2px 8px;border-radius:999px;color:#fff}
.pill.ok{background:var(--ok)}.pill.bad{background:var(--bad)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-family:var(--mono);font-size:.82rem;font-variant-numeric:tabular-nums}
.kv span:first-child{color:var(--muted)}
.row{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:14px}
@media (max-width:820px){.row{grid-template-columns:1fr}}
canvas{width:100%;height:auto;display:block;background:var(--panel);border:1px solid var(--line);border-radius:6px}
.ctl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-family:var(--mono);font-size:.8rem;margin-top:8px}
.ctl input[type=range]{flex:1;min-width:140px}
button{font:inherit;padding:4px 12px;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:4px;cursor:pointer}
button:focus-visible{outline:2px solid var(--tag)}
table{border-collapse:collapse;font-family:var(--mono);font-size:.78rem;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
.tw{overflow-x:auto}
.legend{display:flex;gap:16px;font-family:var(--mono);font-size:.75rem;color:var(--muted);flex-wrap:wrap}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}
.gal{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.fig{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px;min-width:0}
.fig img{width:100%;height:auto;border-radius:3px}
.fig .cap{font-family:var(--mono);font-size:.74rem;color:var(--muted);margin-top:6px;overflow-wrap:anywhere}
.note{color:var(--muted);font-size:.85rem;max-width:70ch}
@media (prefers-reduced-motion:reduce){*{animation:none!important}}
</style>
<div class="wrap">
<div><h1>__TITLE__</h1><div class="sub" id="sub"></div></div>
<div class="tiles" id="tiles"></div>
<div class="row">
 <div class="tile"><h2>Арена, вид сверху</h2>
  <canvas id="arena" width="900" height="900"></canvas>
  <div class="ctl"><button id="play">▶ пуск</button><input type="range" id="time" min="0" value="0" step="1"><span id="tlabel"></span></div>
  <div class="legend"><span><i class="sw" style="background:var(--tag)"></i>tag_rover</span><span><i class="sw" style="background:var(--opp)"></i>opponent</span><span>крест — эталон, кольцо — оценка тракта, хвост — последние 2 с</span><span>□ камеры, ◇ лидар</span></div>
 </div>
 <div class="tile"><h2>Ошибка XY по времени, см</h2>
  <canvas id="err" width="900" height="520"></canvas>
  <h2 style="margin-top:14px">Лидар и источники измерений</h2><div class="kv" id="src"></div>
 </div>
</div>
<div class="tile"><h2>Камеры: режимы окон и время кадра</h2><div class="tw"><table id="cams"></table></div>
 <p class="note">acquire — полный кадр, roi — окно по прогнозу, idle — прогноз вне камеры, watchdog — контрольный полный кадр. Время кадра здесь — этот ноутбук, 6 потоков; на RTX-машине будет иначе.</p></div>
<div><h2>Кадры с окнами и проекцией эталона</h2><p class="note">Синее окно — где искали маркер, оранжевое — где искали соперника. Крест — эталон, спроецированный через модель линзы; кольцо — выход тракта в тот же момент. Подпись — ошибка XY в этот момент.</p>
 <div class="gal" id="gal"></div></div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
document.getElementById('sub').textContent=`реплей ${D.runtime} · датасет ${D.dataset}`+(D.timing.realtime_ratio?` · реплей ${D.timing.realtime_ratio.toFixed(2)}× сим-времени`:'');
const tiles=document.getElementById('tiles');
for(const [name,s] of Object.entries(D.summary)){
 const t=document.createElement('div');t.className='tile';
 const rows=[['XY P50 / P95',s.p50_cm!=null?`${s.p50_cm} / ${s.p95_cm} см`:'нет выходов'],['XY max',s.max_cm!=null?`${s.max_cm} см`:'—'],['valid по сим-времени',`${s.valid_pct} %`],['yaw P95',s.yaw_p95!=null?`${s.yaw_p95}°`:'не публикуется'],['возраст измерения P95',s.age_p95!=null?`${s.age_p95} мс`:'—'],['подмены >100 мс',String(s.swaps)]];
 t.innerHTML=`<h2>${name}<span class="pill ${s.pass?'ok':'bad'}">${s.pass?'пороги README пройдены':'пороги не пройдены'}</span></h2><div class="kv">${rows.map(r=>`<span>${r[0]}</span><span>${r[1]}</span>`).join('')}</div>`;
 tiles.appendChild(t);}
const src=document.getElementById('src');
const parts=[];
for(const [name,o] of Object.entries(D.observations)){for(const [m,n] of Object.entries(o.method||{})){parts.push([`${name} · ${m}`,`${n}`]);}parts.push([`${name} · принято фильтром`,`${o.accepted} из ${o.total}`]);}
for(const [k,v] of Object.entries(D.lidar))parts.push([`лидар · ${k}`,String(v)]);
src.innerHTML=parts.map(p=>`<span>${p[0]}</span><span>${p[1]}</span>`).join('');
const cams=document.getElementById('cams');
const modes=['acquire','roi','roi_exhausted_base','idle','watchdog'];
cams.innerHTML=`<tr><th>камера</th><th>кадров</th>${modes.map(m=>`<th>${m}</th>`).join('')}<th>кадр P95, мс</th><th>отказы PnP</th></tr>`+
 Object.entries(D.camera_frames).map(([c,v])=>`<tr><td>${c}</td><td>${v.frames}</td>${modes.map(m=>`<td>${v.mode_share[m]!=null?(100*v.mode_share[m]).toFixed(1)+' %':'—'}</td>`).join('')}<td>${v.latency_p95!=null?v.latency_p95.toFixed(1):'—'}</td><td style="white-space:normal;text-align:left">${Object.entries(v.pnp_rejections).map(([k,n])=>`${k} ${n}`).join(', ')||'—'}</td></tr>`).join('');
const gal=document.getElementById('gal');
for(const im of D.images){const f=document.createElement('div');f.className='fig';f.innerHTML=`<img alt="${im.camera} t=${im.t}" src="data:image/jpeg;base64,${im.jpeg}"><div class="cap">${im.camera} · t=${im.t} с · маркер: ${im.mode}, детекций ${im.detections} · соперник: ${im.opponent_mode||'—'} / ${im.opponent_method||'—'}<br>${im.labels.join(' · ')}</div>`;gal.appendChild(f);}
// timeline
const names=Object.keys(D.series);const S=D.series;const T=S[names[0]].map(r=>r[0]);
const slider=document.getElementById('time');slider.max=T.length-1;const tl=document.getElementById('tlabel');
const A=document.getElementById('arena'),ac=A.getContext('2d');const E=document.getElementById('err'),ec=E.getContext('2d');
const M=50,W=A.width-2*M,sc=W/12;const px=x=>M+x*sc,py=y=>A.height-M-y*sc;
function drawArena(i){ac.clearRect(0,0,A.width,A.height);ac.fillStyle=css('--panel');ac.fillRect(0,0,A.width,A.height);
 ac.strokeStyle=css('--line');ac.lineWidth=1;for(let g=0;g<=12;g++){ac.beginPath();ac.moveTo(px(g),py(0));ac.lineTo(px(g),py(12));ac.stroke();ac.beginPath();ac.moveTo(px(0),py(g));ac.lineTo(px(12),py(g));ac.stroke();}
 ac.strokeStyle=css('--fg');ac.lineWidth=2;ac.strokeRect(px(0),py(12),12*sc,12*sc);
 ac.fillStyle=css('--muted');ac.font='13px '+css('--mono');for(let g=0;g<=12;g+=2){ac.fillText(g+' м',px(g)-8,A.height-M+20);ac.fillText(g+' м',8,py(g)+4);}
 for(const c of D.cameras){ac.strokeStyle=css('--muted');ac.lineWidth=1.5;ac.strokeRect(px(c.xy[0])-7,py(c.xy[1])-7,14,14);ac.fillText(c.id.replace('camera_','K'),px(c.xy[0])+10,py(c.xy[1])-8);}
 if(D.lidar_xy[0]!=null){ac.save();ac.translate(px(D.lidar_xy[0]),py(D.lidar_xy[1]));ac.rotate(Math.PI/4);ac.strokeRect(-6,-6,12,12);ac.restore();}
 for(const n of names){const col=n==='tag_rover'?css('--tag'):css('--opp');const s=S[n];const j0=Math.max(0,i-100);
  ac.strokeStyle=col;ac.lineWidth=1.2;ac.globalAlpha=.5;ac.beginPath();let started=false;
  for(let j=j0;j<=i;j++){const e=s[j][1];if(!e){started=false;continue;}if(!started){ac.moveTo(px(e[0]),py(e[1]));started=true;}else ac.lineTo(px(e[0]),py(e[1]));}
  ac.stroke();ac.globalAlpha=1;
  const r=s[i];if(!r)continue;const g=r[2];ac.strokeStyle=col;ac.lineWidth=2.5;ac.beginPath();ac.moveTo(px(g[0])-12,py(g[1]));ac.lineTo(px(g[0])+12,py(g[1]));ac.moveTo(px(g[0]),py(g[1])-12);ac.lineTo(px(g[0]),py(g[1])+12);ac.stroke();
  const yaw=r[6];ac.beginPath();ac.moveTo(px(g[0]),py(g[1]));ac.lineTo(px(g[0]+0.5*Math.cos(yaw)),py(g[1]+0.5*Math.sin(yaw)));ac.stroke();
  if(r[1]){ac.beginPath();ac.arc(px(r[1][0]),py(r[1][1]),10,0,2*Math.PI);ac.stroke();
   ac.fillStyle=col;ac.font='600 14px '+css('--mono');ac.fillText(`${n} ${(r[3]*100).toFixed(1)} см · ${r[5]}`,px(r[1][0])+14,py(r[1][1])+5);}
  else{ac.fillStyle=col;ac.font='600 14px '+css('--mono');ac.fillText(`${n} ${r[5]||'нет выхода'}`,px(g[0])+14,py(g[1])+5);}}
}
function drawErr(i){const w=E.width,h=E.height,m=42;ec.clearRect(0,0,w,h);ec.fillStyle=css('--panel');ec.fillRect(0,0,w,h);
 const ymax=Math.max(5,...names.flatMap(n=>S[n].map(r=>r[3]||0)))*100;const yTop=Math.ceil(ymax/5)*5;
 const X=j=>m+(j/(T.length-1))*(w-2*m),Y=v=>h-m-(v/yTop)*(h-2*m);
 ec.strokeStyle=css('--line');ec.fillStyle=css('--muted');ec.font='12px '+css('--mono');
 for(let v=0;v<=yTop;v+=yTop/5){ec.beginPath();ec.moveTo(m,Y(v));ec.lineTo(w-m,Y(v));ec.stroke();ec.fillText(v.toFixed(0),6,Y(v)+4);}
 for(let k=0;k<=4;k++){const j=Math.round(k*(T.length-1)/4);ec.fillText(T[j].toFixed(0)+' с',X(j)-12,h-m+18);}
 ec.beginPath();ec.moveTo(m,Y(15));ec.lineTo(w-m,Y(15));ec.strokeStyle=css('--bad');ec.setLineDash([5,5]);ec.stroke();ec.setLineDash([]);ec.fillStyle=css('--bad');ec.fillText('порог P95 15 см',w-m-120,Y(15)-5);
 for(const n of names){const col=n==='tag_rover'?css('--tag'):css('--opp');ec.strokeStyle=col;ec.lineWidth=1.2;ec.beginPath();let st=false;
  S[n].forEach((r,j)=>{if(r[3]==null){st=false;return;}const y=Y(Math.min(r[3]*100,yTop));if(!st){ec.moveTo(X(j),y);st=true;}else ec.lineTo(X(j),y);});ec.stroke();}
 ec.strokeStyle=css('--fg');ec.lineWidth=1;ec.beginPath();ec.moveTo(X(i),m);ec.lineTo(X(i),h-m);ec.stroke();}
let i=0,playing=false,timer=null;
function show(k){i=k;slider.value=k;tl.textContent=`t = ${T[k].toFixed(2)} с`;drawArena(k);drawErr(k);}
slider.addEventListener('input',e=>{show(+e.target.value);});
document.getElementById('play').addEventListener('click',e=>{playing=!playing;e.target.textContent=playing?'∎ стоп':'▶ пуск';if(playing){timer=setInterval(()=>{show((i+2)%T.length);},40);}else clearInterval(timer);});
show(Math.min(400,T.length-1));
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>show(i));
</script>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runtime", type=Path)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default="Реплей арены IMX219")
    p.add_argument("--at", action="append", default=[], metavar="camera_N:seconds",
                   help="force a frame of this camera at this sim time into the gallery")
    a = p.parse_args()
    at = [(item.split(":")[0], float(item.split(":")[1])) for item in a.at]
    build(a.runtime, a.truth, a.dataset, a.output, a.title, at)


if __name__ == "__main__":
    main()
