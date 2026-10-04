// The node's settings page, served at "/": live picture, histogram, region
// statistics for tuning exposure on the arena, and the stream settings.
#pragma once

namespace mocap {

inline const char* kWebPage = R"HTML(<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Camera Setup</title>
<style>
:root{--bg:#f5f6f8;--fg:#14171c;--mut:#5d6673;--card:#fff;--line:#d8dce2;--acc:#2563eb;--warn:#c2410c;--ok:#15803d;--hist:#64748b}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f1216;--fg:#e7e9ec;--mut:#98a1ad;--card:#181c22;--line:#2a3038;--acc:#60a5fa;--warn:#fb923c;--ok:#4ade80;--hist:#94a3b8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{padding:12px 16px;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}h1{font-size:17px;margin:0}
.mut{color:var(--mut);font-size:12px}
main{display:grid;gap:14px;padding:0 16px 16px;grid-template-columns:minmax(0,1fr) 340px}
@media (max-width:980px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.view{position:relative}.view img{width:100%;display:block;border-radius:6px;background:#000}
.view canvas{position:absolute;left:12px;top:12px;cursor:crosshair}
h2{font-size:13px;margin:0 0 8px;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.row{display:grid;grid-template-columns:110px 1fr 90px;gap:8px;align-items:center;margin:6px 0}
.row label{font-size:13px}input[type=range]{width:100%}
input[type=number],select{width:100%;font:inherit;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
.val{font-variant-numeric:tabular-nums;font-size:12px;color:var(--mut);text-align:right}
button{font:inherit;padding:7px 12px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.k{font-size:11px;color:var(--mut)}
.n{font-size:18px;font-weight:600;font-variant-numeric:tabular-nums}
.warn{color:var(--warn)}.ok{color:var(--ok)}
#hist{width:100%;height:90px;display:block}
.below{display:grid;gap:14px;grid-template-columns:1fr 1fr;margin-top:12px}@media (max-width:700px){.below{grid-template-columns:1fr}}
#msg{min-height:1.4em}
</style></head><body>
<header><h1 id="title">Камера</h1><span class="mut" id="mode"></span></header>
<main>
<section class="card">
 <div class="view"><img id="img" alt="кадр камеры"><canvas id="ov"></canvas></div>
 <p class="mut">Выделите мышью область (например, маркер или пол арены), чтобы видеть её статистику. Картинка — то, что уходит в кодер, уменьшенная для просмотра.</p>
 <div class="below">
  <div><h2>Гистограмма кадра</h2><canvas id="hist" width="400" height="90"></canvas>
   <div class="grid" style="margin-top:8px">
    <div><div class="k">среднее</div><div class="n" id="mean">—</div></div>
    <div><div class="k">99-й перц.</div><div class="n" id="p99">—</div></div>
    <div><div class="k">пересвет ≥250</div><div class="n" id="sat">—</div></div>
    <div><div class="k">тёмное ≤5</div><div class="n" id="dark">—</div></div></div></div>
  <div><h2>Выделенная область</h2><div id="roihint" class="mut">не выбрана</div>
   <div class="grid" style="margin-top:8px">
    <div><div class="k">среднее</div><div class="n" id="rmean">—</div></div>
    <div><div class="k">P5 / P95</div><div class="n" id="rp">—</div></div>
    <div><div class="k">контраст</div><div class="n" id="rc">—</div></div>
    <div><div class="k">пересвет</div><div class="n" id="rsat">—</div></div></div>
   <p class="mut">Для маркера: пересвет 0 %, контраст как можно выше, P95 около 200–235.</p></div>
 </div>
</section>
<section>
 <div class="card"><h2>Экспозиция</h2>
  <div class="row"><label for="exp">Выдержка, мкс</label><input type="range" id="expr" min="20" max="5000" step="1"><input type="number" id="exp" min="20" step="10"></div>
  <div class="row"><span></span><span class="mut" id="blur"></span><span class="val" id="expv"></span></div>
  <div class="row"><label for="gain">Усиление</label><input type="range" id="gainr" min="1" max="10.67" step="0.01"><input type="number" id="gain" min="1" max="10.67" step="0.05"></div>
  <div class="row"><span></span><span></span><span class="val" id="gainv"></span></div>
 </div>
 <div class="card" style="margin-top:14px"><h2>Поток</h2>
  <div class="row"><label for="res">Разрешение</label><select id="res"></select><span class="val">весь кадр</span></div>
  <div class="row"><label for="fps">Кадров/с</label><input type="number" id="fps" min="5" max="83" step="1"><span class="val" id="fpsv"></span></div>
  <div class="row"><label for="br">Битрейт, Мбит/с</label><input type="number" id="br" min="1" max="40" step="1"><span class="val" id="brv"></span></div>
  <div class="row"><label for="color">Цвет</label><span><input type="checkbox" id="color"> <span class="mut">иначе серое</span></span><span></span></div>
  <div class="row"><label for="rg">Баланс: красный</label><input type="range" id="rgr" min="0.25" max="4" step="0.01"><input type="number" id="rg" min="0.25" max="4" step="0.01"></div>
  <div class="row"><label for="bg">Баланс: синий</label><input type="range" id="bgr" min="0.25" max="4" step="0.01"><input type="number" id="bg" min="0.25" max="4" step="0.01"></div>
  <p style="display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 0"><button id="wbroi">Баланс белого по области</button><button id="wball">по всему кадру</button></p>
  <p class="mut">Выделите на картинке белый или серый лист на арене и нажмите «по области». Баланс влияет только на цвет, не на яркость (маркер читается по яркости).</p>
  <p style="display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 0"><button class="primary" id="save">Сохранить в конфиг</button><button id="key">Опорный кадр</button></p>
  <p class="mut" id="msg"></p>
 </div>
 <div class="card" style="margin-top:14px"><h2>Узел сейчас</h2>
  <div class="grid">
   <div><div class="k">к/с</div><div class="n" id="sfps">—</div></div>
   <div><div class="k">Мбит/с</div><div class="n" id="smb">—</div></div>
   <div><div class="k">кодер, мс</div><div class="n" id="senc">—</div></div>
   <div><div class="k">эксп.→отпр., мс</div><div class="n" id="slat">—</div></div>
   <div><div class="k">CPU %</div><div class="n" id="scpu">—</div></div>
   <div><div class="k">°C</div><div class="n" id="stemp">—</div></div>
   <div><div class="k">пропуски</div><div class="n" id="sdrop">—</div></div>
   <div><div class="k">клиент</div><div class="n" id="scli">—</div></div>
   <div style="grid-column:1/-1"><div class="k">PTP</div><div class="n" id="sptp" style="font-size:14px">—</div></div></div>
 </div>
</section>
</main>
<script>
const $=id=>document.getElementById(id);let roi=null,drag=null,state=null,editing=0,pend={},timer=null;
const RES=[[1640,1232],[1280,960],[1024,768],[820,616],[640,480]];
RES.forEach(([w,h])=>{const o=document.createElement('option');o.value=w+'x'+h;o.textContent=w+'×'+h+(w==1640?' (полное)':w==820?' (2×2 ячейка)':'');$('res').appendChild(o)});
function send(){const body=JSON.stringify(pend);pend={};fetch('/api/apply',{method:'POST',body}).then(r=>r.json()).then(s=>{state=s;$('msg').textContent=s.message||'';$('msg').className='mut'+(s.error?' warn':'');show(s,true)}).catch(e=>$('msg').textContent='ошибка: '+e)}
function queue(k,v){pend[k]=v;editing=Date.now();clearTimeout(timer);timer=setTimeout(send,250)}
function pair(r,n,k,f=Number){$(r).oninput=()=>{$(n).value=$(r).value;queue(k,f($(r).value))};$(n).onchange=()=>{$(r).value=$(n).value;queue(k,f($(n).value))}}
pair('expr','exp','exposure_us',v=>Math.round(v));pair('gainr','gain','gain',Number);pair('rgr','rg','red_gain',Number);pair('bgr','bg','blue_gain',Number);
function wb(r){fetch('/api/apply',{method:'POST',body:JSON.stringify({wb_roi:r})}).then(x=>x.json()).then(s=>{$('msg').textContent=s.message||'';$('msg').className='mut'+(s.error?' warn':' ok');show(s,true)})}
$('wbroi').onclick=()=>{if(!roi){$('msg').textContent='сначала выделите область';return}wb([roi.x,roi.y,roi.w,roi.h])};$('wball').onclick=()=>wb([]);
$('fps').onchange=()=>queue('fps',Number($('fps').value));$('br').onchange=()=>queue('bitrate',Math.round(Number($('br').value)*1e6));
$('res').onchange=()=>{const[w,h]=$('res').value.split('x').map(Number);queue('width',w);pend.height=h};
$('color').onchange=()=>queue('color',$('color').checked);
$('save').onclick=()=>fetch('/api/save',{method:'POST'}).then(r=>r.json()).then(s=>{$('msg').textContent=s.message;$('msg').className='mut'+(s.error?' warn':' ok')});
$('key').onclick=()=>fetch('/api/keyframe',{method:'POST'});
function f1(x){return x==null?'—':(+x).toFixed(1)}
function show(s,force){
 const st=s.settings;$('title').textContent=s.camera_id;$('mode').textContent=s.mode;
 if(force||Date.now()-editing>1500){$('exp').value=st.exposure_us;$('expr').max=s.max_exposure_us;$('expr').value=st.exposure_us;$('gain').value=st.gain.toFixed(2);$('gainr').value=st.gain;
  $('fps').value=st.fps;$('br').value=st.bitrate/1e6;$('res').value=st.width+'x'+st.height;$('color').checked=st.color;
  $('rg').value=st.red_gain.toFixed(2);$('rgr').value=st.red_gain;$('bg').value=st.blue_gain.toFixed(2);$('bgr').value=st.blue_gain}
 $('expv').textContent='сенсор: '+s.actual.exposure_us+' мкс';$('gainv').textContent='сенсор: ×'+s.actual.gain.toFixed(2);
 $('fpsv').textContent=s.actual.fps.toFixed(1);$('brv').textContent='';
 const b=11*s.actual.exposure_us/1000;$('blur').innerHTML='смаз при 11 м/с: <b class="'+(b>11?'warn':'')+'">'+b.toFixed(1)+' мм</b>';
 const im=s.image;if(im){$('mean').textContent=f1(im.mean);$('p99').textContent=im.p99;$('sat').textContent=f1(im.saturated_pct)+'%';$('sat').className='n'+(im.saturated_pct>0.5?' warn':'');$('dark').textContent=f1(im.dark_pct)+'%';drawHist(im.hist)}
 const r=s.roi;if(r){$('roihint').textContent='область '+r.w+'×'+r.h+' пикс. кадра';$('rmean').textContent=f1(r.mean);$('rp').textContent=r.p5+' / '+r.p95;$('rc').textContent=(r.contrast*100).toFixed(0)+'%';$('rsat').textContent=f1(r.saturated_pct)+'%';$('rsat').className='n'+(r.saturated_pct>0?' warn':'')}
 const n=s.status||{};$('sfps').textContent=n.encoded??'—';$('smb').textContent=f1(n.mbit_s);$('senc').textContent=n.encoder_ms?f1(n.encoder_ms.p50):'—';
 $('slat').textContent=n.exp_to_encoded_ms?f1(n.exp_to_encoded_ms.p50)+'/'+f1(n.exp_to_encoded_ms.p95):'—';$('scpu').textContent=f1(n.cpu_percent);$('stemp').textContent=f1(n.temp_c);
 $('sdrop').textContent=n.frames!=null?(n.dropped_prep+n.dropped_encoder+n.dropped_link):'—';$('scli').textContent=s.client?'есть':'нет';
 const p=n.ptp||{};$('sptp').textContent=p.state?(p.state+(p.offset_ns!=null?', смещение '+(p.offset_ns/1000).toFixed(1)+' мкс':'')):'—';
 $('sptp').className='n'+((p.state==='SLAVE'&&Math.abs(p.offset_ns||0)<10000)||p.state==='MASTER'?' ok':' warn')}
function drawHist(h){const c=$('hist'),g=c.getContext('2d'),W=c.width,H=c.height;g.clearRect(0,0,W,H);const cs=getComputedStyle(document.documentElement);
 const m=Math.max(...h.map(v=>Math.sqrt(v)))||1,bw=W/h.length;h.forEach((v,i)=>{g.fillStyle=i==h.length-1&&v>0?cs.getPropertyValue('--warn'):cs.getPropertyValue('--hist');const y=Math.sqrt(v)/m*(H-4);g.fillRect(i*bw,H-y,bw-1,y)})}
function poll(){const q=roi?`?roi=${roi.x.toFixed(4)},${roi.y.toFixed(4)},${roi.w.toFixed(4)},${roi.h.toFixed(4)}`:'';
 fetch('/api/state'+q).then(r=>r.json()).then(s=>{state=s;show(s)}).catch(()=>{$('msg').textContent='нет связи с узлом'}).finally(()=>setTimeout(poll,500))}
const img=$('img'),ov=$('ov');function next(){img.src='/preview.jpg?'+Date.now()}img.onload=()=>{sizeOv();setTimeout(next,150)};img.onerror=()=>setTimeout(next,1000);
function sizeOv(){ov.width=img.clientWidth;ov.height=img.clientHeight;ov.style.width=img.clientWidth+'px';ov.style.height=img.clientHeight+'px';drawRoi()}
function drawRoi(){const g=ov.getContext('2d');g.clearRect(0,0,ov.width,ov.height);const r=drag||roi;if(!r)return;g.strokeStyle='#22d3ee';g.lineWidth=2;g.strokeRect(r.x*ov.width,r.y*ov.height,r.w*ov.width,r.h*ov.height)}
ov.onmousedown=e=>{const b=ov.getBoundingClientRect();drag={x0:(e.clientX-b.left)/b.width,y0:(e.clientY-b.top)/b.height,x:0,y:0,w:0,h:0}};
ov.onmousemove=e=>{if(!drag)return;const b=ov.getBoundingClientRect(),x=(e.clientX-b.left)/b.width,y=(e.clientY-b.top)/b.height;Object.assign(drag,{x:Math.min(x,drag.x0),y:Math.min(y,drag.y0),w:Math.abs(x-drag.x0),h:Math.abs(y-drag.y0)});drawRoi()};
ov.onmouseup=()=>{if(drag&&drag.w>0.01&&drag.h>0.01)roi={x:drag.x,y:drag.y,w:drag.w,h:drag.h};else roi=null;drag=null;drawRoi();if(!roi)$('roihint').textContent='не выбрана'};
window.onresize=sizeOv;next();poll();
</script></body></html>)HTML";

}  // namespace mocap
