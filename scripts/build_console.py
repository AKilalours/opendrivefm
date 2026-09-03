#!/usr/bin/env python3
"""Emit the OpenDriveFM validation console as one self-contained HTML file."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "outputs/console/bundle.json"
OUT = ROOT / "outputs/console/index.html"

HEAD = r"""<title>OpenDriveFM Validation Console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{
  --bg:#0A0C11; --surf:#12161F; --surf2:#171C27; --line:#232A36; --line2:#2E3747;
  --ink:#E6EBF3; --dim:#8A97AB; --dim2:#5C687C;
  --acc:#5BD2E8; --good:#7EE2A8; --warn:#FF9C5C; --bad:#FF6A7A; --viol:#B49BFF;
  --rail:56px; --bar:52px;
  color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:13px;line-height:1.45;-webkit-font-smoothing:antialiased}
.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
button{font:inherit;color:inherit;background:none;border:none;cursor:pointer}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

/* ---------- command bar ---------- */
.bar{position:sticky;top:0;z-index:40;height:var(--bar);display:flex;align-items:center;
  gap:22px;padding:0 16px;background:#0C1017;border-bottom:1px solid var(--line)}
.brand{font-weight:700;letter-spacing:.02em;font-size:15px}
.brand span{color:var(--acc)}
.grp{display:flex;align-items:center;gap:7px}
.lab{font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim2)}
.pill{padding:3px 9px;border:1px solid var(--line2);border-radius:3px;background:var(--surf);
  font-size:11.5px}
.pill.on{border-color:var(--acc);color:var(--acc);background:rgba(91,210,232,.09)}
.step{width:24px;height:24px;border:1px solid var(--line2);border-radius:3px;background:var(--surf);
  display:grid;place-items:center;font-size:11px}
.step:hover{border-color:var(--acc);color:var(--acc)}
.live{display:flex;align-items:center;gap:6px;margin-left:auto;font-size:11px;color:var(--dim)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--good)}

/* ---------- shell ---------- */
.shell{display:flex;min-height:calc(100vh - var(--bar))}
.rail{width:var(--rail);flex:0 0 var(--rail);background:#0C1017;border-right:1px solid var(--line);
  display:flex;flex-direction:column;align-items:center;padding-top:8px;gap:2px;position:sticky;
  top:var(--bar);height:calc(100vh - var(--bar))}
.rb{width:44px;padding:8px 0;border-radius:4px;font-size:8.5px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--dim2);text-align:center;line-height:1.25}
.rb:hover{background:var(--surf);color:var(--dim)}
.rb.on{background:rgba(91,210,232,.12);color:var(--acc)}
main{flex:1;min-width:0;padding:14px 16px 40px}
.page{display:none}.page.on{display:block}

/* ---------- primitives ---------- */
.sec{display:flex;align-items:baseline;gap:12px;margin:0 0 8px}
.sec h2{margin:0;font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:var(--dim)}
.sec .note{font-size:11px;color:var(--dim2)}
.card{background:var(--surf);border:1px solid var(--line);border-radius:6px}
.pad{padding:12px 14px}
table{border-collapse:collapse;width:100%}
th{font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--dim2);
  text-align:left;font-weight:500;padding:0 10px 6px 0;border-bottom:1px solid var(--line)}
td{padding:5px 10px 5px 0;font-family:"IBM Plex Mono",monospace;font-size:12px;
  font-variant-numeric:tabular-nums;border-bottom:1px solid rgba(35,42,54,.5)}
tr:last-child td{border-bottom:none}
.g{color:var(--good)}.w{color:var(--warn)}.b{color:var(--bad)}.a{color:var(--acc)}.d{color:var(--dim)}

/* ---------- cameras ---------- */
.cams{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.cam{position:relative;background:#000;border:1px solid var(--line);border-radius:4px;overflow:hidden;
  aspect-ratio:16/9}
.cam img{width:100%;height:100%;display:block;object-fit:cover}
.cam .tag{position:absolute;left:0;top:0;padding:3px 7px;background:rgba(10,12,17,.82);
  font-size:9px;letter-spacing:.11em;color:var(--dim);font-family:"IBM Plex Mono",monospace}
.cam .vis{position:absolute;right:0;top:0;padding:3px 7px;background:rgba(10,12,17,.82);
  font-size:9px;color:var(--dim2);font-family:"IBM Plex Mono",monospace}
.cam .hit{position:absolute;border:2px solid var(--acc);border-radius:2px;pointer-events:none;
  box-shadow:0 0 0 1px rgba(10,12,17,.8) inset;display:none}
.cam .hit.on{display:block}
.cam .zone{position:absolute;cursor:pointer}

/* ---------- maps ---------- */
.maps{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.mapwrap{position:relative;background:#0B0E14;border:1px solid var(--line);border-radius:5px;
  overflow:hidden}
.mapwrap img{width:100%;display:block}
.mapwrap .cap{display:flex;justify-content:space-between;align-items:baseline;
  padding:7px 10px;border-top:1px solid var(--line);background:var(--surf)}
.mapwrap .cap b{font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;font-weight:600}
.mark{position:absolute;width:22px;height:22px;margin:-11px 0 0 -11px;border:2px solid var(--acc);
  border-radius:50%;pointer-events:none;display:none}
.mark.on{display:block}

/* ---------- temporal ---------- */
.tl{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;align-items:start}
.tl figure{margin:0}
.tl img{width:100%;display:block;border:1px solid var(--line);border-radius:4px;background:#0B0E14}
.tl figcaption{display:flex;justify-content:space-between;padding:5px 2px 0;font-size:10.5px;
  color:var(--dim);font-family:"IBM Plex Mono",monospace}

/* ---------- kpi ---------- */
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:5px;overflow:hidden}
.kpi div{background:var(--surf);padding:9px 12px}
.kpi .k{font-size:9px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim2)}
.kpi .v{font-family:"IBM Plex Mono",monospace;font-size:19px;font-weight:600;
  font-variant-numeric:tabular-nums;margin-top:1px}
.kpi .u{font-size:10px;color:var(--dim2);margin-left:3px;font-weight:400}

/* ---------- bars ---------- */
.br{display:grid;grid-template-columns:132px 1fr 62px;align-items:center;gap:10px;margin:5px 0}
.br .t{font-size:11.5px;color:var(--dim)}
.br .track{height:11px;background:#1B212C;border-radius:2px;overflow:hidden}
.br .fill{height:100%;border-radius:2px}
.br .n{font-family:"IBM Plex Mono",monospace;font-size:11.5px;text-align:right}

/* ---------- inspector ---------- */
.insp{position:fixed;right:0;top:var(--bar);width:270px;height:calc(100vh - var(--bar));
  background:#0D1119;border-left:1px solid var(--line2);padding:14px;overflow:auto;
  transform:translateX(100%);transition:transform .16s ease;z-index:30}
.insp.on{transform:none}
.insp h3{margin:0 0 2px;font-size:15px}
.insp .cls{font-size:11px;color:var(--acc);letter-spacing:.1em;text-transform:uppercase}
.kv{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(35,42,54,.6);
  font-size:12px}
.kv span:first-child{color:var(--dim)}
.kv span:last-child{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.close{position:absolute;right:10px;top:10px;color:var(--dim2);font-size:15px}
.objlist{max-height:210px;overflow:auto;border:1px solid var(--line);border-radius:4px}
.objrow{display:grid;grid-template-columns:1fr 46px 46px 42px;gap:6px;padding:4px 8px;font-size:11px;
  font-family:"IBM Plex Mono",monospace;cursor:pointer;border-bottom:1px solid rgba(35,42,54,.5)}
.objrow:hover{background:var(--surf2)}
.objrow.on{background:rgba(91,210,232,.12);color:var(--acc)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.hr{height:1px;background:var(--line);margin:14px 0}
.warnbox{border-left:2px solid var(--warn);padding:8px 12px;background:rgba(255,156,92,.05);
  font-size:11.5px;color:var(--dim);border-radius:0 4px 4px 0}
.badbox{border-left:2px solid var(--bad);padding:8px 12px;background:rgba(255,106,122,.05);
  font-size:11.5px;color:var(--dim);border-radius:0 4px 4px 0}
@media(max-width:1100px){.maps{grid-template-columns:1fr}.cams{grid-template-columns:repeat(2,1fr)}}
</style>"""

BODY = r"""
<div class="bar">
  <div class="brand">OpenDrive<span>FM</span></div>
  <div class="grp"><span class="lab">Mode</span><span class="pill on">Replay</span></div>
  <div class="grp"><span class="lab">Dataset</span><span class="pill mono">nuScenes mini</span></div>
  <div class="grp"><span class="lab">Scene</span>
    <button class="step" id="sp">&#9664;</button>
    <span class="pill mono" id="sceneLbl">--</span>
    <button class="step" id="sn">&#9654;</button></div>
  <div class="grp"><span class="lab">Frame</span>
    <button class="step" id="fp">&#9664;</button>
    <span class="pill mono" id="frameLbl">--</span>
    <button class="step" id="fn">&#9654;</button>
    <button class="pill" id="play">&#9654; Play</button></div>
  <div class="grp"><span class="lab">Overlay</span>
    <button class="pill ov" data-ov="plain">Raw</button>
    <button class="pill ov" data-ov="lidar">LiDAR</button>
    <button class="pill ov on" data-ov="boxes">3D Boxes</button></div>
  <div class="live"><span class="dot"></span><span class="mono" id="ts">--</span></div>
</div>

<div class="shell">
  <nav class="rail" id="rail"></nav>
  <main>
    <!-- ============ OVERVIEW ============ -->
    <section class="page on" data-p="overview">
      <div class="sec"><h2>Sensor input</h2><span class="note" id="ovNote"></span></div>
      <div class="cams" id="cams"></div>

      <div class="sec" style="margin-top:16px"><h2>World state</h2>
        <span class="note">click any object to trace it across every view</span></div>
      <div class="maps" id="maps"></div>

      <div class="sec" style="margin-top:16px"><h2>Temporal &mdash; persistence forecast vs realised future</h2>
        <span class="note">green correct &middot; orange false occupied &middot; red missed</span></div>
      <div class="tl" id="tl"></div>

      <div style="margin-top:16px" class="kpi" id="kpi"></div>
    </section>

    <!-- ============ PERCEPTION ============ -->
    <section class="page" data-p="perception">
      <div class="sec"><h2>Multi-sweep BEV</h2><span class="note" id="pcNote"></span></div>
      <div class="grid2">
        <div class="mapwrap"><img id="pcBev" alt="BEV"><div class="cap">
          <b>Points + dynamic</b><span class="mono d" id="pcBevN"></span></div></div>
        <div class="mapwrap"><img id="pcOcc" alt="Occupancy"><div class="cap">
          <b>Ray-cast occupancy</b><span class="mono d" id="pcOccN"></span></div></div>
      </div>
      <div class="hr"></div>
      <div class="sec"><h2>Objects in frame</h2></div>
      <div class="objlist" id="objTable" style="max-height:none"></div>
    </section>

    <!-- ============ FORECAST ============ -->
    <section class="page" data-p="forecast">
      <div class="sec"><h2>Predicted vs ground truth</h2>
        <span class="note">persistence baseline &middot; ground truth is the LiDAR that actually arrived</span></div>
      <div id="fcRows"></div>
      <div class="hr"></div>
      <div class="grid2">
        <div class="card pad"><div class="sec"><h2>Occupancy IoU &mdash; 30 keyframes</h2></div>
          <table id="fcTable"></table>
          <div class="badbox" style="margin-top:10px">Persistence wins at every horizon.
            Recall is unchanged (0.4727 &rarr; 0.4726) while precision falls 0.576 &rarr; 0.496:
            advection moves correct cells to wrong places. Root cause is motion-label precision
            of 0.56, not the advection.</div></div>
        <div class="card pad"><div class="sec"><h2>Ego trajectory ADE &mdash; 404 keyframes</h2></div>
          <table id="adeTable"></table>
          <div class="warnbox" style="margin-top:10px">The GPT-2 checkpoint in this repo is
            <b>not evaluated</b>: it was fine-tuned from manifest keys that do not exist, so every
            waypoint fell back to (0,0) and it saw 404 copies of one all-zero trajectory. Real
            waypoints live in the label files. Retraining is the prerequisite for a learned number.</div></div>
      </div>
    </section>

    <!-- ============ MODELS ============ -->
    <section class="page" data-p="models">
      <div class="sec"><h2>Learned trajectory model vs geometric baselines</h2>
        <span class="note" id="mdNote"></span></div>
      <div class="grid2">
        <div class="card pad"><table id="mdAde"></table>
          <div id="mdBars" style="margin-top:12px"></div></div>
        <div class="card pad">
          <div class="sec"><h2>BEV occupancy head</h2></div>
          <table id="mdOcc"></table>
          <div class="warnbox" style="margin-top:10px" id="mdLoad"></div></div>
      </div>
      <div class="hr"></div>
      <div class="grid2">
        <div class="card pad"><div class="sec"><h2>VLA &mdash; vision to action through a language model</h2></div>
          <div class="mono" style="font-size:11px;color:var(--dim);line-height:1.7;margin-bottom:10px"
               id="vlaArch"></div>
          <table id="vlaTable"></table>
          <div id="vlaNotes" style="margin-top:10px"></div></div>
        <div class="card pad"><div class="sec"><h2>VLM &mdash; scene captioning</h2></div>
          <div id="vlmBox"></div></div>
      </div>
    </section>

    <!-- ============ INTEGRITY ============ -->
    <section class="page" data-p="integrity">
      <div class="grid2">
        <div class="mapwrap"><img id="inMap" alt="Integrity"><div class="cap">
          <b>Perception integrity</b><span class="mono d" id="inMean"></span></div></div>
        <div>
          <div class="card pad"><div class="sec"><h2>Per-camera visible fraction of frustum</h2></div>
            <div id="inCams"></div></div>
          <div class="card pad" style="margin-top:10px">
            <div class="sec"><h2>Validated against human visibility labels</h2></div>
            <div id="inAuroc"></div>
            <div class="badbox" style="margin-top:10px" id="inVerdict"></div></div>
        </div>
      </div>
      <div class="hr"></div>
      <div class="sec"><h2>Mean integrity by annotator visibility bucket</h2></div>
      <div class="card pad"><div id="inLevels"></div></div>
    </section>

    <!-- ============ VALIDATION ============ -->
    <section class="page" data-p="validation">
      <div class="grid2">
        <div class="card pad"><div class="sec"><h2>Motion separation</h2></div>
          <table id="vaMotion"></table></div>
        <div class="card pad"><div class="sec"><h2>Occupancy forecast</h2></div>
          <table id="vaFc"></table></div>
      </div>
      <div class="hr"></div>
      <div class="card pad"><div class="sec"><h2>Ego trajectory &mdash; all horizons</h2></div>
        <table id="vaAde"></table></div>
    </section>

    <!-- ============ FAILURES ============ -->
    <section class="page" data-p="failures">
      <div class="sec"><h2>Hard cases &mdash; ranked by integrity at the object footprint</h2>
        <span class="note">click to jump to the frame</span></div>
      <div class="card"><div class="objlist" id="hardList" style="max-height:none"></div></div>
      <div class="hr"></div>
      <div class="grid3" id="failStats"></div>
    </section>

    <!-- ============ PERFORMANCE ============ -->
    <section class="page" data-p="performance">
      <div class="sec"><h2>Per-keyframe latency</h2><span class="note" id="perfNote"></span></div>
      <div class="card pad"><div id="perfBars"></div></div>
      <div class="hr"></div>
      <div class="grid2">
        <div class="card pad"><div class="sec"><h2>Optimisation</h2></div><table id="perfTable"></table></div>
        <div class="card pad"><div class="sec"><h2>Conditions</h2></div>
          <div id="perfCond" style="font-size:11.5px;color:var(--dim);line-height:1.65"></div></div>
      </div>
    </section>

    <!-- ============ SYSTEM ============ -->
    <section class="page" data-p="system">
      <div class="sec"><h2>What runs, and what does not</h2>
        <span class="note">every row is a component that exists in this repository</span></div>
      <div class="card pad"><table id="sysTable"></table></div>
    </section>
  </main>
</div>

<aside class="insp" id="insp">
  <button class="close" id="inspClose">&times;</button>
  <div class="cls" id="iCls">&nbsp;</div>
  <h3 id="iId">&nbsp;</h3>
  <div id="iKv" style="margin-top:10px"></div>
  <div class="hr"></div>
  <div class="lab" style="margin-bottom:6px">Objects in frame</div>
  <div class="objlist" id="iList"></div>
</aside>
"""

SCRIPT = r"""
<script>
const D = window.__ODFM__;
const F = D.frames, R = D.reports;
const CAMS = ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"];
const PAGES = ["overview","perception","forecast","models","integrity","validation","failures","performance","system"];
let fi = 0, ov = "boxes", sel = null, timer = null;

const scenes = [...new Set(F.map(f => f.scene))];
const $ = s => document.querySelector(s);
const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c;
  if (h !== undefined) e.innerHTML = h; return e; };
const fx = (v, n = 3) => (v === undefined || v === null) ? "--" : Number(v).toFixed(n);
const pc = v => (100 * v).toFixed(1) + "%";

/* ---------- nav rail ---------- */
PAGES.forEach((p, i) => {
  const b = el("button", "rb" + (i === 0 ? " on" : ""), p);
  b.onclick = () => { document.querySelectorAll(".rb").forEach(x => x.classList.remove("on"));
    b.classList.add("on");
    document.querySelectorAll(".page").forEach(x => x.classList.toggle("on", x.dataset.p === p)); };
  $("#rail").appendChild(b);
});

/* ---------- camera grid ---------- */
CAMS.forEach(c => {
  const d = el("div", "cam");
  d.innerHTML = `<img data-cam="${c}" alt="${c}"><div class="tag">${c.replace("CAM_","")}</div>
    <div class="vis" data-vis="${c}"></div><div class="hit" data-hit="${c}"></div>`;
  $("#cams").appendChild(d);
});

/* ---------- map panels ---------- */
$("#maps").innerHTML = `
  <div class="mapwrap"><img id="ovBev" alt="BEV"><div class="mark" id="mkBev"></div>
    <div class="cap"><b>Multi-sweep BEV</b><span class="mono d" id="ovBevN"></span></div></div>
  <div class="mapwrap"><img id="ovInt" alt="Integrity"><div class="mark" id="mkInt"></div>
    <div class="cap"><b>Perception integrity</b><span class="mono d" id="ovIntN"></span></div></div>`;

/* ---------- transport ---------- */
function setFrame(i) {
  fi = (i + F.length) % F.length; sel = null; draw();
}
$("#fn").onclick = () => setFrame(fi + 1);
$("#fp").onclick = () => setFrame(fi - 1);
$("#sn").onclick = () => jumpScene(1);
$("#sp").onclick = () => jumpScene(-1);
function jumpScene(d) {
  const cur = scenes.indexOf(F[fi].scene);
  const s = scenes[(cur + d + scenes.length) % scenes.length];
  setFrame(F.findIndex(f => f.scene === s));
}
$("#play").onclick = e => {
  if (timer) { clearInterval(timer); timer = null; e.target.innerHTML = "&#9654; Play"; }
  else { timer = setInterval(() => setFrame(fi + 1), 900); e.target.innerHTML = "&#10074;&#10074; Pause"; }
};
document.querySelectorAll(".ov").forEach(b => b.onclick = () => {
  ov = b.dataset.ov;
  document.querySelectorAll(".ov").forEach(x => x.classList.toggle("on", x === b));
  draw();
});
$("#inspClose").onclick = () => { sel = null; draw(); };
addEventListener("keydown", e => {
  if (e.key === "ArrowRight") setFrame(fi + 1);
  if (e.key === "ArrowLeft") setFrame(fi - 1);
  if (e.key === "Escape") { sel = null; draw(); }
});

/* ---------- object selection ---------- */
function pick(id) { sel = (sel === id) ? null : id; draw(); }

function objRows(container, objs, compact) {
  container.innerHTML = "";
  objs.forEach(o => {
    const r = el("div", "objrow" + (sel === o.id ? " on" : ""));
    r.innerHTML = `<span>${o.cat}</span><span>${o.range}m</span>
      <span>${o.speed.toFixed(1)}</span><span>${fx(o.integrity,2)}</span>`;
    r.onclick = () => pick(o.id);
    container.appendChild(r);
  });
}

/* ---------- main draw ---------- */
function draw() {
  const f = F[fi];
  const idxInScene = F.filter(x => x.scene === f.scene).indexOf(f) + 1;
  const nInScene = F.filter(x => x.scene === f.scene).length;
  $("#sceneLbl").textContent = f.scene.replace("scene-", "");
  $("#frameLbl").textContent = String(idxInScene).padStart(2, "0") + " / " + nInScene;
  const t = new Date(f.timestamp_us / 1000);
  $("#ts").textContent = t.toISOString().substr(11, 12);

  CAMS.forEach(c => {
    const img = document.querySelector(`img[data-cam="${c}"]`);
    img.src = f.cameras[c][ov];
    document.querySelector(`[data-vis="${c}"]`).textContent = fx(f.cameras[c].visible_frac, 2);
    const hit = document.querySelector(`[data-hit="${c}"]`);
    const o = sel !== null ? f.objects.find(x => x.id === sel) : null;
    if (o && o.cams[c]) {
      const [x0, y0, x1, y1] = o.cams[c];
      hit.style.left = (100 * x0) + "%"; hit.style.top = (100 * y0) + "%";
      hit.style.width = (100 * (x1 - x0)) + "%"; hit.style.height = (100 * (y1 - y0)) + "%";
      hit.classList.add("on");
    } else hit.classList.remove("on");
  });

  $("#ovBev").src = f.maps.bev; $("#ovInt").src = f.maps.integrity;
  $("#pcBev").src = f.maps.bev; $("#pcOcc").src = f.maps.occupancy;
  $("#inMap").src = f.maps.integrity;
  const s = f.stats;
  $("#ovBevN").textContent = `${s.returns.toLocaleString()} returns · ${s.dynamic} dynamic`;
  $("#ovIntN").textContent = `mean over drivable ${fx(s.integrity_mean_drivable)}`;
  $("#pcBevN").textContent = `${s.returns.toLocaleString()} returns · 10 sweeps · ${s.dynamic} dynamic`;
  $("#pcOccN").textContent = `free ${pc(s.free)} · occ ${pc(s.occupied)} · unknown ${pc(s.unknown)}`;
  $("#inMean").textContent = `mean over drivable ${fx(s.integrity_mean_drivable)}`;
  $("#ovNote").textContent = `${s.boxes_observed} of ${s.boxes} annotated objects have LiDAR returns`;
  $("#pcNote").textContent = `ground plane tilt ${s.plane_tilt_deg}° · ${pc(s.ground_frac)} of returns are ground`;

  /* object marker on both maps */
  const o = sel !== null ? f.objects.find(x => x.id === sel) : null;
  [["#mkBev", "#ovBev"], ["#mkInt", "#ovInt"]].forEach(([mk, im]) => {
    const m = $(mk);
    if (!o) { m.classList.remove("on"); return; }
    const box = $(im).getBoundingClientRect();
    m.style.left = (box.width * (0.5 - o.y / (2 * f.rng_m))) + "px";
    m.style.top = (box.height * (0.5 - o.x / (2 * f.rng_m))) + "px";
    m.classList.add("on");
  });

  /* temporal strip */
  const tl = $("#tl"); tl.innerHTML = "";
  const cur = el("figure");
  cur.innerHTML = `<img src="${f.maps.occupancy}" alt="current occupancy">
    <figcaption><span>CURRENT</span><span>0.0s</span></figcaption>`;
  tl.appendChild(cur);
  f.forecast.forEach(h => {
    const g = el("figure");
    g.innerHTML = `<img src="${h.error}" alt="forecast error T+${h.h}">
      <figcaption><span>T+${h.h}</span><span>+${h.dt}s · IoU ${fx(h.iou)}</span></figcaption>`;
    tl.appendChild(g);
  });

  /* forecast page */
  const fr = $("#fcRows"); fr.innerHTML = "";
  f.forecast.forEach(h => {
    const row = el("div", "card pad");
    row.style.marginBottom = "10px";
    row.innerHTML = `<div class="sec"><h2>T+${h.h} &middot; +${h.dt}s</h2>
      <span class="note mono">IoU ${fx(h.iou)} &middot; precision ${fx(h.precision)} &middot; recall ${fx(h.recall)}</span></div>
      <div class="grid3">
        <figure style="margin:0"><img src="${h.pred}" style="width:100%;border-radius:4px">
          <figcaption class="lab" style="margin-top:5px">Predicted</figcaption></figure>
        <figure style="margin:0"><img src="${h.truth}" style="width:100%;border-radius:4px">
          <figcaption class="lab" style="margin-top:5px">Ground truth</figcaption></figure>
        <figure style="margin:0"><img src="${h.error}" style="width:100%;border-radius:4px">
          <figcaption class="lab" style="margin-top:5px">Error</figcaption></figure>
      </div>`;
    fr.appendChild(row);
  });

  /* kpi strip */
  const fc1 = R.occupancy_forecast_report?.results?.["T+1"];
  const ade = R.trajectory_ade_report?.results?.["T+1 (0.5s)"];
  const mo = R.motion_separation_report?.best;
  const iv = R.integrity_visibility_report?.auroc?.integrity_occlusion_aware;
  const tm = R.timing_report;
  const K = [
    ["Occupancy IoU T+1", fx(fc1?.persistence?.iou), ""],
    ["Forecast F1 T+1", fx(fc1?.persistence?.f1), ""],
    ["Ego ADE T+1", fx(R.learned_model_report?.trajectory_ade?.["T+1 (0.5s)"]?.learned_v11_temporal?.ade_m, 3), "m"],
    ["Motion F1", fx(mo?.f1), ""],
    ["Integrity AUROC", fx(iv?.auroc), ""],
    ["Integrity (frame)", fx(s.integrity_mean_drivable), ""],
    ["Latency", fx(tm?.total_ms_per_keyframe, 0), "ms"],
    ["Free space", pc(s.free), ""],
  ];
  $("#kpi").innerHTML = K.map(([k, v, u]) =>
    `<div><div class="k">${k}</div><div class="v">${v}<span class="u">${u}</span></div></div>`).join("");

  objRows($("#iList"), f.objects, true);
  objRows($("#objTable"), f.objects, false);

  /* inspector */
  const ins = $("#insp");
  if (o) {
    ins.classList.add("on");
    $("#iCls").textContent = o.group;
    $("#iId").textContent = o.cat + " #" + o.id;
    const visLbl = { 1: "0-40%", 2: "40-60%", 3: "60-80%", 4: "80-100%" }[o.vis] || "--";
    const seen = CAMS.filter(c => o.cams[c]).map(c => c.replace("CAM_", "")).join(", ") || "none";
    $("#iKv").innerHTML = [
      ["range", o.range + " m"], ["speed", o.speed.toFixed(2) + " m/s"],
      ["position", `${o.x}, ${o.y}`], ["heading", o.yaw.toFixed(2) + " rad"],
      ["footprint", `${o.wl[1]} × ${o.wl[0]} m`],
      ["lidar returns", o.pts],
      ["annotator visibility", visLbl],
      ["integrity", fx(o.integrity, 3)],
      ["seen by", seen],
    ].map(([k, v]) => `<div class="kv"><span>${k}</span><span>${v}</span></div>`).join("");
  } else ins.classList.remove("on");
}

/* ---------- static report-driven pages ---------- */
function bar(label, v, max, col) {
  return `<div class="br"><span class="t">${label}</span>
    <span class="track"><span class="fill" style="width:${Math.max(2, 100 * v / max)}%;background:${col}"></span></span>
    <span class="n">${typeof v === "number" ? v.toFixed(3) : v}</span></div>`;
}
function fillStatic() {
  const fcr = R.occupancy_forecast_report;
  if (fcr) {
    $("#fcTable").innerHTML = `<tr><th>horizon</th><th>persistence</th><th>const-vel</th><th>delta</th></tr>` +
      ["T+1", "T+2", "T+3"].map(h => { const r = fcr.results[h];
        return `<tr><td>${h}</td><td>${fx(r.persistence.iou)}</td><td>${fx(r.constant_velocity.iou)}</td>
          <td class="b">${r.relative_gain_pct.toFixed(1)}%</td></tr>`; }).join("");
    $("#vaFc").innerHTML = `<tr><th>horizon</th><th>IoU</th><th>prec</th><th>recall</th><th>F1</th></tr>` +
      ["T+1", "T+2", "T+3"].map(h => { const r = fcr.results[h].persistence;
        return `<tr><td>${h}</td><td>${fx(r.iou)}</td><td>${fx(r.precision)}</td>
          <td>${fx(r.recall)}</td><td>${fx(r.f1)}</td></tr>`; }).join("");
  }
  const ad = R.trajectory_ade_report;
  if (ad) {
    const hs = Object.keys(ad.results);
    const mk = keys => `<tr><th>horizon</th><th>static</th><th>const-vel</th><th>const-turn*</th></tr>` +
      keys.map(h => { const r = ad.results[h];
        return `<tr><td>${h}</td><td class="d">${r.static.ade_m.toFixed(2)}</td>
          <td class="a">${r.constant_velocity.ade_m.toFixed(2)}</td>
          <td class="d">${r.constant_turn_oracle.ade_m.toFixed(2)}</td></tr>`; }).join("");
    $("#adeTable").innerHTML = mk(hs.slice(0, 3));
    $("#vaAde").innerHTML = mk(hs);
  }
  const mo = R.motion_separation_report;
  if (mo) $("#vaMotion").innerHTML =
    `<tr><th>metric</th><th>value</th><th>metric</th><th>value</th></tr>
     <tr><td>precision</td><td>${fx(mo.best.precision)}</td><td>frames</td><td>${mo.frames}</td></tr>
     <tr><td>recall</td><td>${fx(mo.best.recall)}</td><td>boxes scored</td><td>${mo.boxes_scored}</td></tr>
     <tr><td>F1</td><td class="g">${fx(mo.best.f1)}</td><td>coverage</td><td>${pc(mo.coverage)}</td></tr>`;

  const iv = R.integrity_visibility_report;
  if (iv) {
    const a = iv.auroc;
    $("#inAuroc").innerHTML =
      bar("occlusion-aware", a.integrity_occlusion_aware.auroc, 0.75, "var(--acc)") +
      bar("no occlusion", a.integrity_no_occlusion.auroc, 0.75, "#4A5568") +
      bar("object range alone", a.range_only_baseline.auroc, 0.75, "var(--warn)");
    $("#inVerdict").innerHTML = `The occlusion ray-cast adds real signal
      (<b>${iv.occlusion_term_delta_auroc > 0 ? "+" : ""}${iv.occlusion_term_delta_auroc}</b> AUROC,
      CI [${iv.occlusion_term_delta_ci95.join(", ")}]). The map as a whole does <b>not</b> beat
      object range alone (${iv.beats_range_baseline_by}). It measures ground-plane observability,
      which is not object visibility &mdash; use it as an observability prior, not a detector.
      ${iv.boxes_scored} objects.`;
    const lv = iv.per_visibility_level;
    $("#inLevels").innerHTML = Object.keys(lv).map(k =>
      bar(k + "  (n=" + lv[k].n + ")", lv[k].mean_integrity, 0.5, "var(--acc)")).join("");
  }
  const tm = R.timing_report;
  if (tm) {
    const ms = tm.per_stage_ms, mx = Math.max(...Object.values(ms));
    $("#perfBars").innerHTML = Object.keys(ms).map(k =>
      `<div class="br"><span class="t">${k.replace(/_/g, " ")}</span>
       <span class="track"><span class="fill" style="width:${Math.max(2, 100 * ms[k] / mx)}%;
       background:${ms[k] > 300 ? "var(--warn)" : "var(--acc)"}"></span></span>
       <span class="n">${ms[k].toFixed(1)} ms</span></div>`).join("");
    $("#perfNote").textContent = `${tm.total_ms_per_keyframe} ms total · ${tm.hz} Hz`;
    const bf = tm.per_stage_ms_before_optimisation;
    $("#perfTable").innerHTML = `<tr><th>stage</th><th>before</th><th>after</th><th>speedup</th></tr>` +
      [["occupancy_logodds_0.20m", tm.occupancy_speedup_x], ["ground_plane_ransac", null],
       ["integrity_6cam_0.50m", null]].map(([k, sp]) =>
      `<tr><td>${k.replace(/_/g, " ")}</td><td class="d">${bf[k].toFixed(0)} ms</td>
       <td>${ms[k].toFixed(0)} ms</td><td class="${sp ? "g" : "d"}">${sp ? sp.toFixed(2) + "×" : "—"}</td></tr>`).join("") +
      `<tr><td><b>full keyframe</b></td><td class="d">${tm.total_ms_before.toFixed(0)} ms</td>
       <td>${tm.total_ms_per_keyframe.toFixed(0)} ms</td>
       <td class="g">${tm.end_to_end_speedup_x.toFixed(2)}×</td></tr>`;
    $("#perfCond").innerHTML = [tm.hardware, tm.implementation, tm.grids, tm.measured,
      tm.optimisation, tm.caveat].map(t => "<div style='margin-bottom:7px'>" + t + "</div>").join("");
  }

  /* ---------- models page ---------- */
  const lm = R.learned_model_report;
  if (lm) {
    $("#mdNote").textContent = `${lm.samples} keyframes · ${lm.checkpoint} · ${lm.inference_ms_per_sample} ms/sample on ${lm.hardware.split(",")[0]}`;
    const hs = Object.keys(lm.trajectory_ade);
    $("#mdAde").innerHTML = `<tr><th>horizon</th><th>static</th><th>const-vel</th><th>learned</th><th>vs CV</th></tr>` +
      hs.map(h => { const r = lm.trajectory_ade[h];
        return `<tr><td>${h}</td><td class="d">${r.static.ade_m.toFixed(2)}</td>
          <td>${r.constant_velocity.ade_m.toFixed(3)}</td>
          <td class="a">${r.learned_v11_temporal.ade_m.toFixed(3)}</td>
          <td class="g">${r.delta_vs_cv_pct.toFixed(1)}%</td></tr>`; }).join("");
    const mx = Math.max(...hs.map(h => lm.trajectory_ade[h].constant_velocity.ade_m));
    $("#mdBars").innerHTML = hs.slice(0, 3).map(h => { const r = lm.trajectory_ade[h];
      return bar(h + " const-vel", r.constant_velocity.ade_m, mx, "var(--warn)") +
             bar(h + " learned", r.learned_v11_temporal.ade_m, mx, "var(--acc)"); }).join("");
    const o = lm.occupancy.best;
    $("#mdOcc").innerHTML = `<tr><th>metric</th><th>value</th></tr>` +
      [["IoU", o.iou], ["precision", o.precision], ["recall", o.recall], ["F1", o.f1],
       ["threshold", o.threshold]].map(([k, v]) =>
      `<tr><td>${k}</td><td class="${k === "IoU" ? "a" : ""}">${v.toFixed ? v.toFixed(4) : v}</td></tr>`).join("");
    $("#mdLoad").textContent = lm.weights_loaded;
  }
  const vla = R.vla_report;
  if (vla) {
    $("#vlaArch").textContent = vla.architecture;
    const hs = Object.keys(vla.val_ade_m);
    $("#vlaTable").innerHTML = `<tr><th>horizon</th><th>const-vel</th><th>VLA</th></tr>` +
      hs.map(h => { const r = vla.val_ade_m[h];
        const better = r.vla_gpt2_projector < r.constant_velocity;
        return `<tr><td>${h}</td><td>${r.constant_velocity.toFixed(3)}</td>
          <td class="${better ? "g" : "b"}">${r.vla_gpt2_projector.toFixed(3)}</td></tr>`; }).join("");
    $("#vlaNotes").innerHTML =
      `<div class="warnbox"><b>${vla.trainable_params_m}M trainable</b> projector,
       ${vla.frozen_params_m}M frozen · ${vla.train_samples} train / ${vla.val_samples} val ·
       ${vla.steps} steps. Runs end to end and beats the prior only at 6 s. With 64 training
       samples and a frozen LM this repo previously damaged by an all-zero fine-tune, that is
       what it should do &mdash; this demonstrates the mechanism, not a model result.</div>`;
  }
  const vlm = R.vlm_report;
  if (vlm) {
    $("#vlmBox").innerHTML =
      `<div style="display:flex;gap:8px;align-items:center;margin-bottom:10px">
         <span class="pill" style="border-color:var(--bad);color:var(--bad)">${vlm.status}</span>
         <span class="mono" style="font-size:11.5px">${vlm.model}</span></div>
       <div class="badbox">${vlm.reason}</div>
       <div class="warnbox" style="margin-top:8px">${vlm.not_the_reason}</div>
       <div style="margin-top:10px;font-size:11.5px;color:var(--dim)">
         <b>Unblocks with:</b> ${vlm.unblocks_with}</div>
       <div style="margin-top:8px;font-size:11.5px;color:var(--dim2)">${vlm.would_not_substitute}</div>`;
  }

  /* hard-case mining across every loaded frame */
  const hard = [];
  F.forEach((f, i) => f.objects.forEach(o => hard.push({ ...o, fi: i, scene: f.scene })));
  hard.sort((a, b) => a.integrity - b.integrity);
  const hl = $("#hardList"); hl.innerHTML =
    `<div class="objrow" style="color:var(--dim2);cursor:default">
      <span>OBJECT</span><span>INTEG</span><span>VIS</span><span>RANGE</span></div>`;
  hard.slice(0, 24).forEach(o => {
    const r = el("div", "objrow");
    const visLbl = { 1: "0-40", 2: "40-60", 3: "60-80", 4: "80-100" }[o.vis] || "--";
    r.innerHTML = `<span>${o.scene.replace("scene-", "")} · ${o.cat}</span>
      <span class="${o.integrity < 0.2 ? "b" : "w"}">${fx(o.integrity, 2)}</span>
      <span class="d">${visLbl}</span><span>${o.range}m</span>`;
    r.onclick = () => { setFrame(o.fi); sel = o.id; draw();
      document.querySelectorAll(".rb").forEach((x, i) => x.classList.toggle("on", i === 0));
      document.querySelectorAll(".page").forEach(x => x.classList.toggle("on", x.dataset.p === "overview")); };
    hl.appendChild(r);
  });
  const lo = hard.filter(o => o.integrity < 0.2).length;
  const occl = hard.filter(o => o.vis <= 2).length;
  $("#failStats").innerHTML = [
    ["objects indexed", hard.length], ["integrity below 0.20", lo],
    ["annotator visibility ≤ 60%", occl]
  ].map(([k, v]) => `<div class="card pad"><div class="k lab">${k}</div>
    <div class="mono" style="font-size:24px;font-weight:600;margin-top:4px">${v}</div></div>`).join("");

  /* system inventory */
  const SYS = [
    ["LiDAR pose chain + multi-sweep", "runs", "sensor→ego→global→ego_ref, verified to 1.3e-13 m"],
    ["Ground plane + ray-cast occupancy", "runs", "log-odds inverse sensor model, 540² @ 0.20 m"],
    ["Static/dynamic separation", "runs", "free-space consistency, F1 0.61"],
    ["Camera projection + integrity map", "runs", "6-camera noisy-OR with occlusion ray-cast"],
    ["Occupancy forecast", "runs", "persistence + constant-velocity, geometric"],
    ["Ego trajectory baselines", "runs", "static / const-velocity / const-turn oracle"],
    ["v11_temporal checkpoint (occ + traj)", "runs", "151/169 tensors loaded; ADE beats const-velocity at every horizon"],
    ["Trust head weights", "blocked", "checkpoint stores trust_scorer.cnn.*, model defines trust_scorer.trunk.* — name drift, 18 tensors"],
    ["GPT-2 trajectory LM (as trained)", "blocked", "fine-tuned on all-zero waypoints — needs retraining from the label npz"],
    ["VLA projector (LLaVA pattern)", "runs", "1.77M trainable on frozen backbone + frozen GPT-2; mechanism verified, 80 samples"],
    ["BLIP vision-language captioning", "blocked", "huggingface.co 403 at the egress proxy — policy denial, not a missing dependency"],
    ["Sparse causal trajectory head", "not run", "torch present now; no evaluation written yet"],
    ["Trust-weighted BEV pooling kernel", "not run", "claims 4.5x on MPS; unverified on this CPU"],
    ["C++ runner (SPSC ring, latency stats)", "partial", "ring + latency stats present; integrity monitor not ported"],
  ];
  $("#sysTable").innerHTML = `<tr><th>component</th><th>state</th><th>detail</th></tr>` +
    SYS.map(([a, b, c]) => `<tr><td>${a}</td>
      <td class="${b === "runs" ? "g" : b === "blocked" ? "b" : "w"}">${b}</td>
      <td class="d" style="font-family:'IBM Plex Sans'">${c}</td></tr>`).join("");
}

fillStatic();
draw();
addEventListener("resize", draw);
</script>
"""

def main():
    bundle = json.loads(BUNDLE.read_text())
    html = HEAD + BODY + "<script>window.__ODFM__=" + json.dumps(bundle) + ";</script>" + SCRIPT
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(ROOT)}  {OUT.stat().st_size/1e6:.1f} MB")

if __name__ == "__main__":
    main()
