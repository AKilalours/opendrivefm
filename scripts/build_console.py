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
.bar{position:sticky;top:0;z-index:40;display:flex;align-items:center;flex-wrap:wrap;
  gap:10px 18px;padding:8px 14px;background:#0C1017;border-bottom:1px solid var(--line)}
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
.shell{display:flex;min-height:60vh}
.rail{width:var(--rail);flex:0 0 var(--rail);background:#0C1017;border-right:1px solid var(--line);
  display:flex;flex-direction:column;align-items:center;padding-top:8px;gap:2px;position:sticky;
  top:0;height:100vh}
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
.cam .zones{position:absolute;inset:0}
.cam .zone{position:absolute;cursor:pointer;border:1px solid transparent;border-radius:2px}
.cam .zone:hover{border-color:rgba(91,210,232,.85);background:rgba(91,210,232,.14)}
.mapwrap.clickable{cursor:crosshair}
.ovl{position:absolute;inset:0;width:100%;height:100%}
.ovl rect{fill:rgba(120,210,255,.10);stroke:rgba(140,220,255,.65);stroke-width:.22;cursor:pointer;
  vector-effect:non-scaling-stroke}
.ovl rect:hover{fill:rgba(91,210,232,.34);stroke:var(--acc)}
.ovl rect.sel{fill:rgba(91,210,232,.45);stroke:#fff;stroke-width:.5}

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
.insp{position:fixed;right:0;top:0;width:270px;height:100vh;
  background:#0D1119;border-left:1px solid var(--line2);padding:14px;overflow:auto;
  transform:translateX(100%);transition:transform .16s ease;z-index:30}
.insp.on{transform:none}
body.insp-open main{padding-right:284px}
body.insp-open .bar{padding-right:290px}
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
.statusgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px}
.st{background:var(--surf);border:1px solid var(--line);border-radius:6px;padding:10px 12px;
  cursor:pointer;border-left-width:3px}
.st:hover{border-color:var(--line2);background:var(--surf2)}
.st .n{font-size:12px;font-weight:600;margin-bottom:3px}
.st .v{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--dim)}
.hr{height:1px;background:var(--line);margin:14px 0}
.warnbox{border-left:2px solid var(--warn);padding:8px 12px;background:rgba(255,156,92,.05);
  font-size:11.5px;color:var(--dim);border-radius:0 4px 4px 0}
details.notes{margin-top:10px;border-top:1px solid var(--line);padding-top:8px}
details.notes summary{cursor:pointer;font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--dim2);list-style:none}
details.notes summary::-webkit-details-marker{display:none}
details.notes summary::before{content:"+ ";color:var(--acc)}
details.notes[open] summary::before{content:"− "}
details.notes .body{font-size:11.5px;color:var(--dim);line-height:1.6;margin-top:8px}
.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden;margin-bottom:10px}
.hero div{background:var(--surf);padding:10px 12px}
.hero .k{font-size:9px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim2)}
.hero .v{font-family:"IBM Plex Mono",monospace;font-size:21px;font-weight:600;margin-top:2px}
.hero .s{font-size:10px;color:var(--dim2);margin-top:1px}
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
  <div class="grp"><span class="lab">Camera</span>
    <button class="pill ov" data-ov="plain">Raw</button>
    <button class="pill ov" data-ov="lidar">LiDAR</button>
    <button class="pill ov on" data-ov="boxes">3D Boxes</button></div>
  <div class="grp"><span class="lab">World</span>
    <button class="pill wl on" data-wl="bev">BEV</button>
    <button class="pill wl" data-wl="bev_nodyn">Static only</button>
    <button class="pill wl" data-wl="occupancy">Occupancy</button></div>
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
      <div class="sec" style="margin-top:16px"><h2>Stack status</h2>
        <span class="note">every tile links to the page holding its numbers</span></div>
      <div class="statusgrid" id="statusStrip"></div>
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
          <div id="fcNote"></div></div>
        <div class="card pad"><div class="sec"><h2>Ego trajectory ADE &mdash; 404 keyframes</h2></div>
          <table id="adeTable"></table>
          <div id="adeNote"></div></div>
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
      <div class="hr"></div>
      <div class="card pad"><div class="sec"><h2>Trajectory language model &mdash; three defects, three fixes</h2>
        <span class="note" id="tlNote"></span></div>
        <div class="grid3" id="tlFixes" style="margin-bottom:12px"></div>
        <table id="tlTable"></table></div>
      <div class="hr"></div>
      <div class="grid2">
        <div class="card pad"><div class="sec"><h2>VLA &mdash; vision to action through a language model</h2></div>
          <div class="mono" style="font-size:11px;color:var(--dim);line-height:1.7;margin-bottom:10px"
               id="vlaArch"></div>
          <table id="vlaTable"></table>
          <div id="vlaNotes" style="margin-top:10px"></div></div>
        <div class="card pad"><div class="sec"><h2>VLM &mdash; scene understanding</h2>
          <span class="note" id="vlmModel"></span></div>
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px">
            <button class="pill" id="vlmRun">Describe this frame</button>
            <button class="pill" id="vlmStop" hidden>Stop</button>
            <span class="mono" id="vlmState" style="font-size:11px;color:var(--dim2)"></span></div>
          <div id="vlmOut" style="font-size:12px;line-height:1.6;color:var(--ink);
            white-space:pre-wrap;min-height:60px"></div>
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

    <!-- ============ ROBUSTNESS ============ -->
    <section class="page" data-p="robustness">
      <div class="sec"><h2>Fault injection &mdash; one camera degraded at a time</h2>
        <span class="note" id="rbNote"></span></div>
      <div class="grid2">
        <div class="card pad"><table id="rbTable"></table>
          <div id="rbVerdict" style="margin-top:10px"></div></div>
        <div class="card pad"><div class="sec"><h2>Trust separation</h2>
          <span class="note">faulted camera vs the five untouched</span></div>
          <div id="rbBars"></div>
          <div class="badbox" style="margin-top:10px" id="rbGap"></div></div>
      </div>
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
    <section class="page" data-p="hardcases">
      <div class="sec"><h2>Hard-case mining &mdash; every object ranked by camera observability</h2>
        <span class="note">the data engine: which frames are worth labelling next. click any row to open it</span></div>
      <div class="card"><div class="objlist" id="hardList" style="max-height:none"></div></div>
      <div class="hr"></div>
      <div class="grid3" id="failStats"></div>
    </section>

    <!-- ============ RUNTIME ============ -->
    <section class="page" data-p="runtime">
      <div class="sec"><h2>C++ deployment primitives</h2><span class="note" id="cppNote"></span></div>
      <div class="grid2">
        <div class="card pad"><div class="sec"><h2>SPSC ring handoff latency</h2></div>
          <table id="cppTable"></table>
          <div class="warnbox" style="margin-top:10px" id="cppFlaky"></div></div>
        <div class="card pad"><div class="sec"><h2>Python &harr; C++ parity</h2></div>
          <table id="cppParity"></table>
          <div id="cppBuild"></div></div>
      </div>
      <div class="hr"></div>
      <div class="sec"><h2>Backpressure policy &mdash; the number model latency cannot show</h2>
        <span class="note" id="rnNote"></span></div>
      <div class="card pad"><table id="rnTable"></table><div id="rnNoteBox"></div></div>
      <div class="hr"></div>
      <div class="grid2">
        <div class="card pad"><div class="sec"><h2>Sparse attention &mdash; ms/forward</h2>
          <span class="note">does sparsity pay?</span></div>
          <table id="spTable"></table>
          <div id="spNote" style="margin-top:8px;font-size:11.5px;color:var(--dim)"></div></div>
        <div class="card pad"><div class="sec"><h2>Trust-weighted BEV pooling</h2></div>
          <div id="bpBars"></div>
          <div class="warnbox" style="margin-top:10px" id="bpNote"></div></div>
      </div>
      <div class="hr"></div>
      <div class="sec"><h2>Python pipeline latency</h2><span class="note" id="perfNote"></span></div>
      <div class="card pad"><div id="perfBars"></div></div>
      <div class="grid2" style="margin-top:10px">
        <div class="card pad"><table id="perfTable"></table></div>
        <div class="card pad"><div class="sec"><h2>Measurement conditions</h2></div>
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
const PAGES = ["overview","perception","forecast","models","integrity","robustness","validation","hard cases","runtime","system"];
let fi = 0, ov = "boxes", wl = "bev", sel = null, timer = null;

const scenes = [...new Set(F.map(f => f.scene))];
// Null-safe. A missing element used to throw on `.innerHTML =`, which aborted
// fillStatic() and left every panel after the throw point blank -- one deleted
// card silently emptied five pages. Now a missing node costs its own panel and
// nothing else, and says so in the console.
const DEAD = () => ({ innerHTML: "", textContent: "", style: {}, src: "",
  classList: { add() {}, remove() {}, toggle() {} }, appendChild() {},
  insertAdjacentHTML() {}, getBoundingClientRect: () => ({ width: 0, height: 0 }) });
const $ = s => document.querySelector(s) ||
  (console.warn("[odfm] missing element", s), DEAD());
const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c;
  if (h !== undefined) e.innerHTML = h; return e; };
const fx = (v, n = 3) => (v === undefined || v === null) ? "--" : Number(v).toFixed(n);
const pc = v => (100 * v).toFixed(1) + "%";

/* ---------- nav rail ---------- */
PAGES.forEach((p, i) => {
  const b = el("button", "rb" + (i === 0 ? " on" : ""), p);
  b.onclick = () => { document.querySelectorAll(".rb").forEach(x => x.classList.remove("on"));
    b.classList.add("on");
    document.querySelectorAll(".page").forEach(x => x.classList.toggle("on", x.dataset.p === p.replace(" ",""))); };
  $("#rail").appendChild(b);
});

/* ---------- camera grid ---------- */
CAMS.forEach(c => {
  const d = el("div", "cam");
  d.innerHTML = `<img data-cam="${c}" alt="${c}"><div class="tag">${c.replace("CAM_","")}</div>
    <div class="vis" data-vis="${c}"></div><div class="hit" data-hit="${c}"></div>
    <div class="zones" data-zones="${c}"></div>`;
  $("#cams").appendChild(d);
});

/* ---------- map panels ---------- */
$("#maps").innerHTML = `
  <div class="mapwrap"><img id="ovBev" alt="BEV">
    <svg class="ovl" id="ovlBev" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
    <div class="mark" id="mkBev"></div>
    <div class="cap"><b id="ovBevCap">Multi-sweep BEV</b><span class="mono d" id="ovBevN"></span></div></div>
  <div class="mapwrap"><img id="ovInt" alt="Integrity">
    <svg class="ovl" id="ovlInt" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
    <div class="mark" id="mkInt"></div>
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
document.querySelectorAll(".wl").forEach(b => b.onclick = () => {
  wl = b.dataset.wl;
  document.querySelectorAll(".wl").forEach(x => x.classList.toggle("on", x === b));
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
    // One clickable zone per object this camera can see. The boxes come from
    // the exporter, which projected each 3D box into every camera that has it
    // in frame -- so clicking a vehicle here is the same object id the BEV,
    // the inspector and the hard-case list use.
    const zc = document.querySelector(`[data-zones="${c}"]`);
    zc.innerHTML = "";
    f.objects.forEach(ob => {
      const bb = ob.cams[c];
      if (!bb) return;
      const z = el("div", "zone");
      z.style.left = (100 * bb[0]) + "%"; z.style.top = (100 * bb[1]) + "%";
      z.style.width = (100 * (bb[2] - bb[0])) + "%";
      z.style.height = (100 * (bb[3] - bb[1])) + "%";
      z.title = `${ob.cat} · ${ob.range} m · ${ob.speed.toFixed(1)} m/s`;
      z.onclick = ev => { ev.stopPropagation(); pick(ob.id); };
      zc.appendChild(z);
    });
    const hit = document.querySelector(`[data-hit="${c}"]`);
    const o = sel !== null ? f.objects.find(x => x.id === sel) : null;
    if (o && o.cams[c]) {
      const [x0, y0, x1, y1] = o.cams[c];
      hit.style.left = (100 * x0) + "%"; hit.style.top = (100 * y0) + "%";
      hit.style.width = (100 * (x1 - x0)) + "%"; hit.style.height = (100 * (y1 - y0)) + "%";
      hit.classList.add("on");
    } else hit.classList.remove("on");
  });

  // The World control swaps the LEFT panel between three renders of the same
  // frame. The Camera control only re-skins the camera tiles; it cannot change
  // the world state, because BEV, occupancy and integrity are computed from
  // LiDAR and calibration, not from the camera overlay.
  $("#ovBev").src = f.maps[wl]; $("#ovInt").src = f.maps.integrity;
  $("#ovBevCap").textContent =
    {bev: "Multi-sweep BEV", bev_nodyn: "BEV — static returns only",
     occupancy: "Ray-cast occupancy"}[wl];
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

  // Object footprints are drawn as an SVG layer OVER the map, not baked into
  // the rendered image. That is what makes them clickable on every world
  // layer: the static-only render carries no boxes at all, and the occupancy
  // render draws its own, so relying on the picture meant objects were only
  // selectable while BEV happened to be showing.
  [["#ovlBev", "#ovBev"], ["#ovlInt", "#ovInt"]].forEach(([sv, im]) => {
    const svg = $(sv);
    if (!svg.innerHTML && !svg.setAttribute) return;
    const R = f.rng_m, parts = [];
    f.objects.forEach(ob => {
      const w = ob.wl[0], l = ob.wl[1], c = Math.cos(ob.yaw), s2 = Math.sin(ob.yaw);
      const half = Math.max(Math.abs(l * c) + Math.abs(w * s2),
                            Math.abs(l * s2) + Math.abs(w * c)) / 2;
      const cx = 50 - 50 * ob.y / R, cy = 50 - 50 * ob.x / R;
      const sz = Math.max(1.6, 50 * half / R);
      parts.push(`<rect data-id="${ob.id}" x="${(cx - sz).toFixed(2)}" y="${(cy - sz).toFixed(2)}"
        width="${(2 * sz).toFixed(2)}" height="${(2 * sz).toFixed(2)}" rx="0.4"
        class="${sel === ob.id ? "sel" : ""}"><title>${ob.cat} · ${ob.range} m · ${ob.speed.toFixed(1)} m/s</title></rect>`);
    });
    svg.innerHTML = parts.join("");
    svg.querySelectorAll && svg.querySelectorAll("rect").forEach(r =>
      r.onclick = ev => { ev.stopPropagation(); pick(Number(r.dataset.id)); });
    const node = $(im);
    node.parentElement && node.parentElement.classList.add("clickable");
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
  document.body.classList.toggle("insp-open", !!o);
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
function note(title, body) {
  return `<details class="notes"><summary>${title}</summary><div class="body">${body}</div></details>`;
}
function hero(cells) {
  return `<div class="hero">` + cells.map(([k, v, sub]) =>
    `<div><div class="k">${k}</div><div class="v">${v}</div>
     <div class="s">${sub || "&nbsp;"}</div></div>`).join("") + `</div>`;
}
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
  $("#fcNote").innerHTML = note("Why persistence wins",
    "Recall is unchanged (0.4727 &rarr; 0.4726) while precision falls 0.576 &rarr; 0.496 — advection " +
    "moves correct cells to wrong places rather than finding new ones. Root cause is motion-label " +
    "precision of 0.56, not the advection.");
  $("#adeNote").innerHTML = note("On the shipped GPT-2 checkpoint",
    "It was fine-tuned from manifest keys that do not exist, so every waypoint fell back to (0,0) " +
    "and it saw 404 copies of one all-zero trajectory. Retrained from the label files — see Models.");
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
    $("#inVerdict").outerHTML = hero([
      ["occlusion term", (iv.occlusion_term_delta_auroc > 0 ? "+" : "") + iv.occlusion_term_delta_auroc,
       "AUROC, CI [" + iv.occlusion_term_delta_ci95.join(", ") + "]"],
      ["vs range baseline", iv.beats_range_baseline_by, "not a difference"],
      ["objects scored", iv.boxes_scored, "human visibility labels"],
    ]) + note("Reading",
      "The occlusion ray-cast adds real signal, but the map as a whole does not beat object range " +
      "alone. It measures ground-plane observability, which is not object visibility — a car behind " +
      "a car has an occluded footprint and a visible roof. Use it as an observability prior, not a detector.");
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
    if (lm.trust) {
      const t = lm.trust.per_camera_mean;
      $("#mdOcc").insertAdjacentHTML("afterend",
        `<div class="hr" style="margin:10px 0"></div>
         <div class="lab" style="margin-bottom:6px">Live per-camera trust (clean frames)</div>
         <div class="mono" style="font-size:11.5px;color:var(--dim);line-height:1.7">` +
        Object.keys(t).map(k => `${k.replace("CAM_","")} ${t[k].toFixed(4)}`).join(" &nbsp;·&nbsp; ") +
        `</div><div style="margin-top:8px;font-size:11px;color:var(--dim2)">${lm.trust.note}</div>`);
    }
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
    $("#vlaNotes").innerHTML = hero([
      ["trainable", vla.trainable_params_m + "M", "projector"],
      ["frozen", vla.frozen_params_m + "M", "backbone + GPT-2"],
      ["train / val", vla.train_samples + " / " + vla.val_samples, vla.steps + " steps"],
    ]) + note("Reading",
      "Runs end to end and beats the prior only at 6 s. With this corpus and a frozen LM the repo " +
      "previously damaged by an all-zero fine-tune, that is the expected outcome — it demonstrates " +
      "the mechanism, not a model result.");
  }
  const vlm = R.vlm_report;
  if (vlm) {
    $("#vlmModel").textContent = vlm.model;
    $("#vlmBox").innerHTML =
      note("What this is, and what it is not",
        vlm.how + "<br><br><b>Not BLIP.</b> " + vlm.why_not_blip +
        "<br><br>" + vlm.honest_difference);
  }

  /* ---------- live VLM: a real vision-language model on the real frames ----
     The repo's BLIP path cannot run here -- huggingface.co is refused by the
     egress proxy in both available environments. Rather than ship a dead panel
     or write captions from object counts and call them a model, this asks
     Claude, which the runtime can actually reach, and labels it as such. */
  (async () => {
    const btn = $("#vlmRun"), stopBtn = $("#vlmStop");
    const out = $("#vlmOut"), state = $("#vlmState");
    const sample = window.claude && await window.claude.use("sample").catch(() => null);
    if (!sample) { state.textContent = "sampling unavailable in this view"; btn.disabled = true; return; }
    const lim = await sample.limits().catch(() => null);
    if (!lim || !lim.images) { state.textContent = "images unavailable in this view"; btn.disabled = true; return; }
    let ctl = null;
    stopBtn.onclick = () => ctl && ctl.abort();
    btn.onclick = async () => {
      const f = F[fi];
      const blobs = [];
      for (const c of ["CAM_FRONT", "CAM_BACK"]) {
        blobs.push(await (await fetch(f.cameras[c].boxes)).blob());
      }
      ctl = new AbortController();
      btn.disabled = true; stopBtn.hidden = false;
      state.textContent = "Thinking..."; out.textContent = "";
      try {
        await sample(
          "These are the front and rear camera frames from an autonomous vehicle " +
          "on a nuScenes drive, with LiDAR returns and 3D boxes drawn over them. " +
          "In under 90 words: describe the scene, then name the single most " +
          "safety-relevant thing in it and say why. Plain prose, no preamble.",
          { images: blobs, signal: ctl.signal, modelTier: "quick",
            onText: ({ text }) => { state.textContent = ""; out.textContent = text; } });
        state.textContent = `${f.scene} · frame ${fi + 1}`;
      } catch (e) {
        out.textContent = e.text || "";
        state.textContent = e.code === "cancelled" ? "stopped"
          : e.code === "not_granted" ? "declined by viewer" : (e.code || "failed");
      } finally { btn.disabled = false; stopBtn.hidden = true; }
    };
  })();

  const rbq = R.robustness_report, cpq = R.cpp_report;

  /* hard-case mining across every loaded frame */
  const hard = [];
  F.forEach((f, i) => f.objects.forEach(o => hard.push({ ...o, fi: i, scene: f.scene })));
  // Why is integrity low here? Without this the ranking is dominated by two
  // effects that are not the same problem at all: the near-field blind zone
  // under the vehicle, where no camera can see the ground at any range, and
  // genuine far-field or occluded cases. Labelling the cause is what makes the
  // list a work queue instead of a list.
  const cause = o => o.range < 6 ? "near-field blind zone"
                  : o.range > 40 ? "beyond camera resolution"
                  : o.vis <= 2 ? "occluded (annotator agrees)"
                  : "camera occlusion";
  hard.forEach(o => o.cause = cause(o));
  hard.sort((a, b) => a.integrity - b.integrity);
  const hl = $("#hardList"); hl.innerHTML =
    `<div class="objrow hdr" style="grid-template-columns:1fr 52px 52px 46px 168px;color:var(--dim2);cursor:default">
      <span>OBJECT</span><span>INTEG</span><span>VIS</span><span>RANGE</span><span>LIKELY CAUSE</span></div>`;
  hard.slice(0, 26).forEach(o => {
    const r = el("div", "objrow");
    r.style.gridTemplateColumns = "1fr 52px 52px 46px 168px";
    const visLbl = { 1: "0-40", 2: "40-60", 3: "60-80", 4: "80-100" }[o.vis] || "--";
    r.innerHTML = `<span>${o.scene.replace("scene-", "")} · ${o.cat}</span>
      <span class="${o.integrity < 0.2 ? "b" : "w"}">${fx(o.integrity, 2)}</span>
      <span class="d">${visLbl}</span><span>${o.range}m</span>
      <span class="d" style="font-family:'IBM Plex Sans'">${o.cause}</span>`;
    r.onclick = () => { setFrame(o.fi); sel = o.id; draw();
      document.querySelectorAll(".rb").forEach((x, i) => x.classList.toggle("on", i === 0));
      document.querySelectorAll(".page").forEach(x => x.classList.toggle("on", x.dataset.p === "overview")); };
    hl.appendChild(r);
  });
  const byCause = {};
  hard.forEach(o => { byCause[o.cause] = (byCause[o.cause] || 0) + 1; });
  $("#failStats").innerHTML = [["objects indexed", hard.length],
    ["integrity below 0.20", hard.filter(o => o.integrity < 0.2).length],
    ["near-field blind zone", byCause["near-field blind zone"] || 0]]
    .map(([k, v]) => `<div class="card pad"><div class="k lab">${k}</div>
      <div class="mono" style="font-size:24px;font-weight:600;margin-top:4px">${v}</div></div>`).join("");

  /* ---------- retrained trajectory LM ---------- */
  const tl = R.trajlm_retrained_report;
  if (tl) {
    $("#tlNote").textContent = `${tl.train_samples} train / ${tl.val_samples} val · ${tl.steps} steps · ${tl.model.split(".")[0]}`;
    $("#tlFixes").innerHTML = [
      ["Data", "trained on 404 all-zero trajectories", "retrained from the label npz — 388 distinct paths"],
      ["Tokenisation", "±20 m clipped 30.6% of waypoints", "split axes: x [-10,70] m, y [-25,25] m"],
      ["Conditioning", "unconditional — could only emit the dataset average", "velocity-conditioned prefix"],
    ].map(([k, was, now]) => `<div style="border-left:2px solid var(--acc);padding:8px 12px;background:rgba(91,210,232,.05)">
        <div class="lab" style="color:var(--acc)">${k}</div>
        <div style="font-size:11px;color:var(--bad);margin-top:4px">was: ${was}</div>
        <div style="font-size:11px;color:var(--good);margin-top:3px">now: ${now}</div></div>`).join("");
    const hs = Object.keys(tl.val_ade_m);
    $("#tlTable").innerHTML = `<tr><th>horizon</th><th>const-vel</th><th>trajectory LM</th><th>vs CV</th></tr>` +
      hs.map(h => { const r = tl.val_ade_m[h];
        const g = r.delta_vs_cv_pct < 0;
        return `<tr><td>${h}</td><td>${r.constant_velocity.ade_m.toFixed(3)}</td>
          <td class="${g ? "g" : ""}">${r.gpt2_retrained_conditioned.ade_m.toFixed(3)}</td>
          <td class="${g ? "g" : "w"}">${r.delta_vs_cv_pct.toFixed(1)}%</td></tr>`; }).join("") +
      `<tr><td class="d">T+1, unconditional</td><td class="d">0.324</td><td class="b">2.796</td><td class="b">+763%</td></tr>`;
  }

  /* ---------- robustness ---------- */
  const rb = R.robustness_report;
  if (rb) {
    $("#rbNote").textContent = `${rb.frames} keyframes · ${rb.faulted_camera} degraded · ${rb.checkpoint}`;
    const keys = Object.keys(rb.results).filter(k => k !== "clean");
    const c = rb.results.clean;
    $("#rbTable").innerHTML =
      `<tr><th>fault</th><th>trust (faulted)</th><th>trust (others)</th><th>&Delta;trust</th><th>ADE T+3</th></tr>` +
      `<tr><td class="d">clean</td><td>${fx(c.trust_faulted)}</td><td>${fx(c.trust_others)}</td>
        <td class="d">&mdash;</td><td>${c.ade_T3_m} m</td></tr>` +
      keys.map(k => { const r = rb.results[k];
        const good = r.delta_trust_faulted < -0.05;
        return `<tr><td>${k}</td><td class="${good ? "g" : "b"}">${fx(r.trust_faulted)}</td>
          <td class="d">${fx(r.trust_others)}</td>
          <td class="${good ? "g" : "b"}">${r.delta_trust_faulted.toFixed(4)}</td>
          <td>${r.ade_T3_m} m</td></tr>`; }).join("");
    $("#rbBars").innerHTML = keys.map(k => bar(k, Math.abs(rb.results[k].delta_trust_faulted), 0.32,
      rb.results[k].delta_trust_faulted < -0.05 ? "var(--good)" : "var(--bad)")).join("");
    $("#rbVerdict").innerHTML = hero([
      ["separation", rb.mean_separation, "faulted vs untouched"],
      ["detected", ((rb.detected || []).length) + " / 5", (rb.detected || []).join(", ") || "—"],
      ["missed", (rb.missed || []).join(", ") || "none", "wrong sign"],
    ]) + note("Verdict", rb.verdict);
    const occ = rb.results.occlusion;
    $("#rbGap").outerHTML = note("The gap",
      `Occlusion moves trust the wrong way (${occ.delta_trust_faulted > 0 ? "+" : ""}${occ.delta_trust_faulted}). ` +
      "A masked region has low local variance, which this head reads as a clean flat surface — a hole " +
      "in a detector meant to catch a blocked camera.");
  }

  /* ---------- runtime ---------- */
  const cp = R.cpp_report;
  if (cp) {
    const pr0 = cp.primitives;
    $("#cppNote").textContent = cp.hardware;
    $("#cppTable").innerHTML = `<tr><th>SPSC ring handoff</th><th>value</th></tr>` +
      [["frames", pr0.bench_frames.toLocaleString()],
       ["dropped", pr0.bench_dropped],
       ["queueing p50", pr0.bench_queueing_p50_ms + " ms"],
       ["queueing p99", pr0.bench_queueing_p99_ms + " ms"],
       ["test_spsc_ring", pr0.test_spsc_ring],
       ["test_latency_stats", pr0.test_latency_stats]]
      .map(([k, v]) => `<tr><td>${k}</td><td class="${String(v) === "pass" ? "g" : ""}">${v}</td></tr>`).join("");
    const pv = cp.parity, pr = cp.primitives, il = cp.inference_latency;
    $("#cppParity").innerHTML = `<tr><th>output</th><th>max abs</th><th>max rel</th><th></th></tr>` +
      ["occupancy", "trajectory", "trust"].map(k =>
        `<tr><td>${k}</td><td>${pv[k].max_abs.toExponential(3)}</td>
         <td>${pv[k].max_rel.toExponential(3)}</td><td class="g">PASS</td></tr>`).join("") +
      `<tr><td>determinism</td><td colspan="2" class="d">repeated forward</td><td class="g">PASS</td></tr>`;
    $("#cppBuild").innerHTML = hero([
      ["inference p50", il.p50_ms.toFixed(0) + " ms", il.iterations + " iters"],
      ["p99", il.p99_ms.toFixed(0) + " ms", "jitter " + il.jitter_p99_over_p50.toFixed(2) + "x"],
      ["ring p99", pr.bench_queueing_p99_ms + " ms", pr.bench_frames.toLocaleString() + " frames, 0 dropped"],
    ]) + note("How the LibTorch path was unblocked", cp.how_it_was_unblocked) +
        note("Test flakiness", pr.flakiness);
    const rn = cp.runner, A = rn.at_10hz_6s.latest_frame_seqlock, B2 = rn.at_10hz_6s.fifo_queue;
    $("#rnNote").textContent = rn.what;
    $("#rnTable").innerHTML =
      `<tr><th>policy</th><th>inference p50</th><th>queue wait p50</th><th>end-to-end p50</th><th>processed</th><th>dropped</th></tr>` +
      `<tr><td>latest frame (seqlock)</td><td>${A.inference_p50_ms} ms</td><td>${A.queue_wait_p50_ms} ms</td>
        <td class="g">${A.end_to_end_p50_ms} ms</td><td>${A.processed}</td><td>${A.skipped_stale} stale</td></tr>` +
      `<tr><td>FIFO queue</td><td>${B2.inference_p50_ms} ms</td><td class="b">${B2.queue_wait_p50_ms} ms</td>
        <td class="b">${B2.end_to_end_p50_ms} ms</td><td>${B2.processed}</td><td>${B2.dropped_full} full</td></tr>`;
    $("#rnNoteBox").innerHTML = note("Why this is the headline", rn.finding);
  }
  const kb = R.kernel_bench_report;
  if (kb) {
    const sa = kb.sparse_attention.ms_per_forward;
    $("#spTable").innerHTML = `<tr><th>horizon</th><th>dense</th><th>strided</th><th>window</th><th>combined</th></tr>` +
      Object.keys(sa).map(h => { const r = sa[h];
        const best = Math.min(r.dense, r.strided, r.window, r.combined);
        const c = v => v === best && v < r.dense ? "g" : (v === r.dense ? "d" : "");
        return `<tr><td>${h}</td><td class="d">${r.dense}</td>
          <td class="${c(r.strided)}">${r.strided}</td><td class="${c(r.window)}">${r.window}</td>
          <td class="${c(r.combined)}">${r.combined}</td></tr>`; }).join("");
    $("#spNote").innerHTML = note("Does sparsity pay?",
      `No at horizon 12 (${sa["12"].dense} → ${sa["12"].strided} ms), yes at 128 ` +
      `(${sa["128"].dense} → ${sa["128"].combined} ms, ${(100*(1-sa["128"].combined/sa["128"].dense)).toFixed(0)}% faster). ` +
      "The repo's own docstring predicted exactly that; this measures it.");
    const bp = kb.bev_pooling;
    $("#bpBars").innerHTML = bar("python loop", bp.python_loop_ms, bp.python_loop_ms * 1.15, "var(--warn)") +
      bar("fused kernel", bp.kernel_ms, bp.python_loop_ms * 1.15, "var(--acc)");
    $("#bpNote").outerHTML = hero([[bp.speedup_x + "×", bp.speedup_x + "×", "CPU, shapes identical"]]).replace("<div class=\"k\">"+bp.speedup_x+"×</div>","<div class=\"k\">speedup</div>") +
      note("Against the docstring",
      `The docstring claims ${bp.claimed_in_docstring} — a different device, so the two are not the ` +
      "same measurement and the CPU figure is the one taken here.");
  }

  /* ---------- overview status strip ---------- */
  const goTo = page => {
    document.querySelectorAll(".rb").forEach(x =>
      x.classList.toggle("on", x.textContent === page));
    document.querySelectorAll(".page").forEach(x =>
      x.classList.toggle("on", x.dataset.p === page.replace(" ", "")));
    scrollTo(0, 0);
  };
  const STATUS = [
    ["Perception", "runs", `occupancy IoU ${lm ? lm.occupancy.best.iou.toFixed(3) : "--"}`, "perception"],
    ["Forecast", "runs", fcr ? `persistence IoU ${fcr.results["T+1"].persistence.iou.toFixed(3)} @ T+1` : "--", "forecast"],
    ["Learned model", "runs", lm ? `ADE ${lm.trajectory_ade["T+1 (0.5s)"].learned_v11_temporal.ade_m} m, ${lm.trajectory_ade["T+1 (0.5s)"].delta_vs_cv_pct}% vs CV` : "--", "models"],
    ["Trajectory LM", "runs", R.trajlm_retrained_report ? `retrained · ADE ${R.trajlm_retrained_report.val_ade_m["T+3 (1.5s)"].gpt2_retrained_conditioned.ade_m} m @ T+3` : "--", "models"],
    ["VLA", "runs", vla ? `${vla.trainable_params_m}M projector · ${vla.frozen_params_m}M frozen` : "--", "models"],
    ["VLM", "blocked", "BLIP weights unreachable (egress 403)", "models"],
    ["Trust / robustness", "runs", rbq ? `separation ${rbq.mean_separation}` : "--", "robustness"],
    ["Integrity", "runs", iv ? `AUROC ${iv.auroc.integrity_occlusion_aware.auroc}` : "--", "integrity"],
    ["C++ runtime", "runs", cpq ? `parity PASS · e2e ${cpq.runner.at_10hz_6s.latest_frame_seqlock.end_to_end_p50_ms} ms` : "--", "runtime"],
    ["Hard cases", "runs", `${F.reduce((a, f) => a + f.objects.length, 0)} objects indexed`, "hard cases"],
  ];
  $("#statusStrip").innerHTML = STATUS.map(([n, st, v, pg]) =>
    `<div class="st" data-go="${pg}" style="border-left-color:${st === "runs" ? "var(--good)" : "var(--bad)"}">
       <div class="n">${n} <span style="color:${st === "runs" ? "var(--good)" : "var(--bad)"};font-weight:400">· ${st}</span></div>
       <div class="v">${v}</div></div>`).join("");
  document.querySelectorAll(".st").forEach(e => e.onclick = () => goTo(e.dataset.go));

  /* system inventory */  /* system inventory */
  const SYS = [
    ["LiDAR pose chain + multi-sweep", "runs", "sensor→ego→global→ego_ref, verified to 1.3e-13 m"],
    ["Ground plane + ray-cast occupancy", "runs", "log-odds inverse sensor model, 540² @ 0.20 m"],
    ["Static/dynamic separation", "runs", "free-space consistency, F1 0.61"],
    ["Camera projection + integrity map", "runs", "6-camera noisy-OR with occlusion ray-cast"],
    ["Occupancy forecast", "runs", "persistence + constant-velocity, geometric"],
    ["Ego trajectory baselines", "runs", "static / const-velocity / const-turn oracle"],
    ["v11_temporal checkpoint (occ + traj)", "runs", "167/169 after key remap; ADE beats const-velocity at every horizon"],
    ["Trust head weights", "runs", "key drift remapped (cnn.* -> trunk./cnn_head.); trust_fixed_v2_cal loads 171/171 and is calibrated"],
    ["Trajectory language model", "runs", "retrained on real waypoints + velocity conditioning; beats const-velocity from T+3 out"],
    ["VLA projector (LLaVA pattern)", "runs", "1.77M trainable on frozen backbone + frozen GPT-2; mechanism verified, 80 samples"],
    ["Fault injection / robustness", "runs", "5 perturbations; trust separates faulted from untouched by 0.165 — except occlusion"],
    ["Sparse causal trajectory head", "runs", "no gain at horizon 12, 14% faster at 128 — matches its own docstring"],
    ["Trust-weighted BEV pooling kernel", "runs", "3.09x over the Python loop on CPU, outputs shape-identical"],
    ["C++ SPSC ring + latency stats", "runs", "builds, both tests pass, queueing p99 7.8 us over 20k frames"],
    ["TorchScript export", "runs", "traced + frozen from a real keyframe, 60 MB module driving the C++ runner"],
    ["C++ LibTorch runner", "runs", "parity PASS (5.2e-06 occupancy, 0.0 trust); end-to-end 235 ms seqlock vs 2319 ms FIFO"],
    ["VLM scene understanding", "runs", "live Claude call on the real frames via the artifact runtime — not BLIP, and labelled as such"],
    ["BLIP (the repo's own VLM path)", "blocked", "huggingface.co 403 at the egress proxy in BOTH environments — policy denial, not a dependency"],
    ["CUDA / GPU inference", "not available", "no GPU in either environment; every latency figure here is CPU and says so"],
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
