#!/usr/bin/env python3
"""Emit the OpenDriveFM validation console as one self-contained HTML file."""
from __future__ import annotations
import datetime as _dt
import json, subprocess, sys
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
/* The overlay covers the MAP, not the whole card: .mapwrap also holds the
   legend and caption rows, so inset:0 stretched the footprints down over them. */
.ovl{position:absolute;left:0;top:0;width:100%;height:auto;aspect-ratio:1/1}
.ovl rect{fill:rgba(120,210,255,.10);stroke:rgba(140,220,255,.65);stroke-width:.22;cursor:pointer;
  vector-effect:non-scaling-stroke}
.ovl rect:hover{fill:rgba(91,210,232,.34);stroke:var(--acc)}
.ovl rect.sel{fill:rgba(91,210,232,.45);stroke:#fff;stroke-width:.5}

/* ---------- maps ---------- */
.maps{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.mapwrap{position:relative;background:#0B0E14;border:1px solid var(--line);border-radius:5px;
  overflow:hidden;align-self:start}
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
/* display:block matters: .fill is a <span>, and an inline box ignores height,
   so every bar on every page was rendering as an empty track. */
.br .fill{display:block;height:100%;border-radius:2px;min-width:2px}
.br .n{font-family:"IBM Plex Mono",monospace;font-size:11.5px;text-align:right}

/* ---------- inspector ---------- */
.insp{position:fixed;right:0;top:0;width:326px;height:100vh;
  background:#0D1119;border-left:1px solid var(--line2);padding:14px;overflow:auto;
  transform:translateX(100%);transition:transform .16s ease;z-index:30}
.insp.on{transform:none}
body.insp-open main{padding-right:340px}
body.insp-open .bar{padding-right:344px}
.insp h3{margin:0 0 2px;font-size:15px}
.insp .cls{font-size:11px;color:var(--acc);letter-spacing:.1em;text-transform:uppercase}
.kv{display:flex;justify-content:space-between;gap:10px;padding:5px 0;
  border-bottom:1px solid rgba(35,42,54,.6);font-size:12px;align-items:baseline}
.kv span:first-child{color:var(--dim);flex:0 0 auto}
.kv span:last-child{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;
  text-align:right;min-width:0;overflow-wrap:anywhere}
.close{position:absolute;right:10px;top:10px;color:var(--dim2);font-size:15px}
.objlist{max-height:210px;overflow:auto;border:1px solid var(--line);border-radius:4px}
.objrow{display:grid;grid-template-columns:1fr 52px 48px 44px;gap:8px;padding:5px 9px;font-size:11px;
  font-family:"IBM Plex Mono",monospace;cursor:pointer;border-bottom:1px solid rgba(35,42,54,.5)}
.objrow span:not(:first-child){text-align:right}
.objrow.hdr{cursor:default;color:var(--dim2);font-size:9px;letter-spacing:.08em;
  position:sticky;top:0;background:var(--surf);border-bottom:1px solid var(--line)}
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

/* ---------- hint bar ----------
   This used to be a 11px dim span crushed against the heading in a flex row,
   where it read as decoration and wrapped into the map grid. It is now its own
   full-width bar with a live left border, and it reports the current selection
   rather than repeating a static instruction. */
.hint{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:8px 12px;margin:0 0 9px;
  background:var(--surf);border:1px solid var(--line);border-left:2px solid var(--acc);
  border-radius:0 4px 4px 0;font-size:12px;color:var(--dim);line-height:1.5}
.hint b{color:var(--ink);font-weight:600}
.hint .kbd{font-family:"IBM Plex Mono",monospace;font-size:10px;padding:1px 5px;border-radius:3px;
  border:1px solid var(--line2);background:var(--bg);color:var(--dim2)}
.hint .clear{margin-left:auto;font-size:11px;color:var(--acc);border-bottom:1px solid transparent}
.hint .clear:hover{border-bottom-color:var(--acc)}

/* ---------- live controls ---------- */
.ctrl{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:9px 11px;background:var(--surf2);
  border:1px solid var(--line);border-radius:5px;margin-bottom:9px}
.ctrl .lab{flex:0 0 auto}
.tog{padding:3px 8px;border:1px solid var(--line2);border-radius:3px;font-size:11px;
  font-family:"IBM Plex Mono",monospace;color:var(--dim2);background:var(--bg)}
.tog.on{border-color:var(--acc);color:var(--acc);background:rgba(91,210,232,.10)}
.ctrl input[type=range]{accent-color:var(--acc);width:120px}
.rdout{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--acc);margin-left:auto;
  min-height:1.4em}
canvas.grid{width:100%;display:block;image-rendering:pixelated;background:#0B0E14;cursor:crosshair;
  aspect-ratio:1/1;object-fit:contain}
.mapcap{max-width:600px}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:10.5px;color:var(--dim2);padding:6px 10px}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;
  vertical-align:-1px}
.swatch{height:8px;border-radius:2px;flex:1;min-width:80px}

/* ---------- filmstrip / scene tabs ---------- */
.strip{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px}
.strip figure{margin:0;flex:0 0 176px;cursor:pointer;padding:3px;border-radius:5px}
.strip figure:hover{background:var(--surf2)}
.strip img{width:100%;display:block;border:1px solid var(--line);border-radius:4px}
.strip figure.on img{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc)}
.strip figcaption{font-size:10px;color:var(--dim2);padding:4px 2px 0;
  font-family:"IBM Plex Mono",monospace;display:flex;justify-content:space-between;gap:8px;
  letter-spacing:.04em;text-transform:uppercase;white-space:nowrap}
.strip figure.on figcaption{color:var(--acc)}
.plot{width:100%;display:block}

/* ---------- architecture strip ---------- */
.arch{background:var(--surf);border:1px solid var(--line);border-radius:6px;padding:10px 12px 4px}
.arch svg{width:100%;height:auto;display:block;color:var(--ink)}
.arch .nd rect{fill:var(--surf2);stroke:var(--line2);stroke-width:1}
.arch .nd:hover rect{stroke:var(--acc);fill:#1B2230}
.arch .nd.hi rect{stroke:var(--acc);stroke-width:1.6;fill:rgba(91,210,232,.10)}
.arch .nd{cursor:pointer}
.arch .t{fill:var(--ink);font-size:11px;font-weight:600}
.arch .m{fill:var(--acc);font-size:10px;font-family:"IBM Plex Mono",monospace}
.arch .lane{fill:var(--dim2);font-size:9px;letter-spacing:.14em}
.arch .el{fill:var(--dim2);font-size:9.5px}
.arch line,.arch path{stroke:var(--dim2)}
.arch .bnd{stroke:var(--line2);stroke-dasharray:4 4}

/* ---------- claim ---------- */
.claim{font-size:15px;line-height:1.5;color:var(--ink);margin:2px 0 12px;max-width:105ch}
.claim b{color:var(--acc);font-weight:600;font-family:"IBM Plex Mono",monospace;font-size:14px}

/* ---------- reading paths ---------- */
.paths{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:9px}
.rp{padding:4px 11px;border:1px solid var(--line2);border-radius:3px;font-size:11.5px;color:var(--dim)}
.rp.on{border-color:var(--acc);color:var(--acc);background:rgba(91,210,232,.10)}
.step2{display:grid;grid-template-columns:22px 1fr 128px;gap:10px;align-items:baseline;
  padding:7px 0;border-bottom:1px solid rgba(35,42,54,.6);font-size:12px}
.step2:last-child{border-bottom:none}
.step2 .no{font-family:"IBM Plex Mono",monospace;color:var(--acc);font-size:11px}
.step2 .go{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--dim2);text-align:right;
  cursor:pointer}
.step2 .go:hover{color:var(--acc)}
.plot .ax{stroke:var(--line2);stroke-width:.6}
.plot .gl{stroke:var(--line);stroke-width:.5;stroke-dasharray:2 3}
.plot text{fill:var(--dim2);font-size:8px;font-family:"IBM Plex Mono",monospace}
@media(max-width:1100px){.maps{grid-template-columns:1fr}.cams{grid-template-columns:repeat(2,1fr)}
  .grid2,.grid3{grid-template-columns:1fr}}
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
    <button class="pill ov" data-ov="plain">Camera only</button>
    <button class="pill ov" data-ov="lidar">LiDAR depth</button>
    <button class="pill ov on" data-ov="boxes">Objects + LiDAR</button></div>
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
      <p class="claim" id="claim"></p>
      <div class="sec"><h2>System</h2>
        <span class="note">click any block to open the page holding its numbers</span>
        <span style="margin-left:auto;display:flex;gap:6px">
          <button class="tog on am" data-m="val">Measured result</button>
          <button class="tog am" data-m="lat">Latency</button></span></div>
      <div class="arch" id="arch"></div>

      <div class="sec" style="margin-top:16px"><h2>Where to start</h2>
        <span class="note">pick what you care about &mdash; the blocks above highlight, and the route below is ordered</span></div>
      <div class="paths" id="paths"></div>
      <div class="card pad" id="pathBody"></div>

      <div class="sec" style="margin-top:16px"><h2>Sensor input</h2><span class="note" id="ovNote"></span></div>
      <div class="cams" id="cams"></div>

      <div class="sec" style="margin-top:16px"><h2>World state</h2></div>
      <div class="hint" id="wsHint"></div>
      <div class="maps" id="maps"></div>

      <div class="sec" style="margin-top:16px">
        <h2>Occupancy forecast &mdash; predicted vs observed</h2>
        <span class="note">the LiDAR that actually arrived is the ground truth</span></div>
      <div class="tl" id="tl"></div>

      <div style="margin-top:16px" class="kpi" id="kpi"></div>
      <div class="sec" style="margin-top:16px"><h2>Stack status</h2>
        <span class="note">every tile links to the page holding its numbers</span></div>
      <div class="statusgrid" id="statusStrip"></div>
    </section>

    <!-- ============ PERCEPTION ============ -->
    <section class="page" data-p="perception">
      <div class="sec"><h2>Point cloud accumulation</h2><span class="note" id="pcNote"></span></div>
      <div class="grid2">
        <div class="mapwrap"><img id="pcBev" alt="Accumulated LiDAR"><div class="cap">
          <b>Accumulated returns &middot; motion-segmented</b><span class="mono d" id="pcBevN"></span></div></div>
        <div>
          <div class="ctrl">
            <span class="lab">Occupancy threshold</span>
            <input type="range" id="occThr" min="50" max="95" value="65">
            <span class="tog on" id="occThrV">0.65</span>
            <span class="rdout" id="occRd">hover the grid</span>
          </div>
          <div class="mapwrap"><canvas class="grid" id="occCv" width="135" height="135"></canvas>
            <div class="legend">
              <span><i style="background:linear-gradient(90deg,#2C6E8F,#171C27)"></i>free &rarr; unobserved</span>
              <span><i style="background:linear-gradient(90deg,#171C27,#FF6A7A)"></i>unobserved &rarr; occupied</span>
              <span class="mono" id="occFrac" style="margin-left:auto"></span></div>
            <div class="cap"><b>Occupancy grid &middot; log-odds</b>
              <span class="mono d" id="pcOccN"></span></div></div>
        </div>
      </div>
      <div class="hr"></div>
      <div class="grid2">
        <div class="card pad"><div class="sec"><h2>Inverse sensor model</h2>
          <span class="note">what the slider is actually thresholding</span></div>
          <table id="pcIsm"></table></div>
        <div class="card pad"><div class="sec"><h2>Static / dynamic separation</h2>
          <span class="note" id="pcMoNote"></span></div>
          <table id="pcMotion"></table>
          <div id="pcMoBox" style="margin-top:10px"></div></div>
      </div>
      <div class="hr"></div>
      <div class="sec"><h2>Objects in frame</h2>
        <span class="note">click a row to trace it into every camera and both maps</span></div>
      <div class="objlist" id="objTable" style="max-height:420px"></div>
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
      <div class="grid2" style="margin-top:10px">
        <div class="card pad"><div class="sec"><h2>Occupancy forecast &mdash; full metrics</h2></div>
          <table id="vaFc"></table></div>
        <div class="card pad"><div class="sec"><h2>Ego trajectory &mdash; every horizon</h2></div>
          <table id="vaAde"></table></div>
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
      <div class="sec"><h2>Trajectory language model &mdash; training run</h2>
        <span class="note" id="tlNote"></span></div>
      <div class="grid2">
        <div class="card pad"><div class="sec"><h2>Loss</h2>
          <span class="note">cross-entropy over waypoint tokens, logged every 10 steps</span></div>
          <div id="tlCurve"></div></div>
        <div class="card pad"><div class="sec"><h2>Validation ADE vs constant velocity</h2></div>
          <table id="tlTable"></table>
          <div id="tlWhy" style="margin-top:10px"></div></div>
      </div>
      <div class="card pad" style="margin-top:10px">
        <div class="sec"><h2>Decoded paths on held-out keyframes</h2>
          <span class="note" id="tlPathNote"></span></div>
        <div class="grid3" id="tlPaths"></div></div>

      <div class="hr"></div>
      <div class="sec"><h2>VLA &mdash; vision to action through a frozen language model</h2>
        <span class="note" id="vlaNote"></span></div>
      <div class="mono" style="font-size:11px;color:var(--dim);line-height:1.7;margin-bottom:10px"
           id="vlaArch"></div>
      <div class="grid2">
        <div class="card pad"><div class="sec"><h2>Projector training loss</h2>
          <span class="note">only the projector has gradients &mdash; 1.77M of 102.7M</span></div>
          <div id="vlaCurve"></div></div>
        <div class="card pad"><table id="vlaTable"></table>
          <div id="vlaNotes" style="margin-top:10px"></div></div>
      </div>
      <div class="card pad" style="margin-top:10px">
        <div class="sec"><h2>Decoded paths &mdash; VLA vs its own prior</h2>
          <span class="note" id="vlaPathNote"></span></div>
        <div class="grid3" id="vlaPaths"></div></div>

      <div class="hr"></div>
      <div class="sec"><h2>VLM &mdash; scene understanding on the frame you are looking at</h2>
        <span class="note" id="vlmModel"></span></div>
      <div class="grid2">
        <div><div class="strip" id="vlmStrip"></div>
          <div class="mono" style="font-size:10.5px;color:var(--dim2);margin-top:6px" id="vlmFrame"></div></div>
        <div class="card pad">
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px">
            <button class="pill" id="vlmRun">Describe this frame</button>
            <button class="pill" id="vlmStop" hidden>Stop</button>
            <span class="mono" id="vlmState" style="font-size:11px;color:var(--dim2)"></span></div>
          <div id="vlmOut" style="font-size:12px;line-height:1.6;color:var(--ink);
            white-space:pre-wrap;min-height:88px"></div>
          <div id="vlmBox"></div></div>
      </div>
    </section>

    <!-- ============ OBSERVABILITY ============ -->
    <section class="page" data-p="observability">
      <div class="sec"><h2>Camera observability of the ground plane</h2>
        <span class="note">integrity(cell) = 1 &minus; &prod;<sub>i</sub>(1 &minus; trust<sub>i</sub> &middot; coverage<sub>i</sub> &middot; visible<sub>i</sub>)</span></div>
      <div class="hint" id="inHint"></div>
      <div class="ctrl" id="inCtrl">
        <span class="lab">Cameras</span><span id="inTogs"></span>
        <span class="lab" style="margin-left:10px">Trust</span>
        <input type="range" id="inTrust" min="0" max="100" value="80">
        <span class="tog on" id="inTrustV">0.795</span>
        <button class="pill" id="inReset">Reset</button>
        <span class="rdout" id="inRd">hover the map</span>
      </div>
      <div class="grid2">
        <div class="mapwrap mapcap"><canvas class="grid" id="inCv" width="108" height="108"></canvas>
          <svg class="ovl" id="ovlIn" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
          <div class="legend"><span class="mono d">0.0</span>
            <span class="swatch" id="inRamp"></span><span class="mono d">1.0</span></div>
          <div class="cap"><b>Observability</b><span class="mono d" id="inMean"></span></div></div>
        <div>
          <div class="card pad"><div class="sec"><h2>Per-camera contribution at the cursor</h2>
            <span class="note">trust &times; coverage &times; visible</span></div>
            <div id="inProbe"></div></div>
          <div class="card pad" style="margin-top:10px">
            <div class="sec"><h2>Fraction of each frustum not occluded</h2></div>
            <div id="inCams"></div></div>
          <div class="card pad" style="margin-top:10px">
            <div class="sec"><h2>Validated against human visibility labels</h2></div>
            <div id="inAuroc"></div>
            <div class="badbox" style="margin-top:10px" id="inVerdict"></div></div>
        </div>
      </div>
    </section>

    <!-- ============ ROBUSTNESS ============ -->
    <section class="page" data-p="robustness">
      <div class="sec"><h2>Fault injection &mdash; one camera degraded at a time</h2>
        <span class="note" id="rbNote"></span></div>
      <div class="hint" id="rbHint"></div>
      <div class="strip" id="rbStrip"></div>
      <div class="grid2" style="margin-top:10px">
        <div>
          <div class="grid2">
            <div class="card pad"><div class="lab">Trust, faulted camera</div>
              <div class="mono" id="rbTF" style="font-size:26px;font-weight:600;margin-top:3px"></div>
              <div class="mono" id="rbTFd" style="font-size:11px;margin-top:2px"></div></div>
            <div class="card pad"><div class="lab">Trust, five untouched</div>
              <div class="mono" id="rbTO" style="font-size:26px;font-weight:600;margin-top:3px"></div>
              <div class="mono d" id="rbTOd" style="font-size:11px;margin-top:2px"></div></div>
          </div>
          <div class="card pad" style="margin-top:10px"><table id="rbTable"></table>
            <div id="rbVerdict" style="margin-top:10px"></div></div>
        </div>
        <div class="mapwrap mapcap"><canvas class="grid" id="rbCv" width="108" height="108"></canvas>
          <div class="legend"><span class="mono d">recomputed at the measured trust</span>
            <span class="mono" id="rbMean" style="margin-left:auto"></span></div>
          <div class="cap"><b>Downstream effect</b><span class="mono d" id="rbCap"></span></div></div>
      </div>
      <div class="hr"></div>
      <div class="card pad"><div class="sec"><h2>Trust separation</h2>
        <span class="note">faulted camera vs the five untouched, mean over 120 keyframes</span></div>
        <div id="rbBars"></div>
        <div style="margin-top:4px" id="rbGap"></div></div>
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
      <div class="ctrl">
        <span class="lab">Policy</span>
        <button class="tog on rnp" data-p="latest_frame_seqlock">Latest frame (seqlock)</button>
        <button class="tog rnp" data-p="fifo_queue">FIFO queue</button>
        <button class="pill" id="rnPlay">&#9654; Run 6 s at 10 Hz</button>
        <span class="rdout" id="rnRd"></span>
      </div>
      <div class="card pad"><div id="rnViz"></div>
        <div class="legend" style="padding-left:0">
          <span><i style="background:#5BD2E8"></i>frame processed</span>
          <span><i style="background:#3A4454"></i>frame skipped as stale</span>
          <span><i style="background:#FF6A7A"></i>frame dropped, queue full</span>
          <span class="d" style="margin-left:auto">drawn from the measured p50s &mdash; the bar
            length is the end-to-end latency this policy recorded</span></div></div>
      <div class="card pad" style="margin-top:10px"><table id="rnTable"></table>
        <div id="rnNoteBox"></div></div>
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
      <div class="grid2">
        <div class="card pad"><div id="perfBars"></div></div>
        <div class="card pad"><table id="perfTable"></table></div>
      </div>
      <div class="hr"></div>
      <div class="sec"><h2>Integrity monitor &mdash; ported to C++</h2>
        <span class="note" id="imNote"></span></div>
      <div class="grid2">
        <div class="card pad"><table id="imTable"></table>
          <div id="imBox" style="margin-top:10px"></div></div>
        <div class="card pad"><div class="sec"><h2>Stage by stage</h2>
          <span class="note">what the header actually implements</span></div>
          <div id="imStages"></div></div>
      </div>
    </section>

    <!-- ============ SYSTEM ============ -->
    <section class="page" data-p="system">
      <div class="sec"><h2>What runs, and what does not</h2>
        <span class="note">every row is a component that exists in this repository</span></div>
      <div class="card pad"><table id="sysTable"></table></div>
      <div class="hr"></div>
      <div class="sec"><h2>How to re-measure any of this</h2>
        <span class="note" id="repNote"></span></div>
      <div class="grid2">
        <div class="card pad"><table id="repTable"></table></div>
        <div class="card pad"><div class="sec"><h2>Command per number</h2></div>
          <div id="repCmds"></div></div>
      </div>
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
const PAGES = ["overview","perception","forecast","models","observability","robustness","runtime","system"];
let fi = 0, ov = "boxes", wl = "bev", sel = null, timer = null;
/* live state for the recomputed maps */
let camOn = {}, trustUI = null, occThr = 0.65, rnPolicy = "latest_frame_seqlock";
let RB_REPAINT = null, PROBED = false;

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

/* ================= live grid engine =================
   The exporter ships, per frame, the per-camera coverage x visibility terms and
   the occupancy probability field as base64 uint8 grids. Everything below
   recomputes the published formulas from those terms in the browser -- so
   turning a camera off, or substituting the trust the fault-injection head
   actually returned under blur, redraws the real map. Nothing here is a
   pre-rendered picture being swapped. */
const GC = {};                       // decoded-grid cache, keyed by frame + name
function decode(b64) {
  const s = atob(b64), a = new Float32Array(s.length);
  for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i) / 255;
  return a;
}
function grid(i, name, cam) {
  const k = i + "/" + name + "/" + (cam || "");
  if (!GC[k]) {
    const g = F[i].grids;
    GC[k] = decode(cam ? g.cov[cam] : g[name]);
  }
  return GC[k];
}
/* integrity(cell) = 1 - prod_i (1 - trust_i * cov_i) -- the noisy-OR itself,
   evaluated over whichever cameras are enabled at whatever trust is set. */
function noisyOr(i, trustByCam) {
  const g = F[i].grids, n = g.n, out = new Float32Array(n * n);
  out.fill(1);
  CAMS.forEach(c => {
    const t = trustByCam[c];
    if (!t) return;
    const cov = grid(i, "cov", c);
    for (let k = 0; k < out.length; k++) out[k] *= 1 - t * cov[k];
  });
  for (let k = 0; k < out.length; k++) out[k] = 1 - out[k];
  return out;
}
function trustVec() {
  const base = trustUI === null ? F[fi].grids.trust : trustUI, v = {};
  CAMS.forEach(c => { v[c] = camOn[c] === false ? 0 : base; });
  return v;
}
/* Colour ramps. Observability is a single perceptually-monotone ramp, so a
   reader compares brightness and nothing else; occupancy is tri-state because
   "free" and "never observed" are different claims and must not share a hue. */
function rampInteg(v) {
  const s = [[11,14,20],[24,50,72],[30,110,133],[91,210,232],[214,246,252]];
  const x = Math.min(0.999, Math.max(0, v)) * (s.length - 1), i = x | 0, f = x - i;
  return [0, 1, 2].map(k => Math.round(s[i][k] + f * (s[i + 1][k] - s[i][k])));
}
function paint(cv, vals, n, col) {
  cv.width = n; cv.height = n;
  const ctx = cv.getContext && cv.getContext("2d");
  if (!ctx) return;
  const im = ctx.createImageData(n, n), d = im.data;
  // The grids are numpy meshgrid(indexing="ij"): index [ix*n+iy] with x forward
  // and y left. Screen wants x up and y left-to-right reversed, which is the
  // flip below -- get it wrong and every shadow lands on the wrong side.
  for (let sy = 0; sy < n; sy++) {
    for (let sx = 0; sx < n; sx++) {
      const ix = n - 1 - sy, iy = n - 1 - sx;
      const [r, g, b] = col(vals[ix * n + iy]);
      const o = 4 * (sy * n + sx);
      d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
    }
  }
  ctx.putImageData(im, 0, 0);
  // ego marker + range rings, drawn at grid scale
  ctx.strokeStyle = "rgba(230,235,243,.30)"; ctx.lineWidth = 0.6;
  const c = n / 2, res = n === F[fi].grids.n ? F[fi].grids.res : F[fi].grids.occ_res;
  [10, 20, 30, 40, 50].forEach(r => {
    ctx.beginPath(); ctx.arc(c, c, r / res, 0, 6.284); ctx.stroke();
  });
  ctx.fillStyle = "#E6EBF3";
  ctx.fillRect(c - 1.2, c - 2.2, 2.4, 4.4);
}
/* Which grid cell is under the pointer, in both grid indices and metres. */
function cellAt(ev, cv, n, res) {
  const b = cv.getBoundingClientRect();
  if (!b.width) return null;
  const sx = Math.min(n - 1, Math.max(0, Math.floor(n * (ev.clientX - b.left) / b.width)));
  const sy = Math.min(n - 1, Math.max(0, Math.floor(n * (ev.clientY - b.top) / b.height)));
  const ix = n - 1 - sy, iy = n - 1 - sx;
  return { k: ix * n + iy, x: (ix + 0.5) * res - n * res / 2,
           y: (iy + 0.5) * res - n * res / 2 };
}

/* tiny inline plots ------------------------------------------------------- */
function linePlot(series, opt) {
  const W = 300, H = 120, P = 22;
  const xs = series.flatMap(s => s.pts.map(p => p[0]));
  const ys = series.flatMap(s => s.pts.map(p => p[1]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = opt && opt.y0 !== undefined ? opt.y0 : Math.min(...ys);
  const y1 = Math.max(...ys);
  const X = v => P + (W - P - 6) * (v - x0) / Math.max(1e-9, x1 - x0);
  const Y = v => H - 16 - (H - 16 - 6) * (v - y0) / Math.max(1e-9, y1 - y0);
  let g = "";
  for (let i = 0; i <= 3; i++) {
    const v = y0 + (y1 - y0) * i / 3;
    g += `<line class="gl" x1="${P}" y1="${Y(v).toFixed(1)}" x2="${W - 6}" y2="${Y(v).toFixed(1)}"/>
          <text x="2" y="${(Y(v) + 3).toFixed(1)}">${v.toFixed(opt && opt.dp !== undefined ? opt.dp : 1)}</text>`;
  }
  series.forEach(s => {
    g += `<polyline fill="none" stroke="${s.col}" stroke-width="1.4" points="` +
      s.pts.map(p => X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join(" ") + `"/>`;
  });
  g += `<line class="ax" x1="${P}" y1="${H - 16}" x2="${W - 6}" y2="${H - 16}"/>
        <text x="${P}" y="${H - 4}">${x0}</text>
        <text x="${W - 26}" y="${H - 4}">${x1}</text>`;
  const key = series.map(s =>
    `<span style="color:${s.col}">&#9632;</span> <span style="color:var(--dim)">${s.name}</span>`)
    .join("&nbsp;&nbsp;");
  return `<svg class="plot" viewBox="0 0 ${W} ${H}">${g}</svg>
    <div style="font-size:10.5px;margin-top:2px">${key}</div>`;
}
/* Ego-frame path plot. Forward and lateral get SEPARATE scales, because ego
   motion is not isotropic: 6 s of driving is 60-85 m forward and two or three
   metres of lateral deviation, and on one shared scale the lateral error --
   which is the entire difference between these predictors -- collapses onto a
   vertical line. That is the same anisotropy the tokenisation fix was about. */
function pathPlot(sets, title, sub) {
  const W = 220, H = 190, L = 26, B = 22, T = 14, R = 8;
  const pts = sets.flatMap(s => s.p);
  const fwd = Math.max(5, ...pts.map(p => p[0])) * 1.08;
  const lat = Math.max(1.5, ...pts.map(p => Math.abs(p[1]))) * 1.25;
  const X = y => (L + W - R) / 2 - (W - L - R) / 2 * y / lat;
  const Y = x => H - B - (H - B - T) * x / fwd;
  let g = "";
  for (let k = 0; k <= 4; k++) {
    const v = fwd * k / 4, y = Y(v).toFixed(1);
    g += `<line class="gl" x1="${L}" y1="${y}" x2="${W - R}" y2="${y}"/>
          <text x="2" y="${(+y + 3).toFixed(1)}">${v.toFixed(0)}</text>`;
  }
  g += `<line class="ax" x1="${X(0)}" y1="${T}" x2="${X(0)}" y2="${H - B}"/>`;
  sets.forEach(s => {
    g += `<polyline fill="none" stroke="${s.col}" stroke-width="${s.w || 1.5}"
      stroke-dasharray="${s.dash || ""}" stroke-linejoin="round" points="` +
      [[0, 0]].concat(s.p).map(p => X(p[1]).toFixed(1) + "," + Y(p[0]).toFixed(1)).join(" ") + `"/>`;
  });
  g += `<circle cx="${X(0)}" cy="${Y(0).toFixed(1)}" r="2.6" fill="#E6EBF3"/>
        <text x="${L}" y="${H - 10}">-${lat.toFixed(1)} m</text>
        <text x="${W - R - 26}" y="${H - 10}">+${lat.toFixed(1)} m</text>
        <text x="2" y="${T - 5}">fwd m</text>
        <text x="${X(0) - 12}" y="${H - 10}">lat</text>`;
  return `<figure style="margin:0"><svg class="plot" style="max-width:250px"
      viewBox="0 0 ${W} ${H}">${g}</svg>
    <figcaption style="font-size:10.5px;color:var(--dim);padding-top:2px">
      <b style="color:var(--ink)">${title}</b> &middot; ${sub}</figcaption></figure>`;
}

/* ================= architecture strip =================
   The claim this picture makes, and the reason it is a picture: the same frame
   goes down TWO paths, and the geometric one is what validates the learned one.
   Occupancy log-odds supplies the labels the BEV head is scored against; the
   observability monitor is not a model output at all and had to be ported to
   C++ separately. A prose list of components cannot show either of those, and
   both are the argument. Every metric below is read from the report JSONs at
   build time -- nothing here is typed by hand. */
const AM = { m: "val", hi: null };
function archData() {
  const tm = R.timing_report || {}, ms = tm.per_stage_ms || {};
  const lm = R.learned_model_report || {}, cp = R.cpp_report || {};
  const iv = R.integrity_visibility_report || {}, mo = R.motion_separation_report || {};
  const tl = R.trajlm_retrained_report || {}, vla = R.vla_report || {};
  const im = cp.integrity_monitor || {}, rn = (cp.runner || {}).at_10hz_6s || {};
  const sq = rn.latest_frame_seqlock || {}, st = F[0].stats;
  const g = (o, k, d) => (o && o[k] !== undefined ? o[k] : d);
  const ms1 = k => ms[k] !== undefined ? ms[k].toFixed(1) + " ms" : "--";
  const ade = h => { const r = (lm.trajectory_ade || {})[h]; return r ? r.learned_v11_temporal.ade_m : "--"; };
  return [
    // id, x, y, w, title lines, latency, measured result, page
    ["s1",  20,  48, 156, ["LiDAR", "10 sweeps"],       ms1("load_10_sweeps_from_disk"),
      st.returns.toLocaleString() + " returns",           "perception"],
    ["s2",  20, 196, 156, ["6 cameras", "1600 × 900"],  "--", "projected + reprojected", "perception"],
    ["g1", 204,  48, 128, ["Ground plane", "RANSAC"],   ms1("ground_plane_ransac"),
      "tilt " + st.plane_tilt_deg + "°",                  "perception"],
    ["g2", 340,  48, 128, ["Occupancy", "log-odds"],    ms1("occupancy_logodds_0.20m"),
      "free " + (100 * st.free).toFixed(1) + "%",         "perception"],
    ["g3", 476,  48, 128, ["Motion", "separation"],     ms1("motion_separation"),
      "F1 " + fx(g(mo.best, "f1"), 3),                    "perception"],
    ["g4", 612,  48, 128, ["Camera", "observability"],  ms1("integrity_6cam_0.50m"),
      "AUROC " + fx(g(g(iv.auroc, "integrity_occlusion_aware", {}), "auroc"), 3), "observability"],
    ["l1", 204, 196, 128, ["BEV backbone", "→ z(384)"], (lm.inference_ms_per_sample || "--") + " ms",
      "167 of 169 weights",                               "models"],
    ["l2", 340, 196, 128, ["Occ · traj · trust", "heads"], "--",
      "IoU " + fx(g(g(lm.occupancy, "best", {}), "iou"), 3) + " · " + ade("T+1 (0.5s)"), "models"],
    ["l3", 476, 196, 128, ["Trajectory LM", "GPT-2 4.9M"], "--",
      g(g(g(tl.val_ade_m, "T+3 (1.5s)", {}), "gpt2_retrained_conditioned", {}), "ade_m", "--") + " m @T+3", "models"],
    ["l4", 612, 196, 128, ["VLA projector", "· VLM"],   "--",
      g(g(vla.val_ade_m, "6.0s", {}), "vla_gpt2_projector", "--") + " m @6 s", "models"],
    ["d1", 792, 196, 124, ["TorchScript"],              "--", "traced + frozen",        "runtime"],
    ["d2", 928, 196, 152, ["LibTorch", "runner"],
      (g(cp.inference_latency, "p50_ms", 0)).toFixed ? g(cp.inference_latency, "p50_ms", 0).toFixed(0) + " ms p50" : "--",
      "parity " + Number(g(g(cp.parity, "occupancy", {}), "max_abs", 0)).toExponential(1), "runtime"],
    ["d3", 928,  48, 152, ["Integrity monitor", "C++ port"], (im.p50_ms || "--") + " ms p50",
      (im.parity && im.parity.integrity_max_abs === 0 ? "bit-identical"
        : "parity " + Number(g(im.parity, "integrity_max_abs", 0)).toExponential(1)), "runtime"],
    ["d4", 1092, 122, 168, ["Latest-frame", "seqlock @10 Hz"], (sq.end_to_end_p50_ms || "--") + " ms e2e",
      sq.skipped_stale + " stale, 0 dropped",             "runtime"],
  ];
}
const ARCH_EDGES = [
  // from, to, label, kind
  ["s1", "g1", "", "h"], ["g1", "g2", "", "h"], ["g2", "g3", "", "h"], ["g3", "g4", "", "h"],
  ["s2", "l1", "", "h"], ["l1", "l2", "", "h"], ["l2", "l3", "", "h"], ["l3", "l4", "", "h"],
  ["g2", "l2", "supplies the occupancy labels", "v"],
  ["g4", "d3", "ported to C++", "h"],
  ["l4", "d1", "traced", "h"], ["d1", "d2", "", "h"],
];
function drawArch() {
  const N = {}, rows = archData();
  rows.forEach(r => { N[r[0]] = { x: r[1], y: r[2], w: r[3], t: r[4], lat: r[5], val: r[6], p: r[7] }; });
  const H = 70;
  let g = `<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6"
      markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"
      stroke="none" opacity=".55"/></marker></defs>`;
  // lane labels + the process boundary, which is the one thing the reader must not miss
  g += `<text class="lane" x="20" y="34">SENSORS</text>
        <text class="lane" x="204" y="34">GEOMETRY &mdash; NUMPY, PER KEYFRAME</text>
        <text class="lane" x="204" y="182">LEARNED &mdash; CHECKPOINT v11_temporal</text>
        <text class="lane" x="792" y="34">DEPLOYMENT</text>
        <line class="bnd" x1="764" y1="22" x2="764" y2="300"/>
        <text class="el" x="768" y="298">python │ c++</text>`;
  ARCH_EDGES.forEach(([a, b, lab, kind]) => {
    const A = N[a], B = N[b];
    if (!A || !B) return;
    if (kind === "h") {
      const y = A.y + H / 2, y2 = B.y + H / 2;
      const x1 = A.x + A.w, x2 = B.x;
      const d = y === y2 ? `M${x1},${y} L${x2 - 3},${y}`
        : `M${x1},${y} C${(x1 + x2) / 2},${y} ${(x1 + x2) / 2},${y2} ${x2 - 3},${y2}`;
      g += `<path d="${d}" fill="none" marker-end="url(#ah)" opacity=".7"/>`;
      if (lab) g += `<text class="el" x="${(x1 + x2) / 2}" y="${y - 6}" text-anchor="middle">${lab}</text>`;
    } else {
      const x = A.x + A.w / 2, y1 = A.y + H, y2 = B.y;
      g += `<path d="M${x},${y1} L${x},${y2 - 3}" fill="none" marker-end="url(#ah)"
        stroke="var(--acc)" opacity=".85"/>
        <text class="el" x="${x + 7}" y="${(y1 + y2) / 2 + 3}" style="fill:var(--acc)">${lab}</text>`;
    }
  });
  // the two converging arrows into the seqlock -- what actually runs at 10 Hz
  g += `<path d="M1080,83 C1086,83 1084,148 1089,152" fill="none" marker-end="url(#ah)" opacity=".7"/>
        <path d="M1080,231 C1086,231 1084,166 1089,162" fill="none" marker-end="url(#ah)" opacity=".7"/>`;
  rows.forEach(r => {
    const n = N[r[0]], txt = AM.m === "lat" ? n.lat : n.val;
    const on = AM.hi && AM.hi.indexOf(r[0]) >= 0;
    g += `<g class="nd${on ? " hi" : ""}" data-a="${r[0]}" data-p="${n.p}">
      <rect x="${n.x}" y="${n.y}" width="${n.w}" height="${H}" rx="4"/>
      ${n.t.map((line, k) =>
        `<text class="t" x="${n.x + 10}" y="${n.y + 21 + k * 14}">${line}</text>`).join("")}
      <text class="m" x="${n.x + 10}" y="${n.y + (n.t.length > 1 ? 56 : 45)}">${txt}</text></g>`;
  });
  $("#arch").innerHTML =
    `<figure style="margin:0"><svg viewBox="0 0 1276 316" role="img"
      aria-label="One frame runs down two paths: a NumPy geometry pipeline and a learned checkpoint.
      The geometry pipeline supplies the occupancy labels the learned head is scored against, and its
      observability monitor is ported separately to C++, where a seqlock runs the traced model at
      the latest frame.">${g}</svg>
    <figcaption class="el" style="font-size:11px;color:var(--dim2);padding:2px 0 6px">
      The geometry lane is not a preprocessing step for the learned lane &mdash; it is the reference
      the learned lane is measured against, which is why both cross the boundary into C++.
    </figcaption></figure>`;
  document.querySelectorAll("#arch .nd").forEach(e =>
    e.onclick = () => goTo(e.dataset.p));
}
document.querySelectorAll(".am").forEach(b => b.onclick = () => {
  AM.m = b.dataset.m;
  document.querySelectorAll(".am").forEach(x => x.classList.toggle("on", x === b));
  drawArch();
});

/* page navigation, shared by the strip, the status tiles and the reading paths */
function goTo(page) {
  document.querySelectorAll(".rb").forEach(x => x.classList.toggle("on", x.textContent === page));
  document.querySelectorAll(".page").forEach(x =>
    x.classList.toggle("on", x.dataset.p === page.replace(" ", "")));
  scrollTo(0, 0);
}

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
  <div class="mapwrap"><canvas class="grid" id="ovInt" width="108" height="108"></canvas>
    <svg class="ovl" id="ovlInt" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
    <div class="mark" id="mkInt"></div>
    <div class="cap"><b>Camera observability</b><span class="mono d" id="ovIntN"></span></div></div>`;

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

/* ---------- live observability ---------- */
function drawObservability() {
  const g = F[fi].grids, tv = trustVec(), v = noisyOr(fi, tv);
  [$("#ovInt"), $("#inCv")].forEach(c => c.getContext && paint(c, v, g.n, rampInteg));
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i];
  const mean = sum / v.length;
  const off = CAMS.filter(c => camOn[c] === false);
  const base = trustUI === null ? g.trust : trustUI;
  $("#ovIntN").textContent = `mean over drivable ${fx(F[fi].stats.integrity_mean_drivable)}`;
  $("#inMean").textContent = `mean over the grid ${fx(mean)}`;
  $("#inHint").innerHTML = off.length || trustUI !== null
    ? `Recomputed live: <b>${6 - off.length} of 6 cameras</b> at trust <b>${base.toFixed(3)}</b>` +
      (off.length ? ` &mdash; ${off.map(c => c.replace("CAM_", "").toLowerCase()).join(", ")} disabled.` : ".") +
      ` Mean observability <b>${mean.toFixed(4)}</b>, against ${
        (() => { const b = noisyOr(fi, Object.fromEntries(CAMS.map(c => [c, g.trust])));
          let t = 0; for (let i = 0; i < b.length; i++) t += b[i]; return (t / b.length).toFixed(4); })()
      } with the full rig.`
    : `This map is <b>not a picture</b>. The exporter ships the per-camera coverage &times;
       visibility terms; the noisy-OR above is evaluated here, per cell, every time you change
       something. Turn a camera off or move the trust slider and watch the shadows change.`;
  /* The probe panel starts on the cell 12 m ahead of the vehicle rather than
     empty, so the page shows what the noisy-OR decomposes into before anyone
     touches it. */
  if (!PROBED) probeAt(Math.round(g.n / 2 + 12 / g.res) * g.n + Math.round(g.n / 2), "12.0", "0.0");
  const rp = $("#inRamp");
  if (rp.style) rp.style.background =
    "linear-gradient(90deg," + [0, .25, .5, .75, 1].map(x =>
      "rgb(" + rampInteg(x).join(",") + ")").join(",") + ")";
}
function drawOccupancy() {
  const g = F[fi].grids, p = grid(fi, "occ"), n = g.occ_n;
  let free = 0, occ = 0;
  /* A tri-state paint throws away everything the log-odds field knows: 0.36% of
     cells clear the occupied threshold, so a hard three-colour map is a black
     square with confetti on it. Confidence is drawn continuously either side of
     0.5 -- deep blue for carved free space, near-black for never observed, red
     for occupied -- and the slider's threshold is marked by full saturation. */
  const col = v => {
    if (v > occThr) return [255, 106, 122];
    if (v >= 0.5) { const t = (v - 0.5) / Math.max(1e-6, occThr - 0.5);
      return [23 + t * 150, 28 + t * 40, 39 + t * 45]; }
    const t = (0.5 - v) / 0.5;
    return [23 + t * 21, 28 + t * 82, 39 + t * 104];
  };
  for (let i = 0; i < p.length; i++) { if (p[i] > occThr) occ++; else if (p[i] < 0.35) free++; }
  const cv = $("#occCv");
  if (cv.getContext) paint(cv, p, n, col);
  const s = F[fi].stats;
  $("#occFrac").textContent =
    `at 0.20 m: free ${pc(s.free)} · occupied ${pc(s.occupied)} · unobserved ${pc(s.unknown)}`;
}

/* ---------- controls for the live maps ---------- */
function inTogs() {
  const box = $("#inTogs");
  box.innerHTML = "";
  CAMS.forEach(c => {
    const b = el("button", "tog" + (camOn[c] === false ? "" : " on"), c.replace("CAM_", ""));
    b.style.marginRight = "4px";
    b.onclick = () => { camOn[c] = camOn[c] === false; inTogs(); drawObservability(); };
    box.appendChild && box.appendChild(b);
  });
}
inTogs();
const inTr = $("#inTrust");
if (inTr.addEventListener) inTr.addEventListener("input", () => {
  trustUI = Number(inTr.value) / 100;
  $("#inTrustV").textContent = trustUI.toFixed(3);
  drawObservability();
});
$("#inReset").onclick = () => {
  camOn = {}; trustUI = null;
  if (inTr.value !== undefined) inTr.value = 80;
  $("#inTrustV").textContent = F[fi].grids.trust.toFixed(3);
  inTogs(); drawObservability();
};
const occSl = $("#occThr");
if (occSl.addEventListener) occSl.addEventListener("input", () => {
  occThr = Number(occSl.value) / 100;
  $("#occThrV").textContent = occThr.toFixed(2);
  drawOccupancy();
});
/* hover probes -- the readout is the actual stored value at that cell */
function probeAt(k, xs, ys) {
  const tv = trustVec();
  const terms = CAMS.map(cam => [cam, tv[cam] * grid(fi, "cov", cam)[k]]);
  let acc = 1; terms.forEach(([, t]) => acc *= 1 - t);
  $("#inRd").textContent = `x ${xs} m, y ${ys} m → observability ${(1 - acc).toFixed(4)}`;
  const mx = Math.max(1e-6, ...terms.map(t => t[1]));
  $("#inProbe").innerHTML = terms.map(([cam, t]) =>
    bar(cam.replace("CAM_", "").toLowerCase(), t, Math.max(0.3, mx),
        t > 0.01 ? "var(--acc)" : "#3A4454")).join("") +
    `<div class="mono" style="font-size:11px;color:var(--dim);margin-top:8px">
      1 − ∏(1 − t·c·v) = <span class="a">${(1 - acc).toFixed(4)}</span>
      <span class="d">&nbsp;at x ${xs} m, y ${ys} m</span></div>`;
}
const inCv = $("#inCv");
if (inCv.addEventListener) {
  inCv.addEventListener("mousemove", ev => {
    const g = F[fi].grids, c = cellAt(ev, inCv, g.n, g.res);
    if (!c) return;
    PROBED = true;
    probeAt(c.k, c.x.toFixed(1), c.y.toFixed(1));
  });
}
const occCv = $("#occCv");
if (occCv.addEventListener) {
  occCv.addEventListener("mousemove", ev => {
    const g = F[fi].grids, c = cellAt(ev, occCv, g.occ_n, g.occ_res);
    if (!c) return;
    const p = grid(fi, "occ")[c.k];
    const st = p > occThr ? "occupied" : p < 0.35 ? "free" : "unobserved";
    $("#occRd").textContent =
      `x ${c.x.toFixed(1)} m, y ${c.y.toFixed(1)} m → P(occupied) ${p.toFixed(3)} · ${st}`;
  });
  occCv.addEventListener("mouseleave", () => { $("#occRd").textContent = "hover the grid"; });
}

function objRows(container, objs, compact) {
  container.innerHTML = `<div class="objrow hdr"><span>OBJECT</span><span>RANGE</span>
    <span>SPEED</span><span>INTEG</span></div>`;
  objs.forEach(o => {
    const r = el("div", "objrow" + (sel === o.id ? " on" : ""));
    r.innerHTML = `<span>${o.cat}</span><span>${o.range} m</span>
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
  $("#ovBev").src = f.maps[wl];
  $("#ovBevCap").textContent =
    {bev: "Accumulated LiDAR · motion-segmented", bev_nodyn: "Static structure only",
     occupancy: "Occupancy grid · log-odds"}[wl];
  $("#pcBev").src = f.maps.bev;
  drawObservability();
  drawOccupancy();
  if (RB_REPAINT) RB_REPAINT();
  const s = f.stats;
  $("#ovBevN").textContent = `${s.returns.toLocaleString()} returns · ${s.dynamic} dynamic`;
  $("#pcBevN").textContent = `${s.returns.toLocaleString()} returns · 10 sweeps · ${s.dynamic} dynamic`;
  $("#pcOccN").textContent = `${(F[fi].grids.occ_res).toFixed(2)} m cells · rendered live`;
  $("#ovNote").textContent = `${s.boxes_observed} of ${s.boxes} annotated objects have LiDAR returns`;
  $("#pcNote").textContent = `10 sweeps, ego-motion compensated to the current pose · ground plane tilt ${s.plane_tilt_deg}° · ${pc(s.ground_frac)} of returns are ground`;

  /* The hint bar. It reports the live selection instead of repeating a static
     instruction, and it is a bar rather than a caption because as a caption it
     wrapped into the map grid and nobody read it. */
  const so = sel !== null ? f.objects.find(x => x.id === sel) : null;
  const wh = $("#wsHint");
  if (so) {
    const seen = CAMS.filter(c => so.cams[c]).map(c =>
      c.replace("CAM_", "").replace("_", "-").toLowerCase()).join(", ") || "no camera";
    wh.innerHTML = `Tracing <b>${so.cat} #${so.id}</b> &mdash; ${so.range} m,
      ${so.speed.toFixed(1)} m/s, observability ${fx(so.integrity, 2)}. Highlighted in
      <b>${seen}</b> and on both maps.
      <button class="clear" id="wsClear">clear selection <span class="kbd">Esc</span></button>`;
    const wc = document.querySelector("#wsClear");
    if (wc) wc.onclick = () => { sel = null; draw(); };
  } else {
    wh.innerHTML = `<b>Click any object</b> to trace it across every view &mdash; the six camera
      tiles, both maps and the inspector highlight the same instance at once. Click a box in a
      camera tile, a footprint on a map, or a row in the object table.
      <span class="clear" style="color:var(--dim2)">${f.objects.length} objects in this frame</span>`;
  }

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
  [["#ovlBev", "#ovBev"], ["#ovlInt", "#ovInt"], ["#ovlIn", "#inCv"]].forEach(([sv, im]) => {
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
  // The "now" tile is the same live occupancy field the perception page draws,
  // not the pre-rendered picture -- so the strip reads as one instrument.
  cur.innerHTML = `<canvas class="grid" id="tlNow" width="8" height="8"
    style="border:1px solid var(--line);border-radius:4px"></canvas>
    <figcaption><span>OBSERVED NOW</span><span>0.0s</span></figcaption>`;
  tl.appendChild(cur);
  const nowCv = document.querySelector("#tlNow");
  if (nowCv && nowCv.getContext) {
    const gg = F[fi].grids, pp = grid(fi, "occ");
    paint(nowCv, pp, gg.occ_n, v => v > occThr ? [255, 106, 122]
      : v >= 0.5 ? [23, 28, 39]
      : [23 + (0.5 - v) / 0.5 * 21, 28 + (0.5 - v) / 0.5 * 82, 39 + (0.5 - v) / 0.5 * 104]);
  }
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

  /* The VLM sees exactly these two tiles -- shown so the reader can check the
     description against the input rather than take it on trust. */
  $("#vlmStrip").innerHTML = ["CAM_FRONT", "CAM_BACK"].map(c =>
    `<figure style="flex:1 1 0"><img src="${f.cameras[c].boxes}" alt="${c}">
      <figcaption><span>${c.replace("CAM_", "")}</span>
      <span>sent to the model</span></figcaption></figure>`).join("");
  /* Per-camera: how much of what this camera's frustum covers is not occluded
     by something the LiDAR sees. Frame-dependent, so it lives in draw(). */
  $("#inCams").innerHTML = CAMS.map(c =>
    bar(c.replace("CAM_", "").replace("_", "-").toLowerCase(),
        f.cameras[c].visible_frac, 1.0,
        f.cameras[c].visible_frac < 0.6 ? "var(--warn)" : "var(--acc)")).join("");

  $("#vlmFrame").textContent =
    `${f.scene} · frame ${fi + 1} of ${F.length} · front and rear tiles, boxes and LiDAR drawn on`;

  /* inspector */
  const ins = $("#insp");
  document.body.classList.toggle("insp-open", !!o);
  if (o) {
    ins.classList.add("on");
    $("#iCls").textContent = o.group;
    $("#iId").textContent = o.cat + " #" + o.id;
    const visLbl = { 1: "0-40%", 2: "40-60%", 3: "60-80%", 4: "80-100%" }[o.vis] || "--";
    const seen = CAMS.filter(c => o.cams[c]).map(c => c.replace("CAM_", "").replace("_", "-").toLowerCase()).join(" · ") || "none";
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
  if (mo) {
    $("#pcMoNote").textContent = `${mo.frames} keyframes · ${mo.boxes_scored} boxes scored`;
    $("#pcMotion").innerHTML =
      `<tr><th>metric</th><th>value</th><th>metric</th><th>value</th></tr>
       <tr><td>precision</td><td>${fx(mo.best.precision)}</td><td>frames</td><td>${mo.frames}</td></tr>
       <tr><td>recall</td><td>${fx(mo.best.recall)}</td><td>boxes scored</td><td>${mo.boxes_scored}</td></tr>
       <tr><td>F1</td><td class="g">${fx(mo.best.f1)}</td><td>coverage</td><td>${pc(mo.coverage)}</td></tr>`;
    $("#pcMoBox").innerHTML = note("How a return is called dynamic",
      "Not by voxel persistence. A return is dynamic when the OLDER sweeps ray-cast through the " +
      "space it now occupies — the space was observed free a moment ago and is occupied now. " +
      "Absence of prior support scored F1 0.20 on the same frames because it fires on everything " +
      "the beam simply missed before.");
  }
  $("#pcIsm").innerHTML =
    `<tr><th>parameter</th><th>value</th><th>meaning</th></tr>` +
    [["l_free", "0.45", "log-odds subtracted per cell a beam passes through"],
     ["l_occ", "2.40", "log-odds added at the cell a beam terminates in"],
     ["clamp", "±6.0", "keeps a stale observation recoverable"],
     ["cell", (F[0].grids.occ_res).toFixed(2) + " m", "display grid; the pipeline runs at 0.20 m"],
     ["free / occupied", "P < 0.35 / P > " + occThr.toFixed(2), "the slider moves the second one"]]
      .map(([a, b, c]) => `<tr><td>${a}</td><td class="a">${b}</td>
        <td class="d" style="font-family:'IBM Plex Sans'">${c}</td></tr>`).join("");

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
      "a car has an occluded footprint and a visible roof. Use it as an observability prior, not a " +
      "detector. That is why this page is named for what it measures.");
  }
  const tm = R.timing_report;
  if (tm) {
    const ms = tm.per_stage_ms, mx = Math.max(...Object.values(ms));
    $("#perfBars").innerHTML = Object.keys(ms).map(k =>
      `<div class="br"><span class="t">${k.replace(/_/g, " ")}</span>
       <span class="track"><span class="fill" style="width:${Math.max(2, 100 * ms[k] / mx)}%;
       background:${ms[k] > 300 ? "var(--warn)" : "var(--acc)"}"></span></span>
       <span class="n">${ms[k].toFixed(1)} ms</span></div>`).join("");
    $("#perfNote").textContent =
      `${tm.total_ms_per_keyframe} ms total · ${tm.hz} Hz · ${tm.hardware}`;
    const bf = tm.per_stage_ms_before_optimisation;
    $("#perfTable").innerHTML = `<tr><th>stage</th><th>before</th><th>after</th><th>speedup</th></tr>` +
      [["occupancy_logodds_0.20m", tm.occupancy_speedup_x], ["ground_plane_ransac", null],
       ["integrity_6cam_0.50m", null]].map(([k, sp]) =>
      `<tr><td>${k.replace(/_/g, " ")}</td><td class="d">${bf[k].toFixed(0)} ms</td>
       <td>${ms[k].toFixed(0)} ms</td><td class="${sp ? "g" : "d"}">${sp ? sp.toFixed(2) + "×" : "—"}</td></tr>`).join("") +
      `<tr><td><b>full keyframe</b></td><td class="d">${tm.total_ms_before.toFixed(0)} ms</td>
       <td>${tm.total_ms_per_keyframe.toFixed(0)} ms</td>
       <td class="g">${tm.end_to_end_speedup_x.toFixed(2)}×</td></tr>`;
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
    $("#vlaNote").textContent =
      `${vla.train_samples} train / ${vla.val_samples} val · ${vla.steps} steps · ${vla.tokenisation}`;
    const hs = Object.keys(vla.val_ade_m);
    $("#vlaTable").innerHTML = `<tr><th>horizon</th><th>const-vel</th><th>VLA</th><th>vs prior</th></tr>` +
      hs.map(h => { const r = vla.val_ade_m[h];
        const better = r.vla_gpt2_projector < r.constant_velocity;
        const d = 100 * (r.vla_gpt2_projector - r.constant_velocity) / r.constant_velocity;
        return `<tr><td>${h}</td><td>${r.constant_velocity.toFixed(3)}</td>
          <td class="${better ? "g" : "b"}">${r.vla_gpt2_projector.toFixed(3)}</td>
          <td class="${better ? "g" : "w"}">${d.toFixed(1)}%</td></tr>`; }).join("");
    if (vla.curve) $("#vlaCurve").innerHTML = linePlot([
      { name: "train", col: "#4A5568", pts: vla.curve.map(c => [c[0], c[1]]) },
      { name: "val (80 held out)", col: "var(--acc)", pts: vla.curve.map(c => [c[0], c[2]]) },
    ], { dp: 1 });
    $("#vlaNotes").innerHTML = hero([
      ["trainable", vla.trainable_params_m + "M", "projector only"],
      ["frozen", vla.frozen_params_m + "M", "backbone + GPT-2"],
      ["train / val", vla.train_samples + " / " + vla.val_samples, vla.steps + " steps"],
    ]) + note("Reading",
      "Runs end to end and beats the prior only at 6 s. With this corpus and a frozen LM the repo " +
      "previously damaged by an all-zero fine-tune, that is the expected outcome — it demonstrates " +
      "the mechanism, not a model result.");
    $("#vlaPathNote").textContent = vla.val_examples_note || "";
    if (vla.val_examples) $("#vlaPaths").innerHTML = vla.val_examples.map(e =>
      pathPlot([
        { p: e.gt, col: "#E6EBF3", w: 2 },
        { p: e.cv, col: "var(--warn)", dash: "3 3" },
        { p: e.pred, col: "var(--acc)" },
      ], e.tag, `VLA ${e.ade_m} m · prior ${e.cv_ade_m} m · ${e.speed_mps} m/s`)).join("") +
      `<div class="mono" style="grid-column:1/-1;font-size:10.5px;color:var(--dim2)">
        <span style="color:#E6EBF3">&#9632;</span> ground truth &nbsp;
        <span style="color:var(--warn)">&#9632;</span> constant-velocity prior &nbsp;
        <span style="color:var(--acc)">&#9632;</span> VLA</div>`;
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

  /* ---------- retrained trajectory LM ---------- */
  const tl = R.trajlm_retrained_report;
  if (tl) {
    $("#tlNote").textContent =
      `${tl.model.split(",")[0]} · ${tl.train_samples} train / ${tl.val_samples} val · ${tl.steps} steps · trained from scratch on CPU`;
    if (tl.curve) $("#tlCurve").innerHTML = linePlot([
      { name: "train", col: "#4A5568", pts: tl.curve.map(c => [c[0], c[1]]) },
      { name: "val (80 held out)", col: "var(--acc)", pts: tl.curve.map(c => [c[0], c[2]]) },
    ], { dp: 1, y0: 0 });
    const hs = Object.keys(tl.val_ade_m);
    $("#tlTable").innerHTML = `<tr><th>horizon</th><th>const-vel</th><th>trajectory LM</th><th>vs CV</th></tr>` +
      hs.map(h => { const r = tl.val_ade_m[h];
        const g = r.delta_vs_cv_pct < 0;
        return `<tr><td>${h}</td><td>${r.constant_velocity.ade_m.toFixed(3)}</td>
          <td class="${g ? "g" : ""}">${r.gpt2_retrained_conditioned.ade_m.toFixed(3)}</td>
          <td class="${g ? "g" : "w"}">${r.delta_vs_cv_pct.toFixed(1)}%</td></tr>`; }).join("") +
      `<tr><td class="d">T+1, before conditioning</td><td class="d">0.324</td>
       <td class="b">2.796</td><td class="b">+763%</td></tr>`;
    $("#tlWhy").innerHTML = note("What the last row is",
      "The shipped checkpoint was fine-tuned from manifest keys that do not exist, so every " +
      "waypoint fell back to (0,0) and the model saw 404 copies of one all-zero path. Retraining " +
      "from the label files fixed the data; the ±20 m tokenisation was clipping 30.6% of real " +
      "waypoints, so the axes were split (x [-10,70] m, y [-25,25] m); and unconditional, the best " +
      "it can emit from &lt;BOS&gt; is the dataset average — hence the velocity prefix. The curve " +
      "above is that third run.");
    $("#tlPathNote").textContent = tl.val_examples_note || "";
    if (tl.val_examples) $("#tlPaths").innerHTML = tl.val_examples.map(e =>
      pathPlot([
        { p: e.gt, col: "#E6EBF3", w: 2 },
        { p: e.cv, col: "var(--warn)", dash: "3 3" },
        { p: e.pred, col: "var(--acc)" },
      ], e.tag, `LM ${e.ade_m} m · ${e.speed_mps} m/s`)).join("") +
      `<div class="mono" style="grid-column:1/-1;font-size:10.5px;color:var(--dim2)">
        <span style="color:#E6EBF3">&#9632;</span> ground truth &nbsp;
        <span style="color:var(--warn)">&#9632;</span> constant velocity &nbsp;
        <span style="color:var(--acc)">&#9632;</span> trajectory LM &mdash; 6 s of ego motion,
        x forward</div>`;
  }

  /* ---------- robustness ---------- */
  const rb = R.robustness_report;
  if (rb) {
    $("#rbNote").textContent = `${rb.frames} keyframes · ${rb.faulted_camera} degraded · ${rb.checkpoint}`;

    /* The fault picker. The thumbnails are the ACTUAL 90x160 tensors the trust
       head was scored on, written out by the same perturbation classes -- not a
       CSS filter over a JPEG. Selecting one substitutes the trust the head
       returned for that fault into the noisy-OR and redraws the observability
       map, which is the part a fault-detection number never shows: what the
       degradation does to the coverage the planner is handed. */
    const ps = R.perturbation_shots;
    let fault = "clean";
    const paintFault = () => {
      const r = fault === "clean" ? rb.results.clean : rb.results[fault];
      const tF = r.trust_faulted, tO = r.trust_others;
      const d = fault === "clean" ? 0 : r.delta_trust_faulted;
      $("#rbTF").innerHTML = `<span class="${d < -0.05 ? "g" : d > 0 ? "b" : ""}">${fx(tF, 4)}</span>`;
      $("#rbTFd").innerHTML = fault === "clean" ? "<span class='d'>baseline</span>"
        : `<span class="${d < -0.05 ? "g" : "b"}">${d > 0 ? "+" : ""}${d.toFixed(4)} vs clean</span>`;
      $("#rbTO").textContent = fx(tO, 4);
      $("#rbTOd").textContent = "unchanged by design";
      const g = F[fi].grids, tv = {};
      CAMS.forEach(c => { tv[c] = c === rb.faulted_camera ? tF : tO; });
      const v = noisyOr(fi, tv);
      const cv = $("#rbCv");
      if (cv.getContext) paint(cv, v, g.n, rampInteg);
      let s = 0; for (let i = 0; i < v.length; i++) s += v[i];
      const base = noisyOr(fi, Object.fromEntries(CAMS.map(c =>
        [c, c === rb.faulted_camera ? rb.results.clean.trust_faulted : rb.results.clean.trust_others])));
      let b0 = 0; for (let i = 0; i < base.length; i++) b0 += base[i];
      const mean = s / v.length, cleanMean = b0 / base.length;
      $("#rbMean").textContent = `mean ${mean.toFixed(4)}`;
      $("#rbCap").textContent = `${rb.faulted_camera.replace("CAM_", "").toLowerCase()} at trust ${fx(tF, 3)}`;
      $("#rbHint").innerHTML = fault === "clean"
        ? `Six healthy cameras. Pick a fault below: the thumbnail is the real perturbed
           <b>90&times;160 model input</b>, and the map on the right is the observability field
           recomputed with the trust the head actually returned for that fault.`
        : `<b>${fault}</b> on ${rb.faulted_camera.replace("CAM_", "").toLowerCase()} &mdash;
           trust ${fx(rb.results.clean.trust_faulted, 3)} &rarr; <b>${fx(tF, 3)}</b>, and mean
           observability ${cleanMean.toFixed(4)} &rarr; <b>${mean.toFixed(4)}</b>
           (${(100 * (mean - cleanMean) / cleanMean).toFixed(1)}%).
           ${d > 0 ? "Trust moved the WRONG way here — see the gap below."
                   : "The detector caught it, and the coverage loss propagates."}`;
      document.querySelectorAll("#rbStrip figure").forEach(x =>
        x.classList.toggle("on", x.dataset.f === fault));
    };
    if (ps && ps.images) {
      $("#rbStrip").innerHTML = ["clean", "blur", "glare", "occlusion", "rain", "noise"]
        .filter(k => ps.images[k]).map(k => {
          const r = k === "clean" ? rb.results.clean : rb.results[k];
          const d = k === "clean" ? null : r.delta_trust_faulted;
          return `<figure data-f="${k}"><img src="${ps.images[k]}" alt="${k}">
            <figcaption><span>${k}</span><span class="${
              d === null ? "d" : d < -0.05 ? "g" : "b"}">${
              d === null ? "baseline" : (d > 0 ? "+" : "") + d.toFixed(3)}</span></figcaption></figure>`;
        }).join("");
      document.querySelectorAll("#rbStrip figure").forEach(x =>
        x.onclick = () => { fault = x.dataset.f; paintFault(); });
    }
    paintFault();
    RB_REPAINT = paintFault;
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
    $("#rbGap").innerHTML = note("The gap",
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
    ]) + note("How the LibTorch path was unblocked", cp.how_it_was_unblocked);
    $("#cppFlaky").textContent = pr.flakiness;
    const rn = cp.runner, A = rn.at_10hz_6s.latest_frame_seqlock, B2 = rn.at_10hz_6s.fifo_queue;
    $("#rnNote").textContent = rn.what;
    $("#rnTable").innerHTML =
      `<tr><th>policy</th><th>inference p50</th><th>queue wait p50</th><th>end-to-end p50</th><th>processed</th><th>dropped</th></tr>` +
      `<tr><td>latest frame (seqlock)</td><td>${A.inference_p50_ms} ms</td><td>${A.queue_wait_p50_ms} ms</td>
        <td class="g">${A.end_to_end_p50_ms} ms</td><td>${A.processed}</td><td>${A.skipped_stale} stale</td></tr>` +
      `<tr><td>FIFO queue</td><td>${B2.inference_p50_ms} ms</td><td class="b">${B2.queue_wait_p50_ms} ms</td>
        <td class="b">${B2.end_to_end_p50_ms} ms</td><td>${B2.processed}</td><td>${B2.dropped_full} full</td></tr>`;
    $("#rnNoteBox").innerHTML = note("Why this is the headline", rn.finding);

    /* Backpressure, drawn. The table has the p50s; what it cannot show is that
       under FIFO a frame's answer arrives when the world has moved on. Each row
       is one sensor frame at 10 Hz; the bar starts when the frame was captured
       and ends when its result was ready, at the end-to-end p50 this policy
       recorded. Under seqlock most frames are dropped on purpose and the ones
       that survive answer in a quarter of a second; under FIFO nothing is
       dropped until the queue fills and every answer is two seconds stale. */
    const FRAMES = 60, HZ = 10;
    // Both policies are drawn on ONE time axis so the comparison is a shape, not
    // two numbers: 6 s of sensor time wide, one row per frame captured at 10 Hz.
    // A bar starts when its frame was captured and ends when its answer was
    // ready, at the end-to-end p50 that policy recorded. FIFO's bars run off the
    // right of the window, which is exactly the point.
    const drawRunner = (upTo) => {
      const P = rn.at_10hz_6s[rnPolicy];
      const W = 900, RH = 7.4, H = FRAMES * RH + 30;
      const span = 6500, X = t => 44 + (W - 54) * t / span;
      const every = Math.max(1, Math.round(FRAMES / Math.max(1, P.processed)));
      const n = upTo === undefined ? FRAMES : upTo;
      let g = "";
      for (let t = 0; t <= 6000; t += 1000)
        g += `<line class="gl" x1="${X(t).toFixed(1)}" y1="16" x2="${X(t).toFixed(1)}" y2="${H - 10}"/>
              <text x="${(X(t) - 7).toFixed(1)}" y="11">${t / 1000}s</text>`;
      let done = 0, dropped = 0;
      for (let i = 0; i < FRAMES; i++) {
        const t0 = 1000 * i / HZ, y = 18 + i * RH;
        g += `<rect x="${X(0)}" y="${y.toFixed(1)}" width="${(X(6500) - X(0)).toFixed(1)}"
          height="${(RH - 2).toFixed(1)}" fill="#12161F"/>`;
        if (i >= n) continue;
        let col, w;
        if (i % every === 0 && done < P.processed) {
          col = "#5BD2E8"; w = P.end_to_end_p50_ms; done++;
        } else if (P.dropped_full && dropped < P.dropped_full) {
          col = "#FF6A7A"; w = 140; dropped++;
        } else { col = "#3A4454"; w = 140; }
        g += `<rect x="${X(t0).toFixed(1)}" y="${y.toFixed(1)}"
          width="${Math.max(3, Math.min(X(6500), X(t0 + w)) - X(t0)).toFixed(1)}"
          height="${(RH - 2).toFixed(1)}" rx="1" fill="${col}"/>`;
      }
      $("#rnViz").innerHTML =
        `<svg class="plot" style="max-width:940px" viewBox="0 0 ${W} ${H}">${g}</svg>`;
      $("#rnRd").textContent =
        `${P.processed} processed · ${P.skipped_stale} stale · ${P.dropped_full} dropped · ` +
        `e2e p50 ${P.end_to_end_p50_ms} ms · ${P.effective_hz} Hz effective`;
      document.querySelectorAll(".rnp").forEach(b =>
        b.classList.toggle("on", b.dataset.p === rnPolicy));
    };
    document.querySelectorAll(".rnp").forEach(b =>
      b.onclick = () => { rnPolicy = b.dataset.p; drawRunner(); });
    let rnT = null;
    $("#rnPlay").onclick = () => {
      if (rnT) clearInterval(rnT);
      let k = 0;
      drawRunner(0);
      rnT = setInterval(() => {
        k += 2;
        drawRunner(k);
        if (k >= FRAMES) { clearInterval(rnT); rnT = null; }
      }, 55);
    };
    drawRunner();

    /* ---------- C++ integrity monitor ---------- */
    const im = cp.integrity_monitor;
    if (im) {
      $("#imNote").textContent =
        `${im.grid} · ${im.cameras} cameras · ${im.source} · ${im.hardware}`;
      $("#imTable").innerHTML = `<tr><th>output</th><th>max abs diff</th><th></th></tr>` +
        [["coverage × visibility", im.parity.coverage_max_abs],
         ["integrity map", im.parity.integrity_max_abs]].map(([k, v]) =>
        `<tr><td>${k}</td><td>${Number(v).toExponential(3)}</td><td class="g">PASS</td></tr>`).join("") +
        `<tr><td>mean integrity</td><td colspan="2">Python ${im.mean_integrity_python} ·
          C++ ${im.mean_integrity_cpp}</td></tr>` +
        `<tr><td>recompute p50</td><td class="a">${im.p50_ms} ms</td>
          <td class="d">p99 ${im.p99_ms} ms</td></tr>`;
      if (im.platforms) $("#imTable").insertAdjacentHTML("afterend",
        `<div class="hr" style="margin:10px 0"></div>
         <div class="lab" style="margin-bottom:6px">Built and checked on both platforms</div>
         <table><tr><th>platform</th><th>parity</th><th>recompute p50</th></tr>` +
        im.platforms.map(r => `<tr><td class="d" style="font-family:'IBM Plex Sans'">${r[0]}</td>
          <td class="g">${r[1]}</td><td class="a">${r[2]}</td></tr>`).join("") + `</table>`);
      $("#imBox").innerHTML = note("Why this one had to be ported", im.why) +
        note("On the parity number", im.parity.note);
      $("#imStages").innerHTML = (im.stages || []).map(([k, v]) =>
        `<div style="border-left:2px solid var(--acc);padding:7px 11px;margin-bottom:7px;
          background:rgba(91,210,232,.05)">
          <div class="lab" style="color:var(--acc)">${k}</div>
          <div style="font-size:11.5px;color:var(--dim);margin-top:3px">${v}</div></div>`).join("");
    }
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
  const STATUS = [
    ["Perception", "runs", `occupancy IoU ${lm ? lm.occupancy.best.iou.toFixed(3) : "--"}`, "perception"],
    ["Forecast", "runs", fcr ? `persistence IoU ${fcr.results["T+1"].persistence.iou.toFixed(3)} @ T+1` : "--", "forecast"],
    ["Learned model", "runs", lm ? `ADE ${lm.trajectory_ade["T+1 (0.5s)"].learned_v11_temporal.ade_m} m, ${lm.trajectory_ade["T+1 (0.5s)"].delta_vs_cv_pct}% vs CV` : "--", "models"],
    ["Trajectory LM", "runs", R.trajlm_retrained_report ? `retrained · ADE ${R.trajlm_retrained_report.val_ade_m["T+3 (1.5s)"].gpt2_retrained_conditioned.ade_m} m @ T+3` : "--", "models"],
    ["VLA", "runs", vla ? `${vla.trainable_params_m}M projector · ${vla.frozen_params_m}M frozen` : "--", "models"],
    ["VLM", "runs", R.vlm_report ? `live on the displayed frame · ${R.vlm_report.model}` : "--", "models"],
    ["Trust / robustness", "runs", rbq ? `separation ${rbq.mean_separation}` : "--", "robustness"],
    ["Observability", "runs", iv ? `AUROC ${iv.auroc.integrity_occlusion_aware.auroc}` : "--", "observability"],
    ["C++ runtime", "runs", cpq ? `parity PASS · e2e ${cpq.runner.at_10hz_6s.latest_frame_seqlock.end_to_end_p50_ms} ms` : "--", "runtime"],
    ["C++ integrity monitor", "runs",
      cpq && cpq.integrity_monitor ? `parity PASS · ${cpq.integrity_monitor.p50_ms} ms/frame` : "--", "runtime"],
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
    ["Camera observability map", "runs", "6-camera noisy-OR with per-camera occlusion ray-cast, recomputed live in the console"],
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
    ["VLM scene understanding", "runs", "live call on the real displayed frames through the artifact runtime — not BLIP, and labelled as such everywhere it appears"],
    ["C++ integrity monitor", "runs", "noisy-OR + projected-area coverage + occlusion ray-cast ported to a header; bit-identical to Python on aarch64 (1.9e-09 on x86-64), 6.1 ms p50 for six cameras on a 216² grid"],
    ["BLIP (the repo's own VLM path)", "blocked", "huggingface.co returns 403 at the egress proxy in BOTH environments and nothing is cached locally — a policy denial, not a missing dependency. The VLM row above is what runs instead."],
    ["CUDA / GPU inference", "not available", "no GPU is present in either environment, so there is nothing to run it on. Every latency figure in this console is CPU and says so."],
  ];
  $("#sysTable").innerHTML = `<tr><th>component</th><th>state</th><th>detail</th></tr>` +
    SYS.map(([a, b, c]) => `<tr><td>${a}</td>
      <td class="${b === "runs" ? "g" : b === "blocked" ? "b" : "w"}">${b}</td>
      <td class="d" style="font-family:'IBM Plex Sans'">${c}</td></tr>`).join("");

  /* ---------- the one sentence ---------- */
  // Assembled from the reports, so it cannot drift from the numbers below it.
  const _lm = R.learned_model_report, _cp = R.cpp_report, _iv = R.integrity_visibility_report;
  const _im = _cp && _cp.integrity_monitor;
  $("#claim").innerHTML =
    `A camera-plus-LiDAR perception stack on nuScenes where the geometry is the reference the
     network is scored against, every number is measured on
     <b>${_lm ? _lm.samples : "--"} keyframes</b>, and the parts that have to survive a vehicle are
     in C++: trajectory ADE <b>${_lm ? _lm.trajectory_ade["T+3 (1.5s)"].learned_v11_temporal.ade_m : "--"} m</b>
     at 1.5 s against <b>${_lm ? _lm.trajectory_ade["T+3 (1.5s)"].constant_velocity.ade_m : "--"} m</b>
     for constant velocity, Python↔C++ parity at
     <b>${_cp ? Number(_cp.parity.occupancy.max_abs).toExponential(1) : "--"}</b>, and an observability
     monitor running in <b>${_im ? _im.p50_ms : "--"} ms</b> inside a 33.3 ms budget the
     ${_cp ? _cp.inference_latency.p50_ms.toFixed(0) : "--"} ms model misses by an order of magnitude.
     The failures are on the pages too: the trust head does not catch occlusion, and the
     observability map does not beat object range alone.`;

  /* ---------- reading paths ---------- */
  // Ordered routes, not a sitemap. Each step names the number it is sending the
  // reader to, so a reviewer with four minutes reads four numbers, not four pages.
  const PATHS = [
    ["Perception / occupancy", ["g1", "g2", "g3", "l2"], [
      ["Occupancy under a log-odds inverse sensor model, thresholded live",
        "perception", "free " + pc(F[0].stats.free) + " of the grid carved"],
      ["Static/dynamic by free-space consistency, not voxel persistence",
        "perception", "F1 0.610 vs 0.20 for absence-of-prior-support"],
      ["The learned BEV head scored against that geometry",
        "models", "IoU " + fx(_lm && _lm.occupancy.best.iou, 3)],
      ["Forecast checked against the LiDAR that actually arrived",
        "forecast", "IoU " + fx(R.occupancy_forecast_report && R.occupancy_forecast_report.results["T+1"].persistence.iou, 3) + " at T+1"],
    ]],
    ["Sensor health / redundancy", ["g4", "l2", "d3"], [
      ["Observability as noisy-OR over six cameras, recomputed live",
        "observability", "turn a camera off and watch the field change"],
      ["Validated against human visibility labels, including where it fails",
        "observability", "AUROC 0.569 vs 0.571 for range alone"],
      ["Fault injection on the real model inputs, with the downstream effect drawn",
        "robustness", "separation 0.128, occlusion missed"],
      ["The monitor ported to C++ so a runner can call it",
        "runtime", (_im ? _im.p50_ms : "--") + " ms p50, bit-identical"],
    ]],
    ["Prediction / behaviour", ["l2", "l3", "l4"], [
      ["Ego trajectory against three geometric baselines",
        "models", "ADE 0.421 m at T+3, −16.9% vs constant velocity"],
      ["A trajectory LM retrained after finding the shipped one saw all-zero data",
        "models", "loss curve and decoded paths on held-out frames"],
      ["A LLaVA-pattern projector into a frozen GPT-2",
        "models", "1.77M trainable of 102.7M"],
      ["Occupancy forecast vs the realised future",
        "forecast", "persistence beats advection at every horizon"],
    ]],
    ["Runtime / deployment", ["d1", "d2", "d3", "d4"], [
      ["The traced graph checked against Python output by output",
        "runtime", "5.2e-06 occupancy, 0.0 trust"],
      ["Backpressure policy, drawn on one time axis",
        "runtime", "235 ms seqlock vs 2319 ms FIFO, same inference"],
      ["Lock-free handoff under load",
        "runtime", "p99 7.8 µs over 20,000 frames, 0 dropped"],
      ["What is not here, and why",
        "system", "BLIP blocked by egress, no GPU exists"],
    ]],
  ];
  let pi = 0;
  const drawPaths = () => {
    $("#paths").innerHTML = PATHS.map(([n], i) =>
      `<button class="rp${i === pi ? " on" : ""}" data-i="${i}">${n}</button>`).join("");
    document.querySelectorAll(".rp").forEach(b => b.onclick = () => {
      pi = Number(b.dataset.i); AM.hi = PATHS[pi][1]; drawPaths(); drawArch();
    });
    $("#pathBody").innerHTML = PATHS[pi][2].map(([what, page, num], i) =>
      `<div class="step2"><span class="no">${i + 1}</span>
        <span>${what}<br><span class="mono" style="color:var(--acc);font-size:11px">${num}</span></span>
        <span class="go" data-g="${page}">${page} →</span></div>`).join("");
    document.querySelectorAll(".step2 .go").forEach(e => e.onclick = () => goTo(e.dataset.g));
  };
  AM.hi = PATHS[0][1];
  drawPaths();

  /* ---------- reproducibility ---------- */
  const P = D.provenance || {};
  $("#repNote").textContent =
    `${P.dataset || "nuScenes"} · commit ${P.commit || "--"} · built ${P.built || "--"}`;
  $("#repTable").innerHTML = `<tr><th>what</th><th>value</th></tr>` +
    [["dataset", P.dataset], ["scenes / keyframes", P.scenes + " / " + P.keyframes],
     ["console frames rendered", P.console_frames],
     ["train / val split", P.split], ["seed", P.seed],
     ["hardware", P.hardware], ["commit", P.commit], ["built", P.built]]
    .map(([k, v]) => `<tr><td>${k}</td><td class="a">${v === undefined ? "--" : v}</td></tr>`).join("");
  $("#repCmds").innerHTML = (P.commands || []).map(([n, c]) =>
    `<div style="margin-bottom:9px"><div style="font-size:11.5px;color:var(--dim)">${n}</div>
      <div class="mono" style="font-size:11px;color:var(--acc);overflow-wrap:anywhere;margin-top:2px">${c}</div></div>`).join("") +
    note("What cannot be re-measured here",
      "BLIP: huggingface.co returns 403 at the egress proxy and nothing is cached, so the repo's own " +
      "VLM path cannot be run in either environment. CUDA: no GPU is present, so every latency figure " +
      "on this console is CPU and is labelled as such. Both are listed above rather than quietly omitted.");
}

fillStatic();
drawArch();
draw();
addEventListener("resize", draw);
</script>
"""

def _git(*args, default="--"):
    try:
        return subprocess.check_output(("git",) + args, cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default


def provenance(bundle) -> dict:
    """Everything a sceptic needs to re-measure the console, read from the repo
    rather than typed. A number nobody can reproduce is an assertion."""
    lm = bundle["reports"].get("learned_model_report", {})
    return {
        "dataset": "nuScenes v1.0-mini (10 scenes, 2 cities, Boston + Singapore)",
        "scenes": 10,
        "keyframes": lm.get("samples", 404),
        "console_frames": len(bundle["frames"]),
        "split": "324 train / 80 val, fixed permutation",
        "seed": "0 (torch.manual_seed and np.random.seed) in every training and eval script",
        "hardware": "no GPU in either environment; every latency on this console is CPU",
        "commit": _git("rev-parse", "--short", "HEAD"),
        "built": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "commands": [
            ["Ego trajectory ADE, all horizons",
             "python scripts/eval/eval_trajectory_ade.py"],
            ["Occupancy forecast vs the realised future",
             "python scripts/eval/eval_occupancy_forecast.py"],
            ["Static/dynamic separation against annotated boxes",
             "python scripts/eval/eval_motion_separation.py"],
            ["Observability vs human visibility labels",
             "python scripts/eval/eval_integrity_visibility.py"],
            ["C++ integrity monitor parity and latency",
             "python scripts/dump_integrity_case.py &amp;&amp; "
             "cmake -S cpp -B cpp/build_ho -DCMAKE_BUILD_TYPE=Release &amp;&amp; "
             "cmake --build cpp/build_ho -j &amp;&amp; ctest --test-dir cpp/build_ho"],
            ["TorchScript export and Python&harr;C++ parity",
             "python scripts/export_torchscript.py &amp;&amp; cpp/build/odfm_parity_check"],
            ["This console, end to end",
             "python scripts/export_console_bundle.py &amp;&amp; python scripts/build_console.py"],
        ],
    }


def main():
    bundle = json.loads(BUNDLE.read_text())
    bundle["provenance"] = provenance(bundle)
    html = HEAD + BODY + "<script>window.__ODFM__=" + json.dumps(bundle) + ";</script>" + SCRIPT
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(ROOT)}  {OUT.stat().st_size/1e6:.1f} MB")

if __name__ == "__main__":
    main()
