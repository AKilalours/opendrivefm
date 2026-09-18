#!/usr/bin/env python3
"""OpenDriveFM HD frame renderer.

Layout borrows the grid discipline of a sensor-data reel and the chase-camera
perspective of a production AV visualiser, but every panel shows something this
project actually measured, from cameras only:

  A  forward camera
  B  chase-camera view of the PREDICTED occupancy field, class coloured,
     brightness modulated by how much camera evidence each surface has
  C  observability, bird's eye
  D  obstacle triage + the verified-free corridor

There is no LiDAR here and no planner. The green corridor is not a planned
path -- it is the forward distance the cameras have POSITIVELY VERIFIED as
free: contiguous cells ahead of the ego that are predicted free AND carry
observability above a stated threshold. That is the honest camera-only analogue
of a planned path, and it is a number, not a decoration.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FREE, RES, RNG, NZ, Z0, N = 17, 0.4, 40.0, 16, -1.0, 200

NAMES = ["other", "barrier", "bicycle", "bus", "car", "constr veh", "motorcycle",
         "pedestrian", "traffic cone", "trailer", "truck", "driveable",
         "other flat", "sidewalk", "terrain", "manmade", "vegetation", "free"]
PAL = np.array([
 [150,150,150],[255,140, 70],[255,160,205],[255,235, 80],[ 60,175,255],
 [ 80,240,240],[255,150, 50],[255, 80, 80],[255,225,150],[190,105, 45],
 [200,110,255],[150,142,168],[128,122,146],[112, 98,138],[112,162,100],
 [200,205,225],[ 70,175, 85],[ 15, 16, 22]], np.float32) / 255.0

OBST = np.arange(0, 11)
MOVER = np.array([2, 3, 4, 5, 6, 7, 9, 10])
GROUND = np.array([11, 12, 13, 14])
STRUCT = np.array([15, 16])
BG = (0.035, 0.040, 0.052)


def qrot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], np.float64)


def bev(a):
    return a[::-1, ::-1]


def look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f], 0)          # rows: right, up, forward


def chase_render(ax, cls, conf, obs, top, has, eye=(-14., 0., 9.),
                 target=(26., 0., 0.), fov=58.0, W=1.0):
    """Painter's-algorithm perspective view of the predicted surface."""
    ii, jj = np.nonzero(has)
    k = top[ii, jj]
    cx = (ii + .5) * RES - RNG
    cy = (jj + .5) * RES - RNG
    cz = (k + .5) * RES + Z0
    c = cls[ii, jj, k] if cls.ndim == 3 else cls[ii, jj]
    o = obs[ii, jj, k]

    h = RES / 2.0
    # top face corners, in ego metres
    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    P = np.stack([cx, cy, cz + h], 1)
    M = look_at(np.asarray(eye, float), np.asarray(target, float))
    def proj(pts):
        v = (pts - np.asarray(eye, float)) @ M.T
        d = np.maximum(v[..., 2], 1e-3)
        s = 1.0 / np.tan(np.radians(fov) / 2.0)
        return np.stack([-v[..., 0] / d * s, v[..., 1] / d * s], -1), v[..., 2]

    quad = np.repeat(P[:, None, :], 4, 1)
    quad[:, :, 0] += corners[None, :, 0]
    quad[:, :, 1] += corners[None, :, 1]
    xy, depth = proj(quad)
    dep = depth.mean(1)
    keep = (dep > 0.5) & (dep < 90)
    xy, dep, c, o = xy[keep], dep[keep], c[keep], o[keep]
    order = np.argsort(-dep)
    xy, c, o, dep = xy[order], c[order], o[order], dep[order]

    col = PAL[c].copy()
    shade = 0.42 + 0.58 * np.clip(o, 0, 1)[:, None]      # evidence -> brightness
    fog = np.clip(1.0 - (dep[:, None] - 12) / 95.0, 0.35, 1.0)
    col = col * shade * fog + np.array(BG) * (1 - shade * fog)
    ax.add_collection(PolyCollection(xy, facecolors=col, edgecolors="none",
                                     antialiased=False))
    # ego car
    ex, ey, ez = 4.7 / 2, 2.0 / 2, 1.6
    box = np.array([[-ex, -ey, 0], [ex, -ey, 0], [ex, ey, 0], [-ex, ey, 0]])
    for dz, cc, al in ((0.05, "#0d1b22", 1.0), (ez, "#25e0ff", .95)):
        q = box.copy(); q[:, 2] += dz
        p, d = proj(q[None])
        ax.add_collection(PolyCollection(p, facecolors=cc, edgecolors="#25e0ff",
                                         linewidths=1.1, alpha=al, zorder=5))
    ax.set_xlim(-0.95, 0.95); ax.set_ylim(-0.60, 0.40)
    ax.set_aspect("auto"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)


def verified_free(cls, obs, tau=0.15, halfw=1.4, clear_from=0.4):
    """Longest contiguous forward run that is predicted free AND observed.

    A column is NOT "all levels free" -- the road surface itself is class 11,
    so requiring free everywhere returns 0 m for every frame. The question a
    planner asks is whether the DRIVING ENVELOPE is clear: nothing solid from
    `clear_from` metres above the ground up to the top of the grid.
    """
    j0 = int((RNG - halfw) / RES); j1 = int((RNG + halfw) / RES)
    i0 = int(RNG / RES)
    k0 = max(int((clear_from - Z0) / RES), 0)
    free = (cls[:, :, k0:] == FREE).all(-1)
    seen = (obs.max(-1) >= tau)
    ok = (free & seen)[:, j0:j1].mean(1) >= 0.8
    # The cells immediately in front of the bumper are ALWAYS unverifiable: the
    # cameras sit 1.5 m up and cannot see the ground at their own feet. Scanning
    # from i0 therefore returns 0 m on every frame, which is true but useless.
    # Report both numbers instead: where verification starts, and how far it runs.
    # LONGEST contiguous verified run, not the first one. Observability
    # fluctuates cell to cell, so "first run" latches onto a two-cell patch and
    # the number jumps around between frames without meaning anything.
    best_a = best_n = 0
    cur_a = None
    for i in range(i0, N):
        if ok[i]:
            if cur_a is None:
                cur_a = i
            if i - cur_a + 1 > best_n:
                best_a, best_n = cur_a, i - cur_a + 1
        else:
            cur_a = None
    if best_n == 0:
        return 0.0, 0.0, (j0, j1)
    return (best_a - i0) * RES, best_n * RES, (j0, j1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene-0916")
    ap.add_argument("--obs", default="data/pack/obs_max")
    ap.add_argument("--out", default="outputs/viz_hd")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--lo", type=float, default=0.35)
    ap.add_argument("--conf", type=float, default=0.70)
    ap.add_argument("--dpi", type=int, default=160)
    a = ap.parse_args()

    idx = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    rows = sorted([r for r in idx if r["scene"] == a.scene],
                  key=lambda r: r["timestamp"])
    sub = rows[a.start:a.start + a.limit] if a.limit else rows
    outdir = os.path.join(ROOT, a.out, a.scene); os.makedirs(outdir, exist_ok=True)

    for k, r in enumerate(sub):
        n = a.start + k
        tok = r["token"]
        pz = np.load(os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz"))
        cls = pz["cls"].astype(np.int16)
        conf = pz["conf"].astype(np.float32) / 255.0
        obs = np.load(os.path.join(ROOT, a.obs, tok + ".npy")).astype(np.float32) / 255.0

        occ = cls != FREE
        top = np.where(occ.any(-1), 15 - np.argmax(occ[:, :, ::-1], -1), 0)
        has = occ.any(-1)
        take = lambda A: np.take_along_axis(A, top[..., None], -1)[..., 0]
        s_obs = np.where(has, take(obs), np.nan)

        is_ob = np.isin(cls, OBST)
        zi = np.argmax(np.where(is_ob, conf, -1.0), -1)
        ob = is_ob.any(-1)
        pick = lambda A: np.take_along_axis(A, zi[..., None], -1)[..., 0]
        o_cls, o_conf, o_obs = pick(cls), pick(conf), pick(obs)

        # ego speed from consecutive global poses
        spd = float("nan")
        if n > 0:
            p0 = np.asarray(rows[n-1]["cams"]["CAM_FRONT"]["ego2global_translation"])
            p1 = np.asarray(r["cams"]["CAM_FRONT"]["ego2global_translation"])
            dt = (r["timestamp"] - rows[n-1]["timestamp"]) / 1e6
            spd = float(np.linalg.norm(p1 - p0) / max(dt, 1e-3) * 3.6)
        vstart, vfree, (jl, jr) = verified_free(cls, obs)
        t_rel = (r["timestamp"] - rows[0]["timestamp"]) / 1e6

        fig = plt.figure(figsize=(16, 9), facecolor=BG)
        gs = fig.add_gridspec(2, 2, left=.018, right=.982, top=.845, bottom=.105,
                              wspace=.065, hspace=.26,
                              width_ratios=[1.02, 1.0], height_ratios=[1.0, 1.0])

        def head(ax, t, s):
            ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(BG)
            ax.set_title(t, color="#dfe6ef", fontsize=12.5, loc="left", pad=21,
                         fontweight="600")
            ax.text(0, 1.012, s, transform=ax.transAxes, color="#6f7d8c",
                    fontsize=8.6, va="bottom")
            for sp in ax.spines.values():
                sp.set_color("#1d2530"); sp.set_linewidth(1.0)

        # A  camera
        axa = fig.add_subplot(gs[0, 0])
        im = Image.open(os.path.join(ROOT, "data/nuscenes",
                                     r["cams"]["CAM_FRONT"]["filename"]))
        axa.imshow(np.asarray(im.resize((1120, 630))))
        head(axa, "RGB CAMERA  ·  front", "one of six; no LiDAR, no radar, no HD map")

        # B  chase view
        axb = fig.add_subplot(gs[0, 1])
        chase_render(axb, cls, conf, obs, top, has)
        head(axb, "PREDICTED OCCUPANCY  ·  chase view",
             "class colour; brightness = camera evidence for that surface")

        # C  observability
        axc = fig.add_subplot(gs[1, 0])
        m = axc.imshow(bev(s_obs), cmap="magma", vmin=0, vmax=1,
                       interpolation="nearest")
        for rr in (10, 20, 30, 40):
            axc.add_patch(plt.Circle((N/2, N/2), rr/RES, fc="none",
                                     ec="#ffffff12", lw=.7))
            axc.text(N/2 + rr/RES*.707, N/2 - rr/RES*.707, f"{rr}m",
                     color="#49525f", fontsize=6.5)
        axc.add_patch(Rectangle((N/2-2.3/RES/2, N/2-4.8/RES/2), 2.3/RES, 4.8/RES,
                                fc="#25e0ff33", ec="#25e0ff", lw=1.1))
        head(axc, "CAMERA OBSERVABILITY  ·  bird's eye",
             "max single camera, occlusion-aware  ·  0 = no camera sees it")
        cb = fig.colorbar(m, ax=axc, fraction=.032, pad=.008)
        cb.ax.tick_params(colors="#6f7d8c", labelsize=7.5)
        cb.outline.set_edgecolor("#1d2530")

        # D  triage + verified corridor
        axd = fig.add_subplot(gs[1, 1])
        img = np.tile(np.array(BG, np.float32), (N, N, 1))
        img[np.isin(cls, GROUND).any(-1)] = (0.085, 0.095, 0.115)
        img[np.isin(cls, STRUCT).any(-1)] = (0.16, 0.17, 0.20)
        t_blind = ob & (o_obs == 0)
        t_dim = ob & (o_obs > 0) & (o_obs <= a.lo)
        t_risk = t_dim & (o_conf >= a.conf)
        t_ok = ob & (o_obs > a.lo)
        img[t_ok] = (0.16, 0.86, 0.47); img[t_blind] = (0.36, 0.41, 0.56)
        img[t_dim] = (0.97, 0.72, 0.10); img[t_risk] = (1.0, 0.16, 0.16)
        corr = np.zeros((N, N), bool)
        near = np.zeros((N, N), bool)
        i0 = int(RNG / RES)
        near[i0:i0 + max(int(vstart / RES), 0), jl:jr] = True
        if vfree > 0:
            s_ = i0 + int(vstart / RES)
            corr[s_:s_ + int(vfree / RES), jl:jr] = True
        base = bev(img).copy()
        cm = bev(corr)
        nm_ = bev(near)
        base[nm_] = base[nm_] * 0.55 + np.array([0.95, 0.65, 0.12]) * 0.45
        base[cm] = base[cm] * 0.40 + np.array([0.15, 0.95, 0.45]) * 0.60
        axd.imshow(base, interpolation="nearest")
        axd.add_patch(Rectangle((N/2-2.3/RES/2, N/2-4.8/RES/2), 2.3/RES, 4.8/RES,
                                fc="#25e0ff33", ec="#25e0ff", lw=1.1))
        head(axd, "OBSTACLE TRIAGE  ·  verified-free corridor",
             "green = free AND observed  ·  amber = near field no camera can verify")
        for c_, t_ in (((0.16, 0.86, 0.47), f"well observed  {int(t_ok.sum())}"),
                       ((0.97, 0.72, 0.10), f"barely observed  {int(t_dim.sum()-t_risk.sum())}"),
                       ((1.0, 0.16, 0.16), f"asserted yet barely observed  {int(t_risk.sum())}"),
                       ((0.36, 0.41, 0.56), f"no camera coverage  {int(t_blind.sum())}")):
            axd.plot([], [], "s", color=c_, ms=6.5, label=t_)
        axd.legend(loc="lower center", bbox_to_anchor=(.5, -.145), ncol=2,
                   frameon=False, fontsize=7.6, labelcolor="#93a0b0",
                   handletextpad=.4, columnspacing=1.2)

        fig.text(.018, .951, "OpenDriveFM", color="#ffffff", fontsize=21,
                 fontweight="bold")
        fig.text(.018, .925, "camera-only 3D occupancy, scored by what the "
                 "cameras could actually see", color="#5fae95", fontsize=10.5)
        rt = (f"{t_rel:06.2f} s   |   {spd:05.1f} km/h   |   {a.scene}   "
              f"|   frame {n+1:02d}/{len(rows)}")
        fig.text(.982, .953, rt, color="#7d8b9c", fontsize=10.5, ha="right",
                 family="monospace")
        fig.text(.018, .030,
                 f"verified-free corridor {vstart:4.1f}-{vstart+vfree:5.1f} m ahead      "
                 f"near-field unverifiable {vstart:4.1f} m      "
                 f"obstacle columns {int(ob.sum()):5d}      "
                 f"no camera coverage {100*t_blind.sum()/max(ob.sum(),1):4.1f}%      "
                 f"asserted-yet-barely-observed {int(t_risk.sum()):4d}",
                 color="#93a0b0", fontsize=9.2, family="monospace")
        fig.text(.982, .008, "6 cameras · 200x200x16 voxels @ 0.4 m · FB-OCC r50",
                 color="#4e5766", fontsize=8.6, ha="right", family="monospace")

        p = os.path.join(outdir, f"f_{n:03d}.png")
        fig.savefig(p, dpi=a.dpi, facecolor=BG); plt.close(fig)
        print(f"{n+1:3d}/{len(rows)} {tok} corridor {vstart:4.1f}-{vstart+vfree:5.1f}m "
              f"risk={int(t_risk.sum()):5d}",
              flush=True)


if __name__ == "__main__":
    main()
