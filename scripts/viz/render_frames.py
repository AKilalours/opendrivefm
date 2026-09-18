#!/usr/bin/env python3
"""OpenDriveFM per-frame visualiser.

One figure per keyframe, four panels:
  A  forward camera strip (FRONT_LEFT / FRONT / FRONT_RIGHT)
  B  predicted occupancy, bird's eye, obstacles over ground and structure
  C  camera observability of the same scene -- the quantity this project adds
  D  obstacle triage: which asserted obstacles the cameras could actually see

Everything drawn is read from the packed artefacts, nothing is recomputed by eye:
  data/preds/preds_voxel/<token>.npz   cls / conf, uint8 = round(255*p)
  data/pack/obs_ray/<token>.npy        observability, uint8 = round(255*o)
  data/occ3d/.../gts/<scene>/<token>/labels.npz   semantics + mask_camera

Grid convention (verified against the images, not assumed):
  index0 = x forward, index1 = y left, 0.4 m voxels over +-40 m, 16 z levels.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FREE, RES, RNG, N = 17, 0.4, 40.0, 200

NAMES = ["other","barrier","bicycle","bus","car","constr veh","motorcycle",
         "pedestrian","traffic cone","trailer","truck","driveable","other flat",
         "sidewalk","terrain","manmade","vegetation","free"]
PAL = np.array([
 [140,140,140],[255,120, 50],[255,150,200],[255,235, 60],[ 30,160,255],
 [ 60,235,235],[255,140, 30],[255, 60, 60],[255,225,140],[180, 90, 30],
 [190, 90,255],[255,  0,255],[139,137,137],[ 75,  0, 75],[150,240, 80],
 [230,230,250],[  0,175,  0],[ 18, 18, 24]], np.float32) / 255.0

OBST  = np.array([0,1,2,3,4,5,6,7,8,9,10])      # anything you would have to avoid
MOVER = np.array([2,3,4,5,6,7,9,10])            # of those, the ones that move
GROUND = np.array([11,12,13,14])
STRUCT = np.array([15,16])

C_BG, C_GROUND, C_STRUCT = (0.055,0.06,0.075), (0.16,0.175,0.20), (0.30,0.32,0.36)
C_OK, C_DIM, C_RISK, C_BLIND = (0.15,0.85,0.45), (0.95,0.70,0.10), (1.00,0.15,0.15), (0.35,0.40,0.55)


def bev(a):
    """grid[x,y] -> image rows=down, cols=right, with +x up and +y left."""
    return a[::-1, ::-1]


def frame_axes(ax, title, sub=None):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(C_BG)
    ax.set_title(title, color="#e8eaf0", fontsize=11.5, pad=21, loc="left")
    if sub:
        ax.text(0.0, 1.008, sub, transform=ax.transAxes, color="#7d8799", fontsize=8.2,
                va="bottom")
    for s in ax.spines.values():
        s.set_color("#2a2e36"); s.set_linewidth(0.9)


def overlay(ax):
    c = N / 2.0
    for r in (10, 20, 30, 40):
        ax.add_patch(plt.Circle((c, c), r / RES, fc="none", ec="#ffffff14", lw=0.7))
        ax.text(c + r / RES * .707, c - r / RES * .707, f"{r}m",
                color="#4b515e", fontsize=6.2, ha="left", va="bottom")
    ax.add_patch(Rectangle((c - 2.3/RES/2, c - 4.8/RES/2), 2.3/RES, 4.8/RES,
                           fc="#00e5ff22", ec="#00e5ff", lw=1.1))
    ax.plot([c], [c - 4.8/RES/2 - 3.5], marker="^", color="#00e5ff", ms=5.5)
    ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(N - 0.5, -0.5)


def columns(cls, conf, obs):
    """Collapse each (x,y) column to what a planner would care about."""
    occ = cls != FREE
    top = np.where(occ.any(-1), 15 - np.argmax(occ[:, :, ::-1], -1), 0)
    has_top = occ.any(-1)
    obs_top = np.take_along_axis(obs, top[..., None], -1)[..., 0]

    is_obst = np.isin(cls, OBST)
    scored = np.where(is_obst, conf, -1.0)
    zi = np.argmax(scored, -1)
    has_obst = is_obst.any(-1)
    pick = lambda a: np.take_along_axis(a, zi[..., None], -1)[..., 0]
    return dict(has_top=has_top, obs_top=np.where(has_top, obs_top, np.nan),
                has_obst=has_obst, o_cls=pick(cls), o_conf=pick(conf), o_obs=pick(obs),
                ground=np.isin(cls, GROUND).any(-1), struct=np.isin(cls, STRUCT).any(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene-0553")
    ap.add_argument("--out", default="outputs/viz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lo", type=float, default=0.35)
    ap.add_argument("--conf", type=float, default=0.70)
    a = ap.parse_args()

    idx = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    rows = sorted([r for r in idx if r["scene"] == a.scene], key=lambda r: r["timestamp"])
    if a.limit: rows = rows[:a.limit]
    outdir = os.path.join(ROOT, a.out, a.scene); os.makedirs(outdir, exist_ok=True)
    stats = []

    for k, r in enumerate(rows):
        tok = r["token"]
        pz = np.load(os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz"))
        cls = pz["cls"].astype(np.int16)
        conf = pz["conf"].astype(np.float32) / 255.0
        obs = np.load(os.path.join(ROOT, "data/pack/obs_ray", tok + ".npy")).astype(np.float32) / 255.0
        gt = np.load(os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                                 a.scene, tok, "labels.npz"))
        gcls, mcam = gt["semantics"].astype(np.int16), gt["mask_camera"].astype(bool)
        C = columns(cls, conf, obs)

        # -- honest accuracy: per voxel, on ground-truth-occupied voxels the
        #    benchmark itself says a camera should be able to see.
        ev = mcam & (gcls != FREE)
        corr = (cls == gcls)
        clear, dim, blind = ev & (obs > a.lo), ev & (obs > 0) & (obs <= a.lo), ev & (obs == 0)
        acc = lambda m: float(corr[m].mean()) if m.any() else float("nan")
        a_clear, a_dim, a_blind = acc(clear), acc(dim), acc(blind)

        # -- obstacle triage
        ob = C["has_obst"]
        o_obs, o_conf = C["o_obs"], C["o_conf"]
        t_blind = ob & (o_obs == 0)
        t_dim   = ob & (o_obs > 0) & (o_obs <= a.lo)
        t_risk  = t_dim & (o_conf >= a.conf)
        t_ok    = ob & (o_obs > a.lo)

        fig = plt.figure(figsize=(16, 9), facecolor="#0a0b0f")
        gs = fig.add_gridspec(2, 3, height_ratios=[.80, 1.32],
                              left=.022, right=.978, top=.845, bottom=.135,
                              wspace=.075, hspace=.235)

        # A ------------------------------------------------------------
        axc = fig.add_subplot(gs[0, :])
        strip = [np.asarray(Image.open(os.path.join(ROOT, "data/nuscenes",
                 r["cams"][c]["filename"])).resize((533, 300)))
                 for c in ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")]
        axc.imshow(np.concatenate(strip, 1))
        for i, t in enumerate(("FRONT LEFT", "FRONT", "FRONT RIGHT")):
            axc.text(i * 533 + 9, 20, t, color="#dfe4ec", fontsize=8,
                     bbox=dict(fc="#000000b0", ec="none", pad=1.8))
        frame_axes(axc, "A   what goes in",
                   "six surround cameras, no LiDAR and no radar at inference  (front three shown)")

        # B ------------------------------------------------------------
        axb = fig.add_subplot(gs[1, 0])
        img = np.tile(np.array(C_BG, np.float32), (N, N, 1))
        img[C["ground"]] = C_GROUND
        img[C["struct"]] = C_STRUCT
        img[ob] = PAL[C["o_cls"][ob]]
        axb.imshow(bev(img), interpolation="nearest"); overlay(axb)
        frame_axes(axb, "B   predicted occupancy",
                   "obstacles in class colour, over drivable ground and static structure")
        seen_cls = np.unique(C["o_cls"][ob])
        for ci in seen_cls[:7]:
            axb.plot([], [], "s", color=PAL[ci], ms=6, label=NAMES[ci])
        axb.legend(loc="lower center", bbox_to_anchor=(.5, -.105), ncol=4,
                   frameon=False, fontsize=7, labelcolor="#9aa3b2", handletextpad=.4,
                   columnspacing=1.1)

        # C ------------------------------------------------------------
        axo = fig.add_subplot(gs[1, 1])
        m = axo.imshow(bev(C["obs_top"]), cmap="magma", vmin=0, vmax=1,
                       interpolation="nearest")
        overlay(axo)
        frame_axes(axo, "C   camera observability",
                   "how much camera evidence each surface actually has  (0 = none)")
        cb = fig.colorbar(m, ax=axo, fraction=.036, pad=.012)
        cb.ax.tick_params(colors="#7d8799", labelsize=7); cb.outline.set_edgecolor("#2a2e36")

        # D ------------------------------------------------------------
        axr = fig.add_subplot(gs[1, 2])
        img2 = np.tile(np.array(C_BG, np.float32), (N, N, 1))
        img2[C["ground"]] = (0.105, 0.115, 0.135)
        img2[t_ok] = C_OK; img2[t_blind] = C_BLIND
        img2[t_dim] = C_DIM; img2[t_risk] = C_RISK
        axr.imshow(bev(img2), interpolation="nearest"); overlay(axr)
        frame_axes(axr, "D   obstacle triage",
                   "the same obstacles, recoloured by how well the cameras saw them")
        for c, t in ((C_OK,    f"well observed  ({int(t_ok.sum())})"),
                     (C_DIM,   f"barely observed  ({int(t_dim.sum()-t_risk.sum())})"),
                     (C_RISK,  f"asserted p≥{a.conf:.2f} yet barely observed  ({int(t_risk.sum())})"),
                     (C_BLIND, f"no camera coverage  ({int(t_blind.sum())})")):
            axr.plot([], [], "s", color=c, ms=6, label=t)
        axr.legend(loc="lower center", bbox_to_anchor=(.5, -.105), ncol=2,
                   frameon=False, fontsize=7, labelcolor="#9aa3b2", handletextpad=.4,
                   columnspacing=1.1)

        # header / footer ------------------------------------------------
        fig.text(.022, .945, "OpenDriveFM", color="#ffffff", fontsize=20, weight="bold")
        fig.text(.022, .912,
                 "camera-only 3D occupancy, scored by what the cameras could actually see",
                 color="#6f9f8c", fontsize=10)
        fig.text(.978, .947, f"{a.scene}    frame {k+1:02d} / {len(rows)}",
                 color="#5c6474", fontsize=9.5, ha="right")
        fig.text(.978, .918, tok, color="#3c424e", fontsize=7, ha="right", family="monospace")

        def pc(v): return "  n/a" if np.isnan(v) else f"{100*v:4.1f}%"
        fig.text(.022, .040,
                 "voxel class accuracy on ground-truth occupied voxels    "
                 f"well observed {pc(a_clear)}      barely observed {pc(a_dim)}      "
                 f"no camera coverage {pc(a_blind)}",
                 color="#9aa3b2", fontsize=9, family="monospace")
        fig.text(.022, .016,
                 f"obstacle columns {int(ob.sum()):5d}    "
                 f"barely observed {100*t_dim.sum()/max(ob.sum(),1):4.1f}%    "
                 f"uncovered {100*t_blind.sum()/max(ob.sum(),1):4.1f}%    "
                 f"asserted-yet-barely-observed {int(t_risk.sum()):4d}"
                 f"    evaluated voxels {int(ev.sum()):6d}",
                 color="#5c6474", fontsize=8.5, family="monospace")

        p = os.path.join(outdir, f"f_{k:03d}.png")
        fig.savefig(p, dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)
        stats.append(dict(i=k, token=tok, obst=int(ob.sum()), risk=int(t_risk.sum()),
                          blind=int(t_blind.sum()), dim=int(t_dim.sum()),
                          acc_clear=a_clear, acc_dim=a_dim, acc_blind=a_blind,
                          ev=int(ev.sum())))
        print(f"{k+1:3d}/{len(rows)} {tok} obst={int(ob.sum()):5d} risk={int(t_risk.sum()):4d} "
              f"acc {a_clear:.3f}/{a_dim:.3f}/{a_blind:.3f}", flush=True)

    json.dump(stats, open(os.path.join(outdir, "stats.json"), "w"), indent=1)
    print("wrote", outdir)


if __name__ == "__main__":
    main()
