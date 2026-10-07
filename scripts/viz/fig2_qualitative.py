#!/usr/bin/env python3
"""Figure 2: one frame, four views, and the failure the paper is about.

A pedestrian that the camera rig cannot see is absent from the predicted
occupancy field, and the model is maximally confident about the free space
it puts there. Observability marks the same cell as unobserved.

Everything is real: the camera frame, the Occ3D ground truth, the FB-OCC
prediction and the observability map, all for one token on the sealed test
split. Nothing is drawn by hand.

  python3 scripts/viz/fig2_qualitative.py [--token TOK]
"""
from __future__ import annotations
import argparse, glob, importlib.util, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES, RNG, Z0, FREE = 0.4, 40.0, -1.0, 17
DEFAULT_TOKEN = "dc03fe4e0a374592af7701c66d5276c4"

C_GEOM, C_HOT, INK, MUTED, GRID = "#1268a8", "#b3261e", "#1a1a1a", "#6b6b6b", "#d9d9d6"


def _md():
    spec = importlib.util.spec_from_file_location(
        "md", os.path.join(ROOT, "scripts/eval/missed_detection.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DYNC = [2, 3, 4, 5, 6, 7, 9, 10]
OBJC = [1, 8]
STATC = [11, 12, 13, 14, 15, 16]


def tri_volume(cls):
    """Column priority: a dynamic voxel anywhere in a column wins.

    Taking the lowest non-free class instead renders the road surface over
    every column and the prediction panel comes out a uniform grey, which
    is how the first draft of this figure hid its own subject.
    """
    dyn = np.isin(cls, DYNC).any(2)
    obj = np.isin(cls, OBJC).any(2)
    sta = np.isin(cls, STATC).any(2)
    t = np.zeros(cls.shape[:2], np.int16)
    t[sta] = 1
    t[obj] = 2
    t[dyn] = 3
    return t


TRI_CMAP = ListedColormap(["#ffffff", "#e3e3df", "#b9b9b2", "#d1495b"])
TRI_NORM = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)


def draw_bev(ax, img, cmap, norm=None, vmin=None, vmax=None):
    ax.imshow(img, origin="lower", extent=[RNG, -RNG, -RNG, RNG],
              cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlim(19, -19)        # lateral, left of the vehicle on the left
    ax.set_ylim(-5, 33)         # forward
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID); s.set_linewidth(0.6)
    ax.plot(0, 0, marker="^", ms=5, color=INK, zorder=6)   # ego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=DEFAULT_TOKEN)
    a = ap.parse_args()
    md = _md()
    tok = a.token

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    row = next(r for r in index if r["token"] == tok)
    fi = index.index(row)
    gt_map = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}

    g = np.load(gt_map[tok]); gcls = g["semantics"].astype(np.int16)
    p = np.load(os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz"))
    pcls = p["cls"].astype(np.int16)
    pconf = p["conf"].astype(np.float32) / 255.0
    obs = np.load(os.path.join(ROOT, "data/pack/obs_max",
                               tok + ".npy")).astype(np.float32) / 255.0

    ann = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"),
                  allow_pickle=True)
    names = ann["category_names"]; fr = ann["frame"]
    order = np.argsort(fr, kind="stable")
    bounds = np.searchsorted(fr[order], np.arange(len(index) + 1))

    cam = row["cams"]["CAM_FRONT"]        # ego pose only; display cam chosen below
    Re = md.qrot(np.asarray(cam["ego2global_rotation"], np.float64))
    te = np.asarray(cam["ego2global_translation"], np.float64)

    # the object: the worst missed dynamic object in this frame
    best = None
    for k in order[bounds[fi]:bounds[fi + 1]]:
        cid = md.CAT2OCC.get(str(names[ann["category"][k]]))
        if cid is None or cid not in md.DYN:
            continue
        c = Re.T @ (ann["translation"][k].astype(np.float64) - te)
        R = Re.T @ md.qrot(np.asarray(ann["rotation"][k], np.float64))
        vx = md.box_voxels(c, ann["size"][k].astype(np.float64), R, 0.0)
        if vx is None:
            continue
        gm = gcls[vx] == cid
        if gm.sum() < 6:
            continue
        if np.isin(pcls[vx], list(md.DYN)).any():
            continue
        score = float((pcls[vx] == FREE).mean()) * gm.sum()
        if best is None or score > best["score"]:
            best = dict(score=score, c=c, cid=cid, vx=vx, R=R,
                        size=ann["size"][k].astype(np.float64),
                        obs=float(obs[vx][gm].max()),
                        conf=float(pconf[vx].max()),
                        free=float((pcls[vx] == FREE).mean()),
                        rng=float(np.hypot(c[0], c[1])),
                        nvox=int(gm.sum()),
                        pts=int(ann["num_lidar_pts"][k]))
    assert best is not None, "no missed dynamic object in this frame"
    cx, cy = best["c"][0], best["c"][1]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05))
    fig.subplots_adjust(wspace=0.07, left=0.005, right=0.995, top=0.86, bottom=0.10)

    # ---- (a) the camera that SHOULD see it ---------------------------------
    bearing = np.degrees(np.arctan2(cy, cx))
    CAM_BEARING = {"CAM_FRONT": 0.0, "CAM_FRONT_LEFT": 55.0,
                   "CAM_FRONT_RIGHT": -55.0, "CAM_BACK_LEFT": 110.0,
                   "CAM_BACK_RIGHT": -110.0, "CAM_BACK": 180.0}
    def _d(b):
        return abs((bearing - b + 180.0) % 360.0 - 180.0)
    use = min(CAM_BEARING, key=lambda c: _d(CAM_BEARING[c]))
    dcam = row["cams"][use]
    CAMTITLE = "(a)  " + use.replace("_", " ").title().replace("Cam ", "")
    fn = dcam["filename"]
    jp = fn if os.path.exists(fn) else os.path.join(
        ROOT, "dataset/nuscenes", fn.split("nuscenes/")[-1])
    im = plt.imread(jp)
    axes[0].imshow(im)

    # project the ground-truth box into that camera
    K = np.asarray(dcam["intrinsic"], np.float64)
    Rs = md.qrot(np.asarray(dcam["sensor2ego_rotation"], np.float64))
    ts = np.asarray(dcam["sensor2ego_translation"], np.float64)
    hw, hl, hh = (best["size"][0] / 2, best["size"][1] / 2, best["size"][2] / 2)
    corn = np.array([[sx * hl, sy * hw, sz * hh]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    W = (best["R"] @ corn.T).T + best["c"]          # ego frame
    Cm = (W - ts) @ Rs                              # ego -> camera
    if (Cm[:, 2] > 0.1).all():
        uv = (K @ Cm.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        u0, v0 = uv[:, 0].min(), uv[:, 1].min()
        u1, v1 = uv[:, 0].max(), uv[:, 1].max()
        axes[0].add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False,
                                    lw=1.1, ec=C_HOT, zorder=5))
        axes[0].annotate("", xy=((u0 + u1) / 2, v0 - 4),
                         xytext=((u0 + u1) / 2, max(v0 - 60, 8)),
                         arrowprops=dict(arrowstyle="->", lw=0.8, color=C_HOT))
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for s in axes[0].spines.values():
        s.set_color(GRID); s.set_linewidth(0.6)
    axes[0].set_title(CAMTITLE, fontsize=8, loc="left", pad=3)

    # ---- (b,c,d) the three grids -------------------------------------------
    gb, pb = tri_volume(gcls), tri_volume(pcls)
    ob = obs.max(axis=2)

    draw_bev(axes[1], gb, TRI_CMAP, TRI_NORM)
    axes[1].set_title("(b)  ground truth", fontsize=8, loc="left", pad=3)

    draw_bev(axes[2], pb, TRI_CMAP, TRI_NORM)
    axes[2].set_title("(c)  prediction", fontsize=8, loc="left", pad=3)

    draw_bev(axes[3], ob, "magma", vmin=0.0, vmax=1.0)
    axes[3].set_title("(d)  observability", fontsize=8, loc="left", pad=3)

    for ax in axes[1:]:
        ax.add_patch(Circle((cy, cx), 2.6, fill=False, lw=1.1,
                            ec=C_HOT, zorder=8))

    axes[2].annotate("free space,\nconfidence %.2f" % best["conf"],
                     xy=(cy + 2.8, cx + 1.4), xytext=(17.5, 31.0),
                     fontsize=6.4, color=C_HOT, ha="left", va="top",
                     arrowprops=dict(arrowstyle="->", lw=0.7, color=C_HOT,
                                     shrinkA=2, shrinkB=1))
    axes[3].annotate("obs = %.2f" % best["obs"],
                     xy=(cy + 2.8, cx + 1.4), xytext=(17.5, 31.0),
                     fontsize=6.4, color="#ffffff", ha="left", va="top",
                     arrowprops=dict(arrowstyle="->", lw=0.7, color="#ffffff",
                                     shrinkA=2, shrinkB=1))

    cname = {2: "bicycle", 3: "bus", 4: "car", 5: "construction vehicle",
             6: "motorcycle", 7: "pedestrian", 9: "trailer",
             10: "truck"}[best["cid"]]
    fig.text(0.005, -0.04,
             f"{cname} at {best['rng']:.1f} m  ·  {best['nvox']} ground-truth "
             f"voxels, {best['pts']} LiDAR returns  ·  "
             f"{100*best['free']:.0f}% of its box predicted free  ·  "
             f"scene {row['scene']}, token {tok[:12]}",
             fontsize=5.8, color=MUTED, ha="left", va="top")

    out = os.path.join(ROOT, "outputs", "figures")
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig2_qualitative.{ext}"),
                    bbox_inches="tight", dpi=400, facecolor="white")
    plt.close(fig)
    print(f"{cname} rng={best['rng']:.1f} obs={best['obs']:.3f} "
          f"conf={best['conf']:.3f} free={best['free']:.2f} "
          f"nvox={best['nvox']} pts={best['pts']}")
    print("wrote outputs/figures/fig2_qualitative.{pdf,png}")


if __name__ == "__main__":
    main()
