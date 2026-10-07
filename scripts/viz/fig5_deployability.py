#!/usr/bin/env python3
"""Figure 5: why the deployable occluder fails, measured rather than argued.

The first version of this figure asserted that a missed object erases its
own occluder, so the ray passes through and observability is falsely high.
That story is WRONG and the per-object numbers say so: on missed objects
observability is unchanged by the substitution (0.1987 -> 0.2017). What
actually moves is the other side. On DETECTED objects it collapses
(0.5402 -> 0.4586), because predicted occupancy is ~11x denser than the
ground-truth volume and therefore shadows regions the rig genuinely sees.
The separation between seen and unseen shrinks by a quarter and the
ranking degrades.

  python3 scripts/viz/fig5_deployability.py
"""
from __future__ import annotations
import glob, importlib.util, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOKEN = "dc03fe4e0a374592af7701c66d5276c4"
RES, RNG, FREE = 0.4, 40.0, 17
C_GEOM, C_CONF, C_HOT = "#1268a8", "#8d5c0e", "#b3261e"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d6"


def main():
    md_spec = importlib.util.spec_from_file_location(
        "md", os.path.join(ROOT, "scripts/eval/missed_detection.py"))
    md = importlib.util.module_from_spec(md_spec); md_spec.loader.exec_module(md)

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    row = next(r for r in index if r["token"] == TOKEN); fi = index.index(row)
    gtq = glob.glob(os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval",
                                 "gts", "*", TOKEN, "labels.npz"))[0]
    gcls = np.load(gtq)["semantics"].astype(np.int16)
    pcls = np.load(os.path.join(ROOT, "data/preds/preds_voxel",
                                TOKEN + ".npz"))["cls"].astype(np.int16)
    o_gt = np.load(os.path.join(ROOT, "data/pack/obs_max",
                                TOKEN + ".npy")).astype(np.float32) / 255.
    o_pr = np.load(os.path.join(ROOT, "data/pack/obs_pred_fig",
                                TOKEN + ".npy")).astype(np.float32) / 255.
    a55 = json.load(open(os.path.join(ROOT,
                         "outputs/artifacts/a55_pred_occluder.json")))

    ann = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"),
                  allow_pickle=True)
    names = ann["category_names"]; fr = ann["frame"]
    order = np.argsort(fr, kind="stable")
    bounds = np.searchsorted(fr[order], np.arange(len(index) + 1))
    cam0 = row["cams"]["CAM_FRONT"]
    Re = md.qrot(np.asarray(cam0["ego2global_rotation"], np.float64))
    te = np.asarray(cam0["ego2global_translation"], np.float64)
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
        if gm.sum() < 6 or np.isin(pcls[vx], list(md.DYN)).any():
            continue
        sc = float((pcls[vx] == FREE).mean()) * gm.sum()
        if best is None or sc > best[0]:
            best = (sc, c, float(o_gt[vx][gm].max()), float(o_pr[vx][gm].max()))
    _, cen, ogt_v, opr_v = best
    cx, cy = cen[0], cen[1]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.15),
                           gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    fig.subplots_adjust(wspace=0.30, left=0.004, right=0.996,
                        top=0.86, bottom=0.14)

    def bev(a, img, cmap, vmin, vmax):
        a.imshow(img, origin="lower", extent=[RNG, -RNG, -RNG, RNG], cmap=cmap,
                 vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
        a.set_xlim(24, -24); a.set_ylim(-8, 34)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(GRID); sp.set_linewidth(0.6)
        a.add_patch(Circle((cy, cx), 2.8, fill=False, lw=1.2, ec=C_HOT, zorder=8))

    bev(ax[0], o_gt.max(2), "magma", 0, 1)
    ax[0].set_title("(a)  obs from GT geometry", fontsize=7.8, loc="left", pad=3)
    ax[0].text(0.04, 0.05, f"obs = {ogt_v:.2f}", transform=ax[0].transAxes,
               fontsize=7.5, color="#ffffff", fontweight="bold", va="bottom")

    # the occluder set the vehicle actually has
    occ_pred = (pcls != FREE).any(2).astype(float)
    bev(ax[1], o_pr.max(2), "magma", 0, 1)
    ax[1].set_title("(b)  obs from predicted geometry", fontsize=7.8,
                    loc="left", pad=3)
    dens = (pcls != FREE).sum() / max((gcls != FREE).sum(), 1)
    ax[1].text(0.04, 0.05,
               f"obs = {opr_v:.2f}\n{dens:.0f}$\\times$ denser occluder set",
               transform=ax[1].transAxes, fontsize=6.8, color="#ffffff",
               fontweight="bold", va="bottom")

    # (c) the mechanism, measured: which side of the comparison moves
    rows = np.load(os.path.join(ROOT, "outputs/artifacts/a55_rows.npz"))
    miss = ~rows["det"]
    gm_, pm_ = rows["ogt"][miss].mean(), rows["opr"][miss].mean()
    gd_, pd_ = rows["ogt"][~miss].mean(), rows["opr"][~miss].mean()
    c2 = ax[2]
    for (a_, b_, col, lab) in ((gd_, pd_, C_GEOM, "detected"),
                               (gm_, pm_, C_HOT, "missed")):
        c2.plot([0, 1], [a_, b_], "-o", color=col, lw=1.6, ms=4.5,
                mfc=col, mec="#ffffff", mew=0.9, zorder=3)
        c2.text(1.08, b_ + (0.035 if lab == "missed" else -0.035),
                f"{lab}  {b_-a_:+.3f}", fontsize=6.4, color=col,
                va="center", ha="left")
    c2.set_xlim(-0.18, 1.72); c2.set_ylim(0.10, 0.62)
    c2.set_xticks([0, 1]); c2.set_xticklabels(["GT\ngeometry",
                                               "predicted\ngeometry"],
                                              fontsize=6.6)
    c2.set_ylabel("mean observability", fontsize=7.2, labelpad=1)
    c2.tick_params(axis="y", labelsize=6.6, length=2)
    c2.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); c2.set_axisbelow(True)
    for sp in ("top", "right"):
        c2.spines[sp].set_visible(False)
    c2.set_title("(c)  which side moves", fontsize=7.8, loc="left", pad=3)

    # (d) what it costs, measured
    A = a55["auroc"]
    labels = ["obs, GT\ngeometry", "MSP\n(no geometry)", "obs, predicted\ngeometry"]
    vals = [A["obs_gt (evaluation-time)"], A["MSP"], A["obs_pred (deployable)"]]
    cols = [C_GEOM, C_CONF, "#9fb9cc"]
    b = ax[3]
    y = np.arange(3)[::-1]
    b.barh(y, vals, height=0.52, color=cols, zorder=3)
    for yy, v in zip(y, vals):
        b.text(v + 0.004, yy, f"{v:.3f}", va="center", ha="left", fontsize=7.2)
    b.set_yticks(y); b.set_yticklabels(labels, fontsize=6.8)
    b.set_xlim(0.60, 0.80); b.set_xlabel("object-level AUROC", fontsize=7.6,
                                         labelpad=2)
    b.xaxis.grid(True, color=GRID, lw=0.5, zorder=0); b.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        b.spines[sp].set_visible(False)
    b.tick_params(axis="y", length=0)
    b.set_title("(d)  what the substitution costs", fontsize=7.8,
                loc="left", pad=3)
    d = a55["contrasts"]["obs_pred - MSP"]
    b.text(0.5, -0.30, f"obs$_{{pred}}$ $-$ MSP  {d[0]:+.4f} "
                       f"[{d[1]:+.4f}, {d[2]:+.4f}]",
           transform=b.transAxes, ha="center", va="top", fontsize=6.4,
           color=C_HOT)

    fig.text(0.004, -0.12,
             f"(a)-(c) scene {row['scene']}, token {TOKEN[:12]}, the frame of "
             f"Fig. 2.  (d) {a55['objects']:,} objects over the declared "
             f"{a55['sample']}-frame test subsample, seed {a55['seed']}.",
             fontsize=5.7, color=MUTED, ha="left", va="top")

    out = os.path.join(ROOT, "outputs", "figures")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig5_deployability.{ext}"),
                    bbox_inches="tight", dpi=400, facecolor="white")
    plt.close(fig)
    print(f"obs_gt={ogt_v:.3f}  obs_pred={opr_v:.3f}  (same object)")
    print(f"detected {gd_:.4f} -> {pd_:.4f} ({pd_-gd_:+.4f}) | "
          f"missed {gm_:.4f} -> {pm_:.4f} ({pm_-gm_:+.4f})")
    print("wrote outputs/figures/fig5_deployability.{pdf,png}")


if __name__ == "__main__":
    main()
