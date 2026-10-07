#!/usr/bin/env python3
"""Figure 3: why not mask_camera, argued on the images.

The MAD Fig. 3 treatment. Four candidate signals for "can the rig see this
volume", each rendered from real data for one test-split frame, each
annotated on the image with what disqualifies it. Numbers in the panels are
read from outputs/artifacts/ so they cannot drift from the tables.

  python3 scripts/viz/fig3_whynotmask.py
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOKEN = "dc03fe4e0a374592af7701c66d5276c4"
RES, RNG = 0.4, 40.0
BAD, GOOD, INK, MUTED, GRID = "#b3261e", "#1a7f37", "#1a1a1a", "#6b6b6b", "#d9d9d6"
SYM = {"family": "DejaVu Sans"}


def panel_band(ax, title, items, tint="#ffffff", fg=None, h=0.44):
    """Title + verdict stack on a solid band, so text never fights the image.

    The first draft drew these straight onto the map and half of them were
    unreadable. MAD puts its annotations on a flat tinted field for exactly
    this reason.
    """
    fg = fg or INK
    ax.add_patch(plt.Rectangle((0, 1.0 - h), 1.0, h, transform=ax.transAxes,
                               facecolor=tint, edgecolor="none",
                               alpha=0.95, zorder=10))
    ax.text(0.028, 0.975, title, transform=ax.transAxes, fontsize=8.2,
            va="top", ha="left", color=fg, fontweight="bold", zorder=11)
    y0, dy = 0.975 - 0.085, 0.068
    for i, (ok, txt) in enumerate(items):
        y = y0 - i * dy
        if ok is not None:
            ax.text(0.030, y, "\u2713" if ok else "\u2717",
                    transform=ax.transAxes, fontsize=8.6, va="top", ha="left",
                    color=GOOD if ok else BAD, fontweight="bold", zorder=11,
                    **SYM)
        ax.text(0.088, y - 0.003, txt, transform=ax.transAxes, fontsize=6.9,
                va="top", ha="left", color=fg, zorder=11)


def main():
    q = glob.glob(os.path.join(
        ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
        "*", TOKEN, "labels.npz"))[0]
    g = np.load(q)
    maskcam = g["mask_camera"].astype(np.float32)
    obs = np.load(os.path.join(ROOT, "data/pack/obs_max",
                               TOKEN + ".npy")).astype(np.float32) / 255.0

    objres = json.load(open(os.path.join(ROOT, "outputs/artifacts/baselines_object.json")))
    A = objres["auroc"]
    h2 = json.load(open(os.path.join(ROOT, "outputs/artifacts/h2_max_test.json")))

    ii, jj = np.meshgrid(np.arange(200), np.arange(200), indexing="ij")
    X = (ii + .5) * RES - RNG
    Y = (jj + .5) * RES - RNG
    rangemap = np.hypot(X, Y)

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    row = next(r for r in index if r["token"] == TOKEN)
    fn = row["cams"]["CAM_FRONT_LEFT"]["filename"]
    jp = fn if os.path.exists(fn) else os.path.join(
        ROOT, "dataset/nuscenes", fn.split("nuscenes/")[-1])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 3.55))
    fig.subplots_adjust(wspace=0.015, hspace=0.03,
                        left=0.004, right=0.996, top=0.995, bottom=0.055)
    A1, A2, A3, A4 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    def bev(a, img, cmap, vmin, vmax, dark):
        a.imshow(img, origin="lower", extent=[RNG, -RNG, -RNG, RNG],
                 cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
                 aspect="auto")
        a.set_xlim(34, -34); a.set_ylim(-16, 40)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color(GRID); s.set_linewidth(0.6)
        a.plot(0, 0, marker="^", ms=5,
               color="#f5f5f5" if dark else INK, zorder=6)

    # (a) what the model actually gets
    A1.imshow(plt.imread(jp), aspect="auto")
    A1.set_xticks([]); A1.set_yticks([])
    for s in A1.spines.values():
        s.set_color(GRID); s.set_linewidth(0.6)
    panel_band(A1, "the input", [
        (None, "six surround cameras. no LiDAR, no map, no HD prior."),
        (None, "the pedestrian is behind the post, into direct sun."),
    ], tint="#ffffff")

    # (b) mask_camera
    bev(A2, maskcam.max(2), ListedColormap(["#2b2b2b", "#e9e4d6"]), 0, 1, True)
    panel_band(A2, "mask_camera   the dataset's own flag", [
        (False, "binary. a voxel is in or out, nothing between"),
        (False, "built offline from the full sensor suite, including"),
        (None,  "    the LiDAR the model never receives"),
        (False, "not available at inference"),
        (False, "AUROC 0.500 inside itself, by construction"),
    ], tint="#ffffff")

    # (c) range from ego
    bev(A3, rangemap, "magma_r", 0, 45, False)
    panel_band(A3, "range from ego", [
        (True, "continuous, and needs no network"),
        (False, "ignores occlusion entirely: a brick wall and"),
        (None,  "    open road at the same distance score alike"),
        (False, f"AUROC {A['range from ego']:.3f} per object"),
    ], tint="#ffffff")

    # (d) observability
    bev(A4, obs.max(2), "magma", 0, 1, True)
    a55 = json.load(open(os.path.join(ROOT,
                    "outputs/artifacts/a55_pred_occluder.json")))["auroc"]
    panel_band(A4, "observability   ours, evaluation-time", [
        (True, "continuous, and carries one parameter"),
        (True, "independent of the model and of its confidence"),
        (True, "resolves occlusion by first-hit ray marching"),
        (True, f"AUROC {A['observability']:.3f} per object, "
               f"{h2['diff']:+.4f} over mask_camera"),
        (False, "needs external scene geometry. Rebuilt from the"),
        (None,  f"    model's own prediction it falls to "
                f"{a55['obs_pred (deployable)']:.3f}, below MSP"),
    ], tint="#141414", fg="#f2f2f2", h=0.52)

    fig.text(0.004, 0.012,
             f"All four panels are the same frame, token {TOKEN[:12]}, "
             f"scene {row['scene']}, sealed test split.  AUROC values are "
             f"over {objres['objects']:,} objects in {objres['scenes']} scenes.",
             fontsize=5.8, color=MUTED, ha="left", va="bottom")

    out = os.path.join(ROOT, "outputs", "figures")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig3_whynotmask.{ext}"),
                    bbox_inches="tight", dpi=400, facecolor="white")
    plt.close(fig)
    print("wrote outputs/figures/fig3_whynotmask.{pdf,png}")


if __name__ == "__main__":
    main()
