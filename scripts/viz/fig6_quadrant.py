#!/usr/bin/env python3
"""Figure 6: confidence and observability are different axes, with real counts.

The reviewer asked for a 2x2 conceptual taxonomy. A drawn one would be a
claim; this is the same layout populated with measured objects, so each
cell carries its own n and its own dynamic-miss rate. The populated version does NOT support the taxonomy one would draw: the
worst cell is low-confidence/low-observability (39.8%), not high-confidence
/low-observability (23.6%). What it does support is the paper's actual
claim. Hold confidence HIGH and vary observability: 23.6% against 3.7%, a
6.4x difference at matched confidence. The axes are not redundant.

Right panel: the per-class boundary, including the classes where
observability loses.

  python3 scripts/viz/fig6_quadrant.py
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C_GEOM, C_CONF, C_HOT = "#1268a8", "#8d5c0e", "#b3261e"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d6"


def main():
    r = np.load(os.path.join(ROOT, "outputs/artifacts/a55_rows.npz"))
    obs, msp, det = r["ogt"], r["msp"], r["det"]
    miss = ~det
    # split at the medians of this population; stated in the caption
    o_t, c_t = float(np.median(obs)), float(np.median(msp))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.85),
                                 gridspec_kw={"width_ratios": [1.05, 1]})
    fig.subplots_adjust(wspace=0.42, left=0.085, right=0.985,
                        top=0.875, bottom=0.255)

    cells = [((msp >= c_t) & (obs < o_t),  0, 1, "BLIND SPOT",
              "sure, no evidence", C_HOT),
             ((msp >= c_t) & (obs >= o_t), 1, 1, "trusted", "sure, well seen",
              C_GEOM),
             ((msp < c_t) & (obs < o_t),   0, 0, "unknown space",
              "unsure, no evidence", MUTED),
             ((msp < c_t) & (obs >= o_t),  1, 0, "model error",
              "unsure, well seen", MUTED)]
    worst = max(100 * miss[m].mean() for m, *_ in cells)
    for m, cxp, cyp, name, sub, col in cells:
        n = int(m.sum()); mr = 100 * miss[m].mean()
        shade = 0.06 + 0.52 * (mr / worst)
        ax.add_patch(plt.Rectangle((cxp, cyp), 1, 1, facecolor=C_HOT,
                                   alpha=shade, edgecolor="#ffffff", lw=2))
        ax.text(cxp + .5, cyp + .74, name, ha="center", va="center",
                fontsize=8.4, color=col,
                fontweight="bold" if col == C_HOT else "normal")
        ax.text(cxp + .5, cyp + .56, sub, ha="center", va="center",
                fontsize=6.2, color=MUTED, style="italic")
        ax.text(cxp + .5, cyp + .34, f"{mr:.1f}%", ha="center", va="center",
                fontsize=12.5, color=INK, fontweight="bold")
        ax.text(cxp + .5, cyp + .17, f"missed   n = {n:,}", ha="center",
                va="center", fontsize=6.2, color=MUTED)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["low", "high"], fontsize=7.5)
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["low", "high"], fontsize=7.5)
    ax.set_xlabel("observability  $O_{\\mathrm{GT}}$   \"was there evidence?\"",
                  fontsize=7.8, labelpad=3)
    ax.set_ylabel("confidence  (MSP)\n\"is my prediction reliable?\"",
                  fontsize=7.8, labelpad=3)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    hi_lo = 100 * miss[(msp >= c_t) & (obs < o_t)].mean()
    hi_hi = 100 * miss[(msp >= c_t) & (obs >= o_t)].mean()
    ax.annotate("", xy=(0.52, 1.035), xytext=(0.98, 1.035),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=INK),
                annotation_clip=False, zorder=7)
    ax.text(0.5, 1.045, f"at matched high confidence, "
            f"{hi_lo/hi_hi:.1f}$\\times$ the miss rate",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.8,
            color=INK, fontweight="bold")
    ax.set_title("(a)  The two axes are not the same question",
                 fontsize=8.2, loc="left", pad=20)

    # ---- (b) per class, where it wins and where it loses ---------------
    d = json.load(open(os.path.join(ROOT,
                  "outputs/artifacts/baselines_by_class.json")))["classes"]
    d = sorted(d, key=lambda c: c["obs_minus_msp"][0])
    nm = {"constr_veh": "constr. vehicle"}
    y = np.arange(len(d))
    vals = [c["obs_minus_msp"][0] for c in d]
    lo = [c["obs_minus_msp"][1] for c in d]
    hi = [c["obs_minus_msp"][2] for c in d]
    cols = [C_GEOM if v > 0 else C_CONF for v in vals]
    bx.barh(y, vals, height=0.6, color=cols, zorder=3)
    bx.hlines(y, lo, hi, color="#55555533", lw=3.2, zorder=4)
    bx.axvline(0, color=INK, lw=0.8, zorder=5)
    for yy, c, v in zip(y, d, vals):
        off = 0.012 if v > 0 else -0.012
        bx.text(v + off, yy, f"{v:+.3f}", va="center",
                ha="left" if v > 0 else "right", fontsize=6.6, color=INK)
    bx.set_yticks(y)
    bx.set_yticklabels([f"{nm.get(c['cls'], c['cls'])}  ({c['n']:,})"
                        for c in d], fontsize=7)
    bx.set_xlim(-0.345, 0.285)
    bx.set_xlabel("AUROC:  observability $-$ confidence", fontsize=7.8,
                  labelpad=24)
    bx.xaxis.grid(True, color=GRID, lw=0.5, zorder=0); bx.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        bx.spines[sp].set_visible(False)
    bx.tick_params(axis="y", length=0)
    # the sort puts the most negative at the bottom, so the labels go
    # bottom-left and top-right. The first draft had them swapped.
    bx.annotate("geometry wins", xy=(0.5, -0.165), xycoords="axes fraction",
                xytext=(14, 0), textcoords="offset points", ha="left",
                va="center", fontsize=6.8, color=C_GEOM, fontweight="bold",
                annotation_clip=False)
    bx.annotate("confidence wins", xy=(0.5, -0.165), xycoords="axes fraction",
                xytext=(-14, 0), textcoords="offset points", ha="right",
                va="center", fontsize=6.8, color=C_CONF, fontweight="bold",
                annotation_clip=False)
    bx.set_title("(b)  and neither wins everywhere", fontsize=8.2,
                 loc="left", pad=20)

    fig.text(0.085, -0.03,
             f"(a) {len(obs):,} test objects split at the population medians, "
             f"$O$ {o_t:.2f} and MSP {c_t:.2f} \u2014 more than half of all "
             f"objects carry the maximum softmax value, which is itself the "
             f"problem; shading tracks the miss rate.  (b) bars are the "
             f"paired scene-bootstrap mean, rules the 95% interval.",
             fontsize=5.7, color=MUTED, ha="left", va="top")

    out = os.path.join(ROOT, "outputs", "figures")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig6_quadrant.{ext}"),
                    bbox_inches="tight", dpi=400, facecolor="white")
    plt.close(fig)
    for m, _, _, name, _, _ in cells:
        print(f"  {name:<14} n={int(m.sum()):>6,}  missed {100*miss[m].mean():5.1f}%")
    print("wrote outputs/figures/fig6_quadrant.{pdf,png}")


if __name__ == "__main__":
    main()
