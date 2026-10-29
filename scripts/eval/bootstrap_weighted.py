#!/usr/bin/env python3
"""Paired bootstrap over scenes for the A68 weighted-mIoU comparison.

A71 reported point estimates only and said plainly that the collapse of
FB-OCC's lead over FlashOcc M2 (+0.93 standard, -0.01 weighted) was not yet a
claim. This supplies the intervals.

Resampling is over the 75 TEST scenes, with replacement, and the SAME draw is
applied to every model in the same replicate. That is what makes the
difference-of-differences paired: the question is not "are these two numbers
far apart" but "does weighting change the gap between these two models by
more than scene-to-scene variation explains".

Reads the per-scene sufficient statistics that weighted_miou.py writes, so no
voxel is touched here.

    python3 scripts/eval/bootstrap_weighted.py \
        outputs/artifacts/weighted_miou_fbocc.json \
        outputs/artifacts/weighted_miou_flashocc_m2.json \
        outputs/artifacts/weighted_miou_flashocc_m1.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

FREE = 17


def miou(inter, union):
    """mIoU over the 17 non-free classes, classes with empty union skipped."""
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), np.nan)
    return float(np.nanmean(iou[:FREE]) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    models, scenes_ref = {}, None
    for f in args.files:
        d = json.load(open(f))
        name = os.path.basename(f).replace("weighted_miou_", "").replace(".json", "")
        ps = d["per_scene"]
        sc = sorted(ps)
        if scenes_ref is None:
            scenes_ref = sc
        elif sc != scenes_ref:
            raise SystemExit(f"{name}: scene set differs; cannot pair")
        models[name] = {k: np.array([ps[s][k] for s in sc], np.float64)
                        for k in ("inter", "union", "inter_w", "union_w")}
    S = len(scenes_ref)
    print(f"{len(models)} models, {S} scenes, {args.boot} replicates, "
          f"paired on identical scene draws\n")

    rs = np.random.default_rng(args.seed)
    draws = rs.integers(0, S, size=(args.boot, S))

    def dist(m, weighted):
        i = m["inter_w"] if weighted else m["inter"]
        u = m["union_w"] if weighted else m["union"]
        return np.array([miou(i[d].sum(0), u[d].sum(0)) for d in draws])

    point, boot = {}, {}
    for name, m in models.items():
        for tag, wt in (("std", False), ("wt", True)):
            i = m["inter_w"] if wt else m["inter"]
            u = m["union_w"] if wt else m["union"]
            point[(name, tag)] = miou(i.sum(0), u.sum(0))
            boot[(name, tag)] = dist(m, wt)

    def ci(x):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    print("per model")
    for name in models:
        for tag in ("std", "wt"):
            lo, hi = ci(boot[(name, tag)])
            print(f"  {name:<14} {tag:<4} {point[(name,tag)]:6.2f}  "
                  f"[{lo:6.2f}, {hi:6.2f}]")

    out = {"scenes": S, "boot": args.boot,
           "point": {f"{k[0]}|{k[1]}": v for k, v in point.items()},
           "ci": {f"{k[0]}|{k[1]}": list(ci(v)) for k, v in boot.items()},
           "pairs": {}}

    print("\npairwise, PAIRED on the same scene draws")
    print("  gap_std = A-B standard | gap_wt = A-B weighted | "
          "shift = gap_wt - gap_std")
    for a, b in itertools.combinations(models, 2):
        gs = boot[(a, "std")] - boot[(b, "std")]
        gw = boot[(a, "wt")] - boot[(b, "wt")]
        sh = gw - gs
        ps_ = point[(a, "std")] - point[(b, "std")]
        pw_ = point[(a, "wt")] - point[(b, "wt")]
        for label, pt, bs in (("gap_std", ps_, gs), ("gap_wt", pw_, gw),
                              ("shift", pw_ - ps_, sh)):
            lo, hi = ci(bs)
            excl = "excludes 0" if (lo > 0 or hi < 0) else "SPANS 0"
            print(f"  {a} vs {b:<14} {label:<8} {pt:+6.2f}  "
                  f"[{lo:+6.2f}, {hi:+6.2f}]  {excl}")
        out["pairs"][f"{a}|{b}"] = {
            "gap_std": [ps_, *ci(gs)], "gap_wt": [pw_, *ci(gw)],
            "shift": [pw_ - ps_, *ci(sh)]}
        print()

    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
