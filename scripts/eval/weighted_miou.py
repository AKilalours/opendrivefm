#!/usr/bin/env python3
"""Standard and observability-weighted Occ3D mIoU. Registered in A68.

The question: how much of the mIoU separating camera-only occupancy models is
earned in voxels where the rig supplies little or no geometric evidence?

Standard mIoU follows the benchmark exactly as scripts/eval/eval_occ3d_miou.py
does: intersection over union per class, over voxels where mask_camera is set,
averaged across the 17 non-free classes. The weighted variant is the same
computation with each voxel contributing weight O(v) instead of 1:

    inter_w[c] = sum_v w(v) * [pred(v)==c and gt(v)==c]
    union_w[c] = sum_v w(v) * [pred(v)==c or  gt(v)==c]

w(v) = O(v) from data/pack/obs_max, uint8 / 255, no exponent, no threshold, no
rescaling. A68 fixes the weighting function in advance precisely so that a
family of weightings cannot be tried until one separates. If this one shows
nothing, that is the answer.

    python3 scripts/eval/weighted_miou.py --scenes test \
        --preds data/preds_flashocc_m2 \
        --out outputs/artifacts/weighted_miou_flashocc_m2.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

N_CLS = 18
FREE = 17
NAMES = ['others', 'barrier', 'bicycle', 'bus', 'car', 'constr_veh',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation', 'free']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gts",
                    default="data/occ3d/Occupancy3D-nuScenes-trainval/gts")
    ap.add_argument("--obs", default="data/pack/obs_max")
    ap.add_argument("--scenes", default="test", choices=["all", "dev", "test"])
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gt_map = {os.path.basename(os.path.dirname(p)): p
              for p in glob.glob(os.path.join(args.gts, "*", "*", "labels.npz"))}
    index = json.load(open(os.path.join(args.pack, "index.json")))

    keep = None
    if args.scenes != "all":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import split as _sp
        keep = _sp.dev_scenes() if args.scenes == "dev" else _sp.test_scenes()
        print(f"[A45] {args.scenes.upper()} scenes only, {len(keep)} of 150, "
              f"split digest {_sp.digest()}", flush=True)

    inter = np.zeros(N_CLS, np.float64)
    union = np.zeros(N_CLS, np.float64)
    inter_w = np.zeros(N_CLS, np.float64)
    union_w = np.zeros(N_CLS, np.float64)

    used, t0 = 0, time.time()
    for row in index:
        if keep is not None and row["scene"] not in keep:
            continue
        if args.frames and used >= args.frames:
            break
        tok = row["token"]
        pp = os.path.join(args.preds, f"{tok}.npz")
        op = os.path.join(args.obs, f"{tok}.npy")
        gp = gt_map.get(tok)
        if not (gp and os.path.exists(pp) and os.path.exists(op)):
            continue

        g = np.load(gp)
        gt = g["semantics"]
        mask = g["mask_camera"].astype(bool)
        cls = np.load(pp)["cls"]
        w = np.load(op).astype(np.float32) / 255.0

        gtm, pm, wm = gt[mask], cls[mask], w[mask]
        for c in range(N_CLS):
            p, q = pm == c, gtm == c
            both, either = p & q, p | q
            inter[c] += np.count_nonzero(both)
            union[c] += np.count_nonzero(either)
            inter_w[c] += float(wm[both].sum())
            union_w[c] += float(wm[either].sum())

        used += 1
        if used % 250 == 0:
            el = time.time() - t0
            print(f"  {used} frames  {el/used:.3f}s/frame", flush=True)

    def miou(i, u):
        iou = np.where(u > 0, i / np.maximum(u, 1e-12), np.nan)
        return float(np.nanmean(iou[:FREE]) * 100), (iou * 100).tolist()

    m_std, per_std = miou(inter, union)
    m_wt, per_wt = miou(inter_w, union_w)

    print(f"\nframes {used}")
    print(f"  standard mIoU   {m_std:6.2f}")
    print(f"  weighted mIoU   {m_wt:6.2f}")
    print(f"  delta           {m_wt - m_std:+6.2f}\n")
    for c in range(FREE):
        print(f"  {NAMES[c]:<14} std {per_std[c]:6.2f}   wt {per_wt[c]:6.2f}"
              f"   {per_wt[c]-per_std[c]:+6.2f}")

    res = dict(preds=args.preds, scenes=args.scenes, frames=used,
               miou_standard=m_std, miou_weighted=m_wt,
               delta=m_wt - m_std,
               per_class_standard=per_std, per_class_weighted=per_wt,
               weighting="O(v) from obs_max, uint8/255, no exponent (A68)")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
