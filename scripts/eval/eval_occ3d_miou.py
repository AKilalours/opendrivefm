"""Reproduce FB-OCC's published Occ3D-nuScenes mIoU. The end-to-end check.

Why this comes before any observability result
----------------------------------------------
Everything downstream assumes the export is faithful, the grid orientation is
identity, the class indices line up with Occ3D's, and `mask_camera` is applied
the way the benchmark applies it. Each of those was established separately and
some only narrowly -- the orientation margin was 0.055 against a 0.05 rule.

A published number checks all of them at once. FB-OCC r50 reports mIoU in the
high 30s on this benchmark. If we land there, the pipeline is sound. If we land
at half that, something is wrong and no calibration result computed on top of
it means anything, however tidy it looks.

This is also the reason to run all eight transforms once more: orientation was
resolved against LiDAR, which is an indirect proxy. Occ3D semantics is the
direct answer, and the two should agree. If they disagree, the LiDAR probe was
measuring something else and A3 in the freeze document has to be reopened.

mIoU protocol follows the benchmark: intersection over union per class, over
voxels where mask_camera is set, averaged across the 17 non-free classes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

N_CLS = 18
FREE = 17
NAMES = ['others', 'barrier', 'bicycle', 'bus', 'car', 'constr_veh',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation', 'free']

TRANSFORMS = {
    "identity": lambda a: a,
    "rot90":    lambda a: np.rot90(a, 1, axes=(0, 1)),
    "rot180":   lambda a: np.rot90(a, 2, axes=(0, 1)),
    "rot270":   lambda a: np.rot90(a, 3, axes=(0, 1)),
    "transpose": lambda a: np.swapaxes(a, 0, 1),
    "flip_x":   lambda a: a[::-1],
    "flip_y":   lambda a: a[:, ::-1],
    "anti_transpose": lambda a: np.swapaxes(a[::-1, ::-1], 0, 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="data/preds/preds_voxel")
    ap.add_argument("--gts", default="data/occ3d/Occupancy3D-nuScenes-trainval/gts")
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--all-transforms", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print("indexing gts ...", flush=True)
    gt_map = {os.path.basename(os.path.dirname(p)): p
              for p in glob.glob(os.path.join(args.gts, "*", "*", "labels.npz"))}
    toks = sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(os.path.join(args.preds, "*.npz")))
    toks = [t for t in toks if t in gt_map]
    print(f"{len(gt_map):,} gt frames, {len(toks):,} matched to predictions")
    if args.frames:
        rs = np.random.default_rng(0)
        toks = [toks[i] for i in sorted(rs.choice(len(toks),
                min(args.frames, len(toks)), replace=False))]
    print(f"scoring {len(toks):,} frames\n", flush=True)

    names = list(TRANSFORMS) if args.all_transforms else ["identity"]
    inter = {k: np.zeros(N_CLS, np.int64) for k in names}
    union = {k: np.zeros(N_CLS, np.int64) for k in names}

    for i, t in enumerate(toks):
        g = np.load(gt_map[t])
        gt = g["semantics"]
        m = g["mask_camera"].astype(bool)
        cls = np.load(os.path.join(args.preds, t + ".npz"))["cls"]
        gtm = gt[m]
        for k in names:
            pm = np.ascontiguousarray(TRANSFORMS[k](cls))[m]
            for c in range(N_CLS):
                p, q = pm == c, gtm == c
                inter[k][c] += np.count_nonzero(p & q)
                union[k][c] += np.count_nonzero(p | q)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(toks)}", flush=True)

    res = {}
    for k in names:
        iou = np.where(union[k] > 0, inter[k] / np.maximum(union[k], 1), np.nan)
        res[k] = float(np.nanmean(iou[:FREE]) * 100)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16} mIoU {v:6.2f}")

    best = max(res, key=res.get)
    iou = np.where(union[best] > 0,
                   inter[best] / np.maximum(union[best], 1), np.nan)
    print(f"\nper-class IoU ({best})")
    for c in range(N_CLS):
        print(f"  {c:>2} {NAMES[c]:<13} {iou[c]*100:6.2f}")
    print(f"\nmIoU over 17 classes: {res[best]:.2f}")
    print("FB-OCC r50 publishes high-30s on this benchmark.")
    print("Within a couple of points -> the whole pipeline is validated.")
    if args.out:
        json.dump({"miou": res, "per_class": iou.tolist(),
                   "frames": len(toks), "best": best}, open(args.out, "w"),
                  indent=2)


if __name__ == "__main__":
    main()
