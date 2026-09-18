"""Settle the Occ3D orientation against LiDAR, in 3D, on real per-voxel data.

The BEV version of this probe could not separate the eight dihedral transforms
(p_occ AUROC spread 0.0015, height spearman negative for all eight) -- but that
was a verdict on the column-collapsed export, not on the geometry. With a real
per-voxel volume there is a much stronger signal available: LiDAR says which
VOXELS contain a return, the model says which voxels are non-free, and the two
should agree only under the correct orientation.

Matthews correlation is the statistic, because the classes are wildly
imbalanced and accuracy would be dominated by empty space.

Scope: r < 20 m only. Two reasons, both fixed before running. Inside 20 m the
cameras are supervised (free 74% at 0-10 m, 64% at 10-20 m) and LiDAR is dense;
beyond it the model fills unsupervised interior with `manmade` and LiDAR
returns thin out, so agreement there measures neither geometry nor orientation.

DECISION RULE, from docs/METHOD_FREEZE.md and not revisable here:
the winner must beat the runner-up by at least 0.05. Below that, the
orientation is NOT resolved and nothing downstream may be produced.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OCC_N, OCC_RES, OCC_RNG = 200, 0.4, 40.0
OCC_Z0, OCC_ZRES, OCC_NZ = -1.0, 0.4, 16
FREE = 17
MARGIN_REQUIRED = 0.05

# Acting on the first two axes only; the vertical axis is already confirmed
# ascending by the height profile (ground solid at level 0, air at level 12).
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


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "odfm_eval", os.path.join(HERE, "eval_observability_scaled.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_eval_module()


def lidar_voxels(pts_ego):
    x, y, z = pts_ego[:, 0], pts_ego[:, 1], pts_ego[:, 2]
    k = ((np.abs(x) < OCC_RNG) & (np.abs(y) < OCC_RNG) &
         (z >= OCC_Z0) & (z < OCC_Z0 + OCC_NZ * OCC_ZRES))
    ix = ((x[k] + OCC_RNG) / OCC_RES).astype(np.int32)
    iy = ((y[k] + OCC_RNG) / OCC_RES).astype(np.int32)
    iz = ((z[k] - OCC_Z0) / OCC_ZRES).astype(np.int32)
    occ = np.zeros((OCC_N, OCC_N, OCC_NZ), bool)
    occ[ix, iy, iz] = True
    return occ


def mcc(pred, truth):
    tp = float(np.count_nonzero(pred & truth))
    tn = float(np.count_nonzero(~pred & ~truth))
    fp = float(np.count_nonzero(pred & ~truth))
    fn = float(np.count_nonzero(~pred & truth))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--preds", default="data/preds/preds_voxel")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--max-range", type=float, default=20.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    index = json.load(open(os.path.join(args.pack, "index.json")))
    rs = np.random.default_rng(0)
    rows = [index[i] for i in rs.choice(len(index), args.frames, replace=False)]

    ax = (np.arange(OCC_N) + 0.5) * OCC_RES - OCC_RNG
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    R = np.hypot(X, Y)
    near = (R < args.max_range) & (R > 2.0)          # exclude the ego box
    near3 = np.repeat(near[:, :, None], OCC_NZ, axis=2)

    acc = {k: [[0, 0, 0, 0]] for k in TRANSFORMS}    # tp tn fp fn
    truth_all, pred_all = [], {k: [] for k in TRANSFORMS}
    used = 0
    for row in rows:
        tok = row["token"]
        pp = os.path.join(args.preds, f"{tok}.npz")
        lp = os.path.join(args.pack, "lidar", f"{tok}.npy")
        if not (os.path.exists(pp) and os.path.exists(lp)):
            continue
        gt = lidar_voxels(M.lidar_to_ego(np.load(lp), row["lidar"]))[near3]
        cls = np.load(pp)["cls"]
        truth_all.append(gt)
        for name, fn in TRANSFORMS.items():
            pred_all[name].append(
                (np.ascontiguousarray(fn(cls)) != FREE)[near3])
        used += 1

    truth = np.concatenate(truth_all)
    print(f"frames {used}   voxels {truth.size:,}   "
          f"lidar-occupied {truth.mean()*100:.2f}%   r < {args.max_range:g} m\n")
    res = {}
    for name in TRANSFORMS:
        p = np.concatenate(pred_all[name])
        res[name] = mcc(p, truth)
    for name, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<16} MCC {v:+.4f}")

    ranked = sorted(res.items(), key=lambda kv: -kv[1])
    best, second = ranked[0], ranked[1]
    margin = best[1] - second[1]
    print(f"\nwinner {best[0]}   margin over {second[0]}  {margin:+.4f}"
          f"   (rule: >= {MARGIN_REQUIRED})")
    ok = margin >= MARGIN_REQUIRED
    print("ORIENTATION RESOLVED: " + best[0] if ok else
          "NOT RESOLVED -- margin below the pre-registered threshold, stop")
    if args.out:
        json.dump({"mcc": res, "winner": best[0], "margin": margin,
                   "resolved": bool(ok), "frames": used,
                   "max_range": args.max_range}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
