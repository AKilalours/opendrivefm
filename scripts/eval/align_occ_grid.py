"""Resolve the orientation of FB-OCC's BEV output empirically, not by convention.

FB-OCC returns a (200, 200, 16) occupancy volume on the Occ3D grid: 0.4 m
voxels spanning +/-40 m, ego-framed. The observability cache uses a different
grid -- 0.5 m over +/-54 m -- with the convention flat = i*n + j, where
x = (i+0.5)*res - rng_m and y = (j+0.5)*res - rng_m, i.e. axis 0 is forward
and axis 1 is left.

Whether Occ3D uses that same (x, y) order, or (y, x), or flips either axis, is
a documentation question, and documentation is exactly the sort of thing that
is wrong in a way nobody notices until a headline number is built on it. A
90-degree error is invisible in aggregate statistics and fatal to every claim
downstream.

So this measures it. LiDAR gives an independent occupancy map on the Occ3D
grid; the model's own p_occ should agree with it. Score all eight dihedral
transforms by AUROC and take the winner. The correct one should not be close.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

OCC3D_N = 200
OCC3D_RES = 0.4
OCC3D_RNG = 40.0
# Occ3D's vertical extent: 16 levels of 0.4 m starting at z = -1.0 in the ego
# frame. z_occ is the index of the least-free level, so this converts it to a
# height that can be compared against a LiDAR column top.
Z0 = -1.0
Z_RES = 0.4


def _load_eval_module():
    path = os.path.join(HERE, "eval_observability_scaled.py")
    spec = importlib.util.spec_from_file_location("odfm_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_eval_module()

# The eight ways a square array can be laid on a square grid. Named by what
# they do, so a result can be quoted without decoding an index.
TRANSFORMS = {
    "identity":        lambda a: a,
    "rot90":           lambda a: np.rot90(a, 1),
    "rot180":          lambda a: np.rot90(a, 2),
    "rot270":          lambda a: np.rot90(a, 3),
    "transpose":       lambda a: a.T,
    "flip_x":          lambda a: a[::-1, :],
    "flip_y":          lambda a: a[:, ::-1],
    "anti_transpose":  lambda a: a[::-1, ::-1].T,
}


def _rank(v: np.ndarray) -> np.ndarray:
    """Ranks for a Spearman correlation, ties averaged."""
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    ranks = np.empty(sv.size, np.float64)
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j)
        i = j + 1
    out = np.empty_like(ranks)
    out[order] = ranks
    return out


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUROC with tie handling. Ties matter here: p_occ is saturated."""
    pos = int(label.sum())
    neg = label.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(s.size, np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    return float((r[label].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--preds", default="data/preds/fbocc_val")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--min-pts", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    index = json.load(open(os.path.join(args.pack, "index.json")))
    rng = np.random.default_rng(0)
    rows = [index[i] for i in rng.choice(len(index), args.frames, replace=False)]

    # A cell right under the ego vehicle is occupied by the ego vehicle in
    # every frame and in every orientation, so it discriminates nothing and
    # dilutes the contrast. Drop the 4 m box around the origin.
    gax = (np.arange(OCC3D_N) + 0.5) * OCC3D_RES - OCC3D_RNG
    X, Y = np.meshgrid(gax, gax, indexing="ij")
    keep_mask = (np.abs(X) > 2.0) | (np.abs(Y) > 2.0)

    scores = {k: [] for k in TRANSFORMS}
    heights = {k: [] for k in TRANSFORMS}
    labels = []
    ztops = []
    used = 0
    for row in rows:
        tok = row["token"]
        lp = os.path.join(args.pack, "lidar", f"{tok}.npy")
        pp = os.path.join(args.preds, f"{tok}.npz")
        if not (os.path.exists(lp) and os.path.exists(pp)):
            continue
        pts = M.lidar_to_ego(np.load(lp), row["lidar"])
        occ, z_top, _ = M.occupancy_from_lidar(
            pts, OCC3D_N, None, res=OCC3D_RES, rng_m=OCC3D_RNG,
            min_pts=args.min_pts)
        d = np.load(pp)
        p = d["p_occ"].astype(np.float32)
        h = d["z_occ"].astype(np.float32) * Z_RES + Z0
        # Only columns with a real LiDAR return have a height to compare
        # against; empty road gives z_top = 0 by convention, not by
        # measurement, and would reward any transform equally.
        hm = keep_mask & occ
        labels.append(occ[keep_mask])
        ztops.append(z_top[hm])
        for name, fn in TRANSFORMS.items():
            scores[name].append(np.ascontiguousarray(fn(p))[keep_mask])
            heights[name].append(np.ascontiguousarray(fn(h))[hm])
        used += 1

    y = np.concatenate(labels)
    print(f"frames {used}   cells {y.size}   occupied {y.mean():.4f}")
    res = {}
    for name in TRANSFORMS:
        res[name] = auroc(np.concatenate(scores[name]), y)
    for name, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<15} AUROC {v:.4f}")

    zt = np.concatenate(ztops)
    print(f"\nheight probe: {zt.size} columns with a LiDAR return")
    hres = {}
    for name in TRANSFORMS:
        hh = np.concatenate(heights[name])
        hres[name] = float(np.corrcoef(_rank(hh), _rank(zt))[0, 1])
    for name, v in sorted(hres.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<15} spearman {v:+.4f}")

    res = hres          # the height probe decides; p_occ cannot
    best, second = sorted(res.values(), reverse=True)[:2]
    print(f"\nbest {max(res, key=res.get)}  margin over runner-up {best - second:+.4f}")
    if best - second < 0.05:
        print("MARGIN TOO SMALL -- orientation is NOT resolved, do not proceed")
    if args.out:
        json.dump({"p_occ_auroc": {k: auroc(np.concatenate(scores[k]), y)
                                   for k in TRANSFORMS},
                   "height_spearman": hres, "frames": used},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
