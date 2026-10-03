"""H2: does continuous observability beat Occ3D's binary mask_camera at
predicting where the model is wrong?

This is the experiment the contribution turns on. If a binary mask that ships
with the benchmark predicts model error as well as our continuous, resolution-
and trust-weighted, noisy-OR measure, then the extra machinery is not earning
its place and the paper has to say so. Pre-registered in METHOD_FREEZE A5 as
reported whichever way it comes out.

SCOPE CORRECTION, recorded rather than quietly applied
------------------------------------------------------
A5 said "on the same voxels". Inside mask_camera the mask is constant 1, so its
AUROC is undefined there. H2 therefore runs over the FULL volume, which is also
the honest operational question: at inference time, with no ground truth, which
signal better tells the stack where to distrust the model? Both signals are
available then; neither needs GT to compute.

The cost of full-volume scoring is that "error" outside mask_camera includes
the unsupervised `manmade` fill, which no one supervised and which arguably
should not count against the model. So both restrictions are reported:

  full     every voxel. The operational question.
  masked   mask_camera only. Here mask_camera has no variance and cannot be
           scored, so only observability gets an AUROC -- reported to show
           our measure still discriminates inside the region the binary mask
           calls uniformly visible, which is precisely what a binary mask
           cannot do.

Scores are oriented so that HIGHER means MORE LIKELY WRONG:
    observability -> 1 - obs
    mask_camera   -> 1 - mask
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import sys
import numpy as np

OCC_N, OCC_RES, OCC_RNG = 200, 0.4, 40.0
OCC_NZ = 16
OBS_N, OBS_RES, OBS_RNG = 216, 0.5, 54.0
FREE = 17
NB = 256          # histogram bins; exact AUROC on quantised uint8 scores


def occ_to_obs_index():
    ax = (np.arange(OCC_N) + 0.5) * OCC_RES - OCC_RNG
    oi = np.floor((ax + OBS_RNG) / OBS_RES).astype(np.int32)
    assert oi.min() >= 0 and oi.max() < OBS_N
    return oi


def auroc_from_hist(pos, neg):
    """AUROC from per-bin counts, ties handled as half-credit.

    Histograms rather than raw arrays because the full volume is 3.85 billion
    voxels; the scores are uint8-quantised anyway, so this is exact, not an
    approximation.
    """
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    cneg = np.cumsum(neg) - neg          # strictly-lower-scoring negatives
    return float((pos * (cneg + 0.5 * neg)).sum() / (P * N))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--preds", default="data/preds/preds_voxel")
    ap.add_argument("--gts",
                    default="data/occ3d/Occupancy3D-nuScenes-trainval/gts")
    ap.add_argument("--obs", default="", help="per-voxel observability dir. "
                    "Empty = the old 2D broadcast.")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="")
    # A45: H2 was the paper's central claim and had no dev/test separation at
    # all. The split existed in outputs/artifacts/val_dev_test.json from 11
    # September and no script read it. It does now.
    ap.add_argument("--scenes", default="all", choices=["all", "dev", "test"])
    args = ap.parse_args()

    gt_map = {os.path.basename(os.path.dirname(p)): p
              for p in glob.glob(os.path.join(args.gts, "*", "*", "labels.npz"))}
    index = json.load(open(os.path.join(args.pack, "index.json")))
    if args.frames:
        rs0 = np.random.default_rng(1234)
        sel = sorted(rs0.choice(len(index), min(args.frames, len(index)),
                                replace=False))
        index = [index[i] for i in sel]
    oi = occ_to_obs_index()

    # per scene: histograms of each score, split by whether the voxel is wrong
    H = {}
    def blank():
        return {k: np.zeros((2, NB), np.int64)
                for k in ("obs_full", "mask_full", "obs_masked")}

    keep = None
    if args.scenes != "all":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import split as _sp
        keep = _sp.dev_scenes() if args.scenes == "dev" else _sp.test_scenes()
        print(f"[A45] {args.scenes.upper()} scenes only, {len(keep)} of 150, "
              f"split digest {_sp.digest()}", flush=True)
    used = 0
    t0 = time.time()
    for row in index:
        if keep is not None and row["scene"] not in keep:
            continue
        tok = row["token"]
        pp = os.path.join(args.preds, f"{tok}.npz")
        op = (os.path.join(args.obs, f"{tok}.npy") if args.obs else
              os.path.join(args.pack, "observability", f"{tok}.npy"))
        gp = gt_map.get(tok)
        if not (gp and os.path.exists(pp) and os.path.exists(op)):
            continue
        g = np.load(gp)
        gt, mask = g["semantics"], g["mask_camera"].astype(bool)
        cls = np.load(pp)["cls"]
        wrong = (cls != gt)

        if args.obs:
            obs = np.load(op).astype(np.float32) / 255.0
        else:
            obs2d = np.load(op).astype(np.float32)[np.ix_(oi, oi)]
            obs = np.repeat(obs2d[:, :, None], OCC_NZ, axis=2)
        s_obs = np.clip(np.rint((1.0 - obs) * (NB - 1)), 0, NB - 1).astype(np.uint8)
        s_msk = np.where(mask, 0, NB - 1).astype(np.uint8)

        h = H.setdefault(row["scene"], blank())
        w, nw = wrong.ravel(), ~wrong.ravel()
        h["obs_full"][1] += np.bincount(s_obs.ravel()[w], minlength=NB)
        h["obs_full"][0] += np.bincount(s_obs.ravel()[nw], minlength=NB)
        h["mask_full"][1] += np.bincount(s_msk.ravel()[w], minlength=NB)
        h["mask_full"][0] += np.bincount(s_msk.ravel()[nw], minlength=NB)
        wm = wrong[mask]
        sm = s_obs[mask]
        h["obs_masked"][1] += np.bincount(sm[wm], minlength=NB)
        h["obs_masked"][0] += np.bincount(sm[~wm], minlength=NB)

        used += 1
        if used % 1000 == 0:
            el = time.time() - t0
            print(f"  {used}  {el/used:.3f}s/frame", flush=True)

    scenes = sorted(H)
    print(f"\nframes {used}   scenes {len(scenes)}")

    def agg(keys, sel):
        out = {}
        for k in keys:
            a = np.zeros((2, NB), np.int64)
            for i in sel:
                a += H[scenes[i]][k]
            out[k] = auroc_from_hist(a[1], a[0])
        return out

    allsel = np.arange(len(scenes))
    keys = ("obs_full", "mask_full", "obs_masked")
    point = agg(keys, allsel)

    rs = np.random.default_rng(0)
    diffs, obs_b, msk_b = [], [], []
    for _ in range(args.boot):
        s = rs.integers(0, len(scenes), len(scenes))
        r = agg(("obs_full", "mask_full"), s)
        obs_b.append(r["obs_full"]); msk_b.append(r["mask_full"])
        diffs.append(r["obs_full"] - r["mask_full"])
    diffs = np.array(diffs)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])

    print(f"\nFULL VOLUME  (predicting per-voxel error)")
    print(f"  observability   AUROC {point['obs_full']:.4f}   "
          f"[{np.nanpercentile(obs_b,2.5):.4f}, {np.nanpercentile(obs_b,97.5):.4f}]")
    print(f"  mask_camera     AUROC {point['mask_full']:.4f}   "
          f"[{np.nanpercentile(msk_b,2.5):.4f}, {np.nanpercentile(msk_b,97.5):.4f}]")
    d = point['obs_full'] - point['mask_full']
    print(f"  difference      {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("  " + ("observability WINS -- CI excludes zero"
                  if lo > 0 else
                  "mask_camera WINS -- CI excludes zero" if hi < 0 else
                  "NO DIFFERENCE SHOWN -- CI spans zero"))

    print(f"\nINSIDE mask_camera  (where the binary mask is constant and"
          f" cannot be scored at all)")
    print(f"  observability   AUROC {point['obs_masked']:.4f}")

    if args.out:
        json.dump({"point": point, "diff": d, "ci": [float(lo), float(hi)],
                   "frames": used, "scenes": len(scenes)},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
