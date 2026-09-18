"""Is the model's confidence miscalibrated where the cameras cannot see?

This is the experiment the observability measure was built for, so the analysis
was written down before the numbers existed and is not adjusted now that they
do.

PRE-REGISTERED (docs/METHOD_FREEZE.md), fixed before any prediction existed
--------------------------------------------------------------------------
Binning. 78.7% of occupied cells have observability exactly zero, so equal-width
or equal-count bins would put four fifths of the data in one bucket:

    bin 0        observability == 0
    bins 1..10   deciles WITHIN observability > 0

Confidence. Max-softmax over the 18 channels, per voxel. The column-collapsed
p_occ is excluded outright: it saturated, and the class it reported was the road
rather than what stood on it.

Headline. gap = confidence - accuracy, and the statistic is

    gap(obs = 0) - gap(obs > 0)

Positive means the model is more overconfident where no camera can see.
Significance is a scene-level bootstrap: consecutive keyframes in a scene
contain the same objects, so frames are not independent.

Ground truth.
  --gt lidar   SMOKE TEST ONLY. LiDAR shares the camera's occlusion geometry,
               so the effect cannot be separated from GT error. Not publishable,
               and the script says so on every run.
  --gt occ3d   The publishable mode, and the one that permits the comparison
               that matters: ours versus Occ3D's own binary mask_camera.

Implementation note
-------------------
Nothing is concatenated across frames. 6,019 frames x 640,000 voxels is 3.85
billion values per field, and holding that costs tens of gigabytes for
statistics that are sums. Everything accumulates per (scene, bin) instead,
which makes the scene bootstrap exact rather than approximate, because the mean
of a union of scenes is recoverable from their sums.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OCC_N, OCC_RES, OCC_RNG = 200, 0.4, 40.0
OCC_Z0, OCC_ZRES, OCC_NZ = -1.0, 0.4, 16
OBS_N, OBS_RES, OBS_RNG = 216, 0.5, 54.0
FREE = 17
N_DECILES = 10


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "odfm_eval", os.path.join(HERE, "eval_observability_scaled.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_eval_module()


def occ_to_obs_index():
    """Occ3D BEV column -> observability grid cell. Both ego-framed and centred.

    Occ3D's +/-40 m sits strictly inside the observability grid's +/-54 m, so
    every column has a home. The assertion stays anyway: a silent clip would
    hand the outer ring somebody else's observability.
    """
    ax = (np.arange(OCC_N) + 0.5) * OCC_RES - OCC_RNG
    oi = np.floor((ax + OBS_RNG) / OBS_RES).astype(np.int32)
    assert oi.min() >= 0 and oi.max() < OBS_N, (oi.min(), oi.max())
    return oi


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


def decile_edges(pack, index, oi, n_sample=300, seed=0, obs_dir=""):
    """Deciles of observability > 0, from the maps alone -- no predictions.

    Deliberately computed without touching a single model output, so the bin
    boundaries cannot be influenced by the thing being measured.
    """
    rs = np.random.default_rng(seed)
    vals = []
    for i in rs.choice(len(index), min(n_sample, len(index)), replace=False):
        tk = index[i]["token"]
        p = (os.path.join(obs_dir, f"{tk}.npy") if obs_dir else
             os.path.join(pack, "observability", f"{tk}.npy"))
        if not os.path.exists(p):
            continue
        v = (np.load(p).astype(np.float32).ravel() / 255.0 if obs_dir else
             np.load(p).astype(np.float32)[np.ix_(oi, oi)].ravel())
        vals.append(v[v > 0])
    v = np.concatenate(vals)
    qs = np.linspace(0, 100, N_DECILES + 1)[1:-1]
    return np.unique(np.percentile(v, qs)), v.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--preds", default="data/preds/preds_voxel")
    ap.add_argument("--gt", choices=["lidar", "occ3d"], default="lidar")
    ap.add_argument("--occ3d", default="data/occ3d/gts")
    ap.add_argument("--obs", default="", help="per-voxel observability dir "
                    "(200x200x16 uint8). Empty = the old 2D broadcast.")
    ap.add_argument("--frames", type=int, default=0, help="0 = all")
    ap.add_argument("--split-zero", action="store_true",
                    help="split the zero bin into OUT-OF-VIEW (no frustum) "
                         "and OCCLUDED (in view, behind the first surface). "
                         "They behave oppositely and pooling them is what "
                         "made H1 fail (A13).")
    ap.add_argument("--nonfree", action="store_true",
                    help="score only voxels whose GT is not free. Tests "
                         "whether the decile hump is a class-composition "
                         "artifact rather than a visibility effect.")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.gt == "lidar":
        print("*** SMOKE TEST. LiDAR GT shares camera occlusion geometry.")
        print("*** Nothing from this run is publishable.\n")

    calib = json.load(open(os.path.join(args.pack, "calib.json")))
    index = json.load(open(os.path.join(args.pack, "index.json")))
    if args.frames:
        # Sample, do not take the head. Consecutive keyframes belong to the
        # same scene, so index[:300] is ~8 scenes, and the scene bootstrap
        # then has 8 independent units and no power worth the name.
        rs0 = np.random.default_rng(1234)
        sel = sorted(rs0.choice(len(index), min(args.frames, len(index)),
                                replace=False))
        index = [index[i] for i in sel]
    oi = occ_to_obs_index()

    occ3d_map = {}
    if args.gt == "occ3d":
        # One walk, then a token -> path dict. The scene directory name is not
        # derivable from the sample token, so it has to be discovered.
        for scene in sorted(os.listdir(args.occ3d)):
            d = os.path.join(args.occ3d, scene)
            if not os.path.isdir(d):
                continue
            for t in os.listdir(d):
                f = os.path.join(d, t, "labels.npz")
                if os.path.exists(f):
                    occ3d_map[t] = f
        print(f"occ3d: {len(occ3d_map):,} labelled frames found under "
              f"{args.occ3d}")
        have = sum(1 for r in index if r["token"] in occ3d_map)
        print(f"       {have:,} of {len(index):,} split frames have GT\n")

    edges, n_pos = decile_edges(args.pack, index, oi, obs_dir=args.obs)
    print(f"decile edges from {n_pos:,} positive-observability columns "
          f"(no predictions touched):")
    print("  " + "  ".join(f"{e:.3f}" for e in edges) + "\n")

    import frustum_util as FU
    n_bins = len(edges) + 2 + (1 if args.split_zero else 0)
    # per scene: [n, sum_conf, sum_correct] for each bin
    acc = defaultdict(lambda: np.zeros((n_bins, 4), np.float64))
    used = skipped = 0
    t0 = time.time()
    for r, row in enumerate(index):
        tok = row["token"]
        pp = os.path.join(args.preds, f"{tok}.npz")
        op = (os.path.join(args.obs, f"{tok}.npy") if args.obs else
              os.path.join(args.pack, "observability", f"{tok}.npy"))
        if not (os.path.exists(pp) and os.path.exists(op)):
            skipped += 1
            continue
        z = np.load(pp)
        cls = z["cls"]
        conf = z["conf"].astype(np.float32) / 255.0

        if args.gt == "lidar":
            lp = os.path.join(args.pack, "lidar", f"{tok}.npy")
            if not os.path.exists(lp):
                skipped += 1
                continue
            gt_occ = lidar_voxels(M.lidar_to_ego(np.load(lp), row["lidar"]))
            correct = (cls != FREE) == gt_occ
            keep = None
        else:
            gp = occ3d_map.get(tok)
            if gp is None:
                skipped += 1
                continue
            g = np.load(gp)
            sem = g["semantics"]
            correct = (cls == sem)
            # mask_camera marks the voxels Occ3D itself considers camera
            # observable. Scoring outside it would grade the model on voxels
            # its training loss never supervised, which is how you end up
            # measuring the unsupervised `manmade` fill instead of the model.
            keep = g["mask_camera"].astype(bool)

        if args.obs:
            # Per-voxel measure: no column broadcast, no regridding. The 3D
            # map is already on the Occ3D grid by construction.
            obs = np.load(os.path.join(args.obs, f"{tok}.npy")
                          ).astype(np.float32) / 255.0
        else:
            obs2d = np.load(op).astype(np.float32)[np.ix_(oi, oi)]
            obs = np.repeat(obs2d[:, :, None], OCC_NZ, axis=2)

        gtfree = (sem == FREE) if args.gt == "occ3d" else ~gt_occ
        if keep is not None:
            conf, correct, obs = conf[keep], correct[keep], obs[keep]
            gtfree = gtfree[keep]
        conf, correct = conf.ravel(), correct.ravel()
        obs, gtfree = obs.ravel(), gtfree.ravel()
        if args.nonfree:
            k2 = ~gtfree
            conf, correct, obs, gtfree = conf[k2], correct[k2], obs[k2], gtfree[k2]

        if args.split_zero:
            fr = FU.in_any_frustum(calib[tok]["cams"], M.quat_to_rot)
            fr = fr.ravel()[keep.ravel()] if keep is not None else fr.ravel()
            if args.nonfree:
                fr = fr[k2]
            # 0 = out of view, 1 = in view but occluded, 2.. = deciles
            b = np.where(obs <= 0, np.where(fr, 1, 0),
                         np.digitize(obs, edges) + 2)
        else:
            b = np.where(obs <= 0, 0, np.digitize(obs, edges) + 1)
        a = acc[row["scene"]]
        np.add.at(a[:, 0], b, 1.0)
        np.add.at(a[:, 1], b, conf.astype(np.float64))
        np.add.at(a[:, 2], b, correct.astype(np.float64))
        np.add.at(a[:, 3], b, gtfree.astype(np.float64))
        used += 1
        if used % 500 == 0:
            el = time.time() - t0
            print(f"  {used}/{len(index)}  {el/used:.3f}s/frame  "
                  f"eta {(len(index)-used)*el/used/60:.1f} min", flush=True)

    if not used:
        sys.exit("no frames processed")
    scenes = sorted(acc)
    A = np.stack([acc[s] for s in scenes])       # (S, bins, 3)
    tot = A.sum(0)
    print(f"\nframes {used}  skipped {skipped}  scenes {len(scenes)}  "
          f"voxels {tot[:,0].sum():,.0f}")

    names = (["out of view", "occluded"] if args.split_zero else ["obs = 0"])
    names += [f"decile {i+1}" for i in range(n_bins - len(names))]
    print(f"\n{'bin':<12}{'n':>16}{'conf':>9}{'acc':>9}{'gap':>9}"
          f"{'%GTfree':>9}")
    rows_out = []
    for i, nm in enumerate(names):
        n, sc, sa, sf = tot[i]
        if n == 0:
            continue
        c, a_, fr = sc / n, sa / n, sf / n
        print(f"{nm:<12}{n:>16,.0f}{c:>9.4f}{a_:>9.4f}{c-a_:>+9.4f}"
              f"{fr*100:>9.1f}")
        rows_out.append({"bin": nm, "n": int(n), "conf": c, "acc": a_,
                         "gap": c - a_, "gt_free_share": fr})

    nz = 2 if args.split_zero else 1      # how many bins are "observability 0"

    def delta(sel):
        z = A[sel].sum(0)
        zz = z[:nz].sum(0)
        n0, c0, a0 = zz[0], zz[1], zz[2]
        rest = z[nz:].sum(0)
        n1, c1, a1 = rest[0], rest[1], rest[2]
        if n0 == 0 or n1 == 0:
            return np.nan
        return (c0 - a0) / n0 - (c1 - a1) / n1

    point = delta(np.arange(len(scenes)))
    rs = np.random.default_rng(0)
    boot = np.array([delta(rs.integers(0, len(scenes), len(scenes)))
                     for _ in range(args.boot)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    print(f"\ngap(obs=0) - gap(obs>0) = {point:+.4f}   "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]   {len(scenes)} scenes")
    print("CI excludes zero -- the effect is real"
          if (lo > 0) == (hi > 0) else "CI spans zero -- no effect shown")

    if args.out:
        json.dump({"gt": args.gt, "frames": used, "scenes": len(scenes),
                   "edges": edges.tolist(), "bins": rows_out,
                   "delta": float(point), "ci": [float(lo), float(hi)]},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
