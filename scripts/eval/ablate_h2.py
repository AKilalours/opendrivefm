"""Which part of observability carries H2 -- and is it just range in disguise?

H2 shows the measure beats mask_camera at predicting per-voxel error by
+0.0527. "We built a complicated thing and it worked" is not a contribution;
"this specific component is what matters" is. And there is a sharper worry:
earlier in this project a grazing-angle feature was RETRACTED once it turned
out to be a pure range proxy (all six lenses sit within 9.5 cm of each other in
height). The same trap is open here, because coverage goes as 1/Z^2. If a
plain range score matches the full measure, the occlusion machinery is
decoration.

Variants, all scored identically against per-voxel error:

    full        1 - prod_i (1 - T * cov_i * vis_i)        the measure
    no_occ      1 - prod_i (1 - T * cov_i * infrustum_i)  occlusion removed
    no_cov      1 - prod_i (1 - T * vis_i)                resolution removed
    max_cam     max_i (T * cov_i * vis_i)                 noisy-OR removed
    range_only  max_i cov_i                               PURE RANGE PROXY
    n_vis       (# cameras that see it) / 6               pure redundancy

Everything is accumulated as per-scene score histograms in one pass, so the
AUROCs are exact on the uint8 grid and the scene bootstrap is exact too.
Nothing is written to disk.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "bray", os.path.join(HERE, "build_observability_ray.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

NB = 256
VARIANTS = ["full", "no_occ", "no_cov", "max_cam", "range_only", "n_vis"]


def auroc_from_hist(pos, neg):
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    cneg = np.cumsum(neg) - neg
    return float((pos * (cneg + 0.5 * neg)).sum() / (P * N))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--preds", default="data/preds/preds_voxel")
    ap.add_argument("--gts",
                    default="data/occ3d/Occupancy3D-nuScenes-trainval/gts")
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    index = json.load(open(os.path.join(args.pack, "index.json")))
    calib = json.load(open(os.path.join(args.pack, "calib.json")))
    gt_map = {os.path.basename(os.path.dirname(q)): q
              for q in glob.glob(os.path.join(args.gts, "*", "*",
                                              "labels.npz"))}
    if args.frames:
        rs = np.random.default_rng(1234)
        sel = sorted(rs.choice(len(index), min(args.frames, len(index)),
                               replace=False))
        index = [index[i] for i in sel]

    vox = B.voxel_centres()
    H = {}
    used, t0 = 0, time.time()
    for row in index:
        tok = row["token"]
        pp = os.path.join(args.preds, f"{tok}.npz")
        gp = gt_map.get(tok)
        if not (gp and os.path.exists(pp) and tok in calib):
            continue
        g = np.load(gp)
        sem = g["semantics"]
        occ_flat = (sem.reshape(-1) != 17)
        wrong = (np.load(pp)["cls"] != sem).ravel()

        miss_full = np.ones(vox.shape[0], np.float32)
        miss_noocc = np.ones(vox.shape[0], np.float32)
        miss_nocov = np.ones(vox.shape[0], np.float32)
        best = np.zeros(vox.shape[0], np.float32)
        cov_max = np.zeros(vox.shape[0], np.float32)
        nvis = np.zeros(vox.shape[0], np.float32)

        for c in B.CAMS:
            cam = calib[tok]["cams"].get(c)
            if cam is None:
                continue
            W, H_ = cam.get("width", 1600), cam.get("height", 900)
            u, v, z, good, K, ct = B.project(vox, cam)
            ins = good & (u >= 0) & (u < W) & (v >= 0) & (v < H_)
            if not ins.any():
                continue
            vis = B.march_visibility(occ_flat, cam, W, H_) & ins
            area = (K[0, 0] * K[1, 1] * B.OCC_RES ** 2) / np.maximum(z, 0.1) ** 2
            cov = np.clip(area / B.FULL_PX, 0.0, 1.0) * ins
            t = B.TRUST * cov * vis
            miss_full *= (1.0 - t)
            miss_noocc *= (1.0 - B.TRUST * cov * ins)
            miss_nocov *= (1.0 - B.TRUST * vis)
            best = np.maximum(best, t)
            cov_max = np.maximum(cov_max, cov)
            nvis += vis

        S = {"full": 1.0 - miss_full, "no_occ": 1.0 - miss_noocc,
             "no_cov": 1.0 - miss_nocov, "max_cam": best,
             "range_only": cov_max, "n_vis": nvis / 6.0}

        h = H.setdefault(row["scene"],
                         {k: np.zeros((2, NB), np.int64) for k in VARIANTS})
        for k, s in S.items():
            # higher score = more likely WRONG, so invert observability
            q = np.clip(np.rint((1.0 - s) * (NB - 1)), 0, NB - 1).astype(np.uint8)
            h[k][1] += np.bincount(q[wrong], minlength=NB)
            h[k][0] += np.bincount(q[~wrong], minlength=NB)
        used += 1
        if used % 100 == 0:
            el = time.time() - t0
            print(f"  {used}/{len(index)}  {el/used:.2f}s/frame  "
                  f"eta {(len(index)-used)*el/used/60:.1f} min", flush=True)

    scenes = sorted(H)
    print(f"\nframes {used}   scenes {len(scenes)}\n")

    def agg(sel):
        out = {}
        for k in VARIANTS:
            a = np.zeros((2, NB), np.int64)
            for i in sel:
                a += H[scenes[i]][k]
            out[k] = auroc_from_hist(a[1], a[0])
        return out

    point = agg(np.arange(len(scenes)))
    rs = np.random.default_rng(0)
    boot = {k: [] for k in VARIANTS}
    for _ in range(args.boot):
        r = agg(rs.integers(0, len(scenes), len(scenes)))
        for k in VARIANTS:
            boot[k].append(r[k])

    print(f"{'variant':<12}{'AUROC':>9}{'95% CI':>22}{'vs full':>10}")
    for k in VARIANTS:
        b = np.array(boot[k])
        d = np.array(boot["full"]) - b
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        dd = "" if k == "full" else f"{point['full']-point[k]:+.4f}"
        print(f"{k:<12}{point[k]:>9.4f}   [{lo:.4f}, {hi:.4f}]{dd:>10}")
        if k != "full":
            l2, h2 = np.nanpercentile(d, [2.5, 97.5])
            tag = ("full is better" if l2 > 0 else
                   "variant is better" if h2 < 0 else "no difference")
            print(f"{'':<12}   diff CI [{l2:+.4f}, {h2:+.4f}]  {tag}")

    print("\nThe one that matters: if range_only is close to full, the "
          "occlusion machinery is decoration and the measure is a range proxy.")
    if args.out:
        json.dump({"auroc": point, "frames": used, "scenes": len(scenes)},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
