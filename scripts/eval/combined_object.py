#!/usr/bin/env python3
"""Observability and confidence combined, at object level. A76.

baselines.py reports a combined score per VOXEL (0.8811 against MSP's 0.8797,
a negligible gain). No combined score has ever been computed per OBJECT --
the granularity where the paper's claim lives and where the two signals are
furthest apart: observability 0.7623, MSP 0.7193.

A76 fixes the method in advance: logistic regression on two standardised
features, observability and MSP, fit on the 75 DEV scenes and applied
unchanged to the 75 TEST scenes. No third feature, no interactions, no
hyperparameter search, one fit.

Acceptance, registered before this existed:
    AUROC(combined) - AUROC(observability) >= +0.01 on TEST, 95% paired
    scene-bootstrap interval excluding zero.

    python3 scripts/eval/combined_object.py --boot 2000 \
        --out outputs/artifacts/combined_object.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from missed_detection import CAT2OCC, DYN, box_voxels, qrot  # noqa: E402
from ensemble_object import auroc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))


def extract(half, obs_dir, minpts, preds_dir):
    """Per-object observability, MSP and miss label for one split half."""
    import split as _sp
    keep = _sp.dev_scenes() if half == "dev" else _sp.test_scenes()
    print(f"[A45] {half.upper()} scenes, {len(keep)} of 150, "
          f"digest {_sp.digest()}", flush=True)

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    ann = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"),
                  allow_pickle=True)
    names = ann["category_names"]
    gt_map = {os.path.basename(os.path.dirname(p)): p
              for p in glob.glob(os.path.join(
                  ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                  "*", "*", "labels.npz"))}
    fr = ann["frame"]
    order = np.argsort(fr, kind="stable")
    bounds = np.searchsorted(fr[order], np.arange(len(index) + 1))

    rows, used, t0 = [], 0, time.time()
    for fi, row in enumerate(index):
        if row["scene"] not in keep:
            continue
        tok = row["token"]
        gp = gt_map.get(tok)
        pp = os.path.join(ROOT, preds_dir, tok + ".npz")
        op = os.path.join(ROOT, obs_dir, tok + ".npy")
        if not (gp and os.path.exists(pp) and os.path.exists(op)):
            continue
        ks = order[bounds[fi]:bounds[fi + 1]]
        if len(ks) == 0:
            continue
        g = np.load(gp)
        gcls = g["semantics"].astype(np.int16)
        P = np.load(pp)
        pcls = P["cls"].astype(np.int16)
        conf = P["conf"].astype(np.float32) / 255.0
        obs = np.load(op).astype(np.float32) / 255.0
        cam0 = row["cams"]["CAM_FRONT"]
        Re = qrot(np.asarray(cam0["ego2global_rotation"], np.float64))
        te = np.asarray(cam0["ego2global_translation"], np.float64)
        for k in ks:
            cid = CAT2OCC.get(str(names[ann["category"][k]]))
            if cid is None or cid not in DYN:
                continue
            if ann["num_lidar_pts"][k] < minpts:
                continue
            c = Re.T @ (ann["translation"][k].astype(np.float64) - te)
            R = Re.T @ qrot(np.asarray(ann["rotation"][k], np.float64))
            vx = box_voxels(c, ann["size"][k].astype(np.float64), R, 0.0)
            if vx is None:
                continue
            gm = gcls[vx] == cid
            if not gm.any():
                continue
            rows.append((row["scene"], float(obs[vx][gm].max()),
                         float(conf[vx].max()),
                         bool(np.isin(pcls[vx], list(DYN)).any())))
        used += 1
        if used % 500 == 0:
            print(f"  {used} frames  {len(rows):,} objects  "
                  f"{(time.time()-t0)/used:.3f}s/frame", flush=True)
    sc = np.array([r[0] for r in rows])
    X = np.array([[r[1], r[2]] for r in rows], np.float64)
    y = ~np.array([r[3] for r in rows], bool)
    print(f"  {half}: {len(rows):,} objects, {len(set(sc))} scenes, "
          f"{y.mean()*100:.1f}% missed\n")
    return sc, X, y


def fit_logistic(X, y, iters=200, lr=0.5):
    """Plain gradient descent. Two features; nothing here needs sklearn."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    yy = y.astype(np.float64)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        w -= lr * (Xb.T @ (p - yy)) / len(Xb)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="data/pack/obs_max")
    ap.add_argument("--preds", default="data/preds/preds_voxel")
    ap.add_argument("--minpts", type=int, default=1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    sc_d, X_d, y_d = extract("dev", a.obs, a.minpts, a.preds)
    sc_t, X_t, y_t = extract("test", a.obs, a.minpts, a.preds)

    # standardise on DEV only; the same constants are applied to TEST.
    mu, sd = X_d.mean(0), X_d.std(0)
    sd[sd == 0] = 1.0
    w = fit_logistic((X_d - mu) / sd, y_d)
    print(f"dev fit: w_obs {w[0]:+.4f}  w_msp {w[1]:+.4f}  bias {w[2]:+.4f}")
    print("  (both weights negative = low observability AND low confidence "
          "each raise the predicted miss probability)\n")

    Z = (X_t - mu) / sd
    comb = 1.0 / (1.0 + np.exp(-(np.hstack([Z, np.ones((len(Z), 1))]) @ w)))
    cols = {"observability": -X_t[:, 0],
            "MSP (max conf in box)": -X_t[:, 1],
            "combined (dev-fit)": comb}
    point = {k: auroc(v, y_t) for k, v in cols.items()}
    for k, v in point.items():
        print(f"  {k:<26} AUROC {v:.4f}")

    scenes = sorted(set(sc_t))
    idx = {s: np.where(sc_t == s)[0] for s in scenes}
    rs = np.random.default_rng(a.seed)
    draws = rs.integers(0, len(scenes), size=(a.boot, len(scenes)))
    boot = {k: np.empty(a.boot) for k in cols}
    for b, d in enumerate(draws):
        sel = np.concatenate([idx[scenes[i]] for i in d])
        yy = y_t[sel]
        for k, v in cols.items():
            boot[k][b] = auroc(v[sel], yy)

    def ci(x):
        x = x[~np.isnan(x)]
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    print("\n--- paired bootstrap over TEST scenes ---")
    for k in cols:
        lo, hi = ci(boot[k])
        print(f"  {k:<26} {point[k]:.4f}  [{lo:.4f}, {hi:.4f}]")
    print()
    out = {"objects": int(len(y_t)), "scenes": len(scenes), "boot": a.boot,
           "dev_weights": {"obs": w[0], "msp": w[1], "bias": w[2]},
           "auroc": point, "ci": {k: list(ci(v)) for k, v in boot.items()},
           "contrasts": {}}
    for ref in ("observability", "MSP (max conf in box)"):
        d = boot["combined (dev-fit)"] - boot[ref]
        lo, hi = ci(d)
        pt = point["combined (dev-fit)"] - point[ref]
        tag = "excludes 0" if (lo > 0 or hi < 0) else "SPANS 0"
        print(f"  combined - {ref:<26} {pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {tag}")
        out["contrasts"][f"combined - {ref}"] = [pt, lo, hi]

    d = point["combined (dev-fit)"] - point["observability"]
    lo, hi = ci(boot["combined (dev-fit)"] - boot["observability"])
    ok = (d >= 0.01) and (lo > 0 or hi < 0)
    print(f"\n--- A76 acceptance ---")
    print(f"  gain over observability >= +0.01   {d:+.4f}   "
          f"{'yes' if d >= 0.01 else 'NO'}")
    print(f"  interval excludes zero             [{lo:+.4f}, {hi:+.4f}]   "
          f"{'yes' if (lo > 0 or hi < 0) else 'NO'}")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    out["verdict"] = "PASS" if ok else "FAIL"

    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
