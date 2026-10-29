#!/usr/bin/env python3
"""Heterogeneous-ensemble baseline at the object level. A69, fixed by A73.

The sampling family of uncertainty methods was never run (main.tex line 940
calls it the largest gap in the evaluation). MC-dropout is unavailable: the
occupancy path of both BEVDet and FlashOcc has no dropout. A seed ensemble is
unavailable: no seed replicas, and training them is out of scope.

What exists is three independently trained camera-only occupancy models on the
identical sealed split. That is a HETEROGENEOUS ensemble, not a deep ensemble
in Lakshminarayanan's sense, and A69 requires the paper to say so in those
words.

Scores, per A69:
    E1  modal disagreement   1 - (models agreeing with the modal class) / 3
    E2  free-space spread    std of p_free across models

Per A73: target is FB-OCC's miss, miss means no dynamic voxel in the box, and
both mean and max aggregation are reported for both scores.

A73 also gates this script on reproducing FB-OCC's published object-level
AUROCs -- observability 0.7623, MSP 0.7193, tolerance 0.005. Outside that the
pipeline is not the published one and nothing it prints counts.

    python3 scripts/eval/ensemble_object.py --scenes test --boot 2000 \
        --out outputs/artifacts/ensemble_object.json
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
from missed_detection import CAT2OCC, DYN, FREE, box_voxels, qrot  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
REF = {"observability": 0.7623, "MSP (max conf in box)": 0.7193}
TOL = 0.005


def auroc(score, y):
    """AUROC with ties as half credit. y True = positive = MISSED."""
    s = np.asarray(score, float)
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="test", choices=["all", "dev", "test"])
    ap.add_argument("--obs", default="data/pack/obs_max")
    ap.add_argument("--minpts", type=int, default=1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    MODELS = {"fbocc": "data/preds/preds_voxel",
              "m2": "data/preds_flashocc_m2",
              "m1": "data/preds_flashocc_m1"}

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    ann = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"),
                  allow_pickle=True)
    names = ann["category_names"] if "category_names" in ann else ann["names"]
    gt_map = {os.path.basename(os.path.dirname(p)): p
              for p in glob.glob(os.path.join(
                  ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                  "*", "*", "labels.npz"))}

    keep = None
    if a.scenes != "all":
        import split as _sp
        keep = _sp.dev_scenes() if a.scenes == "dev" else _sp.test_scenes()
        print(f"[A45] {a.scenes.upper()} scenes only, {len(keep)} of 150, "
              f"split digest {_sp.digest()}", flush=True)

    # annotations are keyed by FRAME INDEX into index.json, not by token.
    # Same construction missed_detection.py uses, so the object set matches.
    fr = ann["frame"]
    order = np.argsort(fr, kind="stable")
    bounds = np.searchsorted(fr[order], np.arange(len(index) + 1))

    recs, used, t0 = [], 0, time.time()
    for fi, row in enumerate(index):
        if keep is not None and row["scene"] not in keep:
            continue
        if a.frames and used >= a.frames:
            break
        tok = row["token"]
        gp = gt_map.get(tok)
        paths = {m: os.path.join(ROOT, d, tok + ".npz") for m, d in MODELS.items()}
        op = os.path.join(ROOT, a.obs, tok + ".npy")
        if not (gp and os.path.exists(op) and all(map(os.path.exists, paths.values()))):
            continue
        ks = order[bounds[fi]:bounds[fi + 1]]
        if len(ks) == 0:
            continue

        g = np.load(gp)
        gcls = g["semantics"].astype(np.int16)
        P = {m: np.load(p) for m, p in paths.items()}
        cls = {m: P[m]["cls"].astype(np.int16) for m in MODELS}
        pfree = np.stack([P[m]["p_free"].astype(np.float32) / 255.0
                          for m in MODELS])
        conf = P["fbocc"]["conf"].astype(np.float32) / 255.0
        obs = np.load(op).astype(np.float32) / 255.0

        # E1: how many models disagree with the modal class, per voxel.
        stack = np.stack([cls[m] for m in MODELS])            # (3, X, Y, Z)
        agree = np.maximum.reduce([(stack == stack[i]).sum(0) for i in range(3)])
        e1 = 1.0 - agree / 3.0
        # E2: spread of the free-space probability across the committee.
        e2 = pfree.std(0)

        cam0 = row["cams"]["CAM_FRONT"]
        Re = qrot(np.asarray(cam0["ego2global_rotation"], np.float64))
        te = np.asarray(cam0["ego2global_translation"], np.float64)

        for k in ks:
            cid = CAT2OCC.get(str(names[ann["category"][k]]))
            if cid is None or cid not in DYN:
                continue
            if ann["num_lidar_pts"][k] < a.minpts:
                continue
            c = Re.T @ (ann["translation"][k].astype(np.float64) - te)
            R = Re.T @ qrot(np.asarray(ann["rotation"][k], np.float64))
            vx = box_voxels(c, ann["size"][k].astype(np.float64), R, 0.0)
            if vx is None:
                continue
            gm = gcls[vx] == cid
            if not gm.any():
                continue
            recs.append((
                row["scene"],
                float(obs[vx][gm].max()),            # observability, as published
                float(conf[vx].max()),               # MSP, as published
                float(e1[vx].mean()), float(e1[vx].max()),
                float(e2[vx].mean()), float(e2[vx].max()),
                bool(np.isin(cls["fbocc"][vx], list(DYN)).any()),
            ))
        used += 1
        if used % 250 == 0:
            print(f"  {used} frames  {len(recs):,} objects  "
                  f"{(time.time()-t0)/used:.3f}s/frame", flush=True)

    if not recs:
        sys.exit("no objects scored")
    scene = np.array([r[0] for r in recs])
    cols = {"observability": np.array([r[1] for r in recs]),
            "MSP (max conf in box)": np.array([r[2] for r in recs]),
            "E1 modal disagreement (mean)": np.array([r[3] for r in recs]),
            "E1 modal disagreement (max)": np.array([r[4] for r in recs]),
            "E2 p_free spread (mean)": np.array([r[5] for r in recs]),
            "E2 p_free spread (max)": np.array([r[6] for r in recs])}
    missed = ~np.array([r[7] for r in recs], bool)

    # observability and MSP are LOW when a miss is likely; the ensemble scores
    # are HIGH. Signs are set here so every AUROC answers the same question.
    sign = {"observability": -1.0, "MSP (max conf in box)": -1.0}
    print(f"\n{len(recs):,} objects, {len(set(scene))} scenes, "
          f"{missed.mean()*100:.1f}% missed\n")

    point = {k: auroc(sign.get(k, 1.0) * v, missed) for k, v in cols.items()}
    for k in cols:
        print(f"  {k:<32} AUROC {point[k]:.4f}")

    # An AUROC below 0.5 means the score carries signal in the INVERTED
    # direction. A69 fixed the direction (high disagreement = uncertain) and
    # that is what is reported, but a baseline should not be judged by a sign
    # convention: 1 - AUROC is printed alongside so the ensemble is always
    # read at its strongest. Whichever is larger is the fair comparison
    # against observability, and both go in the record.
    inverted = [k for k in cols if k not in REF and point[k] < 0.5]
    if inverted:
        print("\n  below 0.5 -- signal is inverted; best-case reading:")
        for k in inverted:
            print(f"  {k:<32} 1 - AUROC {1 - point[k]:.4f}")

    print("\n--- A73 validation gate ---")
    bad = []
    for k, ref in REF.items():
        d = abs(point[k] - ref)
        ok = d <= TOL
        print(f"  {k:<32} {point[k]:.4f} vs published {ref:.4f}  "
              f"diff {d:.4f}  {'OK' if ok else 'FAIL'}")
        if not ok:
            bad.append(k)
    if bad:
        sys.exit("\nA73 GATE FAILED on " + ", ".join(bad) +
                 ".\nThe pipeline is not the one that produced the published "
                 "table, so the ensemble numbers are NOT comparable to it and "
                 "are not reported. Understand the discrepancy first.")
    print("  gate passes -- ensemble numbers below are comparable")

    scenes = sorted(set(scene))
    idx = {s: np.where(scene == s)[0] for s in scenes}
    rs = np.random.default_rng(a.seed)
    draws = rs.integers(0, len(scenes), size=(a.boot, len(scenes)))
    boot = {k: np.empty(a.boot) for k in cols}
    for b, d in enumerate(draws):
        sel = np.concatenate([idx[scenes[i]] for i in d])
        y = missed[sel]
        for k, v in cols.items():
            boot[k][b] = auroc(sign.get(k, 1.0) * v[sel], y)

    def ci(x):
        x = x[~np.isnan(x)]
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    print("\n--- paired bootstrap over scenes ---")
    out = {"objects": len(recs), "scenes": len(scenes), "boot": a.boot,
           "missed_rate": float(missed.mean()),
           "auroc": point, "ci": {k: list(ci(v)) for k, v in boot.items()},
           "auroc_best_case": {k: max(point[k], 1 - point[k]) for k in cols},
           "contrasts": {}}
    for k in cols:
        lo, hi = ci(boot[k])
        print(f"  {k:<32} {point[k]:.4f}  [{lo:.4f}, {hi:.4f}]")
    print()
    # contrasts use the ensemble's BEST-CASE orientation, so the comparison
    # cannot be accused of being won on a sign convention.
    flip = {k: (point[k] < 0.5 and k not in REF) for k in cols}
    for ref in ("observability", "MSP (max conf in box)"):
        for k in cols:
            if k == ref:
                continue
            bk = (1 - boot[k]) if flip[k] else boot[k]
            d = bk - boot[ref]
            lo, hi = ci(d)
            pt = (1 - point[k] if flip[k] else point[k]) - point[ref]
            tag = "excludes 0" if (lo > 0 or hi < 0) else "SPANS 0"
            mark = " (best-case)" if flip[k] else ""
            print(f"  {k:<32} - {ref:<24} {pt:+.4f}  "
                  f"[{lo:+.4f}, {hi:+.4f}]  {tag}{mark}")
            out["contrasts"][f"{k} - {ref}"] = [pt, lo, hi]
        print()

    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
