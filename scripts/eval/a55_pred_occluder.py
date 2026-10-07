#!/usr/bin/env python3
"""A55: does the object-level result survive a DEPLOYABLE occluder?

The frozen observability maps resolve occlusion against the Occ3D
ground-truth volume, which is not available on a vehicle (A54). This
rebuilds nothing; it scores the already-rebuilt predicted-occluder maps
against the same objects, paired on the same 800 frames, so the only thing
that differs between the two arms is where occ() came from.

Pre-registered acceptance (A54/A55, before this ran):
    HOLDS  AUROC(obs_pred) - AUROC(MSP) > 0 at object level, 95% paired
           scene-bootstrap interval excluding zero.
"""
from __future__ import annotations
import glob, importlib.util, json, os
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FREE = 17
PRED_DIR = "data/pack/obs_pred_s800"
GT_DIR = "data/pack/obs_max"

_spec = importlib.util.spec_from_file_location(
    "md", os.path.join(HERE, "missed_detection.py"))
md = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(md)

_W = {}


def _init(index, gt_map, order, bounds):
    _W.update(index=index, gt=gt_map, order=order, bounds=bounds)
    an = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"),
                 allow_pickle=True)
    _W["ann"] = {k: an[k] for k in ("category", "translation", "rotation",
                                    "size", "num_lidar_pts")}
    _W["names"] = an["category_names"]


def _one(fi):
    row = _W["index"][fi]; tok = row["token"]
    ann, names = _W["ann"], _W["names"]
    g = np.load(_W["gt"][tok]); gcls = g["semantics"].astype(np.int16)
    p = np.load(os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz"))
    pcls = p["cls"].astype(np.int16)
    pconf = p["conf"].astype(np.float32) / 255.0
    o_gt = np.load(os.path.join(ROOT, GT_DIR, tok + ".npy")).astype(np.float32) / 255.
    o_pr = np.load(os.path.join(ROOT, PRED_DIR, tok + ".npy")).astype(np.float32) / 255.
    cam0 = row["cams"]["CAM_FRONT"]
    Re = md.qrot(np.asarray(cam0["ego2global_rotation"], np.float64))
    te = np.asarray(cam0["ego2global_translation"], np.float64)
    out = []
    for k in _W["order"][_W["bounds"][fi]:_W["bounds"][fi + 1]]:
        cid = md.CAT2OCC.get(str(names[ann["category"][k]]))
        if cid is None or cid not in md.DYN:
            continue
        if ann["num_lidar_pts"][k] < 1:
            continue
        c = Re.T @ (ann["translation"][k].astype(np.float64) - te)
        R = Re.T @ md.qrot(np.asarray(ann["rotation"][k], np.float64))
        vx = md.box_voxels(c, ann["size"][k].astype(np.float64), R, 0.0)
        if vx is None:
            continue
        gm = gcls[vx] == cid
        if not gm.any():
            continue
        out.append((row["scene"],
                    float(o_gt[vx][gm].max()),      # evaluation-time
                    float(o_pr[vx][gm].max()),      # deployable
                    float(pconf[vx].max()),         # MSP
                    bool(np.isin(pcls[vx], list(md.DYN)).any())))
    return out


def auroc(score, pos):
    """Exact rank AUROC with tie handling. pos = the event being ranked."""
    s = np.asarray(score, np.float64); y = np.asarray(pos, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), np.float64); r[order] = np.arange(1, len(s) + 1)
    ss = s[order]; i = 0
    while i < len(ss):                       # average ranks within ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    gt_map = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}
    toks = {os.path.basename(q)[:-4]
            for q in glob.glob(os.path.join(ROOT, PRED_DIR, "*.npy"))}
    ann = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"),
                  allow_pickle=True)
    fr = ann["frame"]; order = np.argsort(fr, kind="stable")
    bounds = np.searchsorted(fr[order], np.arange(len(index) + 1))
    fis = [i for i, r in enumerate(index)
           if r["token"] in toks and r["token"] in gt_map
           and os.path.exists(os.path.join(ROOT, "data/preds/preds_voxel",
                                           r["token"] + ".npz"))]
    print(f"frames {len(fis)}")
    with Pool(initializer=_init, initargs=(index, gt_map, order, bounds)) as pool:
        rows = [r for part in pool.map(_one, fis, chunksize=8) for r in part]

    scene = np.array([r[0] for r in rows])
    ogt = np.array([r[1] for r in rows]); opr = np.array([r[2] for r in rows])
    msp = np.array([r[3] for r in rows]); det = np.array([r[4] for r in rows])
    miss = ~det                                   # the event we rank
    print(f"objects {len(rows):,}  scenes {len(set(scene))}  "
          f"missed {100*miss.mean():.2f}%")

    # observability and MSP both rank "missed" when INVERTED (low obs -> miss)
    A = {"obs_gt (evaluation-time)": auroc(-ogt, miss),
         "obs_pred (deployable)":    auroc(-opr, miss),
         "MSP":                      auroc(-msp, miss)}
    for k, v in A.items():
        print(f"  {k:<28} {v:.4f}")

    uq = np.array(sorted(set(scene)))
    idx = {s: np.flatnonzero(scene == s) for s in uq}
    rng = np.random.default_rng(7)

    def boot(a, b):
        d = []
        for _ in range(2000):
            pick = rng.choice(len(uq), len(uq), replace=True)
            sel = np.concatenate([idx[uq[i]] for i in pick])
            d.append(auroc(-a[sel], miss[sel]) - auroc(-b[sel], miss[sel]))
        d = np.array(d)
        return [float(np.nanmean(d)), float(np.nanpercentile(d, 2.5)),
                float(np.nanpercentile(d, 97.5))]

    C = {"obs_pred - MSP": boot(opr, msp),
         "obs_gt - MSP":   boot(ogt, msp),
         "obs_pred - obs_gt": boot(opr, ogt)}
    for k, v in C.items():
        verdict = "HOLDS" if v[1] > 0 else ("FAILS" if v[2] < 0 else "NULL")
        print(f"  {k:<22} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]  {verdict}")

    np.savez_compressed(os.path.join(ROOT, "outputs/artifacts/a55_rows.npz"),
                        scene=scene, ogt=ogt, opr=opr, msp=msp, det=det)

    # A55b: is the stated mechanism true? If a miss erases its own occluder,
    # obs_pred should be systematically HIGHER than obs_gt on MISSED objects
    # and similar on detected ones. Measured, not asserted.
    for nm, m in (("missed", miss), ("detected", ~miss)):
        print(f"  {nm:<9} n={m.sum():>6,}  obs_gt {ogt[m].mean():.4f}  "
              f"obs_pred {opr[m].mean():.4f}  "
              f"delta {opr[m].mean()-ogt[m].mean():+.4f}  "
              f"frac obs_pred>obs_gt {100*(opr[m]>ogt[m]+0.02).mean():.1f}%")

    out = dict(level="object", scope="test subsample", frames=len(fis),
               objects=len(rows), scenes=int(len(set(scene))),
               sample=800, seed=11, boot=2000, auroc=A, contrasts=C,
               missed_rate=float(miss.mean()))
    dst = os.path.join(ROOT, "outputs/artifacts/a55_pred_occluder.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
