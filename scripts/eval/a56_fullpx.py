#!/usr/bin/env python3
"""A56: how much does pi_full matter?

Analytically, coverage is min(1, A/pi). Where no camera saturates, changing
pi rescales every coverage by one constant, leaves the argmax camera
unchanged, and is therefore a strictly monotone transform of the score --
rank metrics are EXACTLY invariant. pi can only act through the saturated
set, which is 11.9% of in-frustum voxel-camera pairs at the registered
value. This measures what that 11.9% is worth at object level.
"""
from __future__ import annotations
import glob, importlib.util, json, os
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DIRS = {965: "data/pack/obs_max", 500: "data/pack/obs_px500",
        2000: "data/pack/obs_px2000"}
sp = importlib.util.spec_from_file_location("a55", os.path.join(HERE, "a55_pred_occluder.py"))
a55 = importlib.util.module_from_spec(sp); sp.loader.exec_module(a55)
md = a55.md
_W = {}


def _init(index, gt_map, order, bounds, dirs):
    _W.update(index=index, gt=gt_map, order=order, bounds=bounds, dirs=dirs)
    an = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"), allow_pickle=True)
    _W["ann"] = {k: an[k] for k in ("category", "translation", "rotation",
                                    "size", "num_lidar_pts")}
    _W["names"] = an["category_names"]


def _one(fi):
    row = _W["index"][fi]; tok = row["token"]; ann = _W["ann"]; names = _W["names"]
    gcls = np.load(_W["gt"][tok])["semantics"].astype(np.int16)
    pk = np.load(os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz"))
    pcls = pk["cls"].astype(np.int16); pconf = pk["conf"].astype(np.float32) / 255.
    O = {k: np.load(os.path.join(ROOT, d, tok + ".npy")).astype(np.float32) / 255.
         for k, d in _W["dirs"].items()}
    cam0 = row["cams"]["CAM_FRONT"]
    Re = md.qrot(np.asarray(cam0["ego2global_rotation"], np.float64))
    te = np.asarray(cam0["ego2global_translation"], np.float64)
    out = []
    for k in _W["order"][_W["bounds"][fi]:_W["bounds"][fi + 1]]:
        cid = md.CAT2OCC.get(str(names[ann["category"][k]]))
        if cid is None or cid not in md.DYN or ann["num_lidar_pts"][k] < 1:
            continue
        c = Re.T @ (ann["translation"][k].astype(np.float64) - te)
        R = Re.T @ md.qrot(np.asarray(ann["rotation"][k], np.float64))
        vx = md.box_voxels(c, ann["size"][k].astype(np.float64), R, 0.0)
        if vx is None:
            continue
        gm = gcls[vx] == cid
        if not gm.any():
            continue
        out.append((row["scene"], float(pconf[vx].max()),
                    bool(np.isin(pcls[vx], list(md.DYN)).any()),
                    *[float(O[k][vx][gm].max()) for k in sorted(_W["dirs"])]))
    return out


def main():
    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    gt_map = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}
    have = set.intersection(*[{os.path.basename(q)[:-4]
                               for q in glob.glob(os.path.join(ROOT, d, "*.npy"))}
                              for d in DIRS.values()])
    ann = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"), allow_pickle=True)
    fr = ann["frame"]; order = np.argsort(fr, kind="stable")
    bounds = np.searchsorted(fr[order], np.arange(len(index) + 1))
    fis = [i for i, r in enumerate(index) if r["token"] in have and r["token"] in gt_map]
    print(f"frames {len(fis)} (common to all pi values)")
    with Pool(initializer=_init, initargs=(index, gt_map, order, bounds, DIRS)) as pool:
        rows = [r for part in pool.map(_one, fis, chunksize=8) for r in part]
    keys = sorted(DIRS)
    scene = np.array([r[0] for r in rows]); msp = np.array([r[1] for r in rows])
    miss = ~np.array([r[2] for r in rows])
    O = {k: np.array([r[3 + i] for r in rows]) for i, k in enumerate(keys)}
    print(f"objects {len(rows):,}  scenes {len(set(scene))}  missed {100*miss.mean():.1f}%")
    base = None
    res = {}
    for k in keys:
        a = a55.auroc(-O[k], miss); res[str(k)] = a
        if k == 965:
            base = a
    for k in keys:
        print(f"  pi_full = {k:>5}   object AUROC {res[str(k)]:.4f}"
              f"   delta vs 965 {res[str(k)]-base:+.4f}")
    print(f"  MSP on the same objects      {a55.auroc(-msp, miss):.4f}")
    json.dump(dict(scope="test subsample", frames=len(fis), objects=len(rows),
                   scenes=int(len(set(scene))), auroc_by_full_px=res,
                   msp=a55.auroc(-msp, miss),
                   saturated_frac_at_965=0.11907),
              open(os.path.join(ROOT, "outputs/artifacts/a56_fullpx.json"), "w"),
              indent=1)
    print("wrote outputs/artifacts/a56_fullpx.json")


if __name__ == "__main__":
    main()
