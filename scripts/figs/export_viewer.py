#!/usr/bin/env python3
"""Export one scene for the interactive web viewer.

Payload has to fit a browser, and a 200x200x16 grid at 41 frames does not.
Three reductions, none of which change what is shown:

  surface only    an occupied voxel with six occupied neighbours is never
                  visible; dropping interiors removes 63% of them
  ground as image the drivable surface, sidewalk and terrain are a near-flat
                  sheet, so they ship as a 200x200 PNG heightmap plus class
                  map rather than as voxels
  PNG container   remaining voxels are packed into RGBA pixels sorted by
                  index, so PNG's own filters delta-encode them and the
                  browser decodes natively, with no custom decompressor

    python3 scripts/figs/export_viewer.py --scene scene-0553
"""
import argparse, glob, importlib.util, json, os, sys
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts/eval"))
_s = importlib.util.spec_from_file_location(
    "co", os.path.join(ROOT, "scripts/eval/combined_object.py"))
co = importlib.util.module_from_spec(_s); _s.loader.exec_module(co)

FREE, GROUND, TAU = 17, (11, 12, 13, 14), 0.15
N, NZ = 200, 16


def surface(occ):
    p = np.pad(occ, 1, constant_values=False)
    inner = (p[2:, 1:-1, 1:-1] & p[:-2, 1:-1, 1:-1] & p[1:-1, 2:, 1:-1] &
             p[1:-1, :-2, 1:-1] & p[1:-1, 1:-1, 2:] & p[1:-1, 1:-1, :-2])
    return occ & ~inner


def pack_voxels(cls, obs, path):
    occ = cls != FREE
    keep = surface(occ) & ~np.isin(cls, GROUND)
    i, j, k = np.nonzero(keep)
    idx = (i.astype(np.uint32) * N + j) * NZ + k          # < 2^20
    o = np.argsort(idx)
    idx = idx[o]; i, j, k = i[o], j[o], k[o]
    lab = cls[i, j, k].astype(np.uint8)
    ob = np.clip(obs[i, j, k] * 7.0, 0, 7).astype(np.uint8)
    n = len(idx)
    wpx = 512
    hpx = int(np.ceil(n / wpx))
    buf = np.zeros((hpx * wpx, 4), np.uint8)
    buf[:n, 0] = (idx & 0xFF)
    buf[:n, 1] = ((idx >> 8) & 0xFF)
    buf[:n, 2] = ((idx >> 16) & 0xFF)
    buf[:n, 3] = (lab << 3) | ob
    Image.fromarray(buf.reshape(hpx, wpx, 4), "RGBA").save(path, optimize=True)
    return n


def pack_ground(cls, obs, pg, po):
    """Top ground class per column, and O(v) max over height, as two PNGs."""
    g = np.full((N, N), 255, np.uint8)
    for k in range(NZ):
        m = np.isin(cls[:, :, k], GROUND)
        g[m] = cls[:, :, k][m]
    Image.fromarray(g, "L").save(pg, optimize=True)
    o = np.clip(obs.max(-1) * 255, 0, 255).astype(np.uint8)
    free = cls == FREE
    bad = (free & (obs < TAU)).any(-1) & ~(free & (obs >= TAU)).any(-1)
    rgba = np.dstack([o, np.where(bad, 255, 0).astype(np.uint8),
                      np.zeros_like(o), np.full_like(o, 255)])
    Image.fromarray(rgba, "RGBA").save(po, optimize=True)
    return float(bad.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene-0553")
    ap.add_argument("--out", default="docs/viewer/data")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    ann = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"), allow_pickle=True)
    names = ann["category_names"]
    fr = ann["frame"]; order = np.argsort(fr, kind="stable")
    bounds = np.searchsorted(fr[order], np.arange(len(index) + 1))
    gt_map = {os.path.basename(os.path.dirname(p)): p for p in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}
    rows = sorted([(i, r) for i, r in enumerate(index) if r["scene"] == a.scene],
                  key=lambda t: t[1]["timestamp"])

    meta = {"scene": a.scene, "res": 0.4, "rng": 40.0, "z0": -1.0,
            "n": N, "nz": NZ, "frames": []}
    nice = {"rigid": "bus", "bendy": "bus", "adult": "pedestrian",
            "child": "pedestrian", "police_officer": "pedestrian",
            "construction_worker": "pedestrian", "wheelchair": "pedestrian",
            "stroller": "pedestrian", "personal_mobility": "pedestrian"}
    prev = None
    for n, (fi, row) in enumerate(rows):
        tok = row["token"]
        pp = os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz")
        op = os.path.join(ROOT, "data/pack/obs_max", tok + ".npy")
        if not (os.path.exists(pp) and os.path.exists(op) and tok in gt_map):
            continue
        cls = np.load(pp)["cls"].astype(np.int16)
        obs = np.load(op).astype(np.float32) / 255.0
        gcls = np.load(gt_map[tok])["semantics"].astype(np.int16)
        nv = pack_voxels(cls, obs, os.path.join(out, f"v{n:03d}.png"))
        uf = pack_ground(cls, obs, os.path.join(out, f"g{n:03d}.png"),
                         os.path.join(out, f"o{n:03d}.png"))

        cam = row["cams"]["CAM_FRONT"]
        Re = co.qrot(np.asarray(cam["ego2global_rotation"], np.float64))
        te = np.asarray(cam["ego2global_translation"], np.float64)
        boxes = []
        for k in order[bounds[fi]:bounds[fi+1]]:
            cat = str(names[ann["category"][k]])
            cid = co.CAT2OCC.get(cat)
            if cid is None or cid not in co.DYN or ann["num_lidar_pts"][k] < 1:
                continue
            c = Re.T @ (ann["translation"][k].astype(np.float64) - te)
            R = Re.T @ co.qrot(np.asarray(ann["rotation"][k], np.float64))
            vx = co.box_voxels(c, ann["size"][k].astype(np.float64), R, 0.0)
            if vx is None or not (gcls[vx] == cid).any():
                continue
            leaf = cat.split(".")[-1]
            boxes.append({"c": [round(float(x), 2) for x in c],
                          "s": [round(float(x), 2) for x in ann["size"][k]],
                          "y": round(float(np.arctan2(R[1, 0], R[0, 0])), 3),
                          "d": bool(np.isin(cls[vx], list(co.DYN)).any()),
                          "l": nice.get(leaf, leaf)})
        p1 = np.asarray(cam["ego2global_translation"], np.float64)
        spd = 0.0 if prev is None else float(np.linalg.norm(p1 - prev[0]) /
                                             max((row["timestamp"]-prev[1])/1e6, 1e-3))
        prev = (p1, row["timestamp"])
        meta["frames"].append({
            "i": n, "voxels": nv, "unverified": round(uf, 4),
            "meanObs": round(float(obs.max(-1).mean()), 4),
            "speed": round(spd, 2),
            "missed": sum(1 for b in boxes if not b["d"]),
            "cam": row["cams"]["CAM_FRONT"]["filename"], "boxes": boxes})
        print(f"  frame {n:02d}  {nv:6d} voxels  unverified {uf*100:4.1f}%", flush=True)

    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    tot = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"\n{len(meta['frames'])} frames | payload {tot/1e6:.1f} MB")


if __name__ == "__main__":
    main()
