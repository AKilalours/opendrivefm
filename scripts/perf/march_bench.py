#!/usr/bin/env python3
"""Make the ray marcher fast, and prove the answer did not change.

`build_observability_ray.py` carries this line in its own docstring:

    "Affordable because grid and cameras are both ego-fixed, so sample indices
     are identical every frame: precompute once, then gather and any() per
     frame."

That optimisation was described and never implemented. The shipped marcher
recomputes every ray direction and every voxel index on every frame, for every
camera, for all 6,019 frames. The full build takes ~100 minutes.

The precondition turns out to be stronger than the docstring claims. Measured
over all 150 scenes: camera extrinsics and intrinsics are **constant within a
scene** (max variation 0.0) and there are only **two distinct rigs** in the
whole validation split. So the ray -> voxel index table can be built twice and
reused 6,019 times.

    table[ray, step] = flat voxel index, plus a validity mask

Per frame the march becomes: gather occupancy through the table, find each
ray's first hit, mark everything up to and including it. No trigonometry, no
projection, no per-step index arithmetic.

**A speedup that changes the answer is worthless**, so `verify` asserts the new
visibility is bit-identical to the old on every frame it is given, and the
benchmark refuses to report a number until that passes.
"""
from __future__ import annotations
import argparse, glob, hashlib, importlib.util, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location(
    "build_obs_ray", os.path.join(ROOT, "scripts/eval/build_observability_ray.py"))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
FREE = 17
_TABLES = {}


def rig_key(cam):
    h = hashlib.sha1()
    for k in ("sensor2ego_translation", "sensor2ego_rotation"):
        h.update(np.asarray(cam[k], np.float64).tobytes())
    K = cam.get("cam_intrinsic", cam.get("intrinsic"))
    h.update(np.asarray(K, np.float64).tobytes())
    h.update(str((cam.get("width", 1600), cam.get("height", 900))).encode())
    return h.hexdigest()[:16]


def build_table(cam, stride=4, near=0.5, far=62.0):
    """Precompute, once per rig, the flat voxel index every ray visits."""
    R = B.M.quat_to_rot(cam["sensor2ego_rotation"])
    t = np.asarray(cam["sensor2ego_translation"], np.float32)
    K = np.asarray(cam.get("cam_intrinsic", cam.get("intrinsic")), np.float32)
    W, H = cam.get("width", 1600), cam.get("height", 900)
    uu, vv = np.meshgrid(np.arange(0, W, stride, dtype=np.float32) + 0.5,
                         np.arange(0, H, stride, dtype=np.float32) + 0.5,
                         indexing="xy")
    d = np.stack([(uu.ravel() - K[0, 2]) / K[0, 0],
                  (vv.ravel() - K[1, 2]) / K[1, 1],
                  np.ones(uu.size, np.float32)], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d @ R.T
    steps = np.arange(near, far, B.STEP, dtype=np.float32)
    pts = t[None, None, :] + d[:, None, :] * steps[None, :, None]
    idx, ok = B.to_index(pts.reshape(-1, 3))
    nr, ns = d.shape[0], steps.size
    return idx.reshape(nr, ns).astype(np.int32), ok.reshape(nr, ns)


def march_fast(occ_flat, table, valid):
    """Same semantics as B.march_visibility: mark every cell a ray reaches, up
    to and including its first occupied one."""
    hit = np.zeros(table.shape, bool)
    np.take(occ_flat, table, out=hit)          # gather; invalid entries are junk
    hit &= valid                                # ...and are masked out here
    any_hit = hit.any(1)
    first = np.where(any_hit, hit.argmax(1), table.shape[1] - 1)
    reached = valid & (np.arange(table.shape[1])[None, :] <= first[:, None])
    vis = np.zeros(occ_flat.size, bool)
    vis[table[reached]] = True
    return vis


def get_tables(cam):
    k = rig_key(cam)
    if k not in _TABLES:
        _TABLES[k] = build_table(cam)
    return _TABLES[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", default="outputs/artifacts/march_bench.json")
    a = ap.parse_args()

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    calib = json.load(open(os.path.join(ROOT, "data/pack/calib.json")))
    gt = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}
    rng = np.random.default_rng(31)
    rows = [r for r in index if r["token"] in gt and r["token"] in calib]
    rows = [rows[i] for i in rng.choice(len(rows), a.frames, replace=False)]

    rigs = {rig_key(c) for r in rows for c in calib[r["token"]]["cams"].values()}
    print(f"{len(rows)} frames | {len(rigs)} distinct camera configurations "
          f"across the sampled frames")

    t0 = time.perf_counter()
    for r in rows:
        for c in B.CAMS:
            cam = calib[r["token"]]["cams"].get(c)
            if cam: get_tables(cam)
    t_build = time.perf_counter() - t0
    nbytes = sum(t.nbytes + v.nbytes for t, v in _TABLES.values())
    print(f"table build: {t_build:.2f} s once, {nbytes/1e6:.0f} MB resident "
          f"for {len(_TABLES)} camera tables")

    print("\nverifying the fast path is BIT-IDENTICAL before timing anything")
    t_old = t_new = 0.0
    mism = 0
    for n, r in enumerate(rows):
        sem = np.load(gt[r["token"]])["semantics"]
        occ_flat = (sem.reshape(-1) != FREE)
        for c in B.CAMS:
            cam = calib[r["token"]]["cams"].get(c)
            if cam is None: continue
            W, H = cam.get("width", 1600), cam.get("height", 900)
            s = time.perf_counter()
            v_old = B.march_visibility(occ_flat, cam, W, H)
            t_old += time.perf_counter() - s
            tab, val = get_tables(cam)
            s = time.perf_counter()
            v_new = march_fast(occ_flat, tab, val)
            t_new += time.perf_counter() - s
            d = int((v_old != v_new).sum())
            mism += d
        if (n + 1) % 4 == 0:
            print(f"  {n+1}/{len(rows)} frames, mismatched voxels so far {mism}",
                  flush=True)

    ok = mism == 0
    print(f"\n{'IDENTICAL' if ok else 'MISMATCH'}: {mism} differing voxels "
          f"out of {len(rows)*6*640000:,} compared")
    if not ok:
        print("refusing to report a speedup on a changed answer")
        raise SystemExit(1)

    nf = len(rows)
    print("=" * 62)
    print(f"{'implementation':<34}{'s / frame':>13}{'speedup':>13}")
    print("-" * 62)
    print(f"{'shipped: per-frame projection':<34}{t_old/nf:>13.3f}{'1.00x':>13}")
    print(f"{'precomputed ray-to-voxel table':<34}{t_new/nf:>13.3f}"
          f"{t_old/max(t_new,1e-9):>12.2f}x")
    print("-" * 62)
    full = 6019
    print(f"full 6,019-frame build: {t_old/nf*full/60:.0f} min  ->  "
          f"{(t_new/nf*full + t_build)/60:.0f} min  (table build included)")
    print("=" * 62)
    json.dump(dict(frames=nf, cameras_per_frame=6, identical=ok,
                   mismatched_voxels=mism,
                   table_build_s=t_build, table_mb=nbytes / 1e6,
                   s_per_frame_shipped=t_old / nf, s_per_frame_fast=t_new / nf,
                   speedup=t_old / max(t_new, 1e-9),
                   full_build_min_shipped=t_old / nf * full / 60,
                   full_build_min_fast=(t_new / nf * full + t_build) / 60),
              open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
