#!/usr/bin/env python3
"""Where should the seventh camera go? A rig recommendation from fleet data.

A21 asked what the rig loses when a camera fails. This asks the design question
instead: given the observability field over real driving, where does an
ADDITIONAL camera buy the most, and is raising or widening an existing one
cheaper than adding hardware?

The measure makes this answerable analytically. A candidate camera is just
another term in the max: place it, march it, recombine. No retraining, no new
data, no simulator -- the same ray marcher that produced every frozen map.

Candidates are chosen from findings, not from a grid sweep:

  bumper_low   A30 measured a near-field band of 1.2 m median (12.8 m at p90)
               that NO camera can verify, because the cameras sit 1.5 m up.
               A low bumper camera is the direct test of that finding.
  front_wide   A21 showed five 65-degree cameras cannot close a 360 ring and
               the 89-degree back camera carries the gap alone. Does widening
               the front help, or is the gap elsewhere?
  roof_high    Height buys sightline over parked cars. A29 found vehicles cause
               45.4% of hidden drivable surface. Does 2.4 m fix that?
  side_left / side_right   fill the 65-degree seams between front and back.
  rear_tele    narrow, long-range rear, for the merge case.

Scored on what A21 and A29 established matters: obstacle columns with zero
coverage, mean observability over obstacles, and drivable surface recovered.
"""
from __future__ import annotations
import argparse, glob, importlib.util, json, os, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location(
    "build_obs_ray", os.path.join(HERE, "build_observability_ray.py"))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)

FREE, RES, RNG, NZ, Z0, N = 17, 0.4, 40.0, 16, -1.0, 200
OBST = np.arange(0, 11)
DRIVABLE = 11
_G = {}


def rot_zy(yaw_deg, pitch_deg):
    """Camera-to-ego rotation for a camera looking along +x, yawed then pitched.

    The optical axis is +z in camera frame, x right, y down -- the nuScenes
    convention -- so the base is the transform that sends camera +z to ego +x.
    """
    base = np.array([[0., 0., 1.],
                     [-1., 0., 0.],
                     [0., -1., 0.]])
    y = np.radians(yaw_deg); p = np.radians(pitch_deg)
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    return Rz @ Ry @ base


def make_cam(t, yaw, pitch, hfov_deg, W=1600, H=900):
    f = W / (2 * np.tan(np.radians(hfov_deg) / 2))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
    return dict(R=rot_zy(yaw, pitch), t=np.asarray(t, np.float32), K=K, W=W, H=H)


CANDIDATES = {
    "bumper_low":  make_cam((3.4, 0.0, 0.55), 0, 8, 100),
    "front_wide":  make_cam((1.7, 0.0, 1.50), 0, 0, 120),
    "roof_high":   make_cam((1.0, 0.0, 2.40), 0, 10, 90),
    "side_left":   make_cam((1.5, 0.9, 1.50), 90, 5, 90),
    "side_right":  make_cam((1.5, -0.9, 1.50), -90, 5, 90),
    "rear_tele":   make_cam((0.0, 0.0, 1.60), 180, 0, 45),
}


def _init(obs_dir):
    _G["obs"] = obs_dir
    _G["gt"] = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}
    _G["calib"] = json.load(open(os.path.join(ROOT, "data/pack/calib.json")))
    _G["vox"] = B.voxel_centres()


def factor_for(cam_R, cam_t, K, W, H, occ_flat, vox):
    """coverage * visible for an arbitrary camera pose. Same terms as A23."""
    p = (vox - cam_t) @ cam_R
    z = p[:, 2]
    good = z > 0.1
    zz = np.where(good, z, 1.0)
    u = K[0, 0] * p[:, 0] / zz + K[0, 2]
    v = K[1, 1] * p[:, 1] / zz + K[1, 2]
    inside = good & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inside.any():
        return np.zeros(vox.shape[0], np.float32)
    fake = dict(sensor2ego_rotation=None, sensor2ego_translation=cam_t,
                cam_intrinsic=K)
    vis = march_generic(occ_flat, cam_R, cam_t, K, W, H)
    area = (K[0, 0] * K[1, 1] * B.OCC_RES ** 2) / np.maximum(z, 0.1) ** 2
    cov = np.clip(area / B.FULL_PX, 0.0, 1.0) * inside
    return (cov * (vis & inside)).astype(np.float32)


def march_generic(occ_flat, R, t, K, W, H, stride=4, near=0.5, far=62.0):
    uu, vv = np.meshgrid(np.arange(0, W, stride, dtype=np.float32) + 0.5,
                         np.arange(0, H, stride, dtype=np.float32) + 0.5,
                         indexing="xy")
    d = np.stack([(uu.ravel() - K[0, 2]) / K[0, 0],
                  (vv.ravel() - K[1, 2]) / K[1, 1],
                  np.ones(uu.size, np.float32)], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d @ R.T
    vis = np.zeros(occ_flat.size, bool)
    alive = np.ones(d.shape[0], bool)
    for s in np.arange(near, far, B.STEP, dtype=np.float32):
        if not alive.any(): break
        idx, ok = B.to_index(t + d[alive] * s)
        idx = idx[ok]
        if idx.size == 0: continue
        vis[idx] = True
        hit = occ_flat[idx]
        if hit.any():
            live = np.flatnonzero(alive)
            alive[live[np.flatnonzero(ok)[hit]]] = False
    return vis


def _one(row):
    tok, scene = row
    g = np.load(_G["gt"][tok])
    gcls = g["semantics"].astype(np.int16)
    occ_flat = (gcls.reshape(-1) != FREE)
    vox = _G["vox"]

    base = np.load(os.path.join(ROOT, _G["obs"], tok + ".npy")).astype(np.float32) / 255.0
    obst = np.isin(gcls, OBST)
    road = gcls == DRIVABLE
    out = dict(token=tok, scene=scene,
               obst_n=int(obst.sum()),
               obst_dark=int((obst & (base <= 0)).sum()),
               obst_mean=float(base[obst].mean()) if obst.any() else 0.0,
               road_dark=int((road & (base <= 0)).sum()),
               cand={})
    for name, c in CANDIDATES.items():
        f = factor_for(c["R"], c["t"], c["K"], c["W"], c["H"], occ_flat, vox)
        new = np.maximum(base, f.reshape(base.shape))      # A23: max, not noisy-OR
        out["cand"][name] = dict(
            obst_dark=int((obst & (new <= 0)).sum()),
            obst_mean=float(new[obst].mean()) if obst.any() else 0.0,
            road_dark=int((road & (new <= 0)).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="data/pack/obs_max")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--chunk", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--cache", default="outputs/artifacts/placement_cache.jsonl")
    ap.add_argument("--out", default="outputs/artifacts/camera_placement.json")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    cache = os.path.join(ROOT, a.cache)
    os.makedirs(os.path.dirname(cache), exist_ok=True)

    if not a.report:
        index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
        gt = {os.path.basename(os.path.dirname(q)) for q in glob.glob(
            os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                         "*", "*", "labels.npz"))}
        rows = [(r["token"], r["scene"]) for r in index if r["token"] in gt
                and os.path.exists(os.path.join(ROOT, a.obs, r["token"] + ".npy"))]
        rng = np.random.default_rng(a.seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), min(a.frames, len(rows)),
                                                   replace=False))]
        done = set()
        if os.path.exists(cache):
            done = {json.loads(l)["token"] for l in open(cache)}
        todo = [r for r in rows if r[0] not in done][:a.chunk]
        print(f"{len(rows)} frames | cached {len(done)} | this run {len(todo)} | "
              f"{len(CANDIDATES)} candidates | {os.cpu_count()} cores", flush=True)
        if not todo:
            print("complete -- run --report"); return
        t0 = time.time()
        with open(cache, "a") as fh, Pool(max(os.cpu_count() - 1, 1),
                                          initializer=_init, initargs=(a.obs,)) as pool:
            for n, r in enumerate(pool.imap_unordered(_one, todo, chunksize=1)):
                fh.write(json.dumps(r) + "\n"); fh.flush()
                if (n + 1) % 10 == 0:
                    el = time.time() - t0
                    print(f"  {n+1}/{len(todo)}  {el/(n+1):.2f}s/frame  "
                          f"eta {(len(todo)-n-1)*el/(n+1)/60:.1f} min", flush=True)
        left = len(rows) - len(done) - len(todo)
        print(f"chunk done, {left} remain" if left > 0 else
              "complete -- run --report", flush=True)
        return

    R = [json.loads(l) for l in open(cache)]
    bn = sum(r["obst_n"] for r in R)
    bd = sum(r["obst_dark"] for r in R)
    brd = sum(r["road_dark"] for r in R)
    bm = float(np.mean([r["obst_mean"] for r in R]))
    print(f"\n{len(R)} frames | baseline six-camera rig")
    print(f"  obstacle voxels with NO coverage: {bd:,} of {bn:,} "
          f"({100*bd/bn:.2f}%)   mean observability on obstacles {bm:.4f}")
    print("=" * 88)
    print(f"{'add a 7th camera':<14}{'obstacles recovered':>21}{'road recovered':>17}"
          f"{'mean obs':>12}{'gain':>10}")
    print("-" * 88)
    res = {}
    rows_out = []
    for name in CANDIDATES:
        cd = sum(r["cand"][name]["obst_dark"] for r in R)
        crd = sum(r["cand"][name]["road_dark"] for r in R)
        cm = float(np.mean([r["cand"][name]["obst_mean"] for r in R]))
        rec = bd - cd
        rows_out.append((rec, name, cd, crd, cm))
    for rec, name, cd, crd, cm in sorted(rows_out, reverse=True):
        print(f"{name:<14}{rec:>13,} {100*rec/max(bd,1):>6.1f}%"
              f"{brd-crd:>12,} {100*(brd-crd)/max(brd,1):>4.1f}%"
              f"{cm:>12.4f}{cm-bm:>+10.4f}")
        res[name] = dict(obst_dark_after=cd, obstacles_recovered=int(rec),
                         obstacles_recovered_frac=float(rec / max(bd, 1)),
                         road_recovered=int(brd - crd),
                         road_recovered_frac=float((brd - crd) / max(brd, 1)),
                         mean_obs_after=cm, mean_obs_gain=cm - bm)
    print("=" * 88)
    json.dump(dict(frames=len(R), baseline=dict(
        obst_n=bn, obst_dark=bd, obst_dark_frac=bd / bn, road_dark=brd,
        mean_obs=bm), candidates=res),
        open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
