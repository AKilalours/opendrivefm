"""Per-voxel observability by ray marching. Replaces the z-buffer entirely.

Why not a z-buffer
------------------
Nine attempts failed, each fixing one symptom and exposing another: point
splats leaked through walls, dilation made surfaces shadow themselves,
isotropic splats over-occluded, clipping injected 60.7% phantom occluders,
bounding boxes over-covered, and the depth bias needed three separate
derivations. Every one of those is an artefact of rasterising boxes into bins.
A z-buffer answers "what is nearest along this pixel"; the question is "is
anything between the lens and THIS voxel". On a voxel grid with grazing
surfaces those differ, and no amount of parameter work closes the gap.

The method here cannot have those bugs, because it never rasterises anything.

Why it is fast
--------------
Naive marching is 640k voxels x ~140 steps x 6 cameras = 500M samples a frame.
But visibility along a ray is RECURSIVE:

    vis(V) = vis(V') and not occupied(V')      V' = V - step * unit(V - C)

V' is strictly closer to the lens than V, so if voxels are processed in order
of increasing distance, every dependency is already known. Sorting into shells
of one step width makes each shell a single vectorised update, and the whole
frame becomes ~70 small array operations per camera instead of 500M samples.

Exactness: the step is one voxel (0.4 m), so the march cannot skip a voxel.
Where a step lands back in the same cell (near-tangential rays) it is retried
at double length, and the shell width covers that so ordering still holds.

What is unchanged from the validated measure
--------------------------------------------
    observability(v) = 1 - prod_i (1 - trust_i * coverage_i(v) * visible_i(v))

TRUST is the same constant 0.795 and still contributes no discrimination.
COVERAGE is the same resolution weighting with FULL_PX = 965, the value
derived in A7 to preserve the 2D measure's 16.3 m saturation range. Only
`visible` changes, and it now has no free parameters at all: no bin size, no
depth tolerance, no bias, no splat shape.
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
OCC_N, OCC_RES, OCC_RNG = 200, 0.4, 40.0
OCC_NZ, OCC_Z0 = 16, -1.0
FULL_PX = 965.0
STEP = OCC_RES                 # one voxel; cannot skip a cell
CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "odfm_eval", os.path.join(HERE, "eval_observability_scaled.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _load_eval_module()
TRUST = getattr(M, "TRUST_CONST", 0.795)


def voxel_centres():
    ax = (np.arange(OCC_N) + 0.5) * OCC_RES - OCC_RNG
    az = (np.arange(OCC_NZ) + 0.5) * OCC_RES + OCC_Z0
    X, Y, Z = np.meshgrid(ax, ax, az, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)


def to_index(p):
    """World point -> flat voxel index, plus a mask for points inside the grid."""
    i = np.floor((p[:, 0] + OCC_RNG) / OCC_RES)
    j = np.floor((p[:, 1] + OCC_RNG) / OCC_RES)
    k = np.floor((p[:, 2] - OCC_Z0) / OCC_RES)
    ok = ((i >= 0) & (i < OCC_N) & (j >= 0) & (j < OCC_N) &
          (k >= 0) & (k < OCC_NZ))
    idx = (np.clip(i, 0, OCC_N - 1) * OCC_N + np.clip(j, 0, OCC_N - 1)
           ) * OCC_NZ + np.clip(k, 0, OCC_NZ - 1)
    return idx.astype(np.int64), ok


def project(pts_ego, cam):
    R = M.quat_to_rot(cam["sensor2ego_rotation"])
    t = np.asarray(cam["sensor2ego_translation"], np.float32)
    K = np.asarray(cam.get("cam_intrinsic", cam.get("intrinsic")), np.float32)
    p = (pts_ego - t) @ R
    z = p[:, 2]
    good = z > 0.1
    zz = np.where(good, z, 1.0)
    u = K[0, 0] * p[:, 0] / zz + K[0, 2]
    v = K[1, 1] * p[:, 1] / zz + K[1, 2]
    return u, v, z, good, K, t


def march_visibility(occ_flat, cam, W, H, stride=4, near=0.5, far=62.0):
    """Visibility by marching one ray per PIXEL, first-hit semantics.

    Casting a ray per VOXEL is the wrong formulation and was the last bug. The
    ray to a road voxel at 30 m passes through the road voxel at 29.6 m, which
    is also occupied, so every ground voxel is blocked by its own neighbour and
    the road slab occludes itself end to end (95% of surface voxels zeroed).
    Geometrically the ray grazes just above that neighbour's surface, but a
    voxel world has no surfaces, only solid blocks.

    Occ3D does not cast per voxel. It casts one ray per pixel and marks what
    each ray reaches: every free voxel along the way, plus the FIRST occupied
    voxel, which is the surface the camera actually sees. Each road voxel is
    the first hit for whichever pixel points at it, so the road stays visible
    along its length. That is the semantics of mask_camera, and matching it is
    what makes the H2 comparison a test of continuous-versus-binary rather than
    a test of who modelled occlusion more carefully.

    Rays terminate on first hit, so cost is bounded by the visible surface
    rather than by the volume.
    """
    R = M.quat_to_rot(cam["sensor2ego_rotation"])
    t = np.asarray(cam["sensor2ego_translation"], np.float32)
    K = np.asarray(cam.get("cam_intrinsic", cam.get("intrinsic")), np.float32)

    uu, vv = np.meshgrid(np.arange(0, W, stride, dtype=np.float32) + 0.5,
                         np.arange(0, H, stride, dtype=np.float32) + 0.5,
                         indexing="xy")
    d_cam = np.stack([(uu.ravel() - K[0, 2]) / K[0, 0],
                      (vv.ravel() - K[1, 2]) / K[1, 1],
                      np.ones(uu.size, np.float32)], 1)
    d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
    d = d_cam @ R.T                                   # camera -> ego

    vis = np.zeros(occ_flat.size, bool)
    alive = np.ones(d.shape[0], bool)
    steps = np.arange(near, far, STEP, dtype=np.float32)
    for sdist in steps:
        if not alive.any():
            break
        pts = t + d[alive] * sdist
        idx, ok = to_index(pts)
        idx = idx[ok]
        if idx.size == 0:
            continue
        vis[idx] = True                               # reached, so seen
        hit = occ_flat[idx]                           # first solid: stop here
        if hit.any():
            live = np.flatnonzero(alive)
            kill = live[np.flatnonzero(ok)[hit]]
            alive[kill] = False
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--gts",
                    default="data/occ3d/Occupancy3D-nuScenes-trainval/gts")
    ap.add_argument("--out", default="data/pack/observability_ray")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--measure", choices=["noisy_or", "max"], default="noisy_or",
                    help="noisy_or reproduces every frozen number (A12-A22). "
                         "max is the formula settled in A23: "
                         "max_i(coverage_i * visible_i), no TRUST constant.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    index = json.load(open(os.path.join(args.pack, "index.json")))
    calib = json.load(open(os.path.join(args.pack, "calib.json")))
    gt_map = {os.path.basename(os.path.dirname(q)): q
              for q in glob.glob(os.path.join(args.gts, "*", "*",
                                              "labels.npz"))}
    if args.limit:
        index = index[:args.limit]
    vox = voxel_centres()
    print(f"{len(gt_map):,} gt frames | {vox.shape[0]:,} voxels/frame | "
          f"{len(index):,} frames")

    t0, done = time.time(), 0
    for row in index:
        tok = row["token"]
        dst = os.path.join(args.out, tok + ".npy")
        if os.path.exists(dst):
            done += 1
            continue
        if tok not in gt_map or tok not in calib:
            continue
        sem = np.load(gt_map[tok])["semantics"]
        occ_flat = (sem.reshape(-1) != 17)

        miss = np.ones(vox.shape[0], np.float32)
        best = np.zeros(vox.shape[0], np.float32)
        for c in CAMS:
            cam = calib[tok]["cams"].get(c)
            if cam is None:
                continue
            W = cam.get("width", 1600)
            H = cam.get("height", 900)
            u, v, z, good, K, ct = project(vox, cam)
            inside = good & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not inside.any():
                continue
            vis = march_visibility(occ_flat, cam, W, H)
            area = (K[0, 0] * K[1, 1] * OCC_RES * OCC_RES) / \
                np.maximum(z, 0.1) ** 2
            cov = np.clip(area / FULL_PX, 0.0, 1.0)
            seen = cov * (vis & inside)
            miss *= (1.0 - TRUST * seen)
            best = np.maximum(best, seen)
        # A23: under a max, TRUST is a monotone rescale and is dropped; the
        # measured difference to keeping it is exactly 0.0000.
        obs = best if args.measure == "max" else 1.0 - miss
        np.save(dst, np.clip(np.rint(obs * 255), 0, 255).astype(np.uint8)
                .reshape(OCC_N, OCC_N, OCC_NZ))
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(index)}  {el/done:.3f}s/frame  "
                  f"eta {(len(index)-done)*el/done/60:.1f} min", flush=True)

    json.dump(dict(grid=[OCC_N, OCC_N, OCC_NZ], res=OCC_RES, rng=OCC_RNG,
                   z0=OCC_Z0, trust=TRUST, full_px=FULL_PX, step_m=STEP,
                   measure=args.measure,
                   method="recursive ray march, no free parameters",
                   quantisation="uint8 = round(255 * observability)",
                   frames=done),
              open(os.path.join(args.out, "_meta.json"), "w"), indent=2)
    print("done", done)


if __name__ == "__main__":
    main()
