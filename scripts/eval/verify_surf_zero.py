#!/usr/bin/env python3
"""Is the first-hit zeroing correct, or is it a bug?

The one acceptance gate still failing is surf_zero: 70.4% of ground-truth
occupied voxels that Occ3D marks camera-visible get observability exactly 0
under the ray marcher.  Two possible causes:

  (a) correct.  A camera sees the front face of a thing, not its interior or
      its far side.  Occ3D's mask_camera is more permissive than that, so a
      strict first-hit measure legitimately zeroes most of them.
  (b) a bug in the marcher.

This script decides it WITHOUT using the marcher.  For every zeroed voxel it
walks the straight segment from each camera's optical centre to that voxel and
asks the ground-truth occupancy volume whether anything sits strictly in
between.  Independent geometry, independent code path.

Each zeroed voxel lands in exactly one bucket:

  out_of_fov    no camera has it inside the image at positive depth
  occluded      in at least one camera's image, and EVERY such camera is
                blocked by a ground-truth occupied voxel before reaching it
  unexplained   some camera has a clear line of sight to it  ->  a real bug

A correct measure puts ~everything in out_of_fov + occluded.
"""
import argparse, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FREE, RES, RNG, NZ = 17, 0.4, 40.0, 16
Z0 = -1.0
NSTEP = 420          # samples along each lens->voxel segment; >= dist/0.175 m


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], np.float64)


def to_index(P):
    """ego metres -> (i,j,k) voxel index, plus in-bounds mask."""
    i = np.floor((P[:, 0] + RNG) / RES).astype(np.int32)
    j = np.floor((P[:, 1] + RNG) / RES).astype(np.int32)
    k = np.floor((P[:, 2] - Z0) / RES).astype(np.int32)
    ok = ((i >= 0) & (i < 200) & (j >= 0) & (j < 200) & (k >= 0) & (k < NZ))
    return i, j, k, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--per-frame", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/artifacts/surf_zero_verdict.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    idx = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    pick = rng.choice(len(idx), size=min(a.frames, len(idx)), replace=False)

    tot = dict(zeroed=0, out_of_fov=0, occluded=0, unexplained=0, tested=0)
    per_frame = []
    unexp_depth = []

    for n, fi in enumerate(pick):
        r = idx[int(fi)]
        tok, scene = r["token"], r["scene"]
        gp = os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                          scene, tok, "labels.npz")
        op = os.path.join(ROOT, "data/pack/obs_ray", tok + ".npy")
        if not (os.path.exists(gp) and os.path.exists(op)):
            continue
        g = np.load(gp)
        gcls, mcam = g["semantics"].astype(np.int16), g["mask_camera"].astype(bool)
        obs = np.load(op)
        occ = gcls != FREE

        target = mcam & occ & (obs == 0)
        ii, jj, kk = np.nonzero(target)
        tot["zeroed"] += ii.size
        if ii.size == 0:
            continue
        if ii.size > a.per_frame:
            sel = rng.choice(ii.size, a.per_frame, replace=False)
            ii, jj, kk = ii[sel], jj[sel], kk[sel]
        M = ii.size
        tot["tested"] += M

        P = np.stack([(ii + .5) * RES - RNG, (jj + .5) * RES - RNG,
                      (kk + .5) * RES + Z0], 1).astype(np.float64)

        in_fov = np.zeros(M, bool)     # seen by at least one camera's image
        clear  = np.zeros(M, bool)     # at least one camera reaches it unblocked
        best_clear_d = np.full(M, np.inf)

        for cam in r["cams"].values():
            R = quat_to_rot(np.asarray(cam["sensor2ego_rotation"], np.float64))
            t = np.asarray(cam["sensor2ego_translation"], np.float64)
            K = np.asarray(cam.get("cam_intrinsic", cam.get("intrinsic")), np.float64)
            W, H = cam["width"], cam["height"]

            Pc = (P - t) @ R                       # ego -> camera frame
            z = Pc[:, 2]
            front = z > 0.1
            u = np.where(front, K[0, 0] * Pc[:, 0] / np.where(front, z, 1) + K[0, 2], -1)
            v = np.where(front, K[1, 1] * Pc[:, 1] / np.where(front, z, 1) + K[1, 2], -1)
            vis = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            in_fov |= vis

            cand = np.flatnonzero(vis & ~clear)
            if cand.size == 0:
                continue
            d = np.linalg.norm(P[cand] - t, axis=1)
            # sample strictly between the lens and the target, stopping one
            # voxel short so the target itself never counts as its own occluder
            tmax = np.clip(1.0 - (RES * 1.05) / np.maximum(d, 1e-6), 0.0, 1.0)
            frac = np.linspace(0.0, 1.0, NSTEP)[None, :] * tmax[:, None]
            near = np.clip(0.6 / np.maximum(d, 1e-6), 0, 1)[:, None]
            frac = np.maximum(frac, near)          # skip the lens housing
            S = t[None, None, :] + (P[cand] - t)[:, None, :] * frac[:, :, None]
            flat = S.reshape(-1, 3)
            si, sj, sk, sok = to_index(flat)
            hit = np.zeros(flat.shape[0], bool)
            hit[sok] = occ[si[sok], sj[sok], sk[sok]]
            blocked = hit.reshape(cand.size, NSTEP).any(1)
            ok = cand[~blocked]
            clear[ok] = True
            best_clear_d[ok] = np.minimum(best_clear_d[ok], d[~blocked])

        out = ~in_fov
        occl = in_fov & ~clear
        unexp = clear
        tot["out_of_fov"] += int(out.sum())
        tot["occluded"] += int(occl.sum())
        tot["unexplained"] += int(unexp.sum())
        if unexp.any():
            unexp_depth.append(best_clear_d[unexp])
        per_frame.append(dict(token=tok, tested=int(M), out=int(out.sum()),
                              occl=int(occl.sum()), unexp=int(unexp.sum())))
        print(f"{n+1:3d}/{len(pick)} {tok} tested {M:5d}  "
              f"out_of_fov {100*out.mean():5.1f}%  occluded {100*occl.mean():5.1f}%  "
              f"unexplained {100*unexp.mean():5.1f}%", flush=True)

    T = max(tot["tested"], 1)
    print("\n" + "=" * 66)
    print(f"zeroed voxels found      {tot['zeroed']:,}")
    print(f"tested                   {tot['tested']:,}")
    print(f"out of every FOV         {tot['out_of_fov']:,}  ({100*tot['out_of_fov']/T:.2f}%)")
    print(f"occluded by GT geometry  {tot['occluded']:,}  ({100*tot['occluded']/T:.2f}%)")
    print(f"UNEXPLAINED (clear line) {tot['unexplained']:,}  ({100*tot['unexplained']/T:.2f}%)")
    if unexp_depth:
        d = np.concatenate(unexp_depth)
        print(f"  unexplained range  p50 {np.median(d):.1f} m  p90 {np.percentile(d,90):.1f} m")
    tot["pct_unexplained"] = 100 * tot["unexplained"] / T
    os.makedirs(os.path.join(ROOT, os.path.dirname(a.out)), exist_ok=True)
    json.dump(dict(summary=tot, frames=per_frame, nstep=NSTEP,
                   seed=a.seed, per_frame_cap=a.per_frame),
              open(os.path.join(ROOT, a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
