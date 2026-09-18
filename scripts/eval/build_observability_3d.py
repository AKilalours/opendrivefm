"""Per-VOXEL observability on the Occ3D grid. Replaces the BEV broadcast.

Why
---
The 2D measure answers "can a camera see this patch of GROUND". Broadcasting
that answer up a 16-level column asserts that a voxel of air five metres above
an occluded pavement is itself unseen, which is false and which corrupted 65%
of camera-visible voxels (350M of 536M got observability exactly 0). H2 then
compared that 2D measure against Occ3D's true-3D mask_camera on a 3D task and
lost by 0.072 AUROC. This is the fix that defect deserves, and it was named
before H2 ran, not after.

Method
------
Per camera, a z-buffer rather than per-ray tracing: 640k voxels x 6 cameras x
6,019 frames is 23M rays a frame and would never finish. Instead

  1. project the LiDAR-occupied voxel centres into the image,
  2. scatter-min their depths into a coarse pixel grid -> a depth map,
  3. project every voxel and call it visible if its own depth does not exceed
     the buffered depth by more than a tolerance.

That is the same occlusion logic Occ3D uses for mask_camera, which makes the
H2 comparison fair: identical geometry, differing only in that ours is
weighted by resolution and trust and theirs is thresholded to one bit.

The buffer is deliberately COARSE (8x8 px bins). LiDAR returns are sparse, so
a full-resolution buffer is full of holes and every hole lets a wall leak
through as visible. Coarsening trades a little spatial precision for not
hallucinating sight lines through solid objects -- the same reasoning that
made the 2D version cast angular width rather than infinitely thin rays.

Everything else is carried over unchanged from the validated measure, so the
only thing that differs is the dimensionality:

    observability(v) = 1 - prod_i (1 - trust_i * coverage_i(v) * visible_i(v))

TRUST is the same constant 0.795 and still contributes no discrimination; it
stays in the formula for continuity with Stage 1, not because it earns a place.
COVERAGE is the same resolution weighting, now evaluated at the voxel's own
depth instead of its ground column's: a res x res face at depth Z projects to
about fx*fy*res^2/Z^2 pixels, normalised by the same full_px = 140.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OCC_N, OCC_RES, OCC_RNG = 200, 0.4, 40.0
OCC_NZ, OCC_Z0 = 16, -1.0
# Saturation area for coverage, DERIVED, not carried over.
#
# The 2D measure used FULL_PX = 140 for a 0.5 m patch of GROUND. A ground patch
# is viewed nearly edge-on, so its projected area falls as fx*fy*res^2*h/r^3 --
# the cube law, from foreshortening. Setting that equal to 140 with fx = 1266,
# res = 0.5, h = 1.5 gives r = 16.3 m: the range at which the validated measure
# stopped saturating and began to grade.
#
# A voxel face is viewed frontally and falls off only as fx*fy*res^2/Z^2. Using
# 140 there puts saturation at 42.8 m, outside the +/-40 m grid, so coverage is
# 1 everywhere and observability collapses to trust*visible -- measured as
# p50 = p90 = 0.796, a two-valued measure pretending to be continuous.
#
# Porting the CONSTANT is wrong; porting its MEANING is right. Preserving the
# same 16.3 m saturation range under the square law:
#     FULL_PX = fx*fy*res^2 / r_sat^2 = 1266^2 * 0.4^2 / 16.3^2 ~= 965
# This is fixed by the old measure's behaviour, not by tuning an outcome.
FULL_PX = 965.0
BIN = 2              # depth-buffer resolution, pixels. See note.
#
# This was 8, and it was the root cause of all five failed attempts. A coarse
# bin forces ground voxels at very different depths into one cell; the nearest
# wins and shadows the rest, which is the grazing-angle failure. At or near
# full pixel resolution the problem does not arise: a pixel looking at the road
# sees exactly one ground voxel and the others project elsewhere, so there is
# nothing to self-shadow. The coarse bin existed only to hide gaps left by
# point-splatting, and the corner bounding-box splat closes those gaps
# geometrically, so it is no longer buying anything. Finer is strictly more
# correct here, bounded only by compute; the derived bias scales with BIN on
# its own.
DEPTH_TOL = 0.55     # metres; half a voxel diagonal (0.35) plus slack
CAM_H = 1.5          # nominal lens height, for the grazing-angle bias below

# SLOPE-SCALED DEPTH BIAS, derived not tuned.
#
# A z-buffer cannot resolve visibility on a surface seen at a grazing angle:
# foreshortening packs tens of metres of road into a few pixels near the
# horizon, so many ground voxels at very different depths share one bin, the
# nearest wins, and the rest are shadowed although the camera plainly sees
# them. Measured as 68-93% of SURFACE voxels zeroed across three attempts,
# against 8-62% of free ones.
#
# The remedy is the standard shadow-map fix: tolerate the depth spread that a
# single bin actually spans. For ground at range r viewed from height h,
# dr/dpixel = r^2/(f*h), so across BIN pixels the spread is
#
#     bias(r) = BIN * r^2 / (f * h)
#
# ~0.1 m at 5 m, ~1.7 m at 20 m, ~6.7 m at 40 m. A flat 0.55 m was roughly a
# tenth of what grazing geometry needs at range, which is why the failure grew
# with distance. Capped so it cannot swallow genuine occlusion by a wall.
BIAS_CAP = 15.0
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


def lidar_occupied_centres(pts_ego):
    """Voxel centres containing at least one return. The occluder set."""
    x, y, z = pts_ego[:, 0], pts_ego[:, 1], pts_ego[:, 2]
    k = ((np.abs(x) < OCC_RNG) & (np.abs(y) < OCC_RNG) &
         (z >= OCC_Z0) & (z < OCC_Z0 + OCC_NZ * OCC_RES))
    ix = ((x[k] + OCC_RNG) / OCC_RES).astype(np.int32)
    iy = ((y[k] + OCC_RNG) / OCC_RES).astype(np.int32)
    iz = ((z[k] - OCC_Z0) / OCC_RES).astype(np.int32)
    flat = np.unique((ix * OCC_N + iy) * OCC_NZ + iz)
    ci = flat // (OCC_N * OCC_NZ)
    cj = (flat // OCC_NZ) % OCC_N
    ck = flat % OCC_NZ
    return np.stack([(ci + 0.5) * OCC_RES - OCC_RNG,
                     (cj + 0.5) * OCC_RES - OCC_RNG,
                     (ck + 0.5) * OCC_RES + OCC_Z0], 1).astype(np.float32)


def gt_occupied_centres(sem):
    """Non-free voxels of the dense Occ3D volume. The occluder set.

    Single-sweep LiDAR leaves most depth-buffer bins empty, so walls stop
    casting shadows and 93% of all voxels came back visible against Occ3D's
    own 13.9%. Occ3D builds mask_camera against its dense accumulated volume;
    matching that is what makes H2 a test of continuous-versus-binary rather
    than a test of who had denser geometry.

    CONSEQUENCE, stated rather than buried: observability computed this way is
    an OFFLINE quantity. It needs ground-truth scene geometry and cannot be
    evaluated at inference time. That suits the validation and sensor-placement
    use, and rules out the runtime-signal framing until a proxy exists.
    """
    ii, jj, kk = np.nonzero(sem != 17)
    return np.stack([(ii + 0.5) * OCC_RES - OCC_RNG,
                     (jj + 0.5) * OCC_RES - OCC_RNG,
                     (kk + 0.5) * OCC_RES + OCC_Z0], 1).astype(np.float32)


def project(pts_ego, cam):
    """ego -> camera -> pixels. Returns u, v, depth, in-front mask."""
    R = M.quat_to_rot(cam["sensor2ego_rotation"])
    t = np.asarray(cam["sensor2ego_translation"], np.float32)
    K = np.asarray(cam["cam_intrinsic"] if "cam_intrinsic" in cam
                   else cam["intrinsic"], np.float32)
    p = (pts_ego - t) @ R                      # R^T x, written as x R
    z = p[:, 2]
    good = z > 0.1
    u = np.full(z.shape, -1.0, np.float32)
    v = np.full(z.shape, -1.0, np.float32)
    zz = np.where(good, z, 1.0)
    u[good] = (K[0, 0] * p[good, 0] / zz[good] + K[0, 2])
    v[good] = (K[1, 1] * p[good, 1] / zz[good] + K[1, 2])
    return u, v, z, good, K


def camera_term(vox, occ_pts, cam, W, H, dilate=True, occlude=True):
    """trust * coverage * visible, for every voxel, for one camera."""
    out = np.zeros(vox.shape[0], np.float32)
    u, v, z, good, K = project(vox, cam)
    inside = good & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inside.any():
        return out

    nbu, nbv = int(np.ceil(W / BIN)), int(np.ceil(H / BIN))
    buf = np.full((nbv, nbu), np.inf, np.float32)
    if occ_pts.shape[0] and occlude:
        # CORNER-PROJECTED BOUNDING BOX.
        #
        # Three earlier attempts all approximated a voxel's image footprint
        # with a shape it does not have, and each failed differently:
        #   point splat, no dilation  -> walls leak between projected points
        #   point splat + 3x3 dilate  -> nearer depths drag sideways, surfaces
        #                                shadow themselves (68% zeroed)
        #   isotropic depth splat     -> a square footprint; a road voxel 2 m
        #                                ahead smears its 2 m depth from the
        #                                frame bottom to the horizon (85%)
        #
        # A voxel's real footprint is the projection of its eight corners,
        # which is wide and short for near ground and small for distant
        # geometry. Projecting the corners and splatting that box respects
        # perspective and obliquity, so gaps close without depth spreading
        # where it does not belong.
        #
        # Depth stored is the NEAREST corner: the voxel occludes what lies
        # beyond its front face, not beyond its centre.
        c = OCC_RES * 0.5
        us = np.empty((occ_pts.shape[0], 8), np.float32)
        vs = np.empty((occ_pts.shape[0], 8), np.float32)
        zs = np.empty((occ_pts.shape[0], 8), np.float32)
        gs = np.zeros((occ_pts.shape[0], 8), bool)
        for n, (dx, dy, dz) in enumerate(
                [(sx * c, sy * c, sz * c)
                 for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]):
            cu, cv, cz, cg, _ = project(
                occ_pts + np.array([dx, dy, dz], np.float32), cam)
            us[:, n], vs[:, n], zs[:, n], gs[:, n] = cu, cv, cz, cg

        ok = gs.all(1)                       # whole voxel in front of the lens
        # DISCARD off-screen occluders before clipping. Clipping is not
        # intersection: np.clip squashed a voxel projecting to u = -3000 --
        # beside or behind the car, invisible to this camera -- onto column 0,
        # where it stamped its very small depth along the image border.
        # Measured: 60.7% of occluders (12,606 of 20,766) were fully
        # off-screen and being injected as phantom geometry. That, not the
        # splat shape or the bin size or the corner choice, is what zeroed
        # three quarters of all surface voxels.
        if ok.any():
            ru0, ru1 = us[ok].min(1), us[ok].max(1)
            rv0, rv1 = vs[ok].min(1), vs[ok].max(1)
            on = (ru1 >= 0) & (ru0 <= W - 1) & (rv1 >= 0) & (rv0 <= H - 1)
            ru0, ru1, rv0, rv1 = ru0[on], ru1[on], rv0[on], rv1[on]
            zn_all = zs[ok].max(1)[on]
            u0 = np.clip(ru0, 0, W - 1)
            u1 = np.clip(ru1, 0, W - 1)
            v0 = np.clip(rv0, 0, H - 1)
            v1 = np.clip(rv1, 0, H - 1)
            # FAR face, not near face.
            #
            # Storing the nearest corner was the last live bug. A voxel is a
            # 3D box, so its projected bounding box includes its TOP face,
            # which for a ground voxel reaches up-image into the region where
            # the road BEHIND it appears. Painting the near-face depth there
            # made every ground voxel shadow its own continuation -- which is
            # why surf_zero stayed at 0.72-0.93 no matter how fine the bins got
            # (0.793 at 2 px, 0.715 at 1 px: resolution was never the cause).
            #
            # Occlusion means "blocks what lies BEYOND it", so the depth a
            # voxel contributes is its far face. A surface then stops
            # shadowing itself, while a wall at 10 m still occludes a target at
            # 30 m, since 30 > 10.2 + tol + bias.
            zn = zn_all
            bu0 = (u0 // BIN).astype(np.int32)
            bu1 = (u1 // BIN).astype(np.int32)
            bv0 = (v0 // BIN).astype(np.int32)
            bv1 = (v1 // BIN).astype(np.int32)
            wu = bu1 - bu0 + 1
            wv = bv1 - bv0 + 1

            # Two rasterisation paths, chosen by box area only.
            #
            # np.minimum.at is a scattered read-modify-write and costs roughly
            # a microsecond per element, so a near occluder covering 443x450
            # bins would alone need ~0.2 s. Those boxes are few, and each is a
            # contiguous rectangle, so a direct slice-min runs at memory speed.
            # Small boxes are many, so they keep the grouped vectorised path.
            big = (wu.astype(np.int64) * wv) > 64
            if (~big).any():
                su, sv = wu[~big], wv[~big]
                gu0, gv0, gz = bu0[~big], bv0[~big], zn[~big]
                key = su.astype(np.int64) * 64 + sv
                for k in np.unique(key):
                    g = key == k
                    a, b = int(k // 64), int(k % 64)
                    cu_, cv_, cz_ = gu0[g], gv0[g], gz[g]
                    for dv in range(b):
                        tv = cv_ + dv
                        okv = (tv >= 0) & (tv < nbv)
                        for du in range(a):
                            tu = cu_ + du
                            m2 = okv & (tu >= 0) & (tu < nbu)
                            if m2.any():
                                np.minimum.at(buf, (tv[m2], tu[m2]), cz_[m2])
            if big.any():
                for a0, a1, b0, b1, zq in zip(bv0[big], bv1[big],
                                              bu0[big], bu1[big], zn[big]):
                    a0 = max(int(a0), 0); a1 = min(int(a1) + 1, nbv)
                    b0 = max(int(b0), 0); b1 = min(int(b1) + 1, nbu)
                    if a1 > a0 and b1 > b0:
                        sl = buf[a0:a1, b0:b1]
                        np.minimum(sl, zq, out=sl)
    buf = buf.ravel()

    bi = (v[inside].astype(np.int32) // BIN) * nbu + \
         (u[inside].astype(np.int32) // BIN)
    zi = z[inside]
    # Bias from the OCCLUDER'S OWN PROJECTED HEIGHT, not the bin size.
    #
    # A voxel's bounding box includes its top face, so at range r it covers the
    # image rows where ground out to r + res*r/h appears: measured, ~17-30 px
    # tall at 30 m while consecutive ground voxels sit 0.84 px apart, so each
    # one blankets 20-35 voxels of road behind it. The earlier BIN-based bias
    # gave 0.95 m at 30 m against a shadow reaching ~14 m, which is why
    # surf_zero never fell below 0.68 through eight attempts.
    #
    #     bias(r) = OCC_RES * r / CAM_H
    #
    # 1.3 m at 5 m, 8 m at 30 m, 10.7 m at 40 m. This does weaken genuine
    # occlusion at long range -- a wall at 30 m no longer hides a target at
    # 38 m -- which is inherent to voxel-grid rendering of grazing surfaces and
    # must be stated as a limitation rather than hidden.
    bias = np.minimum(OCC_RES * zi / CAM_H, BIAS_CAP)
    visible = zi <= (buf[bi] + DEPTH_TOL + bias)

    # resolution-weighted coverage at the voxel's OWN depth
    area = (K[0, 0] * K[1, 1] * OCC_RES * OCC_RES) / np.maximum(z[inside], 0.1) ** 2
    cov = np.clip(area / FULL_PX, 0.0, 1.0)

    idx = np.flatnonzero(inside)
    out[idx] = (TRUST * cov * visible).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--out", default="data/pack/observability_3d")
    ap.add_argument("--occluder", choices=["lidar", "occ3d"], default="occ3d",
                    help="what blocks sight lines. See the note below.")
    ap.add_argument("--no-occlusion", action="store_true",
                    help="ISOLATION TEST: skip the depth test entirely, so "
                         "every in-frustum voxel counts as visible. Separates "
                         "frustum coverage from occlusion as causes.")
    ap.add_argument("--bin", type=int, default=0,
                    help="override depth-buffer resolution in pixels")
    ap.add_argument("--no-dilate", action="store_true",
                    help="skip the 3x3 depth-buffer dilation. Diagnostic: the "
                         "dilation drags a nearer neighbour's depth sideways "
                         "into a surface voxel's own bin, so surfaces shadow "
                         "themselves -- the 3D form of the self-occlusion bug "
                         "already fixed once in the 2D per-box measure.")
    ap.add_argument("--gts",
                    default="data/occ3d/Occupancy3D-nuScenes-trainval/gts")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    index = json.load(open(os.path.join(args.pack, "index.json")))
    calib = json.load(open(os.path.join(args.pack, "calib.json")))
    if args.limit:
        index = index[:args.limit]
    gt_map = {}
    if args.occluder == "occ3d":
        import glob as _g
        gt_map = {os.path.basename(os.path.dirname(q)): q
                  for q in _g.glob(os.path.join(args.gts, "*", "*",
                                                "labels.npz"))}
        print(f"occluder: occ3d dense GT, {len(gt_map):,} frames indexed")
    if args.bin:
        globals()["BIN"] = args.bin
    print(f"depth-buffer resolution: {BIN} px")
    vox = voxel_centres()
    print(f"{vox.shape[0]:,} voxels per frame, {len(index):,} frames")

    t0 = time.time()
    done = 0
    for row in index:
        tok = row["token"]
        dst = os.path.join(args.out, tok + ".npy")
        if os.path.exists(dst):
            done += 1
            continue
        lp = os.path.join(args.pack, "lidar", f"{tok}.npy")
        if tok not in calib or not os.path.exists(lp):
            continue
        if args.occluder == "occ3d":
            gp = gt_map.get(tok)
            if gp is None:
                continue
            occ_pts = gt_occupied_centres(np.load(gp)["semantics"])
        else:
            occ_pts = lidar_occupied_centres(
                M.lidar_to_ego(np.load(lp), row["lidar"]))

        miss = np.ones(vox.shape[0], np.float32)     # prod (1 - term)
        for c in CAMS:
            cam = calib[tok]["cams"].get(c)
            if cam is None:
                continue
            W = cam.get("width", 1600)
            H = cam.get("height", 900)
            miss *= (1.0 - camera_term(vox, occ_pts, cam, W, H,
                                       dilate=not args.no_dilate,
                                       occlude=not args.no_occlusion))
        obs = (1.0 - miss).reshape(OCC_N, OCC_N, OCC_NZ)
        np.save(dst, np.clip(np.rint(obs * 255), 0, 255).astype(np.uint8))
        done += 1
        if done % 100 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(index)}  {el/done:.3f}s/frame  "
                  f"eta {(len(index)-done)*el/done/60:.1f} min", flush=True)

    meta = dict(grid=[OCC_N, OCC_N, OCC_NZ], res=OCC_RES, rng=OCC_RNG,
                z0=OCC_Z0, frame="ego", trust=TRUST, full_px=FULL_PX,
                depth_bin_px=int(BIN), depth_tol_m=DEPTH_TOL,
                bias_cap_m=BIAS_CAP, cam_h=CAM_H,
                quantisation="uint8 = round(255 * observability)",
                frames=done)
    json.dump(meta, open(os.path.join(args.out, "_meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
