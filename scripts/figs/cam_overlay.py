#!/usr/bin/env python3
"""Project the occupancy grid onto the camera image, as a shaded ground plane.

A voxel cloud is hard to read. A photograph with the road shaded is not, which
is why every AV reel does it this way. For each pixel we cast a ray through the
camera model, intersect it with the road plane fitted to this frame's drivable
voxels, and look up what the model says about the grid cell it lands in:

    teal     predicted free AND camera-observed  -> verified clear
    orange   predicted free, almost no evidence  -> unverified free

Fully vectorised: one ray bundle per frame, no per-cell polygon drawing.
"""
from __future__ import annotations
import numpy as np

RES, RNG, Z0, NZ, FREE, TAU = 0.4, 40.0, -1.0, 16, 17, 0.15
GROUND = (11, 12, 13, 14)
TEAL = np.array([0, 168, 180], np.float32)
ORANGE = np.array([237, 161, 0], np.float32)


def road_height(cls):
    """Median height of drivable-surface voxels: the plane the rays hit."""
    i, j, k = np.nonzero(cls == 11)
    if len(k) < 50:
        i, j, k = np.nonzero(np.isin(cls, GROUND))
    if len(k) < 50:
        return Z0 + 0.2
    return float(np.median(k) * RES + Z0 + RES / 2)


def overlay(img, cls, obs, K, Rs, ts, max_range=38.0,
            a_teal=0.30, a_orange=0.52, step=2):
    """img: HxWx3 uint8 camera frame. Returns a new image with the plane shaded.

    K is the 3x3 intrinsic at the IMAGE's current resolution. Rs, ts take a
    point from camera frame to ego frame.
    """
    H, W = img.shape[:2]
    zr = road_height(cls)

    vs, us = np.mgrid[0:H:step, 0:W:step]
    ones = np.ones_like(us, np.float64)
    pix = np.stack([us, vs, ones], -1).reshape(-1, 3).T          # 3 x N
    d_cam = np.linalg.inv(K) @ pix
    d_ego = Rs @ d_cam                                            # 3 x N
    dz = d_ego[2]
    t = np.where(np.abs(dz) > 1e-6, (zr - ts[2]) / np.where(dz == 0, 1e-6, dz),
                 -1.0)
    ok = (t > 0.5) & (t < max_range) & (dz < 0)
    P = ts[:, None] + d_ego * t[None, :]

    gi = np.floor((P[0] + RNG) / RES).astype(np.int64)
    gj = np.floor((P[1] + RNG) / RES).astype(np.int64)
    n = int(2 * RNG / RES)
    ok &= (gi >= 0) & (gi < n) & (gj >= 0) & (gj < n)

    free = cls == FREE
    seen = (free & (obs >= TAU)).any(-1)
    blind = (free & (obs < TAU)).any(-1) & ~seen
    drivable = (cls == 11).any(-1)          # the road itself, not verge or kerb

    # Occlusion, narrowly. The plane fit says nothing about what stands on it,
    # so without a test the shading paints over cars. The test has to be for
    # OBSTACLES only: a first attempt rejected any non-free voxel above the
    # cell, which on this tree-lined scene blocked 32k of 40k columns on
    # overhanging canopy that occludes nothing at camera height.
    OBSTACLE = tuple(range(1, 11))      # barrier .. truck; not road furniture
    band = cls[:, :, max(int((zr + 0.2 - Z0) / RES), 0):
                 min(int((zr + 2.6 - Z0) / RES), NZ)]
    blocked = (np.isin(band, OBSTACLE).any(-1) if band.shape[2]
               else np.zeros_like(drivable))

    gi_s = np.where(ok, gi, 0); gj_s = np.where(ok, gj, 0)
    free_col = ~blocked[gi_s, gj_s]
    is_unv = ok & free_col & blind[gi_s, gj_s]
    is_clear = ok & free_col & ~is_unv & drivable[gi_s, gj_s]

    # Distance fade, so the overlay dissolves instead of ending on a hard line
    # where the 40 m grid runs out.
    fade = np.clip((max_range - t) / 14.0, 0.0, 1.0)

    small = (H + step - 1) // step, (W + step - 1) // step
    lay = np.zeros(small + (4,), np.float32)
    f = lay.reshape(-1, 4)
    f[is_clear, :3] = TEAL
    f[is_clear, 3] = a_teal * fade[is_clear]
    f[is_unv, :3] = ORANGE
    f[is_unv, 3] = a_orange * fade[is_unv]

    if step > 1:                       # nearest-neighbour back to full size
        lay = np.repeat(np.repeat(lay, step, 0), step, 1)[:H, :W]
    a = lay[..., 3:4]
    return np.clip(img * (1 - a) + lay[..., :3] * a, 0, 255).astype(np.uint8)
