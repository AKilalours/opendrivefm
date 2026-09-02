"""Ground-plane segmentation and ray-cast free-space occupancy.

Why this exists: a LiDAR BEV that draws every return the same way cannot
distinguish "there is nothing here" from "I cannot see here". Those are very
different for a planner -- the first is drivable, the second is where a cyclist
emerges from behind a van. Separating ground from obstacles, and then tracing
what the sensor could actually observe, is what turns a point cloud into an
occupancy grid a planner can consume.

UNKNOWN = 0, FREE = 1, OCCUPIED = 2.
"""
from __future__ import annotations

import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


def fit_ground_plane(pts_xyz, n_iter=120, tol=0.18, seed=0, max_range=40.0):
    """RANSAC plane fit, returned as (normal, d) with |normal| = 1 and
    normal . x + d = 0.

    Seeded only from returns that are plausibly ground -- inside `max_range`
    and in the lower part of the height distribution. A plain RANSAC over the
    whole cloud will happily fit a building facade, which is a perfectly good
    plane and completely wrong. Constraining the seed set and then rejecting
    any candidate whose normal is not roughly vertical removes that failure.
    """
    p = np.asarray(pts_xyz, np.float64)
    r = np.linalg.norm(p[:, :2], axis=1)
    seed_mask = (r < max_range) & (p[:, 2] < np.percentile(p[:, 2], 55))
    cand = p[seed_mask]
    if len(cand) < 50:
        cand = p
    rng = np.random.default_rng(seed)

    best_n, best_d, best_cnt = np.array([0.0, 0.0, 1.0]), -float(np.median(cand[:, 2])), -1
    for _ in range(n_iter):
        idx = rng.choice(len(cand), 3, replace=False)
        a, b, c = cand[idx]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        if abs(nrm[2]) < 0.90:          # reject near-vertical planes (facades)
            continue
        d = -float(nrm @ a)
        cnt = int((np.abs(cand @ nrm + d) < tol).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = nrm, d, cnt

    if best_n[2] < 0:                   # keep the normal pointing up
        best_n, best_d = -best_n, -best_d

    # one least-squares refit on the inliers: RANSAC picks a good support set,
    # it does not give a good plane
    inl = np.abs(p @ best_n + best_d) < tol
    if inl.sum() >= 3:
        q = p[inl]
        c = q.mean(axis=0)
        _, _, vt = np.linalg.svd(q - c, full_matrices=False)
        nrm = vt[-1]
        if nrm[2] < 0:
            nrm = -nrm
        if abs(nrm[2]) > 0.90:
            best_n, best_d = nrm, -float(nrm @ c)
    return best_n, float(best_d)


def height_above_ground(pts_xyz, plane) -> np.ndarray:
    n, d = plane
    return np.asarray(pts_xyz, np.float64) @ n + d


def label_ground(pts_xyz, plane, tol=0.25) -> np.ndarray:
    """True where a return lies within `tol` of the fitted plane."""
    return np.abs(height_above_ground(pts_xyz, plane)) < tol


def occupancy_grid(pts_xyz, ground_mask, rng_m=54.0, res=0.25,
                   n_azimuth=720, min_obstacle_h=0.25, max_obstacle_h=3.0,
                   plane=None, visible=None):
    """Ray-cast occupancy from a single sensor origin at the ego.

    Rather than stepping a DDA along every one of a few hundred thousand rays,
    this reduces the cloud to a polar range image: for each of `n_azimuth`
    bearings, the nearest obstacle return defines how far the sensor could see.
    Cells nearer than that are FREE, the boundary ring is OCCUPIED, everything
    beyond is UNKNOWN -- which is exactly the occlusion shadow behind a parked
    vehicle. This is O(cells + points) instead of O(rays x steps), and the
    result is the same because a single-origin scan has one visibility limit
    per bearing by construction.

    Bearings with no obstacle return still get free space out to the furthest
    ground return, so open road does not read as unobserved.

    `visible` selects the returns that define VISIBILITY, and should be the
    current sweep alone even when `pts_xyz` carries accumulated history. What
    the sensor can see right now is a property of this scan; taking the nearest
    obstacle across half a second of accumulation lets a vehicle that has since
    driven past go on blocking that bearing forever, which under-reports free
    space and invents occlusion shadows behind objects that are no longer
    there. History is for density, not for visibility.
    """
    p = np.asarray(pts_xyz, np.float64)
    g = np.asarray(ground_mask, bool)
    vis = np.ones(len(p), bool) if visible is None else np.asarray(visible, bool)

    h = height_above_ground(p, plane) if plane is not None else p[:, 2]
    r = np.linalg.norm(p[:, :2], axis=1)
    az = np.arctan2(p[:, 1], p[:, 0])
    bin_ = np.clip(((az + np.pi) / (2 * np.pi) * n_azimuth).astype(np.int64),
                   0, n_azimuth - 1)

    # Obstacles are non-ground returns in the height band a vehicle actually
    # has to avoid. Overhanging foliage and gantries sit above it and must not
    # cast an occlusion shadow across drivable road.
    obst = vis & (~g) & (h > min_obstacle_h) & (h < max_obstacle_h) & (r < rng_m)
    # Obstacles are gated to the current scan; ground is not. Free space is a
    # claim about static road surface and stays true across half a second,
    # whereas an obstacle that has driven on must stop casting a shadow. Using
    # the single sweep for BOTH actually reduces free space, because sparse
    # single-sweep ground returns shorten the fallback horizon -- measured 7.6%
    # free against 8.6% for the accumulated cloud. Splitting them is what gets
    # the physics and the density right at the same time.
    free_src = g & (r < rng_m)

    r_obst = np.full(n_azimuth, np.inf)
    np.minimum.at(r_obst, bin_[obst], r[obst])
    r_gnd = np.zeros(n_azimuth)
    np.maximum.at(r_gnd, bin_[free_src], r[free_src])

    # A single grazing return should not blank out a whole bearing. Take the
    # nearest obstacle only where at least a few returns agree on it.
    cnt = np.bincount(bin_[obst], minlength=n_azimuth)
    r_obst[cnt < 2] = np.inf
    horizon = np.where(np.isfinite(r_obst), r_obst, r_gnd)

    # Close sampling gaps along azimuth. A beam that happens to land on tarmac
    # 40 m out gives one bearing a long horizon while its neighbours, which
    # missed, get a short one -- rendering as a fan of one-cell spikes rather
    # than an observed wedge. The road is continuous; the sampling is not. A
    # circular median over a few bearings restores the surface without
    # extending free space past a genuine obstacle, because a real obstacle
    # occupies many consecutive bearings and survives the median.
    k = 5
    stack = np.stack([np.roll(horizon, i) for i in range(-(k // 2), k // 2 + 1)])
    horizon = np.median(stack, axis=0)
    r_obst_s = np.median(np.stack(
        [np.roll(r_obst, i) for i in range(-(k // 2), k // 2 + 1)]), axis=0)
    horizon = np.minimum(horizon, np.where(np.isfinite(r_obst_s), r_obst_s, np.inf))

    n = int(2 * rng_m / res)
    ax = (np.arange(n) + 0.5) * res - rng_m
    X, Y = np.meshgrid(ax, ax, indexing="ij")           # X forward, Y left
    R = np.hypot(X, Y)
    B = np.clip(((np.arctan2(Y, X) + np.pi) / (2 * np.pi) * n_azimuth
                 ).astype(np.int64), 0, n_azimuth - 1)
    H = horizon[B]

    grid = np.full((n, n), UNKNOWN, np.uint8)
    grid[(R < H - res) & (R < rng_m)] = FREE
    grid[(np.abs(R - H) <= res * 1.5) & np.isfinite(r_obst_s)[B] & (R < rng_m)] = OCCUPIED
    grid[R >= rng_m] = UNKNOWN
    return grid, {"res": res, "rng_m": rng_m, "n": n}


def grid_stats(grid) -> dict:
    tot = int(grid.size)
    u = int((grid == UNKNOWN).sum()); f = int((grid == FREE).sum())
    o = int((grid == OCCUPIED).sum())
    return {"cells": tot,
            "free_frac": f / tot, "occupied_frac": o / tot, "unknown_frac": u / tot,
            "observed_frac": (f + o) / tot}
