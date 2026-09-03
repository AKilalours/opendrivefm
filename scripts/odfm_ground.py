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


# ---------------------------------------------------------------------------
# proper inverse sensor model: per-ray log-odds occupancy
# ---------------------------------------------------------------------------

# A hit must outweigh the free evidence from beams that merely pass nearby.
# A vertical surface is struck by few beams but grazed by many, so with
# l_occ ~ l_free an obstacle is voted away by its own neighbours and the grid
# comes out 0.2% occupied -- Bayesian-correct given equal evidence weights,
# and wrong as a sensor model. A return that terminates on a surface is far
# stronger evidence than a return that passes through a cell on its way
# elsewhere, so the weights are asymmetric.
L_FREE, L_OCC, L_CLAMP = 0.45, 2.4, 6.0


def occupancy_logodds(points, ground_mask, plane, rng_m=54.0, res=0.20,
                      z_band=(-0.4, 2.5), step_frac=0.6, hit_ages=(0, 1, 2),
                      beam_fill=True, beam_iters=2,
                      free_ages=None, l_free=L_FREE, l_occ=L_OCC,
                      clamp=L_CLAMP, chunk=40000):
    """Occupancy as accumulated log-odds, one ray per return.

    This replaces a polar range-image approach that reduced each bearing to a
    single horizon: nearest obstacle, everything beyond it unknown. That is
    wrong in a way that shows. A 0.2 m signpost narrower than an azimuth bin
    blanked the entire bin out to 54 m, so a street with poles and parked cars
    came out 91% unknown and rendered as a spider of thin spokes. Real beams
    pass BETWEEN obstacles, and the only way to represent that is to let each
    return carve its own ray.

    The model is the textbook one. A return at range r means the beam travelled
    r metres unobstructed, so every cell along the way gets evidence of free
    (-l_free) and the endpoint gets evidence of occupied (+l_occ). Evidence
    accumulates across beams and sweeps and is squashed to probability at the
    end, which is why two beams disagreeing about a cell produce an
    intermediate value rather than whichever fired last.

    Height band: only returns between `z_band` above the fitted ground take
    part. A ray to tree canopy 8 m up passes over the road on its way, so
    carving the ground plane along it claims free space that was never
    observed at ground level; and canopy endpoints are not obstacles a vehicle
    must avoid. Excluding them from both roles is the fix.

    Ground returns carve free space but never mark occupied -- tarmac is the
    ray terminating on drivable surface, not on an obstacle.

    Hits come from `hit_ages` -- the three most recent sweeps, about 0.15 s --
    while free-space carving uses every sweep. A moving object should not leave
    a long trail of occupied cells behind it, but the road it drove over is
    genuinely free and later beams re-carve it. Three sweeps is the compromise:
    enough returns to give a vertical surface a continuous rim, short enough
    that a vehicle at 10 m/s smears by 1.5 m rather than 4.5 m.
    """
    p = np.asarray(points, np.float32)
    age = p[:, 5].astype(np.int64) if p.shape[1] > 5 else np.zeros(len(p), np.int64)
    h = height_above_ground(p[:, :3], plane)
    r = np.linalg.norm(p[:, :2], axis=1)

    in_band = (h > z_band[0]) & (h < z_band[1]) & (r > 0.5) & (r < rng_m)
    free_sel = in_band if free_ages is None else in_band & np.isin(age, free_ages)
    hit_sel = in_band & (~np.asarray(ground_mask, bool)) & np.isin(age, hit_ages)

    n = int(2 * rng_m / res)
    lo = np.zeros((n, n), np.float32)

    def to_idx(x, y):
        ix = np.clip(((x + rng_m) / res).astype(np.int32), 0, n - 1)
        iy = np.clip(((y + rng_m) / res).astype(np.int32), 0, n - 1)
        return ix, iy

    # --- free-space carving, chunked so the sample tensor stays bounded ---
    step = res * step_frac
    max_steps = int(np.ceil(rng_m / step))
    idx = np.where(free_sel)[0]
    for s in range(0, len(idx), chunk):
        sl = idx[s:s + chunk]
        rr = r[sl].astype(np.float32)
        ux, uy = (p[sl, 0] / rr).astype(np.float32), (p[sl, 1] / rr).astype(np.float32)
        t = (np.arange(1, max_steps + 1, dtype=np.float32) * step)[None, :]
        # stop one cell short of the endpoint: the last cell is the return
        # itself and belongs to the hit, not to the free run leading up to it
        valid = t < (rr[:, None] - res)
        if not valid.any():
            continue
        xs = (ux[:, None] * t)[valid]
        ys = (uy[:, None] * t)[valid]
        ix, iy = to_idx(xs, ys)
        np.add.at(lo, (ix, iy), -l_free)

    ix, iy = to_idx(p[hit_sel, 0], p[hit_sel, 1])
    np.add.at(lo, (ix, iy), l_occ)

    if beam_fill:
        lo = _close_beam_gaps(lo, iters=beam_iters, l_free=l_free)

    np.clip(lo, -clamp, clamp, out=lo)
    prob = 1.0 / (1.0 + np.exp(-lo))
    meta = {"res": res, "rng_m": rng_m, "n": n, "z_band": list(z_band),
            "l_free": l_free, "l_occ": l_occ, "clamp": clamp,
            "rays_free": int(free_sel.sum()), "rays_hit": int(hit_sel.sum())}
    return prob, lo, meta


def prob_to_tristate(prob, free_below=0.35, occ_above=0.65):
    g = np.full(prob.shape, UNKNOWN, np.uint8)
    g[prob < free_below] = FREE
    g[prob > occ_above] = OCCUPIED
    return g


def _close_beam_gaps(lo, iters=2, l_free=L_FREE, min_free_neighbours=5):
    """Fill the unobserved slivers between adjacent beams.

    A 32-beam spinning LiDAR has roughly 0.1-0.4 deg of azimuth spacing, which
    at 40 m is 7-28 cm -- one to two cells at 20 cm resolution. Nothing is
    sampled in between, so a pure per-ray carve leaves the free region shot
    through with one-cell radial slivers and renders as a fan of spokes rather
    than a surface.

    Those slivers are not unknown in any meaningful sense: they are bracketed
    on both sides by beams that travelled the full distance unobstructed, and a
    real beam has angular width besides. So a cell carrying no evidence whose
    neighbourhood is mostly confidently-free inherits free evidence, capped at
    one unit so an inferred cell never outvotes a directly observed one.

    Deliberately one-directional: gaps are only filled towards FREE, never
    towards occupied. Inventing occupancy would fabricate obstacles, which is
    the one error a planner must never be handed.
    """
    out = lo.astype(np.float32).copy()
    thresh = -l_free * 1.5
    for _ in range(iters):
        free = (out < thresh).astype(np.float32)
        k = np.ones((3, 3), np.float32)
        try:
            from scipy.ndimage import convolve
            cnt = convolve(free, k, mode="constant", cval=0.0)
        except Exception:                       # numpy fallback, same result
            cnt = np.zeros_like(free)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cnt += np.roll(np.roll(free, dx, 0), dy, 1)
        gap = (np.abs(out) < 1e-6) & (cnt >= min_free_neighbours)
        out[gap] = -l_free
    return out


def visibility_from(occ_mask, origin_xy, rng_m=54.0, res=0.20,
                    n_azimuth=2048, slack=1.0):
    """Which ground cells are visible from an arbitrary point, given obstacles.

    The Perception Integrity Map was, before this, pure frustum geometry: it
    asked whether a cell falls inside a camera's field of view and how many
    pixels it subtends, and nothing else. That produces the same six-petal
    rosette on every frame of every scene, because nothing in it depends on
    what is actually in front of the vehicle. A camera cannot see the road
    behind a parked van, and a map claiming otherwise is worse than no map --
    it reports coverage over exactly the cells where a pedestrian can be
    hiding.

    So visibility is ray-cast from the CAMERA's position, not the LiDAR's.
    That distinction matters: the cameras sit up to 1.7 m forward and 1 m to
    the side of the LiDAR, so their occlusion shadows fall differently, and
    computing them from the sensor origin would put every shadow in the wrong
    place.

    A cell is visible when no occupied cell lies nearer along the same bearing
    from the origin. `slack` in cells keeps an obstacle's own front face
    visible -- the van's surface is seen, it is the road behind it that is not.
    """
    occ = np.asarray(occ_mask, bool)
    n = occ.shape[0]
    ax = (np.arange(n) + 0.5) * res - rng_m
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    dx, dy = X - float(origin_xy[0]), Y - float(origin_xy[1])
    R = np.hypot(dx, dy)
    B = np.clip(((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * n_azimuth
                 ).astype(np.int64), 0, n_azimuth - 1)

    horizon = np.full(n_azimuth, np.inf)
    if occ.any():
        np.minimum.at(horizon, B[occ], R[occ])
        # An obstacle one cell wide subtends fewer than one azimuth bin at
        # range, so its shadow would be a bin-wide sliver with gaps either
        # side. Taking a running minimum over neighbouring bearings closes
        # those, which is what a solid object actually does to the light.
        k = 3
        horizon = np.min(np.stack(
            [np.roll(horizon, i) for i in range(-(k // 2), k // 2 + 1)]), axis=0)
    return R <= (horizon[B] + slack * res)
