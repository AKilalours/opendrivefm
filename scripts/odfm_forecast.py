"""Occupancy forecasting: where will the occupied cells be in 0.5 / 1.0 / 1.5 s.

Deliberately geometric, and labelled as such. No learned world model is
involved -- the repo's GPT-2 trajectory head cannot be run on this machine
(PyTorch's package index is unreachable from it), and inventing numbers for a
model that did not run would be worse than having none. What this provides is
the honest floor that any learned forecaster has to beat, measured against real
future LiDAR rather than against itself.

Two forecasters:

  PERSISTENCE   the world does not change. Every occupied cell stays put.
                This is the baseline, and it is a stronger one than it sounds:
                most of a street is parked cars, kerbs and buildings.

  CONSTANT-VELOCITY  static structure persists; returns labelled dynamic are
                clustered, given a velocity estimated from where those clusters
                were in earlier sweeps, and advected. Velocity comes from the
                LiDAR alone -- never from annotations, which would be using the
                answer to compute the prediction.

Both are scored on the cells where the future scan actually observed something,
because crediting a forecaster for correctly predicting "unknown" would reward
it for the sensor's blind spots.
"""
from __future__ import annotations

import numpy as np

import odfm_ground as G
import odfm_lidar as L

SWEEP_DT = 0.05           # LIDAR_TOP runs at 20 Hz


def cluster_dynamic(points, motion, res=0.6, min_pts=8, max_gap=1):
    """Connected components over the dynamic returns of the current sweep.

    Grid-based rather than a proper density clustering: at 0.6 m a vehicle is a
    handful of connected cells and two vehicles a metre apart stay separate,
    which is all this needs. Returns a list of index arrays into `points`.
    """
    cur = np.where((points[:, 5] == 0) & (motion == 1))[0]
    if len(cur) == 0:
        return []
    q = np.floor(points[cur, :2] / res).astype(np.int64)
    key = q[:, 0] * 1000003 + q[:, 1]
    uniq, inv = np.unique(key, return_inverse=True)
    cells = {int(k): i for i, k in enumerate(uniq)}
    qs = np.stack([uniq // 1000003, uniq % 1000003], axis=1)
    # recover signed coordinates: the hash is only injective for the range used
    qx = np.floor(points[cur, 0] / res).astype(np.int64)
    qy = np.floor(points[cur, 1] / res).astype(np.int64)
    coord = {}
    for i, (a, b) in enumerate(zip(qx, qy)):
        coord[(int(a), int(b))] = coord.get((int(a), int(b)), [])
        coord[(int(a), int(b))].append(i)

    seen, groups = set(), []
    for c in coord:
        if c in seen:
            continue
        stack, comp = [c], []
        seen.add(c)
        while stack:
            cc = stack.pop()
            comp.extend(coord[cc])
            for dx in range(-max_gap, max_gap + 1):
                for dy in range(-max_gap, max_gap + 1):
                    nb = (cc[0] + dx, cc[1] + dy)
                    if nb in coord and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(comp) >= min_pts:
            groups.append(cur[np.array(comp)])
    return groups


def estimate_velocity(points, group, obstacle=None, old_from=5, search_r=4.0,
                      max_speed=30.0, min_match=5):
    """Velocity of one cluster, from where its returns were in earlier sweeps.

    The old-sweep points it matches against MUST be restricted to returns that
    are themselves plausibly the moving object. An earlier version searched all
    returns within the radius, which in a street means mostly road surface and
    kerb: the "previous centroid" was really the centroid of the local static
    structure, so the velocity was noise. It showed as 183 of 183 clusters
    receiving a velocity -- a match rate that should have been suspicious on
    its own -- and the resulting forecast lost to persistence by 9% IoU while
    leaving recall untouched, which is the signature of moving cells to the
    wrong place rather than to a better one.

    Restricting to old returns that are non-ground and inside the obstacle
    height band is the fix. Note the motion labels CANNOT be used for this:
    classify_motion only labels the current sweep, so every old return is
    UNDECIDED and a "not static" test passes all of them -- an attempted fix
    that changed the match rate not at all, which is how it was caught.
    Clusters with no obstacle support get no velocity and stay where they are,
    which is the correct fallback.

    Known bias, not corrected: LiDAR sees a vehicle's near face, so a changing
    aspect shifts the centroid even for a stationary object. `search_r` bounds
    it and `max_speed` rejects nonsense.
    """
    p = points
    c_new = p[group, :2].mean(axis=0)
    old = np.where(p[:, 5] >= old_from)[0]
    if obstacle is not None:
        old = old[np.asarray(obstacle, bool)[old]]
    if len(old) == 0:
        return None
    d = np.linalg.norm(p[old, :2] - c_new, axis=1)
    near = old[d < search_r]
    if len(near) < min_match:
        return None
    c_old = p[near, :2].mean(axis=0)
    dt = SWEEP_DT * (p[near, 5].mean() - p[group, 5].mean())
    if dt <= 1e-3:
        return None
    v = (c_new - c_old) / dt
    if np.linalg.norm(v) > max_speed:
        return None
    return v


def forecast_occupancy(prob, points, motion, plane, horizon_s, rng_m=54.0,
                       res=0.20, mode="constant_velocity", occ_thresh=0.65,
                       ego_v=None, ground_mask=None, z_band=(0.25, 2.5)):
    """Forecast the occupied set `horizon_s` into the future.

    `ego_v` shifts the whole grid to account for the vehicle having moved by
    the time the future arrives. It is only correct when the forecast is being
    compared against ground truth expressed in the FUTURE ego frame. When
    ground truth is resolved back into the present ego frame -- which is what
    eval_occupancy_forecast.py does -- static structure does not move in that
    frame and this must be left off. Applying it there double-compensates:
    measured, it drove persistence IoU from 0.229 down to 0.008 at 0.2 m, by
    sliding every prediction 3.9 m away from where it belonged.
    """
    n = prob.shape[0]
    occ = prob > occ_thresh
    out = np.zeros_like(occ)

    def shift(mask_or_idx, dx, dy):
        sx = int(round(dx / res))
        sy = int(round(dy / res))
        return np.roll(np.roll(mask_or_idx, sx, axis=0), sy, axis=1)

    obstacle = None
    if ground_mask is not None:
        h = G.height_above_ground(points[:, :3], plane)
        obstacle = (~np.asarray(ground_mask, bool)) & (h > z_band[0]) & (h < z_band[1])

    dyn_cells = np.zeros_like(occ)
    moved = np.zeros_like(occ)
    n_clusters = n_moved = 0

    if mode == "constant_velocity":
        for grp in cluster_dynamic(points, motion):
            n_clusters += 1
            v = estimate_velocity(points, grp, obstacle=obstacle)
            ix = np.clip(((points[grp, 0] + rng_m) / res).astype(int), 0, n - 1)
            iy = np.clip(((points[grp, 1] + rng_m) / res).astype(int), 0, n - 1)
            m = np.zeros_like(occ)
            m[ix, iy] = True
            dyn_cells |= m
            if v is None:
                moved |= m
                continue
            n_moved += 1
            moved |= shift(m, v[0] * horizon_s, v[1] * horizon_s)
        out = (occ & ~dyn_cells) | moved
    else:
        out = occ.copy()

    if ego_v is not None:
        out = shift(out, -ego_v[0] * horizon_s, -ego_v[1] * horizon_s)

    return out, {"clusters": n_clusters, "clusters_with_velocity": n_moved}
