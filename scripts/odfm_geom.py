"""Camera projection geometry and the Perception Integrity Map.

Two jobs:

  1. Move points and boxes between the LiDAR-ego frame and each camera's image
     plane, so LiDAR depth and 3D boxes can be drawn on the camera views and
     the two sensors can be checked against each other.

  2. Compute the Perception Integrity Map -- for every cell of the ground
     plane, how much trustworthy camera evidence covers it:

         integrity(cell) = 1 - prod_i (1 - trust_i * covers_i(cell))

     This is the noisy-OR of independent evidence. A cell seen by two cameras
     of trust 0.5 reaches 0.75, not 1.0; a cell seen only by a blinded camera
     inherits that camera's low trust however well-framed it is; a cell outside
     every frustum is 0 regardless of how healthy the rig is. The point is to
     separate "nothing is there" from "nothing can be seen there", which a
     detection count alone cannot express.

     The independence assumption is the weak part and is stated plainly: two
     cameras on the same vehicle share weather, share a power rail, and share
     the sun. Their failures are correlated, so noisy-OR is an upper bound on
     integrity, not an unbiased estimate.
"""
from __future__ import annotations

import numpy as np

from odfm_tables import quat_to_rot


def ego_to_cam(pts_ego, rec) -> np.ndarray:
    """Ego frame -> camera frame. `rec` is an odfm_tables sensor_record."""
    R = quat_to_rot(rec["cal_rot"])
    t = np.asarray(rec["cal_trans"], dtype=np.float64)
    return (np.asarray(pts_ego, np.float64) - t) @ R


def project_to_image(pts_ego, rec, min_depth=0.5):
    """Project ego-frame points into a camera.

    Returns (uv, depth, valid) with uv shaped (N, 2). `valid` requires positive
    depth AND being inside the image: a pinhole model happily produces finite
    pixel coordinates for points behind the camera, which is the classic way a
    projection overlay ends up with objects mirrored into the sky.
    """
    cam = ego_to_cam(pts_ego, rec)
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uvw = cam @ rec["intrinsic"].T
    uv = uvw[:, :2] / safe[:, None]
    W, H = rec["width"], rec["height"]
    valid = (z > min_depth) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & \
            (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, z, valid


def box_corners_ego(box) -> np.ndarray:
    """Eight corners of a 3D box in the ego frame, in the order
    (front face 0-3, back face 4-7) so edges can be drawn by index."""
    w, l, h = box["wlh"]
    x = np.array([l, l, l, l, -l, -l, -l, -l], np.float64) / 2
    y = np.array([w, -w, -w, w, w, -w, -w, w], np.float64) / 2
    z = np.array([h, h, -h, -h, h, h, -h, -h], np.float64) / 2
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # the annotation centre is the box centre in x/y but sits at mid-height in
    # z, so corners are offset symmetrically about it in all three axes
    return (np.stack([x, y, z], axis=1) @ R.T) + np.asarray(box["centre"])


def camera_ground_coverage(rec, rng_m=54.0, res=0.5, z_ground=0.0,
                           soft=True, full_px=140.0):
    """Per-cell coverage weight for one camera over a BEV grid.

    With soft=False this is a boolean frustum-intersection test: is the cell
    centre inside the image with positive depth.

    That binary test flatters the rig badly. The ground plane runs to the
    horizon in every camera, so a cell 54 m away passes it exactly as a cell
    5 m away does -- measured, the six-camera union then "covers" 99.2% of a
    108 m square, which is true as geometry and useless as a statement about
    what can be perceived. A distant cell is not merely far, it is viewed at
    grazing incidence, so it projects to a sliver a few pixels tall.

    So with soft=True the weight is the cell's actual projected AREA in pixels,
    normalised against `full_px` and clipped to 1. Area is measured by
    projecting the cell's four corners rather than scaling by depth, which
    captures foreshortening and grazing incidence together instead of assuming
    a fronto-parallel patch. A cell that images to fewer pixels contributes
    proportionally less evidence, which is what a detector actually experiences.

    Returns (weights, meta) with weights float in [0, 1].
    """
    n = int(2 * rng_m / res)
    ax = (np.arange(n) + 0.5) * res - rng_m
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    flat = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_ground)], axis=1)
    uv, z, valid = project_to_image(flat, rec)
    meta = {"n": n, "res": res, "rng_m": rng_m, "soft": soft, "full_px": full_px}
    if not soft:
        return valid.reshape(n, n), meta

    h = res / 2.0
    corners = []
    for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h)):
        c = flat.copy()
        c[:, 0] += dx
        c[:, 1] += dy
        cuv, _, _ = project_to_image(c, rec)
        corners.append(cuv)
    c = np.stack(corners, axis=1)                     # (N, 4, 2)
    # shoelace area of the projected quadrilateral
    x1, y1 = c[:, :, 0], c[:, :, 1]
    x2, y2 = np.roll(x1, -1, axis=1), np.roll(y1, -1, axis=1)
    area = 0.5 * np.abs((x1 * y2 - x2 * y1).sum(axis=1))

    w = np.clip(area / float(full_px), 0.0, 1.0)
    w[~valid] = 0.0
    return w.reshape(n, n), meta


def integrity_map(coverages, trusts) -> np.ndarray:
    """Noisy-OR combination of per-camera coverage weighted by per-camera trust.

    coverages: list of (n, n) coverage weights in [0, 1] (boolean masks work
    too). trusts: matching list of floats in [0, 1]. Returns a float (n, n) map
    in [0, 1].
    """
    if not coverages:
        raise ValueError("integrity_map needs at least one camera")
    acc = np.ones_like(np.asarray(coverages[0], np.float64))
    for cov, tr in zip(coverages, trusts):
        acc *= (1.0 - float(tr) * np.asarray(cov, np.float64))
    return 1.0 - acc


def integrity_stats(integ, occupancy=None, free_value=1) -> dict:
    """Summarise an integrity map, and -- when an occupancy grid is supplied --
    restrict the summary to cells the LiDAR reports as free.

    That restriction is the number that matters operationally. Mean integrity
    over the whole square is dominated by cells behind the vehicle and beyond
    the sensors, which no planner will ever route through. Mean integrity over
    *drivable* space answers the question actually being asked: of the road I
    could drive into, how much of it am I seeing with cameras I trust?
    """
    out = {
        "mean": float(integ.mean()),
        "median": float(np.median(integ)),
        "frac_below_0.3": float((integ < 0.3).mean()),
        "frac_above_0.7": float((integ > 0.7).mean()),
    }
    if occupancy is not None:
        free = np.asarray(occupancy) == free_value
        if free.any():
            out["mean_over_free_space"] = float(integ[free].mean())
            out["free_cells_below_0.3"] = float((integ[free] < 0.3).mean())
    return out
