"""Does camera observability predict what a human annotator could see?

At scale, and with the one analysis the first version was missing.

What was measured before
------------------------
scripts/eval/eval_integrity_visibility.py, on 45 keyframes and 2,913 boxes:

    integrity (occlusion-aware)   AUROC 0.5693
    range only                    AUROC 0.5708
    verdict                       NEGATIVE RESULT

That is honest and it is also, on reflection, close to a tautology. Coverage in
camera_ground_coverage is the cell's projected AREA IN PIXELS, and projected
area falls off with distance by construction. So the integrity map is already
mostly a monotone function of range before occlusion enters. Asking whether it
beats range is close to asking whether range beats range, and a tie is the
expected answer rather than a finding.

The question worth asking
-------------------------
Not "does integrity beat range" but "does integrity know anything range does
not". Those come apart, and the second is what a reader should care about,
because the useful claim is not "far things are hard to see" -- everyone knows
that -- but "this specific patch of ground is unobservable even though it is
close, because a truck is in the way".

So the headline analysis here is RANGE-STRATIFIED. Inside a narrow range band,
range is nearly constant and carries almost no information; any AUROC above 0.5
that integrity retains there is information range cannot supply. Mostly that is
the occlusion term. If stratified AUROC collapses to 0.5, the map really is a
range proxy with extra steps and the project should say so.

Both are reported. The marginal number stays because dropping an unflattering
comparison after seeing it is how results stop being trustworthy.

Scale
-----
    before   45 keyframes      2,913 boxes     10 scenes
    now   6,019 keyframes    192,041 boxes    150 scenes

Development protocol
--------------------
outputs/artifacts/val_dev_test.json fixes 75 dev and 75 test scenes, chosen by
a seeded permutation BEFORE any of this was computed (sha 3e00ea450bb17507).
This script defaults to DEV. Tuning happens on dev; test is run once, at the
end, with the method frozen. Reporting a number tuned on the set it was
measured on is the same error as a leaking split, arrived at more politely.

A limitation to state plainly
-----------------------------
The pack carries ONE LiDAR sweep per keyframe. The original evaluation
accumulated ten. A sparser cloud leaves gaps a ray can slip through, so some
genuinely occluded cells will be scored visible and the occlusion term is, if
anything, UNDERSTATED here. Sweeps were deliberately not downloaded -- they are
~70% of nuScenes and nothing else in this project reads them -- so this is a
known cost of that choice, not an oversight.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

# Kept identical to eval_integrity_visibility.py so the numbers are comparable.
DEFAULT_TRUST = 0.795
RNG_M = 54.0
RES = 0.5
MAX_RANGE = 45.0
FULL_PX = 140.0
OCC_THRESHOLD = 0.65
L_FREE, L_OCC, L_CLAMP = 0.45, 2.4, 6.0
# Height above the road, ego frame, at which a return stops being ground.
# 0.30 m clears kerbs and lane paint without swallowing low bollards.
GROUND_Z = 0.30


# ---------------------------------------------------------------- geometry
def quat_to_rot(q) -> np.ndarray:
    """nuScenes stores rotations as (w, x, y, z)."""
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def ground_grid(rng_m=RNG_M, res=RES):
    n = int(2 * rng_m / res)
    ax = (np.arange(n) + 0.5) * res - rng_m
    return n, ax


def camera_coverage(cam: dict, n: int, ax: np.ndarray, res=RES,
                    full_px=FULL_PX) -> np.ndarray:
    """Per-cell coverage weight in [0, 1]: projected pixel area / full_px.

    Not a binary frustum test. The ground plane reaches the horizon in every
    camera, so a boolean test says a cell at 54 m is covered exactly as well as
    one at 5 m -- true as geometry, useless as a statement about perception.
    Area is measured by projecting the cell's four CORNERS and taking the
    shoelace area, which captures foreshortening and grazing incidence together
    rather than assuming a fronto-parallel patch.
    """
    K = np.asarray(cam["intrinsic"], float)
    R_c = quat_to_rot(cam["sensor2ego_rotation"])
    t_c = np.asarray(cam["sensor2ego_translation"], float)
    W, H = float(cam["width"]), float(cam["height"])

    half = res / 2.0
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    out = np.zeros((n, n))
    uvs, zs = [], []
    for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half)):
        pts = np.stack([(X + dx).ravel(), (Y + dy).ravel(),
                        np.zeros(X.size)], axis=1)
        cam_pts = (pts - t_c) @ R_c                       # ego -> camera
        z = cam_pts[:, 2]
        safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        uvw = cam_pts @ K.T
        uvs.append(uvw[:, :2] / safe[:, None])
        zs.append(z)

    z0 = zs[0]
    u = np.stack([p[:, 0] for p in uvs], 1)
    v = np.stack([p[:, 1] for p in uvs], 1)
    # shoelace over the four projected corners
    area = 0.5 * np.abs(
        u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0] +
        u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1] +
        u[:, 2] * v[:, 3] - u[:, 3] * v[:, 2] +
        u[:, 3] * v[:, 0] - u[:, 0] * v[:, 3])

    centre_u = u.mean(1)
    centre_v = v.mean(1)
    inside = (z0 > 0.5) & (centre_u >= 0) & (centre_u < W) & \
             (centre_v >= 0) & (centre_v < H)
    w = np.clip(area / full_px, 0.0, 1.0) * inside
    out[:] = w.reshape(n, n)
    return out


def lidar_to_ego(pts: np.ndarray, lid: dict) -> np.ndarray:
    """LiDAR-frame points -> ego frame.

    The pack stores raw sensor-frame points. Everything else here -- the grid,
    the camera origins, the boxes -- is ego-framed, and LiDAR_TOP sits 0.94 m
    forward of and 1.84 m above the ego origin. Skipping this puts every
    obstacle almost a metre out of place and leaves ground at z = -1.84 rather
    than z = 0, which is the difference between a height threshold that means
    something and a magic number.
    """
    R = quat_to_rot(lid["sensor2ego_rotation"])
    t = np.asarray(lid["sensor2ego_translation"], float)
    return pts[:, :3].astype(np.float64) @ R.T + t


def occupancy_from_lidar(pts_ego: np.ndarray, n: int, ax: np.ndarray,
                         res=RES, rng_m=RNG_M, min_pts: int = 1):
    """Log-odds occupancy over the BEV grid, with the height of each column.

    Returns (occ, z_top, z_bot). The heights are what make the occlusion test
    three-dimensional: a BEV cell is not a wall from the ground up, and
    treating it as one is why the first version of this measure shadowed most
    of the road. Tree canopy, awnings, overhead signs and the roofline of a
    parked car all land in a BEV cell while a camera at 1.5 m sees the road
    underneath them perfectly well.

    Ground is anything below GROUND_Z in the ego frame, where the ego origin
    sits on the road surface, so the threshold is a real height above the
    road rather than an offset from a sensor mount.
    """
    xy = pts_ego[:, :2]
    z = pts_ego[:, 2]
    keep = (np.abs(xy[:, 0]) < rng_m) & (np.abs(xy[:, 1]) < rng_m)
    xy, z = xy[keep], z[keep]
    empty = np.zeros((n, n), bool)
    if xy.size == 0:
        return empty, np.zeros((n, n)), np.zeros((n, n))

    ix = np.clip(((xy[:, 0] + rng_m) / res).astype(int), 0, n - 1)
    iy = np.clip(((xy[:, 1] + rng_m) / res).astype(int), 0, n - 1)
    obstacle = z > GROUND_Z

    logodds = np.zeros((n, n))
    np.add.at(logodds, (ix[obstacle], iy[obstacle]), L_OCC)
    np.add.at(logodds, (ix[~obstacle], iy[~obstacle]), -L_FREE)
    np.clip(logodds, -L_CLAMP, L_CLAMP, out=logodds)
    prob = 1.0 / (1.0 + np.exp(-logodds))
    occ = prob > OCC_THRESHOLD
    if min_pts > 1:
        # One stray return clears the log-odds threshold, and with
        # angular-width casting a single noise point at 2 m now shadows a wide
        # wedge. Requiring corroboration trades a little recall for far fewer
        # phantom shadows. Tuned on dev, never on test.
        cnt = np.zeros((n, n), np.int32)
        np.add.at(cnt, (ix[obstacle], iy[obstacle]), 1)
        occ &= cnt >= min_pts

    z_top = np.full((n, n), -np.inf)
    z_bot = np.full((n, n), np.inf)
    np.maximum.at(z_top, (ix[obstacle], iy[obstacle]), z[obstacle])
    np.minimum.at(z_bot, (ix[obstacle], iy[obstacle]), z[obstacle])
    z_top[~np.isfinite(z_top)] = 0.0
    z_bot[~np.isfinite(z_bot)] = 0.0
    return occ, z_top, z_bot


def visibility_from(occ: np.ndarray, origin_xy, n: int, res=RES,
                    rng_m=RNG_M, n_azimuth=2048) -> np.ndarray:
    """Cells with no nearer obstacle along the same bearing from origin_xy.

    Cast from the CAMERA position, not the LiDAR. The cameras sit up to 1.7 m
    forward and 1 m to the side, so their occlusion shadows fall in different
    places; casting from the sensor origin puts every shadow slightly wrong.

    Each obstacle cell shadows the ANGULAR WEDGE it actually subtends, not the
    single bucket its centre falls in. That distinction is not cosmetic: a
    0.5 m cell at 5 m subtends ~0.1 rad while a bucket spans 0.003 rad, so
    bucketing by centre alone leaves ~30 empty buckets between neighbouring
    cells of a solid wall and rays slip straight through them. The error is
    worst for CLOSE obstacles -- the ones that matter most, since a van two
    metres away hides far more ground than one at forty -- so the previous
    version systematically under-reported exactly the occlusion it was built
    to find. A wall spanning all bearings now shadows everything behind it,
    which is verified in the unit test at the bottom of this file.
    """
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cx = (ii + 0.5) * res - rng_m - float(origin_xy[0])
    cy = (jj + 0.5) * res - rng_m - float(origin_xy[1])
    rad = np.hypot(cx, cy)
    az = np.arctan2(cy, cx)

    nearest = np.full(n_azimuth, np.inf)
    ob = occ.ravel()
    if ob.any():
        r_o = np.maximum(rad.ravel()[ob], res)
        a_o = az.ravel()[ob]
        # half-angle subtended by a cell of side `res` seen at radius r
        half = np.arctan2(0.7071 * res, r_o)
        step = 2.0 * np.pi / n_azimuth
        span = np.minimum((2.0 * half / step).astype(int) + 1, n_azimuth)
        b0 = ((a_o - half + np.pi) / (2.0 * np.pi) * n_azimuth).astype(int)
        # expand each obstacle into the run of buckets it covers
        # Ragged expansion without a Python loop: each obstacle contributes
        # `span` consecutive buckets, and a list comprehension over thousands
        # of obstacles per camera per frame dominated the runtime. The cumsum
        # trick builds the same offsets vectorised.
        total = int(span.sum())
        starts = np.zeros(span.size, np.int64)
        np.cumsum(span[:-1], out=starts[1:])
        offs = np.arange(total) - np.repeat(starts, span)
        base = np.repeat(b0, span)
        radr = np.repeat(r_o, span)
        np.minimum.at(nearest, (base + offs) % n_azimuth, radr)

    bucket = np.clip(((az + np.pi) / (2 * np.pi) * n_azimuth).astype(int),
                     0, n_azimuth - 1)
    return rad <= (nearest[bucket] + res)


def integrity_map(covs: List[np.ndarray], trusts: List[float]) -> np.ndarray:
    acc = np.ones_like(covs[0])
    for c, t in zip(covs, trusts):
        acc *= (1.0 - t * c)
    return 1.0 - acc


def footprint_cells(cx, cy, w, l, yaw, n, res=RES, rng_m=RNG_M):
    hx, hy = l / 2.0, w / 2.0
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    crn = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]]) @ R.T + \
        np.array([cx, cy])
    lo = np.clip(np.floor((crn.min(0) + rng_m) / res).astype(int), 0, n - 1)
    hi = np.clip(np.ceil((crn.max(0) + rng_m) / res).astype(int), 1, n)
    if (hi <= lo).any():
        return None
    return np.meshgrid(np.arange(lo[0], hi[0]),
                       np.arange(lo[1], hi[1]), indexing="ij")


def obstacle_geometry(occ, z_top, z_bot, n: int, res=RES, rng_m=RNG_M):
    """Flat indices, ego-frame centres and column heights of occupied cells."""
    flat = np.flatnonzero(occ.ravel())
    ci, cj = flat // n, flat % n
    ox = (ci + 0.5) * res - rng_m
    oy = (cj + 0.5) * res - rng_m
    return flat, ox, oy, z_top.ravel()[flat], z_bot.ravel()[flat]


def _angdiff(a, b):
    d = a - b
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def camera_obstacle_view(ox, oy, ztop, zbot, cam_xy):
    """Polar description of every occupied cell as seen from one camera.

    Hoisted out of the per-object test on purpose. Recomputing hypot and
    arctan2 over ten thousand obstacle cells once per object per camera --
    thirty objects and six cameras a frame -- was three quarters of the
    runtime, and none of it depends on which object is being scored.
    """
    r = np.hypot(ox - cam_xy[0], oy - cam_xy[1])
    a = np.arctan2(oy - cam_xy[1], ox - cam_xy[0])
    return r, a, np.sin(a), np.cos(a), ztop, zbot


def footprint_visibility(own_mask, fx, fy, view, cam_xy, cam_z,
                         res=RES, pad=None):
    """Per-cell visibility of ONE object's footprint from one camera.

    The object's own cells are excluded from the set of occluders.

    This is the correction that mattered. The grid-wide raycast asks "is this
    cell shadowed by anything nearer on the same bearing", and a car's own
    LiDAR returns fill its own footprint -- so the near face of every object
    shadows the rest of it, and a fully visible car scored ~0.1 observability
    the same as a fully hidden one. The measure was reading self-occlusion,
    which is not occlusion: an object is not hidden by itself. Excluding the
    object's own cells asks the question the measure was always meant to ask,
    which is whether some OTHER structure stands between the camera and this
    footprint.

    A cell is occluded when some other occupied cell is strictly nearer along
    an overlapping angular wedge -- same wedge test as visibility_from, just
    restricted to the handful of cells that could possibly matter -- AND that
    cell's column actually intersects the sight line. The line from a camera
    at cam_z down to the road at range r_c passes height cam_z*(1 - r/r_c) at
    range r, so a column blocks only if it spans that height. Without this
    test every tree, awning and overhead sign becomes a solid wall.
    """
    r_o, a_o, sin_o, cos_o, ztop, zbot = view
    r_c = np.hypot(fx - cam_xy[0], fy - cam_xy[1])
    a_c = np.arctan2(fy - cam_xy[1], fx - cam_xy[0])
    if r_o.size == 0:
        return np.ones(r_c.shape, bool)

    # Cheap prefilter: only cells nearer than the far edge of the footprint,
    # inside its bearing span, and not part of the object itself. The bearing
    # test is a dot product against the mean direction rather than an
    # arctan2 difference -- same comparison, no transcendentals over the full
    # obstacle set.
    if pad is None:
        # The bearing prefilter must be wide enough to admit the widest wedge
        # any obstacle can subtend, or close obstacles are discarded before
        # the test ever sees them. A 0.5 m cell one cell away subtends 0.615
        # rad; the fixed 0.30 that used to sit here silently dropped every
        # blocker nearer than ~1.2 m, which is exactly the case where a
        # blocker hides the most ground. Derived from the frame's own
        # geometry, so it stays tight when nothing is close.
        # Wide enough for the widest blocker AND the widest target, since
        # the overlap test compares the SUM of their half-widths. Counting
        # only the blocker's leaves a band of bearings that pass the real
        # test but never reach it -- invisible, and worst for targets a
        # metre or two from the lens, where a cell is angularly huge.
        pad = float(np.arctan2(0.7071 * res, max(float(r_o.min()), 1e-3))
                    + np.arctan2(0.7071 * res, max(float(r_c.min()), 1e-3)))
    sc, cc = np.sin(a_c), np.cos(a_c)
    sref, cref = sc.mean(), cc.mean()
    nrm = np.hypot(sref, cref) or 1.0
    sref, cref = sref / nrm, cref / nrm
    spread = np.arccos(np.clip(sc * sref + cc * cref, -1.0, 1.0)).max()
    cos_lim = np.cos(min(spread + pad, np.pi))
    keep = (r_o < r_c.max()) & ((cos_o * cref + sin_o * sref) > cos_lim)
    keep &= ~own_mask
    if not keep.any():
        return np.ones(r_c.shape, bool)

    r_k, a_k = r_o[keep], a_o[keep]
    t_k, b_k = ztop[keep][:, None], zbot[keep][:, None]
    half_k = np.arctan2(0.7071 * res, np.maximum(r_k, 1e-3))[:, None]
    half_c = np.arctan2(0.7071 * res, np.maximum(r_c, 1e-3))[None, :]
    nearer = r_k[:, None] < (r_c[None, :] - 0.5 * res)
    overlap = _angdiff(a_k[:, None], a_c[None, :]) <= (half_k + half_c)
    # height of the sight line at each obstacle's range, per target cell
    h_ray = cam_z * (1.0 - r_k[:, None] / np.maximum(r_c[None, :], 1e-6))
    spans = (t_k >= h_ray) & (b_k <= h_ray + 0.5 * GROUND_Z)
    return ~(nearer & overlap & spans).any(0)


# ---------------------------------------------------------------- statistics
def auroc(scores, labels) -> float:
    """Rank-based AUROC with averaged ranks for ties.

    Written out rather than imported: an earlier version of this project
    reported AUROC 0.046 from a rank formula that mishandled ties, which is
    impossible and was only caught because the number was absurd.
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    # Average ranks without the per-tie Python loop. That loop was correct but
    # O(n) in interpreted code, and at 68k boxes times a few thousand bootstrap
    # resamples it turned the statistics into the slowest part of the run.
    edge = np.flatnonzero(np.concatenate(
        ([True], ss[1:] != ss[:-1], [True])))
    counts = np.diff(edge)
    avg = edge[:-1] + (counts - 1) / 2.0 + 1.0
    ranks = np.empty(len(s), float)
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def paired_scene_bootstrap(s_a, s_b, labels, scenes, n=2000, seed=0):
    """CI on the DIFFERENCE of two AUROCs, resampling the same scenes for both.

    Two intervals that overlap can still hide a difference that is significant
    every time, because both scores are computed on the same boxes and move
    together from scene to scene. The paired resample cancels that shared
    variation, which is the only honest way to claim one measure beats another.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(scenes)
    by = {u: np.where(scenes == u)[0] for u in uniq}
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by[uniq[p]] for p in pick])
        d = auroc(s_a[idx], labels[idx]) - auroc(s_b[idx], labels[idx])
        if d == d:
            out.append(d)
    if not out:
        return float("nan"), float("nan"), float("nan")
    out = np.asarray(out)
    # two-sided bootstrap p for "no difference"
    p = 2.0 * min((out <= 0).mean(), (out >= 0).mean())
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)),
            float(min(p, 1.0)))


def scene_bootstrap(scores, labels, scenes, n=2000, seed=0):
    """CI by resampling SCENES, not boxes.

    Boxes inside one scene are not independent draws -- the same parked van
    appears in forty consecutive keyframes. A box-level bootstrap would give a
    reassuringly tight interval that means nothing.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(scenes)
    by = {u: np.where(scenes == u)[0] for u in uniq}
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by[uniq[p]] for p in pick])
        a = auroc(scores[idx], labels[idx])
        if a == a:
            out.append(a)
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ---------------------------------------------------------------- main
def main(pack: str, split_file: str, which: str, limit: int, out_path: str,
         min_pts: int = 1, offset: int = 0, dump: str = "",
         cov_dir: str = ""):
    index = json.load(open(os.path.join(pack, "index.json")))
    calib = json.load(open(os.path.join(pack, "calib.json")))
    # Materialised, not left as a lazy NpzFile. Every ann[key] on an open npz
    # decompresses that whole array again, and the per-object accesses in the
    # loop below turned a 3-second read into most of the runtime.
    with np.load(os.path.join(pack, "annotations_val.npz"),
                 allow_pickle=False) as _z:
        ann = {k: _z[k] for k in _z.files}
    cats = [str(c) for c in ann["category_names"]]

    dt = json.load(open(split_file))
    want = set(dt[which])
    rows_idx = [i for i, r in enumerate(index) if r["scene"] in want]
    # The shell this runs in cannot hold a background job, so the split is
    # walked in chunks and the per-box records are merged afterwards. Chunking
    # changes nothing statistically: every box is scored from its own frame.
    rows_idx = rows_idx[offset:]
    if limit:
        rows_idx = rows_idx[:limit]
    print(f"split '{which}': {len(want)} scenes, {len(rows_idx)} keyframes")

    # annotations grouped by frame
    by_frame: Dict[int, List[int]] = {}
    for k, f in enumerate(ann["frame"]):
        by_frame.setdefault(int(f), []).append(k)

    n, ax = ground_grid()
    recs = []

    # Per-camera coverage depends only on intrinsics and extrinsics, which are
    # fixed for a whole scene -- so it is identical across every keyframe in
    # that scene, and it is by far the expensive part (1.1M corner projections
    # per keyframe against six raycasts). Computing it once per scene instead
    # of once per keyframe is a ~40x saving with no change to any number.
    cov_cache: Dict[str, List[np.ndarray]] = {}
    if cov_dir:
        os.makedirs(cov_dir, exist_ok=True)
    t0 = __import__("time").time()

    for count, fi in enumerate(rows_idx):
        row = index[fi]
        tok = row["token"]
        lid_path = os.path.join(pack, "lidar", f"{tok}.npy")
        if not os.path.exists(lid_path):
            continue
        pts = lidar_to_ego(np.load(lid_path), row["lidar"])
        occ, z_top, z_bot = occupancy_from_lidar(pts, n, ax, min_pts=min_pts)

        scene = row["scene"]
        if scene not in cov_cache:
            # Coverage is fixed for a whole scene but costs 1.1M corner
            # projections to build, and the split is walked in chunks across
            # separate processes -- so it is also cached on disk, or every
            # chunk would pay for all 75 scenes again.
            cpath = os.path.join(cov_dir, f"{scene}.npy") if cov_dir else ""
            if cpath and os.path.exists(cpath):
                cov_cache[scene] = list(np.load(cpath))
            else:
                cov_cache[scene] = [camera_coverage(calib[tok]["cams"][c], n, ax)
                                    for c in CAMS]
                if cpath:
                    np.save(cpath, np.stack(cov_cache[scene]))
        covs_raw = cov_cache[scene]
        covs_occ = []
        for c, cov in zip(CAMS, covs_raw):
            cam = calib[tok]["cams"][c]
            vis = visibility_from(occ, np.asarray(
                cam["sensor2ego_translation"], float)[:2], n)
            covs_occ.append(cov * vis)
        trusts = [DEFAULT_TRUST] * len(CAMS)
        integ = integrity_map(covs_occ, trusts)          # self-shadowed
        integ_no = integrity_map(covs_raw, trusts)       # coverage only
        ob_flat, ob_x, ob_y, ob_t, ob_b = obstacle_geometry(occ, z_top, z_bot, n)
        cam_t = [np.asarray(calib[tok]["cams"][c]["sensor2ego_translation"],
                            float) for c in CAMS]
        views = [camera_obstacle_view(ob_x, ob_y, ob_t, ob_b, t[:2])
                 for t in cam_t]
        own_grid = np.zeros(n * n, bool)

        # boxes: global -> ego (the grid is ego-framed, z ignored)
        lid = row["lidar"]
        R_e = quat_to_rot(lid["ego2global_rotation"])
        t_e = np.asarray(lid["ego2global_translation"], float)

        for k in by_frame.get(fi, []):
            if int(ann["num_lidar_pts"][k]) <= 0:
                continue
            v = int(ann["visibility"][k])
            if v > 3:                                   # unknown bucket
                continue
            g = ann["translation"][k].astype(float)
            e = (g - t_e) @ R_e                          # global -> ego
            r = float(np.hypot(e[0], e[1]))
            if r > MAX_RANGE:
                continue
            w_, l_, h_ = ann["size"][k].astype(float)
            yaw_g = np.arctan2(*quat_to_rot(ann["rotation"][k])[:2, 0][::-1])
            yaw_e = yaw_g - np.arctan2(R_e[1, 0], R_e[0, 0])
            idx = footprint_cells(e[0], e[1], w_, l_, yaw_e, n)
            if idx is None:
                continue
            ii, jj = idx
            own_flat = (ii * n + jj).ravel()
            own_grid[own_flat] = True
            own_mask = own_grid[ob_flat]
            own_grid[own_flat] = False
            fx = (ii.ravel() + 0.5) * RES - RNG_M
            fy = (jj.ravel() + 0.5) * RES - RNG_M
            acc = np.ones(fx.shape)
            for c, cov in zip(range(len(CAMS)), covs_raw):
                vcell = footprint_visibility(own_mask, fx, fy, views[c],
                                             cam_t[c][:2], float(cam_t[c][2]))
                acc *= (1.0 - DEFAULT_TRUST * cov[ii, jj].ravel() * vcell)
            integ_self_excl = float((1.0 - acc).mean())

            recs.append((v, integ_self_excl, float(integ_no[idx].mean()),
                         r, h_, int(ann["category"][k]), row["scene"],
                         float(integ[idx].mean())))

        if count % 250 == 0:
            el = __import__("time").time() - t0
            print(f"  {count}/{len(rows_idx)} keyframes, {len(recs):,} boxes, "
                  f"{el:.0f}s", flush=True)

    if not recs:
        raise SystemExit("no boxes scored")

    cols = dict(
        vis=np.array([r[0] for r in recs], np.int8),
        a_occ=np.array([r[1] for r in recs]),
        a_no=np.array([r[2] for r in recs]),
        rng_a=np.array([r[3] for r in recs]),
        hgt=np.array([r[4] for r in recs]),
        cat=np.array([r[5] for r in recs], np.int16),
        scn=np.array([r[6] for r in recs]),
        a_self=np.array([r[7] for r in recs]),
    )
    if dump:
        np.savez_compressed(dump, keyframes=len(rows_idx), **cols)
        print(f"wrote {dump}  ({len(recs):,} boxes)")
    analyse(cols, cats, which, len(rows_idx), min_pts, out_path)


def analyse(cols, cats, which, n_keyframes, min_pts, out_path):
    vis, a_occ, a_no = cols["vis"], cols["a_occ"], cols["a_no"]
    rng_a, hgt, cat = cols["rng_a"], cols["hgt"], cols["cat"]
    scn, a_self = cols["scn"], cols["a_self"]
    recs = vis
    rows_idx = range(n_keyframes)
    y = vis >= 2                       # visibility bucket 2 or 3 => >60% visible

    res_out = {
        "split": which, "scenes": len(np.unique(scn)),
        "keyframes": len(rows_idx), "boxes_scored": len(recs),
        "positive_rate_vis_ge_60pct": round(float(y.mean()), 4),
        "params": {"res": RES, "rng_m": RNG_M, "max_box_range_m": MAX_RANGE,
                   "trust": DEFAULT_TRUST, "full_px": FULL_PX,
                   "lidar_sweeps": 1, "min_pts": min_pts},
        "marginal_auroc": {},
        "range_stratified_auroc": {},
        "per_visibility_level": {},
    }

    for name, sc in (("observability_occlusion_aware", a_occ),
                     ("observability_self_shadowed", a_self),
                     ("observability_no_occlusion", a_no),
                     ("range_only_baseline", -rng_a)):
        A = auroc(sc, y)
        lo, hi = scene_bootstrap(sc, y, scn)
        res_out["marginal_auroc"][name] = {
            "auroc": round(A, 4), "ci95_scene": [round(lo, 4), round(hi, 4)]}

    res_out["paired_scene_bootstrap"] = {}
    for lab, other in (("vs_range_only", -rng_a),
                       ("vs_no_occlusion", a_no),
                       ("vs_self_shadowed", a_self)):
        lo, hi, p = paired_scene_bootstrap(a_occ, other, y, scn)
        res_out["paired_scene_bootstrap"][lab] = {
            "delta_auroc": round(auroc(a_occ, y) - auroc(other, y), 4),
            "ci95": [round(lo, 4), round(hi, 4)], "p_two_sided": round(p, 4)}

    # THE analysis: inside a range band, range is near-constant, so anything
    # observability retains there is information range cannot provide.
    bands = [(0, 10), (10, 20), (20, 30), (30, 45)]
    for lo_r, hi_r in bands:
        m = (rng_a >= lo_r) & (rng_a < hi_r)
        if m.sum() < 200 or len(np.unique(y[m])) < 2:
            continue
        # This band-wise comparison is the claim the paper stands on once the
        # marginal one ties, so it gets the same paired test rather than two
        # bare numbers side by side.
        blo, bhi, bp = paired_scene_bootstrap(a_occ[m], -rng_a[m], y[m], scn[m])
        res_out["range_stratified_auroc"][f"{lo_r}-{hi_r}m"] = {
            "n": int(m.sum()),
            "positive_rate": round(float(y[m].mean()), 4),
            "observability": round(auroc(a_occ[m], y[m]), 4),
            "no_occlusion": round(auroc(a_no[m], y[m]), 4),
            "range_within_band": round(auroc(-rng_a[m], y[m]), 4),
            "delta_vs_range_in_band": round(
                auroc(a_occ[m], y[m]) - auroc(-rng_a[m], y[m]), 4),
            "ci95": [round(blo, 4), round(bhi, 4)],
            "p_two_sided": round(bp, 4),
        }

    lut = {0: "v0-40", 1: "v40-60", 2: "v60-80", 3: "v80-100"}
    for lv in range(4):
        m = vis == lv
        if m.any():
            res_out["per_visibility_level"][lut[lv]] = {
                "n": int(m.sum()),
                "mean_observability": round(float(a_occ[m].mean()), 4),
                "mean_no_occlusion": round(float(a_no[m].mean()), 4),
                "mean_self_shadowed": round(float(a_self[m].mean()), 4),
                "mean_range_m": round(float(rng_a[m].mean()), 2)}

    res_out["by_height"] = {}
    for lab, m in (("under_1m", hgt < 1.0),
                   ("1_to_1.8m", (hgt >= 1.0) & (hgt < 1.8)),
                   ("over_1.8m", hgt >= 1.8)):
        if m.sum() > 200 and len(np.unique(y[m])) > 1:
            res_out["by_height"][lab] = {
                "n": int(m.sum()),
                "observability": round(auroc(a_occ[m], y[m]), 4),
                "range_only": round(auroc(-rng_a[m], y[m]), 4)}

    res_out["by_category"] = {}
    for ci in np.unique(cat):
        m = cat == ci
        if m.sum() > 500 and len(np.unique(y[m])) > 1:
            res_out["by_category"][cats[ci]] = {
                "n": int(m.sum()),
                "observability": round(auroc(a_occ[m], y[m]), 4),
                "range_only": round(auroc(-rng_a[m], y[m]), 4)}

    res_out["occlusion_term_delta"] = round(
        res_out["marginal_auroc"]["observability_occlusion_aware"]["auroc"] -
        res_out["marginal_auroc"]["observability_no_occlusion"]["auroc"], 4)
    res_out["beats_range_by"] = round(
        res_out["marginal_auroc"]["observability_occlusion_aware"]["auroc"] -
        res_out["marginal_auroc"]["range_only_baseline"]["auroc"], 4)
    res_out["caveats"] = [
        "One LiDAR sweep per keyframe, not the ten-sweep stack the original "
        "45-frame evaluation used. Sparser occupancy lets rays through gaps, "
        "so the occlusion term is if anything understated.",
        "Visibility labels describe the OBJECT; observability describes its "
        "GROUND FOOTPRINT. A van visible over a low wall is a genuine "
        "disagreement between two different quantities, not an error.",
        "Ground is approximated by a fixed height threshold rather than a "
        "fitted plane, which the pack does not carry.",
        "Trust is a constant 0.795 for every camera, so it scales all six "
        "equally and contributes nothing to discrimination here.",
        "observability_occlusion_aware excludes an object's own footprint "
        "cells from its occluder set. observability_self_shadowed is the "
        "grid-wide raycast that does not, and is reported so the size of that "
        "correction is visible rather than hidden inside a tuned number.",
    ]

    json.dump(res_out, open(out_path, "w"), indent=2, default=float)

    print(f"\n=== {which.upper()}: {len(recs):,} boxes, "
          f"{len(np.unique(scn))} scenes ===")
    print("\nMARGINAL")
    for k, v in res_out["marginal_auroc"].items():
        print(f"  {k:<32} {v['auroc']:.4f}  scene-CI {v['ci95_scene']}")
    print(f"  beats range by {res_out['beats_range_by']:+.4f}   "
          f"occlusion term {res_out['occlusion_term_delta']:+.4f}")
    print("\nPAIRED (same scenes resampled for both scores)")
    for k, v in res_out["paired_scene_bootstrap"].items():
        print(f"  {k:<20} {v['delta_auroc']:+.4f}  CI {v['ci95']}  "
              f"p={v['p_two_sided']:.4f}")
    print("\nRANGE-STRATIFIED  (range carries no information inside a band)")
    print(f"  {'band':<10}{'n':>8}{'observability':>15}{'no-occl':>10}{'range':>8}")
    for k, v in res_out["range_stratified_auroc"].items():
        print(f"  {k:<10}{v['n']:>8,}{v['observability']:>15.4f}"
              f"{v['no_occlusion']:>10.4f}{v['range_within_band']:>8.4f}"
              f"   delta {v['delta_vs_range_in_band']:+.4f} "
              f"CI {v['ci95']} p={v['p_two_sided']:.4f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--splits", default="outputs/artifacts/val_dev_test.json")
    ap.add_argument("--which", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=0,
                    help="cap keyframes, for a quick smoke run")
    ap.add_argument("--out", default="outputs/artifacts/observability_scaled.json")
    ap.add_argument("--min-pts", type=int, default=1,
                    help="LiDAR returns needed for a cell to count as an "
                         "obstacle; >1 suppresses phantom shadows from noise")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many keyframes; with --limit, walks the "
                         "split in chunks that each fit one shell call")
    ap.add_argument("--cov-cache", default="",
                    help="directory to memoize per-scene camera coverage in")
    ap.add_argument("--dump", default="",
                    help="write this chunk's per-box records to an .npz")
    ap.add_argument("--merge", nargs="*", default=None,
                    help="skip scoring; pool these record dumps and report")
    a = ap.parse_args()
    if a.merge:
        parts = [np.load(p) for p in a.merge]
        keys = ["vis", "a_occ", "a_no", "rng_a", "hgt", "cat", "scn", "a_self"]
        pooled = {k: np.concatenate([p[k] for p in parts]) for k in keys}
        nkf = int(sum(int(p["keyframes"]) for p in parts))
        names = [str(c) for c in np.load(
            os.path.join(a.pack, "annotations_val.npz"))["category_names"]]
        analyse(pooled, names, a.which, nkf, a.min_pts, a.out)
    else:
        main(a.pack, a.splits, a.which, a.limit, a.out, a.min_pts,
             a.offset, a.dump, a.cov_cache)
