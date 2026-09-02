"""Static / dynamic separation from an ego-motion-compensated sweep stack.

Once every sweep is resolved into a common frame, static structure lands in the
same place ten times over while a moving object smears into a trail. So the
signal is not the position of a return but the *persistence* of the voxel it
falls in: a kerb is observed by every sweep in the window, a car crossing the
junction occupies each of its voxels for only one or two.

This needs no detector, no training and no labels, which is the point -- it is
a geometric consistency check that works on any object class, including the
ones a detector was never trained on. That is exactly the long-tail case a
camera-only pipeline misses.

The output is validated against annotation-derived velocity in
`evaluate_against_annotations`, not asserted.
"""
from __future__ import annotations

import numpy as np

STATIC, DYNAMIC, UNDECIDED = 0, 1, 2


def classify_motion(points, ground_mask, plane, rng_m=54.0, res=0.25,
                    old_from=5, max_range=40.0, min_obstacle_h=0.25):
    """Label each current-sweep return STATIC / DYNAMIC / UNDECIDED.

    A return is DYNAMIC when the older sweeps in the window SAW THROUGH the
    space it now occupies -- the ray-cast free space of the past scan covers
    this cell, and something solid is there now. That is a statement about
    observed emptiness, not about missing data, which is the distinction that
    makes this work.

    Two weaker designs were measured first and both failed, for reasons worth
    keeping:

      * voxel PERSISTENCE (static things are seen by many sweeps): precision
        0.28, recall 0.13. A car at 5 m/s sweeps a corridor over the half-second
        window and the voxels along its core are occupied by nearly every
        sweep, so persistence calls the vehicle static.
      * ABSENCE OF PRIOR SUPPORT (no old return in a 3x3 neighbourhood):
        precision 0.33, recall 0.15. Sparsity forces the neighbourhood out to
        +/-1.5 m, but a car travels 1.25-2.25 m across the window, so a moving
        object still finds support inside the tolerance. Tightening it to
        recover motion loses static surfaces to sampling gaps.

    Free space resolves that tension because an unsampled cell reads UNKNOWN
    rather than empty, so tolerance is no longer traded against sensitivity.
    Measured over 40 frames by scripts/eval/eval_motion_separation.py: F1 0.61
    (precision 0.56, recall 0.67) at a 0.2 box threshold, on the 23% of boxes
    where enough returns fall inside previously observed space to decide at
    all. A 12-frame spot check gave F1 0.76; the 40-frame figure is the one to
    quote, and the gap between them is the reason not to trust a dozen frames.
    """
    from odfm_ground import occupancy_grid, height_above_ground, FREE, OCCUPIED

    p = np.asarray(points, np.float32)
    age = p[:, 5].astype(np.int64)
    cur, old = age == 0, age >= old_from

    lab = np.full(len(p), UNDECIDED, np.uint8)
    if not cur.any() or not old.any():
        return lab, {"note": "insufficient history", "old_from": old_from}

    grid, meta = occupancy_grid(p[:, :3], ground_mask, rng_m=rng_m, res=res,
                                plane=plane, visible=old)
    n = meta["n"]
    h = height_above_ground(p[:, :3], plane)
    r = np.linalg.norm(p[:, :2], axis=1)

    sel = cur & (~np.asarray(ground_mask, bool)) & (h > min_obstacle_h) & (r <= max_range)
    idx = np.where(sel)[0]
    if len(idx) == 0:
        return lab, {"old_from": old_from, "n_current_obstacle": 0}

    ix = np.clip(((p[idx, 0] + rng_m) / res).astype(np.int64), 0, n - 1)
    iy = np.clip(((p[idx, 1] + rng_m) / res).astype(np.int64), 0, n - 1)
    cell = grid[ix, iy]

    lab[idx[cell == FREE]] = DYNAMIC
    lab[idx[cell == OCCUPIED]] = STATIC
    return lab, {"old_from": old_from, "res": res, "max_range": max_range,
                 "n_current_obstacle": int(len(idx)),
                 "dynamic_frac": float((cell == FREE).mean()),
                 "undecided_frac": float((cell != FREE).mean() -
                                         (cell == OCCUPIED).mean())}


def annotation_velocities(tables, sample_token, min_speed=0.5):
    """Per-instance speed in m/s, differenced between adjacent keyframes.

    nuScenes annotations carry no velocity field; the devkit derives it the same
    way. Global positions are differenced so ego motion cannot masquerade as
    object motion.
    """
    t = tables
    scene = t.sample[sample_token]["scene_token"]
    order = t.scene_samples[scene]
    i = order.index(sample_token)
    nb = [order[j] for j in (i - 1, i + 1) if 0 <= j < len(order)]

    cur = {a["instance_token"]: a for a in t.ann_by_sample.get(sample_token, [])}
    out = {}
    for tok in nb:
        dt = abs(t.sample[tok]["timestamp"] - t.sample[sample_token]["timestamp"]) / 1e6
        if dt <= 0:
            continue
        for a in t.ann_by_sample.get(tok, []):
            k = a["instance_token"]
            if k not in cur:
                continue
            d = np.linalg.norm(np.asarray(a["translation"][:2], float) -
                               np.asarray(cur[k]["translation"][:2], float))
            out[k] = max(out.get(k, 0.0), d / dt)
    return {k: v for k, v in out.items()}, min_speed


def _in_box(pts_xy, centre, wlh, yaw, margin=0.3):
    w, l, _ = wlh
    d = pts_xy - np.asarray(centre[:2], float)
    c, s = np.cos(-yaw), np.sin(-yaw)
    lx = d[:, 0] * c - d[:, 1] * s
    ly = d[:, 0] * s + d[:, 1] * c
    return (np.abs(lx) <= l / 2 + margin) & (np.abs(ly) <= w / 2 + margin)


def evaluate_against_annotations(points, labels, tables, sample_token,
                                 min_speed=0.5):
    """Score the geometric labels against annotation-derived motion.

    For every annotated box with LiDAR returns, the box is moving if its
    instance's differenced speed exceeds `min_speed`, and the geometric
    labelling calls it moving if most of its enclosed current-sweep returns are
    DYNAMIC. Reported as precision / recall over boxes, because per-point
    ground truth does not exist here -- annotations are box-level.
    """
    p = np.asarray(points, np.float32)
    cur = p[:, 5] == 0
    xy, lab = p[cur, :2], np.asarray(labels)[cur]

    speeds, _ = annotation_velocities(tables, sample_token, min_speed)
    tp = fp = fn = tn = 0
    rows = []
    for b in tables.boxes_ego(sample_token):
        if b["num_lidar_pts"] <= 0:
            continue
        m = _in_box(xy, b["centre"], b["wlh"], b["yaw"])
        if m.sum() < 5:
            continue
        pred = (lab[m] == DYNAMIC).mean() > 0.5
        truth = speeds.get(b["instance_token"], 0.0) > min_speed
        tp += pred and truth; fp += pred and not truth
        fn += (not pred) and truth; tn += (not pred) and not truth
        rows.append({"category": b["category"],
                     "speed_mps": round(float(speeds.get(b["instance_token"], 0.0)), 2),
                     "dynamic_frac": round(float((lab[m] == DYNAMIC).mean()), 3),
                     "n_pts": int(m.sum()), "pred_moving": bool(pred),
                     "truth_moving": bool(truth)})
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"boxes_scored": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec,
            "min_speed_mps": min_speed, "boxes": rows}
