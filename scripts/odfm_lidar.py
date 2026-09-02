"""Real nuScenes LIDAR_TOP loading, multi-sweep aggregation and pose maths.

No nuscenes-devkit, no pyquaternion, no torch. Everything here is numpy over
the raw `.pcd.bin` files and the pose chain already captured in
`outputs/artifacts/lidar_manifest.json` (written by precompute_lidar_manifest.py).

Frame conventions, stated once because every sign error this file could have
comes from getting one of them wrong:

  * a `.pcd.bin` row is (x, y, z, intensity, ring_index), float32, in the
    LIDAR_TOP *sensor* frame.
  * `cal_rot` / `cal_trans` take sensor -> ego  (calibrated_sensor).
  * `ego_rot` / `ego_trans` take ego -> global  (ego_pose).
  * quaternions are nuScenes order, (w, x, y, z).
  * the nuScenes ego frame is x-forward, y-left, z-up.

To put a sweep captured at time t into the ego frame of the keyframe at time
t_ref you compose all three: sensor_t -> ego_t -> global -> ego_ref. That last
leg is what makes accumulated sweeps line up instead of smearing; skipping it
is the usual reason a "10-sweep" BEV looks like a blurred single sweep.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/artifacts/lidar_manifest.json"

# The ego vehicle's own bodywork returns. nuScenes does not remove them.
#
# These must be excluded by FOOTPRINT, in the capture-time ego frame, not by
# radius in the sensor frame. Two reasons, both of which bit this file:
#   * the sensor sits at (0.94, 0.00, 1.84) in the ego frame, so a radius
#     around the sensor is not a radius around the car -- self-returns were
#     measured out to 3.0 m along +x while a 1.0 m sensor-frame cut kept them.
#   * left in, they read as an obstacle at ~0.6 m on nearly every bearing, so
#     the occupancy ray-cast shadowed the ego vehicle behind itself and the
#     grid came out 94% unknown.
# Extents are the nuScenes Renault Zoe footprint about the rear-axle ego
# origin, with a small margin.
EGO_BOX_X = (-1.4, 3.3)
EGO_BOX_Y = (-1.0, 1.0)


def quat_to_rot(q) -> np.ndarray:
    """(w, x, y, z) -> 3x3 rotation matrix. nuScenes quaternion order."""
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float64)


def yaw_from_quat(q) -> float:
    """Heading about +z, radians. Reads it off the rotated x-axis rather than
    using an euler formula, so it stays correct under the roll/pitch the ego
    pose actually carries."""
    R = quat_to_rot(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def load_manifest(path=MANIFEST) -> dict:
    with open(path) as fh:
        return json.load(fh)


def load_bin(path) -> np.ndarray:
    """Read a LIDAR_TOP .pcd.bin as (N, 5): x, y, z, intensity, ring."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    a = np.fromfile(str(p), dtype=np.float32)
    if a.size % 5 != 0:
        raise ValueError(f"{p}: {a.size} floats is not a multiple of 5")
    return a.reshape(-1, 5)


def _sensor_to_global(pts_xyz: np.ndarray, rec: dict) -> np.ndarray:
    R_cal = quat_to_rot(rec["cal_rot"])
    t_cal = np.asarray(rec["cal_trans"], dtype=np.float64)
    R_ego = quat_to_rot(rec["ego_rot"])
    t_ego = np.asarray(rec["ego_trans"], dtype=np.float64)
    p = pts_xyz @ R_cal.T + t_cal          # sensor -> ego at capture time
    return p @ R_ego.T + t_ego             # ego -> global


def _global_to_ego(pts_xyz: np.ndarray, ref: dict) -> np.ndarray:
    R_ego = quat_to_rot(ref["ego_rot"])
    t_ego = np.asarray(ref["ego_trans"], dtype=np.float64)
    return (pts_xyz - t_ego) @ R_ego       # inverse rotation == right-multiply by R


def load_sweep_stack(sample_token: str, manifest: dict, n_sweeps: int = 10,
                     drop_self_returns: bool = True) -> np.ndarray:
    """Keyframe plus up to `n_sweeps - 1` prior sweeps, all resolved into the
    keyframe's ego frame.

    Returns (N, 6): x, y, z, intensity, ring, sweep_age
    where sweep_age is 0 for the keyframe and increases into the past. Keeping
    age per point is what lets the renderer fade history instead of drawing a
    uniform smear that hides which returns are current.
    """
    ref = manifest[sample_token]
    recs = [ref] + list(ref.get("sweeps", []))[: max(0, n_sweeps - 1)]

    out = []
    for age, rec in enumerate(recs):
        raw = load_bin(rec["path"])
        xyz = raw[:, :3].astype(np.float64)

        # sensor -> the ego frame AT CAPTURE TIME. Self-returns are fixed
        # relative to the vehicle, so this is the only frame in which a fixed
        # footprint removes them; in the reference ego frame an older sweep's
        # self-returns land wherever the car used to be and smear into a trail.
        cap = xyz @ quat_to_rot(rec["cal_rot"]).T + np.asarray(
            rec["cal_trans"], dtype=np.float64)
        if drop_self_returns:
            keep = ~((cap[:, 0] > EGO_BOX_X[0]) & (cap[:, 0] < EGO_BOX_X[1]) &
                     (cap[:, 1] > EGO_BOX_Y[0]) & (cap[:, 1] < EGO_BOX_Y[1]))
            raw, xyz, cap = raw[keep], xyz[keep], cap[keep]

        if age == 0:
            ego = cap
        else:
            ego = _global_to_ego(_sensor_to_global(xyz, rec), ref)
        block = np.empty((ego.shape[0], 6), dtype=np.float32)
        block[:, :3] = ego
        block[:, 3] = raw[:, 3]            # intensity
        block[:, 4] = raw[:, 4]            # ring
        block[:, 5] = age
        out.append(block)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 6), np.float32)


# --------------------------------------------------------------------------
# ground-truth boxes, in the keyframe ego frame, with TRUE yaw
# --------------------------------------------------------------------------

def load_annotations(sample_token: str, ref: dict, tables_dir=None,
                     _cache={}) -> list:
    """3D boxes for one keyframe, transformed global -> ego.

    nuScenes stores annotations in the global frame; the yaw that matters for a
    BEV is the box heading *relative to the ego*, which is the global box yaw
    minus the ego yaw. Deriving it this way is exact, and replaces the
    cross-frame nearest-match heading estimate used before the manifest existed.
    """
    tables = Path(tables_dir) if tables_dir else ROOT / "data/nuscenes/v1.0-mini"
    key = str(tables)
    if key not in _cache:
        with open(tables / "sample_annotation.json") as fh:
            anns = json.load(fh)
        with open(tables / "instance.json") as fh:
            inst = {i["token"]: i for i in json.load(fh)}
        with open(tables / "category.json") as fh:
            cats = {c["token"]: c["name"] for c in json.load(fh)}
        by_sample = {}
        for a in anns:
            by_sample.setdefault(a["sample_token"], []).append(a)
        _cache[key] = (by_sample, inst, cats)
    by_sample, inst, cats = _cache[key]

    R_ego = quat_to_rot(ref["ego_rot"])
    t_ego = np.asarray(ref["ego_trans"], dtype=np.float64)
    ego_yaw = yaw_from_quat(ref["ego_rot"])

    boxes = []
    for a in by_sample.get(sample_token, []):
        c = np.asarray(a["translation"], dtype=np.float64)
        c_ego = (c - t_ego) @ R_ego
        w, l, h = [float(v) for v in a["size"]]       # nuScenes order: w, l, h
        boxes.append({
            "centre": c_ego,
            "wlh": (w, l, h),
            "yaw": yaw_from_quat(a["rotation"]) - ego_yaw,
            "category": cats.get(inst[a["instance_token"]]["category_token"], "?"),
            "num_lidar_pts": int(a.get("num_lidar_pts", 0)),
        })
    return boxes
