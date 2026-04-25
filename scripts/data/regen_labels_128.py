"""
regen_labels_128.py — Regenerate BEV labels at 128×128 resolution.

This takes ~5 minutes. Run BEFORE training v10.

Usage:
    python scripts/regen_labels_128.py \
        --data_root data/nuscenes \
        --manifest artifacts/nuscenes_mini_manifest.jsonl \
        --out_dir artifacts/nuscenes_labels_128

What changes at 128×128 vs 64×64:
  - BEV resolution: 0.78125 m/pixel (was 1.5625 m/pixel)
  - Same 100m×100m extent
  - Dilation radius: 1 pixel (was 2) — scaled proportionally
  - Expected occupancy density: similar ~4-5% (objects same size, finer grid)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


def _dilate_binary(mask01: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask01
    k = 2 * r + 1
    t = torch.from_numpy(mask01[None, None].astype(np.float32))
    t = F.max_pool2d(t, kernel_size=k, stride=1, padding=r)
    return (t[0, 0].numpy() > 0.0).astype(np.uint8)


def _lidar_sd(nusc, sample):
    return nusc.get("sample_data", sample["data"]["LIDAR_TOP"])


def _ego_pose_from_sd(nusc, sd):
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    R = Quaternion(pose["rotation"]).rotation_matrix.astype(np.float32)
    t = np.array(pose["translation"], dtype=np.float32)
    return R, t


def _timestamp_sec(sd):
    return float(sd["timestamp"]) / 1e6


def build_occ_128(nusc, sample, bev=128, extent_m=50.0, z_min=-1.2, z_max=3.0, dilate_r=1):
    """Build occupancy at 128×128 from LiDAR."""
    sd = _lidar_sd(nusc, sample)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    pc = LidarPointCloud.from_file(str(Path(nusc.dataroot) / sd["filename"]))
    R  = Quaternion(cs["rotation"]).rotation_matrix
    t  = np.array(cs["translation"])
    pc.rotate(R); pc.translate(t)

    pts  = pc.points[:3, :]
    x, y, z = pts[0], pts[1], pts[2]

    keep = (z >= z_min) & (z <= z_max)
    x, y = x[keep], y[keep]
    keep2 = (x >= -extent_m) & (x <= extent_m) & (y >= -extent_m) & (y <= extent_m)
    x, y  = x[keep2], y[keep2]

    occ = np.zeros((bev, bev), dtype=np.uint8)
    if x.size == 0:
        return occ[None].astype(np.float32)

    s = (2.0 * extent_m) / bev
    r = np.floor((extent_m - x) / s).astype(np.int32)
    c = np.floor((y + extent_m) / s).astype(np.int32)
    r = np.clip(r, 0, bev-1)
    c = np.clip(c, 0, bev-1)
    occ[r, c] = 1
    occ = _dilate_binary(occ, dilate_r)
    return occ[None].astype(np.float32)


def build_traj_and_motion(nusc, sample0, horizon=12):
    """Unchanged from v8 — trajectory is ego-frame, independent of BEV resolution."""
    sd0   = _lidar_sd(nusc, sample0)
    R0, x0 = _ego_pose_from_sd(nusc, sd0)
    t0    = _timestamp_sec(sd0)

    traj, t_rel = [], []
    s = sample0
    last_xy, last_tr = np.zeros(2, np.float32), 0.0

    for _ in range(horizon):
        nxt = s.get("next", "")
        if not nxt:
            traj.append(last_xy.copy()); t_rel.append(last_tr); continue
        s    = nusc.get("sample", nxt)
        sdk  = _lidar_sd(nusc, s)
        _, xk = _ego_pose_from_sd(nusc, sdk)
        tk   = _timestamp_sec(sdk)
        p_rel = (R0.T @ (xk - x0)).astype(np.float32)
        last_xy = p_rel[:2]; last_tr = float(tk - t0)
        traj.append(last_xy.copy()); t_rel.append(last_tr)

    prev_tok = sample0.get("prev", "")
    if prev_tok:
        s_prev = nusc.get("sample", prev_tok)
        sdp    = _lidar_sd(nusc, s_prev)
        tp     = _timestamp_sec(sdp)
        dt     = float(_timestamp_sec(sd0) - tp)
        _, xp  = _ego_pose_from_sd(nusc, sdp)
        disp   = (R0.T @ (x0 - xp)).astype(np.float32)
        vxy    = (disp[:2] / max(dt, 1e-6)).astype(np.float32)
    else:
        dt, vxy = 0.0, np.zeros(2, np.float32)

    return (np.stack(traj).astype(np.float32),
            np.array(t_rel, np.float32),
            float(dt),
            vxy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--version",   default="v1.0-mini")
    ap.add_argument("--manifest",  required=True)
    ap.add_argument("--out_dir",   default="artifacts/nuscenes_labels_128")
    ap.add_argument("--bev",       type=int, default=128)
    ap.add_argument("--extent_m",  type=float, default=50.0)
    ap.add_argument("--dilate",    type=int, default=1)
    args = ap.parse_args()

    nusc    = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows    = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]

    print(f"Regenerating {len(rows)} labels at {args.bev}×{args.bev} → {out_dir}")
    for i, r in enumerate(rows):
        sample0 = nusc.get("sample", r["sample_token"])
        occ     = build_occ_128(nusc, sample0, bev=args.bev,
                                extent_m=args.extent_m, dilate_r=args.dilate)
        traj, t_rel, dt_prev, vxy_prev = build_traj_and_motion(nusc, sample0)

        np.savez_compressed(
            out_dir / f"{r['sample_token']}.npz",
            occ=occ, traj=traj, t_rel=t_rel,
            dt_prev=np.float32(dt_prev), vxy_prev=vxy_prev,
        )
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}")

    # Sanity check
    sample_npz = np.load(out_dir / f"{rows[0]['sample_token']}.npz")
    print(f"\nDone. Sample occ shape: {sample_npz['occ'].shape}")
    mean_occ = np.mean([np.load(out_dir/f"{r['sample_token']}.npz")['occ'].mean()
                        for r in rows[:20]])
    print(f"Mean occupancy (first 20): {mean_occ:.3f}")
    print(f"\nResolution: {100*2/args.bev:.4f} m/pixel "
          f"(was 1.5625 m/pixel at 64×64)")


if __name__ == "__main__":
    main()
