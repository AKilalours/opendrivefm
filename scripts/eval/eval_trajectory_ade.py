#!/usr/bin/env python3
"""Ego-trajectory ADE/FDE for geometric baselines, against real waypoints.

WHY THIS EXISTS
---------------
The repo already contains a fine-tuned GPT-2 at outputs/artifacts/traj_lm_gpt2/.
It cannot be used, and the reason is not that PyTorch is missing here:

  scripts/traj_lm.py builds its dataset from outputs/artifacts/
  nuscenes_mini_manifest.jsonl, reading row["ego_future"] and falling back to
  row["ego_pose"]["translation"]. That manifest has neither key -- its rows
  carry only cams, extrinsics, intrinsics, sample_token and scene. Verified
  directly: all 404 rows. So the fallback produced (0, 0) for every waypoint
  and the model was fine-tuned on 404 copies of one all-zero trajectory. It is
  a real fine-tune of a real model on degenerate data, and its outputs are
  worthless regardless of hardware.

The genuine waypoints were sitting one directory away the whole time, in
outputs/artifacts/nuscenes_labels_128/*.npz under key "traj": 404 samples, 388
distinct, 12 horizons from 0.5 s to 6.0 s, mean final displacement 27.5 m.
This script scores geometric baselines against those, so there is a real floor
on record before any learned model is retrained on the correct source.

BASELINES
  STATIC            the ego does not move. The trivial floor.
  CONSTANT VELOCITY  waypoint(t) = v_prev * t, from the vxy_prev field.
  CONSTANT TURN      constant speed and yaw rate, yaw rate estimated from the
                     curvature of the first second of the true path.

CONSTANT TURN USES FUTURE INFORMATION and is labelled an oracle: it is an upper
bound on what curvature modelling can buy, not a deployable predictor.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "outputs/artifacts/nuscenes_labels_128"


def ade_fde(pred, true, upto):
    """Average and final displacement error over the first `upto` waypoints."""
    d = np.linalg.norm(pred[:, :upto] - true[:, :upto], axis=-1)
    return float(d.mean()), float(d[:, -1].mean())


def run():
    fs = sorted(glob.glob(str(LABELS / "*.npz")))
    if not fs:
        raise SystemExit(f"no label files under {LABELS}")
    traj = np.array([np.load(f)["traj"] for f in fs])          # (N, 12, 2)
    vprev = np.array([np.load(f)["vxy_prev"] for f in fs])     # (N, 2)
    t_rel = np.load(fs[0])["t_rel"]                            # (12,)

    N = len(traj)
    static = np.zeros_like(traj)
    cv = vprev[:, None, :] * t_rel[None, :, None]

    # constant-turn ORACLE: speed and yaw rate fitted to the true path's first
    # second, then integrated forward. Uses future information by construction.
    speed = np.linalg.norm(vprev, axis=-1)
    hd0 = np.arctan2(traj[:, 0, 1], traj[:, 0, 0])
    hd1 = np.arctan2(traj[:, 1, 1] - traj[:, 0, 1], traj[:, 1, 0] - traj[:, 0, 0])
    omega = (hd1 - hd0) / max(1e-6, float(t_rel[1] - t_rel[0]))
    ct = np.zeros_like(traj)
    for i in range(N):
        x = y = th = 0.0
        prev_t = 0.0
        for k, tt in enumerate(t_rel):
            dt = float(tt - prev_t)
            prev_t = float(tt)
            th += omega[i] * dt
            x += speed[i] * np.cos(th) * dt
            y += speed[i] * np.sin(th) * dt
            ct[i, k] = (x, y)

    horizons = {"T+1 (0.5s)": 1, "T+2 (1.0s)": 2, "T+3 (1.5s)": 3,
                "3.0s": 6, "6.0s": 12}
    out = {
        "samples": N,
        "distinct_trajectories": int(len(np.unique(traj.reshape(N, -1), axis=0))),
        "horizons_s": [round(float(v), 2) for v in t_rel],
        "mean_speed_mps": round(float(speed.mean()), 2),
        "source": "outputs/artifacts/nuscenes_labels_128/*.npz key 'traj'",
        "results": {},
    }
    for name, k in horizons.items():
        row = {}
        for mname, pred in (("static", static), ("constant_velocity", cv),
                            ("constant_turn_oracle", ct)):
            a, f = ade_fde(pred, traj, k)
            row[mname] = {"ade_m": round(a, 3), "fde_m": round(f, 3)}
        out["results"][name] = row

    out["gpt2_checkpoint_status"] = (
        "NOT EVALUATED, and not because of missing hardware. The checkpoint at "
        "outputs/artifacts/traj_lm_gpt2/ was fine-tuned via scripts/traj_lm.py, "
        "which reads waypoints from manifest keys ('ego_future', then "
        "'ego_pose') that do not exist in outputs/artifacts/"
        "nuscenes_mini_manifest.jsonl -- its rows carry only cams, extrinsics, "
        "intrinsics, sample_token, scene. Every waypoint fell back to (0, 0), so "
        "the model saw 404 copies of one all-zero trajectory. Retraining against "
        "the label npz files is the prerequisite for any learned number here."
    )
    out["caveats"] = [
        "constant_turn is an ORACLE: its yaw rate is fitted to the true future "
        "path, so it bounds what curvature modelling could buy rather than "
        "predicting anything.",
        "Ego trajectory only. This is not object trajectory prediction.",
        "nuScenes mini: 404 keyframes over 10 scenes, 2 cities.",
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    r = run()
    p = ROOT / "outputs/artifacts/trajectory_ade_report.json"
    p.write_text(json.dumps(r, indent=2))
    print(f"samples {r['samples']}  distinct {r['distinct_trajectories']}  "
          f"mean speed {r['mean_speed_mps']} m/s\n")
    print("%-12s %18s %18s %18s" % ("horizon", "static", "const-velocity",
                                    "const-turn (oracle)"))
    for h, row in r["results"].items():
        print("%-12s %10.2f m ADE %10.2f m ADE %10.2f m ADE" % (
            h, row["static"]["ade_m"], row["constant_velocity"]["ade_m"],
            row["constant_turn_oracle"]["ade_m"]))
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
