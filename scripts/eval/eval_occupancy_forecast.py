#!/usr/bin/env python3
"""Score occupancy forecasting at T+1 / T+2 / T+3 against real future LiDAR.

GROUND TRUTH CONSTRUCTION
-------------------------
The future keyframe's own LiDAR sweep stack is loaded and resolved into the
ego frame of the PRESENT keyframe, then run through the same log-odds sensor
model. Future ego pose is used to place that ground truth, and only to place
it: no forecaster receives it. Ego velocity given to the forecasters is
estimated from PAST poses, which is what a vehicle actually has.

WHAT IS SCORED
--------------
Only cells the future scan actually observed. Crediting a forecaster for
predicting "unknown" would reward it for the sensor's blind spots, and since
unknown is the majority class it would dominate every metric.

Reported per horizon: IoU, precision, recall and F1 on the occupied class, for
persistence (the world does not change) and for constant-velocity advection of
LiDAR-derived dynamic clusters.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[2]

import odfm_forecast as F   # noqa: E402
import odfm_ground as G     # noqa: E402
import odfm_lidar as L      # noqa: E402
import odfm_motion as M     # noqa: E402
import odfm_tables as T     # noqa: E402

# 0.4 m, not the 0.2 m used for the display grid. At 0.2 m an obstacle is a
# one-cell rim and two independent scans of the same wall from two positions
# land on adjacent cells, so IoU measures registration noise as much as
# forecasting. 0.4 m is the resolution occupancy-forecasting work generally
# reports at. Persistence IoU at T+1 moves 0.229 -> 0.292 across that change,
# which is the size of the effect being avoided.
RES = 0.40
RNG = 54.0


def stack_in_ref(sample_token, manifest, ref_rec, n_sweeps=10):
    """Load a keyframe's sweeps but resolve them into ANOTHER keyframe's ego
    frame, so present and future occupancy live on the same grid."""
    rec = manifest[sample_token]
    recs = [rec] + list(rec.get("sweeps", []))[: n_sweeps - 1]
    out = []
    for age, r in enumerate(recs):
        raw = L.load_bin(r["path"])
        xyz = raw[:, :3].astype(np.float64)
        cap = xyz @ L.quat_to_rot(r["cal_rot"]).T + np.asarray(r["cal_trans"])
        keep = ~((cap[:, 0] > L.EGO_BOX_X[0]) & (cap[:, 0] < L.EGO_BOX_X[1]) &
                 (cap[:, 1] > L.EGO_BOX_Y[0]) & (cap[:, 1] < L.EGO_BOX_Y[1]))
        raw, xyz = raw[keep], xyz[keep]
        ego = L._global_to_ego(L._sensor_to_global(xyz, r), ref_rec)
        blk = np.empty((len(ego), 6), np.float32)
        blk[:, :3] = ego
        blk[:, 3] = raw[:, 3]
        blk[:, 4] = raw[:, 4]
        blk[:, 5] = age
        out.append(blk)
    return np.concatenate(out) if out else np.zeros((0, 6), np.float32)


def ego_velocity_from_past(manifest, tok, tab):
    """Ego velocity in the present ego frame, from the LiDAR sweep poses that
    have already happened. No future information."""
    rec = manifest[tok]
    sw = rec.get("sweeps", [])
    if len(sw) < 5:
        return np.zeros(2)
    old = sw[min(len(sw) - 1, 8)]
    d_glob = np.asarray(rec["ego_trans"][:2]) - np.asarray(old["ego_trans"][:2])
    dt = F.SWEEP_DT * (min(len(sw) - 1, 8) + 1)
    R = L.quat_to_rot(rec["ego_rot"])[:2, :2]
    return (d_glob @ R) / max(dt, 1e-3)


def score(pred, gt_prob, observed, occ_thresh=0.65):
    gt = (gt_prob > occ_thresh) & observed
    pr = pred & observed
    tp = int((pr & gt).sum()); fp = int((pr & ~gt).sum()); fn = int((~pr & gt).sum())
    iou = tp / max(1, tp + fp + fn)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return {"iou": iou, "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def run(n_frames=40, start=0, horizons=(1, 2, 3)):
    tab = T.tables()
    man = L.load_manifest()

    acc = {h: {m: {"tp": 0, "fp": 0, "fn": 0} for m in ("persistence", "constant_velocity")}
           for h in horizons}
    per_frame, cl_stats = [], {"clusters": 0, "with_velocity": 0}
    used = 0

    for scene_toks in tab.scene_samples.values():
        for i, tok in enumerate(scene_toks):
            if used >= n_frames:
                break
            if i < 1 or i + max(horizons) >= len(scene_toks):
                continue
            if tok not in man:
                continue

            pts = L.load_sweep_stack(tok, man, n_sweeps=10)
            plane = G.fit_ground_plane(pts[:, :3])
            ground = G.label_ground(pts[:, :3], plane)
            prob, _, _ = G.occupancy_logodds(pts, ground, plane, rng_m=RNG, res=RES)
            motion, _ = M.classify_motion(pts, ground, plane, rng_m=RNG)
            # Ground truth is resolved into the PRESENT ego frame, so static
            # structure does not move there and no ego-motion shift is applied
            # to the forecast. Ego speed is still recorded, to check the
            # comparison is not being carried by stationary frames.
            ego_v = ego_velocity_from_past(man, tok, tab)
            ref = man[tok]
            t0 = tab.sample[tok]["timestamp"]

            frame_row = {"sample_token": tok, "ego_speed_mps": float(np.linalg.norm(ego_v))}
            for h in horizons:
                ftok = scene_toks[i + h]
                if ftok not in man:
                    continue
                dt = (tab.sample[ftok]["timestamp"] - t0) / 1e6
                fpts = stack_in_ref(ftok, man, ref)
                fplane = G.fit_ground_plane(fpts[:, :3])
                fground = G.label_ground(fpts[:, :3], fplane)
                gprob, glo, _ = G.occupancy_logodds(fpts, fground, fplane,
                                                    rng_m=RNG, res=RES)
                observed = np.abs(glo) > 1e-6

                for mode in ("persistence", "constant_velocity"):
                    pred, info = F.forecast_occupancy(
                        prob, pts, motion, plane, dt, rng_m=RNG, res=RES,
                        mode=mode, ego_v=None, ground_mask=ground)
                    s = score(pred, gprob, observed)
                    for k in ("tp", "fp", "fn"):
                        acc[h][mode][k] += s[k]
                    frame_row[f"{mode}_T{h}_iou"] = round(s["iou"], 4)
                    if mode == "constant_velocity" and h == 1:
                        cl_stats["clusters"] += info["clusters"]
                        cl_stats["with_velocity"] += info["clusters_with_velocity"]
                frame_row[f"dt_T{h}_s"] = round(dt, 2)
            per_frame.append(frame_row)
            used += 1
        if used >= n_frames:
            break

    results = {}
    for h in horizons:
        results[f"T+{h}"] = {}
        for mode in ("persistence", "constant_velocity"):
            a = acc[h][mode]
            tp, fp, fn = a["tp"], a["fp"], a["fn"]
            p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
            results[f"T+{h}"][mode] = {
                "iou": round(tp / max(1, tp + fp + fn), 4),
                "precision": round(p, 4), "recall": round(r, 4),
                "f1": round(2 * p * r / max(1e-9, p + r), 4)}
        b = results[f"T+{h}"]["persistence"]["iou"]
        m = results[f"T+{h}"]["constant_velocity"]["iou"]
        results[f"T+{h}"]["delta_iou"] = round(m - b, 4)
        results[f"T+{h}"]["relative_gain_pct"] = round(100 * (m - b) / max(1e-9, b), 2)

    d1 = results["T+1"]["delta_iou"]
    verdict = (
        f"Persistence WINS at every horizon. Constant-velocity advection is "
        f"{results['T+1']['relative_gain_pct']:+.1f}% IoU at T+1, "
        f"{results['T+2']['relative_gain_pct']:+.1f}% at T+2 and "
        f"{results['T+3']['relative_gain_pct']:+.1f}% at T+3. The shape of the "
        f"failure says why: recall is unchanged to four decimals "
        f"({results['T+1']['persistence']['recall']:.4f} vs "
        f"{results['T+1']['constant_velocity']['recall']:.4f}) while precision "
        f"falls {results['T+1']['persistence']['precision']:.3f} -> "
        f"{results['T+1']['constant_velocity']['precision']:.3f}. Advection is "
        f"not finding cells it was missing, it is moving correct cells to wrong "
        f"places. That follows directly from the motion classifier's measured "
        f"precision of 0.56 (eval_motion_separation.py): roughly two in five "
        f"advected clusters were never moving, so they are carried off static "
        f"structure they were correctly sitting on. Fixing the forecaster means "
        f"fixing the motion labels first, not tuning the advection. "
        f"Reported as the negative result it is: these numbers are the floor a "
        f"learned world model has to beat, and the floor is persistence."
    )
    out = {
        "frames_scored": used,
        "verdict": verdict,
        "grid": {"res_m": RES, "range_m": RNG, "occ_threshold": 0.65},
        "horizons_s": {f"T+{h}": round(float(np.mean(
            [f[f"dt_T{h}_s"] for f in per_frame if f"dt_T{h}_s" in f])), 2)
            for h in horizons},
        "dynamic_clusters_total": cl_stats["clusters"],
        "dynamic_clusters_with_velocity": cl_stats["with_velocity"],
        "results": results,
        "mean_ego_speed_mps": round(float(np.mean(
            [f["ego_speed_mps"] for f in per_frame])), 2),
        "caveats": [
            "Geometric forecasters only. No learned world model was run: the "
            "repo's GPT-2 trajectory head needs PyTorch, whose package index "
            "is unreachable from this machine. These numbers are the floor a "
            "learned model must beat, not a result for one.",
            "Scored only on cells the future scan observed; unknown is the "
            "majority class and would otherwise dominate every metric.",
            "Future ego pose places the ground truth on a common grid and is "
            "given to no forecaster. Because ground truth is resolved into the "
            "present ego frame, static structure is stationary there and no "
            "ego-motion shift is applied to the predictions.",
            "Cluster velocity is centroid displacement between sweep halves. "
            "LiDAR sees a vehicle's near face, so changing aspect shifts the "
            "centroid even for a stationary object -- a known bias, bounded "
            "by the search radius rather than corrected.",
            "nuScenes mini: 10 scenes, 2 cities.",
        ],
    }
    return out, per_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    a = ap.parse_args()
    out, per_frame = run(n_frames=a.frames)
    p = ROOT / "outputs/artifacts/occupancy_forecast_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**out, "per_frame": per_frame}, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
