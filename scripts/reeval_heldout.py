"""Recompute every headline number on data the model was not fitted to.

The problem this fixes
----------------------
scripts/train/train_v11_temporal.py holds out whole scenes:

    VAL_SCENES = {"scene-0655", "scene-1077"}

so the model itself was trained correctly. But every metric that reached the
console was then computed over ALL 404 keyframes -- the 322 training keyframes
included. learned_model_report.json says so in its own caveats: "All 404
keyframes of nuScenes v1.0-mini". calibration_report.json likewise reports 404
keyframes and 6,619,136 cells. And the robustness sweep used keyframes 0..119,
which are scenes 0061, 0103 and 0553: three TRAINING scenes, all daylight.

A number measured on the training set is not wrong arithmetic, it is the wrong
question. It says how well the model reproduces what it was shown, not how well
it works. Published that way it is the kind of thing a reviewer finds in the
supplementary and stops reading.

This script recomputes ADE, occupancy IoU and calibration three ways --
all 404, the 322 train keyframes, and the 82 genuinely held-out ones -- so the
size of the inflation is measured rather than assumed. Predictions are cached
so that every later analysis (calibration stratified by observability, tau(o),
risk-coverage) is free and runs on CPU.

Confidence intervals
--------------------
Two bootstraps are reported and they disagree on purpose:

  keyframe-level  resamples keyframes. This is what everybody does and it is
                  too narrow here, because keyframes 0.5 s apart are nearly the
                  same picture, so they are not independent draws.

  scene-level     resamples whole scenes. With two validation scenes this is
                  brutally wide -- which is the honest width. If a result only
                  survives the keyframe-level interval, it is not a result, it
                  is a property of two clips.

Reporting the narrow one alone would repeat the mistake that produced a
+0.0108 occlusion delta presented as a finding.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from opendrivefm.models.model import OpenDriveFM      # noqa: E402
from opendrivefm.data import splits as S              # noqa: E402
from fix_trust import remap_trust_keys                # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(2)

ROOT = os.environ.get("ODFM_ROOT", ".")
ART = os.environ.get("ODFM_ARTIFACTS",
                     "/mnt/user-data/uploads/Projects/opendrivefm/outputs/artifacts")
LABELS = os.environ.get("ODFM_LABELS", os.path.join(ROOT, "artifacts", "scene_labels.json"))
OUT = os.environ.get("ODFM_OUT", ".")
CACHE = os.path.join(OUT, "preds_v11_404.npz")
NBOOT = int(os.environ.get("ODFM_NBOOT", "10000"))

D = np.load(os.path.join(ROOT, "all_frames.npz"), allow_pickle=True)
IMG, CH, ED, MO = D["img"], D["chain"], D["ego_deltas"], D["motion"]
TR, TREL, OCCGT = D["traj"], D["t_rel"], D["occ"]
TOK = [str(t) for t in D["tokens"]]
N = len(IMG)

scenes = np.array(S.scene_of(TOK, LABELS))
val_scenes = set(S.LEGACY_V11["val"])
VAL = np.where(np.isin(scenes, list(val_scenes)))[0]
TRN = np.where(~np.isin(scenes, list(val_scenes)))[0]
ALL = np.arange(N)
print(f"{N} keyframes | train {TRN.size} | HELD OUT {VAL.size} "
      f"({', '.join(sorted(val_scenes))})")


def window(i):
    w = np.transpose(IMG[CH[i]], (1, 0, 2, 3, 4))
    return (torch.from_numpy(np.ascontiguousarray(w))
            .float().div_(255.).permute(0, 1, 4, 2, 3).unsqueeze(0))


# ---------------------------------------------------------------- predictions
if os.path.exists(CACHE):
    c = np.load(CACHE)
    RES, OCCP = c["res"], c["occp"]
    print(f"loaded cached predictions from {CACHE}")
else:
    ck = torch.load(f"{ART}/checkpoints_v11_temporal/best_val_ade.ckpt",
                    map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters") or {}
    m = OpenDriveFM(d=hp.get("d", 384), bev_h=128, bev_w=128,
                    horizon=hp.get("horizon", 12), enable_trust=True)
    sd = {k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    sd, _, _ = remap_trust_keys(sd, m.state_dict())
    miss, unexp = m.load_state_dict(sd, strict=False)
    m.eval()
    print(f"v11_temporal: {len(sd) - len(miss)}/{len(sd)} loaded, {len(unexp)} unexpected")

    RES = np.zeros((N, 12, 2), np.float32)
    OCCP = np.zeros((N, 128, 128), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(N):
            occ, res, _, _ = m(window(i),
                               velocity=torch.from_numpy(MO[i][1:3]).unsqueeze(0),
                               ego_deltas=torch.from_numpy(ED[i]).unsqueeze(0))
            RES[i] = res[0].numpy()
            OCCP[i] = torch.sigmoid(occ[0, 0]).numpy().astype(np.float16)
            if i % 100 == 0:
                print(f"  {i}/{N}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(CACHE, res=RES, occp=OCCP)
    print(f"cached predictions -> {CACHE}  ({time.time() - t0:.0f}s)")

dtp = MO[:, 0:1]
vxy = MO[:, 1:3] * (dtp > 0)
CV = TREL[:, :, None] * vxy[:, None, :]
LEARNED = CV + RES
GT = (OCCGT[:, 0] > 0)


# ---------------------------------------------------------------- statistics
def boot_ci(per_item, idx, n=NBOOT, groups=None, rng_seed=0):
    """Percentile CI for a mean.

    groups=None resamples items (keyframes). groups=array resamples GROUPS
    (scenes) with all their members, which is the honest unit here because
    keyframes inside a scene are near-duplicates.
    """
    rng = np.random.default_rng(rng_seed)
    v = np.asarray(per_item, float)[idx]
    if v.size == 0:
        return float("nan"), float("nan")
    if groups is None:
        draws = rng.choice(v, size=(n, v.size), replace=True).mean(1)
    else:
        g = np.asarray(groups)[idx]
        uniq = np.unique(g)
        buckets = [v[g == u] for u in uniq]
        draws = np.empty(n)
        for k in range(n):
            pick = rng.integers(0, len(buckets), len(buckets))
            draws[k] = np.concatenate([buckets[j] for j in pick]).mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def ade_per_frame(pred, k):
    return np.linalg.norm(pred[:, :k] - TR[:, :k], axis=-1).mean(1)


def occ_stats(idx, thr):
    pr = OCCP[idx].astype(np.float32) > thr
    g = GT[idx]
    tp = float((pr & g).sum()); fp = float((pr & ~g).sum()); fn = float((~pr & g).sum())
    iou = tp / max(1.0, tp + fp + fn)
    p = tp / max(1.0, tp + fp)
    r = tp / max(1.0, tp + fn)
    return {"threshold": thr, "iou": round(iou, 4), "precision": round(p, 4),
            "recall": round(r, 4), "f1": round(2 * p * r / max(1e-9, p + r), 4)}


def calibration(idx, bins=15):
    """ECE / MCE over occupancy cells, plus the base rate."""
    p = OCCP[idx].astype(np.float32).ravel()
    y = GT[idx].ravel().astype(np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    mce = 0.0
    total = p.size
    hist = []
    for b in range(bins):
        sel = which == b
        n_b = int(sel.sum())
        if n_b == 0:
            continue
        conf = float(p[sel].mean())
        acc = float(y[sel].mean())
        gap = abs(acc - conf)
        ece += (n_b / total) * gap
        mce = max(mce, gap)
        hist.append({"bin": round(float(edges[b]), 3), "n": n_b,
                     "confidence": round(conf, 4), "actual": round(acc, 4)})
    return {"ece": round(ece, 4), "mce": round(mce, 4),
            "base_rate": round(float(y.mean()), 4),
            "cells": int(p.size), "histogram": hist}


# ---------------------------------------------------------------- report
HOR = [("T+1 (0.5s)", 1), ("T+2 (1.0s)", 2), ("T+3 (1.5s)", 3),
       ("3.0s", 6), ("6.0s", 12)]
SETS = [("all_404_as_published", ALL), ("train_322", TRN), ("held_out_82", VAL)]

traj = {}
for label, k in HOR:
    cv_pf = ade_per_frame(CV, k)
    lr_pf = ade_per_frame(LEARNED, k)
    row = {}
    for name, idx in SETS:
        klo, khi = boot_ci(lr_pf, idx)
        slo, shi = boot_ci(lr_pf, idx, groups=scenes)
        row[name] = {
            "constant_velocity_m": round(float(cv_pf[idx].mean()), 3),
            "learned_m": round(float(lr_pf[idx].mean()), 3),
            "delta_vs_cv_pct": round(100 * (lr_pf[idx].mean() - cv_pf[idx].mean())
                                     / max(1e-9, cv_pf[idx].mean()), 2),
            "ci95_keyframe": [round(klo, 3), round(khi, 3)],
            "ci95_scene": [round(slo, 3), round(shi, 3)],
        }
    row["inflation_pct"] = round(
        100 * (row["held_out_82"]["learned_m"] - row["all_404_as_published"]["learned_m"])
        / max(1e-9, row["all_404_as_published"]["learned_m"]), 1)
    traj[label] = row

occ = {}
for name, idx in SETS:
    best = max((occ_stats(idx, t) for t in (0.3, 0.4, 0.5, 0.6, 0.7)),
               key=lambda d: d["iou"])
    occ[name] = best

cal = {name: calibration(idx) for name, idx in SETS}

report = {
    "status": "MEASURED",
    "what": ("Every headline number recomputed on the 82 keyframes the "
             "v11_temporal checkpoint was never fitted to, beside the all-404 "
             "figures that were published, so the train-set inflation is "
             "measured rather than assumed."),
    "checkpoint": "checkpoints_v11_temporal/best_val_ade.ckpt",
    "split": {"name": "legacy_v11",
              "source": "scripts/train/train_v11_temporal.py:31",
              "val_scenes": sorted(val_scenes),
              "train_keyframes": int(TRN.size), "val_keyframes": int(VAL.size)},
    "trajectory_ade": traj,
    "occupancy": occ,
    "calibration": cal,
    "bootstrap_resamples": NBOOT,
    "caveats": [
        "The held-out set is two scenes. The scene-level interval is the honest "
        "one and it is very wide; the keyframe-level interval is reported only "
        "to show how misleading it is at this sample size.",
        "Occupancy labels are the geometric LiDAR reconstruction, so calibration "
        "measures agreement with that reconstruction, not with absolute truth.",
        "scene-0655 is day and scene-1077 is night, so the held-out set is "
        "lighting-balanced but covers one city.",
    ],
}
path = os.path.join(OUT, "heldout_reeval_report.json")
# numpy scalars survive the arithmetic above; make them plain floats on the
# way out rather than hunting each one down.
json.dump(report, open(path, "w"), indent=2, default=float)

print("\n=== TRAJECTORY ADE ===")
print(f"{'horizon':<12}{'published(404)':>16}{'train(322)':>13}{'HELD OUT(82)':>15}{'inflation':>11}")
for label, _ in HOR:
    r = traj[label]
    print(f"{label:<12}{r['all_404_as_published']['learned_m']:>16.3f}"
          f"{r['train_322']['learned_m']:>13.3f}"
          f"{r['held_out_82']['learned_m']:>15.3f}{r['inflation_pct']:>10.1f}%")
print("\n=== OCCUPANCY IoU ===")
for name, _ in SETS:
    print(f"  {name:<24} IoU {occ[name]['iou']:.4f}  P {occ[name]['precision']:.4f}  "
          f"R {occ[name]['recall']:.4f}  @thr {occ[name]['threshold']}")
print("\n=== CALIBRATION ===")
for name, _ in SETS:
    c = cal[name]
    print(f"  {name:<24} ECE {c['ece']:.4f}  MCE {c['mce']:.4f}  "
          f"base {c['base_rate']:.4f}  cells {c['cells']:,}")
print("\nwrote", path)
