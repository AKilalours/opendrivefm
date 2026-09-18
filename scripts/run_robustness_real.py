"""Does the trust head react to REAL degradation the way it reacts to ours?

Every robustness number this project has published was measured on keyframes
0..119 of the export, which are scenes 0061, 0103 and 0553 -- all three of them
clear daylight. The degradations were synthetic: rain, fog and snow rendered
onto clean daytime frames by opendrivefm.robustness.weather. So the claim
"the trust head detects degraded cameras" was only ever tested against
degradation this repository wrote itself, which is close to circular.

nuScenes mini contains three real night scenes (1077, 1094, 1100), 121
keyframes, 30% of the data, and scene-1094 was recorded after rain. They had
never been evaluated. This script uses them as the real condition.

Two things are measured.

1. REAL. Mean per-camera trust on clear daytime scenes, on night scenes, and on
   the after-rain night scene, with no perturbation applied to anything. If the
   trust head is measuring image quality rather than memorising scenes, night
   should score lower than day.

2. SYNTHETIC, made comparable. The old sweep corrupted ONE camera and compared
   it against the other five. Real night degrades all six at once, so a
   one-camera number cannot be compared against it. Here every synthetic
   condition is applied to ALL SIX cameras of the daytime frames, which is the
   like-for-like version of what night does.

The comparison that matters is the effect size: if synthetic rain moves trust
about as far as real night does, the synthetic pipeline is a fair proxy and the
earlier results stand. If it moves it much further, the synthetic corruptions
are easier to catch than reality and the earlier numbers were flattering.
Either answer is worth reporting; the current state, which is no answer, is not.

Every figure carries a 95% bootstrap confidence interval over keyframes,
because a mean over ~120 keyframes without one cannot distinguish an effect
from sampling noise -- which is how a +0.0108 occlusion delta came to be
reported as a finding.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from opendrivefm.models.model import OpenDriveFM          # noqa: E402
from opendrivefm.robustness.perturbations import PERTURBATIONS  # noqa: E402
from opendrivefm.data import splits as S                  # noqa: E402
from weather import WEATHER                               # noqa: E402
from fix_trust import remap_trust_keys                    # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(2)

ROOT = os.environ.get("ODFM_ROOT", ".")
ART = os.environ.get("ODFM_ARTIFACTS",
                     "/mnt/user-data/uploads/Projects/opendrivefm/outputs/artifacts")
LABELS = os.environ.get("ODFM_LABELS", os.path.join(ROOT, "artifacts", "scene_labels.json"))
OUT = os.environ.get("ODFM_OUT", ".")
NBOOT = int(os.environ.get("ODFM_NBOOT", "10000"))

# ---------------------------------------------------------------- data
D = np.load(os.path.join(ROOT, "all_frames.npz"), allow_pickle=True)
IMG, CH, ED, MO = D["img"], D["chain"], D["ego_deltas"], D["motion"]
TOK = [str(t) for t in D["tokens"]]

scene_of = S.scene_of(TOK, LABELS)
scenes = np.array(scene_of)
day_idx = np.where(np.isin(scenes, S.DAY_SCENES))[0]
night_idx = np.where(np.isin(scenes, S.NIGHT_SCENES))[0]
wet_idx = np.where(np.isin(scenes, S.WET_SCENES))[0]
dry_night_idx = np.array([i for i in night_idx if i not in set(wet_idx)])

print(f"day {len(day_idx)} | night {len(night_idx)} "
      f"(after-rain {len(wet_idx)}, dry night {len(dry_night_idx)}) keyframes")


def window(i):
    w = np.transpose(IMG[CH[i]], (1, 0, 2, 3, 4))
    return (torch.from_numpy(np.ascontiguousarray(w))
            .float().div_(255.).permute(0, 1, 4, 2, 3).unsqueeze(0))


# ---------------------------------------------------------------- model
ck = torch.load(f"{ART}/checkpoints_v11_trustfix2/trust_fixed_v2_cal.ckpt",
                map_location="cpu", weights_only=False)
m = OpenDriveFM(d=384, bev_h=128, bev_w=128, horizon=12, enable_trust=True)
sd = {k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
sd, _, _ = remap_trust_keys(sd, m.state_dict())
m.load_state_dict(sd, strict=False)
m.eval()
print("trust_fixed_v2_cal loaded, calibrated =",
      bool(m.backbone.trust_scorer.stat_calibrated))


def make(name):
    if name is None:
        return None
    return WEATHER[name]() if name in WEATHER else PERTURBATIONS[name]()


@torch.no_grad()
def trust_over(indices, cond=None, all_cameras=True):
    """Mean trust per keyframe (averaged over the six cameras).

    cond=None leaves the frames exactly as recorded, which is what makes the
    night measurement a measurement of reality rather than of our renderer.
    """
    P = make(cond)
    out = []
    for i in indices:
        x = window(int(i))
        if P is not None:
            B, NC, T, C, H, W = x.shape
            cams = range(NC) if all_cameras else (0,)
            x = x.clone()
            for c in cams:
                v = x[:, c]
                x[:, c] = P(v.reshape(B * T, C, H, W)).reshape(B, T, C, H, W).clamp(0, 1)
        _, _, tr, _ = m(x,
                        velocity=torch.from_numpy(MO[int(i)][1:3]).unsqueeze(0),
                        ego_deltas=torch.from_numpy(ED[int(i)]).unsqueeze(0))
        out.append(float(tr[0].numpy().mean()))
    return np.asarray(out)


# ---------------------------------------------------------------- statistics
def boot_ci(a, n=NBOOT, alpha=0.05, rng=None):
    """95% percentile bootstrap CI for a mean."""
    rng = rng or np.random.default_rng(0)
    a = np.asarray(a, float)
    if a.size == 0:
        return (float("nan"),) * 2
    draws = rng.choice(a, size=(n, a.size), replace=True).mean(1)
    return float(np.percentile(draws, 100 * alpha / 2)), float(np.percentile(draws, 100 * (1 - alpha / 2)))


def boot_diff_ci(a, b, n=NBOOT, alpha=0.05, rng=None):
    """CI for mean(a) - mean(b) with independent resampling of each group.

    If this interval contains zero the difference is not distinguishable from
    sampling noise, whatever the point estimate looks like.
    """
    rng = rng or np.random.default_rng(1)
    a, b = np.asarray(a, float), np.asarray(b, float)
    da = rng.choice(a, size=(n, a.size), replace=True).mean(1)
    db = rng.choice(b, size=(n, b.size), replace=True).mean(1)
    d = da - db
    return float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2)))


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(1, na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def block(name, vals, ref=None):
    lo, hi = boot_ci(vals)
    row = {"condition": name, "keyframes": int(vals.size),
           "trust_mean": round(float(vals.mean()), 4),
           "ci95": [round(lo, 4), round(hi, 4)],
           "std": round(float(vals.std(ddof=1)), 4)}
    if ref is not None:
        dlo, dhi = boot_diff_ci(vals, ref)
        row["delta_vs_clean_day"] = round(float(vals.mean() - ref.mean()), 4)
        row["delta_ci95"] = [round(dlo, 4), round(dhi, 4)]
        row["significant"] = bool(dlo > 0 or dhi < 0)
        row["cohen_d"] = round(cohen_d(vals, ref), 3)
    return row


# ---------------------------------------------------------------- run
t0 = time.time()
print("\n[1/2] real conditions, nothing rendered")
clean_day = trust_over(day_idx)
clean_night = trust_over(dry_night_idx)
clean_wet = trust_over(wet_idx)

real = [block("clean day (7 scenes)", clean_day),
        block("REAL night, dry (1077, 1100)", clean_night, clean_day),
        block("REAL night after rain (1094)", clean_wet, clean_day),
        block("REAL night, all three", np.concatenate([clean_night, clean_wet]), clean_day)]
for r in real:
    print("   ", json.dumps(r))

print("\n[2/2] synthetic conditions on the SAME daytime frames, all six cameras")
synth = []
for cond in ["rain", "fog", "snow", "noise", "blur", "glare", "occlusion"]:
    v = trust_over(day_idx, cond=cond, all_cameras=True)
    row = block(f"synthetic {cond} (all cameras)", v, clean_day)
    synth.append(row)
    print("   ", json.dumps(row))

# ---------------------------------------------------------------- verdict
real_night_delta = float(np.concatenate([clean_night, clean_wet]).mean() - clean_day.mean())
closest = min(synth, key=lambda r: abs(r["delta_vs_clean_day"] - real_night_delta))
harsher = [r["condition"] for r in synth
           if abs(r["delta_vs_clean_day"]) > abs(real_night_delta) * 1.5]

report = {
    "status": "MEASURED",
    "what": ("Trust-head response to REAL night and after-rain keyframes, against "
             "synthetic corruptions applied to the same daytime frames on all six "
             "cameras. Replaces a sweep that used only daytime scenes 0061/0103/0553."),
    "scenes": {"day": list(S.DAY_SCENES), "night": list(S.NIGHT_SCENES),
               "after_rain": list(S.WET_SCENES)},
    "keyframes": {"day": int(day_idx.size), "night": int(night_idx.size)},
    "bootstrap_resamples": NBOOT,
    "real_conditions": real,
    "synthetic_conditions_all_cameras": synth,
    "real_night_delta": round(real_night_delta, 4),
    "closest_synthetic_proxy": closest["condition"],
    "synthetic_harsher_than_reality": harsher,
    "reading": (
        "A synthetic condition is a fair proxy for real degradation only if it moves "
        "trust by a comparable amount. Conditions listed under "
        "synthetic_harsher_than_reality move it more than 1.5x further than real night "
        "does, so trust-detection rates measured on them are optimistic."),
    "caveats": [
        "Three night scenes, all from singapore-hollandvillage. Night here is "
        "confounded with location; a larger split is needed to separate them.",
        "Trust is averaged over six cameras per keyframe, so a single badly "
        "degraded camera is diluted. This is the comparison real night forces, "
        "since night degrades all six at once.",
        "Bootstrap CIs are over keyframes, which are correlated within a scene. "
        "They therefore understate the true uncertainty; a scene-level bootstrap "
        "would be wider.",
    ],
    "seconds": round(time.time() - t0, 1),
}

path = os.path.join(OUT, "robustness_real_report.json")
json.dump(report, open(path, "w"), indent=2)
print(f"\nreal night delta {real_night_delta:+.4f} | closest synthetic proxy: "
      f"{closest['condition']}")
if harsher:
    print("synthetic conditions harsher than reality:", ", ".join(harsher))
print("wrote", path)
