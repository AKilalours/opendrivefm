#!/usr/bin/env python3
"""Score free-space motion separation against annotation-derived velocity.

There is no per-point motion ground truth in nuScenes, so this scores at box
level: a box is truly moving when its instance's differenced global position
exceeds `--min-speed`, and predicted moving when most of its enclosed
current-sweep returns that the classifier COMMITTED on are labelled DYNAMIC.

Boxes where the classifier abstains on nearly everything are excluded from the
score and counted separately, because scoring an abstention as a wrong answer
and scoring it as a right one are both misleading. Coverage is reported next to
accuracy so the two can be read together.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[2]

import odfm_ground as G      # noqa: E402
import odfm_lidar as L       # noqa: E402
import odfm_motion as M      # noqa: E402
import odfm_tables as T      # noqa: E402


def run(n_frames=40, start=0, min_speed=0.5, thresholds=(0.2, 0.35, 0.5),
        min_committed=5):
    tab = T.tables()
    man = L.load_manifest()
    toks = tab.frames_with_history(9)[start:start + n_frames]

    per_frame = []
    for tok in toks:
        pts = L.load_sweep_stack(tok, man, n_sweeps=10)
        plane = G.fit_ground_plane(pts[:, :3])
        ground = G.label_ground(pts[:, :3], plane)
        lab, _ = M.classify_motion(pts, ground, plane)
        cur = pts[:, 5] == 0
        speeds, _ = M.annotation_velocities(tab, tok, min_speed)
        rows = []
        for b in tab.boxes_ego(tok):
            if b["num_lidar_pts"] <= 0:
                continue
            m = M._in_box(pts[cur][:, :2], b["centre"], b["wlh"], b["yaw"])
            dec = lab[cur][m]
            dec = dec[dec != M.UNDECIDED]
            rows.append({"n_committed": int(len(dec)),
                         "dyn_frac": float((dec == M.DYNAMIC).mean()) if len(dec) else None,
                         "speed": float(speeds.get(b["instance_token"], 0.0)),
                         "category": b["category"]})
        per_frame.append({"sample_token": tok, "boxes": rows})

    all_rows = [r for f in per_frame for r in f["boxes"]]
    committed = [r for r in all_rows if r["n_committed"] >= min_committed]
    out = {
        "frames": len(toks), "min_speed_mps": min_speed,
        "min_committed_points": min_committed,
        "boxes_with_returns": len(all_rows),
        "boxes_scored": len(committed),
        "coverage": round(len(committed) / max(1, len(all_rows)), 4),
        "moving_boxes_in_scored_set": sum(1 for r in committed if r["speed"] > min_speed),
        "sweeps": [],
    }
    for thr in thresholds:
        tp = fp = fn = tn = 0
        for r in committed:
            pred = r["dyn_frac"] > thr
            truth = r["speed"] > min_speed
            tp += pred and truth; fp += pred and not truth
            fn += (not pred) and truth; tn += (not pred) and not truth
        pr = tp / (tp + fp) if tp + fp else float("nan")
        rc = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else float("nan")
        out["sweeps"].append({"threshold": thr, "tp": tp, "fp": fp, "fn": fn,
                              "tn": tn, "precision": round(pr, 4),
                              "recall": round(rc, 4), "f1": round(f1, 4)})

    best = max(out["sweeps"], key=lambda s: (s["f1"] if s["f1"] == s["f1"] else -1))
    out["best"] = best
    out["caveats"] = [
        "Box-level, not point-level: nuScenes has no per-point motion label.",
        "Speed is differenced between adjacent keyframes at 2 Hz, so slow "
        "objects near the 0.5 m/s threshold are noisy ground truth.",
        f"The classifier abstains outside the previous scan's observed free "
        f"space; only {out['coverage']:.0%} of boxes with returns reach "
        f"{min_committed} committed points and enter the score.",
        "Ground truth is annotation-derived, so any nuScenes annotation error "
        "propagates directly into these numbers.",
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--min-speed", type=float, default=0.5)
    a = ap.parse_args()
    r = run(n_frames=a.frames, start=a.start, min_speed=a.min_speed)
    p = ROOT / "outputs/artifacts/motion_separation_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, indent=2))
    print(json.dumps({k: v for k, v in r.items() if k != "sweeps"}, indent=2))
    print("\nthreshold sweep:")
    for s in r["sweeps"]:
        print("  thr %.2f  P %.3f  R %.3f  F1 %.3f  (tp %d fp %d fn %d tn %d)" % (
            s["threshold"], s["precision"], s["recall"], s["f1"],
            s["tp"], s["fp"], s["fn"], s["tn"]))
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
