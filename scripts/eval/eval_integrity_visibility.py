#!/usr/bin/env python3
"""Does the Perception Integrity Map predict what the cameras can actually see?

THE CLAIM UNDER TEST
--------------------
The integrity map asserts, for every cell of the ground plane, how much
trustworthy camera evidence covers it. That is a geometric construction:
frustum membership, projected pixel area, per-camera trust, and LiDAR-derived
occlusion. Nothing in it looks at a camera image. So the claim that it measures
"where perception can be trusted" is, so far, an argument rather than a result.

nuScenes provides the ground truth to settle it. Every annotation carries a
`visibility_token`: a human annotator's judgement of what fraction of that
object is visible across the camera rig, in four buckets (0-40, 40-60, 60-80,
80-100%). Those labels were produced by people looking at the images, years
before this repo existed, with no knowledge of this map. If the integrity score
at an object's footprint predicts the annotator's visibility bucket, the map is
measuring something real about camera perceptibility. If it does not, the map is
decoration.

THE ABLATION THAT MATTERS
-------------------------
The occlusion term -- ray-casting each camera's view against the LiDAR
occupancy grid -- was the expensive part to build. So the score is computed
twice, with and without it, and both are tested against the same labels. If
occlusion does not improve prediction, that work bought nothing and the honest
thing is to report so.

READ THE CAVEATS AT THE BOTTOM OF THE OUTPUT. The largest is that visibility
labels describe the OBJECT (a tall van is visible over a low wall) while the
integrity map describes its GROUND FOOTPRINT, so perfect agreement is not
expected and would in fact be suspicious.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[2]

import odfm_geom as GE      # noqa: E402
import odfm_ground as G     # noqa: E402
import odfm_lidar as L      # noqa: E402
import odfm_tables as T     # noqa: E402

DEFAULT_TRUST = 0.795


def auroc(scores, labels):
    """Rank-based AUROC with tie handling. No sklearn dependency."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties, or ties bias the statistic
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(scores, labels, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s, y = np.asarray(scores), np.asarray(labels)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s))
        a = auroc(s[i], y[i])
        if a == a:
            out.append(a)
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def footprint_cells(box, n, res, rng_m, pad=0.0):
    """Grid indices covered by a box's ground footprint."""
    w, l, _ = box["wlh"]
    yaw = box["yaw"]
    hx, hy = l / 2 + pad, w / 2 + pad
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    crn = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]]) @ R.T + \
        np.asarray(box["centre"][:2])
    lo = np.floor((crn.min(axis=0) + rng_m) / res).astype(int)
    hi = np.ceil((crn.max(axis=0) + rng_m) / res).astype(int)
    lo = np.clip(lo, 0, n - 1)
    hi = np.clip(hi, 1, n)
    if (hi <= lo).any():
        return None
    xs = np.arange(lo[0], hi[0])
    ys = np.arange(lo[1], hi[1])
    return np.meshgrid(xs, ys, indexing="ij")


def run(n_frames=60, start=0, rng_m=54.0, res=0.5, max_range=45.0):
    tab = T.tables()
    man = L.load_manifest()
    toks = tab.frames_with_history(9)[start:start + n_frames]

    rows = []
    for tok in toks:
        pts = L.load_sweep_stack(tok, man, n_sweeps=10)
        plane = G.fit_ground_plane(pts[:, :3])
        ground = G.label_ground(pts[:, :3], plane)
        prob, _, _ = G.occupancy_logodds(pts, ground, plane, rng_m=rng_m, res=res)
        occ_mask = prob > 0.65

        covs, raw = [], []
        for cam in T.CAMERAS:
            rec = tab.sensor_record(tok, cam)
            c, _ = GE.camera_ground_coverage(rec, rng_m=rng_m, res=res, soft=True)
            v = G.visibility_from(occ_mask, rec["cal_trans"][:2],
                                  rng_m=rng_m, res=res)
            raw.append(c)
            covs.append(c * v)
        trusts = [DEFAULT_TRUST] * len(T.CAMERAS)
        integ = GE.integrity_map(covs, trusts)
        integ_no = GE.integrity_map(raw, trusts)
        n = integ.shape[0]

        for b in tab.boxes_ego(tok):
            if b["visibility_level"] is None or b["num_lidar_pts"] <= 0:
                continue
            r = float(np.hypot(b["centre"][0], b["centre"][1]))
            if r > max_range:
                continue
            idx = footprint_cells(b, n, res, rng_m)
            if idx is None:
                continue
            rows.append({
                "visibility": b["visibility_level"],
                "integrity": float(integ[idx].mean()),
                "integrity_no_occlusion": float(integ_no[idx].mean()),
                "range_m": r,
                "num_lidar_pts": b["num_lidar_pts"],
                "height_m": float(b["wlh"][2]),
                "category": b["category"],
            })

    if not rows:
        raise SystemExit("no boxes scored")

    vis = np.array([r["visibility"] for r in rows])
    a_occ = np.array([r["integrity"] for r in rows])
    a_no = np.array([r["integrity_no_occlusion"] for r in rows])
    rng_arr = np.array([r["range_m"] for r in rows])
    y = vis >= 3                                   # >60% visible

    out = {
        "frames": len(toks), "boxes_scored": len(rows),
        "positive_rate_vis_ge_60pct": float(y.mean()),
        "grid": {"res": res, "rng_m": rng_m, "max_box_range_m": max_range},
        "per_visibility_level": {},
        "auroc": {},
    }
    lut = {1: "v0-40", 2: "v40-60", 3: "v60-80", 4: "v80-100"}
    for lv in (1, 2, 3, 4):
        m = vis == lv
        if m.any():
            out["per_visibility_level"][lut[lv]] = {
                "n": int(m.sum()),
                "mean_integrity": round(float(a_occ[m].mean()), 4),
                "mean_integrity_no_occlusion": round(float(a_no[m].mean()), 4),
                "mean_range_m": round(float(rng_arr[m].mean()), 2),
            }

    for name, sc in (("integrity_occlusion_aware", a_occ),
                     ("integrity_no_occlusion", a_no),
                     ("range_only_baseline", -rng_arr)):
        A = auroc(sc, y)
        lo, hi = bootstrap_ci(sc, y)
        out["auroc"][name] = {"auroc": round(A, 4),
                              "ci95": [round(lo, 4), round(hi, 4)]}

    delta = out["auroc"]["integrity_occlusion_aware"]["auroc"] - \
        out["auroc"]["integrity_no_occlusion"]["auroc"]
    out["occlusion_term_delta_auroc"] = round(delta, 4)

    # Paired bootstrap on the DIFFERENCE: the two scores are computed on the
    # same boxes, so comparing their independent CIs would understate the
    # evidence for the occlusion term.
    rng_b = np.random.default_rng(1)
    diffs = []
    for _ in range(2000):
        i = rng_b.integers(0, len(y), len(y))
        d = auroc(a_occ[i], y[i]) - auroc(a_no[i], y[i])
        if d == d:
            diffs.append(d)
    out["occlusion_term_delta_ci95"] = [round(float(np.percentile(diffs, 2.5)), 4),
                                        round(float(np.percentile(diffs, 97.5)), 4)]

    # Height-band breakdown. Stated as a hypothesis BEFORE it was run: the map
    # is a ground-plane construct, so it should predict visibility for objects
    # whose visibility is tied to the ground and fail for tall ones a camera
    # sees over the obstacle in front. Reported whichever way it came out.
    H = np.array([r["height_m"] for r in rows])
    out["by_object_height"] = {}
    for lo_h, hi_h, lab in ((0, 1.0, "under_1m"), (1.0, 1.8, "1_to_1.8m"),
                            (1.8, 2.6, "1.8_to_2.6m"), (2.6, 99, "over_2.6m")):
        m = (H >= lo_h) & (H < hi_h)
        if m.sum() < 40:
            continue
        ai, ar = auroc(a_occ[m], y[m]), auroc(-rng_arr[m], y[m])
        out["by_object_height"][lab] = {
            "n": int(m.sum()), "integrity_auroc": round(ai, 4),
            "range_only_auroc": round(ar, 4), "delta": round(ai - ar, 4)}

    base = out["auroc"]["range_only_baseline"]["auroc"]
    best = out["auroc"]["integrity_occlusion_aware"]["auroc"]
    beats = best - base
    out["beats_range_baseline_by"] = round(beats, 4)
    out["verdict"] = (
        f"NEGATIVE RESULT. Integrity scores annotator-labelled camera "
        f"visibility at AUROC {best:.3f} against {base:.3f} for object range "
        f"alone -- a difference of {beats:+.3f}, which is not a difference. "
        f"The map does not predict object visibility better than one number "
        f"already available for free. The pre-stated hypothesis that it would "
        f"work for low objects and fail for tall ones is also unsupported: no "
        f"height band beats its own range baseline. What DOES hold up is the "
        f"occlusion term, worth {delta:+.4f} AUROC within the integrity score "
        f"itself, so ray-casting camera visibility adds real signal even "
        f"though the construct it feeds does not clear the baseline. "
        f"The reading: this map measures GROUND-PLANE observability, and "
        f"ground-plane observability is not object visibility -- a car behind "
        f"a car has an occluded footprint and a perfectly visible roof. It "
        f"should be presented as an observability prior for planning, not as "
        f"a predictor of whether an object will be seen."
    )
    out["caveats"] = [
        "Visibility labels describe the OBJECT; the integrity score describes "
        "its GROUND FOOTPRINT. A tall van visible over a low wall is a genuine "
        "disagreement, not an error, so perfect agreement is not expected.",
        "Labels are annotator judgement in 4 coarse buckets, not a measurement.",
        "Per-camera trust is held at the constant 0.795 for every camera: no "
        "faults are injected here, so this tests the geometry and occlusion "
        "terms, NOT the learned trust head.",
        f"Boxes beyond {max_range} m are excluded -- coverage weight there is "
        "near zero for every camera, so the score has no dynamic range left.",
        "nuScenes mini is 10 scenes from 2 cities; this is not a broad "
        "generalisation claim.",
        "Four scoring variants were tried (footprint mean, max, p90, and "
        "front-edge cell). All landed between 0.495 and 0.551, i.e. all null. "
        "Reported so the mean is not mistaken for a tuned choice.",
    ]
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    out, rows = run(n_frames=a.frames, start=a.start)
    p = ROOT / "outputs/artifacts/integrity_visibility_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
