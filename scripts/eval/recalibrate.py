"""Does observability-conditioned recalibration beat mask_camera-conditioned?

The argument for a continuous measure that does not depend on winning an AUROC
contest. A binary mask can only ever give you TWO corrections: one for
"visible", one for "not". A continuous measure gives a curve. If calibration
error really varies with how well a voxel is seen -- and A12 measured a gap
running +0.0338 to +0.0992 -- then a two-valued correction must leave most of
that on the table, by construction.

PRE-REGISTRATION, written before this script was ever run
---------------------------------------------------------
The Occ3D data already exists on disk, so this is weaker than A5's
pre-registration and is labelled as such. What it does guarantee: the schemes,
the split, the metric and the acceptance thresholds below were fixed before a
single number came out, and are not revised afterwards.

Four schemes, each fit on the dev scenes and scored on the test scenes:

    A  none        raw confidence, no correction
    B  global      one recalibration map for every voxel
    C  mask_camera two maps, inside and outside the binary mask   <- baseline
    D  observability  one map per observability decile            <- ours

Recalibration is histogram binning on max-softmax: within a group, the
calibrated probability for a confidence bin is the empirical accuracy of that
bin on DEV. Chosen over temperature scaling because the export kept the argmax
probability, not the logits, and histogram binning needs only what we have. It
is also the weaker, more conservative choice: it cannot exploit a
parametric form that happens to suit us.

Split: scenes, not frames, alternating by sorted scene name. Consecutive
keyframes share objects, so a frame-level split would leak.

Metric: expected calibration error on the test scenes, 15 equal-width bins.

ACCEPTANCE, fixed here:
  * D must beat C. If a continuous measure cannot beat two numbers, the
    contribution does not exist and the paper says so.
  * D must cut ECE by >= 40% against A.
  * Reported whichever way it comes out, with C and B alongside. A null here is
    a finding about the size of the contribution, not a reason to go looking
    for a friendlier statistic.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

OCC_NZ = 16
FREE = 17
NCONF = 50          # recalibration-map resolution
NECE = 15           # ECE bins, standard


def ece_from_counts(pred_p, n, correct):
    """ECE given, per bin: the probability we assert, the count, the hits."""
    tot = n.sum()
    if tot == 0:
        return float("nan")
    idx = np.clip((pred_p * NECE).astype(int), 0, NECE - 1)
    out = 0.0
    for b in range(NECE):
        m = idx == b
        nb = n[m].sum()
        if nb == 0:
            continue
        conf = (pred_p[m] * n[m]).sum() / nb
        acc = correct[m].sum() / nb
        out += (nb / tot) * abs(conf - acc)
    return float(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--preds", default="data/preds/preds_voxel")
    ap.add_argument("--obs", default="data/pack/obs_ray")
    ap.add_argument("--gts",
                    default="data/occ3d/Occupancy3D-nuScenes-trainval/gts")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--nonfree", action="store_true")
    ap.add_argument("--split-zero", action="store_true",
                    help="scheme D splits the zero bin into out-of-view and "
                         "occluded. A13 showed those two populations behave "
                         "oppositely, so pooling them makes a bad group to "
                         "recalibrate, however well observability ranks.")
    ap.add_argument("--scope", choices=["full", "mask"], default="full",
                    help="'mask' scores only inside mask_camera, which makes "
                         "the mask grouping CONSTANT and silently collapses "
                         "scheme C into scheme B -- they printed identical "
                         "ECE to five decimals. The mask baseline only exists "
                         "over the full volume, so that is the default.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import importlib.util as _il
    _sp = _il.spec_from_file_location("odfm_eval", os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "eval_observability_scaled.py"))
    _m = _il.module_from_spec(_sp); _sp.loader.exec_module(_m)
    quat_to_rot = _m.quat_to_rot
    import frustum_util as FU
    calib = json.load(open(os.path.join(args.pack, "calib.json")))
    index = json.load(open(os.path.join(args.pack, "index.json")))
    if args.frames:
        rs = np.random.default_rng(1234)
        sel = sorted(rs.choice(len(index), min(args.frames, len(index)),
                               replace=False))
        index = [index[i] for i in sel]
    gt_map = {os.path.basename(os.path.dirname(q)): q
              for q in glob.glob(os.path.join(args.gts, "*", "*",
                                              "labels.npz"))}

    # decile edges from observability alone, no predictions touched
    rs = np.random.default_rng(0)
    vals = []
    for i in rs.choice(len(index), min(300, len(index)), replace=False):
        p = os.path.join(args.obs, f"{index[i]['token']}.npy")
        if os.path.exists(p):
            v = np.load(p).astype(np.float32).ravel() / 255.0
            vals.append(v[v > 0])
    edges = np.unique(np.percentile(np.concatenate(vals),
                                    np.linspace(0, 100, 11)[1:-1]))
    NOBS = len(edges) + 2
    print(f"observability bins: {NOBS} (zero + {NOBS-1} deciles)\n")

    scenes = sorted({r["scene"] for r in index})
    dev = {s for i, s in enumerate(scenes) if i % 2 == 0}
    print(f"{len(scenes)} scenes -> {len(dev)} dev / {len(scenes)-len(dev)} test")

    # counts[split][scheme_group][conf_bin] = (n, hits)
    NOBS_EFF = NOBS + (1 if args.split_zero else 0)
    shapes = {"global": 1, "mask": 2, "obs": NOBS_EFF}
    C = {sp: {k: np.zeros((v, NCONF, 2), np.float64)
              for k, v in shapes.items()} for sp in ("dev", "test")}

    used, t0 = 0, time.time()
    for row in index:
        tok = row["token"]
        pp = os.path.join(args.preds, f"{tok}.npz")
        op = os.path.join(args.obs, f"{tok}.npy")
        gp = gt_map.get(tok)
        if not (gp and os.path.exists(pp) and os.path.exists(op)):
            continue
        g = np.load(gp)
        sem, mk = g["semantics"], g["mask_camera"].astype(bool)
        z = np.load(pp)
        cls = z["cls"]
        conf = z["conf"].astype(np.float32) / 255.0
        obs = np.load(op).astype(np.float32) / 255.0
        correct = (cls == sem)

        if args.scope == "mask":
            keep = mk if not args.nonfree else (mk & (sem != FREE))
        else:
            keep = (np.ones_like(mk) if not args.nonfree else (sem != FREE))
        cf, co, ob = conf[keep], correct[keep], obs[keep]
        mkk = mk[keep]                          # the actual binary baseline
        cb = np.clip((cf * NCONF).astype(np.int32), 0, NCONF - 1)
        if args.split_zero:
            fr = FU.in_any_frustum(calib[tok]["cams"], quat_to_rot).ravel()[
                keep.ravel()]
            ob_b = np.where(ob <= 0, np.where(fr, 1, 0),
                            np.digitize(ob, edges) + 2)
        else:
            ob_b = np.where(ob <= 0, 0, np.digitize(ob, edges) + 1)

        sp = "dev" if row["scene"] in dev else "test"
        for name, grp in (("global", np.zeros_like(cb)),
                          ("mask", mkk.astype(np.int32)),
                          ("obs", ob_b.astype(np.int32))):
            a = C[sp][name]
            np.add.at(a[..., 0], (grp, cb), 1.0)
            np.add.at(a[..., 1], (grp, cb), co.astype(np.float64))
        used += 1
        if used % 500 == 0:
            el = time.time() - t0
            print(f"  {used}/{len(index)}  {el/used:.3f}s/frame", flush=True)

    centres = (np.arange(NCONF) + 0.5) / NCONF
    res = {}

    # A: no correction. assert the raw confidence.
    t = C["test"]["global"]
    res["A none"] = ece_from_counts(np.tile(centres, (1, 1)).ravel(),
                                    t[0, :, 0], t[0, :, 1])

    for name, label in (("global", "B global"), ("mask", "C mask_camera"),
                        ("obs", "D observability")):
        d, t = C["dev"][name], C["test"][name]
        # map: dev accuracy per (group, conf bin); fall back to the bin centre
        # where dev has no support, which is the honest no-information choice
        with np.errstate(invalid="ignore", divide="ignore"):
            m = d[..., 1] / d[..., 0]
        m = np.where(d[..., 0] > 0, m, np.tile(centres, (m.shape[0], 1)))
        res[label] = ece_from_counts(m.ravel(), t[..., 0].ravel(),
                                     t[..., 1].ravel())

    print(f"\nframes {used}   scheme ECE on held-out scenes")
    for k in ("A none", "B global", "C mask_camera", "D observability"):
        print(f"  {k:<18} {res[k]:.5f}")
    a, b_, c, d = (res["A none"], res["B global"], res["C mask_camera"],
                   res["D observability"])
    print(f"\n  D vs A: {100*(a-d)/a:+.1f}%   (acceptance: >= 40% reduction)")
    print(f"  D vs C: {100*(c-d)/c:+.1f}%   (acceptance: D must beat C)")
    print(f"  D vs B: {100*(b_-d)/b_:+.1f}%   (MANDATORY: D must beat the "
          f"unconditioned map)")
    ok = d < c and d < b_ and (a - d) / a >= 0.40
    print("  " + ("ALL ACCEPTANCE CRITERIA MET" if ok else
                  "acceptance NOT met -- report as is"))
    # A13: the original criteria omitted B, so a result where conditioning
    # actively HURTS printed as a pass. B is mandatory from here on.
    if args.out:
        json.dump({"ece": res, "frames": used, "nonfree": args.nonfree,
                   "edges": edges.tolist()}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
