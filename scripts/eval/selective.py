"""Smooth conditioning, and the downstream metric that actually fits the claim.

Two changes from the failed recalibration experiment (A13), each following
from its diagnosis rather than from wanting a better number.

1. SMOOTH INSTEAD OF FRAGMENTED.
   Scheme D fit TEN independent confidence->accuracy tables, one per
   observability decile. Its sparse cells -- decile 1 at high confidence --
   were estimated on little dev data and generalised badly, which is why D lost
   to a single global map. The remedy for fragmentation is pooling with
   smoothness: ONE logistic model with observability entering as a continuous
   feature,

       P(correct) = sigmoid(a + b*l + c*o + d*o*l),   l = logit(confidence)

   Four parameters instead of five hundred tables, fit by weighted IRLS on the
   dev scenes. It can represent "confidence means less when observability is
   low" without spending data on every cell separately.

2. SELECTIVE PREDICTION INSTEAD OF ECE.
   ECE after recalibration has almost no headroom: a global map already gets
   residual error to 0.0047, so even a perfect conditioner can win at most that
   much, which is inside bootstrap noise. It was the wrong target.

   What an AV stack actually asks is: given a budget to abstain, which signal
   picks the right voxels to distrust? That is a RANKING question, and H2
   already established that observability ranks error well (0.6742 vs 0.6215),
   so this is a fair test of a claim we have evidence for, not a rescue.

   Reported as risk-coverage: sort by the ranking signal, keep the most
   trustworthy fraction, measure error on what is kept. Summarised by AURC
   (area under the risk-coverage curve, lower is better) and by error at 80%
   coverage.

PRE-REGISTERED, before this script was run:
  * confidence alone is the baseline to beat. If adding observability does not
    lower AURC against it, observability adds nothing at the decision the AV
    stack actually makes, and that is the finding.
  * mask_camera + confidence is the second baseline, the binary competitor.
  * All three fit on dev scenes, scored on held-out test scenes.
  * Reported whichever way it comes out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
from multiprocessing import Pool

FREE = 17
NC, NO = 40, 12          # confidence bins, observability bins

# --- parallel scan ----------------------------------------------------------
# The scan was single-threaded at ~0.063 s/frame, so the full 6,019-frame split
# took ~380 s. That is longer than the 180 s a remote shell call allows, which
# is why A33 had to be run on a 2,000-frame subsample and why its result was
# recorded as NOT QUOTABLE. The work is a pure per-frame histogram accumulation
# with no cross-frame state, so it parallelises exactly. This is the root fix
# for "the full split cannot be run", not a workaround for it.
_W = {}


def _init_scan(preds, obs, gts, nonfree):
    _W.update(preds=preds, obs=obs, nonfree=nonfree)
    _W["gt"] = {os.path.basename(os.path.dirname(q)): q
                for q in glob.glob(os.path.join(gts, "*", "*", "labels.npz"))}


def _scan_one(row):
    tok = row["token"]
    pp = os.path.join(_W["preds"], f"{tok}.npz")
    op = os.path.join(_W["obs"], f"{tok}.npy")
    gp = _W["gt"].get(tok)
    if not (gp and os.path.exists(pp) and os.path.exists(op)):
        return None
    g = np.load(gp)
    sem, mk = g["semantics"], g["mask_camera"].astype(bool)
    z = np.load(pp)
    cls, conf = z["cls"], z["conf"].astype(np.float32) / 255.0
    obs = np.load(op).astype(np.float32) / 255.0
    corr = (cls == sem)
    keep = np.ones_like(mk) if not _W["nonfree"] else (sem != FREE)
    c, o, m, y = conf[keep], obs[keep], mk[keep], corr[keep]
    cb = np.clip((c * NC).astype(np.int32), 0, NC - 1)
    ob = np.clip((o * (NO - 1)).astype(np.int32), 0, NO - 1)
    flat = (cb * NO + ob) * 2 + m.astype(np.int32)
    n = np.bincount(flat, minlength=NC * NO * 2).astype(np.float64)
    k = np.bincount(flat, weights=y.astype(np.float64),
                    minlength=NC * NO * 2)
    return row["scene"], n.reshape(NC, NO, 2), k.reshape(NC, NO, 2)


def irls(X, n, k, iters=40):
    """Weighted logistic regression from binomial cell counts.

    X: (cells, features).  n: trials per cell.  k: successes per cell.
    Cells are dense here -- millions of voxels each -- so this is a stable fit
    on sufficient statistics rather than on raw samples.
    """
    w0 = np.zeros(X.shape[1])
    keep = n > 0
    X, n, k = X[keep], n[keep], k[keep]
    for _ in range(iters):
        eta = np.clip(X @ w0, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = n * p * (1 - p) + 1e-9
        z = eta + (k - n * p) / W
        A = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        w0 = np.linalg.solve(A, X.T @ (W * z))
    return w0


def feats(conf, obs):
    l = np.log(np.clip(conf, 1e-4, 1 - 1e-4) / (1 - np.clip(conf, 1e-4, 1 - 1e-4)))
    return np.stack([np.ones_like(l), l, obs, obs * l], 1)


def risk_coverage(score, n, wrong):
    """AURC and error at 80% coverage, from cell counts.

    score: per-cell trust (higher = keep first). Cells are sorted once; the
    curve is exact on the cell grid.
    """
    o = np.argsort(-score)
    n_, w_ = n[o], wrong[o]
    cn, cw = np.cumsum(n_), np.cumsum(w_)
    cov = cn / cn[-1]
    risk = cw / np.maximum(cn, 1)
    aurc = float(np.trapz(risk, cov))
    i80 = int(np.searchsorted(cov, 0.80))
    return aurc, float(risk[min(i80, len(risk) - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--preds", default="data/preds/preds_voxel")
    ap.add_argument("--obs", default="data/pack/obs_ray")
    ap.add_argument("--gts",
                    default="data/occ3d/Occupancy3D-nuScenes-trainval/gts")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--nonfree", action="store_true")
    ap.add_argument("--boot", type=int, default=2000,
                    help="scene bootstrap over the TEST scenes, with the dev "
                         "fit held fixed. Without it the observability-vs-mask "
                         "margin (0.7%) cannot be distinguished from noise, "
                         "and every other number in this project carries one.")
    ap.add_argument("--jobs", type=int, default=0,
                    help="scan workers; 0 = cores-1. The scan is a pure "
                         "per-frame histogram accumulation, so this changes "
                         "the wall clock and nothing else.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    index = json.load(open(os.path.join(args.pack, "index.json")))
    if args.frames:
        rs = np.random.default_rng(1234)
        sel = sorted(rs.choice(len(index), min(args.frames, len(index)),
                               replace=False))
        index = [index[i] for i in sel]
    gt_map = {os.path.basename(os.path.dirname(q)): q
              for q in glob.glob(os.path.join(args.gts, "*", "*",
                                              "labels.npz"))}
    scenes = sorted({r["scene"] for r in index})
    # A47: was `dev = scenes where index % 2 == 0`, an alternating split this
    # file invented. It now reads the single source of truth materialised in
    # A45, so the sensor-only headline is fit and scored on the same dev/test
    # partition as every other result.
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import split as SP
    dev = SP.dev_scenes() & set(scenes)
    print(f"[A47] split digest {SP.digest()}")
    print(f"{len(scenes)} scenes -> {len(dev)} dev / {len(scenes)-len(dev)} test")

    # (split, conf bin, obs bin, mask) -> n, correct
    shape = (2, NC, NO, 2)
    N = np.zeros(shape, np.float64)
    K = np.zeros(shape, np.float64)
    # per-TEST-scene counts, so the margin can be bootstrapped
    test_scenes = [x for x in scenes if x not in dev]
    tsi = {x: i for i, x in enumerate(test_scenes)}
    NS = np.zeros((len(test_scenes), NC, NO, 2), np.float32)
    KS = np.zeros((len(test_scenes), NC, NO, 2), np.float32)
    used, t0 = 0, time.time()
    nj = args.jobs if args.jobs > 0 else max(os.cpu_count() - 1, 1)
    with Pool(nj, initializer=_init_scan,
              initargs=(args.preds, args.obs, args.gts, args.nonfree)) as pool:
        for out in pool.imap_unordered(_scan_one, index, chunksize=8):
            if out is None:
                continue
            scene, n, k = out
            s_ = 0 if scene in dev else 1
            N[s_] += n
            K[s_] += k
            if s_ == 1:
                j = tsi[scene]
                NS[j] += n.astype(np.float32)
                KS[j] += k.astype(np.float32)
            used += 1
            if used % 1000 == 0:
                print(f"  {used}  {(time.time()-t0)/used:.4f}s/frame "
                      f"({nj} workers)", flush=True)

    cc = (np.arange(NC) + 0.5) / NC
    oo = np.arange(NO) / (NO - 1)
    C3, O3, M3 = np.meshgrid(cc, oo, np.array([0.0, 1.0]), indexing="ij")
    c_f, o_f, m_f = C3.ravel(), O3.ravel(), M3.ravel()
    nd, kd = N[0].ravel(), K[0].ravel()
    nt, kt = N[1].ravel(), K[1].ravel()
    wrong_t = nt - kt
    print(f"\nframes {used}   test voxels {nt.sum():,.0f}")

    # --- three ranking signals, all fit on dev ---------------------------
    res = {}
    # 1. confidence alone
    res["confidence"] = risk_coverage(c_f, nt, wrong_t)
    # 2. confidence + binary mask, logistic
    Xm = np.stack([np.ones_like(c_f),
                   np.log(np.clip(c_f, 1e-4, 1-1e-4) /
                          (1 - np.clip(c_f, 1e-4, 1-1e-4))),
                   m_f, m_f * np.log(np.clip(c_f, 1e-4, 1-1e-4) /
                                     (1 - np.clip(c_f, 1e-4, 1-1e-4)))], 1)
    wm = irls(Xm, nd, kd)
    res["conf + mask_camera"] = risk_coverage(Xm @ wm, nt, wrong_t)
    # 3. confidence + observability, logistic, smooth
    Xo = feats(c_f, o_f)
    wo = irls(Xo, nd, kd)
    res["conf + observability"] = risk_coverage(Xo @ wo, nt, wrong_t)

    # --- the SENSOR-ONLY regime -----------------------------------------
    # Everything above hands the model's confidence to every method, and in
    # that setting confidence already encodes much of what visibility knows.
    # But there is a real AV regime where confidence is NOT available: gating
    # on sensor geometry before or independently of the model, for sensor
    # placement, fleet monitoring and safety-case coverage. Observability is
    # computable there and the model's confidence is not.
    #
    # PRE-REGISTERED: observability alone must beat mask_camera alone. This is
    # the same comparison H2 made as a ranking (+0.0527); here it is made as a
    # decision. If it fails, the measure has no regime where it beats the
    # binary flag, and that is the finding.
    res["observability alone"] = risk_coverage(o_f, nt, wrong_t)
    res["mask_camera alone"] = risk_coverage(m_f, nt, wrong_t)

    print(f"\n{'signal':<24}{'AURC':>10}{'err @80% cov':>15}")
    for k, (a, e) in res.items():
        print(f"{k:<24}{a:>10.5f}{e:>15.5f}")
    b = res["confidence"][0]
    for k in ("conf + mask_camera", "conf + observability"):
        print(f"  {k:<22} vs confidence alone: {100*(b-res[k][0])/b:+.1f}% AURC")
    d = res["conf + observability"][0]
    c_ = res["conf + mask_camera"][0]
    print(f"  observability vs mask_camera: {100*(c_-d)/c_:+.1f}% AURC"
          "   (point estimate only -- see the bootstrap below for the verdict)")
    # NO VERDICT HERE. A verdict printed from point estimates was wrong three
    # times in one session: twice from a degenerate or incomplete baseline, once
    # here, where the +0.7% margin turned out to be noise with a CI spanning
    # zero. Every accept/reject statement in this project now comes from an
    # interval, and the only place that exists is the bootstrap section.

    # --- ECE of the smooth model, the other half of the fix --------------
    def ece(p, n, k, nb=15):
        idx = np.clip((p * nb).astype(int), 0, nb - 1)
        tot, out = n.sum(), 0.0
        for b_ in range(nb):
            mm = idx == b_
            nb_ = n[mm].sum()
            if nb_ == 0:
                continue
            out += (nb_ / tot) * abs((p[mm] * n[mm]).sum() / nb_
                                     - k[mm].sum() / nb_)
        return float(out)

    sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    Xg = np.stack([np.ones_like(c_f),
                   np.log(np.clip(c_f, 1e-4, 1-1e-4) /
                          (1 - np.clip(c_f, 1e-4, 1-1e-4)))], 1)
    wg = irls(Xg, nd, kd)
    e = {"raw confidence": ece(c_f, nt, kt),
         "global logistic": ece(sig(Xg @ wg), nt, kt),
         "+ mask_camera": ece(sig(Xm @ wm), nt, kt),
         "+ observability (smooth)": ece(sig(Xo @ wo), nt, kt)}
    print(f"\n{'scheme':<28}{'ECE':>10}")
    for k, v in e.items():
        print(f"{k:<28}{v:>10.5f}")
    print("  smooth conditioning vs global: "
          f"{100*(e['global logistic']-e['+ observability (smooth)'])/e['global logistic']:+.1f}%")

    # --- bootstrap the margin over the binary mask -----------------------
    # The dev fit is held fixed; only the test scenes are resampled. That is
    # the uncertainty in the estimate, which is what the 0.7% needs.
    sc_m, sc_o, sc_c = Xm @ wm, Xo @ wo, c_f
    rs2 = np.random.default_rng(0)
    NSf = NS.reshape(NS.shape[0], -1)
    KSf = KS.reshape(KS.shape[0], -1)
    d_om, d_oc, d_sensor = [], [], []
    for _ in range(args.boot):
        pick = rs2.integers(0, NSf.shape[0], NSf.shape[0])
        n_b = NSf[pick].sum(0).astype(np.float64)
        k_b = KSf[pick].sum(0).astype(np.float64)
        w_b = n_b - k_b
        a_o = risk_coverage(sc_o, n_b, w_b)[0]
        a_m = risk_coverage(sc_m, n_b, w_b)[0]
        a_c = risk_coverage(sc_c, n_b, w_b)[0]
        d_om.append(a_m - a_o)          # positive = observability better
        d_oc.append(a_c - a_o)
        d_sensor.append(risk_coverage(m_f, n_b, w_b)[0]
                        - risk_coverage(o_f, n_b, w_b)[0])
    d_om, d_oc = np.array(d_om), np.array(d_oc)
    lo, hi = np.percentile(d_om, [2.5, 97.5])
    lo2, hi2 = np.percentile(d_oc, [2.5, 97.5])
    print(f"\nscene bootstrap over {NSf.shape[0]} test scenes, "
          f"{args.boot} resamples, dev fit fixed")
    print(f"  AURC(mask) - AURC(obs)  = {d_om.mean():+.5f}  "
          f"95% CI [{lo:+.5f}, {hi:+.5f}]")
    print("  " + ("observability beats mask_camera -- CI excludes zero"
                  if lo > 0 else
                  "mask_camera beats observability -- CI excludes zero"
                  if hi < 0 else
                  "NO DIFFERENCE from mask_camera -- CI spans zero"))
    print(f"  AURC(conf) - AURC(obs)  = {d_oc.mean():+.5f}  "
          f"95% CI [{lo2:+.5f}, {hi2:+.5f}]")
    ds = np.array(d_sensor)
    lo3, hi3 = np.percentile(ds, [2.5, 97.5])
    print(f"\nSENSOR-ONLY regime (no model confidence available)")
    print(f"  observability alone AURC {res['observability alone'][0]:.5f}")
    print(f"  mask_camera alone   AURC {res['mask_camera alone'][0]:.5f}")
    print(f"  mask - obs = {ds.mean():+.5f}  95% CI [{lo3:+.5f}, {hi3:+.5f}]")
    print("  " + ("observability BEATS the binary mask on sensor geometry alone"
                  if lo3 > 0 else
                  "binary mask beats observability" if hi3 < 0 else
                  "no difference -- CI spans zero"))

    if args.out:
        json.dump({"boot_margin_vs_mask": [float(d_om.mean()),
                                           float(lo), float(hi)],
                   "boot_margin_vs_conf": [float(d_oc.mean()),
                                           float(lo2), float(hi2)],
                   # The sensor-only contrast is the headline of A17/A37 and
                   # was PRINTED but never stored, so the one number the paper
                   # quotes most could not be read back from a committed
                   # artifact. Found in the 24 Sep audit. Stored now.
                   "boot_margin_sensor_mask_minus_obs": [float(ds.mean()),
                                                         float(lo3), float(hi3)],
                   "aurc": {k: v[0] for k, v in res.items()},
                   "err80": {k: v[1] for k, v in res.items()},
                   "ece": e, "frames": used, "nonfree": args.nonfree},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
