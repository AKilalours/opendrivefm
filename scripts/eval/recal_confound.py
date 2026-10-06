#!/usr/bin/env python3
"""Is mask_camera winning the recalibration because it is a BETTER measure,
or because it leaks the validity of the label the error is measured against?

Three attempts at observability-conditioned recalibration lost to the binary
mask_camera flag, and A34 closed the question with "there is no residual to
model". That conclusion has a hole in it that none of the four analyses tested.

The hole
--------
Error is defined as `predicted class != Occ3D ground-truth class`. Occ3D builds
`mask_camera` OFFLINE from the full sensor suite and uses it to mark WHICH
VOXELS THE LABEL IS VALID FOR -- it is a label-validity flag, and A34 says so
in its own text. Outside the mask the semantics were never supervised, so a
"wrong" voxel there is partly a statement about the ground truth rather than
about the model.

So when mask_camera is handed to a recalibrator as a feature, it can do
something observability structurally cannot: point at the region where the
TARGET is unreliable. That is not a better visibility measure winning. That is
a feature leaking the construction of the label.

The test
--------
Restrict to `mask_camera == 1`, where every label is valid by construction, and
ask the A34 question again:

    conditional on predicted class and the model's own confidence,
    does accuracy still depend on observability?

Inside that stratum mask_camera is constant and cannot compete, which is
exactly the A13/A34 trap when the question was "obs vs mask" -- but it is NOT a
trap here, because the comparison is obs vs NOTHING. The baseline is
class+confidence, which is fully expressible inside the stratum.

Three outcomes, all informative:

  1. observability adds inside the mask     -> the signal is real, the previous
     failures were about the target, and Stage 5 reopens on valid labels only.
  2. observability adds nothing inside      -> A34's conclusion survives the
     strongest objection to it and the question is closed for good.
  3. mask_camera's whole advantage sits in the mask == 0 stratum -> quantifies
     the leak directly, whichever way (1) and (2) land.

No new pass over the data: this reads the same sufficient statistics A34 built.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import split as SP                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NC, NCONF, NOBS, NM = 18, 20, 10, 2


def logloss(p, n, k):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum() / max(n.sum(), 1)


def ece(p, n, k, nb=15):
    e = np.linspace(0, 1, nb + 1)
    b = np.clip(np.digitize(p, e[1:-1]), 0, nb - 1)
    tot = np.bincount(b, weights=n, minlength=nb)
    sp = np.bincount(b, weights=p * n, minlength=nb)
    sk = np.bincount(b, weights=k, minlength=nb)
    m = tot > 0
    return float((tot[m] / tot.sum() * np.abs(sp[m] / tot[m] - sk[m] / tot[m])).sum())


def models(nt, kt, shape):
    """Laplace-smoothed accuracy tables, pooled over the axes each model ignores."""
    axes_all = tuple(range(1, len(shape)))
    cls_n = nt.sum(axes_all, keepdims=True)
    cls_k = kt.sum(axes_all, keepdims=True)
    prior = (cls_k + 1) / (cls_n + 2)

    def pool(axes):
        n_ = nt.sum(axes, keepdims=True); k_ = kt.sum(axes, keepdims=True)
        return np.broadcast_to((k_ + 20 * prior) / (n_ + 20), shape)
    return prior, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="outputs/artifacts/recal_rc_store.npz")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--out", default="outputs/artifacts/recal_confound.json")
    a = ap.parse_args()

    z = np.load(os.path.join(ROOT, a.store), allow_pickle=True)
    scenes = list(z["scenes"]); N, K = z["n"], z["k"]
    full = (NC, NCONF, NOBS, NM)
    sub = (NC, NCONF, NOBS)
    rng = np.random.default_rng(a.seed)
    # A47: the split now comes from scripts/eval/split.py, the single
    # source of truth materialised in A45, instead of this file
    # inventing its own.
    dset, tset = SP.dev_scenes(), SP.test_scenes()
    dev = np.array([i for i, s in enumerate(scenes) if str(s) in dset])
    tst = np.array([i for i, s in enumerate(scenes) if str(s) in tset])
    print(f"[A47] split digest {SP.digest()} | dev {len(dev)} / test {len(tst)} scenes")
    if len(dev) == 0 or len(tst) == 0:
        raise SystemExit("split did not match the store scene names")

    def stratum(idx_tr, idx_te, mk):
        """mk = 1 (labels valid), 0 (unsupervised), or None (both, A34's scope)."""
        nt = N[idx_tr].sum(0).reshape(full)
        kt = K[idx_tr].sum(0).reshape(full)
        ne = N[idx_te].sum(0).reshape(full)
        ke = K[idx_te].sum(0).reshape(full)
        if mk is None:
            return nt, kt, ne, ke, full
        return (nt[..., mk], kt[..., mk], ne[..., mk], ke[..., mk], sub)

    def fit(idx_tr, idx_te, mk):
        nt, kt, ne, ke, shape = stratum(idx_tr, idx_te, mk)
        prior, pool = models(nt, kt, shape)
        if mk is None:
            P = {"class + confidence": pool((2, 3)),
                 "+ mask_camera": pool((2,)),
                 "+ observability": pool((3,))}
        else:
            P = {"class + confidence": pool((2,)),
                 "+ observability": np.broadcast_to(
                     (kt + 20 * pool((2,))) / (nt + 20), shape)}
        return ne, ke, P

    out = {}
    print(f"\n{len(scenes)} scenes | held out {len(tst)} | "
          f"{int(N.sum()):,} voxels total")

    # --- how big is each stratum, and how does raw accuracy differ -----------
    tot = N.sum(0).reshape(full)
    hit = K.sum(0).reshape(full)
    n1, n0 = int(tot[..., 1].sum()), int(tot[..., 0].sum())
    a1 = hit[..., 1].sum() / max(n1, 1)
    a0 = hit[..., 0].sum() / max(n0, 1)
    print("=" * 74)
    print(f"{'stratum':<34}{'voxels':>16}{'share':>10}{'accuracy':>12}")
    print("-" * 74)
    print(f"{'mask_camera = 1  labels valid':<34}{n1:>16,}{100*n1/(n1+n0):>9.1f}%{a1:>12.4f}")
    print(f"{'mask_camera = 0  unsupervised':<34}{n0:>16,}{100*n0/(n1+n0):>9.1f}%{a0:>12.4f}")
    print(f"{'accuracy difference':<34}{'':>16}{'':>10}{a1-a0:>+12.4f}")
    out["stratum"] = dict(n_valid=n1, n_unsupervised=n0,
                          acc_valid=float(a1), acc_unsupervised=float(a0))

    # --- the A34 comparison, reproduced, then repeated inside the mask ------
    for label, mk in (("ALL voxels (A34's scope)", None),
                      ("mask_camera == 1 only (labels valid)", 1),
                      ("mask_camera == 0 only (unsupervised)", 0)):
        ne, ke, P = fit(dev, tst, mk)
        m = ne > 0
        print("\n" + "=" * 74)
        print(f"{label}   held-out voxels {int(ne.sum()):,}")
        print("-" * 74)
        print(f"{'model for p(correct)':<34}{'log-loss':>13}{'ECE':>13}{'vs base':>14}")
        base = logloss(P["class + confidence"][m], ne[m], ke[m])
        rows = {}
        for nm, p_ in P.items():
            ll = logloss(p_[m], ne[m], ke[m]); ec = ece(p_[m], ne[m], ke[m])
            d = "" if nm == "class + confidence" else f"{100*(base-ll)/base:+.3f}%"
            print(f"{nm:<34}{ll:>13.6f}{ec:>13.6f}{d:>14}")
            rows[nm] = dict(logloss=float(ll), ece=float(ec))
        out[label] = rows

    # --- bootstrap the one contrast this file exists to settle --------------
    d_in, d_leak = [], []
    for _ in range(a.boot):
        dd = rng.choice(dev, len(dev), replace=True)
        tt = rng.choice(tst, len(tst), replace=True)
        ne, ke, P = fit(dd, tt, 1)
        m = ne > 0
        d_in.append(logloss(P["class + confidence"][m], ne[m], ke[m])
                    - logloss(P["+ observability"][m], ne[m], ke[m]))
        ne, ke, P = fit(dd, tt, None)
        m = ne > 0
        d_leak.append(logloss(P["+ observability"][m], ne[m], ke[m])
                      - logloss(P["+ mask_camera"][m], ne[m], ke[m]))
    for nm, arr in (("observability over class+confidence, INSIDE the mask", d_in),
                    ("mask_camera over observability, ALL voxels", d_leak)):
        v = np.asarray(arr); lo, hi = np.percentile(v, [2.5, 97.5])
        verdict = ("real, interval excludes zero" if lo > 0 else
                   "absent, interval excludes zero" if hi < 0 else
                   "null, interval spans zero")
        print(f"\n{nm}\n  {v.mean():+.6f} log-loss  95% CI "
              f"[{lo:+.6f}, {hi:+.6f}]  -- {verdict}")
        out.setdefault("contrasts", {})[nm] = [float(v.mean()), float(lo), float(hi)]

    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
