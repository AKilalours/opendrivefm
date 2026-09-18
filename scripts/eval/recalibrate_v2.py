#!/usr/bin/env python3
"""Reopening the recalibration failure, with a variable that did not exist then.

A13 closed this: conditioning recalibration on observability lost to a single
global map (fragmentation), and the smooth four-parameter replacement only TIED
the binary mask_camera baseline. The ECE target was never met and Stage 5 was
marked closed.

It is reopened for one specific reason, not out of optimism. A25 showed the
blind set is TWO populations: a cell the cameras cannot see now but saw 0.5 s
ago has a 13.63% error rate, a cell no camera has seen in 4 s has 23.83%
(+9.20 pts [+8.08, +10.34]). Every recalibration tried so far was blind to that
split, because age did not exist as a feature. Observability alone cannot
separate those two groups -- it assigns both exactly 0.

Schemes, all Platt-style logistic maps fitted on l = logit(confidence):

    A  none          raw model confidence
    B  global        l                            one map for everything
    C  mask_camera   l, m, l*m                    the binary baseline to beat
    D  observability l, o, l*o                    A13's best, which only tied C
    E  + staleness   l, o, l*o, a, l*a            NEW

Fitted on a dev half of the SCENES, scored on the held-out half. Scene-disjoint,
because voxels within a frame are anything but independent.

Acceptance, fixed here before the numbers are printed, and written to require
beating the do-nothing control -- the rule A13 had to learn twice:

    E must beat A, B and C on held-out ECE, or it has not worked.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def logit(p, e=1e-4):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def irls(X, y, w=None, iters=60):
    """Weighted logistic regression, Newton steps, ridge-stabilised."""
    n, d = X.shape
    w = np.ones(n) if w is None else w
    b = np.zeros(d)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ b))
        g = X.T @ (w * (y - p))
        s = w * p * (1 - p) + 1e-9
        H = (X * s[:, None]).T @ X + 1e-6 * np.eye(d)
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def ece(p, y, nb=15):
    edges = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, nb - 1)
    n = np.bincount(idx, minlength=nb).astype(float)
    sp = np.bincount(idx, weights=p, minlength=nb)
    sy = np.bincount(idx, weights=y.astype(float), minlength=nb)
    m = n > 0
    return float((n[m] / n.sum() * np.abs(sp[m] / n[m] - sy[m] / n[m])).sum())


def brier(p, y):
    return float(np.mean((p - y.astype(float)) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/artifacts/temporal_cache.npz")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs/artifacts/recalibrate_v2.json")
    a = ap.parse_args()

    z = np.load(os.path.join(ROOT, a.cache), allow_pickle=True)
    toks = list(z["tok"])
    index = {r["token"]: r["scene"] for r in
             json.load(open(os.path.join(ROOT, "data/pack/index.json")))}
    cat = lambda k: np.concatenate(list(z[k]))
    age, obs, conf, wrong = cat("age"), cat("obs"), cat("conf"), cat("wrong")
    fidx = np.concatenate([np.full(len(v), i) for i, v in enumerate(z["age"])])
    # integer scene codes: np.isin over 1.7M PYTHON STRINGS inside a 2000-draw
    # bootstrap is minutes of pure string comparison. Codes make it milliseconds.
    names = sorted({index[str(t)] for t in toks})
    code = {n: i for i, n in enumerate(names)}
    scenes_of_frame = np.array([code[index[str(t)]] for t in toks], np.int32)
    scn = scenes_of_frame[fidx]
    y = (~wrong).astype(np.float64)              # 1 = the model was right

    uscn = np.arange(len(names), dtype=np.int32)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(uscn))
    dev_s = uscn[perm[: len(uscn) // 2]]
    ismem = np.zeros(len(names), bool); ismem[dev_s] = True
    dev = ismem[scn]
    tst = ~dev
    print(f"{len(toks)} frames, {len(uscn)} scenes -> "
          f"{len(dev_s)} dev / {len(uscn)-len(dev_s)} test", flush=True)
    print(f"dev {dev.sum():,} voxels   test {tst.sum():,} voxels")

    NEVER = age.max()
    l = logit(conf)
    o = obs.astype(np.float64)
    m = (obs > 0).astype(np.float64)             # the binary mask analogue
    an = np.clip(age / max(NEVER, 1e-6), 0, 1)   # 0 = seen now, 1 = never seen

    one = np.ones_like(l)
    feats = {
        "B  global":              np.stack([one, l], 1),
        "C  mask_camera":         np.stack([one, l, m, l * m], 1),
        "D  observability":       np.stack([one, l, o, l * o], 1),
        "E  + staleness":         np.stack([one, l, o, l * o, an, l * an], 1),
    }

    P = {"A  none": conf.astype(np.float64)}
    coef = {}
    for k, X in feats.items():
        b = irls(X[dev], y[dev])
        coef[k] = [float(v) for v in b]
        P[k] = 1.0 / (1.0 + np.exp(-(X @ b)))

    blind = obs <= 0.0
    order = ["A  none", "B  global", "C  mask_camera", "D  observability",
             "E  + staleness"]

    print("\nHELD-OUT scenes only")
    print("=" * 86)
    print(f"{'scheme':<20}{'ECE':>10}{'vs C':>10}{'Brier':>11}"
          f"{'ECE | blind':>14}{'ECE | seen':>13}")
    print("-" * 86)
    res = {}
    eC = ece(P["C  mask_camera"][tst], y[tst])
    for k in order:
        e_all = ece(P[k][tst], y[tst])
        e_bl = ece(P[k][tst & blind], y[tst & blind])
        e_se = ece(P[k][tst & ~blind], y[tst & ~blind])
        res[k] = dict(ece=e_all, brier=brier(P[k][tst], y[tst]),
                      ece_blind=e_bl, ece_seen=e_se,
                      coef=coef.get(k))
        print(f"{k:<20}{e_all:>10.5f}{100*(e_all-eC)/eC:>9.1f}%"
              f"{brier(P[k][tst], y[tst]):>11.5f}{e_bl:>14.5f}{e_se:>13.5f}")
    print("=" * 86)

    # paired scene bootstrap on the contrast that decides it
    us_t = np.sort(np.unique(scn[tst]))
    def boot(kA, kB, mask):
        d = []
        for _ in range(a.boot):
            cnt = np.bincount(rng.choice(us_t, len(us_t), replace=True),
                              minlength=len(names))
            sel = (cnt[scn] > 0) & mask
            if sel.sum() < 5000:
                continue
            d.append(ece(P[kA][sel], y[sel]) - ece(P[kB][sel], y[sel]))
        return np.percentile(d, [2.5, 97.5]) if d else (np.nan, np.nan)

    print("\nacceptance: E must beat A, B and C on held-out ECE")
    ok = True
    for base in ("A  none", "B  global", "C  mask_camera", "D  observability"):
        d = res["E  + staleness"]["ece"] - res[base]["ece"]
        lo, hi = boot("E  + staleness", base, tst)
        verdict = "E WINS" if hi < 0 else ("E LOSES" if lo > 0 else "tie")
        if base != "D  observability" and hi >= 0:
            ok = False
        print(f"  E vs {base:<18}{d:>+10.5f}  [{lo:+.5f}, {hi:+.5f}]   {verdict}")
        res[f"E_vs_{base.split()[0]}"] = dict(delta=float(d),
                                              ci=[float(lo), float(hi)],
                                              verdict=verdict)
    lo, hi = boot("E  + staleness", "D  observability", tst & blind)
    dbl = res["E  + staleness"]["ece_blind"] - res["D  observability"]["ece_blind"]
    print(f"  E vs D, BLIND SET only   {dbl:>+10.5f}  [{lo:+.5f}, {hi:+.5f}]")
    res["E_vs_D_blind"] = dict(delta=float(dbl), ci=[float(lo), float(hi)])

    print("\nACCEPTANCE MET" if ok else
          "\nACCEPTANCE NOT MET -- recorded as a failure, not reframed")
    res["acceptance_met"] = bool(ok)
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
