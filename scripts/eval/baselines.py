#!/usr/bin/env python3
"""A46. Uncertainty baselines, and a combined score, on held-out scenes.

Every baseline this project had measured against was a dataset artifact or a
trivial feature. This adds the post-hoc uncertainty family, which needs no
re-inference and no GPU because conf, conf2 and p_free are already stored per
voxel, and then asks the obvious question four amendments imply and none of
them measured: what does the BEST available error predictor look like when
sensor geometry and model confidence are combined?

Protocol, fixed in A46 before this ran. Coefficients are fit on the 75 DEV
scenes and every number reported is scored on the 75 TEST scenes, under the
split materialised in A45. Scoring is inside mask_camera, A5's scope.

Temperature scaling is deliberately absent from the AUROC table. It is a
monotone transform of confidence, so it cannot change a ranking or an AUROC.
It appears in the calibration table, where it is the correct baseline.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import split as SP                                       # noqa: E402

FREE, RES, RNG, NZ, Z0, N = 17, 0.4, 40.0, 16, -1.0, 200
NB = 256
SCORES = ["MSP", "margin", "p_free", "range", "mask_camera",
          "observability", "COMBINED"]
_G = {}


def _range_grid():
    ax = (np.arange(N) + 0.5) * RES - RNG
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2)
    return np.broadcast_to(r[:, :, None], (N, N, NZ)).astype(np.float32)


RGRID = _range_grid()
RMAX = float(RGRID.max())


def _init(obs_dir, coef):
    _G["obs"] = obs_dir
    _G["coef"] = coef
    _G["gt"] = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}


def _feats(tok):
    g = np.load(_G["gt"][tok])
    gt = g["semantics"].astype(np.int16)
    mk = g["mask_camera"].astype(bool)
    p = np.load(os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz"))
    cls = p["cls"].astype(np.int16)
    conf = p["conf"].astype(np.float32) / 255.0
    conf2 = p["conf2"].astype(np.float32) / 255.0
    pfree = p["p_free"].astype(np.float32) / 255.0
    obs = np.load(os.path.join(ROOT, _G["obs"],
                               tok + ".npy")).astype(np.float32) / 255.0
    wrong = (cls != gt)[mk]
    c, m = conf[mk], (conf[mk] - conf2[mk])
    return dict(wrong=wrong, conf=c, margin=m, pfree=pfree[mk],
                rng=RGRID[mk] / RMAX, obs=obs[mk], mask=mk[mk].astype(np.float32))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _design(f):
    """Features for the combined score. Fit on dev, applied unchanged on test."""
    return np.stack([_logit(f["conf"]), f["margin"], f["obs"], f["rng"]], 1)


def _combined(f, coef):
    if coef is None:
        return np.zeros_like(f["conf"])
    X = _design(f)
    return 1.0 / (1.0 + np.exp(-(X @ coef[:-1] + coef[-1])))


def _score_vec(f, name, coef):
    """Every score oriented so HIGHER means MORE LIKELY WRONG."""
    if name == "MSP":            return 1.0 - f["conf"]
    if name == "margin":         return 1.0 - f["margin"]
    if name == "p_free":         return 1.0 - np.abs(2 * f["pfree"] - 1.0)
    if name == "range":          return f["rng"]
    if name == "mask_camera":    return 1.0 - f["mask"]
    if name == "observability":  return 1.0 - f["obs"]
    if name == "COMBINED":       return _combined(f, coef)
    raise KeyError(name)


def _hist(tok):
    f = _feats(tok)
    w = f["wrong"]
    out = {}
    for nm in SCORES:
        s = _score_vec(f, nm, _G["coef"])
        q = np.clip(np.rint(s * (NB - 1)), 0, NB - 1).astype(np.uint8)
        out[nm] = (np.bincount(q[w], minlength=NB).astype(np.int64),
                   np.bincount(q[~w], minlength=NB).astype(np.int64))
    return out


def _sample(tok):
    f = _feats(tok)
    rs = np.random.default_rng(abs(hash(tok)) % (2 ** 31))
    n = f["wrong"].size
    k = min(40000, n)
    idx = rs.choice(n, k, replace=False)
    return (_design({kk: vv[idx] for kk, vv in f.items()}),
            f["wrong"][idx].astype(np.float64))


def auroc(pos, neg):
    p, n = pos.astype(np.float64), neg.astype(np.float64)
    cn = np.cumsum(n) - n
    P, Nn = p.sum(), n.sum()
    if P == 0 or Nn == 0:
        return float("nan")
    return float((p * (cn + n / 2.0)).sum() / (P * Nn))


def fit_logistic(X, y, iters=300, lr=0.5):
    """Plain gradient descent. No sklearn on this machine, and a four-feature
    logistic does not need one."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = np.zeros(Xs.shape[1]); b = 0.0
    for _ in range(iters):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (Xs.T @ g) / len(y)
        b -= lr * g.mean()
    return np.concatenate([w / sd, [b - float((mu / sd) @ w)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="data/pack/obs_max")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--fit-frames", type=int, default=400)
    ap.add_argument("--out", default="outputs/artifacts/baselines.json")
    # The dev fit is ~100 s and the shell here caps at 180 s, so the fitted
    # coefficients are cached. They are a DEV-only quantity; caching them
    # changes nothing about the protocol and makes the scoring pass rerunnable.
    ap.add_argument("--coef", default="outputs/artifacts/baselines_coef.json")
    a = ap.parse_args()

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    gt = {os.path.basename(os.path.dirname(q)) for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}
    have = lambda r: (r["token"] in gt and os.path.exists(os.path.join(
        ROOT, "data/preds/preds_voxel", r["token"] + ".npz")) and
        os.path.exists(os.path.join(ROOT, a.obs, r["token"] + ".npy")))
    dev_s, test_s = SP.dev_scenes(), SP.test_scenes()
    dev = [r["token"] for r in index if have(r) and r["scene"] in dev_s]
    test = [(r["token"], r["scene"]) for r in index if have(r) and r["scene"] in test_s]
    print(f"[A46] split digest {SP.digest()} | dev {len(dev)} frames | "
          f"test {len(test)} frames / {len({s for _, s in test})} scenes", flush=True)

    # ---- fit the combined score on DEV ONLY -----------------------------
    coef_path = os.path.join(ROOT, a.coef)
    if os.path.exists(coef_path):
        cj = json.load(open(coef_path))
        coef = np.asarray(cj["coef"], np.float64)
        fit_toks = cj["fit_frames"] * [None]
        print(f"loaded dev-fit coefficients from {a.coef} "
              f"({cj['fit_frames']} dev frames, {cj['voxels']:,} voxels)")
        print("  coefficients  logit(conf) %+.4f   margin %+.4f   obs %+.4f   "
              "range %+.4f   bias %+.4f" % tuple(coef))
        return _score(a, coef, test, index)
    rs = np.random.default_rng(46)
    fit_toks = list(rs.choice(dev, min(a.fit_frames, len(dev)), replace=False))
    t0 = time.time()
    Xs, ys = [], []
    with Pool(max(os.cpu_count() - 1, 1), initializer=_init,
              initargs=(a.obs, None)) as pool:
        for X, y in pool.imap_unordered(_sample, fit_toks, chunksize=4):
            Xs.append(X); ys.append(y)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    coef = fit_logistic(X, y)
    print(f"fit on {len(fit_toks)} dev frames, {len(y):,} sampled voxels, "
          f"{time.time()-t0:.0f}s")
    print("  coefficients  logit(conf) %+.4f   margin %+.4f   obs %+.4f   "
          "range %+.4f   bias %+.4f" % tuple(coef))
    json.dump(dict(coef=[float(c) for c in coef], fit_frames=len(fit_toks),
                   voxels=int(len(y)), split_digest=SP.digest()),
              open(coef_path, "w"), indent=1)
    return _score(a, coef, test, index)


def _score(a, coef, test, index):
    # ---- score on TEST ---------------------------------------------------
    scenes = sorted({s for _, s in test})
    sidx = {s: i for i, s in enumerate(scenes)}
    H = {nm: np.zeros((len(scenes), 2, NB), np.int64) for nm in SCORES}
    t0 = time.time()
    with Pool(max(os.cpu_count() - 1, 1), initializer=_init,
              initargs=(a.obs, coef)) as pool:
        for n, (out, (tok, sc)) in enumerate(
                zip(pool.imap(_hist, [t for t, _ in test], chunksize=4), test)):
            j = sidx[sc]
            for nm in SCORES:
                H[nm][j, 0] += out[nm][0]; H[nm][j, 1] += out[nm][1]
            if (n + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  {n+1}/{len(test)}  {el/(n+1):.3f}s/frame", flush=True)

    pt = {nm: auroc(H[nm].sum(0)[0], H[nm].sum(0)[1]) for nm in SCORES}
    rng2 = np.random.default_rng(46)
    idx = [rng2.choice(len(scenes), len(scenes), replace=True)
           for _ in range(a.boot)]
    print("\n" + "=" * 76)
    print("PREDICTING PER-VOXEL ERROR -- 75 HELD-OUT TEST SCENES, inside mask_camera")
    print("-" * 76)
    print(f"{'score':<26}{'AUROC':>9}   {'vs observability':>26}")
    for nm in sorted(SCORES, key=lambda k: -pt[k]):
        if nm == "observability":
            print(f"{nm:<26}{pt[nm]:>9.4f}   {'(reference)':>26}")
            continue
        d = np.array([auroc(H[nm][i].sum(0)[0], H[nm][i].sum(0)[1])
                      - auroc(H["observability"][i].sum(0)[0],
                              H["observability"][i].sum(0)[1]) for i in idx])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{nm:<26}{pt[nm]:>9.4f}   {d.mean():+8.4f} [{lo:+.4f},{hi:+.4f}]")
    print("=" * 76)

    res = {"split_digest": SP.digest(), "test_frames": len(test),
           "test_scenes": len(scenes),
           "coef": [float(c) for c in coef], "auroc": pt, "contrasts": {}}
    for base, hyp in (("observability", "H5"), ("MSP", "H6")):
        d = np.array([auroc(H["COMBINED"][i].sum(0)[0], H["COMBINED"][i].sum(0)[1])
                      - auroc(H[base][i].sum(0)[0], H[base][i].sum(0)[1])
                      for i in idx])
        lo, hi = np.percentile(d, [2.5, 97.5])
        holds = lo > 0
        print(f"\n{hyp}: COMBINED - {base} = {d.mean():+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  -> {'HOLDS' if holds else 'FAILS'}")
        res["contrasts"][f"combined_minus_{base}"] = [float(d.mean()),
                                                      float(lo), float(hi)]
        res[f"{hyp}_holds"] = bool(holds)
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
