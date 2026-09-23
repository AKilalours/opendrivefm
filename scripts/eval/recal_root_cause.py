#!/usr/bin/env python3
"""Why does observability-conditioned recalibration keep failing? Settle it.

Three attempts have now tied or lost to the binary mask_camera flag on ECE
(A13, A27, A33) while the same measure wins on ranking. Each attempt fixed a
modelling problem -- fragmentation, then smoothing, then staleness -- and each
still failed. At some point the honest hypothesis is not "the model is wrong"
but "there is nothing left to model".

This tests that directly, and it is the test none of the three ran:

    CONDITIONAL ON THE PREDICTED CLASS AND THE MODEL'S OWN CONFIDENCE,
    does accuracy still depend on observability?

If it does, the signal exists and every previous failure was a modelling
failure worth another attempt. If it does not, then FB-OCC -- which sees the
same six cameras the measure is computed from -- has already internalised
visibility into its confidence, there is no residual to recalibrate, and the
question is CLOSED rather than open. Either answer ends it.

Method: exact sufficient statistics, no sampling. Every voxel inside
mask_camera is binned by (predicted class, confidence bin, observability bin)
and only counts are kept, so the whole validation split fits in a few hundred
kilobytes and every number below is computed on all of it.

    baseline   p(correct | class, confidence)          -- the no-observability model
    candidate  p(correct | class, confidence, obs)     -- adds the measure

Compared on held-out scenes by log-loss and ECE, and separately by the raw
within-cell accuracy spread, which needs no model at all.
"""
from __future__ import annotations
import argparse, glob, json, os, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FREE = 17
NC, NCONF, NOBS, NM = 18, 20, 10, 2   # class, confidence, observability, mask_camera
# NM is why this file was rewritten once. The first version filtered to voxels
# INSIDE mask_camera, which makes mask_camera CONSTANT in the sample -- so the
# binary baseline the previous three attempts lost to could not even be
# expressed, and "observability helps" would have been measured against a
# baseline that was missing its competitor. A13 recorded exactly this trap
# ("--scope mask made the mask grouping constant") and it was walked into again.
# All voxels are kept now and mask_camera is a feature.
_G = {}


def _init(obs_dir):
    _G["obs"] = obs_dir
    _G["gt"] = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}


def _one(row):
    tok, scene = row
    g = np.load(_G["gt"][tok])
    gc = g["semantics"].astype(np.int16).ravel()
    mk = g["mask_camera"].astype(np.int64).ravel()
    p = np.load(os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz"))
    pc = p["cls"].astype(np.int16).ravel()
    cf = p["conf"].astype(np.float32).ravel() / 255.0
    ob = np.load(os.path.join(ROOT, _G["obs"], tok + ".npy")).ravel().astype(np.float32) / 255.0
    ok = (pc == gc).astype(np.int64)

    ci = np.clip((cf * NCONF).astype(np.int64), 0, NCONF - 1)
    oi = np.clip((ob * NOBS).astype(np.int64), 0, NOBS - 1)
    flat = ((pc.astype(np.int64) * NCONF + ci) * NOBS + oi) * NM + mk
    SZ = NC * NCONF * NOBS * NM
    n = np.bincount(flat, minlength=SZ)
    k = np.bincount(flat, weights=ok, minlength=SZ)
    s = np.bincount(flat, weights=cf, minlength=SZ)
    return scene, n.astype(np.int64), k.astype(np.int64), s.astype(np.float64)


def cmd_build(a):
    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    gt = {os.path.basename(os.path.dirname(q)) for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}
    rows = [(r["token"], r["scene"]) for r in index if r["token"] in gt
            and os.path.exists(os.path.join(ROOT, "data/preds/preds_voxel",
                                            r["token"] + ".npz"))
            and os.path.exists(os.path.join(ROOT, a.obs, r["token"] + ".npy"))]
    store = os.path.join(ROOT, a.store)
    os.makedirs(os.path.dirname(store), exist_ok=True)
    scenes, N, K, S, done = [], None, None, None, set()
    if os.path.exists(store):
        z = np.load(store, allow_pickle=True)
        scenes = list(z["scenes"]); N, K, S = z["n"], z["k"], z["s"]
        done = set(z["done"].tolist())
    todo = [r for r in rows if r[0] not in done][:a.chunk]
    print(f"{len(rows)} frames | stored {len(done)} | this run {len(todo)} | "
          f"{os.cpu_count()} cores", flush=True)
    if not todo:
        print("build complete -- run analyse"); return
    sidx = {s: i for i, s in enumerate(scenes)}
    if N is None:
        SZ = NC * NCONF * NOBS * NM
        N = np.zeros((0, SZ), np.int64)
        K = np.zeros((0, SZ), np.int64)
        S = np.zeros((0, SZ), np.float64)
    t0 = time.time()
    with Pool(max(os.cpu_count() - 1, 1), initializer=_init,
              initargs=(a.obs,)) as pool:
        for i, (scene, n, k, s) in enumerate(pool.imap_unordered(_one, todo, chunksize=4)):
            if scene not in sidx:
                sidx[scene] = len(scenes); scenes.append(scene)
                N = np.concatenate([N, np.zeros((1, N.shape[1]), np.int64)])
                K = np.concatenate([K, np.zeros((1, K.shape[1]), np.int64)])
                S = np.concatenate([S, np.zeros((1, S.shape[1]), np.float64)])
            j = sidx[scene]
            N[j] += n; K[j] += k; S[j] += s
            if (i + 1) % 400 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(todo)}  {el/(i+1):.3f}s/frame  "
                      f"eta {(len(todo)-i-1)*el/(i+1)/60:.1f} min", flush=True)
    done |= {r[0] for r in todo}
    np.savez_compressed(store, scenes=np.array(scenes), n=N, k=K, s=S,
                        done=np.array(sorted(done)))
    left = len(rows) - len(done)
    print(f"chunk done, {left} remain" if left > 0 else
          "build complete -- run analyse", flush=True)


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


def cmd_analyse(a):
    z = np.load(os.path.join(ROOT, a.store), allow_pickle=True)
    scenes = list(z["scenes"]); N, K, S = z["n"], z["k"], z["s"]
    shape = (NC, NCONF, NOBS, NM)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(scenes))
    dev, tst = perm[:len(scenes) // 2], perm[len(scenes) // 2:]

    def fit_predict(train_idx, eval_idx):
        nt = N[train_idx].sum(0).reshape(shape)
        kt = K[train_idx].sum(0).reshape(shape)
        ne = N[eval_idx].sum(0).reshape(shape)
        ke = K[eval_idx].sum(0).reshape(shape)
        cls_n = nt.sum((1, 2, 3), keepdims=True)
        cls_k = kt.sum((1, 2, 3), keepdims=True)
        prior = (cls_k + 1) / (cls_n + 2)

        def pool(axes):
            """Accuracy pooled over `axes`, Laplace-smoothed toward the class rate."""
            n_ = nt.sum(axes, keepdims=True); k_ = kt.sum(axes, keepdims=True)
            return np.broadcast_to((k_ + 20 * prior) / (n_ + 20), shape)

        A = pool((2, 3))            # class, confidence            -- the baseline
        B = pool((2,))              # + mask_camera                -- the binary flag
        C = pool((3,))              # + observability              -- the measure
        D = (kt + 20 * A) / (nt + 20)   # + both
        return ne, ke, {"class + confidence": A,
                        "+ mask_camera": B,
                        "+ observability": C,
                        "+ both": D}

    ne, ke, P = fit_predict(dev, tst)
    m = ne > 0
    print(f"\n{len(scenes)} scenes, {int(N.sum()):,} voxels (ALL, not just "
          f"inside mask_camera -- see the note at the top of this file)")
    print(f"held out {len(tst)} scenes, {int(ne.sum()):,} voxels")
    print("=" * 74)
    print(f"{'model for p(correct)':<34}{'log-loss':>13}{'ECE':>13}{'vs baseline':>14}")
    print("-" * 74)
    res = {}
    ll0 = logloss(P["class + confidence"][m], ne[m], ke[m])
    for nm, p_ in P.items():
        ll = logloss(p_[m], ne[m], ke[m]); ec = ece(p_[m], ne[m], ke[m])
        res[nm] = dict(logloss=ll, ece=ec)
        d = "" if nm == "class + confidence" else f"{100*(ll0-ll)/ll0:+.2f}%"
        print(f"{nm:<34}{ll:>13.6f}{ec:>13.6f}{d:>14}")
    print("=" * 74)

    boot = {k: [] for k in ("obs_vs_base", "obs_vs_mask", "both_vs_mask")}
    for _ in range(a.boot):
        d = rng.choice(dev, len(dev), replace=True)
        t = rng.choice(tst, len(tst), replace=True)
        n2, k2, P2 = fit_predict(d, t)
        mm = n2 > 0
        L = {kk: logloss(vv[mm], n2[mm], k2[mm]) for kk, vv in P2.items()}
        boot["obs_vs_base"].append(L["class + confidence"] - L["+ observability"])
        boot["obs_vs_mask"].append(L["+ mask_camera"] - L["+ observability"])
        boot["both_vs_mask"].append(L["+ mask_camera"] - L["+ both"])
    print("\nlog-loss reduction, scene bootstrap (positive favours the first named)")
    for kk, lbl in (("obs_vs_base", "observability vs class+confidence"),
                    ("obs_vs_mask", "observability vs mask_camera"),
                    ("both_vs_mask", "both vs mask_camera")):
        v = np.array(boot[kk]); lo, hi = np.percentile(v, [2.5, 97.5])
        verdict = "WINS" if lo > 0 else ("LOSES" if hi < 0 else "tie")
        print(f"  {lbl:<38}{v.mean():+.6f}  [{lo:+.6f}, {hi:+.6f}]  {verdict}")
        res[kk] = dict(delta=float(v.mean()), ci=[float(lo), float(hi)],
                       verdict=verdict)
    ll_b = res["class + confidence"]["logloss"]; ll_c = res["+ observability"]["logloss"]
    ec_b = res["class + confidence"]["ece"]; ec_c = res["+ observability"]["ece"]
    lo, hi = np.percentile(boot["obs_vs_base"], [2.5, 97.5])

    # model-free view: inside a fixed (class, confidence) cell, how much does
    # accuracy move across observability? This needs no fitting at all.
    nt = N.sum(0).reshape(shape).sum(3); kt = K.sum(0).reshape(shape).sum(3)
    rows = []
    for c in range(NC):
        for q in range(NCONF):
            n_, k_ = nt[c, q], kt[c, q]
            live = n_ >= a.min_cell
            if live.sum() < 4:
                continue
            acc = k_[live] / n_[live]
            w = n_[live]
            o = (np.arange(NOBS)[live] + 0.5) / NOBS
            slope = np.polyfit(o, acc, 1, w=w)[0]
            rows.append((n_[live].sum(), c, q, acc.max() - acc.min(), slope))
    rows.sort(reverse=True)
    W = np.array([r[0] for r in rows], float)
    spread = np.array([r[3] for r in rows])
    slope = np.array([r[4] for r in rows])
    print(f"\nmodel-free: inside a fixed (class, confidence) cell, "
          f"{len(rows)} cells with >= {a.min_cell} voxels in >= 4 observability bins")
    print(f"  weighted mean accuracy spread across observability: "
          f"{(W*spread).sum()/W.sum():.4f}")
    print(f"  weighted mean slope (accuracy per unit observability): "
          f"{(W*slope).sum()/W.sum():+.4f}")
    print(f"  cells where accuracy RISES with observability: "
          f"{100*(W*(slope>0)).sum()/W.sum():.1f}% by voxel weight")

    json.dump(dict(scenes=len(scenes), voxels=int(N.sum()),
                   models=res, logloss_gain=ll_b - ll_c,
                   ci=[float(lo), float(hi)],
                   mean_spread=float((W * spread).sum() / W.sum()),
                   mean_slope=float((W * slope).sum() / W.sum()),
                   frac_positive_slope=float((W * (slope > 0)).sum() / W.sum())),
              open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("\nwrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--obs", default="data/pack/obs_max")
    b.add_argument("--chunk", type=int, default=100000)
    b.add_argument("--store", default="outputs/artifacts/recal_rc_store.npz")
    b.set_defaults(fn=cmd_build)
    an = sub.add_parser("analyse")
    an.add_argument("--store", default="outputs/artifacts/recal_rc_store.npz")
    an.add_argument("--boot", type=int, default=400)
    an.add_argument("--seed", type=int, default=9)
    an.add_argument("--min-cell", type=int, default=5000)
    an.add_argument("--out", default="outputs/artifacts/recal_root_cause.json")
    an.set_defaults(fn=cmd_analyse)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
