#!/usr/bin/env python3
"""Why is the calibration decline a HUMP? Root cause, not a rewrite of H1.

H1 was pre-registered as: the gap between confidence and accuracy decreases
monotonically as observability rises. It does not. The deciles run

    .0440 .0561 .0602 .0582 .0583 .0549 .0517 .0461 .0338

which rises, peaks at decile 3 and then falls. H1 is recorded as FAILED and
stays failed. This script asks a different question: WHY.

The suspect is already in the record. %GTfree falls from 91.7% in decile 1 to
85.2% in decile 9, and is only 60.3% at obs = 0. Free space is easy -- high
confidence, high accuracy, small gap. Occupied space is hard. So two effects
run in opposite directions as observability rises:

    composition   fewer free voxels    ->  pushes the gap UP
    visibility    more camera evidence ->  pushes the gap DOWN

A sum of one rising and one falling curve is a hump. If that is the mechanism,
then WITHIN each stratum -- free only, occupied only -- the decline should be
monotone, and the aggregate hump is Simpson's paradox.

This is a falsifiable prediction stated before the numbers are printed. If the
strata are also humped, the composition story is wrong and the mechanism is
something else.

No marching: reads the stored maps.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FREE = 17


def spearman(y):
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float)
    rx = x - x.mean(); ry = y - y.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--chunk", type=int, default=380)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--obs", default="data/pack/obs_ray")
    ap.add_argument("--cache", default="outputs/artifacts/h1_cache.npz")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default="outputs/artifacts/root_cause_h1.json")
    a = ap.parse_args()
    cache = os.path.join(ROOT, a.cache)

    NB, NS = 11, 3          # bins: 0 = obs zero, 1..10 = deciles; strata all/free/occ
    # accumulators per scene: [scene, stratum, bin] -> n, sum_conf, n_correct
    if a.report:
        z = np.load(cache, allow_pickle=True)
        ACC, scenes, seen = z["acc"], list(z["scenes"]), set(z["seen"].tolist())
    else:
        index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
        gt_map = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
            os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                         "*", "*", "labels.npz"))}
        rng = np.random.default_rng(a.seed)
        rows = [r for r in index if r["token"] in gt_map
                and os.path.exists(os.path.join(ROOT, a.obs, r["token"] + ".npy"))
                and os.path.exists(os.path.join(ROOT, "data/preds/preds_voxel",
                                                r["token"] + ".npz"))]
        rows = [rows[i] for i in sorted(rng.choice(len(rows),
                min(a.frames, len(rows)), replace=False))]
        scenes, seen, ACC = [], set(), None
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            ACC, scenes, seen = z["acc"], list(z["scenes"]), set(z["seen"].tolist())
        sidx = {s: i for i, s in enumerate(scenes)}
        todo = [r for r in rows if r["token"] not in seen][:a.chunk]
        print(f"sample {len(rows)} | cached {len(seen)} | this run {len(todo)}",
              flush=True)
        # fixed decile edges, computed once on the first chunk and then frozen
        efile = os.path.join(ROOT, "outputs/artifacts/h1_edges.npy")
        if os.path.exists(efile):
            edges = np.load(efile)
        else:
            s = []
            for r in todo[:60]:
                o = np.load(os.path.join(ROOT, a.obs, r["token"] + ".npy")).ravel()
                s.append(o[o > 0][::37])
            s = np.concatenate(s)
            edges = np.quantile(s, np.arange(1, 10) / 10.0)
            np.save(efile, edges)
            print("frozen decile edges:", np.round(edges / 255, 4), flush=True)

        for r in todo:
            tok, sc = r["token"], r["scene"]
            if sc not in sidx:
                sidx[sc] = len(scenes); scenes.append(sc)
                ACC = np.zeros((len(scenes), NS, NB, 3)) if ACC is None else \
                    np.concatenate([ACC, np.zeros((1, NS, NB, 3))], 0)
            if ACC is None:
                ACC = np.zeros((len(scenes), NS, NB, 3))
            g = np.load(gt_map[tok])
            gcls = g["semantics"].astype(np.int16).ravel()
            m = g["mask_camera"].astype(bool).ravel()
            p = np.load(os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz"))
            pc = p["cls"].astype(np.int16).ravel()
            cf = p["conf"].astype(np.float32).ravel() / 255.0
            ob = np.load(os.path.join(ROOT, a.obs, tok + ".npy")).ravel()

            b = np.where(ob == 0, 0, np.searchsorted(edges, ob, "right") + 1)
            corr = (pc == gcls).astype(np.float32)
            isfree = (gcls == FREE)
            i = sidx[sc]
            for st, sel in ((0, m), (1, m & isfree), (2, m & ~isfree)):
                bb = b[sel]
                ACC[i, st, :, 0] += np.bincount(bb, minlength=NB)
                ACC[i, st, :, 1] += np.bincount(bb, weights=cf[sel], minlength=NB)
                ACC[i, st, :, 2] += np.bincount(bb, weights=corr[sel], minlength=NB)
            seen.add(tok)
        np.savez_compressed(cache, acc=ACC, scenes=np.array(scenes),
                            seen=np.array(sorted(seen)))
        left = len(rows) - len(seen)
        print(f"chunk done, {left} remain" if left > 0 else
              "sample complete -- run with --report", flush=True)
        return

    T = ACC.sum(0)
    # decile 9 is EMPTY: TRUST = 0.795 caps the measure, so a large mass sits at
    # exactly 0.7961 and two quantile edges coincide. An empty bin contributes a
    # spurious 0.0 to any monotonicity statistic, so it is dropped explicitly
    # rather than silently averaged in.
    live = T[0, :, 0] > 0
    gap = lambda A: (A[..., 1] - A[..., 2]) / np.maximum(A[..., 0], 1)
    G = gap(T)
    names = ["ALL voxels (what H1 was scored on)", "GT FREE only", "GT OCCUPIED only"]
    lbl = ["obs = 0"] + [f"decile {i}" for i in range(1, 11)]

    print(f"\n{len(ACC)} scenes, {int(T[0,:,0].sum()):,} voxels inside mask_camera")
    print("=" * 84)
    hdr = f"{'bin':<11}" + "".join(f"{n.split(' ')[0]:>13}" for n in names) + \
          f"{'%free':>9}{'voxels':>14}"
    print(hdr); print("-" * 84)
    pf = T[1, :, 0] / np.maximum(T[0, :, 0], 1)
    for k in range(11):
        if not live[k]:
            print(f"{lbl[k]:<11}{'(empty - TRUST saturation collapses two edges)':>52}")
            continue
        print(f"{lbl[k]:<11}" + "".join(f"{G[st, k]:>13.4f}" for st in range(3))
              + f"{100*pf[k]:>8.1f}%{int(T[0,k,0]):>14,}")
    print("=" * 84)

    rng = np.random.default_rng(3)
    res = {}
    print("\nmonotonicity across deciles 1..10  (rank correlation with decile; "
          "H1 predicts a clear negative)")
    for st in range(3):
        keep = live[1:]
        v = G[st, 1:][keep]
        r = spearman(v)
        bs = []
        for _ in range(a.boot):
            S = ACC[rng.integers(0, len(ACC), len(ACC))].sum(0)
            bs.append(spearman(gap(S)[st, 1:][keep]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        mono = "MONOTONE DECREASING" if hi < 0 else (
               "monotone increasing" if lo > 0 else "not monotone")
        res[names[st]] = dict(gaps=[float(x) for x in G[st]], rho=r,
                              ci=[float(lo), float(hi)], verdict=mono)
        print(f"  {names[st]:<36} rho {r:>+7.3f}  [{lo:+.3f}, {hi:+.3f}]   {mono}")

    json.dump(dict(scenes=len(ACC), pct_free=[float(x) for x in pf],
                   counts=[int(x) for x in T[0, :, 0]], strata=res),
              open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
