#!/usr/bin/env python3
"""How long since a camera last saw this cell? -- and does staleness predict error?

Every observability number in this project so far is PRESENT TENSE: what the six
cameras can see in this frame. But FB-OCC r50 runs with `do_history=True` over a
16-frame temporal window, so a voxel that is occluded right now may have been in
plain view two seconds ago, and the model is entitled to remember it. Present-
tense observability calls both cases blind. They are not the same case.

This adds the missing axis, camera-only and with no re-inference:

    age(v) = seconds since ANY camera last had observability > tau at the
             world location that voxel currently occupies

Past maps are warped into the current ego frame through the recorded ego poses,
so the question is asked about a place in the world, not an index in a grid.

The claim under test, stated before the numbers exist:

    H-T   among voxels the model asserts confidently, error rises with age,
          AND age carries information that present-tense observability does not.

The second half is what makes it a contribution rather than a restatement. It is
tested by AUROC of age alone, and by AUROC of age INSIDE fixed bands of present
observability -- if age only works because stale cells are also unseen cells, it
will collapse inside those bands.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FREE, RES, RNG, NZ, Z0, N = 17, 0.4, 40.0, 16, -1.0, 200
MAXH = 8                      # keyframes of history; nuScenes keyframes are 2 Hz


def qrot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], np.float64)


def pose(row):
    c = row["cams"]["CAM_FRONT"]
    return (qrot(np.asarray(c["ego2global_rotation"], np.float64)),
            np.asarray(c["ego2global_translation"], np.float64))


def centres():
    ax = (np.arange(N) + 0.5) * RES - RNG
    az = (np.arange(NZ) + 0.5) * RES + Z0
    X, Y, Z = np.meshgrid(ax, ax, az, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


def sample_prev(P_glob, Rp, tp, prev):
    """Look up a past observability map at the world points P_glob."""
    L = (P_glob - tp) @ Rp
    i = np.floor((L[:, 0] + RNG) / RES).astype(np.int32)
    j = np.floor((L[:, 1] + RNG) / RES).astype(np.int32)
    k = np.floor((L[:, 2] - Z0) / RES).astype(np.int32)
    ok = ((i >= 0) & (i < N) & (j >= 0) & (j < N) & (k >= 0) & (k < NZ))
    out = np.zeros(P_glob.shape[0], np.uint8)
    out[ok] = prev[i[ok], j[ok], k[ok]]
    return out


def auroc(score, y):
    s = np.asarray(score, np.float64)
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # average ranks over ties
    _, first, cnt = np.unique(s[o], return_index=True, return_counts=True)
    for f, c in zip(first, cnt):
        if c > 1:
            r[o[f:f + c]] = r[o[f:f + c]].mean()
    P, Nn = y.sum(), (~y).sum()
    if P == 0 or Nn == 0:
        return float("nan")
    return float((r[y].sum() - P * (P + 1) / 2) / (P * Nn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=260)
    ap.add_argument("--chunk", type=int, default=90)
    ap.add_argument("--tau", type=float, default=0.10, help="observability that counts as 'seen'")
    ap.add_argument("--conf", type=float, default=0.70)
    ap.add_argument("--sub", type=int, default=13, help="voxel subsample stride")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--cache", default="outputs/artifacts/temporal_cache.npz")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default="outputs/artifacts/temporal_observability.json")
    a = ap.parse_args()
    cache = os.path.join(ROOT, a.cache)

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    by_scene = {}
    for i, r in enumerate(index):
        by_scene.setdefault(r["scene"], []).append(i)
    for s in by_scene:
        by_scene[s].sort(key=lambda i: index[i]["timestamp"])

    if not a.report:
        gt_map = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
            os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                         "*", "*", "labels.npz"))}
        ok_tok = lambda t: (t in gt_map
            and os.path.exists(os.path.join(ROOT, "data/pack/obs_ray", t + ".npy"))
            and os.path.exists(os.path.join(ROOT, "data/preds/preds_voxel", t + ".npz")))
        cand = []
        for s, ids in by_scene.items():
            for pos, i in enumerate(ids):
                if pos >= MAXH and ok_tok(index[i]["token"]) and all(
                        ok_tok(index[j]["token"]) for j in ids[pos-MAXH:pos]):
                    cand.append((s, pos, i))
        rng = np.random.default_rng(a.seed)
        cand = [cand[i] for i in sorted(rng.choice(len(cand),
                min(a.frames, len(cand)), replace=False))]

        store = {"age": [], "obs": [], "conf": [], "wrong": [], "occ": [],
                 "scene": []}
        done = set()
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            for k in store:
                store[k] = list(z[k])
            done = set(z["tok"].tolist())
        toks = []
        if os.path.exists(cache):
            toks = list(np.load(cache, allow_pickle=True)["tok"])
        todo = [c for c in cand if index[c[2]]["token"] not in done][:a.chunk]
        print(f"sample {len(cand)} | cached {len(done)} | this run {len(todo)}",
              flush=True)
        C = centres()
        sub = np.arange(0, C.shape[0], a.sub)
        Cs = C[sub]
        for s, pos, i in todo:
            row = index[i]; tok = row["token"]
            Rc, tc = pose(row)
            Pg = Cs @ Rc.T + tc                       # current ego -> global
            age = np.full(sub.size, np.float32(MAXH * 0.5 + 0.5))
            cur = np.load(os.path.join(ROOT, "data/pack/obs_ray", tok + ".npy"))
            obs_now = cur.ravel()[sub].astype(np.float32) / 255.0
            age[obs_now > a.tau] = 0.0
            ids = by_scene[s]
            for h in range(1, MAXH + 1):
                pj = ids[pos - h]
                dt = (row["timestamp"] - index[pj]["timestamp"]) / 1e6
                Rp, tp = pose(index[pj])
                prev = np.load(os.path.join(ROOT, "data/pack/obs_ray",
                                            index[pj]["token"] + ".npy"))
                v = sample_prev(Pg, Rp, tp, prev).astype(np.float32) / 255.0
                hit = (v > a.tau) & (age > dt)
                age[hit] = dt
            g = np.load(gt_map[tok])
            gc = g["semantics"].astype(np.int16).ravel()[sub]
            m = g["mask_camera"].astype(bool).ravel()[sub]
            p = np.load(os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz"))
            pc = p["cls"].astype(np.int16).ravel()[sub]
            cf = p["conf"].astype(np.float32).ravel()[sub] / 255.0
            keep = m
            store["age"].append(age[keep].astype(np.float32))
            store["obs"].append(obs_now[keep].astype(np.float32))
            store["conf"].append(cf[keep].astype(np.float32))
            store["wrong"].append((pc != gc)[keep])
            store["occ"].append((gc != FREE)[keep])
            store["scene"].append(np.full(keep.sum(), len(toks), np.int32))
            toks.append(tok)
        np.savez_compressed(cache, tok=np.array(toks),
                            **{k: np.array(v, dtype=object) for k, v in store.items()})
        left = len(cand) - len(done) - len(todo)
        print(f"chunk done, {left} remain" if left > 0 else
              "sample complete -- run with --report", flush=True)
        return

    z = np.load(cache, allow_pickle=True)
    cat = lambda k: np.concatenate(list(z[k]))
    age, obs, conf, wrong, occ = (cat("age"), cat("obs"), cat("conf"),
                                  cat("wrong"), cat("occ"))
    toks = z["tok"]
    scn = np.concatenate([np.full(len(v), i) for i, v in enumerate(z["age"])])
    print(f"\n{len(toks)} frames, {len(age):,} sampled voxels inside mask_camera")

    NEVER = MAXH * 0.5 + 0.5
    print("=" * 88)
    print(f"{'age since last seen':<28}{'voxels':>13}{'share':>9}"
          f"{'error rate':>13}{'err | occupied':>17}")
    print("-" * 88)
    bins = [(-.01, .01, "seen right now"), (.01, .6, "0.5 s"), (.6, 1.1, "1.0 s"),
            (1.1, 2.1, "1.5 - 2.0 s"), (2.1, 4.1, "2.5 - 4.0 s"),
            (NEVER - .01, 99, f"never in {MAXH//2} s")]
    rows = []
    for lo, hi, nm in bins:
        m = (age > lo) & (age <= hi)
        if m.sum() < 500: continue
        mo = m & occ
        print(f"{nm:<28}{m.sum():>13,}{100*m.mean():>8.1f}%"
              f"{100*wrong[m].mean():>12.2f}%{100*wrong[mo].mean():>16.2f}%")
        rows.append(dict(band=nm, n=int(m.sum()), err=float(wrong[m].mean()),
                         err_occ=float(wrong[mo].mean())))
    print("=" * 88)

    print("\npredicting a wrong voxel, AUROC")
    res = {}
    for nm, sc in (("present observability (inverted)", -obs),
                   ("age since last seen", age),
                   ("model confidence (inverted)", -conf)):
        res[nm] = auroc(sc, wrong)
        print(f"  {nm:<40}{res[nm]:.4f}")

    print("\n  age INSIDE fixed bands of present observability"
          "  (does it survive holding 'seen now' constant?)")
    for lo, hi, nm in ((-.001, .001, "obs = 0      "), (.001, .25, "obs 0-0.25   "),
                       (.25, .6, "obs 0.25-0.6 "), (.6, 1.01, "obs > 0.6    ")):
        m = (obs > lo) & (obs <= hi)
        if m.sum() < 5000: continue
        v = auroc(age[m], wrong[m])
        res[f"band_{nm.strip()}"] = v
        print(f"    {nm}  n={m.sum():>9,}   AUROC {v:.4f}")

    # The bands above obs>tau are DEGENERATE by construction: age is defined as 0
    # whenever present observability exceeds tau, so it is constant there and the
    # AUROC is exactly 0.5 by definition, not by measurement. The only band where
    # age can carry independent information is the blind set. Isolate it.
    blind = obs <= 0.0
    print("\ninside the blind set (obs = 0) -- where age is the ONLY evidence axis")
    print("-" * 74)
    print(f"{'age since last seen':<26}{'voxels':>12}{'error rate':>14}{'err | occupied':>18}")
    bres = []
    for lo, hi, nm in bins:
        m = blind & (age > lo) & (age <= hi)
        if m.sum() < 500: continue
        mo = m & occ
        e_o = 100*wrong[mo].mean() if mo.sum() > 50 else float("nan")
        print(f"{nm:<26}{m.sum():>12,}{100*wrong[m].mean():>13.2f}%{e_o:>17.2f}%")
        bres.append(dict(band=nm, n=int(m.sum()), err=float(wrong[m].mean())))
    res["blind_bands"] = bres
    fresh_b = blind & (age < NEVER - .01)
    never_b = blind & (age >= NEVER - .01)
    if fresh_b.sum() > 500 and never_b.sum() > 500:
        d = wrong[never_b].mean() - wrong[fresh_b].mean()
        rg = np.random.default_rng(4); ds = []
        fr = np.unique(scn)
        for _ in range(1500):
            pk = np.isin(scn, rg.choice(fr, len(fr), replace=True))
            A, B = never_b & pk, fresh_b & pk
            if A.sum() > 100 and B.sum() > 100:
                ds.append(wrong[A].mean() - wrong[B].mean())
        lo_, hi_ = np.percentile(ds, [2.5, 97.5])
        res["blind_never_minus_remembered"] = dict(delta=float(d),
                                                   ci=[float(lo_), float(hi_)])
        print(f"\n  blind AND never seen  minus  blind but seen within 4 s:"
              f"  {100*d:+.2f} pts  [{100*lo_:+.2f}, {100*hi_:+.2f}]")

    # the deployable number
    stale = (age >= NEVER - .01)
    committed = conf >= a.conf
    print(f"\nvoxels asserted at p>={a.conf:.2f} that no camera has seen in "
          f"{MAXH//2} s: {100*(stale & committed).sum()/max(committed.sum(),1):.2f}%"
          f" of all committed voxels")
    print(f"  their error rate {100*wrong[stale & committed].mean():.2f}%  vs "
          f"{100*wrong[~stale & committed].mean():.2f}% for the rest")
    res["stale_committed_share"] = float((stale & committed).sum() / max(committed.sum(), 1))
    res["stale_committed_err"] = float(wrong[stale & committed].mean())
    res["fresh_committed_err"] = float(wrong[~stale & committed].mean())
    json.dump(dict(frames=len(toks), voxels=int(len(age)), bands=rows,
                   auroc=res, tau=a.tau, horizon_s=MAXH / 2),
              open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
