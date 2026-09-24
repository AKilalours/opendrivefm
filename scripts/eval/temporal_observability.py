#!/usr/bin/env python3
"""A44. Does evidence you had a second ago predict error better than evidence
you have now?

A25 asked "how long since this cell was last seen" and could only ask it of
cells blind right now, so the broad claim was untestable rather than false.
A41 section 6 then registered a repair that reproduced the same degeneracy.
A44 registers the real one, before this ran:

    obs_T(v) = max over the current and past 8 keyframes of
               obs_t(v) * lambda^(seconds since t)

Past frames only, warped through the recorded ego poses, so a cell is tracked
as a place in the world. Defined on every voxel, and nested -- at lambda = 0 it
IS observability, so the contrast is like-for-like.

Half-life pre-named at 2.0 s in A44. The sweep exists to show sensitivity, not
to be mined; the 2.0 s row is the result.

Scored inside mask_camera, A5's pre-registered scope. AUROC by exact histogram
on uint8-quantised scores, accumulated per scene so the bootstrap can pair.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FREE, RES, RNG, NZ, Z0, N = 17, 0.4, 40.0, 16, -1.0, 200
MAXH = 8
HALF_LIVES = (0.5, 1.0, 2.0, 4.0, 1e9)     # 2.0 is the pre-named primary
PRIMARY = 2.0
NB = 256
_G = {}


def qrot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], np.float64)


def pose(row):
    c = row["cams"]["CAM_FRONT"]
    return (qrot(np.asarray(c["ego2global_rotation"], np.float64)),
            np.asarray(c["ego2global_translation"], np.float64))


def _centres():
    ax = (np.arange(N) + 0.5) * RES - RNG
    az = (np.arange(NZ) + 0.5) * RES + Z0
    X, Y, Z = np.meshgrid(ax, ax, az, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


CEN = _centres()


def _sample(P, Rp, tp, arr):
    L = (P - tp) @ Rp
    i = np.floor((L[:, 0] + RNG) / RES).astype(np.int32)
    j = np.floor((L[:, 1] + RNG) / RES).astype(np.int32)
    k = np.floor((L[:, 2] - Z0) / RES).astype(np.int32)
    ok = ((i >= 0) & (i < N) & (j >= 0) & (j < N) & (k >= 0) & (k < NZ))
    out = np.zeros(P.shape[0], np.uint8)
    out[ok] = arr[i[ok], j[ok], k[ok]]
    return out


def _init(obs_dir):
    _G["obs"] = obs_dir
    _G["gt"] = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}


def _one(job):
    tok, scene, cur, past = job
    g = np.load(_G["gt"][tok])
    gc = g["semantics"].astype(np.int16).ravel()
    mk = g["mask_camera"].astype(bool).ravel()
    pc = np.load(os.path.join(ROOT, "data/preds/preds_voxel",
                              tok + ".npz"))["cls"].astype(np.int16).ravel()
    obs0 = np.load(os.path.join(ROOT, _G["obs"], tok + ".npy")).ravel()

    Rc, tc = pose(cur)
    Pg = CEN @ Rc.T + tc
    warped, ages = [], []
    for fr in past:
        dt = abs(cur["timestamp"] - fr["timestamp"]) / 1e6
        warped.append(_sample(Pg, *pose(fr),
                              np.load(os.path.join(ROOT, _G["obs"],
                                                   fr["token"] + ".npy"))))
        ages.append(dt)

    wrong = (pc != gc)[mk]
    out = {}
    for hl in HALF_LIVES:
        if hl >= 1e8:                      # no decay: pure max over the window
            acc = obs0.copy()
            for w in warped:
                acc = np.maximum(acc, w)
        else:
            lam = 0.5 ** (1.0 / hl)
            acc = obs0.astype(np.float32)
            for w, dt in zip(warped, ages):
                acc = np.maximum(acc, w.astype(np.float32) * (lam ** dt))
            acc = np.clip(acc, 0, 255).astype(np.uint8)
        s = acc[mk]
        out[str(hl)] = (np.bincount(s[wrong], minlength=NB).astype(np.int64),
                        np.bincount(s[~wrong], minlength=NB).astype(np.int64))
    s0 = obs0[mk]
    out["obs"] = (np.bincount(s0[wrong], minlength=NB).astype(np.int64),
                  np.bincount(s0[~wrong], minlength=NB).astype(np.int64))
    return scene, out


def auroc(pos, neg):
    """AUROC from histograms, score INVERTED (low observability = error)."""
    p = pos[::-1].astype(np.float64); n = neg[::-1].astype(np.float64)
    cn = np.cumsum(n) - n
    P, Nn = p.sum(), n.sum()
    if P == 0 or Nn == 0:
        return float("nan")
    return float(((p * (cn + n / 2.0)).sum()) / (P * Nn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="data/pack/obs_max")
    ap.add_argument("--frames", type=int, default=700)
    ap.add_argument("--chunk", type=int, default=100000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--store", default="outputs/artifacts/tempobs_store.npz")
    ap.add_argument("--out", default="outputs/artifacts/temporal_observability.json")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    store = os.path.join(ROOT, a.store)
    keys = ["obs"] + [str(h) for h in HALF_LIVES]

    if not a.report:
        index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
        gt = {os.path.basename(os.path.dirname(q)) for q in glob.glob(
            os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                         "*", "*", "labels.npz"))}
        by = {}
        for i, r in enumerate(index):
            by.setdefault(r["scene"], []).append(i)
        for s in by:
            by[s].sort(key=lambda i: index[i]["timestamp"])
        have = lambda t: (t in gt and os.path.exists(os.path.join(
            ROOT, "data/preds/preds_voxel", t + ".npz")) and os.path.exists(
            os.path.join(ROOT, a.obs, t + ".npy")))
        jobs = []
        for s, ids in by.items():
            for n_, i in enumerate(ids):
                if n_ < MAXH:
                    continue
                cur = index[i]; past = [index[ids[n_ - d]] for d in range(1, MAXH + 1)]
                if not have(cur["token"]) or not all(
                        os.path.exists(os.path.join(ROOT, a.obs, p["token"] + ".npy"))
                        for p in past):
                    continue
                jobs.append((cur["token"], s, cur, past))
        rng = np.random.default_rng(a.seed)
        rng.shuffle(jobs)
        jobs = jobs[:a.frames]
        scenes, H, done = [], {}, set()
        if os.path.exists(store):
            z = np.load(store, allow_pickle=True)
            scenes = list(z["scenes"]); done = set(z["done"].tolist())
            H = {k: z[k] for k in keys}
        todo = [j for j in jobs if j[0] not in done][:a.chunk]
        print(f"{len(jobs)} eligible | stored {len(done)} | this run {len(todo)}",
              flush=True)
        if not todo:
            print("build complete -- run --report"); return
        sidx = {s: i for i, s in enumerate(scenes)}
        if not H:
            H = {k: np.zeros((0, 2, NB), np.int64) for k in keys}
        t0 = time.time()
        with Pool(max(os.cpu_count() - 1, 1), initializer=_init,
                  initargs=(a.obs,)) as pool:
            for c, (scene, out) in enumerate(pool.imap_unordered(_one, todo, chunksize=2)):
                if scene not in sidx:
                    sidx[scene] = len(scenes); scenes.append(scene)
                    for k in keys:
                        H[k] = np.concatenate([H[k], np.zeros((1, 2, NB), np.int64)])
                j = sidx[scene]
                for k in keys:
                    H[k][j, 0] += out[k][0]; H[k][j, 1] += out[k][1]
                if (c + 1) % 50 == 0:
                    el = time.time() - t0
                    print(f"  {c+1}/{len(todo)}  {el/(c+1):.2f}s/frame  "
                          f"eta {(len(todo)-c-1)*el/(c+1)/60:.1f} min", flush=True)
        done |= {j[0] for j in todo}
        np.savez_compressed(store, scenes=np.array(scenes),
                            done=np.array(sorted(done)), **H)
        left = len(jobs) - len(done)
        print(f"chunk done, {left} remain" if left > 0 else
              "build complete -- run --report", flush=True)
        return

    z = np.load(store, allow_pickle=True)
    scenes = list(z["scenes"]); H = {k: z[k] for k in keys}
    ns = len(scenes)
    tot = H["obs"].sum()
    print(f"\n{len(z['done'])} frames | {ns} scenes | {int(tot):,} voxels "
          f"inside mask_camera")
    base = auroc(H["obs"].sum(0)[0], H["obs"].sum(0)[1])
    print("=" * 74)
    print(f"{'score':<34}{'AUROC':>10}{'vs obs':>11}{'95% CI':>19}")
    print("-" * 74)
    print(f"{'observability (A26 measure)':<34}{base:>10.4f}")
    rng = np.random.default_rng(a.seed)
    idx = [rng.choice(ns, ns, replace=True) for _ in range(a.boot)]
    res = {"obs": dict(auroc=base)}
    for hl in HALF_LIVES:
        k = str(hl)
        au = auroc(H[k].sum(0)[0], H[k].sum(0)[1])
        d = []
        for ii in idx:
            a1 = auroc(H[k][ii].sum(0)[0], H[k][ii].sum(0)[1])
            a0 = auroc(H["obs"][ii].sum(0)[0], H["obs"][ii].sum(0)[1])
            d.append(a1 - a0)
        d = np.asarray(d); lo, hi = np.percentile(d, [2.5, 97.5])
        nm = ("no decay (plain max)" if hl >= 1e8 else
              f"temporal obs, half-life {hl:.1f} s")
        star = "  <- PRE-NAMED PRIMARY" if hl == PRIMARY else ""
        print(f"{nm:<34}{au:>10.4f}{d.mean():>+11.4f}"
              f"   [{lo:+.4f}, {hi:+.4f}]{star}")
        res[k] = dict(auroc=au, delta=float(d.mean()), lo=float(lo), hi=float(hi))
    print("=" * 74)
    p = res[str(PRIMARY)]
    holds = p["lo"] > 0
    print(f"\nACCEPTANCE (fixed in A44 before this ran): delta > 0 at "
          f"half-life {PRIMARY} s, interval excluding zero")
    print(f"  measured {p['delta']:+.4f}  [{p['lo']:+.4f}, {p['hi']:+.4f}]")
    print(f"\nH4: {'HOLDS' if holds else 'FAILS'}")
    res["primary_half_life"] = PRIMARY
    res["holds"] = bool(holds)
    res["frames"] = int(len(z["done"])); res["scenes"] = ns
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
