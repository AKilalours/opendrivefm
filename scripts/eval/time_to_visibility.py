#!/usr/bin/env python3
"""How long does a blind cell stay blind? Occlusion with a clock on it.

Every occlusion number in this project is a snapshot: this cell is unseen now.
A planner cannot act on that. What it can act on is how long the condition
lasts, because the ego's own motion resolves most occlusions on its own:

    time_to_visibility(v) = seconds until ANY camera first has evidence at the
                            world location this voxel currently occupies

Future observability maps are warped into the current ego frame through the
recorded poses, so the question is asked about a place in the world rather than
an index in a grid. Eight keyframes of look-ahead = 4 s at 2 Hz.

The number that matters is not the average over the whole blind volume -- most
of that is behind buildings the ego will never enter. It is the blind volume
**inside the corridor the ego actually drives through**, which this script
computes from the ego's own future trajectory rather than assuming straight
ahead.

Camera-only, no re-inference, no ground truth.
"""
from __future__ import annotations
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FREE, RES, RNG, NZ, Z0, N = 17, 0.4, 40.0, 16, -1.0, 200
MAXH = 8                      # keyframes of look-ahead; 2 Hz, so 4.0 s
_G = {}


def _init(obs_dir, tau, halfw):
    _G.update(obs=obs_dir, tau=tau, halfw=halfw)


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


CEN = centres()


def sample(P_glob, Rp, tp, arr):
    L = (P_glob - tp) @ Rp
    i = np.floor((L[:, 0] + RNG) / RES).astype(np.int32)
    j = np.floor((L[:, 1] + RNG) / RES).astype(np.int32)
    k = np.floor((L[:, 2] - Z0) / RES).astype(np.int32)
    ok = ((i >= 0) & (i < N) & (j >= 0) & (j < N) & (k >= 0) & (k < NZ))
    out = np.zeros(P_glob.shape[0], np.uint8)
    out[ok] = arr[i[ok], j[ok], k[ok]]
    return out, ok


def _one(job):
    tok, scene, cur_row, fut_rows = job
    NEVER = MAXH * 0.5 + 0.5
    obs = np.load(os.path.join(ROOT, _G["obs"], tok + ".npy"))
    blind = (obs.ravel().astype(np.float32) / 255.0) <= 0.0
    if not blind.any():
        return None
    Rc, tc = pose(cur_row)
    Pb = CEN[blind] @ Rc.T + tc                 # blind cells -> global

    ttv = np.full(Pb.shape[0], np.float32(NEVER))
    # does the ego DRIVE THROUGH this cell? Use the real future trajectory.
    on_path = np.zeros(Pb.shape[0], bool)
    for h, fr in enumerate(fut_rows, start=1):
        dt = (fr["timestamp"] - cur_row["timestamp"]) / 1e6
        Rf, tf = pose(fr)
        o, ok = sample(Pb, Rf, tf, np.load(
            os.path.join(ROOT, _G["obs"], fr["token"] + ".npy")))
        seen = ok & ((o.astype(np.float32) / 255.0) > _G["tau"])
        hit = seen & (ttv > dt)
        ttv[hit] = dt
        # the ego occupies roughly a halfw-wide box around its own origin at
        # each future pose; a blind cell inside that box is one it drives into
        L = (Pb - tf) @ Rf
        on_path |= (np.abs(L[:, 0]) < 2.4) & (np.abs(L[:, 1]) < _G["halfw"]) \
            & (L[:, 2] > -0.5) & (L[:, 2] < 2.0)

    edges = np.array([0.0, 0.6, 1.1, 2.1, 3.1, 4.1, 99.0])
    h_all = np.histogram(ttv, bins=edges)[0]
    h_path = np.histogram(ttv[on_path], bins=edges)[0]
    return dict(token=tok, scene=scene, n_blind=int(blind.sum()),
                n_path=int(on_path.sum()), all=h_all.tolist(),
                path=h_path.tolist(),
                med_all=float(np.median(ttv)),
                med_path=float(np.median(ttv[on_path])) if on_path.any() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="data/pack/obs_max")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=100000)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--halfw", type=float, default=1.4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--cache", default="outputs/artifacts/ttv_cache.jsonl")
    ap.add_argument("--out", default="outputs/artifacts/time_to_visibility.json")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    cache = os.path.join(ROOT, a.cache)
    os.makedirs(os.path.dirname(cache), exist_ok=True)

    if not a.report:
        index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
        by = {}
        for i, r in enumerate(index):
            by.setdefault(r["scene"], []).append(i)
        for s in by:
            by[s].sort(key=lambda i: index[i]["timestamp"])
        has = lambda t: os.path.exists(os.path.join(ROOT, a.obs, t + ".npy"))
        jobs = []
        for s, ids in by.items():
            for pos, i in enumerate(ids[:-MAXH]):
                fut = ids[pos + 1: pos + 1 + MAXH]
                if has(index[i]["token"]) and all(has(index[j]["token"]) for j in fut):
                    jobs.append((index[i]["token"], s, index[i],
                                 [index[j] for j in fut]))
        rng = np.random.default_rng(a.seed)
        if a.frames and a.frames < len(jobs):
            jobs = [jobs[i] for i in sorted(rng.choice(len(jobs), a.frames, replace=False))]
        done = set()
        if os.path.exists(cache):
            done = {json.loads(l)["token"] for l in open(cache)}
        todo = [j for j in jobs if j[0] not in done][:a.chunk]
        print(f"{len(jobs)} eligible | cached {len(done)} | this run {len(todo)} | "
              f"{os.cpu_count()} cores", flush=True)
        if not todo:
            print("complete -- run --report"); return
        t0 = time.time()
        with open(cache, "a") as fh, Pool(max(os.cpu_count() - 1, 1),
                                          initializer=_init,
                                          initargs=(a.obs, a.tau, a.halfw)) as pool:
            for n, r in enumerate(pool.imap_unordered(_one, todo, chunksize=2)):
                if r: fh.write(json.dumps(r) + "\n")
                if (n + 1) % 40 == 0:
                    fh.flush(); el = time.time() - t0
                    print(f"  {n+1}/{len(todo)}  {el/(n+1):.2f}s/frame  "
                          f"eta {(len(todo)-n-1)*el/(n+1)/60:.1f} min", flush=True)
        left = len(jobs) - len(done) - len(todo)
        print(f"chunk done, {left} remain" if left > 0 else
              "complete -- run --report", flush=True)
        return

    R = [json.loads(l) for l in open(cache)]
    A = np.array([r["all"] for r in R]).sum(0).astype(float)
    P = np.array([r["path"] for r in R]).sum(0).astype(float)
    lab = ["revealed within 0.5 s", "by 1.0 s", "by 2.0 s", "by 3.0 s",
           "by 4.0 s", "STILL BLIND after 4 s"]
    print(f"\n{len(R):,} frames | look-ahead {MAXH/2:.1f} s | "
          f"observed means obs > {a.tau}")
    print(f"blind voxels: {int(A.sum()):,} total, {int(P.sum()):,} on the ego's "
          f"own future path")
    print("=" * 74)
    print(f"{'time to visibility':<26}{'all blind':>13}{'share':>9}"
          f"{'on ego path':>14}{'share':>9}")
    print("-" * 74)
    for i, l in enumerate(lab):
        print(f"{l:<26}{int(A[i]):>13,}{100*A[i]/A.sum():>8.1f}%"
              f"{int(P[i]):>14,}{100*P[i]/max(P.sum(),1):>8.1f}%")
    print("=" * 74)
    ma = np.array([r["med_all"] for r in R])
    mp = np.array([r["med_path"] for r in R if r["med_path"] is not None])
    print(f"\nper-frame median time to visibility: {np.median(ma):.2f} s over all "
          f"blind cells, {np.median(mp):.2f} s over cells on the ego's path")
    print(f"share of the ego's own path that is blind now and STILL blind in 4 s: "
          f"{100*P[-1]/max(P.sum(),1):.1f}%")
    json.dump(dict(frames=len(R), horizon_s=MAXH / 2, tau=a.tau,
                   bins=lab, all=A.tolist(), on_path=P.tolist(),
                   median_all_s=float(np.median(ma)),
                   median_path_s=float(np.median(mp)),
                   path_still_blind_frac=float(P[-1] / max(P.sum(), 1))),
              open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
