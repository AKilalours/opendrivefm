#!/usr/bin/env python3
"""The verified-free corridor, given the memory the stack actually has.

A30/A38 measure the corridor from ONE frame's observability. That is a real
quantity and it is the conservative one, but it is not what the deployed system
knows. FB-OCC fuses sixteen frames, the vehicle is moving, and A25 measured how
long that memory stays useful: inside the blind set, a cell seen 0.5 s ago has
a 13.63% error rate against 23.83% for a cell no camera has seen in 4 s.

So the single-frame corridor is a LOWER bound on what the stack can justify,
and the question this file answers is how loose that bound is.

    obs_eff(v, T) = max over the current frame and every past keyframe within
                    T seconds, of the observability at the WORLD LOCATION this
                    cell currently occupies

Past frames only. A deployed stack has memory, not prophecy, and warping future
observability backwards -- which the time-to-visibility script legitimately does
for a different question -- would make this number a fiction. Poses come from
the recorded ego trajectory, so a cell is tracked as a place in the world rather
than an index in a grid.

WHAT THIS IS NOT
----------------
Memory is not evidence of PRESENT clearance. A patch of road seen two seconds
ago can hold a cyclist now. So this is not a better corridor than A38's -- it is
the OPTIMISTIC end of a bracket whose pessimistic end A38 already measured:

    A38  memory-free    every cell must be seen in the current frame
    here memory-perfect a cell counts if it was seen within T seconds and is
                        assumed unchanged since

The truth for a real planner sits between them, closer to A38 in traffic and
closer to this in an empty street. Reporting the bracket is honest; reporting
either end alone is not. T is swept rather than tuned, so the reader sees the
sensitivity instead of one chosen number.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corridor import verified_free, HALF_WIDTH, RES, RNG, NZ, Z0, N  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DECEL = 4.0
HORIZONS = (0, 1, 2, 4, 8)          # keyframes back; 2 Hz, so 0 / 0.5 / 1 / 2 / 4 s
_G = {}


def _init(obs_dir, tau):
    _G.update(obs=obs_dir, tau=tau)


def qrot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], np.float64)


def pose(row):
    c = row["cams"]["CAM_FRONT"]
    return (qrot(np.asarray(c["ego2global_rotation"], np.float64)),
            np.asarray(c["ego2global_translation"], np.float64))


# Only the lane corridor is ever read by verified_free's evidence term, so only
# those columns are warped. 200 x 7 x 16 points instead of 640,000.
J0 = int((RNG - HALF_WIDTH) / RES)
J1 = int((RNG + HALF_WIDTH) / RES)


def _corridor_centres():
    ax = (np.arange(N) + 0.5) * RES - RNG
    ay = (np.arange(J0, J1) + 0.5) * RES - RNG
    az = (np.arange(NZ) + 0.5) * RES + Z0
    X, Y, Z = np.meshgrid(ax, ay, az, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


CEN = _corridor_centres()
SHP = (N, J1 - J0, NZ)


def _sample(P_glob, Rp, tp, arr):
    L = (P_glob - tp) @ Rp
    i = np.floor((L[:, 0] + RNG) / RES).astype(np.int32)
    j = np.floor((L[:, 1] + RNG) / RES).astype(np.int32)
    k = np.floor((L[:, 2] - Z0) / RES).astype(np.int32)
    ok = ((i >= 0) & (i < N) & (j >= 0) & (j < N) & (k >= 0) & (k < NZ))
    out = np.zeros(P_glob.shape[0], np.uint8)
    out[ok] = arr[i[ok], j[ok], k[ok]]
    return out


def _one(job):
    tok, scene, speed, cur_row, past_rows = job
    cls = np.load(os.path.join(ROOT, "data/preds/preds_voxel",
                               tok + ".npz"))["cls"].astype(np.int16)
    obs = np.load(os.path.join(ROOT, _G["obs"], tok + ".npy"))
    Rc, tc = pose(cur_row)
    Pg = CEN @ Rc.T + tc
    acc = obs[:, J0:J1, :].reshape(-1).copy()

    out, prev = {}, 0
    for h in HORIZONS:
        if h > 0:
            if h > len(past_rows):
                break
            # every frame between the last horizon and this one, not just the
            # newest: HORIZONS jumps 2 -> 4 -> 8, so stepping one frame per
            # horizon silently dropped three of the eight.
            for fr in past_rows[prev:h]:
                acc = np.maximum(acc, _sample(
                    Pg, *pose(fr),
                    np.load(os.path.join(ROOT, _G["obs"], fr["token"] + ".npy"))))
            prev = h
        a3 = acc.reshape(SHP)
        o2 = obs.copy()
        o2[:, J0:J1, :] = a3
        start, length, _, sf, ss = verified_free(cls, o2, tau=_G["tau"])
        # corridor columns carrying ANY evidence. Without this the flat reach
        # curve below is indistinguishable from a warp that silently does
        # nothing, and a null result you cannot separate from a no-op is not a
        # result.
        cov = float((a3.max(-1) >= _G["tau"] * 255).mean())
        out[str(h)] = [start, length, sf, ss, cov]
    return dict(token=tok, scene=scene, speed=speed, h=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="data/pack/obs_max")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--chunk", type=int, default=100000)
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--decel", type=float, default=DECEL)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--cache", default="outputs/artifacts/corridor_temporal_cache.jsonl")
    ap.add_argument("--out", default="outputs/artifacts/corridor_temporal.json")
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
        have = lambda t: (os.path.exists(os.path.join(
            ROOT, "data/preds/preds_voxel", t + ".npz")) and
            os.path.exists(os.path.join(ROOT, a.obs, t + ".npy")))
        jobs = []
        for s, ids in by.items():
            for n, i in enumerate(ids):
                if n < max(HORIZONS):
                    continue                      # needs a full history
                cur = index[i]
                past = [index[ids[n - d]] for d in range(1, max(HORIZONS) + 1)]
                if not have(cur["token"]) or not all(have(p["token"]) for p in past):
                    continue
                p0 = np.asarray(index[ids[n-1]]["cams"]["CAM_FRONT"]["ego2global_translation"])
                p1 = np.asarray(cur["cams"]["CAM_FRONT"]["ego2global_translation"])
                dt = (cur["timestamp"] - index[ids[n-1]]["timestamp"]) / 1e6
                jobs.append((cur["token"], s,
                             float(np.linalg.norm(p1 - p0) / max(dt, 1e-3)),
                             cur, past))
        rng = np.random.default_rng(a.seed)
        rng.shuffle(jobs)
        jobs = jobs[:a.frames]
        done = set()
        if os.path.exists(cache):
            done = {json.loads(l)["token"] for l in open(cache)}
        todo = [j for j in jobs if j[0] not in done][:a.chunk]
        print(f"{len(jobs)} eligible | cached {len(done)} | this run {len(todo)}",
              flush=True)
        if not todo:
            print("complete -- run --report"); return
        t0 = time.time()
        with open(cache, "a") as fh, Pool(max(os.cpu_count() - 1, 1),
                                          initializer=_init,
                                          initargs=(a.obs, a.tau)) as pool:
            for n, r in enumerate(pool.imap_unordered(_one, todo, chunksize=2)):
                fh.write(json.dumps(r) + "\n")
                if (n + 1) % 50 == 0:
                    fh.flush(); el = time.time() - t0
                    print(f"  {n+1}/{len(todo)}  {el/(n+1):.2f}s/frame  "
                          f"eta {(len(todo)-n-1)*el/(n+1)/60:.1f} min", flush=True)
        left = len(jobs) - len(done) - len(todo)
        print(f"chunk done, {left} remain" if left > 0 else
              "complete -- run --report", flush=True)
        return

    R = [json.loads(l) for l in open(cache)]
    R = [r for r in R if str(max(HORIZONS)) in r["h"]]
    v = np.array([r["speed"] for r in R])
    dstop = v ** 2 / (2 * a.decel)
    print(f"\n{len(R):,} frames | tau {a.tau} | braking {a.decel} m/s^2")
    print("A38's memory-free corridor is the T = 0.0 s row.")
    print("=" * 78)
    print(f"{'memory horizon':<18}{'evidence':>10}{'med reach':>11}{'p10 reach':>11}"
          f"{'flagged':>10}{'>10 m/s':>10}{'no corr':>9}")
    print("-" * 78)
    rows, fast = [], v >= 10.0
    for h in HORIZONS:
        k = str(h)
        rc = np.array([r["h"][k][1] for r in R])
        cv = np.array([r["h"][k][4] for r in R])
        fl = rc < dstop
        rows.append(dict(keyframes=h, seconds=h * 0.5,
                         corridor_evidence=float(cv.mean()),
                         reach_median=float(np.median(rc)),
                         reach_p10=float(np.percentile(rc, 10)),
                         flagged=float(fl.mean()),
                         flagged_above_10=float(fl[fast].mean()) if fast.any() else None,
                         no_corridor=float((rc <= 0).mean())))
        print(f"{h*0.5:>6.1f} s ({h:>2} kf){100*cv.mean():>9.1f}%"
              f"{np.median(rc):>11.1f}{np.percentile(rc,10):>11.1f}"
              f"{100*fl.mean():>9.1f}%{100*fl[fast].mean():>9.1f}%"
              f"{100*(rc<=0).mean():>8.1f}%")
    print("=" * 78)
    b = rows[0]["flagged_above_10"]; e = rows[-1]["flagged_above_10"]
    print(f"\nabove 10 m/s, the bracket is {100*e:.1f}% to {100*b:.1f}% flagged:")
    print(f"  memory-free (A38)     {100*b:.1f}%   every cell seen this frame")
    print(f"  memory-perfect, 4 s   {100*e:.1f}%   seen within 4 s and assumed unchanged")
    print(f"  {100*(b-e)/max(b,1e-9):.0f}% of the single-frame figure is "
          f"recoverable by memory alone.")
    c0, c1 = rows[0]["corridor_evidence"], rows[-1]["corridor_evidence"]
    print(f"\nmemory fills the corridor: evidence on {100*c0:.1f}% of columns "
          f"-> {100*c1:.1f}%,")
    print(f"and the envelope barely moves, because A38 measured that the "
          f"corridor ends on an\nOBSTACLE 96% of the time. Evidence behind an "
          f"obstacle is not clearance in front of one.")
    json.dump(dict(frames=len(R), tau=a.tau, decel=a.decel,
                   n_above_10=int(fast.sum()), horizons=rows),
              open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
