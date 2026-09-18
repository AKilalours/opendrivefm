#!/usr/bin/env python3
"""Stage 4 -- does low observability predict a MISSED OBJECT, not just a wrong voxel?

Everything measured so far is per voxel. A voxel being mislabelled is a
benchmark quantity; an object being absent from the occupancy field is a thing
a planner would drive into. This is the step that connects the two.

For each ground-truth object:
    detected   at least one voxel inside its box carries a predicted
               non-free class
    obs        camera observability over the ground-truth occupied voxels
               inside that box

Then: is the miss rate concentrated in the BARELY observed band rather than the
blind one? The danger-zone result (A14) says the model commits hardest where it
can just barely see, so the prediction is that misses peak in the low-but-
nonzero band, not at zero.

Boxes arrive in global coordinates and the occupancy grid is ego-framed, so the
transform is verified before it is used (`--probe`): the z offset is swept and
the value that maximises agreement between box interiors and same-class ground
truth must be the one the convention implies. A transform that cannot be
confirmed this way is not used.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FREE, RES, RNG, NZ, Z0 = 17, 0.4, 40.0, 16, -1.0

# nuScenes detection categories -> Occ3D class id
CAT2OCC = {
    "vehicle.car": 4, "vehicle.truck": 10, "vehicle.bus.rigid": 3,
    "vehicle.bus.bendy": 3, "vehicle.construction": 5, "vehicle.trailer": 9,
    "vehicle.motorcycle": 6, "vehicle.bicycle": 2,
    "human.pedestrian.adult": 7, "human.pedestrian.child": 7,
    "human.pedestrian.construction_worker": 7, "human.pedestrian.police_officer": 7,
    "movable_object.barrier": 1, "movable_object.trafficcone": 8,
}
DYN = {2, 3, 4, 5, 6, 7, 9, 10}


def qrot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], np.float64)


def box_voxels(c, size, rot, zoff, pad=0.0):
    """Indices of voxels whose centre lies inside an oriented box."""
    w, l, h = size                      # nuScenes order: width, length, height
    half = np.array([l, w, h]) / 2.0 + pad
    r = float(np.linalg.norm(half))
    lo = np.floor((c - r + np.array([RNG, RNG, -Z0])) / RES).astype(int)
    hi = np.ceil((c + r + np.array([RNG, RNG, -Z0])) / RES).astype(int)
    lo = np.maximum(lo, 0); hi = np.minimum(hi, [200, 200, NZ])
    if np.any(hi <= lo):
        return None
    gi, gj, gk = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]),
                             np.arange(lo[2], hi[2]), indexing="ij")
    P = np.stack([(gi + .5) * RES - RNG, (gj + .5) * RES - RNG,
                  (gk + .5) * RES + Z0], -1).reshape(-1, 3)
    L = (P - c) @ rot                   # world -> box axes
    inside = np.all(np.abs(L) <= half, 1)
    if not inside.any():
        return None
    return (gi.ravel()[inside], gj.ravel()[inside], gk.ravel()[inside])


def frames_with(ann, index):
    return ann["frame"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--minpts", type=int, default=1)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--cache", default="outputs/artifacts/missed_cache.jsonl")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default="outputs/artifacts/missed_detection.json")
    a = ap.parse_args()

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    ann = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"),
                  allow_pickle=True)
    names = ann["category_names"]
    gt_map = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}
    fr = ann["frame"]
    order = np.argsort(fr, kind="stable")
    bounds = np.searchsorted(fr[order], np.arange(len(index) + 1))

    rng = np.random.default_rng(a.seed)
    cand = [i for i in range(len(index))
            if index[i]["token"] in gt_map
            and os.path.exists(os.path.join(ROOT, "data/preds/preds_voxel",
                                            index[i]["token"] + ".npz"))
            and bounds[i + 1] > bounds[i]]
    sel = sorted(rng.choice(len(cand), min(a.frames, len(cand)), replace=False))
    frames = [cand[i] for i in sel]

    cache = os.path.join(ROOT, a.cache)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    done = set()
    if os.path.exists(cache):
        done = {json.loads(l)["token"] for l in open(cache)}
    if a.report:
        frames = []
    if not a.probe and not a.report:
        pend = [f for f in frames if index[f]["token"] not in done]
        n_left = len(pend)
        frames = pend[:a.chunk]
        print(f"sample {len(sel)} | remaining {n_left} | this run {len(frames)}",
              flush=True)

    zoffs = [-2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0] \
        if a.probe else [0.0]
    if a.probe:
        frames = frames[:a.frames]
        print("z-offset probe: fraction of DYNAMIC boxes whose interior contains "
              "at least one ground-truth voxel of the SAME class\n")

    for zoff in zoffs:
        hit = tot = 0
        recs = []
        for fi in frames:
            row = index[fi]
            tok = row["token"]
            g = np.load(gt_map[tok])
            gcls = g["semantics"].astype(np.int16)
            cam0 = row["cams"]["CAM_FRONT"]
            Re = qrot(np.asarray(cam0["ego2global_rotation"], np.float64))
            te = np.asarray(cam0["ego2global_translation"], np.float64)
            if not a.probe:
                pcls = np.load(os.path.join(ROOT, "data/preds/preds_voxel",
                                            tok + ".npz"))["cls"].astype(np.int16)
                obs = np.load(os.path.join(ROOT, "data/pack/obs_ray",
                                           tok + ".npy")).astype(np.float32) / 255.0
            sl = order[bounds[fi]:bounds[fi + 1]]
            for k in sl:
                cid = CAT2OCC.get(str(names[ann["category"][k]]))
                if cid is None or cid not in DYN:
                    continue
                if ann["num_lidar_pts"][k] < a.minpts:
                    continue
                c = Re.T @ (ann["translation"][k].astype(np.float64) - te)
                c[2] += zoff
                R = Re.T @ qrot(np.asarray(ann["rotation"][k], np.float64))
                vx = box_voxels(c, ann["size"][k].astype(np.float64), R, zoff)
                if vx is None:
                    continue
                tot += 1
                same = (gcls[vx] == cid).any()
                hit += bool(same)
                if a.report:
                    pass
                if not a.probe and same:
                    gm = gcls[vx] == cid
                    o = float(obs[vx][gm].max())
                    det = bool((pcls[vx] != FREE).any())
                    dyn = bool(np.isin(pcls[vx], list(DYN)).any())
                    exact = bool((pcls[vx] == cid).any())
                    recs.append((o, det, dyn, exact, int(ann["visibility"][k]),
                                 int(ann["num_lidar_pts"][k]), row["scene"], cid,
                                 tok, float(np.hypot(c[0], c[1]))))
        if a.probe:
            print(f"  z offset {zoff:+.1f} m   agreement {100*hit/max(tot,1):5.1f}%"
                  f"   ({hit:,}/{tot:,})", flush=True)
    if a.probe:
        return

    if not a.report:
        with open(cache, "a") as fh:
            for fi in sorted({r[8] for r in recs}):
                pass
            for r in recs:
                fh.write(json.dumps(dict(token=r[8], obs=r[0], det=r[1],
                                         dyn=r[2], exact=r[3], vis=r[4],
                                         pts=r[5], scene=r[6], cls=r[7],
                                         rng=r[9])) + "\n")
            for fi in frames:
                t = index[fi]["token"]
                if t not in {r[8] for r in recs}:
                    fh.write(json.dumps(dict(token=t, empty=True)) + "\n")
        left = n_left - len(frames)
        print("chunk done, %d remain" % left if left > 0 else
              "sample complete -- run with --report", flush=True)
        return

    recs = []
    for l in open(cache):
        d = json.loads(l)
        if d.get("empty"):
            continue
        recs.append((d["obs"], d["det"], d["dyn"], d["exact"], d["vis"],
                     d["pts"], d["scene"], d["cls"], d["token"], d["rng"]))
    tot = len(recs)
    frames = list({json.loads(l)["token"] for l in open(cache)})

    print(f"\n{len(frames)} frames, {tot:,} dynamic boxes, "
          f"{len(recs):,} confirmed in ground truth")
    o = np.array([r[0] for r in recs])
    det = np.array([r[1] for r in recs])
    dyn = np.array([r[2] for r in recs])
    exact = np.array([r[3] for r in recs])
    scn = np.array([r[6] for r in recs])
    scenes = sorted(set(scn.tolist()))

    nz = o[o > 0]
    edges = np.concatenate([[0.0], np.quantile(nz, [.2, .4, .6, .8]), [1.0001]])
    lab = ["obs = 0  (no camera evidence)"] + \
          [f"obs {edges[i]:.3f}-{edges[i+1]:.3f}" for i in range(5)]
    band = np.where(o <= 0, 0, np.clip(np.searchsorted(edges, o, "right"), 1, 5))

    print("=" * 92)
    print(f"{'observability band':<34}{'objects':>9}{'missed entirely':>18}"
          f"{'no dynamic class':>19}{'wrong class':>12}")
    print("-" * 92)
    rows_out = []
    for b in range(6):
        m = band == b
        if not m.any():
            continue
        print(f"{lab[b]:<34}{m.sum():>9,}{100*(~det[m]).mean():>17.2f}%"
              f"{100*(~dyn[m]).mean():>18.2f}%{100*(dyn[m]&~exact[m]).mean():>11.2f}%")
        rows_out.append(dict(band=lab[b], n=int(m.sum()),
                             miss=float((~det[m]).mean()),
                             no_dyn=float((~dyn[m]).mean()),
                             wrong_cls=float((dyn[m] & ~exact[m]).mean())))
    print("=" * 92)

    # paired scene bootstrap on the contrast that matters
    lowest = (band == 1)
    blind = (band == 0)
    high = (band == 5)
    def boot(mA, mB, vec):
        d = []
        for _ in range(a.boot):
            pick = rng.choice(scenes, len(scenes), replace=True)
            sm = np.isin(scn, pick)
            A, Bm = mA & sm, mB & sm
            if A.any() and Bm.any():
                d.append(vec[A].mean() - vec[Bm].mean())
        return np.percentile(d, [2.5, 97.5]) if d else (np.nan, np.nan)

    miss = ~dyn
    for nm, mA, mB in (("barely seen  vs  best seen", lowest, high),
                       ("barely seen  vs  no evidence", lowest, blind),
                       ("no evidence  vs  best seen", blind, high)):
        lo, hi = boot(mA, mB, miss.astype(float))
        print(f"{nm:<32}{100*(miss[mA].mean()-miss[mB].mean()):>+8.2f} pts"
              f"   [{100*lo:+.2f}, {100*hi:+.2f}]")
    # --- the control a reviewer will demand: is this just range? ---
    R = np.array([r[9] for r in recs])
    V = np.array([r[4] for r in recs])          # nuScenes annotator visibility 0-3
    miss_v = (~dyn).astype(float)
    def auroc_(score, y):
        s_ = np.asarray(score, float); o_ = np.argsort(s_)
        r_ = np.empty(len(s_)); r_[o_] = np.arange(1, len(s_) + 1)
        P, N = y.sum(), (1 - y).sum()
        if P == 0 or N == 0: return float("nan")
        return float((r_[y == 1].sum() - P * (P + 1) / 2) / (P * N))
    print("\npredicting 'no dynamic class was placed here', per object")
    print("-" * 62)
    ctrl = {}
    for nm, sc in (("observability (ours, inverted)", -o),
                   ("range from ego (inverted)", R),
                   ("nuScenes annotator visibility (inverted)", -V.astype(float)),
                   ("lidar points in box (inverted)",
                    -np.array([r[5] for r in recs], float))):
        v = auroc_(sc, miss_v)
        ctrl[nm] = v
        print(f"  {nm:<44}AUROC {v:.4f}")
    # observability within a fixed range shell -- range held constant
    print("\n  observability inside fixed range shells (range held constant)")
    shells = [(0, 10), (10, 20), (20, 30), (30, 55)]
    for a_, b_ in shells:
        m = (R >= a_) & (R < b_)
        if m.sum() < 200: continue
        v = auroc_(-o[m], miss_v[m])
        ctrl[f"shell_{a_}_{b_}"] = v
        print(f"    {a_:>2}-{b_:<3} m   n={m.sum():>6,}   AUROC {v:.4f}")
    json.dump(dict(frames=len(frames), objects=len(recs), bands=rows_out,
                   controls=ctrl),
              open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
