#!/usr/bin/env python3
"""A79: does observability-gating catch real hazards, at a matched alarm budget?

Design is fixed in docs/METHOD_FREEZE.md (A79) BEFORE this ran. Two arms of one
function -- verified_free with tau=0.15 (gated) and tau=0.0 (free-only) -- a
hazard label taken from Occ3D ground truth alone, and a comparison at fixed
flag budgets so the arms are compared as signals rather than as thresholds.

    python3 scripts/eval/a79_safety_downstream.py --boot 2000 \
        --out outputs/artifacts/a79_safety_downstream.json
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corridor import verified_free, FREE, RES, RNG, Z0, NZ, BUMPER, \
    HALF_WIDTH, CLEAR_FROM, CLEAR_TO, OCC_FRAC          # noqa: E402
import split as _sp                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DECEL = 4.0
HORIZONS = (10.0, 20.0, 30.0)     # A81; 20 m is PRIMARY (power), see amendment
_G = {}


def _init(obs_dir, gt_map, decel, horizons):
    _G.update(obs=obs_dir, gt=gt_map, decel=decel, H=horizons)


def _one(args):
    tok, scene, speed = args
    cls = np.load(os.path.join(ROOT, "data/preds/preds_voxel",
                               tok + ".npz"))["cls"].astype(np.int16)
    obs = np.load(os.path.join(ROOT, _G["obs"], tok + ".npy")).astype(np.float32) / 255.0

    _, reach_g, _, _, _ = verified_free(cls, obs, tau=0.15)
    _, reach_f, _, _, _ = verified_free(cls, obs, tau=0.0)
    dstop = speed ** 2 / (2.0 * _G["decel"])

    # hazard: a GT-occupied voxel inside the corridor, within stopping distance
    # of the bumper, that the model called free. GT only; neither arm enters it.
    gp = _G["gt"].get(tok)
    if gp is None:
        return None
    z = np.load(gp)
    gt = z["semantics"] if "semantics" in z.files else z[z.files[0]]
    j0 = int((RNG - HALF_WIDTH) / RES); j1 = int((RNG + HALF_WIDTH) / RES)
    k0 = max(int((CLEAR_FROM - Z0) / RES), 0)
    k1 = min(int((CLEAR_TO - Z0) / RES), NZ)
    i0 = int((RNG + BUMPER) / RES)
    i1 = min(i0 + int(max(dstop, 0.0) / RES) + 1, cls.shape[0])
    haz = False
    if i1 > i0:
        g = gt[i0:i1, j0:j1, k0:k1]
        p = cls[i0:i1, j0:j1, k0:k1]
        haz = bool(((g != FREE) & (p == FREE)).any())
    # A81: fixed-range hazard windows. Speed does not enter these labels.
    hz = {}
    for H in _G["H"]:
        iH = min(i0 + int(H / RES), cls.shape[0])
        g = gt[i0:iH, j0:j1, k0:k1]
        p = cls[i0:iH, j0:j1, k0:k1]
        hz[f"hazard_{int(H)}m"] = bool(((g != FREE) & (p == FREE)).any())
    return dict(token=tok, scene=scene, speed=float(speed), dstop=float(dstop),
                reach_gated=float(reach_g), reach_free=float(reach_f),
                hazard=haz, **hz)


def auroc(s, y):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.empty(len(s), np.float64)
    o = np.argsort(s, kind="mergesort")
    sr = s[o]; i = 0
    while i < len(sr):                      # ties get the mean rank
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    np_, nn = y.sum(), (~y).sum()
    return float((r[y].sum() - np_ * (np_ + 1) / 2.0) / (np_ * nn))


def recall_at(s, y, budget):
    """Hazard recall when the top `budget` fraction of frames is flagged."""
    y = np.asarray(y, bool)
    k = max(int(round(budget * len(s))), 1)
    idx = np.argsort(-s, kind="mergesort")[:k]
    return float(y[idx].sum() / max(y.sum(), 1))


def report_a81(R, a, keep):
    """A81: fixed-range hazard windows. No kinematic term in label or score."""
    sc = np.array([r["scene"] for r in R])
    rg = np.array([r["reach_gated"] for r in R])
    rf = np.array([r["reach_free"] for r in R])
    sp = np.array([r["speed"] for r in R])
    arms = {"gated (tau=0.15)": -rg, "free-only (tau=0.0)": -rf,
            "speed alone (confound check)": sp}
    scenes = sorted(set(sc))
    idx = {s: np.where(sc == s)[0] for s in scenes}
    rs = np.random.default_rng(a.seed)
    draws = rs.integers(0, len(scenes), size=(a.boot, len(scenes)))
    BUD = (0.05, 0.10, 0.20)
    out = {"frames": len(R), "scenes": len(scenes), "boot": a.boot,
           "primary": "H=20m", "horizons": {}}
    print(f"\n{len(R):,} frames | {len(scenes)} scenes | "
          f"PRIMARY endpoint H=20 m (declared on DEV, by power)\n")
    for H in HORIZONS:
        key = f"hazard_{int(H)}m"
        y = np.array([r[key] for r in R], bool)
        tag = "PRIMARY" if H == 20.0 else "secondary"
        print(f"=== H = {int(H)} m  ({tag})   hazard rate {100*y.mean():.1f}%"
              f"  ({int(y.sum()):,} frames) ===")
        point = {k: {"auroc": auroc(v, y),
                     **{f"recall@{int(b*100)}%": recall_at(v, y, b) for b in BUD}}
                 for k, v in arms.items()}
        print(f"{'arm':<30}{'AUROC':>9}" +
              "".join(f"{f'rec@{int(b*100)}%':>11}" for b in BUD))
        print("-" * 74)
        for k, d in point.items():
            print(f"{k:<30}{d['auroc']:>9.4f}" +
                  "".join(f"{d[f'recall@{int(b*100)}%']:>11.4f}" for b in BUD))
        keys = ["auroc"] + [f"recall@{int(b*100)}%" for b in BUD]
        boot = {k: {m: np.empty(a.boot) for m in keys} for k in arms}
        for bi, d in enumerate(draws):
            sel = np.concatenate([idx[scenes[i]] for i in d])
            yy = y[sel]
            for k, v in arms.items():
                vv = v[sel]
                boot[k]["auroc"][bi] = auroc(vv, yy)
                for b in BUD:
                    boot[k][f"recall@{int(b*100)}%"][bi] = recall_at(vv, yy, b)

        def ci(x):
            x = x[~np.isnan(x)]
            return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

        print("  gated - free-only, paired over scenes")
        ctr = {}
        for m in keys:
            dd = boot["gated (tau=0.15)"][m] - boot["free-only (tau=0.0)"][m]
            lo, hi = ci(dd)
            pt = point["gated (tau=0.15)"][m] - point["free-only (tau=0.0)"][m]
            t = "excludes 0" if (lo > 0 or hi < 0) else "SPANS 0"
            print(f"    {m:<14}{pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {t}")
            ctr[m] = [pt, lo, hi]
        out["horizons"][f"{int(H)}m"] = {"hazard_rate": float(y.mean()),
                                         "point": point, "contrasts": ctr}
        print()
    pt, lo, hi = out["horizons"]["20m"]["contrasts"]["auroc"]
    ok = (pt > 0) and (lo > 0 or hi < 0)
    print("--- A81 acceptance (PRIMARY, H = 20 m, AUROC) ---")
    print(f"  gated - free-only  {pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    out["verdict"] = "PASS" if ok else "FAIL"
    if a.out:
        json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
        print("\nwrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="data/pack/obs_max")
    ap.add_argument("--gts", default="data/occ3d/Occupancy3D-nuScenes-trainval/gts")
    ap.add_argument("--decel", type=float, default=DECEL)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--cache", default="outputs/artifacts/a79_cache.jsonl")
    ap.add_argument("--chunk", type=int, default=100000)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--scenes", default="test", choices=["dev", "test"])
    ap.add_argument("--a81", action="store_true",
                    help="A81 report: fixed-range hazard windows, score = -reach")
    ap.add_argument("--out", default="outputs/artifacts/a79_safety_downstream.json")
    a = ap.parse_args()
    cache = os.path.join(ROOT, a.cache)
    os.makedirs(os.path.dirname(cache), exist_ok=True)

    keep = _sp.dev_scenes() if a.scenes == "dev" else _sp.test_scenes()
    print(f"[A45] {a.scenes.upper()} scenes, {len(keep)} of 150, digest {_sp.digest()}",
          flush=True)

    if not a.report:
        gt_map = {os.path.basename(os.path.dirname(p)): p
                  for p in glob.glob(os.path.join(ROOT, a.gts, "*", "*",
                                                  "labels.npz"))}
        index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
        by_scene = {}
        for i, r in enumerate(index):
            by_scene.setdefault(r["scene"], []).append(i)
        for s in by_scene:
            by_scene[s].sort(key=lambda i: index[i]["timestamp"])
        speed = {}
        for s, ids in by_scene.items():
            for n, i in enumerate(ids):
                if n == 0:
                    speed[index[i]["token"]] = float("nan"); continue
                p0 = np.asarray(index[ids[n-1]]["cams"]["CAM_FRONT"]["ego2global_translation"])
                p1 = np.asarray(index[i]["cams"]["CAM_FRONT"]["ego2global_translation"])
                dt = (index[i]["timestamp"] - index[ids[n-1]]["timestamp"]) / 1e6
                speed[index[i]["token"]] = float(np.linalg.norm(p1 - p0) / max(dt, 1e-3))
        rows = [(r["token"], r["scene"], speed[r["token"]]) for r in index
                if r["scene"] in keep
                and r["token"] in gt_map
                and os.path.exists(os.path.join(ROOT, "data/preds/preds_voxel",
                                                r["token"] + ".npz"))
                and os.path.exists(os.path.join(ROOT, a.obs, r["token"] + ".npy"))
                and not np.isnan(speed[r["token"]])]
        if a.frames:
            rows = rows[:a.frames]
        done = set()
        if os.path.exists(cache):
            done = {json.loads(l)["token"] for l in open(cache)}
        todo = [r for r in rows if r[0] not in done][:a.chunk]
        print(f"{len(rows)} test frames | cached {len(done)} | this run {len(todo)}",
              flush=True)
        if not todo:
            print("complete -- run --report"); return
        t0 = time.time()
        with open(cache, "a") as fh, Pool(
                max(os.cpu_count() - 1, 1), initializer=_init,
                initargs=(a.obs, gt_map, a.decel, HORIZONS)) as pool:
            for n, r in enumerate(pool.imap_unordered(_one, todo, chunksize=8)):
                if r is None:
                    continue
                fh.write(json.dumps(r) + "\n")
                if (n + 1) % 250 == 0:
                    fh.flush(); el = time.time() - t0
                    print(f"  {n+1}/{len(todo)}  {el/(n+1):.3f}s/frame  "
                          f"eta {(len(todo)-n-1)*el/(n+1)/60:.1f} min", flush=True)
        print("complete -- run --report", flush=True)
        return

    R = [json.loads(l) for l in open(cache)]
    R = [r for r in R if r["scene"] in keep]
    if a.a81:
        return report_a81(R, a, keep)
    sc = np.array([r["scene"] for r in R])
    y = np.array([r["hazard"] for r in R], bool)
    s_g = np.array([r["dstop"] - r["reach_gated"] for r in R])
    s_f = np.array([r["dstop"] - r["reach_free"] for r in R])
    print(f"\n{len(R):,} test frames | {len(set(sc))} scenes | "
          f"hazard rate {100*y.mean():.1f}%  ({int(y.sum()):,} frames)\n")

    BUDGETS = (0.05, 0.10, 0.20)
    arms = {"gated (tau=0.15)": s_g, "free-only (tau=0.0)": s_f}
    point = {k: {"auroc": auroc(v, y),
                 **{f"recall@{int(b*100)}%": recall_at(v, y, b) for b in BUDGETS}}
             for k, v in arms.items()}
    print(f"{'arm':<22}{'AUROC':>9}" + "".join(f"{f'rec@{int(b*100)}%':>11}"
                                               for b in BUDGETS))
    print("-" * 66)
    for k, d in point.items():
        print(f"{k:<22}{d['auroc']:>9.4f}" +
              "".join(f"{d[f'recall@{int(b*100)}%']:>11.4f}" for b in BUDGETS))

    scenes = sorted(set(sc))
    idx = {s: np.where(sc == s)[0] for s in scenes}
    rs = np.random.default_rng(a.seed)
    draws = rs.integers(0, len(scenes), size=(a.boot, len(scenes)))
    keys = ["auroc"] + [f"recall@{int(b*100)}%" for b in BUDGETS]
    boot = {k: {m: np.empty(a.boot) for m in keys} for k in arms}
    for bi, d in enumerate(draws):
        sel = np.concatenate([idx[scenes[i]] for i in d])
        yy = y[sel]
        for k, v in arms.items():
            vv = v[sel]
            boot[k]["auroc"][bi] = auroc(vv, yy)
            for b in BUDGETS:
                boot[k][f"recall@{int(b*100)}%"][bi] = recall_at(vv, yy, b)

    def ci(x):
        x = x[~np.isnan(x)]
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    print("\n--- paired bootstrap over TEST scenes: gated - free-only ---")
    out = {"frames": len(R), "scenes": len(scenes), "boot": a.boot,
           "hazard_rate": float(y.mean()), "decel": a.decel,
           "point": point, "contrasts": {}}
    for m in keys:
        dd = boot["gated (tau=0.15)"][m] - boot["free-only (tau=0.0)"][m]
        lo, hi = ci(dd)
        pt = point["gated (tau=0.15)"][m] - point["free-only (tau=0.0)"][m]
        tag = "excludes 0" if (lo > 0 or hi < 0) else "SPANS 0"
        print(f"  {m:<14}{pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {tag}")
        out["contrasts"][m] = [pt, lo, hi]

    d10, (lo10, hi10) = out["contrasts"]["recall@10%"][0], \
        (out["contrasts"]["recall@10%"][1], out["contrasts"]["recall@10%"][2])
    ok = (d10 >= 0.05) and (lo10 > 0 or hi10 < 0)
    print("\n--- A79 acceptance ---")
    print(f"  recall@10% gain >= +0.05   {d10:+.4f}   {'yes' if d10 >= 0.05 else 'NO'}")
    print(f"  interval excludes zero     [{lo10:+.4f}, {hi10:+.4f}]   "
          f"{'yes' if (lo10 > 0 or hi10 < 0) else 'NO'}")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    out["verdict"] = "PASS" if ok else "FAIL"
    if a.out:
        json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
