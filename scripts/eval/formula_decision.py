#!/usr/bin/env python3
"""Noisy-OR or single best camera? Settle the formula the ablation questioned.

A15 found that `max_cam` -- taking the single best camera instead of combining
all six with a noisy-OR -- scored +0.0007 ABOVE the full measure, with the
interval excluding zero. A paper whose own ablation argues against its own
formula is a paper a reviewer will take apart, so this decides it.

Five scores, all accumulated in ONE march per frame so they are perfectly
paired:

    full         1 - prod_i (1 - T * cov_i * vis_i)   the current measure
    max_cam      max_i (T * cov_i * vis_i)            noisy-OR removed
    max_noT      max_i (cov_i * vis_i)                and the TRUST constant removed
    max_novis    max_i (cov_i * in_frustum_i)         and occlusion removed (control)
    mask_cam     Occ3D's own binary flag              the baseline that must be beaten

`max_noT` exists to make a claim testable rather than asserted: under a max,
TRUST is a global monotone rescale, so if the formula is honest its AUROC must
be IDENTICAL to max_cam to the last digit. If it is not, something else in the
pipeline depends on the constant and the simplification is not free.

Same frame sample and seed as the frozen ablation (1234), so the numbers are
directly comparable to A15. Per-scene score histograms on the uint8 grid, so
the AUROCs are exact and the paired scene bootstrap is exact. Chunked and
cached: re-run until it reports complete, then --report.
"""
from __future__ import annotations
import argparse, glob, importlib.util, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location(
    "build_obs_ray", os.path.join(HERE, "build_observability_ray.py"))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)

NB = 256
VAR = ["full", "max_cam", "max_noT", "max_novis", "mask_cam"]


def auroc(pos, neg):
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    cneg = np.cumsum(neg) - neg
    return float((pos * (cneg + 0.5 * neg)).sum() / (P * N))


def frame_hist(tok, calib, gp, pp, vox):
    g = np.load(gp)
    sem = g["semantics"]
    mcam = g["mask_camera"].astype(bool).ravel()
    occ_flat = (sem.reshape(-1) != 17)
    wrong = (np.load(pp)["cls"] != sem).ravel()

    miss = np.ones(vox.shape[0], np.float32)
    best = np.zeros(vox.shape[0], np.float32)
    best_noT = np.zeros(vox.shape[0], np.float32)
    best_novis = np.zeros(vox.shape[0], np.float32)
    for c in B.CAMS:
        cam = calib[tok]["cams"].get(c)
        if cam is None:
            continue
        W, H_ = cam.get("width", 1600), cam.get("height", 900)
        u, v, z, good, K, ct = B.project(vox, cam)
        ins = good & (u >= 0) & (u < W) & (v >= 0) & (v < H_)
        if not ins.any():
            continue
        vis = B.march_visibility(occ_flat, cam, W, H_) & ins
        area = (K[0, 0] * K[1, 1] * B.OCC_RES ** 2) / np.maximum(z, 0.1) ** 2
        cov = np.clip(area / B.FULL_PX, 0.0, 1.0) * ins
        miss *= (1.0 - B.TRUST * cov * vis)
        best = np.maximum(best, B.TRUST * cov * vis)
        best_noT = np.maximum(best_noT, cov * vis)
        best_novis = np.maximum(best_novis, cov * ins)

    S = {"full": 1.0 - miss, "max_cam": best, "max_noT": best_noT,
         "max_novis": best_novis, "mask_cam": mcam.astype(np.float32)}
    out = np.zeros((len(VAR), 2, NB), np.int64)
    for i, k in enumerate(VAR):
        q = np.clip(np.rint((1.0 - S[k]) * (NB - 1)), 0, NB - 1).astype(np.uint8)
        out[i, 1] = np.bincount(q[wrong], minlength=NB)
        out[i, 0] = np.bincount(q[~wrong], minlength=NB)
    return out


def report(cdir, boot, seed, out_json):
    files = sorted(glob.glob(os.path.join(cdir, "*.npz")))
    per = {}
    for f in files:
        z = np.load(f)
        per.setdefault(str(z["scene"]), []).append(z["h"])
    scenes = sorted(per)
    Mx = np.stack([np.sum(per[s], 0) for s in scenes])        # (S, V, 2, NB)
    tot = Mx.sum(0)
    A = {k: auroc(tot[i, 1], tot[i, 0]) for i, k in enumerate(VAR)}

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(scenes), (boot, len(scenes)))
    Ab = np.zeros((boot, len(VAR)))
    for b in range(boot):
        T = Mx[draws[b]].sum(0)
        for i in range(len(VAR)):
            Ab[b, i] = auroc(T[i, 1], T[i, 0])

    print(f"\n{len(files)} frames, {len(scenes)} scenes, "
          f"paired scene bootstrap x{boot}")
    print("=" * 78)
    print(f"{'score':<14}{'AUROC':>10}{'vs mask_cam':>16}{'95% CI':>24}")
    print("-" * 78)
    im = VAR.index("mask_cam")
    res = {}
    for i, k in enumerate(VAR):
        d = Ab[:, i] - Ab[:, im]
        lo, hi = np.percentile(d, [2.5, 97.5])
        res[k] = dict(auroc=A[k], vs_mask=A[k] - A["mask_cam"],
                      ci=[float(lo), float(hi)])
        s = "" if k == "mask_cam" else f"{A[k]-A['mask_cam']:+.4f}"
        c = "" if k == "mask_cam" else f"[{lo:+.4f}, {hi:+.4f}]"
        print(f"{k:<14}{A[k]:>10.4f}{s:>16}{c:>24}")
    print("-" * 78)
    for a_, b_ in (("max_cam", "full"), ("max_noT", "max_cam"),
                   ("max_cam", "max_novis")):
        d = Ab[:, VAR.index(a_)] - Ab[:, VAR.index(b_)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{a_} - {b_:<12}{A[a_]-A[b_]:>+10.4f}"
              f"{'':>16}[{lo:+.4f}, {hi:+.4f}]")
        res[f"{a_}_minus_{b_}"] = dict(delta=A[a_] - A[b_], ci=[float(lo), float(hi)])
    print("=" * 78)
    json.dump(dict(frames=len(files), scenes=len(scenes), boot=boot,
                   auroc=A, contrasts=res), open(os.path.join(ROOT, out_json), "w"),
              indent=1)
    print("wrote", out_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cache", default="outputs/artifacts/formula_cache")
    ap.add_argument("--out", default="outputs/artifacts/formula_decision.json")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    cdir = os.path.join(ROOT, a.cache); os.makedirs(cdir, exist_ok=True)
    if a.report:
        return report(cdir, a.boot, 7, a.out)

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    calib = json.load(open(os.path.join(ROOT, "data/pack/calib.json")))
    gt_map = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}
    # same sample and seed as the frozen ablation A15
    rs = np.random.default_rng(a.seed)
    sel = sorted(rs.choice(len(index), min(a.frames, len(index)), replace=False))
    rows = [index[i] for i in sel]

    vox = B.voxel_centres()
    todo, t0 = [], time.time()
    for r in rows:
        tok = r["token"]
        if os.path.exists(os.path.join(cdir, tok + ".npz")):
            continue
        pp = os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz")
        if tok in gt_map and tok in calib and os.path.exists(pp):
            todo.append((r, pp))
    # A2's bug again: taking the head of an index-sorted sample walks scene by
    # scene, so a partial cache covers a handful of scenes and the scene
    # bootstrap is meaningless. Shuffle deterministically before slicing.
    np.random.default_rng(99).shuffle(todo)
    n_left = len(todo); todo = todo[:a.chunk]
    print(f"sample {len(rows)} | remaining {n_left} | this run {len(todo)}", flush=True)
    for n, (r, pp) in enumerate(todo):
        tok = r["token"]
        h = frame_hist(tok, calib, gt_map[tok], pp, vox)
        np.savez_compressed(os.path.join(cdir, tok + ".npz"),
                            h=h, scene=r["scene"])
        if (n + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(todo)}  {el/(n+1):.2f}s/frame", flush=True)
    left = n_left - len(todo)
    print(f"chunk done, {left} remain" if left > 0 else
          "sample complete -- run with --report", flush=True)


if __name__ == "__main__":
    main()
