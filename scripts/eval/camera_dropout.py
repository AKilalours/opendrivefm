#!/usr/bin/env python3
"""What does the ego lose when one camera fails?

This is a COVERAGE study, not a robustness study. The model is not re-run with
a camera removed -- there is one published FB-OCC checkpoint and its
predictions are fixed. What changes is the observability field: the noisy-OR is
recomputed over the five surviving cameras, so every voxel whose only camera
evidence came from the failed unit drops to observability exactly zero.

That answers a deployment question the benchmark cannot:

    if CAM_X dies mid-drive, which part of the scene is the stack now
    asserting with no camera evidence at all, and how good was the model
    in exactly that region while the camera still worked?

The second half matters. A region that goes dark but where the model was
already weak is a different engineering problem from a region that goes dark
where the model was carrying the drive.

Cost is one ray march per camera per frame -- the same work the full build
already does -- because the per-camera factors are kept instead of being
collapsed immediately. All seven configurations come out of one pass.

Run in chunks (`--chunk`) and re-run; finished frames are cached and skipped.
`--report` aggregates the cache and prints the table.
"""
from __future__ import annotations
import argparse, glob, importlib.util, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location(
    "build_obs_ray", os.path.join(HERE, "build_observability_ray.py"))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)

FREE = 17
OBST = np.arange(0, 11)
CAMS = B.CAMS
K1 = ["ev", "dark", "dark_ok", "keep", "keep_ok",
      "obst", "obst_dark", "obst_dark_ok", "obst_keep", "obst_keep_ok"]


def frame_counts(tok, scene, row, calib, gt_path, vox):
    g = np.load(gt_path)
    gcls = g["semantics"].astype(np.int16)
    mcam = g["mask_camera"].astype(bool)
    occ_flat = (gcls.reshape(-1) != FREE)
    pcls = np.load(os.path.join(ROOT, "data/preds/preds_voxel",
                               tok + ".npz"))["cls"].astype(np.int16)

    fac = {}
    for c in CAMS:
        cam = calib[tok]["cams"].get(c)
        if cam is None:
            fac[c] = np.zeros(vox.shape[0], np.float32); continue
        W, H = cam.get("width", 1600), cam.get("height", 900)
        u, v, z, good, K, ct = B.project(vox, cam)
        inside = good & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not inside.any():
            fac[c] = np.zeros(vox.shape[0], np.float32); continue
        vis = B.march_visibility(occ_flat, cam, W, H)
        area = (K[0, 0] * K[1, 1] * B.OCC_RES ** 2) / np.maximum(z, 0.1) ** 2
        cov = np.clip(area / B.FULL_PX, 0.0, 1.0)
        fac[c] = (B.TRUST * cov * (vis & inside)).astype(np.float32)

    miss_all = np.ones(vox.shape[0], np.float32)
    for c in CAMS:
        miss_all *= (1.0 - fac[c])
    shape = gcls.shape
    obs_full = (1.0 - miss_all).reshape(shape)

    ev = mcam & (gcls != FREE) & (obs_full > 0)
    right = (pcls == gcls)
    obst_m = np.isin(gcls, OBST)
    out = {}
    for c in CAMS:
        miss_c = miss_all / np.maximum(1.0 - fac[c], 1e-6)
        obs_c = (1.0 - miss_c).reshape(shape)
        dark = ev & (obs_c <= 1e-6)
        keep = ev & ~dark
        o_ev, o_d, o_k = ev & obst_m, dark & obst_m, keep & obst_m
        out[c] = [int(ev.sum()), int(dark.sum()), int(right[dark].sum()),
                  int(keep.sum()), int(right[keep].sum()), int(o_ev.sum()),
                  int(o_d.sum()), int(right[o_d].sum()),
                  int(o_k.sum()), int(right[o_k].sum())]
    return out


def report(cache, boot, seed, out_json):
    recs = [json.loads(l) for l in open(cache)]
    scenes = sorted({r["scene"] for r in recs})
    sidx = {s: i for i, s in enumerate(scenes)}
    per = {c: np.zeros((len(scenes), len(K1)), np.float64) for c in CAMS}
    for r in recs:
        for c in CAMS:
            per[c][sidx[r["scene"]]] += np.asarray(r["counts"][c], np.float64)

    rng = np.random.default_rng(seed)
    rep = {}
    print(f"\n{len(recs)} frames, {len(scenes)} scenes, "
          f"paired scene bootstrap x{boot}")
    print("=" * 100)
    print(f"{'camera failed':<18}{'field dark':>13}{'obstacles dark':>16}"
          f"{'acc where dark':>16}{'acc where kept':>16}{'gap (kept - dark)':>21}")
    print("-" * 100)
    for c in CAMS:
        Mx = per[c]; S = Mx.sum(0)
        pd = 100 * S[1] / max(S[0], 1)
        po = 100 * S[6] / max(S[5], 1)
        ad = S[2] / max(S[1], 1); ak = S[4] / max(S[3], 1)
        g = []
        for _ in range(boot):
            T = Mx[rng.integers(0, len(scenes), len(scenes))].sum(0)
            if T[1] > 0 and T[3] > 0:
                g.append(T[4] / T[3] - T[2] / T[1])
        lo, hi = np.percentile(g, [2.5, 97.5]) if g else (np.nan, np.nan)
        rep[c] = dict(totals={k: float(v) for k, v in zip(K1, S)},
                      pct_dark=pd, pct_obst_dark=po, acc_dark=ad, acc_keep=ak,
                      gap=ak - ad, gap_ci=[float(lo), float(hi)])
        print(f"{c:<18}{pd:>12.2f}%{po:>15.2f}%{100*ad:>15.1f}%{100*ak:>15.1f}%"
              f"{100*(ak-ad):>10.1f} [{100*lo:>5.1f},{100*hi:>5.1f}]")
    print("=" * 100)
    os.makedirs(os.path.join(ROOT, os.path.dirname(out_json)), exist_ok=True)
    json.dump(dict(frames=len(recs), scenes=len(scenes), boot=boot, seed=seed,
                   per_camera=rep,
                   note="coverage study; the model was NOT re-run without the camera"),
              open(os.path.join(ROOT, out_json), "w"), indent=1)
    print("wrote", out_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=450, help="target sample size")
    ap.add_argument("--chunk", type=int, default=110, help="frames this invocation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--cache", default="outputs/artifacts/dropout_cache.jsonl")
    ap.add_argument("--out", default="outputs/artifacts/camera_dropout.json")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    cache = os.path.join(ROOT, a.cache)
    os.makedirs(os.path.dirname(cache), exist_ok=True)

    if a.report:
        return report(cache, a.boot, a.seed, a.out)

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    calib = json.load(open(os.path.join(ROOT, "data/pack/calib.json")))
    gt_map = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}
    rows = [r for r in index if r["token"] in gt_map and r["token"] in calib
            and os.path.exists(os.path.join(ROOT, "data/preds/preds_voxel",
                                            r["token"] + ".npz"))]
    rng = np.random.default_rng(a.seed)
    if a.frames and a.frames < len(rows):
        rows = [rows[i] for i in sorted(rng.choice(len(rows), a.frames, replace=False))]

    done = set()
    if os.path.exists(cache):
        done = {json.loads(l)["token"] for l in open(cache)}
    todo = [r for r in rows if r["token"] not in done][:a.chunk]
    print(f"sample {len(rows)} | cached {len(done)} | this run {len(todo)}", flush=True)

    vox = B.voxel_centres()
    t0 = time.time()
    with open(cache, "a") as fh:
        for n, r in enumerate(todo):
            cnt = frame_counts(r["token"], r["scene"], r, calib,
                               gt_map[r["token"]], vox)
            fh.write(json.dumps(dict(token=r["token"], scene=r["scene"],
                                     counts=cnt)) + "\n")
            fh.flush()
            if (n + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  {n+1}/{len(todo)}  {el/(n+1):.2f}s/frame", flush=True)
    left = len(rows) - len(done) - len(todo)
    print(f"chunk done. {left} frames remain." if left > 0 else
          "sample complete. run with --report", flush=True)


if __name__ == "__main__":
    main()
