#!/usr/bin/env python3
"""Measured support and error-vs-observability, for Figure 4.

Over a fixed sample of sealed-test frames:
  (1) the value distribution of mask_camera and of O(v;G) on the same voxels
  (2) the per-voxel error rate by observability decile, restricted to
      mask_camera = 1, with a paired-frame bootstrap interval per decile.

Writes outputs/artifacts/maskvsobs.json. No number is typed by hand.
"""
import glob, importlib.util, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NFR, SEED, BOOT = 60, 20261108, 400
EDGES = np.linspace(0, 1, 11)


def main():
    sp = importlib.util.spec_from_file_location(
        "sp", os.path.join(ROOT, "scripts/eval/split.py"))
    S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
    test = set(S.test_scenes())
    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    rows = [r for r in index if r["scene"] in test]
    rng = np.random.default_rng(SEED)
    rows = [rows[i] for i in rng.permutation(len(rows))]

    mk_vals, o_vals = [], []
    per_frame = []            # (hits, counts) per decile, one row per frame
    used = []
    for r in rows:
        if len(used) >= NFR:
            break
        t = r["token"]
        g = glob.glob(os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval",
                                   "gts", "*", t, "labels.npz"))
        pp = os.path.join(ROOT, "data/preds/preds_voxel", t + ".npz")
        op = os.path.join(ROOT, "data/pack/obs_max", t + ".npy")
        if not g or not os.path.exists(pp) or not os.path.exists(op):
            continue
        z = np.load(g[0])
        if "mask_camera" not in z:
            continue
        gc = z["semantics"].reshape(-1)
        mk = z["mask_camera"].reshape(-1).astype(bool)
        pc = np.load(pp)["cls"].reshape(-1)
        o = np.load(op).reshape(-1).astype(np.float32) / 255.
        if mk.sum() < 5000:
            continue
        sel = rng.random(len(o)) < 0.05          # thin for the histograms only
        mk_vals.append(mk[sel].astype(np.float32)); o_vals.append(o[sel])
        err = (pc[mk] != gc[mk])
        b = np.clip(np.digitize(o[mk], EDGES) - 1, 0, 9)
        hit = np.bincount(b, weights=err, minlength=10)
        cnt = np.bincount(b, minlength=10)
        per_frame.append((hit, cnt))
        used.append(t)

    H = np.array([h for h, _ in per_frame]); C = np.array([c for _, c in per_frame])
    rate = H.sum(0) / np.maximum(C.sum(0), 1)
    br = np.empty((BOOT, 10))
    for b in range(BOOT):
        i = rng.integers(0, len(H), len(H))
        br[b] = H[i].sum(0) / np.maximum(C[i].sum(0), 1)
    lo, hi = np.percentile(br, [2.5, 97.5], axis=0)

    rev = int(sum(1 for k in range(9) if rate[k + 1] > rate[k]))
    out = dict(frames=len(used), seed=SEED, boot=BOOT,
               decile_centres=[round(x, 2) for x in (EDGES[:-1] + 0.05)],
               error_rate=[float(x) for x in rate],
               ci_lo=[float(x) for x in lo], ci_hi=[float(x) for x in hi],
               n_per_decile=[int(x) for x in C.sum(0)],
               reversals=rev,
               hist_mask=np.concatenate(mk_vals).tolist()[:0],   # not stored
               mask_frac_one=float(np.concatenate(mk_vals).mean()),
               obs_hist_counts=np.histogram(np.concatenate(o_vals),
                                            bins=50, range=(0, 1))[0].tolist(),
               obs_n=int(len(np.concatenate(o_vals))))
    del out["hist_mask"]
    json.dump(out, open(os.path.join(ROOT, "outputs/artifacts/maskvsobs.json"), "w"),
              indent=2)
    print("frames", len(used), "reversals", rev)
    print("rate", np.round(rate, 4).tolist())


if __name__ == "__main__":
    main()
