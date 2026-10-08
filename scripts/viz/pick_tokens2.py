#!/usr/bin/env python3
"""Pick a distinct sealed-test token for Figures 3, 4 and 5, by stated rules.

Fig 3 (qualitative): needs a camera image, so the pool is test tokens whose
  rig is mirrored locally. Rule: the frame containing the largest strictly
  missed dynamic object (model predicts free throughout the box) whose
  maximum observability is below 0.05. Largest = most legible in print.
Fig 4 (mask_camera vs O): no image needed, full test pool. Rule: maximise
  the standard deviation of O inside mask_camera = 1, i.e. the frame where
  the binary mask hides the most variation.
Fig 5 (deployable occluder): full test pool with a prediction on disk.
  Rule: maximise occupied-fraction gap between predicted and GT occupancy,
  the quantity A55b identifies as the mechanism.
"""
import glob, importlib.util, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(ROOT, "outputs/artifacts/figure_tokens.json")
CAP = 220          # scan budget per rule, keeps a device_bash call inside its limit
FREE = 17


def gts(tok):
    g = glob.glob(os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval",
                               "gts", "*", tok, "labels.npz"))
    return g[0] if g else None


def main():
    cur = json.load(open(P))
    sp = importlib.util.spec_from_file_location(
        "sp", os.path.join(ROOT, "scripts/eval/split.py"))
    S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
    test = set(S.test_scenes())
    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    used = {cur["fig1_token"], "dc03fe4e0a374592af7701c66d5276c4"}

    rows = [r for r in index if r["scene"] in test
            and os.path.exists(os.path.join(ROOT, "data/pack/obs_max",
                                            r["token"] + ".npy"))
            and r["token"] not in used]
    rng = np.random.default_rng(cur["seed"])
    scan = [rows[i] for i in rng.choice(len(rows), min(CAP, len(rows)), replace=False)]

    # ---- Fig 4
    best4 = None
    for r in scan:
        gp = gts(r["token"])
        if gp is None:
            continue
        z = np.load(gp)
        if "mask_camera" not in z:
            continue
        mk = z["mask_camera"].astype(bool)
        o = np.load(os.path.join(ROOT, "data/pack/obs_max",
                                 r["token"] + ".npy")).astype(np.float32) / 255.
        if mk.sum() < 5000:
            continue
        s = float(o[mk].std())
        if best4 is None or s > best4[1]:
            best4 = (r["token"], s)

    # ---- Fig 5
    best5 = None
    for r in scan:
        pp = os.path.join(ROOT, "data/preds/preds_voxel", r["token"] + ".npz")
        gp = gts(r["token"])
        if gp is None or not os.path.exists(pp):
            continue
        g = np.load(gp)["semantics"]
        p = np.load(pp)["cls"]
        gap = float((p != FREE).mean() - (g != FREE).mean())
        if best5 is None or gap > best5[1]:
            best5 = (r["token"], gap)

    cur.update(fig4_token=best4[0], fig4_obs_std_inside_mask=round(best4[1], 4),
               fig5_token=best5[0], fig5_pred_minus_gt_occupied=round(best5[1], 4),
               scan_cap=CAP,
               rules={"fig4": "max std of O inside mask_camera=1",
                      "fig5": "max predicted-minus-GT occupied fraction"})
    json.dump(cur, open(P, "w"), indent=2)
    print(json.dumps({k: cur[k] for k in cur if k.startswith("fig")}, indent=2))


if __name__ == "__main__":
    main()
