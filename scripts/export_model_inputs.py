#!/usr/bin/env python3
"""Export model-ready tensors so the trained checkpoint can be run elsewhere.

The dataset class needs nuscenes-devkit and PyTorch, neither installable on
this machine. Everything it does is reproducible from the raw tables through
odfm_tables, so this writes exactly the arrays the model consumes -- resized
uint8 images, ego deltas, motion, labels -- and the inference runs where torch
is available. Matches NuScenesMiniTemporal: 90x160 bilinear, ToTensor scaling
only (no normalisation), CAMS order, oldest-frame-first, T=4.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
import odfm_tables as T  # noqa: E402

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
HW = (90, 160)
NF = 4


def ego_delta(tab, cur_tok, prev_tok):
    """[dx, dy, dyaw] taking points from prev ego frame into cur ego frame."""
    a = tab.sensor_record(cur_tok, "LIDAR_TOP")
    b = tab.sensor_record(prev_tok, "LIDAR_TOP")
    Rc = T.quat_to_rot(a["ego_rot"]); tc = np.asarray(a["ego_trans"], float)
    tp = np.asarray(b["ego_trans"], float)
    disp = Rc.T @ (tp - tc)
    dyaw = T.yaw_from_quat(b["ego_rot"]) - T.yaw_from_quat(a["ego_rot"])
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
    return np.array([disp[0], disp[1], dyaw], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", default=str(ROOT / "outputs/console/model_inputs.npz"))
    a = ap.parse_args()

    tab = T.tables()
    labels = ROOT / "outputs/artifacts/nuscenes_labels_128"
    toks = [t for toks in tab.scene_samples.values() for t in toks
            if (labels / f"{t}.npz").exists()]
    toks = toks[: a.n]

    X, ED, MO, TR, TREL, OCC, IDS = [], [], [], [], [], [], []
    for tok in toks:
        order = tab.scene_samples[tab.sample[tok]["scene_token"]]
        i = order.index(tok)
        chain = [tok]
        for _ in range(NF - 1):
            j = order.index(chain[-1])
            chain.append(order[j - 1] if j > 0 else chain[-1])
        chain = list(reversed(chain))          # oldest first, current last

        per_cam = []
        for cam in CAMS:
            frames = []
            for s in chain:
                rec = tab.sensor_record(s, cam)
                im = Image.open(rec["path"]).convert("RGB").resize(
                    (HW[1], HW[0]), Image.BILINEAR)
                frames.append(np.asarray(im, np.uint8))
            per_cam.append(np.stack(frames))    # (T, H, W, 3)
        X.append(np.stack(per_cam))             # (V, T, H, W, 3)

        ED.append(np.stack([ego_delta(tab, tok, s) for s in chain[:-1]]))
        z = np.load(labels / f"{tok}.npz")
        vx = z["vxy_prev"].astype(np.float32)
        MO.append(np.array([float(z["dt_prev"]), vx[0], vx[1]], np.float32))
        TR.append(z["traj"].astype(np.float32))
        TREL.append(z["t_rel"].astype(np.float32))
        OCC.append(z["occ"].astype(np.uint8))
        IDS.append(tok)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=np.stack(X), ego_deltas=np.stack(ED),
                        motion=np.stack(MO), traj=np.stack(TR),
                        t_rel=np.stack(TREL), occ=np.stack(OCC),
                        tokens=np.array(IDS))
    print(f"wrote {out.name}  {len(X)} samples  x{np.stack(X).shape}  "
          f"{out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
