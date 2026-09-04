#!/usr/bin/env python3
"""Export model inputs for EVERY keyframe, deduplicated.

The per-sample exporter wrote each temporal window whole: 6 cameras x 4 frames
per sample. Consecutive keyframes overlap, so that stores every image four
times over -- 63 MB for 80 samples, and 320 MB for all 404, which does not move
across the device bridge in one piece.

This writes each (sample, camera) image ONCE plus the chain of sample indices
that forms each window, and the consumer rebuilds the windows. Same tensors,
a quarter of the bytes.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
import odfm_tables as T  # noqa: E402

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
HW, NF = (90, 160), 4


def ego_delta(tab, cur, prev):
    a = tab.sensor_record(cur, "LIDAR_TOP"); b = tab.sensor_record(prev, "LIDAR_TOP")
    Rc = T.quat_to_rot(a["ego_rot"]); tc = np.asarray(a["ego_trans"], float)
    d = Rc.T @ (np.asarray(b["ego_trans"], float) - tc)
    yaw = T.yaw_from_quat(b["ego_rot"]) - T.yaw_from_quat(a["ego_rot"])
    return np.array([d[0], d[1], (yaw + np.pi) % (2*np.pi) - np.pi], np.float32)


def main():
    tab = T.tables()
    labels = ROOT / "outputs/artifacts/nuscenes_labels_128"
    toks, order_of = [], {}
    for stoks in tab.scene_samples.values():
        for t in stoks:
            if (labels / f"{t}.npz").exists():
                order_of[t] = len(toks); toks.append(t)
    N = len(toks)
    print(f"{N} keyframes with labels")

    IMG = np.zeros((N, 6, HW[0], HW[1], 3), np.uint8)
    CHAIN = np.zeros((N, NF), np.int32)
    ED = np.zeros((N, NF - 1, 3), np.float32)
    MO = np.zeros((N, 3), np.float32); TR = np.zeros((N, 12, 2), np.float32)
    TREL = np.zeros((N, 12), np.float32); OCC = np.zeros((N, 1, 128, 128), np.uint8)

    for i, tok in enumerate(toks):
        for c, cam in enumerate(CAMS):
            rec = tab.sensor_record(tok, cam)
            IMG[i, c] = np.asarray(Image.open(rec["path"]).convert("RGB")
                                   .resize((HW[1], HW[0]), Image.BILINEAR), np.uint8)
        sc = tab.scene_samples[tab.sample[tok]["scene_token"]]
        j = sc.index(tok)
        chain = [tok]
        for _ in range(NF - 1):
            k = sc.index(chain[-1]); chain.append(sc[k - 1] if k > 0 else chain[-1])
        chain = list(reversed(chain))
        CHAIN[i] = [order_of.get(t, i) for t in chain]
        ED[i] = np.stack([ego_delta(tab, tok, t) for t in chain[:-1]])
        z = np.load(labels / f"{tok}.npz")
        v = z["vxy_prev"].astype(np.float32)
        MO[i] = [float(z["dt_prev"]), v[0], v[1]]
        TR[i] = z["traj"]; TREL[i] = z["t_rel"]; OCC[i] = z["occ"] > 0.5
        if i % 100 == 0:
            print(f"  {i}/{N}", flush=True)

    out = ROOT / "outputs/console/all_frames.npz"
    np.savez_compressed(out, img=IMG, chain=CHAIN, ego_deltas=ED, motion=MO,
                        traj=TR, t_rel=TREL, occ=OCC, tokens=np.array(toks))
    print(f"wrote {out.name}  {out.stat().st_size/1e6:.1f} MB  img{IMG.shape}")


if __name__ == "__main__":
    main()
