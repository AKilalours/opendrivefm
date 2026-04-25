"""
Build nuScenes mini manifest with calibration data (K, T_ego_cam).
Each JSONL row includes:
  cams, intrinsics (K_v per camera), extrinsics (T_ego_cam_v per camera)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

CAMS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
        "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]

def build_T_ego_cam(nusc, sd):
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    R = Quaternion(cs["rotation"]).rotation_matrix
    t = np.array(cs["translation"], dtype=np.float32)
    T = np.eye(4, dtype=np.float32); T[:3,:3]=R; T[:3,3]=t
    return T

def build_K(nusc, sd):
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    return np.array(cs["camera_intrinsic"], dtype=np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data/nuscenes")
    p.add_argument("--version",   type=str, default="v1.0-mini")
    p.add_argument("--out",       type=str, default="artifacts/nuscenes_mini_manifest.jsonl")
    p.add_argument("--limit",     type=int, default=0)
    args = p.parse_args()

    from nuscenes.nuscenes import NuScenes
    data_root = Path(args.data_root)
    nusc = NuScenes(version=args.version, dataroot=str(data_root), verbose=False)
    out  = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene in tqdm(nusc.scene, desc="scenes"):
        tok = scene["first_sample_token"]
        while tok:
            sample = nusc.get("sample", tok)
            imgs, K_dict, T_dict = {}, {}, {}
            ok = True
            for cam in CAMS:
                sd_tok = sample["data"].get(cam)
                if not sd_tok: ok=False; break
                sd = nusc.get("sample_data", sd_tok)
                imgs[cam]  = str((data_root / sd["filename"]).as_posix())
                K_dict[cam]= build_K(nusc, sd).tolist()
                T_dict[cam]= build_T_ego_cam(nusc, sd).tolist()
            if ok:
                rows.append({"scene": scene["name"], "sample_token": tok,
                             "cams": imgs, "intrinsics": K_dict, "extrinsics": T_dict})
            tok = sample["next"]
            if args.limit and len(rows) >= args.limit: break
        if args.limit and len(rows) >= args.limit: break

    with out.open("w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    print(f"WROTE: {out}  rows= {len(rows)}  (includes K + T_ego_cam per camera)")

if __name__ == "__main__":
    main()
