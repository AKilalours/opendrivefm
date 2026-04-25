"""
build_nuscenes_labels_3class.py — fixed global_to_ego shape bug
Copy to ~/opendrivefm/scripts/build_nuscenes_labels_3class.py
Run:
  python scripts/build_nuscenes_labels_3class.py \
    --data_root data/nuscenes \
    --manifest artifacts/nuscenes_mini_manifest.jsonl \
    --out_dir artifacts/nuscenes_labels_3class \
    --bev 128 --extent_m 20.0
"""
import json, argparse
from pathlib import Path
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from scipy.ndimage import binary_dilation

VEHICLE_CATS = {
    "vehicle.car","vehicle.truck","vehicle.bus.bendy","vehicle.bus.rigid",
    "vehicle.trailer","vehicle.construction","vehicle.motorcycle",
    "vehicle.bicycle","vehicle.emergency.ambulance","vehicle.emergency.police",
}
PEDESTRIAN_CATS = {
    "human.pedestrian.adult","human.pedestrian.child",
    "human.pedestrian.wheelchair","human.pedestrian.stroller",
    "human.pedestrian.personal_mobility","human.pedestrian.police_officer",
    "human.pedestrian.construction_worker",
}

def _lidar_sd(nusc, sample):
    return nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

def _dilate(mask, r):
    if r <= 0: return mask
    struct = np.ones((2*r+1, 2*r+1), dtype=bool)
    return binary_dilation(mask, structure=struct).astype(np.uint8)

def build_3class_occ(nusc, sample, bev=128, extent_m=20.0,
                     z_min=-1.5, z_max=3.0, dilate_r=1):
    occ_veh = np.zeros((bev, bev), dtype=np.uint8)
    occ_ped = np.zeros((bev, bev), dtype=np.uint8)

    sd    = _lidar_sd(nusc, sample)
    ep    = nusc.get("ego_pose", sd["ego_pose_token"])
    R_e2g = Quaternion(ep["rotation"]).rotation_matrix  # (3,3)
    t_e2g = np.array(ep["translation"])                  # (3,)

    def global_to_ego(pt_g):
        """pt_g: (3,) → (3,) in ego frame"""
        return R_e2g.T @ (pt_g - t_e2g)

    s = (2.0 * extent_m) / bev

    for ann_tok in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        cat = ann["category_name"]

        if cat in VEHICLE_CATS:       target = occ_veh
        elif cat in PEDESTRIAN_CATS:  target = occ_ped
        else:                         continue

        centre_g = np.array(ann["translation"])          # (3,)
        centre_e = global_to_ego(centre_g)               # (3,)
        cx, cy   = centre_e[0], centre_e[1]

        # Half-sizes
        w = ann["size"][0] / 2.0
        l = ann["size"][1] / 2.0

        # Yaw in ego frame
        R_box = Quaternion(ann["rotation"]).rotation_matrix
        R_ego = R_e2g.T @ R_box
        yaw   = np.arctan2(R_ego[1, 0], R_ego[0, 0])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        corners_x = np.array([ l*cos_y - w*sin_y,
                               -l*cos_y - w*sin_y,
                               -l*cos_y + w*sin_y,
                                l*cos_y + w*sin_y]) + cx
        corners_y = np.array([ l*sin_y + w*cos_y,
                               -l*sin_y + w*cos_y,
                               -l*sin_y - w*cos_y,
                                l*sin_y - w*cos_y]) + cy

        r_min = int(np.clip((extent_m - corners_x.max()) / s, 0, bev-1))
        r_max = int(np.clip((extent_m - corners_x.min()) / s, 0, bev-1))
        c_min = int(np.clip((corners_y.min() + extent_m) / s, 0, bev-1))
        c_max = int(np.clip((corners_y.max() + extent_m) / s, 0, bev-1))
        if r_min > r_max: r_min, r_max = r_max, r_min
        if c_min > c_max: c_min, c_max = c_max, c_min
        target[r_min:r_max+1, c_min:c_max+1] = 1

    occ_veh = _dilate(occ_veh, dilate_r)
    occ_ped = _dilate(occ_ped, dilate_r)
    occupied = np.maximum(occ_veh, occ_ped)
    free     = (1 - occupied).astype(np.uint8)
    return np.stack([free, occ_veh, occ_ped], axis=0).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",  default="data/nuscenes")
    ap.add_argument("--version",    default="v1.0-mini")
    ap.add_argument("--manifest",   required=True)
    ap.add_argument("--out_dir",    default="artifacts/nuscenes_labels_3class")
    ap.add_argument("--bev",        type=int,   default=128)
    ap.add_argument("--extent_m",   type=float, default=20.0)
    ap.add_argument("--dilate",     type=int,   default=1)
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    veh_counts, ped_counts = [], []
    for i, r in enumerate(rows):
        sample = nusc.get("sample", r["sample_token"])
        occ    = build_3class_occ(nusc, sample,
                                  bev=args.bev,
                                  extent_m=args.extent_m,
                                  dilate_r=args.dilate)

        old = np.load(f"artifacts/nuscenes_labels/{r['sample_token']}.npz")
        np.savez_compressed(
            out / f"{r['sample_token']}.npz",
            occ      = occ,
            traj     = old["traj"],
            t_rel    = old["t_rel"]    if "t_rel"    in old.files else np.arange(1,13)*0.5,
            dt_prev  = old["dt_prev"]  if "dt_prev"  in old.files else np.float32(0.0),
            vxy_prev = old["vxy_prev"] if "vxy_prev" in old.files else np.zeros(2,np.float32),
        )
        veh_counts.append(occ[1].mean())
        ped_counts.append(occ[2].mean())

        if (i+1) % 50 == 0 or (i+1) == len(rows):
            print(f"  {i+1}/{len(rows)}  veh={np.mean(veh_counts):.4f}  ped={np.mean(ped_counts):.4f}")

    print(f"\nDone. Shape: {occ.shape}")
    print(f"Mean vehicle occupancy:    {np.mean(veh_counts):.4f}")
    print(f"Mean pedestrian occupancy: {np.mean(ped_counts):.4f}")

if __name__ == "__main__":
    main()
