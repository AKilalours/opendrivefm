"""
Regenerate BEV occupancy labels as FOREGROUND OBJECTS (vehicles, pedestrians, cyclists)
instead of drivable surface.

This replaces the existing nuscenes_labels/*.npz files.
Old labels: ~79.7% occupied (drivable surface) — degenerate, model predicts "all occupied"
New labels:  ~5-15% occupied (objects only) — sparse, requires real learning

Usage:
    python scripts/regen_labels_objects.py \
        --dataroot /path/to/nuscenes \
        --version v1.0-mini \
        --manifest artifacts/nuscenes_mini_manifest.jsonl \
        --label_root artifacts/nuscenes_labels \
        --bev_size 64 \
        --bev_range 50.0

Run from ~/opendrivefm/
"""

import argparse, json, os, numpy as np
from pathlib import Path
from tqdm import tqdm

# Object categories to treat as foreground
FOREGROUND_CATS = {
    'vehicle.car',
    'vehicle.truck',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.motorcycle',
    'vehicle.bicycle',
    'vehicle.trailer',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.pushable_pullable',
}


def box_to_bev_mask(center_x, center_y, wl, ww, yaw, bev_size, bev_range):
    """
    Rasterise a single 3D bounding box into a BEV binary mask.
    center_x, center_y: box center in ego frame (metres)
    wl, ww: box length and width (metres)
    yaw: rotation in radians
    bev_size: output grid size (e.g. 64)
    bev_range: half-range of BEV (e.g. 50.0 means -50 to +50 metres)
    """
    mask = np.zeros((bev_size, bev_size), dtype=np.float32)

    # BEV pixel size in metres
    res = (2.0 * bev_range) / bev_size   # metres per pixel

    # Box corners in local frame (half-extents)
    half_l = wl / 2.0
    half_w = ww / 2.0
    corners_local = np.array([
        [ half_l,  half_w],
        [ half_l, -half_w],
        [-half_l, -half_w],
        [-half_l,  half_w],
    ])

    # Rotate to ego frame
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners_ego = (R @ corners_local.T).T + np.array([center_x, center_y])

    # Convert to pixel coordinates
    # BEV convention: x=forward, y=left
    # Grid: row 0 = forward-most, col 0 = leftmost
    def ego_to_pixel(ex, ey):
        px = (bev_range - ex) / res   # forward = smaller row index
        py = (bev_range - ey) / res   # left    = smaller col index
        return px, py

    corners_px = np.array([ego_to_pixel(cx, cy) for cx, cy in corners_ego])

    # Rasterise polygon using scanline fill
    from PIL import Image, ImageDraw
    img = Image.new('L', (bev_size, bev_size), 0)
    draw = ImageDraw.Draw(img)
    poly = [(float(py), float(px)) for px, py in corners_px]  # (col, row)
    draw.polygon(poly, fill=1)
    mask = np.array(img, dtype=np.float32)
    return mask


def get_category_name(nusc, ann_token):
    ann = nusc.get('sample_annotation', ann_token)
    inst = nusc.get('instance', ann['instance_token'])
    cat = nusc.get('category', inst['category_token'])
    return cat['name']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',   required=True,
                        help='Path to nuScenes dataset root (contains v1.0-mini/)')
    parser.add_argument('--version',    default='v1.0-mini')
    parser.add_argument('--manifest',   default='artifacts/nuscenes_mini_manifest.jsonl')
    parser.add_argument('--label_root', default='artifacts/nuscenes_labels')
    parser.add_argument('--bev_size',   type=int,   default=64)
    parser.add_argument('--bev_range',  type=float, default=50.0,
                        help='Half-range in metres: BEV covers [-bev_range, +bev_range]')
    parser.add_argument('--dry_run',    action='store_true',
                        help='Print stats without writing files')
    args = parser.parse_args()

    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import Box
        from pyquaternion import Quaternion
    except ImportError:
        print("ERROR: nuscenes-devkit not installed.")
        print("Install: pip install nuscenes-devkit pyquaternion")
        return

    print(f"Loading nuScenes {args.version} from {args.dataroot} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Load manifest to get sample tokens
    rows = [json.loads(l) for l in open(args.manifest)]
    sample_tokens = [r['sample_token'] for r in rows]
    print(f"Manifest samples: {len(sample_tokens)}")

    label_root = Path(args.label_root)
    label_root.mkdir(parents=True, exist_ok=True)

    occupancy_rates = []
    skipped = 0

    for row in tqdm(rows, desc="Generating object labels"):
        token = row['sample_token']
        npz_path = label_root / f"{token}.npz"

        # Load existing npz to preserve traj/motion/t_rel fields
        if not npz_path.exists():
            skipped += 1
            continue

        existing = dict(np.load(str(npz_path), allow_pickle=True))

        # Get sample
        sample = nusc.get('sample', token)

        # Get ego pose from CAM_FRONT (all cams share the same timestamp approx)
        cam_token = sample['data']['CAM_FRONT']
        cam_data  = nusc.get('sample_data', cam_token)
        ego_pose  = nusc.get('ego_pose', cam_data['ego_pose_token'])

        ego_translation = np.array(ego_pose['translation'])
        ego_rotation    = Quaternion(ego_pose['rotation'])

        # Build BEV mask for all foreground annotations
        bev_mask = np.zeros((args.bev_size, args.bev_size), dtype=np.float32)

        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            inst = nusc.get('instance', ann['instance_token'])
            cat  = nusc.get('category', inst['category_token'])
            cat_name = cat['name']

            # Check if foreground
            is_fg = any(cat_name.startswith(fg) for fg in FOREGROUND_CATS)
            if not is_fg:
                continue

            # Box in global frame
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))

            # Transform to ego frame
            box.translate(-ego_translation)
            box.rotate(ego_rotation.inverse)

            cx, cy = box.center[0], box.center[1]
            wl, ww = box.wlh[1], box.wlh[0]   # length, width
            yaw = box.orientation.yaw_pitch_roll[0]

            # Skip boxes outside BEV range
            if abs(cx) > args.bev_range + 5 or abs(cy) > args.bev_range + 5:
                continue

            obj_mask = box_to_bev_mask(
                cx, cy, wl, ww, yaw,
                args.bev_size, args.bev_range)
            bev_mask = np.clip(bev_mask + obj_mask, 0, 1)

        occ_rate = bev_mask.mean()
        occupancy_rates.append(occ_rate)

        if not args.dry_run:
            # Replace occ field, keep everything else
            existing['occ'] = bev_mask
            np.savez_compressed(str(npz_path), **existing)

    print(f"\n── Label Stats ──────────────────────────────")
    print(f"  Processed:        {len(occupancy_rates)}")
    print(f"  Skipped:          {skipped}")
    print(f"  Mean occupancy:   {np.mean(occupancy_rates)*100:.1f}%")
    print(f"  Median occupancy: {np.median(occupancy_rates)*100:.1f}%")
    print(f"  % empty frames:   {(np.array(occupancy_rates)==0).mean()*100:.1f}%")
    print(f"  % <5% occupied:   {(np.array(occupancy_rates)<0.05).mean()*100:.1f}%")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        print(f"\nLabels saved to {args.label_root}")
        print("Next: retrain with --max_epochs 50")


if __name__ == '__main__':
    main()
