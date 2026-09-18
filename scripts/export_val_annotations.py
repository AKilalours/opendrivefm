"""Export ground-truth boxes for the packed validation keyframes.

Why this exists
---------------
pack_nuscenes.py stores images, LiDAR and calibration -- everything needed to
COMPUTE an observability map. It does not store annotations, because the
observability map does not need them.

But two of the strongest planned experiments do:

  * "are missed detections concentrated in low-observability cells?" needs the
    ground-truth boxes to know what was there to be missed.
  * "which object created this blind spot?" needs the boxes to attribute an
    occluded region to the thing that occluded it.

The boxes live in v1.0-trainval/sample_annotation.json, which stays on the
volume after the blobs are deleted, so nothing has to be re-downloaded. This
script pulls out just the validation keyframes and writes a compact archive
small enough to keep on a laptop, which is where the rest of the analysis runs.

It also carries two fields that matter and are easy to lose:

  visibility   the human annotator's 0-40 / 40-60 / 60-80 / 80-100% bucket.
               This is the independent label the observability measure was
               validated against. It is not a model output and not derived
               from anything this project computes, which is exactly what
               makes it worth something. (The 0.954 that once appeared here
               was the VLLM's number, not observability's; the measured value
               on the full dev split is 0.686.)
  instance     the identity of the object across frames, so an occluder can be
               tracked rather than re-discovered each keyframe.

Boxes are stored in the GLOBAL frame as nuScenes provides them. Converting to
ego or LiDAR frame is one rotation and one translation using the pose already
in calib.json, and doing it here would bake in a choice the analysis might
want to make differently.

    python export_val_annotations.py --data /workspace/nuscenes \\
                                     --pack /workspace/pack
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main(data: str, pack: str) -> None:
    root = os.path.join(data, "v1.0-trainval")
    need = ["sample_annotation", "instance", "category", "visibility", "attribute"]
    meta = {n: json.load(open(os.path.join(root, f"{n}.json"))) for n in need}

    index = json.load(open(os.path.join(pack, "index.json")))
    want = {r["token"]: i for i, r in enumerate(index)}
    print(f"{len(want)} packed keyframes")

    inst = {r["token"]: r for r in meta["instance"]}
    cat = {r["token"]: r["name"] for r in meta["category"]}
    vis = {r["token"]: r["level"] for r in meta["visibility"]}
    attr = {r["token"]: r["name"] for r in meta["attribute"]}

    # Stable integer ids so the arrays stay compact and joinable.
    cat_names = sorted(set(cat.values()))
    cat_id = {n: i for i, n in enumerate(cat_names)}
    inst_ids: dict = {}

    rows = []
    for a in meta["sample_annotation"]:
        f = want.get(a["sample_token"])
        if f is None:
            continue                                  # not a packed keyframe
        it = a["instance_token"]
        name = cat[inst[it]["category_token"]]
        rows.append((
            f,                                        # frame index into the pack
            inst_ids.setdefault(it, len(inst_ids)),    # track id
            cat_id[name],
            *[float(x) for x in a["translation"]],     # xyz, global frame
            *[float(x) for x in a["size"]],            # w, l, h
            *[float(x) for x in a["rotation"]],        # quaternion wxyz
            int(a["num_lidar_pts"]),
            int(a["num_radar_pts"]),
            _vis_bucket(vis.get(a["visibility_token"], "")),
        ))

    if not rows:
        raise SystemExit("no annotations matched the packed keyframes -- is "
                         "index.json from the same run as this metadata?")

    arr = np.asarray(rows, np.float64)
    out = os.path.join(pack, "annotations_val.npz")
    np.savez_compressed(
        out,
        frame=arr[:, 0].astype(np.int32),
        track=arr[:, 1].astype(np.int32),
        category=arr[:, 2].astype(np.int16),
        translation=arr[:, 3:6].astype(np.float32),
        size=arr[:, 6:9].astype(np.float32),
        rotation=arr[:, 9:13].astype(np.float32),
        num_lidar_pts=arr[:, 13].astype(np.int32),
        num_radar_pts=arr[:, 14].astype(np.int32),
        visibility=arr[:, 15].astype(np.int8),
        category_names=np.asarray(cat_names),
    )

    per_frame = np.bincount(arr[:, 0].astype(int), minlength=len(index))
    v = arr[:, 15].astype(int)
    print(f"objects           : {len(rows):,}")
    print(f"unique tracks     : {len(inst_ids):,}")
    print(f"objects / keyframe: mean {per_frame.mean():.1f}, max {per_frame.max():.0f}, "
          f"empty frames {int((per_frame == 0).sum())}")
    print("visibility buckets: " + ", ".join(
        f"{lbl} {int((v == i).sum()):,}"
        for i, lbl in enumerate(["0-40%", "40-60%", "60-80%", "80-100%", "unknown"])))
    print(f"categories        : {len(cat_names)}")
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


def _vis_bucket(level: str) -> int:
    """nuScenes visibility levels are strings like 'v0-40'. 4 means unknown."""
    return {"v0-40": 0, "v40-60": 1, "v60-80": 2, "v80-100": 3}.get(level, 4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir containing v1.0-trainval/")
    ap.add_argument("--pack", required=True, help="dir containing index.json")
    a = ap.parse_args()
    main(a.data, a.pack)
