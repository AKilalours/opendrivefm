"""Pack the nuScenes validation split into a small, fixed-size array.

Why pack at all
---------------
Training and inference on multi-camera BEV is dataloader-bound, not GPU-bound:
six JPEGs per keyframe, four keyframes per temporal window, twenty-four decodes
per sample. On a pod with 16-31 vCPU that leaves an A100 idling while the CPU
decodes. Packing once into a flat uint8 memmap turns every later read into a
slice, so the GPU stays fed and the 350 GB of JPEGs never has to exist again.

It also makes the whole project portable: the pack is ~14 GB, small enough to
keep, copy, and analyse on a laptop, which is where the rest of this work runs.

Three stages, run by bootstrap_runpod.sh:

  manifest  read metadata only, decide which files are wanted, write the list
            tar uses to extract just those members. Nothing is downloaded yet.
  pack      copy whatever of that list currently exists on disk into the array,
            then mark those rows done. Called once per blob; blobs each hold a
            slice of the dataset, so rows fill in over several calls.
  verify    confirm every row was filled and report what is missing.

The pack is resumable by construction: `filled.npy` records which rows are
already written, so re-running after an interrupted blob costs nothing.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


# --------------------------------------------------------------------------
def load_meta(data: str) -> Dict[str, list]:
    root = os.path.join(data, "v1.0-trainval")
    need = ["scene", "sample", "sample_data", "calibrated_sensor",
            "ego_pose", "sensor", "log"]
    return {n: json.load(open(os.path.join(root, f"{n}.json"))) for n in need}


def split_scene_names(which: str) -> set:
    """Official scene names for a split, straight from the devkit.

    "val" is all the paper strictly needs: published checkpoints are evaluated
    on it and every downstream analysis is post-hoc. "all" additionally packs
    the 700 training scenes.

    Pack "all" if there is disk for it. The ten blobs have to be downloaded
    either way -- that is the slow, expensive part -- and they are deleted as
    they are consumed. Deciding later that the training split is wanted means
    downloading 350 GB a second time, which is the one mistake in this pipeline
    that cannot be cheaply undone.

      val   150 scenes,  6,019 keyframes,  ~14 GB packed
      all   850 scenes, 34,149 keyframes,  ~72 GB packed
    """
    from nuscenes.utils.splits import train, val
    if which == "val":
        return set(val)
    if which == "train":
        return set(train)
    return set(train) | set(val)


def build_manifest(data: str, pack: str, which: str = "val") -> dict:
    m = load_meta(data)
    scenes = {s["token"]: s for s in m["scene"]}
    samples = {s["token"]: s for s in m["sample"]}
    sensors = {s["token"]: s for s in m["sensor"]}
    calibs = {c["token"]: c for c in m["calibrated_sensor"]}
    poses = {p["token"]: p for p in m["ego_pose"]}

    from nuscenes.utils.splits import val as _val
    want_scenes = split_scene_names(which)
    keep_scene_tokens = {t: s for t, s in scenes.items() if s["name"] in want_scenes}
    expect = {"val": 150, "train": 700, "all": 850}[which]
    print(f"scenes for split '{which}': {len(keep_scene_tokens)} (expected {expect})")
    val_names = set(_val)

    # sample_data rows that are keyframes of the sensors we care about
    by_sample: Dict[str, Dict[str, dict]] = {}
    for sd in m["sample_data"]:
        if not sd["is_key_frame"]:
            continue
        cal = calibs.get(sd["calibrated_sensor_token"])
        if cal is None:
            continue
        chan = sensors[cal["sensor_token"]]["channel"]
        if chan not in CAMS and chan != "LIDAR_TOP":
            continue
        by_sample.setdefault(sd["sample_token"], {})[chan] = sd

    rows: List[dict] = []
    wanted_paths: List[str] = []
    for st, samp in samples.items():
        sc = scenes.get(samp["scene_token"])
        if sc is None or sc["token"] not in keep_scene_tokens:
            continue
        chans = by_sample.get(st, {})
        if not all(c in chans for c in CAMS):
            continue                                   # incomplete rig, skip
        row = {"token": st, "scene": sc["name"], "timestamp": samp["timestamp"],
               "split": "val" if sc["name"] in val_names else "train",
               "cams": {}, "lidar": None}
        for c in CAMS:
            sd = chans[c]
            cal = calibs[sd["calibrated_sensor_token"]]
            pose = poses[sd["ego_pose_token"]]
            row["cams"][c] = {
                "filename": sd["filename"],
                "width": sd["width"], "height": sd["height"],
                "intrinsic": cal["camera_intrinsic"],
                "sensor2ego_translation": cal["translation"],
                "sensor2ego_rotation": cal["rotation"],
                "ego2global_translation": pose["translation"],
                "ego2global_rotation": pose["rotation"],
            }
            wanted_paths.append(sd["filename"])
        if "LIDAR_TOP" in chans:
            sd = chans["LIDAR_TOP"]
            cal = calibs[sd["calibrated_sensor_token"]]
            pose = poses[sd["ego_pose_token"]]
            row["lidar"] = {
                "filename": sd["filename"],
                "sensor2ego_translation": cal["translation"],
                "sensor2ego_rotation": cal["rotation"],
                "ego2global_translation": pose["translation"],
                "ego2global_rotation": pose["rotation"],
            }
            wanted_paths.append(sd["filename"])
        rows.append(row)

    rows.sort(key=lambda r: (r["scene"], r["timestamp"]))
    os.makedirs(pack, exist_ok=True)
    json.dump(rows, open(os.path.join(pack, "index.json"), "w"))
    with open(os.path.join(pack, "wanted_paths.txt"), "w") as fh:
        fh.write("\n".join(wanted_paths) + "\n")

    n_lidar = sum(1 for r in rows if r["lidar"])
    n_val = sum(1 for r in rows if r["split"] == "val")
    print(f"keyframes: {len(rows)} (val {n_val}, train {len(rows) - n_val}) | "
          f"with LiDAR: {n_lidar} | files to extract: {len(wanted_paths)}")
    est = len(rows) * 6 * 256 * 448 * 3 / 1e9  # uint8, six cameras
    print(f"packed image array will be ~{est:.1f} GB")
    return {"rows": rows}


# --------------------------------------------------------------------------
def open_pack(pack: str, n: int, h: int, w: int):
    path = os.path.join(pack, f"images_{h}x{w}.u8")
    shape = (n, 6, h, w, 3)
    mode = "r+" if os.path.exists(path) else "w+"
    arr = np.memmap(path, dtype=np.uint8, mode=mode, shape=shape)
    fpath = os.path.join(pack, "filled.npy")
    filled = np.load(fpath) if os.path.exists(fpath) else np.zeros(n, bool)
    return arr, filled, fpath


# Worker state. Each process opens the memmap once and writes only the rows it
# is given, so the writes are disjoint and no lock is needed. Passing decoded
# images back through IPC instead would move ~2 MB per keyframe between
# processes, which costs more than the decode it is trying to parallelise.
_W = {}


def _init_worker(pack, n, h, w, data, lid_path):
    from PIL import Image                                      # noqa: F401
    path = os.path.join(pack, f"images_{h}x{w}.u8")
    _W["arr"] = np.memmap(path, dtype=np.uint8, mode="r+", shape=(n, 6, h, w, 3))
    _W.update(h=h, w=w, data=data, lid=lid_path)


def _pack_one(task):
    """Decode six cameras for one keyframe, resize, write in place."""
    from PIL import Image
    i, cam_files, lidar_file, token = task
    h, w, data = _W["h"], _W["w"], _W["data"]
    arr = _W["arr"]
    for j, fn in enumerate(cam_files):
        with Image.open(os.path.join(data, fn)) as im:
            arr[i, j] = np.asarray(
                im.convert("RGB").resize((w, h), Image.BILINEAR), np.uint8)
    if lidar_file:
        src = os.path.join(data, lidar_file)
        if os.path.exists(src):
            pts = np.fromfile(src, dtype=np.float32).reshape(-1, 5)[:, :4]
            np.save(os.path.join(_W["lid"], f"{token}.npy"), pts.astype(np.float16))
    return i


def do_pack(data: str, pack: str, h: int, w: int, workers: int = 0):
    import multiprocessing as mp

    rows = json.load(open(os.path.join(pack, "index.json")))
    arr, filled, fpath = open_pack(pack, len(rows), h, w)
    arr.flush()
    del arr                       # workers reopen it; avoid two mappings here

    lid_path = os.path.join(pack, "lidar")
    os.makedirs(lid_path, exist_ok=True)

    # Only rows whose six camera files are present in THIS blob.
    tasks = []
    for i, r in enumerate(rows):
        if filled[i]:
            continue
        fns = [r["cams"][c]["filename"] for c in CAMS]
        if not all(os.path.exists(os.path.join(data, f)) for f in fns):
            continue
        tasks.append((i, fns, r["lidar"]["filename"] if r["lidar"] else None,
                      r["token"]))

    if not tasks:
        print(f"    blob contributed 0 keyframes | "
              f"{int(filled.sum())}/{len(rows)} packed so far")
        return

    n_proc = workers or min(mp.cpu_count(), 64)
    print(f"    {len(tasks)} keyframes to decode on {n_proc} processes", flush=True)

    done = 0
    with mp.Pool(n_proc, initializer=_init_worker,
                 initargs=(pack, len(rows), h, w, data, lid_path)) as pool:
        for i in pool.imap_unordered(_pack_one, tasks, chunksize=8):
            filled[i] = True
            done += 1
            if done % 2000 == 0:
                np.save(fpath, filled)
                print(f"    packed {done}/{len(tasks)} this blob", flush=True)

    np.save(fpath, filled)
    print(f"    blob contributed {done} keyframes | "
          f"{int(filled.sum())}/{len(rows)} packed so far")


# --------------------------------------------------------------------------
def do_verify(data: str, pack: str, h: int, w: int):
    rows = json.load(open(os.path.join(pack, "index.json")))
    filled = np.load(os.path.join(pack, "filled.npy"))
    n = len(rows)
    missing = [rows[i]["token"] for i in np.where(~filled)[0]]

    calib = {r["token"]: {"scene": r["scene"], "timestamp": r["timestamp"],
                          "cams": r["cams"], "lidar": r["lidar"]} for r in rows}
    json.dump(calib, open(os.path.join(pack, "calib.json"), "w"))

    lid = len(os.listdir(os.path.join(pack, "lidar"))) if \
        os.path.isdir(os.path.join(pack, "lidar")) else 0
    scenes = sorted({r["scene"] for r in rows})
    size = os.path.getsize(os.path.join(pack, f"images_{h}x{w}.u8")) / 1e9

    print(f"keyframes packed : {int(filled.sum())}/{n}")
    print(f"scenes           : {len(scenes)}")
    print(f"lidar sweeps     : {lid}")
    print(f"image array      : {size:.1f} GB at {h}x{w}")
    if missing:
        print(f"\nMISSING {len(missing)} keyframes -- a blob probably failed to "
              f"download. Re-run bootstrap_runpod.sh; finished blobs are skipped.")
        print("first few:", missing[:5])
    else:
        print("\nComplete. Every val keyframe is packed.")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["manifest", "pack", "verify"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--workers", type=int, default=0,
                    help="decode processes; 0 picks min(cpu_count, 64)")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"],
                    help="val is all the paper needs; all also packs the 700 "
                         "training scenes, which costs disk now but saves "
                         "re-downloading 350 GB later")
    a = ap.parse_args()
    if a.stage == "manifest":
        build_manifest(a.data, a.pack, a.split)
    elif a.stage == "pack":
        do_pack(a.data, a.pack, a.height, a.width, a.workers)
    else:
        do_verify(a.data, a.pack, a.height, a.width)
