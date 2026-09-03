"""One devkit-free view of the nuScenes tables.

Every other odfm_* module resolves sensors, calibration, poses and annotations
through this file. That is deliberate: before it existed, odfm_lidar read the
pose chain out of a precomputed manifest while anything camera-side would have
had to re-parse the raw JSON, and the two could silently disagree about which
frame they were in. One reader means one set of conventions.

Frames, stated once (odfm_lidar repeats these because it is the entry point
most people read first):
    sensor --cal--> ego --pose--> global
    quaternions are (w, x, y, z); the ego frame is x-forward, y-left, z-up.

Loading is lazy and cached per dataroot, so importing this costs nothing and
the ~40 MB of annotation JSON is parsed at most once per process.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data/nuscenes"
DEFAULT_VERSION = "v1.0-mini"

CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
           "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

_DB = {}


def quat_to_rot(q) -> np.ndarray:
    """(w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([[1 - (yy + zz), xy - wz, xz + wy],
                     [xy + wz, 1 - (xx + zz), yz - wx],
                     [xz - wy, yz + wx, 1 - (xx + yy)]], dtype=np.float64)


def yaw_from_quat(q) -> float:
    R = quat_to_rot(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


class NuScenesTables:
    """Indexed access to one nuScenes split."""

    def __init__(self, dataroot=DEFAULT_ROOT, version=DEFAULT_VERSION):
        self.dataroot = Path(dataroot)
        self.tdir = self.dataroot / version

        def _load(name):
            with open(self.tdir / f"{name}.json") as fh:
                return json.load(fh)

        self.sample = {s["token"]: s for s in _load("sample")}
        self.scene = {s["token"]: s for s in _load("scene")}
        self.ego_pose = {e["token"]: e for e in _load("ego_pose")}
        self.sample_data = {d["token"]: d for d in _load("sample_data")}

        sensors = {s["token"]: s for s in _load("sensor")}
        self.calib = {}
        for c in _load("calibrated_sensor"):
            c = dict(c)
            c["channel"] = sensors[c["sensor_token"]]["channel"]
            self.calib[c["token"]] = c

        cats = {c["token"]: c["name"] for c in _load("category")}
        # Annotator-assessed CAMERA visibility, 4 buckets. Independent of
        # anything this repo computes, which is what makes it usable as ground
        # truth for the integrity map rather than as another of its inputs.
        self.visibility = {v["token"]: v for v in _load("visibility")}
        inst = {i["token"]: i for i in _load("instance")}
        self.ann_by_sample = {}
        for a in _load("sample_annotation"):
            a = dict(a)
            a["category"] = cats[inst[a["instance_token"]]["category_token"]]
            self.ann_by_sample.setdefault(a["sample_token"], []).append(a)

        # sample -> {channel: sample_data_token}. The raw sample.json has no
        # `data` field at all; the devkit synthesises it by reverse-indexing
        # the keyframe rows of sample_data, so this file has to as well.
        self.sample_channels = {}
        for d in self.sample_data.values():
            if d["is_key_frame"]:
                ch = self.calib[d["calibrated_sensor_token"]]["channel"]
                self.sample_channels.setdefault(d["sample_token"], {})[ch] = d["token"]

        # keyframes in capture order within each scene, so "give me a frame
        # with history" is a lookup rather than a scan
        self.scene_samples = {}
        for tok, s in self.sample.items():
            self.scene_samples.setdefault(s["scene_token"], []).append(tok)
        for k in self.scene_samples:
            self.scene_samples[k].sort(key=lambda t: self.sample[t]["timestamp"])

    # ---------------- sensor records ----------------

    def sensor_record(self, sample_token: str, channel: str) -> dict:
        """The sample_data row for one channel of one keyframe, with its
        calibration and ego pose already resolved and the file path made
        absolute. This is the unit every downstream module consumes."""
        sd = self.sample_data[self.sample_channels[sample_token][channel]]
        cal = self.calib[sd["calibrated_sensor_token"]]
        ego = self.ego_pose[sd["ego_pose_token"]]
        return {
            "channel": channel,
            "path": self.dataroot / sd["filename"],
            "width": sd.get("width"), "height": sd.get("height"),
            "timestamp": sd["timestamp"],
            "cal_rot": cal["rotation"], "cal_trans": cal["translation"],
            "intrinsic": np.asarray(cal["camera_intrinsic"], dtype=np.float64)
                         if cal.get("camera_intrinsic") else None,
            "ego_rot": ego["rotation"], "ego_trans": ego["translation"],
            "token": sd["token"],
        }

    def sweeps(self, sample_token: str, channel: str, n: int = 10) -> list:
        """The keyframe record plus up to n-1 prior non-keyframe sweeps."""
        sd = self.sample_data[self.sample_channels[sample_token][channel]]
        out, cur = [], sd
        while len(out) < n:
            cal = self.calib[cur["calibrated_sensor_token"]]
            ego = self.ego_pose[cur["ego_pose_token"]]
            out.append({
                "channel": channel, "path": self.dataroot / cur["filename"],
                "timestamp": cur["timestamp"],
                "cal_rot": cal["rotation"], "cal_trans": cal["translation"],
                "ego_rot": ego["rotation"], "ego_trans": ego["translation"],
            })
            if not cur["prev"]:
                break
            cur = self.sample_data[cur["prev"]]
        return out

    # ---------------- annotations ----------------

    def boxes_ego(self, sample_token: str) -> list:
        """Annotated boxes for a keyframe, in that keyframe's LIDAR-ego frame.

        nuScenes stores boxes in the global frame. Heading relative to the ego
        is the global box yaw minus the ego yaw -- exact, and the reason this
        replaces the cross-frame nearest-match heading estimate used before the
        pose chain was available.
        """
        ref = self.sensor_record(sample_token, "LIDAR_TOP")
        R = quat_to_rot(ref["ego_rot"])
        t = np.asarray(ref["ego_trans"], dtype=np.float64)
        ego_yaw = yaw_from_quat(ref["ego_rot"])
        out = []
        for a in self.ann_by_sample.get(sample_token, []):
            c = (np.asarray(a["translation"], dtype=np.float64) - t) @ R
            w, l, h = [float(v) for v in a["size"]]
            out.append({
                "centre": c, "wlh": (w, l, h),
                "yaw": yaw_from_quat(a["rotation"]) - ego_yaw,
                "global_rot": a["rotation"], "global_trans": a["translation"],
                "category": a["category"],
                "num_lidar_pts": int(a.get("num_lidar_pts", 0)),
                "num_radar_pts": int(a.get("num_radar_pts", 0)),
                "visibility_level": int(a["visibility_token"])
                if a.get("visibility_token") else None,
                "instance_token": a["instance_token"],
            })
        return out

    # ---------------- convenience ----------------

    def frames_with_history(self, min_sweeps: int = 9) -> list:
        """Keyframes that have at least `min_sweeps` prior LIDAR sweeps, which
        is what multi-sweep accumulation and motion separation both need."""
        # Counted on the real prev-chain, not on keyframe position. LIDAR_TOP
        # runs at 20 Hz against 2 Hz keyframes, so even the second keyframe of
        # a scene already has ~10 prior sweeps; assuming otherwise would throw
        # away most of the usable frames.
        out = []
        for tok in self.sample:
            if len(self.sweeps(tok, "LIDAR_TOP", min_sweeps + 1)) > min_sweeps:
                out.append(tok)
        return out

    def scene_name(self, sample_token: str) -> str:
        return self.scene[self.sample[sample_token]["scene_token"]]["name"]


def tables(dataroot=DEFAULT_ROOT, version=DEFAULT_VERSION) -> NuScenesTables:
    key = (str(dataroot), version)
    if key not in _DB:
        _DB[key] = NuScenesTables(dataroot, version)
    return _DB[key]
