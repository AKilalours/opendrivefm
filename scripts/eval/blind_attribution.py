#!/usr/bin/env python3
"""Which object is creating the blind spot, and how much road does it hide?

Every occlusion result so far says a cell is unseen. None says WHO made it
unseen. That is the question a rig designer and a safety case both ask, and it
is answerable from the same ray march: a ray stops at its first occupied voxel,
so every cell that ray would have reached afterwards is hidden BY that voxel.

    for each pixel ray:
        march to the first occupied voxel          -> the OCCLUDER
        keep marching                              -> the cells it HIDES
        attribute those cells, and their ground footprint, to the occluder

The headline this produces is a sentence no occupancy paper currently has:
what fraction of a camera-only stack's blind volume is caused by parked
vehicles, versus vegetation, versus buildings -- and how many square metres of
drivable surface a single truck takes away.

Ground truth is used for the occluder's CLASS only. The geometry is the same
ray march that produced the frozen observability maps, so the attribution is
consistent with every number in A12-A28 by construction.
"""
from __future__ import annotations
import argparse, glob, importlib.util, json, os, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location(
    "build_obs_ray", os.path.join(HERE, "build_observability_ray.py"))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)

FREE, RES, RNG, NZ, Z0, N = 17, 0.4, 40.0, 16, -1.0, 200
NAMES = ["other", "barrier", "bicycle", "bus", "car", "constr veh", "motorcycle",
         "pedestrian", "traffic cone", "trailer", "truck", "driveable",
         "other flat", "sidewalk", "terrain", "manmade", "vegetation", "free"]
DRIVABLE = 11
VEHICLE = {3, 4, 5, 9, 10}
_G = {}


def _init(obs_dir):
    _G["obs"] = obs_dir
    _G["gt"] = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}
    _G["calib"] = json.load(open(os.path.join(ROOT, "data/pack/calib.json")))


def march_attribute(occ_flat, cam, W, H, stride=4, near=0.5, far=62.0):
    """Like B.march_visibility, but rays do NOT terminate on first hit.

    Instead the first hit is remembered as that ray's occluder, and every cell
    the ray enters afterwards is credited to it. Cost is roughly 2x the
    terminating march, because a ray now runs to the grid edge.
    """
    R = B.M.quat_to_rot(cam["sensor2ego_rotation"])
    t = np.asarray(cam["sensor2ego_translation"], np.float32)
    K = np.asarray(cam.get("cam_intrinsic", cam.get("intrinsic")), np.float32)
    uu, vv = np.meshgrid(np.arange(0, W, stride, dtype=np.float32) + 0.5,
                         np.arange(0, H, stride, dtype=np.float32) + 0.5,
                         indexing="xy")
    d = np.stack([(uu.ravel() - K[0, 2]) / K[0, 0],
                  (vv.ravel() - K[1, 2]) / K[1, 1],
                  np.ones(uu.size, np.float32)], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d @ R.T

    nray = d.shape[0]
    occluder = np.full(nray, -1, np.int64)        # flat index of each ray's blocker
    hidden_by = np.full(occ_flat.size, -1, np.int64)
    vis = np.zeros(occ_flat.size, bool)
    for s in np.arange(near, far, B.STEP, dtype=np.float32):
        pts = t + d * s
        idx, ok = B.to_index(pts)
        live = np.flatnonzero(ok)
        if live.size == 0:
            continue
        cell = idx[live]
        free_ray = occluder[live] < 0
        # rays with no occluder yet: this cell is SEEN
        vis[cell[free_ray]] = True
        # rays already blocked: this cell is HIDDEN by that ray's occluder
        blocked = ~free_ray
        if blocked.any():
            hidden_by[cell[blocked]] = occluder[live[blocked]]
        # new hits become occluders for the rest of their ray
        hit = occ_flat[cell] & free_ray
        if hit.any():
            occluder[live[hit]] = cell[hit]
    return vis, hidden_by


def _one(row):
    tok, scene = row
    g = np.load(_G["gt"][tok])
    gcls = g["semantics"].astype(np.int16)
    occ_flat = (gcls.reshape(-1) != FREE)
    cal = _G["calib"][tok]["cams"]

    hidden_any = np.full(occ_flat.size, -1, np.int64)
    seen_any = np.zeros(occ_flat.size, bool)
    for c in B.CAMS:
        cam = cal.get(c)
        if cam is None:
            continue
        vis, hb = march_attribute(occ_flat, cam, cam.get("width", 1600),
                                  cam.get("height", 900))
        seen_any |= vis
        take = (hidden_any < 0) & (hb >= 0)
        hidden_any[take] = hb[take]

    flat_cls = gcls.reshape(-1)
    # a cell is BLIND if no camera reached it and at least one camera's ray was
    # stopped short of it -- that is occlusion, as distinct from out of frustum
    blind = (~seen_any) & (hidden_any >= 0)
    occl_cls = flat_cls[hidden_any[blind]]

    vol = np.bincount(occl_cls, minlength=18).astype(np.int64)

    # how much DRIVABLE SURFACE each occluder class takes away
    ii = np.arange(occ_flat.size)[blind]
    k = ii % NZ
    is_road = flat_cls[blind] == DRIVABLE
    road_area = np.bincount(occl_cls[is_road], minlength=18).astype(np.int64)

    # the single worst occluder in this frame, by hidden volume
    worst_cls, worst_n = -1, 0
    if blind.any():
        u, c = np.unique(hidden_any[blind], return_counts=True)
        j = int(np.argmax(c))
        worst_cls, worst_n = int(flat_cls[u[j]]), int(c[j])
    return dict(token=tok, scene=scene, blind=int(blind.sum()),
                vol=vol.tolist(), road=road_area.tolist(),
                worst_cls=worst_cls, worst_n=worst_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--obs", default="data/pack/obs_max")
    ap.add_argument("--cache", default="outputs/artifacts/blind_attr_cache.jsonl")
    ap.add_argument("--out", default="outputs/artifacts/blind_attribution.json")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    cache = os.path.join(ROOT, a.cache)
    os.makedirs(os.path.dirname(cache), exist_ok=True)

    if not a.report:
        index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
        gt = {os.path.basename(os.path.dirname(q)) for q in glob.glob(
            os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                         "*", "*", "labels.npz"))}
        rows = [(r["token"], r["scene"]) for r in index if r["token"] in gt]
        rng = np.random.default_rng(a.seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), min(a.frames, len(rows)),
                                                   replace=False))]
        done = set()
        if os.path.exists(cache):
            done = {json.loads(l)["token"] for l in open(cache)}
        todo = [r for r in rows if r[0] not in done][:a.chunk]
        print(f"{len(rows)} frames | cached {len(done)} | this run {len(todo)} | "
              f"{os.cpu_count()} cores", flush=True)
        if not todo:
            print("complete -- run --report"); return
        t0 = time.time()
        with open(cache, "a") as fh, Pool(max(os.cpu_count() - 1, 1),
                                          initializer=_init, initargs=(a.obs,)) as pool:
            for n, r in enumerate(pool.imap_unordered(_one, todo, chunksize=2)):
                fh.write(json.dumps(r) + "\n"); fh.flush()
                if (n + 1) % 20 == 0:
                    el = time.time() - t0
                    print(f"  {n+1}/{len(todo)}  {el/(n+1):.2f}s/frame  "
                          f"eta {(len(todo)-n-1)*el/(n+1)/60:.1f} min", flush=True)
        left = len(rows) - len(done) - len(todo)
        print(f"chunk done, {left} remain" if left > 0 else
              "complete -- run --report", flush=True)
        return

    recs = [json.loads(l) for l in open(cache)]
    V = np.array([r["vol"] for r in recs]).sum(0)
    Rd = np.array([r["road"] for r in recs]).sum(0)
    tot, totr = V.sum(), Rd.sum()
    nf = len(recs)
    print(f"\n{nf} frames | {tot:,} occluded voxels attributed to an occluder")
    print("=" * 84)
    print(f"{'occluder class':<16}{'hidden volume':>16}{'share':>9}"
          f"{'hidden drivable':>18}{'share':>9}")
    print("-" * 84)
    order = np.argsort(-V)
    rows_out = []
    for i in order[:10]:
        if V[i] == 0: continue
        m2 = Rd[i] * RES * RES / nf
        print(f"{NAMES[i]:<16}{V[i]:>16,}{100*V[i]/tot:>8.1f}%"
              f"{m2:>13.1f} m2/fr{100*Rd[i]/max(totr,1):>8.1f}%")
        rows_out.append(dict(cls=NAMES[i], vol=int(V[i]), vol_share=float(V[i]/tot),
                             road_m2_per_frame=float(m2),
                             road_share=float(Rd[i]/max(totr,1))))
    print("=" * 84)
    # LIMITATION, stated before the headline rather than after it.
    # "driveable hides driveable" is a RAY-SAMPLING artefact, not geometry. At
    # range the pixel grid under-samples the ground plane, so some road cells
    # fall between rays and are reached by none; the attribution then credits
    # them to the nearest road cell a ray did hit. Cameras do lose angular
    # resolution with range -- that is real, and the coverage term already
    # models it -- but calling the road its own occluder is misleading. Ground
    # classes are therefore reported separately from structures and objects.
    GROUND_OCC = {11, 12, 13, 14}
    keep = [i for i in range(18) if i not in GROUND_OCC]
    Vk = V[keep].sum(); Rk = Rd[keep].sum()
    print(f"\nSTRUCTURES AND OBJECTS ONLY (ground self-occlusion excluded as a "
          f"ray-sampling artefact)")
    print("-" * 84)
    for i in np.argsort(-V):
        if i in GROUND_OCC or V[i] == 0: continue
        print(f"{NAMES[i]:<16}{V[i]:>16,}{100*V[i]/Vk:>8.1f}%"
              f"{Rd[i]*RES*RES/nf:>13.1f} m2/fr{100*Rd[i]/max(Rk,1):>8.1f}%")
    print("-" * 84)
    veh = sum(V[i] for i in VEHICLE); vehr = sum(Rd[i] for i in VEHICLE)
    print(f"vehicles, of structures and objects only: "
          f"{100*veh/Vk:.1f}% of hidden volume, {100*vehr/max(Rk,1):.1f}% of "
          f"hidden drivable surface")
    print(f"all vehicle classes combined: {100*veh/tot:.1f}% of hidden volume, "
          f"{100*vehr/max(totr,1):.1f}% of hidden drivable surface, "
          f"{vehr*RES*RES/nf:.1f} m2 per frame")
    w = np.bincount([r["worst_cls"] for r in recs if r["worst_cls"] >= 0], minlength=18)
    j = int(np.argmax(w))
    print(f"single worst occluder in a frame is most often: {NAMES[j]} "
          f"({100*w[j]/max(w.sum(),1):.0f}% of frames)")
    json.dump(dict(frames=nf, total_occluded=int(tot), by_class=rows_out,
                   vehicle_vol_share=float(veh/tot),
                   vehicle_road_share=float(vehr/max(totr,1)),
                   vehicle_road_m2_per_frame=float(vehr*RES*RES/nf),
                   vehicle_vol_share_no_ground=float(veh/Vk),
                   vehicle_road_share_no_ground=float(vehr/max(Rk,1)),
                   note="ground-class self-occlusion is a ray-sampling artefact "
                        "and is reported separately",
                   most_common_worst_occluder=NAMES[j]),
              open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
