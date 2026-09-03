#!/usr/bin/env python3
"""Render the frames and per-object records the validation console displays.

Writes outputs/console/bundle.json plus base64 JPEGs, so the console is a
single self-contained HTML file with no server and no network. Every number in
the bundle comes from the pipeline or from a report JSON in outputs/artifacts/;
nothing is authored here.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))
ROOT = Path(__file__).resolve().parents[1]

import odfm_bev as B        # noqa: E402
import odfm_forecast as FC  # noqa: E402
import odfm_geom as GE      # noqa: E402
import odfm_ground as G     # noqa: E402
import odfm_lidar as L      # noqa: E402
import odfm_motion as M     # noqa: E402
import odfm_overlay as O    # noqa: E402
import odfm_tables as T     # noqa: E402

RNG = 54.0
RES_O, RES_I, RES_F = 0.20, 0.50, 0.40
TRUST = 0.795


def jpg(arr, size=None, q=74):
    im = Image.fromarray(np.asarray(arr, np.uint8))
    if size:
        im = im.resize(size, Image.LANCZOS)
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def obj_speed(tab, tok, inst):
    """Speed differenced between adjacent keyframes, in m/s."""
    sc = tab.scene_samples[tab.sample[tok]["scene_token"]]
    i = sc.index(tok)
    cur = {a["instance_token"]: a for a in tab.ann_by_sample.get(tok, [])}
    if inst not in cur:
        return 0.0
    best = 0.0
    for j in (i - 1, i + 1):
        if not (0 <= j < len(sc)):
            continue
        dt = abs(tab.sample[sc[j]]["timestamp"] - tab.sample[tok]["timestamp"]) / 1e6
        for a in tab.ann_by_sample.get(sc[j], []):
            if a["instance_token"] == inst and dt > 0:
                d = np.linalg.norm(np.asarray(a["translation"][:2], float) -
                                   np.asarray(cur[inst]["translation"][:2], float))
                best = max(best, d / dt)
    return best


def build_frame(tab, man, tok, cam_size=(336, 189), map_size=456):
    t0 = time.perf_counter()
    pts = L.load_sweep_stack(tok, man, n_sweeps=10)
    plane = G.fit_ground_plane(pts[:, :3])
    ground = G.label_ground(pts[:, :3], plane)
    prob, _, _ = G.occupancy_logodds(pts, ground, plane, rng_m=RNG, res=RES_O)
    motion, _ = M.classify_motion(pts, ground, plane, rng_m=RNG)
    boxes = tab.boxes_ego(tok)

    prob_i, _, _ = G.occupancy_logodds(pts, ground, plane, rng_m=RNG, res=RES_I)
    occ_i = prob_i > 0.65
    covs, raw, vis_frac = [], [], {}
    for cam in T.CAMERAS:
        rec = tab.sensor_record(tok, cam)
        c, _ = GE.camera_ground_coverage(rec, rng_m=RNG, res=RES_I, soft=True)
        v = G.visibility_from(occ_i, rec["cal_trans"][:2], rng_m=RNG, res=RES_I)
        raw.append(c)
        covs.append(c * v)
        vis_frac[cam] = round(float((c * v).sum() / max(1e-9, c.sum())), 3)
    integ = GE.integrity_map(covs, [TRUST] * 6)
    n_i = integ.shape[0]

    # ---- camera panels, with and without overlays so the console can toggle
    cams = {}
    cur = pts[pts[:, 5] == 0][:, :3]
    for cam in T.CAMERAS:
        rec = tab.sensor_record(tok, cam)
        raw_im = np.asarray(Image.open(rec["path"]).convert("RGB"))
        lid = O.draw_lidar_depth(raw_im, cur, rec, radius=2, stride=5)
        box, _ = O.draw_boxes_3d(lid, boxes, rec, label=False)
        cams[cam] = {
            "plain": jpg(raw_im, cam_size),
            "lidar": jpg(lid, cam_size),
            "boxes": jpg(box, cam_size),
            "visible_frac": vis_frac[cam],
        }

    # ---- maps
    pts_h = pts.copy()
    pts_h[:, 2] = G.height_above_ground(pts[:, :3], plane).astype(np.float32)
    st = G.grid_stats(G.prob_to_tristate(prob))
    maps = {
        "bev": jpg(B.render_bev(pts_h, boxes, rng_m=RNG, size=map_size, ss=3,
                                z_lo=-0.3, z_hi=3.0, canopy_above=3.0,
                                motion=motion, dynamic_label=M.DYNAMIC), q=80),
        "bev_nodyn": jpg(B.render_bev(pts_h, None, rng_m=RNG, size=map_size, ss=3,
                                      z_lo=-0.3, z_hi=3.0, canopy_above=3.0), q=80),
        "occupancy": jpg(B.render_occupancy_prob(prob, size=map_size, rng_m=RNG,
                                                 boxes=boxes, title=None,
                                                 subtitle=None, legend=False), q=80),
        "integrity": jpg(B.render_integrity(integ, size=map_size, rng_m=RNG,
                                            occ_mask=occ_i, boxes=boxes,
                                            title=None, subtitle=None,
                                            legend=False), q=80),
    }

    # ---- forecast error maps against the LiDAR that actually arrived
    sc = tab.scene_samples[tab.sample[tok]["scene_token"]]
    i0 = sc.index(tok)
    from eval_occupancy_forecast import stack_in_ref
    prob_f, _, _ = G.occupancy_logodds(pts, ground, plane, rng_m=RNG, res=RES_F)
    fore = []
    for h in (1, 2, 3):
        if i0 + h >= len(sc) or sc[i0 + h] not in man:
            continue
        ftok = sc[i0 + h]
        dt = (tab.sample[ftok]["timestamp"] - tab.sample[tok]["timestamp"]) / 1e6
        pred, _ = FC.forecast_occupancy(prob_f, pts, motion, plane, dt, rng_m=RNG,
                                        res=RES_F, mode="persistence",
                                        ground_mask=ground)
        fpts = stack_in_ref(ftok, man, man[tok])
        fpl = G.fit_ground_plane(fpts[:, :3])
        fg = G.label_ground(fpts[:, :3], fpl)
        gp, glo, _ = G.occupancy_logodds(fpts, fg, fpl, rng_m=RNG, res=RES_F)
        obs = np.abs(glo) > 1e-6
        gt, pr = (gp > 0.65) & obs, pred & obs
        tp = int((pr & gt).sum()); fp = int((pr & ~gt).sum()); fn = int((~pr & gt).sum())

        def grow(m):
            o = m.copy()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    o |= np.roll(np.roll(m, dx, 0), dy, 1)
            return o
        err = np.zeros(gt.shape + (3,), np.uint8)
        err[:, :] = (16, 19, 26)
        err[grow(~pr & gt)] = (255, 106, 122)
        err[grow(pr & ~gt)] = (255, 156, 92)
        err[grow(pr & gt)] = (126, 226, 168)
        truth = np.zeros_like(err); truth[:, :] = (16, 19, 26)
        truth[grow(gt)] = (150, 200, 220)
        predi = np.zeros_like(err); predi[:, :] = (16, 19, 26)
        predi[grow(pr)] = (150, 200, 220)
        flip = lambda a: np.flipud(np.fliplr(a))
        fore.append({
            "h": h, "dt": round(dt, 2),
            "iou": round(tp / max(1, tp + fp + fn), 4),
            "precision": round(tp / max(1, tp + fp), 4),
            "recall": round(tp / max(1, tp + fn), 4),
            "error": jpg(flip(err), (300, 300), 78),
            "pred": jpg(flip(predi), (300, 300), 78),
            "truth": jpg(flip(truth), (300, 300), 78),
        })

    # ---- per-object records, incl. 2D boxes per camera so the console can
    # highlight the same object everywhere at once
    from eval_integrity_visibility import footprint_cells
    objs = []
    for k, b in enumerate(boxes):
        if b["num_lidar_pts"] <= 0:
            continue
        r = float(np.hypot(b["centre"][0], b["centre"][1]))
        if r > 50:
            continue
        idx = footprint_cells(b, n_i, RES_I, RNG)
        integ_v = float(integ[idx].mean()) if idx is not None else 0.0
        proj = {}
        for cam in T.CAMERAS:
            rec = tab.sensor_record(tok, cam)
            corners = GE.box_corners_ego(b)
            camc = GE.ego_to_cam(corners, rec)
            if (camc[:, 2] <= 0.5).any():
                continue
            uv = (camc @ rec["intrinsic"].T)[:, :2] / camc[:, 2:3]
            x0, y0 = uv[:, 0].min(), uv[:, 1].min()
            x1, y1 = uv[:, 0].max(), uv[:, 1].max()
            if x1 < 0 or y1 < 0 or x0 > rec["width"] or y0 > rec["height"]:
                continue
            proj[cam] = [round(float(x0 / rec["width"]), 4),
                         round(float(y0 / rec["height"]), 4),
                         round(float(x1 / rec["width"]), 4),
                         round(float(y1 / rec["height"]), 4)]
        objs.append({
            "id": k,
            "cat": b["category"].split(".")[-1],
            "group": b["category"].split(".")[0],
            "x": round(float(b["centre"][0]), 2),
            "y": round(float(b["centre"][1]), 2),
            "range": round(r, 1),
            "yaw": round(float(b["yaw"]), 3),
            "wl": [round(b["wlh"][0], 2), round(b["wlh"][1], 2)],
            "speed": round(obj_speed(tab, tok, b["instance_token"]), 2),
            "vis": b["visibility_level"],
            "integrity": round(integ_v, 3),
            "pts": b["num_lidar_pts"],
            "cams": proj,
        })

    return {
        "token": tok,
        "scene": tab.scene_name(tok),
        "timestamp_us": tab.sample[tok]["timestamp"],
        "cameras": cams,
        "maps": maps,
        "forecast": fore,
        "objects": objs,
        "stats": {
            "returns": int(len(pts)),
            "dynamic": int((motion == M.DYNAMIC).sum()),
            "free": round(float(st["free_frac"]), 4),
            "occupied": round(float(st["occupied_frac"]), 4),
            "unknown": round(float(st["unknown_frac"]), 4),
            "ground_frac": round(float(ground.mean()), 4),
            "plane_tilt_deg": round(float(np.degrees(np.arccos(abs(plane[0][2])))), 2),
            "integrity_mean_drivable": round(float(GE.integrity_stats(
                integ, G.prob_to_tristate(prob_i), free_value=G.FREE
            )["mean_over_free_space"]), 4),
            "boxes": len(boxes),
            "boxes_observed": sum(1 for b in boxes if b["num_lidar_pts"] > 0),
            "build_ms": round(1000 * (time.perf_counter() - t0), 1),
        },
        "rng_m": RNG,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--per-scene", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "outputs/console/bundle.json"))
    a = ap.parse_args()

    tab = T.tables()
    man = L.load_manifest()
    frames = []
    for si, (stok, toks) in enumerate(list(tab.scene_samples.items())):
        if si >= a.scenes:
            break
        picks = [t for t in toks[6:-4] if t in man][:a.per_scene]
        for tok in picks:
            frames.append(build_frame(tab, man, tok))
            print(f"  {tab.scene_name(tok)}  {tok[:10]}  "
                  f"{frames[-1]['stats']['build_ms']:.0f} ms  "
                  f"{len(frames[-1]['objects'])} objects", flush=True)

    reports = {}
    for name in ("occupancy_forecast_report", "integrity_visibility_report",
                 "motion_separation_report", "timing_report",
                 "trajectory_ade_report", "learned_model_report",
                 "vla_report", "vlm_report", "robustness_report",
                 "trajlm_retrained_report", "kernel_bench_report",
                 "cpp_report"):
        p = ROOT / f"outputs/artifacts/{name}.json"
        if p.exists():
            d = json.loads(p.read_text())
            d.pop("per_frame", None)
            d.pop("boxes", None)
            reports[name] = d

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"frames": frames, "reports": reports}))
    mb = out.stat().st_size / 1e6
    print(f"\nwrote {out.relative_to(ROOT)}  {len(frames)} frames  {mb:.1f} MB")


if __name__ == "__main__":
    main()
