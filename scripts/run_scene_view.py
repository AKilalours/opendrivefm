#!/usr/bin/env python3
"""One command, one keyframe, the whole perception stack.

    python3 scripts/run_scene_view.py --index 30

Produces a single figure holding the six camera views with LiDAR depth and 3D
boxes projected into them, the multi-sweep BEV with dynamic returns separated,
the ray-cast occupancy grid, and the Perception Integrity Map -- plus a JSON of
every number behind them.

The point of building it as one command is that these are not four independent
pictures. The same pose chain places the returns in the BEV and projects them
into the cameras; the same ground-plane fit drives both the occupancy grid and
the motion labels; the occupancy grid decides which cells the integrity summary
is averaged over. If any of it is wrong, the panels disagree with each other
visibly, which is the entire diagnostic value.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import odfm_bev as B          # noqa: E402
import odfm_geom as GE        # noqa: E402
import odfm_ground as G       # noqa: E402
import odfm_lidar as L        # noqa: E402
import odfm_motion as M       # noqa: E402
import odfm_overlay as O      # noqa: E402
import odfm_tables as T       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Per-camera trust from the trained CameraTrustScorer's per-fault constants
# (scripts/gradio_app.py). These are FAULT-LEVEL constants, not live per-frame
# scores -- the integrity map is only as good as the trust feeding it, and this
# is the honest label for what is feeding it today.
DEFAULT_TRUST = 0.795


def build(index=30, n_sweeps=10, rng_m=54.0, res=0.5, out_dir=None,
          fault_camera=None, fault_trust=0.31):
    t0 = time.time()
    tab = T.tables()
    tok = tab.frames_with_history(9)[index]
    man = L.load_manifest()

    pts = L.load_sweep_stack(tok, man, n_sweeps=n_sweeps)
    plane = G.fit_ground_plane(pts[:, :3])
    ground = G.label_ground(pts[:, :3], plane)
    current = pts[:, 5] == 0

    occ, occ_meta = G.occupancy_grid(pts[:, :3], ground, rng_m=rng_m, res=res,
                                     plane=plane, visible=current)
    motion, motion_meta = M.classify_motion(pts, ground, plane, rng_m=rng_m)
    boxes = tab.boxes_ego(tok)

    trusts, covs = [], []
    for cam in T.CAMERAS:
        rec = tab.sensor_record(tok, cam)
        cov, _ = GE.camera_ground_coverage(rec, rng_m=rng_m, res=res, soft=True)
        covs.append(cov)
        trusts.append(fault_trust if cam == fault_camera else DEFAULT_TRUST)
    integ = GE.integrity_map(covs, trusts)

    # ---- panels ----
    cam_imgs = []
    for cam in T.CAMERAS:
        rec = tab.sensor_record(tok, cam)
        im = np.asarray(Image.open(rec["path"]).convert("RGB"))
        im = O.draw_lidar_depth(im, pts[current][:, :3], rec, radius=2, stride=4)
        im, n_drawn = O.draw_boxes_3d(im, boxes, rec)
        tag = "  [FAULT INJECTED]" if cam == fault_camera else ""
        im = O.chrome(im, cam + tag,
                      f"{n_drawn} boxes projected · trust "
                      f"{fault_trust if cam == fault_camera else DEFAULT_TRUST:.3f}")
        cam_imgs.append(Image.fromarray(im).resize((533, 300), Image.LANCZOS))

    n_dyn = int((motion == M.DYNAMIC).sum())
    # z_hi is tightened from the standalone render's 4.5 m: at panel size the
    # height ramp spends most of its range on tree canopy and building upper
    # storeys, which pushes everything at street level into the same narrow
    # band of blues. 3.2 m puts the ramp where the driving-relevant structure is.
    # Colour by height ABOVE THE FITTED GROUND PLANE, not raw z. The road is
    # not level -- this scene's plane is tilted ~1.5 deg -- so raw z shifts the
    # whole ramp across the frame and paints one side of a flat road as if it
    # were higher than the other.
    pts_h = pts.copy()
    pts_h[:, 2] = G.height_above_ground(pts[:, :3], plane).astype(np.float32)
    bev = B.render_bev(
        pts_h, boxes, rng_m=rng_m, size=560, ss=3, z_lo=-0.3, z_hi=3.0,
        canopy_above=3.0, motion=motion,
        dynamic_label=M.DYNAMIC,
        title="Multi-sweep BEV",
        subtitle=f"{len(pts):,} returns · {n_dyn:,} dynamic")
    st = G.grid_stats(occ)
    occ_img = B.render_occupancy(
        occ, size=560, title="Ray-cast occupancy",
        subtitle=f"free {100*st['free_frac']:.1f}%  occupied "
                 f"{100*st['occupied_frac']:.1f}%  unknown {100*st['unknown_frac']:.1f}%")
    ist = GE.integrity_stats(integ, occ, free_value=G.FREE)
    int_img = B.render_integrity(
        integ, size=560, title="Perception Integrity Map",
        subtitle=f"mean over free space {ist['mean_over_free_space']:.3f} · "
                 f"{100*ist['free_cells_below_0.3']:.1f}% of drivable below 0.3")

    W = 533 * 3
    canvas = Image.new("RGB", (W, 300 * 2 + 560), (6, 8, 12))
    for i, im in enumerate(cam_imgs):
        canvas.paste(im, (533 * (i % 3), 300 * (i // 3)))
    for i, arr in enumerate((bev, occ_img, int_img)):
        canvas.paste(Image.fromarray(arr), (int(i * (W - 560) / 2), 600))

    out_dir = Path(out_dir or ROOT / "outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{fault_camera}" if fault_camera else ""
    fig = out_dir / f"scene_view{tag}.png"
    canvas.save(fig)

    report = {
        "sample_token": tok, "scene": tab.scene_name(tok), "index": index,
        "n_sweeps": n_sweeps, "points": int(len(pts)),
        "points_current_sweep": int(current.sum()),
        "ground_plane": {"normal": [round(float(v), 5) for v in plane[0]],
                         "d": round(float(plane[1]), 4),
                         "tilt_deg": round(float(np.degrees(
                             np.arccos(abs(plane[0][2])))), 3),
                         "ground_return_frac": round(float(ground.mean()), 4)},
        "occupancy": {k: round(v, 5) if isinstance(v, float) else v
                      for k, v in st.items()} | occ_meta,
        "motion": {k: round(v, 4) if isinstance(v, float) else v
                   for k, v in motion_meta.items()} | {"n_dynamic": n_dyn},
        "boxes": {"total": len(boxes),
                  "with_lidar_returns": sum(1 for b in boxes if b["num_lidar_pts"] > 0)},
        "cameras": {c: {"trust": tr, "mean_coverage_weight": round(float(cv.mean()), 4)}
                    for c, tr, cv in zip(T.CAMERAS, trusts, covs)},
        "integrity": {k: round(v, 5) for k, v in ist.items()},
        "fault_camera": fault_camera,
        "figure": str(fig.relative_to(ROOT)),
        "seconds": round(time.time() - t0, 2),
    }
    rep = out_dir.parent / "artifacts" / f"scene_view{tag}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=30)
    ap.add_argument("--sweeps", type=int, default=10)
    ap.add_argument("--fault-camera", default=None,
                    help="drop this camera's trust, to see the integrity map respond")
    args = ap.parse_args()
    r = build(index=args.index, n_sweeps=args.sweeps, fault_camera=args.fault_camera)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
