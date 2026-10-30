#!/usr/bin/env python3
"""The SightOcc demo reel, 2560x1440. Every panel is read from the dataset.

The hero is the CAM_FRONT frame at its NATIVE 1600x900 with the occupancy
projected onto the road plane, so nothing in it is upscaled. Around it:

    bird's-eye   observability, range rings, the ego car, and every object at
                 its true plan extent and heading
    bottom       predicted occupancy, Occ3D ground truth, the error map inside
                 the benchmark's own evaluation mask, and unverified free
    right        the full six-camera rig, observability, camera coverage
                 computed from the six intrinsics and extrinsics, telemetry

The detected/missed verdict is imported from combined_object.py, so the reel
cannot disagree with the reported numbers.

    python3 scripts/figs/make_demo_video.py --scene scene-0796
"""
import argparse, glob, importlib.util, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts/eval"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_s = importlib.util.spec_from_file_location(
    "co", os.path.join(ROOT, "scripts/eval/combined_object.py"))
co = importlib.util.module_from_spec(_s); _s.loader.exec_module(co)
import cam_overlay, bev_widgets as bw

W, H = 2560, 1600
BG, INK, INK2, EDGE = (11, 14, 19), (240, 239, 236), (150, 155, 164), (38, 43, 52)
BLUE, ORANGE, WARN, TEAL = (57, 135, 229), (233, 104, 52), (237, 161, 0), (0, 168, 180)
RED = (231, 72, 72)
FREE, TAU, GAP = 17, 0.15, 14
HERO = (32, 124, 1600, 900)
ROW_Y, ROW_H = 1042, 372
KEY_Y = 1436
COL_X, COL_W, COL_H = 1664, 425, 240
TRAIL = 12          # frames of track history drawn in the bird's-eye
SURROUND = (("CAM_FRONT_LEFT", "front left"), ("CAM_FRONT", "front"),
            ("CAM_FRONT_RIGHT", "front right"), ("CAM_BACK_LEFT", "back left"),
            ("CAM_BACK", "back"), ("CAM_BACK_RIGHT", "back right"))
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
NICE = {"rigid": "bus", "bendy": "bus", "adult": "pedestrian",
        "child": "pedestrian", "police_officer": "pedestrian",
        "construction_worker": "pedestrian", "wheelchair": "pedestrian",
        "stroller": "pedestrian", "personal_mobility": "pedestrian"}


def font(sz, bold=False):
    """matplotlib ships DejaVu. macOS system fonts are NOT reachable from the
    sandbox that renders this; /System/Library paths fail silently to PIL's
    bitmap default and every label comes out unreadable."""
    import matplotlib
    b = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf")
    for n in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(os.path.join(b, n), sz)
        except Exception:
            pass
    return ImageFont.load_default()


def panel(d, x, y, w, h, title, f, sub=None):
    d.rectangle([x-1, y-1, x+w, y+h], outline=EDGE, width=1)
    d.text((x + 9, y + 6), title, font=f, fill=INK2)
    if sub:
        d.text((x + w - 9 - f.getlength(sub), y + 6), sub, font=f, fill=(110,118,132))


def square(img, w, h):
    side = min(w, h)
    bg = Image.new("RGB", (w, h), BG)
    bg.paste(img.resize((side, side), Image.NEAREST), ((w - side)//2, 0))
    return bg, side, (w - side)//2


def corners(c, size, R):
    w_, l_, h_ = float(size[0]), float(size[1]), float(size[2])
    X = np.array([[ l_/2, w_/2, h_/2], [ l_/2,-w_/2, h_/2], [-l_/2,-w_/2, h_/2],
                  [-l_/2, w_/2, h_/2], [ l_/2, w_/2,-h_/2], [ l_/2,-w_/2,-h_/2],
                  [-l_/2,-w_/2,-h_/2], [-l_/2, w_/2,-h_/2]])
    return (R @ X.T).T + c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene-0796")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--out", default="docs/figs/demo")
    a = ap.parse_args()

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    ann = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"), allow_pickle=True)
    names = ann["category_names"]
    fr = ann["frame"]; order = np.argsort(fr, kind="stable")
    bounds = np.searchsorted(fr[order], np.arange(len(index) + 1))
    gt_map = {os.path.basename(os.path.dirname(p)): p for p in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}
    f_t, f_s, f_l, f_v = font(46, True), font(21), font(17), font(32, True)
    rows = sorted([(i, r) for i, r in enumerate(index) if r["scene"] == a.scene],
                  key=lambda t: t[1]["timestamp"])
    frames, tot, miss_tot, prev = [], 0, 0, None
    # Real tracking: nuScenes gives a persistent instance id per annotation, so
    # identity across frames is read from the dataset rather than associated by
    # us. hist keeps each track's recent GLOBAL positions, which are projected
    # into the current ego frame when drawn; streak counts consecutive frames a
    # track has been missed, which is the number a safety case cares about.
    disp_id, hist, streak = {}, {}, {}
    hx, hy, hw, hh = HERO
    print(f"{a.scene}: {len(rows)} frames at {W}x{H}")

    for n, (fi, row) in enumerate(rows):
        tok = row["token"]
        pp = os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz")
        op = os.path.join(ROOT, "data/pack/obs_max", tok + ".npy")
        cam = row["cams"]["CAM_FRONT"]
        jp = os.path.join(ROOT, "data/nuscenes", cam["filename"])
        if not (os.path.exists(pp) and os.path.exists(op) and os.path.exists(jp)
                and tok in gt_map):
            continue
        obs = np.load(op).astype(np.float32) / 255.0
        pcls = np.load(pp)["cls"].astype(np.int16)
        z = np.load(gt_map[tok])
        gcls = z["semantics"].astype(np.int16)
        mcam = z["mask_camera"]

        cv = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(cv)

        # ---- hero: native-resolution camera with the occupancy on the road
        im = np.asarray(Image.open(jp).convert("RGB").resize((hw, hh)))
        K = np.asarray(cam["intrinsic"], np.float64).copy()
        K[0] *= hw / cam["width"]; K[1] *= hh / cam["height"]
        Rs = co.qrot(np.asarray(cam["sensor2ego_rotation"], np.float64))
        ts = np.asarray(cam["sensor2ego_translation"], np.float64)
        cv.paste(Image.fromarray(cam_overlay.overlay(im, pcls, obs, K, Rs, ts)),
                 (hx, hy))
        d.rectangle([hx-1, hy-1, hx+hw, hy+hh], outline=EDGE, width=1)

        Re = co.qrot(np.asarray(cam["ego2global_rotation"], np.float64))
        te = np.asarray(cam["ego2global_translation"], np.float64)
        objs, miss_here = [], 0
        for k in order[bounds[fi]:bounds[fi+1]]:
            cat = str(names[ann["category"][k]])
            cid = co.CAT2OCC.get(cat)
            if cid is None or cid not in co.DYN or ann["num_lidar_pts"][k] < 1:
                continue
            c = Re.T @ (ann["translation"][k].astype(np.float64) - te)
            R = Re.T @ co.qrot(np.asarray(ann["rotation"][k], np.float64))
            vx = co.box_voxels(c, ann["size"][k].astype(np.float64), R, 0.0)
            if vx is None or not (gcls[vx] == cid).any():
                continue
            det = bool(np.isin(pcls[vx], list(co.DYN)).any())
            gm = gcls[vx] == cid
            o_obj = float(obs[vx][gm].max()) if gm.any() else 0.0
            trk = int(ann["track"][k])
            if trk not in disp_id:
                disp_id[trk] = len(disp_id) + 1
            streak[trk] = 0 if det else streak.get(trk, 0) + 1
            hist.setdefault(trk, []).append(
                tuple(ann["translation"][k].astype(np.float64)[:2]))
            hist[trk] = hist[trk][-TRAIL:]
            objs.append((c, ann["size"][k], R, det,
                         NICE.get(cat.split(".")[-1], cat.split(".")[-1]),
                         disp_id[trk], trk, o_obj,
                         int(ann["visibility"][k])))
            tot += 1
            if not det:
                miss_here += 1; miss_tot += 1

        ov = Image.new("RGBA", (hw, hh), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
        taken = []
        for c, size_, R, det, lab, did, trk, o_obj, vis in sorted(
                objs, key=lambda o: -np.linalg.norm(o[0][:2])):
            P = corners(c, size_, R)
            Pc = (Rs.T @ (P - ts).T).T
            if (Pc[:, 2] <= 0.8).any():
                continue
            uv = (K @ Pc.T).T; uv = uv[:, :2] / uv[:, 2:3]
            xs, ys = uv[:, 0], uv[:, 1]
            if (max(xs) < 0 or min(xs) > hw or max(ys) < 0 or min(ys) > hh
                    or max(xs)-min(xs) > hw*0.6):
                continue
            col = BLUE if det else ORANGE
            pts = [(u, v) for u, v in uv]
            if det:
                od.polygon([pts[i] for i in (0, 1, 2, 3)], fill=col + (58,))
            for i0, i1 in EDGES:
                od.line([pts[i0], pts[i1]], fill=col + (255,), width=2 if det else 3)
            tx = min(max(min(xs), 2), hw-170); ty = max(min(ys) - 24, 2)
            while any(abs(ty-t) < 24 and abs(tx-u) < 170 for u, t in taken):
                ty += 24
            if ty < hh - 22:
                taken.append((tx, ty))
                s = f"#{did:02d} {lab} {np.linalg.norm(c[:2]):.0f} m"
                od.rectangle([tx, ty, tx + 10 + 8.3*len(s), ty+22], fill=col + (255,))
                od.text((tx+5, ty+3), s, font=f_l, fill=(10, 12, 16, 255))
        cv.paste(ov, (hx, hy), ov)
        d.text((hx+14, hy+11), "CAM_FRONT 1600x900 native   occupancy on the road "
                               "plane:  TEAL verified clear,  ORANGE no camera "
                               "evidence", font=f_l, fill=INK)

        # ---- bird's-eye inset
        IS = 320
        ix, iy = hx + hw - IS - 16, hy + 16
        cv.paste(bw.img_obs(obs, (IS, IS)), (ix, iy))
        d.rectangle([ix-1, iy-1, ix+IS, iy+IS], outline=EDGE, width=1)
        d.text((ix+9, iy+6), "bird's-eye  O(v)   metres", font=f_l, fill=INK)
        ex, ey = ix + IS/2, iy + IS/2
        bw.rings(d, ex, ey, IS/2, f_l)
        bw.draw_trails(d, objs, hist, Re, te, ix, iy, IS, BLUE, ORANGE)
        bw.draw_footprints(d, objs, ix, iy, IS, BLUE, ORANGE, label_font=f_l)
        bw.ego_car(d, ex, ey, 30)

        # ---- bottom row
        PH = ROW_H - 24
        pw_ = (1600 - 3*GAP) // 4
        unv_img, unv_frac = bw.img_unverified(pcls, obs, (PH, PH))
        err_img, ff = bw.img_error(pcls, gcls, (PH, PH), mcam)
        for i, (img, title, sub) in enumerate((
                (bw.img_cls(pcls, (PH, PH)), "predicted occupancy", "model"),
                (bw.img_cls(gcls, (PH, PH)), "ground truth", "Occ3D"),
                (err_img, "error vs ground truth", f"false free {ff*100:.1f}%"),
                (unv_img, "unverified free", f"{unv_frac*100:.1f}%"))):
            x = 32 + i*(pw_ + GAP)
            sq, side, pad = square(img, pw_, PH)
            cv.paste(sq, (x, ROW_Y + 24))
            panel(d, x, ROW_Y, pw_, ROW_H, title, f_l, sub)
            bw.ego_car(d, x + pad + side/2, ROW_Y + 24 + side/2, 17)

        # ---- right column: the rig
        for i, (cn_, lab6) in enumerate(SURROUND):
            x = COL_X + (i % 2)*(COL_W + GAP)
            y = 124 + (i // 2)*(COL_H + GAP)
            cp = os.path.join(ROOT, "data/nuscenes", row["cams"][cn_]["filename"])
            if os.path.exists(cp):
                cv.paste(Image.open(cp).convert("RGB").resize((COL_W, COL_H-24)),
                         (x, y+24))
            panel(d, x, y, COL_W, COL_H, lab6, f_l)

        # ---- right column: observability and camera coverage
        y2 = 124 + 3*(COL_H + GAP)
        EH = 230
        cov, blind = bw.img_coverage(row["cams"], co.qrot, (EH-24, EH-24))
        for i, (img, title, sub) in enumerate((
                (bw.img_obs(obs, (EH-24, EH-24)), "observability  O(v)",
                 f"mean {obs.max(-1).mean():.3f}"),
                (cov, "camera FOV overlap", f"2+ cams {blind*100:.0f}%"))):
            x = COL_X + i*(COL_W + GAP)
            sq, side, pad = square(img, COL_W, EH-24)
            cv.paste(sq, (x, y2 + 24))
            panel(d, x, y2, COL_W, EH, title, f_l, sub)
            bw.rings(d, x + pad + side/2, y2 + 24 + side/2, side/2, f_l,
                     every=20, col=(70, 80, 96), ink=(132, 142, 158))
            bw.ego_car(d, x + pad + side/2, y2 + 24 + side/2, 15)

        # ---- header, telemetry, key
        d.text((32, 34), "SightOcc", font=f_t, fill=INK)
        d.text((272, 54), "what the cameras could actually see", font=f_s, fill=INK2)
        d.text((COL_X, 54), f"{a.scene}   frame {n+1:02d}/{len(rows)}   "
                            f"Occ3D-nuScenes, held-out scene", font=f_s, fill=INK2)
        p1 = np.asarray(cam["ego2global_translation"], np.float64)
        spd = 0.0 if prev is None else np.linalg.norm(p1-prev[0])/max(
            (row["timestamp"]-prev[1])/1e6, 1e-3)
        prev = (p1, row["timestamp"])
        ty0 = y2 + EH + 16
        for i, (lab, val, warn) in enumerate((
                ("SPEED", f"{spd:.1f} m/s", False),
                ("OBJECTS", f"{len(objs)}", False),
                ("MISSED", f"{miss_here}", False),
                ("MEAN O(V)", f"{obs.max(-1).mean():.3f}", False),
                ("UNVERIFIED FREE", f"{unv_frac*100:.1f}%", True),
                ("FALSE FREE", f"{ff*100:.1f}%", True))):
            x = COL_X + (i % 3)*290
            y = ty0 + (i // 3)*70
            d.text((x, y), lab, font=f_l, fill=INK2)
            d.text((x, y+22), val, font=f_v, fill=WARN if warn else INK)
        # the key, one row under the bird's-eye panels
        kx = 32
        for sw, txt, filled in ((BLUE, "model detected it", True),
                                (ORANGE, "model missed it", False),
                                (TEAL, "road verified clear", True),
                                (WARN, "free, no camera evidence", True),
                                (RED, "label occupied, model empty", True),
                                ((150, 120, 220), "wrong class", True)):
            if filled:
                d.rectangle([kx, KEY_Y, kx+22, KEY_Y+16], fill=sw)
            else:
                d.rectangle([kx, KEY_Y, kx+22, KEY_Y+16], outline=sw, width=3)
            d.text((kx+32, KEY_Y-3), txt, font=f_s, fill=INK2)
            kx += 42 + f_s.getlength(txt)
        for i, line in enumerate((
                "Boxes are dataset annotations carrying their nuScenes instance "
                "ids; the colour is the model's verdict. Error is scored inside",
                "Occ3D's camera mask, the 14% it evaluates. The six cameras tile "
                "360 deg, so dark O(v) is resolution and occlusion, not FOV gaps.")):
            d.text((32, KEY_Y + 28 + i*20), line, font=f_l, fill=(108,114,126))

        # ---- the missed-object track table
        TY, TH = ty0 + 150, 300
        panel(d, COL_X, TY, COL_W*2 + GAP, TH, "MISSED OBJECTS  (tracked)", f_l,
              f"{miss_here} now")
        cols = (0, 60, 230, 330, 450, 600)
        for h, cx_ in zip(("ID", "CLASS", "RANGE", "O(v)", "FRAMES MISSED"), cols):
            d.text((COL_X + 12 + cx_, TY + 32), h, font=f_l, fill=(120,128,142))
        d.line([COL_X + 12, TY + 54, COL_X + COL_W*2 + GAP - 12, TY + 54],
               fill=EDGE, width=1)
        shown = sorted([o for o in objs if not o[3]],
                       key=lambda o: np.linalg.norm(o[0][:2]))[:8]
        for r, o in enumerate(shown):
            y = TY + 64 + r*29
            rng_m = np.linalg.norm(o[0][:2])
            st = streak.get(o[6], 1)
            for txt, cx_, col in ((f"#{o[5]:02d}", cols[0], ORANGE),
                                  (o[4], cols[1], INK),
                                  (f"{rng_m:.1f} m", cols[2], INK),
                                  (f"{o[7]:.3f}", cols[3], WARN if o[7] < .2 else INK),
                                  (f"{st}", cols[4], RED if st >= 3 else INK2)):
                d.text((COL_X + 12 + cx_, y), str(txt), font=f_s, fill=col)
        if not shown:
            d.text((COL_X + 12, TY + 64), "none in this frame", font=f_s,
                   fill=(110,118,132))
        frames.append(cv)
        if (n+1) % 10 == 0:
            print(f"  {n+1}/{len(rows)}", flush=True)

    os.makedirs(os.path.join(ROOT, os.path.dirname(a.out)), exist_ok=True)
    import imageio.v2 as iio
    mp4 = os.path.join(ROOT, a.out + ".mp4")
    iio.mimsave(mp4, [np.asarray(f) for f in frames], fps=a.fps,
                codec="libx264", quality=8, macro_block_size=8)
    gif = os.path.join(ROOT, a.out + ".gif")
    gw = 860; gh = int(round(gw * H / W / 2) * 2)
    sm = [f.resize((gw, gh), Image.LANCZOS).convert(
        "P", palette=Image.ADAPTIVE, colors=96) for f in frames]
    sm[0].save(gif, save_all=True, append_images=sm[1:],
               duration=int(1000/a.fps), loop=0, optimize=True)
    print(f"{len(frames)} frames | {tot} objects | {miss_tot} missed "
          f"({100*miss_tot/max(tot,1):.1f}%)")
    for p in (mp4, gif):
        print(f"  {os.path.relpath(p, ROOT)}  {os.path.getsize(p)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
