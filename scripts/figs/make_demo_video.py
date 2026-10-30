#!/usr/bin/env python3
"""The SightOcc demo reel. Every panel is read from the dataset.

    hero      CAM_FRONT with the occupancy projected onto the road plane:
              teal where the model calls it clear AND the cameras saw it,
              ORANGE where it calls it clear with almost no evidence.
              3D boxes over it, filled where the model found the object and
              hollow where it missed it.
    inset     bird's-eye observability with range rings and plan-view
              footprints at true extent and heading.
    bottom    the predicted volume in 3D, observability, unverified free.
    right     the full six-camera rig, telemetry, and the key.

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
import render3d, cam_overlay

W, H = 1920, 1030
BG, INK, INK2, EDGE = (11, 14, 19), (240, 239, 236), (150, 155, 164), (38, 43, 52)
BLUE, ORANGE, WARN, TEAL = (57, 135, 229), (233, 104, 52), (237, 161, 0), (0, 168, 180)
FREE, TAU = 17, 0.15
HERO = (28, 120, 1180, 664)
ROW_Y, ROW_H, GAP = 798, 212, 12
COL_X, COL_W, COL_H = 1236, 322, 186
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
    sandbox that renders this; pointing at /System/Library fails silently to
    PIL's bitmap default and every label comes out unreadable."""
    import matplotlib
    b = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf")
    for n in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(os.path.join(b, n), sz)
        except Exception:
            pass
    return ImageFont.load_default()


def panel(d, x, y, w, h, title, f):
    d.rectangle([x-1, y-1, x+w, y+h], outline=EDGE, width=1)
    d.text((x + 8, y + 5), title, font=f, fill=INK2)


def fit(img, w, h):
    side = min(w, h)
    bg = Image.new("RGB", (w, h), BG)
    bg.paste(img.resize((side, side), Image.NEAREST), ((w - side)//2, 0))
    return bg


def bev_obs(obs, size):
    m = np.clip(obs.max(-1), 0, 1) ** 0.75
    img = np.dstack([(14 + m*116).astype(np.uint8), (20 + m*164).astype(np.uint8),
                     (30 + m*206).astype(np.uint8)])
    return Image.fromarray(np.rot90(img), "RGB").resize(size, Image.NEAREST)


CLS_RGB = np.array([
    (120,120,120),(228, 96, 96),(216,124,208),(252,152, 48),( 64,152,252),
    (252,200, 64),(252, 96,160),(252, 64, 64),(252,228, 92),(180,132, 84),
    ( 84,188,252),( 64, 66, 86),( 92, 84,116),(118, 98,138),( 88,128, 94),
    (148,148,158),( 56,158, 88),( 16, 20, 27)], np.uint8)


def bev_cls(cls, size):
    """Topmost non-free class per column, so the road shows under objects."""
    top = np.full(cls.shape[:2], FREE, np.int16)
    for k in range(cls.shape[2]):
        sl = cls[:, :, k]; m = sl != FREE
        top[m] = sl[m]
    return Image.fromarray(np.rot90(CLS_RGB[top]), "RGB").resize(size, Image.NEAREST)


def bev_unverified(cls, obs, size):
    free = cls == FREE
    seen = (free & (obs >= TAU)).any(-1)
    bad = (free & (obs < TAU)).any(-1) & ~seen
    img = np.full(cls.shape[:2] + (3,), 0, np.uint8)
    img[...] = (16, 20, 27)
    img[seen] = (34, 58, 92)
    img[bad] = WARN
    img[(cls != FREE).any(-1) & ~bad] = (70, 76, 86)
    return Image.fromarray(np.rot90(img), "RGB").resize(size, Image.NEAREST)


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
    f_t, f_s, f_l = font(38, True), font(19), font(15)
    f_v = font(27, True)
    rows = sorted([(i, r) for i, r in enumerate(index) if r["scene"] == a.scene],
                  key=lambda t: t[1]["timestamp"])
    frames, tot, miss_tot, prev = [], 0, 0, None
    hx, hy, hw, hh = HERO
    print(f"{a.scene}: {len(rows)} frames")

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
        gcls = np.load(gt_map[tok])["semantics"].astype(np.int16)

        cv = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(cv)

        # hero: the real frame with the occupancy projected onto the road plane
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
            objs.append((c, ann["size"][k], R, det,
                         NICE.get(cat.split(".")[-1], cat.split(".")[-1])))
            tot += 1
            if not det:
                miss_here += 1; miss_tot += 1

        ov = Image.new("RGBA", (hw, hh), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
        taken = []
        for c, size, R, det, lab in sorted(objs, key=lambda o: -np.linalg.norm(o[0][:2])):
            P = corners(c, size, R)
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
                od.polygon([pts[i] for i in (0, 1, 2, 3)], fill=col + (60,))
            for i0, i1 in EDGES:
                od.line([pts[i0], pts[i1]], fill=col + (255,), width=2 if det else 3)
            tx = min(max(min(xs), 2), hw-150)
            ty = max(min(ys) - 21, 2)
            while any(abs(ty-t) < 21 and abs(tx-u) < 150 for u, t in taken):
                ty += 21
            if ty < hh - 20:
                taken.append((tx, ty))
                s = f"{lab} {np.linalg.norm(c[:2]):.0f} m"
                od.rectangle([tx, ty, tx + 8 + 7.4*len(s), ty+20], fill=col + (255,))
                od.text((tx+4, ty+2), s, font=f_l, fill=(10, 12, 16, 255))
        cv.paste(ov, (hx, hy), ov)
        d.text((hx+12, hy+10), "CAM_FRONT   occupancy on the road plane:  "
                               "TEAL verified clear,  ORANGE no camera evidence",
               font=f_l, fill=INK)

        # bird's-eye inset
        IS = 236
        ix, iy = hx + hw - IS - 14, hy + 14
        cv.paste(bev_obs(obs, (IS, IS)), (ix, iy))
        d.rectangle([ix-1, iy-1, ix+IS, iy+IS], outline=EDGE, width=1)
        d.text((ix+7, iy+5), "bird's-eye  O(v)", font=f_l, fill=INK)
        ex, ey = ix + IS/2, iy + IS/2
        for rng in (10, 20, 30):
            r = rng / 40.0 * IS / 2
            d.ellipse([ex-r, ey-r, ex+r, ey+r], outline=(58, 66, 80), width=1)
            d.text((ex + 4, ey - r - 2), f"{rng}", font=f_l, fill=(112,120,134))
        for c, size, R, det, lab in objs:
            w_, l_ = float(size[0]), float(size[1])
            cn = np.array([[ l_/2, w_/2], [ l_/2,-w_/2], [-l_/2,-w_/2], [-l_/2, w_/2]])
            pw = (R[:2, :2] @ cn.T).T + c[:2]
            q = [(ix + (1 - (y+40)/80)*IS, iy + (1 - (x+40)/80)*IS) for x, y in pw]
            if all(ix <= u <= ix+IS and iy <= v <= iy+IS for u, v in q):
                col = BLUE if det else ORANGE
                if det:
                    d.polygon(q, fill=col)
                else:
                    d.line(q + [q[0]], fill=col, width=2)
        d.polygon([(ex, ey-9), (ex-6, ey+7), (ex+6, ey+7)], fill=INK)

        # bottom row
        # Four square bird's-eye panels. The 3D volume used to sit here, but
        # at 586x192 a perspective render is an unreadable letterbox slit; 3D
        # belongs in the interactive viewer, where it can be orbited.
        PH = ROW_H - 20
        bw = (1180 - 3*GAP) // 4
        free_m = pcls == FREE
        seen_m = (free_m & (obs >= TAU)).any(-1)
        unver_frac = float(((free_m & (obs < TAU)).any(-1) & ~seen_m).mean())
        for i, (img, title) in enumerate((
                (fit(bev_cls(pcls, (PH, PH)), bw, PH), "predicted occupancy"),
                (fit(bev_cls(gcls, (PH, PH)), bw, PH), "ground truth  (Occ3D)"),
                (fit(bev_obs(obs, (PH, PH)), bw, PH), "observability  O(v)"),
                (fit(bev_unverified(pcls, obs, (PH, PH)), bw, PH),
                 "unverified free"))):
            x = 28 + i*(bw + GAP)
            cv.paste(img, (x, ROW_Y + 20))
            panel(d, x, ROW_Y, bw, ROW_H, title, f_l)

        # right column: the full rig
        for i, (cn_, lab6) in enumerate(SURROUND):
            x = COL_X + (i % 2)*(COL_W + GAP)
            y = 120 + (i // 2)*(COL_H + GAP)
            cp = os.path.join(ROOT, "data/nuscenes", row["cams"][cn_]["filename"])
            if os.path.exists(cp):
                cv.paste(Image.open(cp).convert("RGB").resize((COL_W, COL_H-20)),
                         (x, y+20))
            panel(d, x, y, COL_W, COL_H, lab6, f_l)

        # header, telemetry, key
        d.text((28, 30), "SightOcc", font=f_t, fill=INK)
        d.text((226, 46), "what the cameras could actually see", font=f_s, fill=INK2)
        d.text((COL_X, 46), f"{a.scene}   frame {n+1:02d}/{len(rows)}   "
                            f"Occ3D-nuScenes, held-out scene", font=f_s, fill=INK2)
        p1 = np.asarray(cam["ego2global_translation"], np.float64)
        spd = 0.0 if prev is None else np.linalg.norm(p1-prev[0])/max(
            (row["timestamp"]-prev[1])/1e6, 1e-3)
        prev = (p1, row["timestamp"])
        ty0 = 120 + 3*(COL_H + GAP) + 10
        for i, (lab, val, warn) in enumerate((
                ("SPEED", f"{spd:.1f} m/s", False),
                ("OBJECTS", f"{len(objs)}", False),
                ("MISSED", f"{miss_here}", False),
                ("MEAN O(V)", f"{obs.max(-1).mean():.3f}", False),
                ("UNVERIFIED FREE", f"{unver_frac*100:.1f}%", True))):
            x = COL_X + (i % 3)*228
            y = ty0 + (i // 3)*62
            d.text((x, y), lab, font=f_l, fill=INK2)
            d.text((x, y+19), val, font=f_v, fill=WARN if warn else INK)
        ky = ty0 + 136
        for i, (sw, txt, filled) in enumerate((
                (BLUE, "object the model detected", True),
                (ORANGE, "object the model missed", False),
                (TEAL, "road verified clear by the cameras", True),
                (WARN, "called free, no camera evidence", True))):
            y = ky + i*30
            if filled:
                d.rectangle([COL_X, y, COL_X+20, y+15], fill=sw)
            else:
                d.rectangle([COL_X, y, COL_X+20, y+15], outline=sw, width=3)
            d.text((COL_X+32, y-2), txt, font=f_s, fill=INK2)
        d.text((COL_X, ky + 132), "Boxes are dataset annotations; the colour is "
                                  "the model's verdict.", font=f_l, fill=(108,114,126))
        frames.append(cv)
        if (n+1) % 10 == 0:
            print(f"  {n+1}/{len(rows)}", flush=True)

    os.makedirs(os.path.join(ROOT, os.path.dirname(a.out)), exist_ok=True)
    import imageio.v2 as iio
    mp4 = os.path.join(ROOT, a.out + ".mp4")
    iio.mimsave(mp4, [np.asarray(f) for f in frames], fps=a.fps,
                codec="libx264", quality=8, macro_block_size=8)
    gif = os.path.join(ROOT, a.out + ".gif")
    gw = 820; gh = int(round(gw * H / W / 2) * 2)
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
