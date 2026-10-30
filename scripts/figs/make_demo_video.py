#!/usr/bin/env python3
"""The SightOcc demo reel: one held-out scene, every panel from the dataset.

    hero        CAM_FRONT with projected 3D cuboids. Blue = the model found it,
                orange = it did not, labelled with class and range.
    inset       bird's-eye observability with the same markers and the ego.
    panels      front-left / front-right cameras, predicted occupancy, ground
                truth, observability, and UNVERIFIED FREE -- voxels the model
                calls free where the cameras have almost no evidence. That last
                panel is the paper's subject.
    telemetry   speed, mean observability, unverified-free share, miss count.

Nothing is illustrative. The detected/missed label is imported from
combined_object.py, so this cannot disagree with the reported numbers.

    python3 scripts/figs/make_demo_video.py --scene scene-0553
"""
import argparse, glob, importlib.util, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts/eval"))
_s = importlib.util.spec_from_file_location(
    "co", os.path.join(ROOT, "scripts/eval/combined_object.py"))
co = importlib.util.module_from_spec(_s); _s.loader.exec_module(co)

W, H = 1920, 936
BG, INK, INK2, EDGE = (11, 14, 19), (240, 239, 236), (150, 155, 164), (38, 43, 52)
BLUE, ORANGE, WARN = (57, 135, 229), (233, 104, 52), (237, 161, 0)
FREE, TAU = 17, 0.15
HERO = (28, 128, 1180, 664)                      # x, y, w, h
PAN_W, PAN_H, PAN_X, PAN_GAP = 330, 212, 1236, 12

CLS_RGB = np.array([
    (120,120,120),(230, 90, 90),(220,120,210),(255,150, 40),( 60,150,255),
    (255,200, 60),(255, 90,160),(255, 60, 60),(255,230, 90),(180,130, 80),
    ( 80,190,255),( 70, 70, 95),( 95, 85,120),(120,100,140),( 90,130, 95),
    (150,150,160),( 60,160, 90),( 16, 20, 27)], np.uint8)


def font(sz, bold=False):
    """matplotlib ships DejaVu. macOS system fonts are NOT reachable from the
    sandbox that renders this -- pointing at /System/Library fails silently to
    PIL's bitmap default and every label comes out unreadable."""
    import matplotlib
    b = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf")
    for n in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(os.path.join(b, n), sz)
        except Exception:
            pass
    return ImageFont.load_default()


def fit(img, w, h):
    """Letterbox a square BEV into a panel instead of stretching it."""
    side = min(w, h)
    bg = Image.new("RGB", (w, h), (11, 14, 19))
    bg.paste(img.resize((side, side), Image.NEAREST), ((w - side)//2, 0))
    return bg


def panel(d, x, y, w, h, title, f):
    d.rectangle([x-1, y-1, x+w, y+h], outline=EDGE, width=1)
    d.text((x + 7, y + 5), title, font=f, fill=INK2)


def bev_obs(obs, size):
    m = np.clip(obs.max(-1), 0, 1) ** 0.75
    img = np.dstack([(14 + m*116).astype(np.uint8),
                     (20 + m*164).astype(np.uint8),
                     (30 + m*206).astype(np.uint8)])
    return Image.fromarray(np.rot90(img), "RGB").resize(size, Image.NEAREST)


def bev_cls(cls, size):
    """Topmost non-free class per column, so the road is visible under objects."""
    top = np.full(cls.shape[:2], FREE, np.int16)
    for k in range(cls.shape[2]):
        s = cls[:, :, k]
        m = s != FREE
        top[m] = s[m]
    return Image.fromarray(np.rot90(CLS_RGB[top]), "RGB").resize(size, Image.NEAREST)


def bev_unverified(cls, obs, size):
    """Predicted free, but the cameras cannot see it. The paper's subject."""
    free = (cls == FREE)
    blind = obs < TAU
    bad = (free & blind).any(-1) & ~(free & ~blind).any(-1)
    seen = (free & ~blind).any(-1)
    img = np.zeros(cls.shape[:2] + (3,), np.uint8)
    img[...] = (16, 20, 27)
    img[seen] = (34, 58, 92)
    img[bad] = WARN
    img[(cls != FREE).any(-1) & ~bad] = (70, 76, 86)
    return Image.fromarray(np.rot90(img), "RGB").resize(size, Image.NEAREST), bad.mean()


def cuboid(c, size, R, Rs, ts, K, sx, sy, ox, oy):
    w, l, h = float(size[0]), float(size[1]), float(size[2])
    X = np.array([[ l/2, w/2, h/2], [ l/2,-w/2, h/2], [-l/2,-w/2, h/2], [-l/2, w/2, h/2],
                  [ l/2, w/2,-h/2], [ l/2,-w/2,-h/2], [-l/2,-w/2,-h/2], [-l/2, w/2,-h/2]])
    P = (R @ X.T).T + c
    Pc = (Rs.T @ (P - ts).T).T
    if (Pc[:, 2] <= 0.6).any():
        return None
    uv = (K @ Pc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return [(u*sx + ox, v*sy + oy) for u, v in uv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene-0553")
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

    f_t, f_s, f_l, f_n = font(40, True), font(19), font(15), font(16, True)
    rows = sorted([(i, r) for i, r in enumerate(index) if r["scene"] == a.scene],
                  key=lambda t: t[1]["timestamp"])
    frames, tot, miss_tot = [], 0, 0
    prev = None
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
        hx, hy, hw, hh = HERO
        cv.paste(Image.open(jp).convert("RGB").resize((hw, hh)), (hx, hy))
        d.rectangle([hx-1, hy-1, hx+hw, hy+hh], outline=EDGE, width=1)

        Re = co.qrot(np.asarray(cam["ego2global_rotation"], np.float64))
        te = np.asarray(cam["ego2global_translation"], np.float64)
        Rs = co.qrot(np.asarray(cam["sensor2ego_rotation"], np.float64))
        ts = np.asarray(cam["sensor2ego_translation"], np.float64)
        K = np.asarray(cam["intrinsic"], np.float64)
        sx, sy = hw / cam["width"], hh / cam["height"]

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
            objs.append((c, ann["size"][k], R, det, cat.split(".")[-1]))
            tot += 1
            if not det:
                miss_here += 1; miss_tot += 1

        taken = []
        for c, size, R, det, lab in sorted(objs, key=lambda o: -np.linalg.norm(o[0][:2])):
            col = BLUE if det else ORANGE
            pts = cuboid(c, size, R, Rs, ts, K, sx, sy, hx, hy)
            if pts:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                # an object at the frame edge projects to an enormous quad; a
                # box wider than most of the frame, or centred outside it, is
                # that artefact rather than a real detection to draw.
                if (max(xs)-min(xs) > hw*0.75 or max(ys)-min(ys) > hh*0.75
                        or not (hx < (min(xs)+max(xs))/2 < hx+hw)):
                    pts = None
            if pts:
                for i0, i1 in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                               (0,4),(1,5),(2,6),(3,7)):
                    d.line([pts[i0], pts[i1]], fill=col, width=2)
                tx = min(max(min(p[0] for p in pts), hx+2), hx+hw-150)
                ty = max(min(p[1] for p in pts) - 22, hy+2)
                while any(abs(ty-t) < 21 and abs(tx-u) < 150 for u, t in taken):
                    ty += 21
                if ty < hy+hh-20:
                    taken.append((tx, ty))
                    s = f"{lab} {np.linalg.norm(c[:2]):.1f} m"
                    d.rectangle([tx, ty, tx + 8 + 7.4*len(s), ty+20], fill=col)
                    d.text((tx+4, ty+2), s, font=f_l, fill=(10,12,16))

        # bird's-eye inset, top-right of the hero
        IS = 300
        ix, iy = hx + hw - IS - 14, hy + 14
        cv.paste(bev_obs(obs, (IS, IS)), (ix, iy))
        d.rectangle([ix-1, iy-1, ix+IS, iy+IS], outline=EDGE, width=1)
        d.text((ix+7, iy+5), "bird's-eye  observability", font=f_l, fill=INK)
        for c, size, R, det, lab in objs:
            bx, by = (c[0]+40)/80, (c[1]+40)/80
            if 0 <= bx <= 1 and 0 <= by <= 1:
                px, py = ix + (1-by)*IS, iy + (1-bx)*IS
                if det:
                    d.ellipse([px-4, py-4, px+4, py+4], outline=BLUE, width=2)
                else:
                    d.line([px-5,py-5,px+5,py+5], fill=ORANGE, width=3)
                    d.line([px-5,py+5,px+5,py-5], fill=ORANGE, width=3)
        ex, ey = ix+IS/2, iy+IS/2
        d.polygon([(ex, ey-8), (ex-6, ey+7), (ex+6, ey+7)], fill=INK)

        # right-hand panel stack
        py0 = 128
        cams = ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
        for i, cn in enumerate(cams):
            p = os.path.join(ROOT, "data/nuscenes", row["cams"][cn]["filename"])
            x = PAN_X + i*(PAN_W + PAN_GAP)
            if os.path.exists(p):
                cv.paste(Image.open(p).convert("RGB").resize((PAN_W, PAN_H-20)),
                         (x, py0+20))
            panel(d, x, py0, PAN_W, PAN_H, cn.replace("CAM_", "").lower(), f_l)
        PH = PAN_H - 20
        unver, unver_frac = bev_unverified(pcls, obs, (PH, PH))
        for i, (img, title) in enumerate((
                (fit(bev_cls(pcls, (PH, PH)), PAN_W, PH), "predicted occupancy"),
                (fit(bev_cls(gcls, (PH, PH)), PAN_W, PH), "ground truth  (Occ3D)"),
                (fit(bev_obs(obs, (PH, PH)), PAN_W, PH), "observability  O(v)"),
                (fit(unver, PAN_W, PH), "unverified free  -  called free, not seen"))):
            x = PAN_X + (i % 2)*(PAN_W + PAN_GAP)
            y = py0 + (1 + i//2)*(PAN_H + PAN_GAP)
            cv.paste(img, (x, y+20))
            panel(d, x, y, PAN_W, PAN_H, title, f_l)

        # header and telemetry
        d.text((28, 34), "SightOcc", font=f_t, fill=INK)
        d.text((232, 50), "what the cameras could actually see", font=f_s, fill=INK2)
        d.text((PAN_X, 52), f"{a.scene}   frame {n+1:02d}/{len(rows)}   "
                            f"Occ3D-nuScenes, held-out scene", font=f_s, fill=INK2)
        p1 = np.asarray(cam["ego2global_translation"], np.float64)
        spd = 0.0 if prev is None else np.linalg.norm(p1-prev[0])/max(
            (row["timestamp"]-prev[1])/1e6, 1e-3)
        prev = (p1, row["timestamp"])
        ty = hy + hh + 28
        for i, (lab, val) in enumerate((
                ("speed", f"{spd:4.1f} m/s"),
                ("objects", f"{len(objs)}"),
                ("missed", f"{miss_here}"),
                ("mean O(v)", f"{obs.max(-1).mean():.3f}"),
                ("unverified free", f"{unver_frac*100:4.1f}%"))):
            x = 28 + i*236
            d.text((x, ty), lab.upper(), font=f_l, fill=INK2)
            d.text((x, ty+20), val, font=font(26, True),
                   fill=WARN if lab == "unverified free" else INK)
        ly = H - 42
        d.ellipse([28, ly-7, 42, ly+7], outline=BLUE, width=3)
        d.text((52, ly-9), "detected", font=f_s, fill=INK2)
        d.line([172, ly-7, 186, ly+7], fill=ORANGE, width=4)
        d.line([172, ly+7, 186, ly-7], fill=ORANGE, width=4)
        d.text((198, ly-9), "missed by the model", font=f_s, fill=INK2)
        d.rectangle([410, ly-7, 424, ly+7], fill=WARN)
        d.text((436, ly-9), "free space with no camera evidence", font=f_s, fill=INK2)
        frames.append(cv)

    os.makedirs(os.path.join(ROOT, os.path.dirname(a.out)), exist_ok=True)
    import imageio.v2 as iio
    mp4 = os.path.join(ROOT, a.out + ".mp4")
    iio.mimsave(mp4, [np.asarray(f) for f in frames], fps=a.fps,
                codec="libx264", quality=8, macro_block_size=8)
    gif = os.path.join(ROOT, a.out + ".gif")
    gw = 920
    gh = int(round(gw * H / W / 2) * 2)          # keep the real aspect ratio
    sm = [f.resize((gw, gh), Image.LANCZOS).convert(
        "P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    sm[0].save(gif, save_all=True, append_images=sm[1:],
               duration=int(1000/a.fps), loop=0, optimize=True)
    print(f"{len(frames)} frames | {tot} objects | {miss_tot} missed "
          f"({100*miss_tot/max(tot,1):.1f}%)")
    for p in (mp4, gif):
        print(f"  {os.path.relpath(p, ROOT)}  {os.path.getsize(p)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
