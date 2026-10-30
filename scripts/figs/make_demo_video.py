#!/usr/bin/env python3
"""Render the SightOcc demo: a real scene, frame by frame, from the dataset.

Left   CAM_FRONT, with every annotated dynamic object marked by what the model
       did with it: detected (blue circle) or missed (orange cross).
Right  bird's-eye observability O(v), max over height. Dark means the cameras
       have little or no geometric evidence there. The same object markers are
       drawn on top, so a missed object sitting inside a dark wedge is visible
       as the thing it is.

Nothing here is illustrative. The miss definition is imported from
combined_object.py, the observability maps and predictions are the same files
every number in the paper is computed from.

    python3 scripts/figs/make_demo_video.py --scene scene-0553
"""
import argparse, importlib.util, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts/eval"))
_s = importlib.util.spec_from_file_location(
    "co", os.path.join(ROOT, "scripts/eval/combined_object.py"))
co = importlib.util.module_from_spec(_s); _s.loader.exec_module(co)

BG, INK, INK2, PANEL = (13, 17, 23), (240, 239, 236), (160, 163, 170), (22, 27, 34)
BLUE, ORANGE = (57, 135, 229), (233, 104, 52)
W, H = 1680, 772
CAMW, CAMH = 980, 551
BEVPX = 600


def font(sz, bold=False):
    """matplotlib ships DejaVu, which is present wherever this runs.

    macOS system fonts are NOT reachable from the sandboxed shell that renders
    this, so do not reintroduce /System/Library paths -- they fail silently to
    PIL's bitmap default and every label comes out tiny.
    """
    import matplotlib
    base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf")
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for cand in (os.path.join(base, name), os.path.join(base, "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(cand, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def bev_image(obs):
    """O(v) max over height -> a 200x200 blue field, dark = no evidence."""
    m = obs.max(-1)                                   # (200, 200)
    m = np.clip(m, 0, 1) ** 0.75
    r = (18 + m * 120).astype(np.uint8)
    g = (24 + m * 168).astype(np.uint8)
    b = (34 + m * 210).astype(np.uint8)
    img = np.dstack([r, g, b])
    img = np.rot90(img)                               # forward -> up
    return Image.fromarray(img, "RGB").resize((BEVPX, BEVPX), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene-0553")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--out", default="docs/figs/demo")
    a = ap.parse_args()

    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    ann = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"),
                  allow_pickle=True)
    names = ann["category_names"]
    fr = ann["frame"]; order = np.argsort(fr, kind="stable")
    bounds = np.searchsorted(fr[order], np.arange(len(index) + 1))
    import glob
    gt_map = {os.path.basename(os.path.dirname(p)): p for p in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}

    f_big, f_mid, f_sm = font(34, True), font(20), font(17)
    frames, n_miss_tot, n_tot = [], 0, 0
    rows = [(fi, r) for fi, r in enumerate(index) if r["scene"] == a.scene]
    rows.sort(key=lambda t: t[1]["timestamp"])
    print(f"{a.scene}: {len(rows)} frames")

    for n, (fi, row) in enumerate(rows):
        tok = row["token"]
        pp = os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz")
        op = os.path.join(ROOT, "data/pack/obs_max", tok + ".npy")
        cam = row["cams"]["CAM_FRONT"]
        jp = os.path.join(ROOT, "data/nuscenes", cam["filename"])
        if not (os.path.exists(pp) and os.path.exists(op)
                and os.path.exists(jp) and tok in gt_map):
            continue
        obs = np.load(op).astype(np.float32) / 255.0
        pcls = np.load(pp)["cls"].astype(np.int16)
        gcls = np.load(gt_map[tok])["semantics"].astype(np.int16)

        canvas = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(canvas)

        cim = Image.open(jp).convert("RGB").resize((CAMW, CAMH))
        canvas.paste(cim, (24, 96))
        bev = bev_image(obs)
        canvas.paste(bev, (W - BEVPX - 24, 96))

        Re = co.qrot(np.asarray(cam["ego2global_rotation"], np.float64))
        te = np.asarray(cam["ego2global_translation"], np.float64)
        Rs = co.qrot(np.asarray(cam["sensor2ego_rotation"], np.float64))
        ts = np.asarray(cam["sensor2ego_translation"], np.float64)
        K = np.asarray(cam["intrinsic"], np.float64)
        sx, sy = CAMW / cam["width"], CAMH / cam["height"]

        miss_here = 0
        for k in order[bounds[fi]:bounds[fi + 1]]:
            cid = co.CAT2OCC.get(str(names[ann["category"][k]]))
            if cid is None or cid not in co.DYN or ann["num_lidar_pts"][k] < 1:
                continue
            c = Re.T @ (ann["translation"][k].astype(np.float64) - te)
            R = Re.T @ co.qrot(np.asarray(ann["rotation"][k], np.float64))
            vx = co.box_voxels(c, ann["size"][k].astype(np.float64), R, 0.0)
            if vx is None or not (gcls[vx] == cid).any():
                continue
            detected = bool(np.isin(pcls[vx], list(co.DYN)).any())
            col = BLUE if detected else ORANGE
            n_tot += 1
            if not detected:
                miss_here += 1; n_miss_tot += 1

            # BEV marker: ego frame x forward, y left; grid is +/-40 m
            bx = (c[0] + 40.0) / 80.0; by = (c[1] + 40.0) / 80.0
            if 0 <= bx <= 1 and 0 <= by <= 1:
                px = W - BEVPX - 24 + (1 - by) * BEVPX
                py = 96 + (1 - bx) * BEVPX
                if detected:
                    d.ellipse([px-7, py-7, px+7, py+7], outline=col, width=3)
                else:
                    d.line([px-8, py-8, px+8, py+8], fill=col, width=4)
                    d.line([px-8, py+8, px+8, py-8], fill=col, width=4)

            # camera marker
            pc = Rs.T @ (c - ts)
            if pc[2] > 0.6:
                uv = K @ pc; u, v = uv[0]/uv[2]*sx + 24, uv[1]/uv[2]*sy + 96
                if 24 < u < 24 + CAMW and 96 < v < 96 + CAMH:
                    if detected:
                        d.ellipse([u-9, v-9, u+9, v+9], outline=col, width=3)
                    else:
                        d.line([u-10, v-10, u+10, v+10], fill=col, width=4)
                        d.line([u-10, v+10, u+10, v-10], fill=col, width=4)

        # ego marker
        ex, ey = W - BEVPX - 24 + BEVPX/2, 96 + BEVPX/2
        d.polygon([(ex, ey-11), (ex-8, ey+9), (ex+8, ey+9)], fill=INK)

        d.text((24, 26), "SightOcc", font=f_big, fill=INK)
        d.text((200, 38), "what the cameras could actually see",
               font=f_mid, fill=INK2)
        d.text((24, 96 + CAMH + 14), "CAM_FRONT", font=f_sm, fill=INK2)
        d.text((W - BEVPX - 24, 96 + BEVPX + 14),
               "observability  O(v)   dark = no camera evidence",
               font=f_sm, fill=INK2)

        ly = H - 42
        d.ellipse([24, ly-7, 38, ly+7], outline=BLUE, width=3)
        d.text((48, ly-9), "detected", font=f_sm, fill=INK2)
        d.line([150, ly-7, 164, ly+7], fill=ORANGE, width=4)
        d.line([150, ly+7, 164, ly-7], fill=ORANGE, width=4)
        d.text((176, ly-9), "missed by the model", font=f_sm, fill=INK2)
        d.text((420, ly-9), f"{a.scene}   frame {n+1:02d}/{len(rows)}"
                            f"   missed this frame: {miss_here}",
               font=f_sm, fill=INK2)
        d.text((800, ly-9), "Occ3D-nuScenes, held-out scene, real predictions",
               font=f_sm, fill=INK2)
        frames.append(canvas)

    os.makedirs(os.path.join(ROOT, os.path.dirname(a.out)), exist_ok=True)
    mp4 = os.path.join(ROOT, a.out + ".mp4")
    gif = os.path.join(ROOT, a.out + ".gif")
    import imageio.v2 as iio
    iio.mimsave(mp4, [np.asarray(f) for f in frames], fps=a.fps,
                codec="libx264", quality=8, macro_block_size=8)
    small = [f.resize((W//2, H//2), Image.LANCZOS) for f in frames]
    small[0].save(gif, save_all=True, append_images=small[1:],
                  duration=int(1000/a.fps), loop=0, optimize=True)
    print(f"{len(frames)} frames | {n_tot} objects | {n_miss_tot} missed "
          f"({100*n_miss_tot/max(n_tot,1):.1f}%)")
    for p in (mp4, gif):
        print(f"  {os.path.relpath(p, ROOT)}  {os.path.getsize(p)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
