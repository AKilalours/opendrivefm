#!/usr/bin/env python3
"""Figure 4. One sealed-test frame, in image space.

(a) the camera the object falls in
(b) ground-truth 3D boxes projected
(c) the model's predicted occupancy projected, painter's algorithm
(d) geometric observability projected the same way

The object is selected by scripts/viz/pick_tokens2.py: the largest strictly
missed dynamic object in the sealed test split whose maximum observability
is below 0.05. Nothing here is drawn by hand.
"""
import glob, importlib.util, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES, RNG_M, Z0, FREE = 0.4, 40.0, -1.0, 17
INK, C_GT, C_HOT = "#1a1a1a", "#25c04a", "#ff2d2d"
CB = {"CAM_FRONT": 0., "CAM_FRONT_LEFT": 55., "CAM_FRONT_RIGHT": -55.,
      "CAM_BACK_LEFT": 110., "CAM_BACK_RIGHT": -110., "CAM_BACK": 180.}
# nuScenes-Occ3D class colours, 0..16
PAL = np.array([
    [0, 0, 0], [255, 120, 50], [255, 192, 203], [255, 255, 0], [0, 150, 245],
    [0, 255, 255], [255, 127, 0], [255, 0, 0], [255, 240, 150], [135, 60, 0],
    [160, 32, 240], [255, 0, 255], [139, 137, 137], [75, 0, 75], [150, 240, 80],
    [230, 230, 250], [0, 175, 0]], np.float32) / 255.


def project(ijk, cam, md):
    """voxel indices -> pixel coords and depth, in the given camera"""
    g = np.stack(ijk, 1).astype(np.float64) + 0.5
    p = np.empty_like(g)
    p[:, 0] = g[:, 0] * RES - RNG_M
    p[:, 1] = g[:, 1] * RES - RNG_M
    p[:, 2] = g[:, 2] * RES + Z0
    Rs = md.qrot(np.asarray(cam["sensor2ego_rotation"], np.float64))
    ts = np.asarray(cam["sensor2ego_translation"], np.float64)
    K = np.asarray(cam["intrinsic"], np.float64)
    C = (p - ts) @ Rs
    ok = C[:, 2] > 1.0
    C = C[ok]
    uv = (K @ C.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv, C[:, 2], ok, float(K[0, 0])


def surface(ax, ijk, rgb, cam, md, W, H, D=9, alpha=0.55, fill=4.0):
    """Depth-buffered projection: nearest voxel wins each pixel block.

    Voxel centres project to isolated points, so after the z-buffer pass a
    bounded nearest-neighbour fill closes the gaps between them. The fill is
    capped at `fill` blocks, so it never invents surface where none projects.
    """
    from scipy import ndimage
    uv, z, ok, _ = project(ijk, cam, md)
    rgb = rgb[ok]
    u = np.round(uv[:, 0] / D).astype(int)
    v = np.round(uv[:, 1] / D).astype(int)
    h, w = int(np.ceil(H / D)), int(np.ceil(W / D))
    m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z, rgb = u[m], v[m], z[m], rgb[m]
    o = np.argsort(-z)
    col = np.zeros((h, w, 3), np.float32)
    hit = np.zeros((h, w), bool)
    col[v[o], u[o]] = rgb[o]
    hit[v, u] = True
    dist, idx = ndimage.distance_transform_edt(~hit, return_indices=True)
    near = dist <= fill
    col = col[idx[0], idx[1]]
    buf = np.zeros((h, w, 4), np.float32)
    buf[..., :3] = col
    buf[..., 3] = np.where(near, alpha, 0.0)
    ax.imshow(buf, extent=[0, W, H, 0], interpolation="bilinear", zorder=3)


def main():
    sp = importlib.util.spec_from_file_location(
        "md", os.path.join(ROOT, "scripts/eval/missed_detection.py"))
    md = importlib.util.module_from_spec(sp); sp.loader.exec_module(md)
    tk = json.load(open(os.path.join(ROOT, "outputs/artifacts/figure_tokens.json")))
    tok, meta = tk["fig3_token"], tk["fig3_object"]
    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    row = next(r for r in index if r["token"] == tok); fi = index.index(row)

    gc = np.load(glob.glob(os.path.join(
        ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts", "*", tok,
        "labels.npz"))[0])["semantics"].astype(np.int16)
    pc = np.load(os.path.join(ROOT, "data/preds/preds_voxel",
                              tok + ".npz"))["cls"].astype(np.int16)
    ob = np.load(os.path.join(ROOT, "data/pack/obs_max",
                              tok + ".npy")).astype(np.float32) / 255.

    ann = np.load(os.path.join(ROOT, "data/pack/annotations_val.npz"),
                  allow_pickle=True)
    names = ann["category_names"]; fr = ann["frame"]
    order = np.argsort(fr, kind="stable")
    bounds = np.searchsorted(fr[order], np.arange(len(index) + 1))
    cam0 = row["cams"]["CAM_FRONT"]
    Re = md.qrot(np.asarray(cam0["ego2global_rotation"], np.float64))
    te = np.asarray(cam0["ego2global_translation"], np.float64)

    boxes, target = [], None
    for k in order[bounds[fi]:bounds[fi + 1]]:
        cid = md.CAT2OCC.get(str(names[ann["category"][k]]))
        if cid is None or cid not in md.DYN:
            continue
        c = Re.T @ (ann["translation"][k].astype(np.float64) - te)
        R = Re.T @ md.qrot(np.asarray(ann["rotation"][k], np.float64))
        sz = ann["size"][k].astype(np.float64)
        vx = md.box_voxels(c, sz, R, 0.0)
        if vx is None:
            continue
        gm = gc[vx] == cid
        miss = (not np.isin(pc[vx], list(md.DYN)).any()) and float((pc[vx] == FREE).mean()) >= 0.9
        boxes.append((c, R, sz, miss and int(gm.sum()) == meta["n"]))
        if miss and int(gm.sum()) == meta["n"]:
            target = (c, R, sz)
    cx, cy = target[0][0], target[0][1]
    bearing = np.degrees(np.arctan2(cy, cx))
    use = min(CB, key=lambda c: abs((bearing - CB[c] + 180) % 360 - 180))
    cam = row["cams"][use]
    fn = cam["filename"]
    jp = fn if os.path.exists(fn) else os.path.join(
        ROOT, "dataset/nuscenes", fn.split("nuscenes/")[-1])
    im = plt.imread(jp); H, W = im.shape[:2]

    def corners(c, R, sz):
        hw, hl, hh = sz / 2
        q = np.array([[sx * hl, sy * hw, sz_ * hh] for sx in (-1, 1)
                      for sy in (-1, 1) for sz_ in (-1, 1)])
        Rs = md.qrot(np.asarray(cam["sensor2ego_rotation"], np.float64))
        ts = np.asarray(cam["sensor2ego_translation"], np.float64)
        K = np.asarray(cam["intrinsic"], np.float64)
        Cm = ((R @ q.T).T + c - ts) @ Rs
        if not (Cm[:, 2] > 0.5).all():
            return None
        uv = (K @ Cm.T).T
        return uv[:, :2] / uv[:, 2:3]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7.6, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.52))
    fig.subplots_adjust(left=0.004, right=0.996, top=0.90, bottom=0.135,
                        wspace=0.02)
    for a in ax:
        a.imshow(im); a.set_xlim(0, W); a.set_ylim(H, 0)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(0.5); s.set_color("#b5b5b5")

    tgt_uv = corners(*target)

    # (b) ground truth
    for c, R, sz, is_t in boxes:
        uv = corners(c, R, sz)
        if uv is None:
            continue
        u0, v0, u1, v1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
        ax[1].add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False,
                                  lw=1.6 if is_t else 0.9, ec=C_GT, zorder=5))

    # (c) the model's predicted occupancy, nearest surface
    ijk = np.nonzero(pc != FREE)
    surface(ax[2], ijk, PAL[np.clip(pc[ijk], 0, 16)], cam, md, W, H)

    # (d) observability, sampled on the ground-truth surface
    ijk = np.nonzero(gc != FREE)
    surface(ax[3], ijk, plt.get_cmap("viridis")(ob[ijk])[:, :3],
            cam, md, W, H, alpha=0.72)

    for i, lab in ((2, "predicted free, conf 1.00"), (3, "$O = 0.00$")):
        col = C_HOT if i == 2 else "white"
        u0, v0 = tgt_uv[:, 0].min(), tgt_uv[:, 1].min()
        u1, v1 = tgt_uv[:, 0].max(), tgt_uv[:, 1].max()
        pad = 6
        ax[i].add_patch(Rectangle((u0 - pad, v0 - pad), u1 - u0 + 2 * pad,
                                  v1 - v0 + 2 * pad, fill=False, lw=1.3,
                                  ec=col, ls=(0, (3.5, 2.5)), zorder=7))
        ax[i].annotate(lab, xy=((u0 + u1) / 2, v0 - pad), xycoords="data",
                       xytext=(0, 26), textcoords="offset points",
                       ha="center", va="bottom", fontsize=6.4, color=col,
                       arrowprops=dict(arrowstyle="->", color=col, lw=0.9),
                       zorder=8)

    for i, t_ in enumerate(["(a)  input", "(b)  ground truth",
                            "(c)  prediction", "(d)  observability"]):
        ax[i].set_title(t_, fontsize=7.8, loc="left", pad=2.5)

    cax = fig.add_axes([0.800, 0.072, 0.140, 0.026])
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1]); cb.ax.tick_params(labelsize=5.6, length=1.6, pad=1)
    cb.set_label("$O(v;\\mathcal{G})$", fontsize=6.0, labelpad=1.5)
    cb.outline.set_linewidth(0.4)

    fig.text(0.006, 0.022,
             f"{meta['cat'].split('.')[-1]} at {meta['rng']} m  ·  "
             f"{meta['n']} ground-truth voxels  ·  predicted free throughout  ·  "
             f"max O = {meta['obs']:.2f}  ·  {use.replace('_',' ').lower()}  ·  "
             f"token {tok[:8]}", fontsize=5.8, color="#5a5a5a")

    out = os.path.join(ROOT, "outputs/figures/fig2_qualitative.pdf")
    fig.savefig(out, dpi=600); fig.savefig(out.replace(".pdf", ".png"), dpi=300)
    print("token", tok, meta, "| cam", use)
    print("wrote", out)


if __name__ == "__main__":
    main()
