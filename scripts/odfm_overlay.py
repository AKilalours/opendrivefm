"""Camera-view overlays: projected LiDAR depth and 3D box wireframes.

Drawing the LiDAR into the camera is the cheapest honest cross-check there is.
If the calibration, the pose chain or the projection has a sign error, the
points land on the sky or slide off the cars and it is obvious at a glance --
whereas a BEV alone will look perfectly plausible while being wrong.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm

import odfm_geom as GE

CLASS_COLOUR = {
    "vehicle.car": (86, 214, 255), "vehicle.truck": (86, 214, 255),
    "vehicle.bus": (86, 214, 255), "vehicle.trailer": (86, 214, 255),
    "vehicle.construction": (86, 214, 255),
    "vehicle.motorcycle": (255, 178, 64), "vehicle.bicycle": (255, 178, 64),
    "human.pedestrian": (255, 92, 132), "movable_object": (168, 178, 196),
}
_FONTS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
_fc = {}


def _font(sz):
    if sz not in _fc:
        import os
        f = None
        for p in _FONTS:
            if os.path.exists(p):
                try:
                    f = ImageFont.truetype(p, sz); break
                except Exception:
                    pass
        _fc[sz] = f or ImageFont.load_default()
    return _fc[sz]


def class_colour(name):
    for k, v in CLASS_COLOUR.items():
        if name.startswith(k):
            return v
    return (168, 178, 196)


def draw_lidar_depth(img, pts_ego, rec, max_depth=60.0, radius=2, alpha=0.8,
                     stride=3):
    """Splat projected returns coloured by depth (near = warm, far = cool).

    `stride` subsamples the cloud. Every return drawn at full density covers the
    image so completely that the overlay stops being an overlay -- the whole
    point is to check that the points land on the right objects, which requires
    still being able to see the objects.
    """
    im = Image.fromarray(np.asarray(img, np.uint8)).convert("RGB")
    uv, z, valid = GE.project_to_image(np.asarray(pts_ego)[:, :3], rec)
    if not valid.any():
        return np.array(im)
    u, v, d = uv[valid, 0], uv[valid, 1], z[valid]
    if stride > 1:
        u, v, d = u[::stride], v[::stride], d[::stride]

    # far points first, so near returns win the overlap and the depth ordering
    # in the image matches the depth ordering in the world
    order = np.argsort(-d)
    u, v, d = u[order], v[order], d[order]
    t = np.clip(d / max_depth, 0, 1)
    col = (cm.get_cmap("turbo")(1.0 - t)[:, :3] * 255).astype(np.uint8)

    dr = ImageDraw.Draw(im, "RGBA")
    a = int(alpha * 255)
    for (x, y, c) in zip(u, v, col):
        dr.ellipse([x - radius, y - radius, x + radius, y + radius],
                   fill=(int(c[0]), int(c[1]), int(c[2]), a))
    return np.array(im)


_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),          # front face
          (4, 5), (5, 6), (6, 7), (7, 4),          # back face
          (0, 4), (1, 5), (2, 6), (3, 7)]          # connecting


def draw_boxes_3d(img, boxes, rec, min_pts=1, label=True):
    """Wireframe 3D boxes projected into the camera.

    A box is drawn only when ALL eight corners are in front of the camera. A
    box straddling the image plane needs near-plane clipping to draw correctly;
    without it the corners behind the camera project to wild coordinates and
    the wireframe explodes across the frame. Skipping those is the honest
    cheap option -- every other camera in the rig sees them anyway.
    """
    im = Image.fromarray(np.asarray(img, np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(im, "RGBA")
    W, H = im.size
    drawn = 0
    for b in boxes:
        if b.get("num_lidar_pts", 1) < min_pts:
            continue
        corners = GE.box_corners_ego(b)
        cam = GE.ego_to_cam(corners, rec)
        if (cam[:, 2] <= 0.5).any():
            continue
        uvw = cam @ rec["intrinsic"].T
        uv = uvw[:, :2] / cam[:, 2:3]
        if uv[:, 0].max() < 0 or uv[:, 0].min() > W or \
           uv[:, 1].max() < 0 or uv[:, 1].min() > H:
            continue
        col = class_colour(b["category"])
        for i, j in _EDGES:
            dr.line([tuple(uv[i]), tuple(uv[j])], fill=col + (235,), width=2)
        # shade the front face so heading is readable at a glance
        dr.polygon([tuple(uv[k]) for k in (0, 1, 2, 3)], fill=col + (48,))
        drawn += 1
        if label:
            name = b["category"].split(".")[-1]
            x, y = float(uv[:, 0].min()), float(uv[:, 1].min())
            dr.text((max(2, x), max(2, y - 14)), name, fill=col, font=_font(13))
    return np.array(im), drawn


def chrome(img, title, subtitle=None):
    im = Image.fromarray(np.asarray(img, np.uint8)).convert("RGB")
    W, H = im.size
    dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle([0, 0, W, 30], fill=(8, 10, 15, 216))
    dr.text((10, 7), title, fill=(226, 232, 240), font=_font(15))
    if subtitle:
        dr.rectangle([0, H - 24, W, H], fill=(8, 10, 15, 216))
        dr.text((10, H - 18), subtitle, fill=(150, 160, 178), font=_font(12))
    return np.array(im)
