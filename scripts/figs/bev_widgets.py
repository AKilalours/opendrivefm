#!/usr/bin/env python3
"""Bird's-eye widgets: ego car, range rings, error map, camera coverage.

Everything here is drawn from the dataset's own calibration and labels. The
camera-coverage wedges in particular are computed from each sensor's intrinsic
and extrinsic, not drawn by eye, which is what makes them an explanation of
the blind regions rather than a decoration.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

RES, RNG, FREE = 0.4, 40.0, 17
N = 200

CLS_RGB = np.array([
    (120,120,120),(228, 96, 96),(216,124,208),(252,152, 48),( 64,152,252),
    (252,200, 64),(252, 96,160),(252, 64, 64),(252,228, 92),(180,132, 84),
    ( 84,188,252),( 64, 66, 86),( 92, 84,116),(118, 98,138),( 88,128, 94),
    (148,148,158),( 56,158, 88),( 16, 20, 27)], np.uint8)


def ego_car(d, cx, cy, L, col=(238, 240, 245), heading_up=True):
    """A top-down car silhouette, forward = up. L is bumper-to-bumper pixels."""
    w = L * 0.44
    hl, hw = L / 2, w / 2
    body = [(cx - hw*0.74, cy - hl), (cx + hw*0.74, cy - hl),
            (cx + hw, cy - hl*0.52), (cx + hw, cy + hl*0.66),
            (cx + hw*0.84, cy + hl), (cx - hw*0.84, cy + hl),
            (cx - hw, cy + hl*0.66), (cx - hw, cy - hl*0.52)]
    d.polygon(body, fill=col)
    dark = tuple(int(c*0.33) for c in col)
    d.polygon([(cx - hw*0.66, cy - hl*0.46), (cx + hw*0.66, cy - hl*0.46),
               (cx + hw*0.52, cy - hl*0.80), (cx - hw*0.52, cy - hl*0.80)],
              fill=dark)                                       # windscreen
    d.polygon([(cx - hw*0.62, cy + hl*0.30), (cx + hw*0.62, cy + hl*0.30),
               (cx + hw*0.52, cy + hl*0.66), (cx - hw*0.52, cy + hl*0.66)],
              fill=dark)                                       # rear glass
    d.line([(cx - hw*0.66, cy - hl*0.42), (cx + hw*0.66, cy - hl*0.42)],
           fill=dark, width=max(int(L*0.04), 1))


def rings(d, cx, cy, half_px, font, every=10, col=(64, 72, 86),
          ink=(116, 124, 140)):
    for r_m in range(every, int(RNG) + 1, every):
        r = r_m / RNG * half_px
        d.ellipse([cx-r, cy-r, cx+r, cy+r], outline=col, width=1)
        d.text((cx + 5, cy - r - 2), f"{r_m}", font=font, fill=ink)


def top_class(cls):
    top = np.full(cls.shape[:2], FREE, np.int16)
    for k in range(cls.shape[2]):
        sl = cls[:, :, k]; m = sl != FREE
        top[m] = sl[m]
    return top


def img_cls(cls, size):
    return Image.fromarray(np.rot90(CLS_RGB[top_class(cls)]), "RGB").resize(
        size, Image.NEAREST)


def img_obs(obs, size):
    """Observability is continuous, so bilinear here is honest, not a blur."""
    m = np.clip(obs.max(-1), 0, 1) ** 0.75
    a = np.dstack([(14 + m*116).astype(np.uint8), (20 + m*164).astype(np.uint8),
                   (30 + m*206).astype(np.uint8)])
    return Image.fromarray(np.rot90(a), "RGB").resize(size, Image.BILINEAR)


def img_unverified(cls, obs, size, tau=0.15):
    free = cls == FREE
    seen = (free & (obs >= tau)).any(-1)
    bad = (free & (obs < tau)).any(-1) & ~seen
    img = np.full(cls.shape[:2] + (3,), 0, np.uint8)
    img[...] = (16, 20, 27)
    img[seen] = (34, 58, 92)
    img[bad] = (237, 161, 0)
    img[(cls != FREE).any(-1) & ~bad] = (70, 76, 86)
    return Image.fromarray(np.rot90(img), "RGB").resize(size, Image.NEAREST), float(bad.mean())


def img_error(pcls, gcls, size, mask=None):
    """Prediction against Occ3D, inside the benchmark's own evaluation region.

    Two facts make the naive version meaningless and are handled here. Occ3D
    scores only voxels in mask_camera, about 14% of the grid, because the rest
    was never observed and its label is not evidence. And the comparison has to
    be per VOXEL: the prediction is far denser than the label, so a column is
    essentially never entirely free and a column-level diff reports no false
    free at all.

    Per column the most severe category present inside the mask is shown.
    FALSE FREE -- label occupied, model says empty -- is the dangerous one.
    """
    if mask is None:
        mask = np.ones_like(pcls, bool)
    mask = mask.astype(bool)
    pf, gf = pcls == FREE, gcls == FREE
    false_free = mask & gf.__invert__() & pf
    false_solid = mask & gf & ~pf
    wrong_cls = mask & ~gf & ~pf & (pcls != gcls)
    right_occ = mask & ~gf & ~pf & (pcls == gcls)
    img = np.full(pcls.shape[:2] + (3,), 0, np.uint8)
    img[...] = (14, 17, 23)                       # outside the eval region
    img[mask.any(-1)] = (26, 31, 40)
    img[right_occ.any(-1)] = (66, 74, 88)
    img[false_solid.any(-1)] = (72, 146, 214)
    img[wrong_cls.any(-1)] = (150, 120, 220)
    img[false_free.any(-1)] = (231, 72, 72)       # drawn last: most severe
    ev = max(int(mask.sum()), 1)
    return (Image.fromarray(np.rot90(img), "RGB").resize(size, Image.NEAREST),
            float(false_free.sum()) / ev)


def img_coverage(cams, qrot, size, obs=None):
    """The six camera fields of view in plan, from intrinsics and extrinsics.

    Half-angle is atan((w/2)/fx) per camera, direction is the sensor's yaw in
    the ego frame. Overlap is additive, so the bright wedges are where two
    cameras agree and the dark gaps are the structural blind spots that
    observability is measuring.
    """
    S = size[0]
    acc = np.zeros((S, S), np.float32)
    yy, xx = np.mgrid[0:S, 0:S]
    # plan coords: forward = up (-y), left = -x
    fx_m = (S/2 - xx) / (S/2) * RNG          # ego +y (left)
    fy_m = (S/2 - yy) / (S/2) * RNG          # ego +x (forward)
    rad = np.hypot(fx_m, fy_m)
    ang = np.arctan2(fx_m, fy_m)             # 0 = forward, +ve = left
    for cn, c in cams.items():
        K = np.asarray(c["intrinsic"], np.float64)
        half = np.arctan2(c["width"] / 2.0, K[0, 0])
        R = qrot(np.asarray(c["sensor2ego_rotation"], np.float64))
        fwd = R @ np.array([0.0, 0.0, 1.0])          # camera +z in ego
        yaw = np.arctan2(fwd[1], fwd[0])
        dth = np.abs(np.angle(np.exp(1j * (ang - yaw))))
        acc += ((dth < half) & (rad < RNG * 0.98)).astype(np.float32)
    img = np.zeros((S, S, 3), np.uint8)
    img[...] = (15, 19, 26)
    inside = rad < RNG * 0.98
    img[inside & (acc == 0)] = (44, 30, 24)          # structural blind wedge
    for lvl, col in ((1, (34, 78, 128)), (2, (72, 140, 206)), (3, (146, 202, 246))):
        img[acc == lvl] = col
    img[acc >= 4] = (198, 226, 252)
    # The six nuScenes cameras tile 360 degrees, so azimuthal coverage is
    # complete and a "blind fraction" here would always read zero and imply
    # the rig sees everything. What this panel actually shows is where two or
    # more cameras overlap, which is where observability is highest. The dark
    # regions in O(v) come from RESOLUTION and OCCLUSION, not from FOV gaps.
    return Image.fromarray(img, "RGB"), float((acc[inside] >= 2).mean())


def draw_trails(d, objs, hist, Re, te, ox, oy, side, blue, orange):
    """Each track's recent path, held in global coordinates and projected into
    the CURRENT ego frame, so a trail is the object's real motion relative to
    the vehicle rather than a smear of screen positions."""
    for o in objs:
        trk, det = o[6], o[3]
        h = hist.get(trk, [])
        if len(h) < 2:
            continue
        pts = []
        for gx, gy in h:
            e = Re.T @ (np.array([gx, gy, te[2]], np.float64) - te)
            u = ox + (1 - (e[1] + RNG) / (2*RNG)) * side
            v = oy + (1 - (e[0] + RNG) / (2*RNG)) * side
            if ox <= u <= ox+side and oy <= v <= oy+side:
                pts.append((u, v))
        if len(pts) > 1:
            d.line(pts, fill=(blue if det else orange), width=2)


def draw_footprints(d, objs, ox, oy, side, blue, orange, fill_det=True,
                    label_font=None):
    for o in objs:
        c, size_, R, det = o[0], o[1], o[2], o[3]
        did = o[5] if len(o) > 5 else None
        w_, l_ = float(size_[0]), float(size_[1])
        cn = np.array([[ l_/2, w_/2], [ l_/2,-w_/2], [-l_/2,-w_/2], [-l_/2, w_/2]])
        pw = (R[:2, :2] @ cn.T).T + c[:2]
        q = [(ox + (1 - (y+RNG)/(2*RNG))*side, oy + (1 - (x+RNG)/(2*RNG))*side)
             for x, y in pw]
        if all(ox <= u <= ox+side and oy <= v <= oy+side for u, v in q):
            col = blue if det else orange
            if det and fill_det:
                d.polygon(q, fill=col)
            else:
                d.line(q + [q[0]], fill=col, width=2)
            if label_font is not None and not det and did is not None:
                d.text((max(u for u, _ in q) + 3, min(v for _, v in q) - 2),
                       f"#{did:02d}", font=label_font, fill=col)
