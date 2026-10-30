#!/usr/bin/env python3
"""Render the occupancy volume THROUGH the camera, not from above.

The reference AV reels look real because every auxiliary panel is the
photograph with something painted on it, not an abstract plan. To do that from
a voxel grid you have to march each camera ray until it hits something, which
is what this does: one march per frame, from which the semantic, depth and
observability views are all derived.

Ray directions in the EGO frame are fixed, because sensor2ego is a rigid
mount, so they are computed once and reused for every frame of a scene.
"""
from __future__ import annotations
import numpy as np

RES, RNG, Z0, N, NZ, FREE = 0.4, 40.0, -1.0, 200, 16, 17


def ray_dirs(K, Rs, w, h, step):
    """Unit ray directions in the ego frame for a subsampled image."""
    vs, us = np.mgrid[0:h:step, 0:w:step]
    pix = np.stack([us, vs, np.ones_like(us)], -1).reshape(-1, 3).T
    d = Rs @ (np.linalg.inv(K) @ pix)
    d /= np.linalg.norm(d, axis=0, keepdims=True)
    return d, (h + step - 1)//step, (w + step - 1)//step


def march(cls, obs, dirs, ts, t0=1.0, t1=48.0, dt=0.4, chunk=24000,
          extra=None):
    """First non-free voxel along each ray.

    Returns (hit_class, depth, hit_obs); hit_class is FREE where the ray left
    the grid without hitting anything.
    """
    T = np.arange(t0, t1, dt)
    R = dirs.shape[1]
    hc = np.full(R, FREE, np.int16)
    hd = np.full(R, np.inf, np.float32)
    ho = np.zeros(R, np.float32)
    he = ([np.zeros(R, np.float32) for _ in extra] if extra is not None
          else None)
    for s in range(0, R, chunk):
        e = min(s + chunk, R)
        D = dirs[:, s:e]                                   # 3 x n
        P = ts[:, None, None] + D[:, :, None] * T[None, None, :]
        gi = np.floor((P[0] + RNG) / RES).astype(np.int32)
        gj = np.floor((P[1] + RNG) / RES).astype(np.int32)
        gk = np.floor((P[2] - Z0) / RES).astype(np.int32)
        ok = ((gi >= 0) & (gi < N) & (gj >= 0) & (gj < N)
              & (gk >= 0) & (gk < NZ))
        gi = np.clip(gi, 0, N-1); gj = np.clip(gj, 0, N-1); gk = np.clip(gk, 0, NZ-1)
        c = cls[gi, gj, gk]
        solid = ok & (c != FREE)
        any_hit = solid.any(1)
        first = np.argmax(solid, axis=1)
        idx = np.arange(e - s)
        hc[s:e] = np.where(any_hit, c[idx, first], FREE)
        hd[s:e] = np.where(any_hit, T[first], np.inf)
        ho[s:e] = np.where(any_hit, obs[gi[idx, first], gj[idx, first],
                                        gk[idx, first]], 0.0)
        if extra is not None:
            for q, vol in enumerate(extra):
                he[q][s:e] = np.where(any_hit, vol[gi[idx, first],
                                                   gj[idx, first],
                                                   gk[idx, first]], 0.0)
    return (hc, hd, ho) if extra is None else (hc, hd, ho, he)


def upsample(flat, hh, ww, step, w, h):
    a = flat.reshape(hh, ww)
    if step > 1:
        a = np.repeat(np.repeat(a, step, 0), step, 1)
    return a[:h, :w]


def turbo(x):
    """A compact perceptually-ordered ramp for depth. x in [0,1]."""
    stops = np.array([[ 48,  18, 120], [ 62,  86, 212], [ 46, 170, 196],
                      [ 92, 206, 108], [226, 206,  62], [238, 126,  44],
                      [196,  44,  38]], np.float32)
    pos = np.linspace(0, 1, len(stops))
    out = np.empty(x.shape + (3,), np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(x, pos, stops[:, ch])
    return out.astype(np.uint8)


CLS_RGB = np.array([
    (120,120,120),(228, 96, 96),(216,124,208),(252,152, 48),( 64,152,252),
    (252,200, 64),(252, 96,160),(252, 64, 64),(252,228, 92),(180,132, 84),
    ( 84,188,252),( 64, 66, 86),( 92, 84,116),(118, 98,138),( 88,128, 94),
    (148,148,158),( 56,158, 88),( 16, 20, 27)], np.uint8)
SKY = np.array([22, 27, 36], np.uint8)


def view_semantic(hc):
    img = CLS_RGB[hc].copy()
    img[hc == FREE] = SKY
    return img


def view_depth(hc, hd, t1=48.0):
    x = np.clip(np.where(np.isfinite(hd), hd, t1) / t1, 0, 1)
    img = turbo(1.0 - x)
    img[hc == FREE] = SKY
    return img


def view_obs(hc, ho):
    m = np.clip(ho, 0, 1) ** 0.75
    img = np.dstack([(16 + m*120).astype(np.uint8), (22 + m*168).astype(np.uint8),
                     (32 + m*210).astype(np.uint8)])
    img[hc == FREE] = SKY
    return img


def view_error(hc_g, pred_at_g, mask_at_g, hc_p, gt_at_p, maskp_at_p):
    """Prediction against ground truth, compared AT THE HIT VOXEL.

    An earlier version compared hit RANGES and was unusable: along a grazing
    ray down the road a few centimetres of height difference becomes metres of
    range, so the whole road surface read as a false positive. Comparing the
    occupancy of the voxel each ray actually lands in is well conditioned and
    is the same definition the paper uses for false free.

    RED   the label has a surface there and the model calls it empty.
    BLUE  the model puts a surface where the label, inside its own mask, has
          none.
    GREY  both agree, or the voxel is outside Occ3D's evaluated mask.
    """
    img = np.zeros(hc_g.shape + (3,), np.uint8)
    img[...] = SKY
    g_hit = hc_g != FREE
    p_hit = hc_p != FREE
    img[g_hit | p_hit] = (40, 45, 56)
    ev_g = g_hit & (mask_at_g > 0.5)
    img[ev_g] = (68, 76, 90)
    img[ev_g & (pred_at_g >= FREE - 0.5)] = (231, 72, 72)
    ev_p = p_hit & (maskp_at_p > 0.5) & ~ev_g
    img[ev_p & (gt_at_p >= FREE - 0.5)] = (72, 146, 214)
    return img
