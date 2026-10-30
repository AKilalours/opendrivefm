#!/usr/bin/env python3
"""A perspective 3D view of the occupancy volume, z-buffered, pure numpy.

The grid is 200x200x16 at 0.4 m. Drawing every voxel as geometry is far too
slow, so each voxel centre is projected once, depth-sorted, and scattered into
an RGB array with a painter's pass -- one vectorised operation per frame.

Ego frame: x forward, y left, z up. Voxel (i,j,k) centre is
    x = (i+0.5)*RES - RNG,  y = (j+0.5)*RES - RNG,  z = (k+0.5)*RES + Z0
"""
import numpy as np

RES, RNG, Z0, FREE = 0.4, 40.0, -1.0, 17

# Static world is rendered desaturated so the things that matter read against
# it: dynamic objects keep their hue, the unverified-free carpet is the only
# warm colour in the frame. Classes 11-16 are drivable surface, other flat,
# sidewalk, terrain, manmade, vegetation.
STATIC = (11, 12, 13, 14, 15, 16)

CLS_RGB = np.array([
    (120,120,120),(228, 96, 96),(216,124,208),(252,152, 48),( 64,152,252),
    (252,200, 64),(252, 96,160),(252, 64, 64),(252,228, 92),(180,132, 84),
    ( 84,188,252),( 64, 66, 86),( 92, 84,116),(118, 98,138),( 88,128, 94),
    (148,148,158),( 56,158, 88),(  0,  0,  0)], np.float32)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    f = np.asarray(target, np.float64) - np.asarray(eye, np.float64)
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, np.float64)); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, -u, f]), np.asarray(eye, np.float64)   # rows: right, down, fwd


def project(P, Rv, eye, fpx, W, H):
    C = (Rv @ (P - eye).T).T
    z = C[:, 2]
    ok = z > 1.0
    u = np.full(len(P), -1e9); v = np.full(len(P), -1e9)
    u[ok] = C[ok, 0] / z[ok] * fpx + W / 2.0
    v[ok] = C[ok, 1] / z[ok] * fpx + H * 0.40
    return u, v, z, ok


def render(pcls, obs, W=1180, H=620, eye=(-13.0, 0.0, 14.0),
           target=(26.0, 0.0, -0.5), fpx=None, bg=(11, 14, 19), tau=0.15):
    """Returns (RGB uint8 array, projector) so callers can overlay in 3D."""
    fpx = fpx or W * 0.78
    Rv, e = look_at(eye, target)
    img = np.empty((H, W, 3), np.float32); img[...] = bg
    zbuf = np.full((H, W), 1e9, np.float32)

    def splat(P, col, size):
        u, v, z, ok = project(P, Rv, e, fpx, W, H)
        if not ok.any():
            return
        o = np.argsort(-z[ok])                       # far to near
        uu = u[ok][o]; vv = v[ok][o]; zz = z[ok][o]; cc = col[ok][o]
        s = np.clip((size * fpx / zz), 1, 7).astype(np.int32)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                m = (np.abs(dx) <= s // 2) & (np.abs(dy) <= s // 2)
                if not m.any():
                    continue
                xi = (uu[m] + dx).astype(np.int32); yi = (vv[m] + dy).astype(np.int32)
                zi = zz[m]; ci = cc[m]
                inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                xi, yi, zi, ci = xi[inb], yi[inb], zi[inb], ci[inb]
                keep = zi < zbuf[yi, xi]
                xi, yi, zi, ci = xi[keep], yi[keep], zi[keep], ci[keep]
                zbuf[yi, xi] = zi
                img[yi, xi] = ci

    ii, jj, kk = np.nonzero(pcls != FREE)
    if len(ii):
        P = np.stack([(ii + .5) * RES - RNG, (jj + .5) * RES - RNG,
                      (kk + .5) * RES + Z0], 1)
        lab = pcls[ii, jj, kk]
        c = CLS_RGB[lab].copy()
        st = np.isin(lab, STATIC)
        g = c[st] @ np.array([0.30, 0.59, 0.11], np.float32)
        c[st] = np.stack([g, g, g * 1.06], 1) * 0.82
        # brightness carries observability: unseen structure goes dark
        w = (0.26 + 0.74 * np.clip(obs[ii, jj, kk], 0, 1)[:, None])
        splat(P, np.clip(c * w, 0, 255), RES)

    # unverified free: called free, almost no camera evidence. Ground carpet.
    free = pcls == FREE
    bad = (free & (obs < tau)).any(-1) & ~(free & (obs >= tau)).any(-1)
    bi, bj = np.nonzero(bad)
    if len(bi):
        P = np.stack([(bi + .5) * RES - RNG, (bj + .5) * RES - RNG,
                      np.full(len(bi), Z0 + 0.9)], 1)
        splat(P, np.tile(np.array([[237., 161., 0.]]), (len(bi), 1)), RES * 1.1)

    def proj(pts):
        pts = np.atleast_2d(np.asarray(pts, np.float64))
        u, v, z, ok = project(pts, Rv, e, fpx, W, H)
        return u, v, ok
    return img.astype(np.uint8), proj, float(bad.mean())
