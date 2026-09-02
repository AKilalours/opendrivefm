"""Bird's-eye-view rendering of real LiDAR returns.

Takes the (N, 6) stack produced by odfm_lidar.load_sweep_stack and the true
boxes from odfm_lidar.load_annotations, and produces the top-down view.

Three choices here are what separate this from a scatter plot:

  * Points are accumulated into a float weight/colour buffer with
    `np.add.at`, then normalised, rather than painted one over another. Painting
    means the last point drawn wins, which throws away density -- and density is
    the whole signal in a LiDAR BEV. Accumulating means a cell hit by forty
    returns is visibly brighter than one hit by two.
  * Rendering happens at SS x resolution and is downsampled with LANCZOS. A
    single-pixel return at final resolution aliases into a harsh dot; at 3x it
    lands as a properly weighted sub-pixel sample.
  * Colour is height through `turbo`, brightness through intensity, alpha
    through sweep age. The height range is anchored on the measured ground
    plane: in the nuScenes ego frame the road sits at z ~ -0.1, NOT at minus
    the sensor height -- the calibration translation has already been applied.
    Getting that wrong maps tarmac to the middle of the colormap and turns the
    entire road surface green. Height is what makes structure legible (kerbs, vehicle
    roofs, overhanging foliage all separate); intensity is what makes lane
    paint and retroreflective signage pop the way it does in a real stack.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm

BG = (6, 8, 12)
GRID = (38, 46, 60)
TEXT = (226, 232, 240)
DIM = (128, 141, 160)

CLASS_COLOUR = {
    "vehicle.car": (86, 214, 255),
    "vehicle.truck": (86, 214, 255),
    "vehicle.bus": (86, 214, 255),
    "vehicle.trailer": (86, 214, 255),
    "vehicle.construction": (86, 214, 255),
    "vehicle.motorcycle": (255, 178, 64),
    "vehicle.bicycle": (255, 178, 64),
    "human.pedestrian": (255, 92, 132),
    "movable_object": (168, 178, 196),
}

_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_fc = {}


def _font(sz):
    if sz not in _fc:
        import os
        f = None
        for p in _FONTS:
            if os.path.exists(p):
                try:
                    f = ImageFont.truetype(p, sz)
                    break
                except Exception:
                    pass
        _fc[sz] = f or ImageFont.load_default()
    return _fc[sz]


def _class_colour(name):
    for k, v in CLASS_COLOUR.items():
        if name.startswith(k):
            return v
    return (168, 178, 196)


def render_bev(points, boxes=None, rng_m=54.0, size=1100, ss=3,
               z_lo=-0.5, z_hi=4.5, title=None, subtitle=None,
               show_rings=True, max_age=10, motion=None, dynamic_label=1,
               canopy_above=None, canopy_fade=0.30):
    """points: (N, 6) x, y, z, intensity, ring, age -- ego frame, x forward.

    Returns an RGB uint8 array of shape (size, size, 3).
    """
    S = size * ss
    ppm = S / (2.0 * rng_m)                      # pixels per metre at SS scale

    acc = np.zeros((S, S, 3), np.float32)
    wgt = np.zeros((S, S), np.float32)

    if len(points):
        p = np.asarray(points, np.float32)
        x, y, z, inten, age = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 5]

        # ego x-forward -> screen up; ego y-left -> screen left
        col = np.rint(S * 0.5 - y * ppm).astype(np.int64)
        row = np.rint(S * 0.5 - x * ppm).astype(np.int64)
        ok = (col >= 0) & (col < S) & (row >= 0) & (row < S)
        col, row = col[ok], row[ok]
        z, inten, age = z[ok], inten[ok], age[ok]

        t = np.clip((z - z_lo) / max(1e-6, z_hi - z_lo), 0.0, 1.0)
        rgb = cm.get_cmap("turbo")(t)[:, :3].astype(np.float32) * 255.0

        # Motion overrides height. A moving object's height tells you nothing
        # a planner needs; that it is moving tells you everything, so dynamic
        # returns are pulled out of the height ramp entirely rather than tinted
        # within it, where they would be indistinguishable from a kerb.
        if motion is not None:
            mv = np.asarray(motion)[ok] == dynamic_label
            if mv.any():
                rgb[mv] = np.array([255.0, 74.0, 122.0], np.float32)

        # intensity lifts a return without ever letting it vanish: a 0-intensity
        # point is still a real return and must still be drawn.
        b = 0.50 + 0.50 * np.clip(inten / 48.0, 0.0, 1.0)

        # Push tree canopy, awnings and building upper storeys into the
        # background. They are real returns and they are not driving-relevant,
        # and in a scene with heavy foliage they otherwise pin the top of the
        # height ramp and repaint half the picture red -- which reads as
        # dramatic and hides the kerbs and vehicles that actually matter.
        # Dimmed rather than dropped: the canopy still marks where the trees
        # are, it just stops competing for attention.
        if canopy_above is not None:
            b = b * np.where(z > canopy_above, canopy_fade, 1.0)
        # history fades but never to zero, so accumulated structure still reads
        a = (1.0 - 0.62 * np.clip(age / max(1, max_age), 0.0, 1.0)).astype(np.float32)
        w = (a * b).astype(np.float32)

        # Splat each return across a small footprint instead of a single
        # supersampled pixel. At ss=3 a lone pixel is a third of a final pixel,
        # so an unsplatted cloud downsamples into faint dust no matter how the
        # tone curve is set -- the returns are simply not covering the raster.
        # A 3x3 Gaussian-ish footprint makes one return read as roughly one
        # final pixel, which is what every production BEV viewer does.
        K = [(0, 0, 1.00), (-1, 0, 0.55), (1, 0, 0.55), (0, -1, 0.55), (0, 1, 0.55),
             (-1, -1, 0.30), (-1, 1, 0.30), (1, -1, 0.30), (1, 1, 0.30)]
        for dr, dc, kw in K:
            r2, c2 = row + dr, col + dc
            m2 = (r2 >= 0) & (r2 < S) & (c2 >= 0) & (c2 < S)
            ww = (w[m2] * kw).astype(np.float32)
            np.add.at(acc, (r2[m2], c2[m2]), rgb[m2] * ww[:, None])
            np.add.at(wgt, (r2[m2], c2[m2]), ww)

    # Normalise to the mean colour of the returns in a cell, then use the
    # accumulated weight -- log-compressed, because return counts per cell span
    # orders of magnitude -- as the cell's brightness.
    # Reference the 85th percentile of occupied cells, not the 99th. Return
    # counts per cell are heavily skewed -- median 2, p99 above 40, driven by
    # the near-field ground rings -- so normalising on p99 drives the typical
    # cell to near-black and the picture goes dark everywhere except under the
    # vehicle. p85 puts an ordinary return at mid-brightness and lets the dense
    # near field clip, which is the correct thing to sacrifice.
    ref = float(np.percentile(wgt[wgt > 0], 85.0)) if (wgt > 0).any() else 1.0
    dens = np.clip(np.log1p(wgt) / np.log1p(max(1.0, ref)), 0.0, 1.0) ** 0.65
    mean = np.zeros_like(acc)
    nz = wgt > 0
    mean[nz] = acc[nz] / wgt[nz][:, None]
    img = mean * (0.42 + 0.58 * dens)[:, :, None]

    # Gain past 1.0 so bright structure saturates rather than sitting at
    # mid-grey. Clipping the top end of a LiDAR BEV costs nothing -- there is
    # no detail in the hottest cells, only count.
    img = np.clip(img * 1.45, 0.0, 255.0)

    canvas = np.zeros((S, S, 3), np.float32)
    canvas[:, :] = BG
    canvas = np.maximum(canvas, img)

    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    out = out.resize((size, size), Image.LANCZOS)

    d = ImageDraw.Draw(out, "RGBA")
    cx = cy = size / 2.0
    mpp = size / (2.0 * rng_m)                   # pixels per metre, final scale

    if show_rings:
        for r in range(10, int(rng_m) + 1, 10):
            rp = r * mpp
            d.ellipse([cx - rp, cy - rp, cx + rp, cy + rp],
                      outline=GRID + (150,), width=1)
            d.text((cx + 4, cy - rp - 13), f"{r} m", fill=DIM, font=_font(11))
        d.line([cx, 0, cx, size], fill=GRID + (90,), width=1)
        d.line([0, cy, size, cy], fill=GRID + (90,), width=1)

    if boxes:
        for b in boxes:
            _draw_box(d, b, cx, cy, mpp)

    _draw_ego(d, cx, cy, mpp)

    if title:
        f = _font(20)
        d.rectangle([0, 0, size, 40], fill=(8, 10, 15, 232))
        d.text((14, 10), title, fill=TEXT, font=f)
    if subtitle:
        f = _font(13)
        d.rectangle([0, size - 30, size, size], fill=(8, 10, 15, 232))
        d.text((14, size - 22), subtitle, fill=DIM, font=f)

    return np.array(out)


def _draw_box(d, b, cx, cy, mpp):
    w, l, h = b["wlh"]
    yaw = b["yaw"]
    c = b["centre"]
    # footprint corners in the box's own frame, then rotated by yaw into ego
    hx, hy = l / 2.0, w / 2.0
    corners = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]], np.float64)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    pts = corners @ R.T + np.array([c[0], c[1]])
    scr = [(cx - p[1] * mpp, cy - p[0] * mpp) for p in pts]

    col = _class_colour(b["category"])
    empty = b.get("num_lidar_pts", 0) == 0
    # A box with zero LiDAR returns is annotated but unobserved by this sensor.
    # Drawing it identically to an observed box would misrepresent what the
    # LiDAR actually saw, so it is dashed down to an outline hint.
    d.polygon(scr, outline=col + (110 if empty else 255,),
              fill=None if empty else col + (34,))
    if not empty:
        d.line(scr + [scr[0]], fill=col + (255,), width=2)
        # heading tick from centre out through the front face
        fx, fy = (scr[0][0] + scr[1][0]) / 2, (scr[0][1] + scr[1][1]) / 2
        d.line([cx - c[1] * mpp, cy - c[0] * mpp, fx, fy], fill=col + (200,), width=2)


def _draw_ego(d, cx, cy, mpp):
    L, W = 4.084, 1.730                          # nuScenes Renault Zoe footprint
    hx, hy = L / 2 * mpp, W / 2 * mpp
    d.polygon([(cx - hy, cy - hx), (cx + hy, cy - hx),
               (cx + hy, cy + hx), (cx - hy, cy + hx)],
              fill=(255, 255, 255, 40), outline=(255, 255, 255, 210))
    d.polygon([(cx, cy - hx - 9), (cx - 6, cy - hx + 1), (cx + 6, cy - hx + 1)],
              fill=(255, 255, 255, 230))


# ---------------------------------------------------------------------------
# semantic layers: occupancy and integrity
# ---------------------------------------------------------------------------

def _grid_to_screen(a):
    """A BEV array indexed [x_forward, y_left] -> an image with forward up and
    left on the left. Done in one place because getting it wrong produces a
    mirrored map that still looks entirely reasonable."""
    return np.flipud(np.fliplr(np.asarray(a).T)).T[::-1, ::-1].T


def render_occupancy(grid, size=560, title=None, subtitle=None, ego=True):
    """Ray-cast occupancy: UNKNOWN / FREE / OCCUPIED."""
    pal = np.array([[14, 16, 21], [30, 96, 82], [242, 126, 96]], np.uint8)
    img = pal[np.asarray(grid)]
    img = np.flipud(np.fliplr(img))
    out = Image.fromarray(img).resize((size, size), Image.NEAREST)
    d = ImageDraw.Draw(out, "RGBA")
    if ego:
        c = size / 2
        d.polygon([(c, c - 7), (c - 5, c + 6), (c + 5, c + 6)],
                  fill=(255, 255, 255, 235))
    if title:
        d.rectangle([0, 0, size, 28], fill=(8, 10, 15, 224))
        d.text((10, 6), title, fill=TEXT, font=_font(14))
    if subtitle:
        d.rectangle([0, size - 22, size, size], fill=(8, 10, 15, 224))
        d.text((10, size - 17), subtitle, fill=DIM, font=_font(11))
    return np.array(out)


def render_integrity(integ, size=560, title=None, subtitle=None, ego=True):
    """Perception Integrity Map, 0 (no trustworthy camera evidence) to 1."""
    from matplotlib import cm as _cm
    a = np.clip(np.asarray(integ, np.float64), 0, 1)
    img = (_cm.get_cmap("magma")(a)[:, :, :3] * 255).astype(np.uint8)
    img = np.flipud(np.fliplr(img))
    out = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    d = ImageDraw.Draw(out, "RGBA")
    if ego:
        c = size / 2
        d.polygon([(c, c - 7), (c - 5, c + 6), (c + 5, c + 6)],
                  fill=(120, 255, 220, 240))
    if title:
        d.rectangle([0, 0, size, 28], fill=(8, 10, 15, 224))
        d.text((10, 6), title, fill=TEXT, font=_font(14))
    if subtitle:
        d.rectangle([0, size - 22, size, size], fill=(8, 10, 15, 224))
        d.text((10, size - 17), subtitle, fill=DIM, font=_font(11))
    return np.array(out)
