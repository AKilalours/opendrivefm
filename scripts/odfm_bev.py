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



def _bev_chrome(d, size, rng_m, rings=(10, 20, 30, 40, 50), ego=True,
                ring_col=(64, 74, 92), label=True):
    """Range rings, axes and the ego marker, in final-image pixels."""
    c = size / 2.0
    mpp = size / (2.0 * rng_m)
    for r in rings:
        if r > rng_m:
            continue
        rp = r * mpp
        d.ellipse([c - rp, c - rp, c + rp, c + rp], outline=ring_col + (130,), width=1)
        if label:
            d.text((c + 4, c - rp - 12), f"{r}", fill=(120, 132, 150), font=_font(10))
    d.line([c, 0, c, size], fill=ring_col + (70,), width=1)
    d.line([0, c, size, c], fill=ring_col + (70,), width=1)
    if ego:
        d.polygon([(c, c - 8), (c - 6, c + 7), (c + 6, c + 7)],
                  fill=(255, 255, 255, 240))


def _to_screen(a):
    """BEV array indexed [x_forward, y_left] -> image rows/cols with forward
    up and left on the left."""
    return np.flipud(np.fliplr(np.asarray(a)))


def _box_screen(b, size, rng_m):
    c = size / 2.0
    mpp = size / (2.0 * rng_m)
    w, l, _ = b["wlh"]
    yaw = b["yaw"]
    hx, hy = l / 2.0, w / 2.0
    crn = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    q = crn @ R.T + np.array([b["centre"][0], b["centre"][1]])
    return [(c - v[1] * mpp, c - v[0] * mpp) for v in q]


def render_occupancy_prob(prob, size=620, rng_m=54.0, boxes=None,
                          title=None, subtitle=None, legend=True):
    """Probabilistic occupancy: free -> unknown -> occupied.

    Free is the BRIGHT state and unknown the dark one, which is the opposite of
    the obvious mapping and the right way round: the drivable surface is what
    the viewer is looking for, and making the map's most common value (unknown)
    a large pale field would swamp it. Dark-for-unknown also makes occlusion
    shadows read as shadows.

    Colour is anchored on p = 0.5 rather than stretched over the observed
    range, because unknown is genuinely the midpoint of the log-odds scale and
    has to look like the neutral state -- telling "observed empty" from "never
    observed" apart at a glance is the most important thing this panel says.
    """
    p = np.clip(np.asarray(prob, np.float64), 0.0, 1.0)
    free_c = np.array([132, 196, 210], np.float64)
    unk_c = np.array([20, 23, 30], np.float64)
    occ_c = np.array([255, 92, 74], np.float64)

    t = np.clip(p / 0.5, 0, 1)[..., None] ** 0.75
    img = free_c + (unk_c - free_c) * t
    t2 = np.clip((p - 0.5) / 0.5, 0, 1)[..., None]
    img = img + (occ_c - img) * (t2 ** 0.55)

    # Widen occupied cells by one cell before downsampling. An obstacle rim is
    # often a single 20 cm cell; at panel scale that is a third of a pixel and
    # LANCZOS averages it into the free space around it, so the one class the
    # viewer most needs to see disappears. Applied to the RENDER only -- the
    # returned grid and every statistic are untouched.
    occ_m = p > 0.65
    if occ_m.any():
        grown = occ_m.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                grown |= np.roll(np.roll(occ_m, dx, 0), dy, 1)
        img[grown] = occ_c

    out = Image.fromarray(_to_screen(np.clip(img, 0, 255).astype(np.uint8)))
    out = out.resize((size, size), Image.LANCZOS)
    d = ImageDraw.Draw(out, "RGBA")
    _bev_chrome(d, size, rng_m)

    if boxes:
        for b in boxes:
            if b.get("num_lidar_pts", 0) <= 0:
                continue
            scr = _box_screen(b, size, rng_m)
            d.line(scr + [scr[0]], fill=(14, 32, 46, 215), width=2)

    if legend:
        x0, y0 = 12, size - 62
        for i, (lab, col) in enumerate((("free", (132, 196, 210)),
                                        ("unknown", (20, 23, 30)),
                                        ("occupied", (255, 92, 74)))):
            yy = y0 + i * 14
            d.rectangle([x0, yy, x0 + 10, yy + 10], fill=col + (255,),
                        outline=(90, 100, 118, 255))
            d.text((x0 + 15, yy - 1), lab, fill=DIM, font=_font(11))

    if title:
        d.rectangle([0, 0, size, 28], fill=(8, 10, 15, 226))
        d.text((10, 6), title, fill=TEXT, font=_font(14))
    if subtitle:
        d.rectangle([0, size - 22, size, size], fill=(8, 10, 15, 226))
        d.text((10, size - 17), subtitle, fill=DIM, font=_font(11))
    return np.array(out)


def render_integrity(integ, size=620, rng_m=54.0, occ_mask=None, boxes=None,
                     title=None, subtitle=None, ego=True, legend=True):
    """Perception Integrity Map: 0 = no trustworthy camera evidence, 1 = full.

    Obstacles are overprinted in a hard colour rather than left implicit. The
    map's most important feature is the wedge of low integrity stretching away
    behind every vehicle, and a viewer can only read that as an occlusion
    shadow if the thing casting it is visible in the same picture. Without the
    obstacle layer it is just a dark patch of unexplained shape.
    """
    from matplotlib import cm as _cm
    a = np.clip(np.asarray(integ, np.float64), 0, 1)
    img = (_cm.get_cmap("magma")(a)[:, :, :3] * 255.0)

    if occ_mask is not None:
        m = np.asarray(occ_mask, bool)
        grown = m.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                grown |= np.roll(np.roll(m, dx, 0), dy, 1)
        img[grown] = np.array([90, 226, 236], np.float64)

    out = Image.fromarray(_to_screen(np.clip(img, 0, 255).astype(np.uint8)))
    out = out.resize((size, size), Image.LANCZOS)
    d = ImageDraw.Draw(out, "RGBA")
    _bev_chrome(d, size, rng_m, ring_col=(96, 88, 112), ego=False)

    if boxes:
        for b in boxes:
            if b.get("num_lidar_pts", 0) <= 0:
                continue
            scr = _box_screen(b, size, rng_m)
            d.line(scr + [scr[0]], fill=(140, 240, 250, 140), width=1)

    if ego:
        c = size / 2.0
        d.polygon([(c, c - 8), (c - 6, c + 7), (c + 6, c + 7)],
                  fill=(140, 255, 225, 245))

    if legend:
        from matplotlib import cm as _cm2
        bx, by, bw, bh = 12, size - 52, 130, 9
        for i in range(bw):
            col = _cm2.get_cmap("magma")(i / (bw - 1))[:3]
            d.line([bx + i, by, bx + i, by + bh],
                   fill=tuple(int(255 * v) for v in col) + (255,))
        d.rectangle([bx, by, bx + bw, by + bh], outline=(120, 128, 145, 220))
        d.text((bx, by + bh + 2), "0", fill=DIM, font=_font(10))
        d.text((bx + bw - 6, by + bh + 2), "1", fill=DIM, font=_font(10))
        d.text((bx + 34, by + bh + 2), "integrity", fill=DIM, font=_font(10))
        d.rectangle([bx, by - 16, bx + 10, by - 6], fill=(90, 226, 236, 255))
        d.text((bx + 15, by - 18), "occupied", fill=DIM, font=_font(10))

    if title:
        d.rectangle([0, 0, size, 28], fill=(8, 10, 15, 226))
        d.text((10, 6), title, fill=TEXT, font=_font(14))
    if subtitle:
        d.rectangle([0, size - 22, size, size], fill=(8, 10, 15, 226))
        d.text((10, size - 17), subtitle, fill=DIM, font=_font(11))
    return np.array(out)
