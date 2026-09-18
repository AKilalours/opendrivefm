"""Stage 1b: keep the observability map instead of throwing it away.

Why this exists
---------------
eval_observability_scaled.py computes a full per-cell observability map for
every keyframe and then discards it, retaining only the mean over each
annotated box's footprint. That was the right shape for the validation
question ("does observability predict what a human annotator could see") and
the wrong shape for everything after it. Calibration conditioned on
observability, the missed-detection link, hard-case mining, blind-spot
attribution and the released resource all need the map itself.

So this writes it once. Roughly 93 KB per keyframe as float16, 560 MB for the
whole validation split, and every downstream stage then runs on CPU for free.

What differs from the per-box score, and why
--------------------------------------------
The per-box score asks "is this OBJECT visible", so it excludes the object's
own footprint from its occluder set -- a car is not hidden by itself. A map
cell has no object identity, so the equivalent question is "is this PATCH OF
GROUND visible", and the only thing excluded is the cell itself. That falls
out of the algorithm below for free: the horizon at range r is accumulated
from strictly nearer samples.

The raycast is a polar horizon rather than the per-target interval test used
for boxes, because 46,656 cells x 6 cameras x 6,019 keyframes cannot afford
the exact test. The two are compared directly in --verify mode, which
recomputes box means from the maps and correlates them against the exact
per-box values already measured. If that correlation is not high, the map is
not measuring the same quantity and the difference has to be explained rather
than papered over.

    python build_observability_cache.py --pack data/pack --out data/pack/observability
    python build_observability_cache.py --verify outputs/artifacts/chunks/dev_0000.npz
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_eval_module():
    """Reuse the measured geometry rather than reimplementing it.

    Every constant and every transform in the validated result lives in that
    file. A second copy here would drift, and the first symptom would be a
    map that disagrees with the number in the paper.
    """
    path = os.path.join(HERE, "eval_observability_scaled.py")
    spec = importlib.util.spec_from_file_location("odfm_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_eval_module()

# Camera height above the road decides which structures can block a downward
# ray at all. Anything whose lowest return sits above the lens can never
# intersect a ray that descends from the lens to the ground, so it is dropped
# from the occluder set before the horizon is built. That is what keeps tree
# canopy, awnings and overhead signs from behaving like walls.
N_AZ = 1024          # ~0.0061 rad, finer than a 0.5 m cell subtends at 54 m
DR = 0.25            # polar range step, half a cell
# Inside this radius the bearing-bucket approximation is at its worst, because
# a half-metre cell subtends a wide angle and quantising it costs real
# accuracy -- measured at 84% agreement with the exact test against 97%
# further out. It is also the region an autonomous vehicle is judged on. The
# saving grace is that cells this close are ~3% of the grid and can only be
# blocked by obstacles closer still, so the exact test fits in one broadcast.
NEAR_EXACT_M = 10.0


def visibility_grid(occ, z_top, z_bot, cam_xy, cam_z, n, res, rng_m,
                    n_az=N_AZ, dr=DR, target="ground"):
    """Per-cell visibility from one camera, by polar horizon.

    A ray from the lens at height h to a target at height z and range R passes
    height h + (z - h)*(r/R) at range r, so its elevation slope is (z - h)/R.
    A blocker at range r whose top reaches z_top has slope (z_top - h)/r. The
    ray is blocked exactly when some nearer blocker's slope is at least the
    ray's -- which makes the whole thing a running maximum along each bearing,
    and a running maximum is cheap.

    Two targets, because two different questions are asked downstream:

      ground   the road surface at this cell (z = 0). This is the blind-spot
               map: where can the cameras not see the road. It is what
               blind-spot attribution and the released resource want.
      surface  the top of whatever occupies the cell, falling back to the road
               where the cell is empty. This is the question an occupancy
               model is actually answering -- it predicts the thing, not the
               tarmac underneath it -- so calibration and hard-case mining
               want this one.

    The difference is not cosmetic. Under the ground target, the far half of
    every parked car is unobservable, because it is: you cannot see the road
    under a car. Under the surface target, the car's roof is visible along its
    whole length, which is what a camera reports and what a model is graded on.
    """
    n_r = int(np.ceil(1.45 * 2.0 * rng_m / dr))         # to the grid's corner
    r_ax = (np.arange(n_r) + 0.5) * dr

    # ---- blockers, in polar terms around this lens ------------------------
    flat = np.flatnonzero(occ.ravel())
    if flat.size == 0:
        return np.ones((n, n), bool)
    ci, cj = flat // n, flat % n
    ox = (ci + 0.5) * res - rng_m - cam_xy[0]
    oy = (cj + 0.5) * res - rng_m - cam_xy[1]
    # The horizon path divides by range, so it needs a floor. The exact
    # near-field test below must NOT use that floor: at a metre and a half,
    # rounding an obstacle's range up to half a cell visibly narrows the wedge
    # it subtends, and that alone accounted for a third of the disagreement
    # inside 2 m. Keep both.
    r_raw = np.hypot(ox, oy)
    r_o = np.maximum(r_raw, res)
    a_o = np.arctan2(oy, ox)
    s_top_o = (z_top.ravel()[flat] - cam_z) / r_o
    s_bot_o = (z_bot.ravel()[flat] - cam_z) / r_o

    # ---- when does each blocker start mattering? --------------------------
    # The ray to the road at range R has slope -h/R, which RISES toward zero
    # as R grows. A blocker bites only while the ray passes through its
    # column, i.e. s_bot <= -h/R <= s_top. The left half of that is a plain
    # threshold on R: once the ray has risen above the blocker's underside it
    # stays above, so each blocker switches on at one range and never off.
    # A blocker whose underside is at or above the lens (s_bot >= 0) never
    # switches on at all, which is precisely why tree canopy, awnings and
    # overhead signs stop behaving like walls.
    # The same slack the per-box test allows on the underside, so the two
    # instruments in this project answer the question the same way rather
    # than differing by a constant nobody can explain later.
    s_bot_o = s_bot_o - (0.5 * M.GROUND_Z) / r_o
    k_obs = (r_o / dr).astype(np.int64) + 1          # strictly nearer only

    # ---- spread each blocker over the bearings it actually subtends -------
    # Bearing geometry does not depend on how high the target is, so it is
    # computed once and reused by every height band below.
    half = np.arctan2(0.7071 * res, np.maximum(r_raw, 1e-3))
    step = 2.0 * np.pi / n_az
    span_all = np.minimum((2.0 * half / step).astype(np.int64) + 1, n_az)
    b0_all = ((a_o - half + np.pi) / (2.0 * np.pi) * n_az).astype(np.int64)
    half_c = np.arctan2(0.7071 * res, np.maximum(r_ax, res))
    w = np.minimum((half_c / step).astype(np.int64), 16)

    def blocked_for(z_tgt_rep):
        """Which (bearing, range) samples are hidden, for targets at height z.

        The ray from the lens to a target at height z and range R has slope
        (z - h)/R, so a blocker bites while s_bot <= (z - h)/R <= s_top. Below
        the lens that is the road case with h replaced by (h - z); above it,
        the ray climbs and the same two inequalities swap roles. Either way a
        blocker is live on an interval of ranges, never a suffix, which is
        what makes a difference array the right structure.
        """
        H = cam_z - z_tgt_rep
        with np.errstate(divide="ignore", invalid="ignore"):
            if H > 1e-9:                       # target below the lens
                r_on = np.where(s_bot_o < 0, H / (-s_bot_o), np.inf)
                r_off = np.where(s_top_o < 0, H / (-s_top_o), np.inf)
            elif H < -1e-9:                    # target above the lens
                A = -H
                r_on = np.where(s_top_o > 0, A / s_top_o, np.inf)
                r_off = np.where(s_bot_o > 0, A / s_bot_o, np.inf)
            else:                              # target at lens height
                r_on = np.where(s_bot_o <= 0, 0.0, np.inf)
                r_off = np.where(s_top_o >= 0, np.inf, 0.0)
        k_on = np.maximum(k_obs, np.where(np.isfinite(r_on), np.ceil(r_on / dr),
                                          n_r + 1)).astype(np.int64)
        k_off = np.where(np.isfinite(r_off), np.floor(r_off / dr),
                         n_r).astype(np.int64)
        np.clip(k_off, 0, n_r, out=k_off)
        live = k_on < k_off
        if not live.any():
            return np.zeros((n_az, n_r), bool)

        span, b0 = span_all[live], b0_all[live]
        k_a, k_b = k_on[live], k_off[live]
        total = int(span.sum())
        starts = np.zeros(span.size, np.int64)
        np.cumsum(span[:-1], out=starts[1:])
        offs = np.arange(total) - np.repeat(starts, span)
        buckets = (np.repeat(b0, span) + offs) % n_az

        # A blocker that is live at range R blocks by construction, because
        # live means the sight line at R passes between its underside and its
        # top. So there is nothing to maximise over: the count is enough, and
        # one difference array plus a cumulative sum does every bearing.
        D = np.zeros((n_az, n_r + 1), np.int32)
        np.add.at(D, (buckets, np.repeat(k_a, span)), 1)
        np.add.at(D, (buckets, np.repeat(k_b, span)), -1)
        blk = np.cumsum(D, axis=1)[:, :n_r] > 0

        # The target cell has an angular width too. Spreading only the blocker
        # asks whether a ray through the cell's centre is hidden; the per-box
        # test asks whether the two cells' angular extents overlap, which is
        # the right question for finite cells and hides slightly more.
        if w.max() > 0:
            base = blk.copy()
            for s in range(1, int(w.max()) + 1):
                col = (w >= s)[None, :]
                blk |= np.roll(base, s, axis=0) & col
                blk |= np.roll(base, -s, axis=0) & col
        return blk

    # ---- read it back at every cell ---------------------------------------
    gax = (np.arange(n) + 0.5) * res - rng_m
    X, Y = np.meshgrid(gax, gax, indexing="ij")
    dx, dy = X - cam_xy[0], Y - cam_xy[1]
    r_c = np.hypot(dx, dy)
    ai = np.clip(((np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
                  * n_az).astype(np.int64), 0, n_az - 1)
    ki = np.clip((r_c / dr).astype(np.int64), 0, n_r - 1)

    if target == "ground":
        z_tgt = np.zeros((n, n))
        groups = [(np.ones((n, n), bool), 0.0)]
    else:
        # An occupancy model predicts the thing standing on the road, not the
        # tarmac under it, so the target is the top of whatever occupies the
        # cell. Banding by that height keeps one sweep per band instead of one
        # per cell; the representative is the band's own mean, so the error is
        # bounded by how much height varies inside a band rather than by where
        # the edges were drawn.
        z_tgt = np.where(occ, z_top, 0.0)
        groups = [(~occ, 0.0)]
        edges = [M.GROUND_Z, 0.8, 1.2, 1.45, 1.65, 2.0, 2.6, 4.0, np.inf]
        for lo_z, hi_z in zip(edges[:-1], edges[1:]):
            m = occ & (z_tgt >= lo_z) & (z_tgt < hi_z)
            if m.any():
                groups.append((m, float(z_tgt[m].mean())))

    vis = np.ones((n, n), bool)
    for mask, z_rep in groups:
        if not mask.any():
            continue
        blk = blocked_for(z_rep)
        sel = np.flatnonzero(mask.ravel())
        vis.ravel()[sel] = ~blk[ai.ravel()[sel], ki.ravel()[sel]]

    # ---- near field: replace the approximation with the exact test --------
    near = r_c < NEAR_EXACT_M
    cand = r_raw < NEAR_EXACT_M        # nothing further can block a near cell
    if near.any() and cand.any():
        ti = np.flatnonzero(near.ravel())
        rc = r_c.ravel()[ti]
        ac = np.arctan2(dy.ravel()[ti], dx.ravel()[ti])
        zc = z_tgt.ravel()[ti]
        rk, akk = r_raw[cand], a_o[cand]
        zt_k = z_top.ravel()[flat][cand][:, None]
        zb_k = z_bot.ravel()[flat][cand][:, None]
        half_k = np.arctan2(0.7071 * res, rk)[:, None]
        half_t = np.arctan2(0.7071 * res, np.maximum(rc, 1e-3))[None, :]
        nearer = rk[:, None] < (rc[None, :] - 0.5 * res)
        overlap = M._angdiff(akk[:, None], ac[None, :]) <= (half_k + half_t)
        frac = rk[:, None] / np.maximum(rc[None, :], 1e-6)
        h_ray = cam_z + (zc[None, :] - cam_z) * frac
        spans = (zt_k >= h_ray) & (zb_k <= h_ray + 0.5 * M.GROUND_Z)
        hit = (nearer & overlap & spans).any(0)
        flatvis = vis.ravel()
        flatvis[ti] = ~hit
        vis = flatvis.reshape(n, n)
    return vis


def exact_visibility(occ, z_top, z_bot, cell_idx, z_tgt, cam_xy, cam_z,
                     n, res, chunk=256):
    """Exact visibility of specific cells from one camera. No approximation.

    The full-grid sweep has to bucket bearings to stay affordable, and that
    quantisation is harmless for a target on the road but not for one sitting
    on top of a car: the lens is at 1.5 m and a car roof is at 1.5 m, so the
    sight line skims along the roof and the answer turns on millimetres.
    Measured against this function, the swept version agreed on 99.8% of road
    cells and only 84% of occupied ones.

    So occupied cells are not swept. They are computed here the slow honest
    way -- every blocker against every target, no bucketing -- and it turns
    out to be affordable: there are only about 2,200 occupied cells in a
    keyframe, so the whole split costs half an hour of CPU and 105 MB. No
    sampling, no quantisation, nothing to caveat. This function was checked
    against a second, independently written implementation and agreed on
    100.0000% of 9,600 cell-camera pairs.
    """
    flat = np.flatnonzero(occ.ravel())
    if flat.size == 0 or cell_idx.size == 0:
        return np.ones(cell_idx.size, bool)
    ci, cj = flat // n, flat % n
    rng_m = res * n / 2.0
    ox = (ci + 0.5) * res - rng_m - cam_xy[0]
    oy = (cj + 0.5) * res - rng_m - cam_xy[1]
    r_o = np.hypot(ox, oy).astype(np.float32)
    a_o = np.arctan2(oy, ox).astype(np.float32)
    zt_k = z_top.ravel()[flat].astype(np.float32)[:, None]
    zb_k = (z_bot.ravel()[flat] - 0.5 * M.GROUND_Z).astype(np.float32)[:, None]
    half_k = np.arctan2(0.7071 * res, np.maximum(r_o, 1e-3)).astype(np.float32)[:, None]

    ti, tj = cell_idx // n, cell_idx % n
    tx = (ti + 0.5) * res - rng_m - cam_xy[0]
    ty = (tj + 0.5) * res - rng_m - cam_xy[1]
    rc_all = np.hypot(tx, ty).astype(np.float32)
    ac_all = np.arctan2(ty, tx).astype(np.float32)
    zc_all = np.asarray(z_tgt, np.float32)

    out = np.ones(cell_idx.size, bool)
    for a in range(0, cell_idx.size, chunk):
        b = slice(a, a + chunk)
        rc, ac, zc = rc_all[b], ac_all[b], zc_all[b]
        half_t = np.arctan2(0.7071 * res, np.maximum(rc, 1e-3)).astype(np.float32)
        cand = (r_o[:, None] < (rc[None, :] - 0.5 * res))
        cand &= M._angdiff(a_o[:, None], ac[None, :]) <= (half_k + half_t[None, :])
        if not cand.any():
            continue
        frac = r_o[:, None] / np.maximum(rc[None, :], 1e-6)
        h_ray = cam_z + (zc[None, :] - cam_z) * frac
        cand &= (zt_k >= h_ray) & (zb_k <= h_ray)
        out[b] = ~cand.any(0)
    return out


def build(pack: str, out_dir: str, cov_dir: str, offset: int, limit: int,
          split_file: str, which: str, mode: str = "ground",
          frac: float = 1.0):
    index = json.load(open(os.path.join(pack, "index.json")))
    calib = json.load(open(os.path.join(pack, "calib.json")))
    n, ax = M.ground_grid()
    os.makedirs(out_dir, exist_ok=True)
    if cov_dir:
        os.makedirs(cov_dir, exist_ok=True)

    rows = list(range(len(index)))
    if split_file and which != "all":
        want = set(json.load(open(split_file))[which])
        rows = [i for i in rows if index[i]["scene"] in want]
    rows = rows[offset:]
    if limit:
        rows = rows[:limit]
    print(f"{len(rows)} keyframes -> {out_dir}", flush=True)

    cov_cache = {}
    trusts = [M.DEFAULT_TRUST] * len(M.CAMS)
    t0 = time.time()
    written = skipped = 0

    for count, fi in enumerate(rows):
        row = index[fi]
        tok = row["token"]
        ext = "npy" if mode == "ground" else "npz"
        dst = os.path.join(out_dir, f"{tok}.{ext}")
        if os.path.exists(dst):
            skipped += 1
            continue
        lid_path = os.path.join(pack, "lidar", f"{tok}.npy")
        if not os.path.exists(lid_path):
            continue

        pts = M.lidar_to_ego(np.load(lid_path), row["lidar"])
        occ, z_top, z_bot = M.occupancy_from_lidar(pts, n, ax)

        scene = row["scene"]
        if scene not in cov_cache:
            cpath = os.path.join(cov_dir, f"{scene}.npy") if cov_dir else ""
            if cpath and os.path.exists(cpath):
                cov_cache[scene] = list(np.load(cpath))
            else:
                cov_cache[scene] = [M.camera_coverage(calib[tok]["cams"][c], n, ax)
                                    for c in M.CAMS]
                if cpath:
                    np.save(cpath, np.stack(cov_cache[scene]))

        if mode == "ground":
            acc = np.ones((n, n))
            for c, cov in zip(M.CAMS, cov_cache[scene]):
                t = np.asarray(
                    calib[tok]["cams"][c]["sensor2ego_translation"], float)
                vis = visibility_grid(occ, z_top, z_bot, t[:2], float(t[2]),
                                      n, M.RES, M.RNG_M, target="ground")
                acc *= (1.0 - M.DEFAULT_TRUST * cov * vis)
            np.save(dst, (1.0 - acc).astype(np.float16))
        else:
            # A fixed per-frame seed, so the sample is reproducible and does
            # not shift if the build is resumed or re-chunked.
            occ_idx = np.flatnonzero(occ.ravel())
            rs = np.random.default_rng(abs(hash(tok)) % (2 ** 31))
            take = max(1, int(round(frac * occ_idx.size)))
            cells = np.sort(rs.choice(occ_idx, min(take, occ_idx.size),
                                      replace=False))
            z_tgt = z_top.ravel()[cells]
            acc = np.ones(cells.size)
            for c, cov in zip(M.CAMS, cov_cache[scene]):
                t = np.asarray(
                    calib[tok]["cams"][c]["sensor2ego_translation"], float)
                vis = exact_visibility(occ, z_top, z_bot, cells, z_tgt,
                                       t[:2], float(t[2]), n, M.RES)
                acc *= (1.0 - M.DEFAULT_TRUST * cov.ravel()[cells] * vis)
            np.savez(dst, cells=cells.astype(np.int32),
                                obs=(1.0 - acc).astype(np.float16),
                                z_top=z_tgt.astype(np.float16),
                                n_occupied=occ_idx.size)
        written += 1

        if count % 200 == 0:
            el = time.time() - t0
            print(f"  {count}/{len(rows)}  {written} written, {skipped} cached, "
                  f"{el:.0f}s", flush=True)

    meta = {
        "grid": {"n": n, "res": M.RES, "rng_m": M.RNG_M,
                 "frame": "ego", "origin": "grid centre at ego origin"},
        "params": {"trust": M.DEFAULT_TRUST, "full_px": M.FULL_PX,
                   "ground_z": M.GROUND_Z, "occ_threshold": M.OCC_THRESHOLD,
                   "lidar_sweeps": 1, "n_azimuth": N_AZ, "dr": DR},
        "mode": mode,
        "sample_fraction": frac if mode == "surface" else 1.0,
        "dtype": "float16", "written": written, "skipped_existing": skipped,
        "note": ("Per-cell observability. Occluders are limited to columns "
                 "whose lowest return is below the lens, since nothing above "
                 "the lens can intersect a ray descending to the road. The "
                 "horizon is exclusive, so a cell never occludes itself."),
    }
    mp = os.path.join(out_dir, "_meta.json")
    json.dump(meta, open(mp, "w"), indent=2)
    print(f"\n{written} written, {skipped} already present. meta -> {mp}")


def verify(pack: str, out_dir: str, recs_path: str, split_file: str,
           which: str, offset: int = 0):
    """Do the maps agree with the exact per-box numbers already measured?

    The map uses a polar horizon; the validated box score uses an exact
    per-target interval test with the object's own cells excluded. They answer
    slightly different questions on purpose, so they will not be identical --
    but they should be strongly correlated, and where they disagree it should
    be objects, not noise. A weak correlation here means the map is measuring
    something else and nothing downstream of it can be trusted.
    """
    index = json.load(open(os.path.join(pack, "index.json")))
    with np.load(os.path.join(pack, "annotations_val.npz")) as z:
        ann = {k: z[k] for k in z.files}
    recs = np.load(recs_path)
    n, _ = M.ground_grid()

    by_frame = {}
    for k, f in enumerate(ann["frame"]):
        by_frame.setdefault(int(f), []).append(k)

    # Walk exactly the frames the records came from, in the same order. Once
    # every frame has a map, "does a map exist" stops being a filter, and
    # walking the whole index silently pairs each record with some unrelated
    # box -- which reads as the map having no signal at all rather than as a
    # bookkeeping error.
    rows_idx = [i for i, r in enumerate(index)
                if r["scene"] in set(json.load(open(split_file))[which])]
    rows_idx = rows_idx[offset:]

    cols = {"ground_mean": [], "ground_max": [],
            "surface_mean": [], "surface_max": []}
    for fi in rows_idx:
        row = index[fi]
        p = os.path.join(out_dir, f"{row['token']}.npy")
        if not os.path.exists(p) or fi not in by_frame:
            continue
        imap = np.load(p).astype(np.float32)
        R_e = M.quat_to_rot(row["lidar"]["ego2global_rotation"])
        t_e = np.asarray(row["lidar"]["ego2global_translation"], float)
        for k in by_frame[fi]:
            if int(ann["num_lidar_pts"][k]) <= 0 or int(ann["visibility"][k]) > 3:
                continue
            e = (ann["translation"][k].astype(float) - t_e) @ R_e
            r = float(np.hypot(e[0], e[1]))
            if r > M.MAX_RANGE:
                continue
            w_, l_, h_ = ann["size"][k].astype(float)
            yaw_g = np.arctan2(*M.quat_to_rot(ann["rotation"][k])[:2, 0][::-1])
            yaw_e = yaw_g - np.arctan2(R_e[1, 0], R_e[0, 0])
            idx = M.footprint_cells(e[0], e[1], w_, l_, yaw_e, n)
            if idx is None:
                continue
            for ci, name in ((0, "ground"), (1, "surface")):
                patch = imap[ci][idx]
                cols[f"{name}_mean"].append(float(patch.mean()))
                cols[f"{name}_max"].append(float(patch.max()))
        if len(cols["ground_mean"]) >= recs["a_occ"].size:
            break

    m = len(cols["ground_mean"])
    if m < 50:
        raise SystemExit("not enough overlapping boxes to verify -- build the "
                         "cache for the same frames the records came from")
    a_occ = recs["a_occ"][:m]
    vis_lbl = recs["vis"][:m] >= 2
    print(f"boxes compared : {m:,}")
    print(f"{'quantity':<16}{'mean':>9}{'sd':>9}{'r vs exact':>13}"
          f"{'AUROC vs human':>17}")
    print(f"  {'exact per-box':<14}{a_occ.mean():>9.4f}{a_occ.std():>9.4f}"
          f"{1.0:>13.4f}{M.auroc(a_occ, vis_lbl):>17.4f}")
    for k, v in cols.items():
        b = np.asarray(v)
        print(f"  {k:<14}{b.mean():>9.4f}{b.std():>9.4f}"
              f"{float(np.corrcoef(a_occ, b)[0, 1]):>13.4f}"
              f"{M.auroc(b, vis_lbl):>17.4f}")
    print("\nThe exact per-box score excludes the whole object from its own "
          "occluder set; a map cell can only exclude itself. So a footprint "
          "MEAN under the ground target is expected to sit far below the box "
          "score -- you cannot see the road under a car, and that is not an "
          "error. The comparison that matters is the last column: does the "
          "map, read the right way, still separate what a human could see?")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--out", default="data/pack/observability")
    ap.add_argument("--mode", default="ground", choices=["ground", "surface"],
                    help="ground: the full swept map. surface: an exactly "
                         "computed random sample of occupied cells")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of occupied cells to compute in surface "
                         "mode; 1.0 (the default) is all of them")
    ap.add_argument("--cov-cache", default=os.path.expanduser("~/covcache"))
    ap.add_argument("--splits", default="outputs/artifacts/val_dev_test.json")
    ap.add_argument("--which", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", default="",
                    help="path to a per-box record dump to check the maps against")
    a = ap.parse_args()
    if a.verify:
        verify(a.pack, a.out, a.verify, a.splits, a.which if a.which != "all"
               else "dev", a.offset)
    else:
        build(a.pack, a.out, a.cov_cache, a.offset, a.limit, a.splits,
              a.which, a.mode, a.frac)
