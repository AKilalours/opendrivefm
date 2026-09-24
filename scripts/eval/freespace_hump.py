#!/usr/bin/env python3
"""Why does the calibration gap HUMP on free space? A24 left this open.

A24 located H1's failure precisely: the pre-registered monotone decline holds
on ground-truth OCCUPIED voxels (rank correlation -0.865) and is a hump on
ground-truth FREE voxels, which are 85-92% of every decile, so the aggregate
followed the wrong stratum. It then recorded, correctly, that the CAUSE of the
free-space hump "is not established", and it recorded a failed prediction: the
hump is NOT composition along %free.

That open question is worth closing, because two of this project's five failed
predictions trace to the same unexplained phenomenon, and because "our measure
behaves non-monotonically on 86% of the volume and we do not know why" is a
question a reviewer will ask in the first five minutes.

Two decompositions, both of which A24 could have run and did not.

1. GAP = CONFIDENCE - ACCURACY. A hump in the difference can come from either
   term. If confidence is flat and accuracy dips in the middle deciles, the
   model is genuinely worse there. If accuracy is flat and confidence peaks,
   it is an overconfidence artefact. These are different findings.

2. COMPOSITION ALONG A VARIABLE A24 DID NOT TEST. It tested %free and ruled it
   out. But free space is not one thing: air at 4 m is trivially free and the
   model is certain and right, while the 40 cm above the road surface is
   contested. Observability deciles mix HEIGHTS and RANGES unevenly by
   construction -- a ray that grazes a rooftop and a ray that ends in the road
   surface land in different deciles. If the hump vanishes INSIDE fixed height
   bands and range rings but survives in the aggregate, it is Simpson's
   paradox along geometry rather than a property of observability.

Exact sufficient statistics, no sampling: every voxel is binned by
(GT free/occupied, observability decile, height band, range ring, confidence
bin) and only counts are kept.

EXPLORATORY, after seeing data. This does NOT rescue H1. H1 was pre-registered
over all voxels and lost, and understanding why a stratum behaves oddly is not
the same as passing a hypothesis about the aggregate.
"""
from __future__ import annotations
import argparse, glob, json, os, time
from multiprocessing import Pool
import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FREE, RES, RNG, NZ, Z0, N = 17, 0.4, 40.0, 16, -1.0, 200

NOBS = 11          # bin 0 = obs exactly 0, then deciles 1..10 within obs > 0
NGT = 2            # 0 = GT occupied, 1 = GT free
NH = 4             # height bands, metres above the grid floor
NR = 4             # range rings from the ego
NCONF = 20
NP = 3             # proximity to the nearest GT-occupied voxel: 1, 2, >2 cells
SHAPE = (NGT, NOBS, NH, NR, NP, NCONF)
SZ = int(np.prod(SHAPE))

H_EDGES = (0.6, 1.8, 3.0)      # m above Z0: road band / vehicle band / sign band / air
R_EDGES = (10.0, 20.0, 30.0)   # m from the ego
_G = {}


def _geometry():
    ax = (np.arange(N) + 0.5) * RES - RNG
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    rad = np.sqrt(X ** 2 + Y ** 2)
    ring = np.digitize(rad, R_EDGES).astype(np.int8)          # (200,200)
    z = (np.arange(NZ) + 0.5) * RES + Z0 - Z0                 # height above floor
    band = np.digitize(z, H_EDGES).astype(np.int8)            # (16,)
    return (np.broadcast_to(ring[:, :, None], (N, N, NZ)).ravel(),
            np.broadcast_to(band[None, None, :], (N, N, NZ)).ravel())


RING, BAND = _geometry()


def _init(obs_dir, edges):
    _G["obs"] = obs_dir
    _G["edges"] = edges
    _G["gt"] = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}


def _one(row):
    tok, scene = row
    g = np.load(_G["gt"][tok])
    gc = g["semantics"].astype(np.int16).ravel()
    mk = g["mask_camera"].astype(bool).ravel()
    p = np.load(os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz"))
    pc = p["cls"].astype(np.int16).ravel()
    cf = p["conf"].astype(np.float32).ravel() / 255.0
    ob = np.load(os.path.join(ROOT, _G["obs"], tok + ".npy")).ravel().astype(np.float32) / 255.0

    # Proximity to real structure. A free voxel touching a surface is a
    # different object from a free voxel in open air: the model invents
    # occupancy NEAR things, not in the middle of nothing, so if the hump is a
    # false-positive phenomenon it must live in the first shell.
    occ3 = (g["semantics"] != FREE)
    near1 = ndi.binary_dilation(occ3, iterations=1) & ~occ3
    near2 = ndi.binary_dilation(occ3, iterations=2) & ~occ3 & ~near1
    prox = np.where(near1.ravel(), 0, np.where(near2.ravel(), 1, 2)).astype(np.int64)

    # A5 fixed the scoring scope inside mask_camera before the data existed,
    # and A41 showed what ignoring it costs. Honoured here.
    sel = mk
    gc, pc, cf, ob, prox = gc[sel], pc[sel], cf[sel], ob[sel], prox[sel]
    ring, band = RING[sel], BAND[sel]

    ok = (pc == gc).astype(np.int64)
    gt = (gc == FREE).astype(np.int64)
    oi = np.where(ob <= 0, 0, np.digitize(ob, _G["edges"]) + 1).astype(np.int64)
    oi = np.clip(oi, 0, NOBS - 1)
    ci = np.clip((cf * NCONF).astype(np.int64), 0, NCONF - 1)

    flat = (((((gt * NOBS + oi) * NH + band) * NR + ring) * NP + prox)
            * NCONF + ci)
    n = np.bincount(flat, minlength=SZ)
    k = np.bincount(flat, weights=ok, minlength=SZ)
    s = np.bincount(flat, weights=cf.astype(np.float64), minlength=SZ)
    return scene, n.astype(np.int64), k.astype(np.int64), s


def cmd_edges(a):
    """Bin edges within obs > 0, computed on a sample and then FIXED.

    A43 repairs these. Equal-mass deciles cannot be cut from this population:
    23.2% of it sits at exactly 1.0, which left decile 9 EMPTY and decile 10
    holding 28% of the free stratum. Saturation is a distinct population --
    cells a camera sees completely -- so it gets its own bin, and the rest is
    cut into equal-mass octiles. Nine live bins, none empty.
    """
    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    rng = np.random.default_rng(3)
    toks = [r["token"] for r in index
            if os.path.exists(os.path.join(ROOT, a.obs, r["token"] + ".npy"))]
    sel = rng.choice(toks, size=min(120, len(toks)), replace=False)
    v = []
    for t in sel:
        o = np.load(os.path.join(ROOT, a.obs, t + ".npy")).ravel()
        o = o[o > 0]
        v.append(rng.choice(o, size=min(40000, o.size), replace=False))
    v = np.concatenate(v).astype(np.float32) / 255.0
    sub = v[v < 1.0]
    e = [float(q) for q in np.percentile(sub, np.arange(100 / 8, 100, 100 / 8))]
    e.append(1.0)                     # bin 9 is obs == 1.0 exactly
    print(f"saturated at 1.0: {100*(v >= 1.0).mean():.1f}% of the obs > 0 population")
    json.dump(dict(edges=e, n=int(v.size), scheme="A43 octiles + saturation bin"),
              open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("bin edges within obs > 0, from", f"{v.size:,}", "sampled values")
    print("  " + "  ".join(f"{x:.4f}" for x in e))
    print("wrote", a.out)


def cmd_build(a):
    edges = json.load(open(os.path.join(ROOT, a.edges)))["edges"]
    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    gt = {os.path.basename(os.path.dirname(q)) for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}
    rows = [(r["token"], r["scene"]) for r in index if r["token"] in gt
            and os.path.exists(os.path.join(ROOT, "data/preds/preds_voxel",
                                            r["token"] + ".npz"))
            and os.path.exists(os.path.join(ROOT, a.obs, r["token"] + ".npy"))]
    store = os.path.join(ROOT, a.store)
    scenes, N_, K_, S_, done = [], None, None, None, set()
    if os.path.exists(store):
        z = np.load(store, allow_pickle=True)
        scenes = list(z["scenes"]); N_, K_, S_ = z["n"], z["k"], z["s"]
        done = set(z["done"].tolist())
    todo = [r for r in rows if r[0] not in done][:a.chunk]
    print(f"{len(rows)} frames | stored {len(done)} | this run {len(todo)}", flush=True)
    if not todo:
        print("build complete -- run analyse"); return
    sidx = {s: i for i, s in enumerate(scenes)}
    if N_ is None:
        N_ = np.zeros((0, SZ), np.int64); K_ = np.zeros((0, SZ), np.int64)
        S_ = np.zeros((0, SZ), np.float64)
    t0 = time.time()
    with Pool(max(os.cpu_count() - 1, 1), initializer=_init,
              initargs=(a.obs, edges)) as pool:
        for i, (scene, n, k, s) in enumerate(pool.imap_unordered(_one, todo, chunksize=4)):
            if scene not in sidx:
                sidx[scene] = len(scenes); scenes.append(scene)
                N_ = np.concatenate([N_, np.zeros((1, SZ), np.int64)])
                K_ = np.concatenate([K_, np.zeros((1, SZ), np.int64)])
                S_ = np.concatenate([S_, np.zeros((1, SZ), np.float64)])
            j = sidx[scene]
            N_[j] += n; K_[j] += k; S_[j] += s
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(todo)}  {el/(i+1):.3f}s/frame  "
                      f"eta {(len(todo)-i-1)*el/(i+1)/60:.1f} min", flush=True)
    done |= {r[0] for r in todo}
    os.makedirs(os.path.dirname(store), exist_ok=True)
    np.savez_compressed(store, scenes=np.array(scenes), n=N_, k=K_, s=S_,
                        done=np.array(sorted(done)))
    left = len(rows) - len(done)
    print(f"chunk done, {left} remain" if left > 0 else
          "build complete -- run analyse", flush=True)


def _curve(n, k, s):
    """gap, confidence and accuracy by observability decile, given arrays
    already reduced to (NOBS, NCONF) or (NOBS,...,NCONF)."""
    ax = tuple(range(1, n.ndim))
    tot = n.sum(ax); hit = k.sum(ax); cf = s.sum(ax)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(tot > 0, hit / np.maximum(tot, 1), np.nan)
        con = np.where(tot > 0, cf / np.maximum(tot, 1), np.nan)
    return con - acc, con, acc, tot


def _rankcorr(gap, tot, skip_zero_bin=True):
    """Rank correlation of gap against decile index, the statistic H1 was
    pre-registered on. Used instead of the max-rise below wherever cells are
    small: max-rise is biased UPWARD by noise, so comparing a noisy cell's
    max-rise against the aggregate's would manufacture the answer."""
    i0 = 1 if skip_zero_bin else 0
    g, t = gap[i0:], tot[i0:]
    m = (~np.isnan(g)) & (t > 0)
    if m.sum() < 4:
        return np.nan
    x = np.arange(len(g))[m].astype(float)
    y = g[m]
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def _humped(gap):
    """A hump = the decile-1..10 curve is not monotone decreasing. Reported as
    the rise from its minimum to a later maximum, in gap points."""
    g = gap[1:]
    g = g[~np.isnan(g)]
    if g.size < 3:
        return np.nan
    best = 0.0
    for i in range(g.size - 1):
        best = max(best, float(g[i + 1:].max() - g[i]))
    return best


def cmd_analyse(a):
    z = np.load(os.path.join(ROOT, a.store), allow_pickle=True)
    N_, K_, S_ = z["n"], z["k"], z["s"]
    n = N_.sum(0).reshape(SHAPE); k = K_.sum(0).reshape(SHAPE); s = S_.sum(0).reshape(SHAPE)
    pn = ["touching a surface", "one cell away", "open air (2+ cells)"]
    print(f"\n{len(z['scenes'])} scenes, {int(n.sum()):,} voxels inside mask_camera")
    hb = ["road 0.0-0.6", "vehicle 0.6-1.8", "sign 1.8-3.0", "air 3.0+"]
    rb = ["0-10 m", "10-20 m", "20-30 m", "30-57 m"]
    out = {}

    # ---- 1. reproduce the hump, then decompose it -------------------------
    for gi, gname in ((1, "GT FREE"), (0, "GT OCCUPIED")):
        gap, con, acc, tot = _curve(n[gi], k[gi], s[gi])
        print("\n" + "=" * 78)
        print(f"{gname}   {int(tot.sum()):,} voxels")
        print("-" * 78)
        print(f"{'obs bin':<12}{'voxels':>16}{'confidence':>13}{'accuracy':>11}{'gap':>10}")
        for i in range(NOBS):
            lab = "obs = 0" if i == 0 else f"decile {i}"
            print(f"{lab:<12}{int(tot[i]):>16,}{con[i]:>13.4f}{acc[i]:>11.4f}{gap[i]:>10.4f}")
        rc = _rankcorr(gap, tot)
        print(f"hump size (rise after the minimum, deciles 1-10): {_humped(gap):+.4f}"
              f"   |   rank corr vs decile: {rc:+.3f}")
        out[gname] = dict(gap=[float(x) for x in gap], conf=[float(x) for x in con],
                          acc=[float(x) for x in acc], n=[int(x) for x in tot],
                          hump=float(_humped(gap)), rankcorr=rc)

    # ---- 2. is the free-space hump composition along height or range? -----
    print("\n" + "=" * 78)
    print("GT FREE, hump size INSIDE each fixed height band and range ring")
    print("If the hump is composition along geometry it collapses in these cells.")
    print("-" * 78)
    print(f"{'height band':<20}" + "".join(f"{r:>19}" for r in rb))
    print(f"{'':<20}" + "".join(f"{'rankcorr / Mvox':>19}" for _ in rb))
    cell = {}
    for h in range(NH):
        line = f"{hb[h]:<20}"
        for r in range(NR):
            gap, con, acc, tot = _curve(n[1, :, h, r].reshape(NOBS, -1),
                                        k[1, :, h, r].reshape(NOBS, -1),
                                        s[1, :, h, r].reshape(NOBS, -1))
            rc = _rankcorr(gap, tot)
            cell[f"{hb[h]}|{rb[r]}"] = dict(rankcorr=float(rc),
                                            hump=float(_humped(gap)),
                                            n=int(tot.sum()))
            line += (f"{rc:>+11.3f} /{tot.sum()/1e6:>6.1f}"
                     if not np.isnan(rc) else f"{'--':>19}")
        print(line)
    out["free_hump_by_cell"] = cell
    agg = out["GT FREE"]["rankcorr"]
    big = {k_: c for k_, c in cell.items() if c["n"] >= 5_000_000
           and not np.isnan(c["rankcorr"])}
    vals = [c["rankcorr"] for c in big.values()]
    wts = [c["n"] for c in big.values()]
    pooled = float(np.average(vals, weights=wts)) if vals else float("nan")
    print("-" * 78)
    print(f"aggregate rank corr {agg:+.3f}   |   voxel-weighted mean over the "
          f"{len(big)} cells with >= 5M voxels {pooled:+.3f}")
    print("A monotone DECLINE is -1.0. If geometry composition caused the hump,")
    print("the within-cell figure should be far more negative than the aggregate.")
    out["aggregate_rankcorr"] = agg
    out["within_cell_rankcorr"] = pooled

    # ---- 2b. the SAME test on GT OCCUPIED. A24's surviving claim rests on
    # the occupied decline, so it has to face the identical confound check.
    # Reporting the free-space result without this one would be choosing which
    # stratum gets audited.
    print("\n" + "=" * 78)
    print("GT OCCUPIED, rank corr of gap vs decile INSIDE each height x range cell")
    print("-" * 78)
    print(f"{'height band':<20}" + "".join(f"{r:>19}" for r in rb))
    ocell = {}
    for h in range(NH):
        line = f"{hb[h]:<20}"
        for r in range(NR):
            gap, con, acc, tot = _curve(n[0, :, h, r].reshape(NOBS, -1),
                                        k[0, :, h, r].reshape(NOBS, -1),
                                        s[0, :, h, r].reshape(NOBS, -1))
            rc = _rankcorr(gap, tot)
            ocell[f"{hb[h]}|{rb[r]}"] = dict(rankcorr=float(rc), n=int(tot.sum()))
            line += (f"{rc:>+11.3f} /{tot.sum()/1e6:>6.1f}"
                     if not np.isnan(rc) else f"{'--':>19}")
        print(line)
    obig = {k_: c for k_, c in ocell.items() if c["n"] >= 1_000_000
            and not np.isnan(c["rankcorr"])}
    op = float(np.average([c["rankcorr"] for c in obig.values()],
                          weights=[c["n"] for c in obig.values()])) if obig else float("nan")
    print("-" * 78)
    print(f"aggregate rank corr {out['GT OCCUPIED']['rankcorr']:+.3f}   |   "
          f"within-cell mean over {len(obig)} cells >= 1M voxels {op:+.3f}")
    out["occupied_hump_by_cell"] = ocell
    out["occupied_within_cell_rankcorr"] = op

    # ---- 2c. the actual curves in the largest cells, because a rank
    # correlation with no curve under it is not evidence.
    print("\n" + "=" * 78)
    print("gap by decile inside the three largest GT FREE cells")
    print("-" * 78)
    order = sorted(cell.items(), key=lambda kv: -kv[1]["n"])[:3]
    curves = {}
    for key, meta in order:
        hname, rname = key.split("|")
        h = hb.index(hname); r = rb.index(rname)
        gap, con, acc, tot = _curve(n[1, :, h, r].reshape(NOBS, -1),
                                    k[1, :, h, r].reshape(NOBS, -1),
                                    s[1, :, h, r].reshape(NOBS, -1))
        print(f"{key:<34}" + "  ".join(f"{gap[i]:.4f}" for i in range(1, NOBS)
                                       if tot[i] > 0))
        curves[key] = [float(gap[i]) for i in range(1, NOBS) if tot[i] > 0]
    out["largest_free_cell_curves"] = curves

    # ---- 2d. the mechanism test: is the hump a FALSE-POSITIVE phenomenon? --
    print("\n" + "=" * 78)
    print("GT FREE by proximity to the nearest ground-truth occupied voxel")
    print("-" * 78)
    print(f"{'shell':<24}{'voxels':>16}{'share':>9}{'conf':>9}{'acc':>9}"
          f"{'gap':>9}{'rankcorr':>11}")
    prox_rows = {}
    for q in range(NP):
        gap, con, acc, tot = _curve(n[1, :, :, :, q], k[1, :, :, :, q], s[1, :, :, :, q])
        t = tot.sum()
        if t == 0:
            continue
        cw = float(np.nansum(con * tot) / t); aw = float(np.nansum(acc * tot) / t)
        rc = _rankcorr(gap, tot)
        print(f"{pn[q]:<24}{int(t):>16,}{100*t/n[1].sum():>8.1f}%{cw:>9.4f}"
              f"{aw:>9.4f}{cw-aw:>9.4f}{rc:>+11.3f}")
        prox_rows[pn[q]] = dict(n=int(t), share=float(t / n[1].sum()), conf=cw,
                                acc=aw, gap=float(cw - aw), rankcorr=float(rc),
                                gap_by_decile=[float(x) for x in gap])
    out["free_by_proximity"] = prox_rows
    print("\ngap by decile within each shell")
    for q in range(NP):
        gap, _, _, tot = _curve(n[1, :, :, :, q], k[1, :, :, :, q], s[1, :, :, :, q])
        print(f"{pn[q]:<24}" + "  ".join(f"{gap[i]:.4f}" for i in range(1, NOBS)
                                         if tot[i] > 0))

    # ---- 3. what does free space look like in each band -------------------
    print("\n" + "=" * 78)
    print("GT FREE composition and difficulty by height band (all deciles pooled)")
    print("-" * 78)
    print(f"{'band':<20}{'voxels':>16}{'share':>9}{'confidence':>13}{'accuracy':>11}{'gap':>10}")
    band_rows = {}
    for h in range(NH):
        t = n[1, :, h].sum(); hh = k[1, :, h].sum(); cc = s[1, :, h].sum()
        if t == 0:
            continue
        print(f"{hb[h]:<20}{int(t):>16,}{100*t/n[1].sum():>8.1f}%"
              f"{cc/t:>13.4f}{hh/t:>11.4f}{cc/t-hh/t:>10.4f}")
        band_rows[hb[h]] = dict(n=int(t), share=float(t / n[1].sum()),
                                conf=float(cc / t), acc=float(hh / t),
                                gap=float(cc / t - hh / t))
    out["free_by_band"] = band_rows

    # ---- 4. does the decile mix of heights shift? the composition test ----
    print("\n" + "=" * 78)
    print("height mix WITHIN each observability decile, GT FREE (% of the decile)")
    print("-" * 78)
    print(f"{'obs bin':<12}" + "".join(f"{h:>18}" for h in hb))
    mix = {}
    for i in range(NOBS):
        t = n[1, i].sum()
        if t == 0:
            continue
        lab = "obs = 0" if i == 0 else f"decile {i}"
        shares = [float(n[1, i, h].sum() / t) for h in range(NH)]
        print(f"{lab:<12}" + "".join(f"{100*x:>17.1f}%" for x in shares))
        mix[lab] = shares
    out["height_mix_by_decile"] = mix

    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("\nwrote", a.out)


def cmd_h1(a):
    """A43. The single re-test of H1 on the repaired binning. Runs once."""
    z = np.load(os.path.join(ROOT, a.store), allow_pickle=True)
    N_, K_, S_ = z["n"], z["k"], z["s"]
    n = N_.sum(0).reshape(SHAPE); k = K_.sum(0).reshape(SHAPE); s = S_.sum(0).reshape(SHAPE)
    # H1's population: ALL voxels inside mask_camera, both GT strata together.
    gap, con, acc, tot = _curve(n.sum(0).reshape(NOBS, -1),
                                k.sum(0).reshape(NOBS, -1),
                                s.sum(0).reshape(NOBS, -1))
    lab = (["obs = 0"] + [f"bin {i}" for i in range(1, 9)] + ["obs = 1.0"]
           + [f"unused {i}" for i in range(NOBS - 10)])
    print("\nA43  H1 RE-TEST -- all voxels inside mask_camera, repaired binning")
    print("=" * 74)
    print(f"{'bin':<12}{'voxels':>18}{'confidence':>13}{'accuracy':>11}{'gap':>10}")
    print("-" * 74)
    for i in range(NOBS):
        if tot[i] == 0:
            print(f"{lab[i]:<12}{'EMPTY':>18}")
            continue
        print(f"{lab[i]:<12}{int(tot[i]):>18,}{con[i]:>13.4f}{acc[i]:>11.4f}{gap[i]:>10.4f}")
    live = [i for i in range(1, NOBS) if tot[i] > 0]
    g = np.array([gap[i] for i in live])
    steps = np.diff(g)
    rev = int((steps > 0).sum())
    rc = _rankcorr(gap, tot)
    print("=" * 74)
    print(f"live bins {len(live)} of 9   |   reversals {rev}   |   "
          f"Spearman {rc:+.3f}")
    holds = (rev == 0) and (rc <= -0.5)
    print("\nACCEPTANCE (fixed in A43 before this ran):")
    print(f"  strictly decreasing across all live bins   "
          f"{'PASS' if rev == 0 else f'FAIL -- {rev} reversal(s)'}")
    print(f"  Spearman <= -0.50                          "
          f"{'PASS' if rc <= -0.5 else 'FAIL'}  ({rc:+.3f})")
    print(f"\nH1 on the repaired binning: {'HOLDS' if holds else 'FAILS'}")
    json.dump(dict(scheme="A43 octiles + saturation bin",
                   bins=lab, n=[int(x) for x in tot],
                   gap=[float(x) for x in gap], conf=[float(x) for x in con],
                   acc=[float(x) for x in acc], live=len(live),
                   reversals=rev, spearman=float(rc), holds=bool(holds)),
              open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("edges"); e.set_defaults(f=cmd_edges)
    e.add_argument("--obs", default="data/pack/obs_max")
    e.add_argument("--out", default="outputs/artifacts/hump_edges.json")
    b = sub.add_parser("build"); b.set_defaults(f=cmd_build)
    b.add_argument("--obs", default="data/pack/obs_max")
    b.add_argument("--edges", default="outputs/artifacts/hump_edges.json")
    b.add_argument("--chunk", type=int, default=100000)
    b.add_argument("--store", default="outputs/artifacts/hump_store.npz")
    n_ = sub.add_parser("analyse"); n_.set_defaults(f=cmd_analyse)
    n_.add_argument("--store", default="outputs/artifacts/hump_store.npz")
    n_.add_argument("--out", default="outputs/artifacts/freespace_hump.json")
    h = sub.add_parser("h1retest"); h.set_defaults(f=cmd_h1)
    h.add_argument("--store", default="outputs/artifacts/hump_store.npz")
    h.add_argument("--out", default="outputs/artifacts/h1_retest.json")
    a = ap.parse_args(); a.f(a)
