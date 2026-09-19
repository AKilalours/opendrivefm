#!/usr/bin/env python3
"""Hard-case mining and retrieval over the validation fleet.

The question a data engine has to answer is not "where was the model wrong" --
that needs labels, and on real fleet data there are none. It is:

    can an UNLABELLED signal, computable from geometry and the model's own
    output, find the frames where the model is wrong?

So this tool is built in two halves that never touch each other:

  MINING SIGNALS   computed from predictions and camera observability only.
                   No ground truth. These are what would run on the fleet.

  VALIDATION       per-frame error measured against Occ3D. Used ONLY to score
                   how well each mining signal ranks frames. Never an input.

Mixing the two is the failure mode this repo keeps catching in other people's
work and in its own: a "miner" scored against the labels it secretly used.

Retrieval is exact, not approximate. 6,019 frames x 84 dimensions is 2 MB; an
exact inner product over that is sub-millisecond, and an ANN index would be
slower to build than to skip. FAISS earns its place at millions of vectors,
not thousands -- adding it here would be resume-driven engineering.

    build     one pass, parallel over cores, writes SQLite + descriptors
    rank      top frames by a mining signal          (SQL)
    similar   nearest frames to a given frame        (cosine)
    validate  do the unlabelled signals find the errors?
    export    a reproducible mined candidate set
"""
from __future__ import annotations
import argparse, glob, json, os, sqlite3, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FREE, RES, RNG, NZ, N = 17, 0.4, 40.0, 16, 200
OBST = np.arange(0, 11)
DB = os.path.join(ROOT, "outputs/artifacts/mine.sqlite")
VEC = os.path.join(ROOT, "outputs/artifacts/mine_vectors.npz")

SIGNALS = {
    "blind_commit": "voxels asserted p>=0.70 with NO camera evidence",
    "dim_commit":   "voxels asserted p>=0.70 in the barely-observed band",
    "obst_dark":    "obstacle columns with zero camera coverage",
    "low_margin":   "occupied voxels where top-1 and top-2 are within 0.10",
    "mean_obs":     "mean observability over predicted-occupied (low = suspect)",
    "obst_n":       "obstacle columns present",
}

_G = {}


def _init(obs_dir):
    _G["obs"] = obs_dir
    _G["gt"] = {os.path.basename(os.path.dirname(q)): q for q in glob.glob(
        os.path.join(ROOT, "data/occ3d/Occupancy3D-nuScenes-trainval/gts",
                     "*", "*", "labels.npz"))}


def _one(row):
    tok, scene, ts = row
    p = np.load(os.path.join(ROOT, "data/preds/preds_voxel", tok + ".npz"))
    cls = p["cls"].astype(np.int16)
    conf = p["conf"].astype(np.float32) / 255.0
    conf2 = p["conf2"].astype(np.float32) / 255.0
    obs = np.load(os.path.join(ROOT, _G["obs"], tok + ".npy")).astype(np.float32) / 255.0

    occ = cls != FREE
    committed = conf >= 0.70
    blind = obs <= 0.0
    dim = (obs > 0) & (obs <= 0.35)

    # ---- mining signals: predictions + geometry only, no ground truth ----
    sig = dict(
        n_occ=int(occ.sum()),
        blind_commit=int((committed & blind & occ).sum()),
        dim_commit=int((committed & dim & occ).sum()),
        low_margin=int((occ & ((conf - conf2) < 0.10)).sum()),
        mean_obs=float(obs[occ].mean()) if occ.any() else 0.0,
        conf_occ=float(conf[occ].mean()) if occ.any() else 0.0,
        margin_occ=float((conf - conf2)[occ].mean()) if occ.any() else 0.0,
    )
    is_ob = np.isin(cls, OBST)
    ob_col = is_ob.any(-1)
    zi = np.argmax(np.where(is_ob, conf, -1.0), -1)
    o_obs = np.take_along_axis(obs, zi[..., None], -1)[..., 0]
    sig["obst_n"] = int(ob_col.sum())
    sig["obst_dark"] = int((ob_col & (o_obs <= 0.0)).sum())

    # ---- descriptor for retrieval ----
    top = np.where(occ.any(-1), 15 - np.argmax(occ[:, :, ::-1], -1), 0)
    has = occ.any(-1)
    scls = np.take_along_axis(cls, top[..., None], -1)[..., 0][has]
    sobs = np.take_along_axis(obs, top[..., None], -1)[..., 0][has]
    ii, jj = np.nonzero(ob_col)
    x = (ii + .5) * RES - RNG
    y = (jj + .5) * RES - RNG
    rad = np.hypot(x, y)
    az = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
    nz = lambda a, b, r: (np.histogram(a, bins=b, range=r)[0].astype(np.float32)
                          / max(len(a), 1))
    d = np.concatenate([
        nz(scls, 18, (0, 18)),          # what classes are present
        nz(sobs, 16, (0, 1)),           # how well the surface is seen
        nz(rad, 10, (0, 56)),           # where obstacles sit in range
        nz(az, 8, (0, 1)),              # and in azimuth
        nz(top[has], 16, (0, 16)),      # vertical profile
        nz(conf[occ], 16, (0, 1)),      # confidence profile
    ])
    d = d / max(float(np.linalg.norm(d)), 1e-9)

    # ---- validation only. Never fed back into a signal. ----
    err = err_occ = gap_occ = float("nan")
    gp = _G["gt"].get(tok)
    if gp:
        g = np.load(gp)
        gc = g["semantics"].astype(np.int16)
        m = g["mask_camera"].astype(bool)
        wrong = cls != gc
        if m.any():
            err = float(wrong[m].mean())
        mo = m & (gc != FREE)
        if mo.any():
            err_occ = float(wrong[mo].mean())
            # calibration gap, NOT error rate. A14 showed these are different
            # quantities; this file is where that distinction gets tested.
            gap_occ = float(conf[mo].mean() - (~wrong[mo]).mean())
    return tok, scene, ts, sig, d.astype(np.float32), err, err_occ, gap_occ


def cmd_build(a):
    index = json.load(open(os.path.join(ROOT, "data/pack/index.json")))
    rows = [(r["token"], r["scene"], r["timestamp"]) for r in index
            if os.path.exists(os.path.join(ROOT, "data/preds/preds_voxel",
                                           r["token"] + ".npz"))
            and os.path.exists(os.path.join(ROOT, a.obs, r["token"] + ".npy"))]
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS frames(
        token TEXT PRIMARY KEY, scene TEXT, ts INTEGER,
        n_occ INT, blind_commit INT, dim_commit INT, obst_dark INT,
        low_margin INT, obst_n INT, mean_obs REAL, conf_occ REAL,
        margin_occ REAL, err REAL, err_occ REAL, gap_occ REAL)""")
    con.execute("CREATE INDEX IF NOT EXISTS ix_scene ON frames(scene)")
    done = {r[0] for r in con.execute("SELECT token FROM frames")}
    todo = [r for r in rows if r[0] not in done][:a.chunk]
    print(f"{len(rows)} frames | stored {len(done)} | this run {len(todo)} "
          f"| {os.cpu_count()} cores", flush=True)
    if not todo:
        print("build complete -- run validate"); con.close(); return

    vecs = {}
    if os.path.exists(VEC):
        z = np.load(VEC, allow_pickle=True)
        vecs = {t: v for t, v in zip(z["tok"], z["vec"])}
    t0 = time.time()
    with Pool(max(os.cpu_count() - 1, 1), initializer=_init,
              initargs=(a.obs,)) as pool:
        for n, (tok, scene, ts, sig, d, err, err_occ, gap_occ) in enumerate(
                pool.imap_unordered(_one, todo, chunksize=4)):
            con.execute("INSERT OR REPLACE INTO frames VALUES "
                        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (tok, scene, ts, sig["n_occ"], sig["blind_commit"],
                         sig["dim_commit"], sig["obst_dark"], sig["low_margin"],
                         sig["obst_n"], sig["mean_obs"], sig["conf_occ"],
                         sig["margin_occ"], err, err_occ, gap_occ))
            vecs[tok] = d
            if (n + 1) % 200 == 0:
                el = time.time() - t0
                con.commit()
                print(f"  {n+1}/{len(todo)}  {el/(n+1):.3f}s/frame  "
                      f"eta {(len(todo)-n-1)*el/(n+1)/60:.1f} min", flush=True)
    con.commit()
    toks = sorted(vecs)
    np.savez_compressed(VEC, tok=np.array(toks),
                        vec=np.stack([vecs[t] for t in toks]))
    left = len(rows) - len(done) - len(todo)
    print(f"chunk done, {left} remain" if left > 0 else
          "build complete -- run validate", flush=True)
    con.close()


def _load_vecs():
    z = np.load(VEC, allow_pickle=True)
    return list(z["tok"]), np.stack(z["vec"]).astype(np.float32)


def cmd_rank(a):
    con = sqlite3.connect(DB)
    order = "ASC" if a.signal == "mean_obs" else "DESC"
    q = (f"SELECT token, scene, {a.signal}, obst_n, err_occ FROM frames "
         f"WHERE obst_n >= ? ORDER BY {a.signal} {order} LIMIT ?")
    print(f"\ntop {a.top} frames by {a.signal}  ({SIGNALS[a.signal]})")
    print(f"{'token':<34}{'scene':<13}{a.signal:>14}{'obst':>7}{'err|occ':>10}")
    print("-" * 78)
    for tok, sc, v, nb, e in con.execute(q, (a.min_obst, a.top)):
        ev = "  n/a" if e is None or np.isnan(e) else f"{100*e:5.1f}%"
        vv = f"{v:.4f}" if isinstance(v, float) else f"{v:,}"
        print(f"{tok:<34}{sc:<13}{vv:>14}{nb:>7,}{ev:>10}")
    con.close()


def cmd_similar(a):
    toks, V = _load_vecs()
    if a.token not in toks:
        sys.exit(f"token not in index: {a.token}")
    i = toks.index(a.token)
    s = V @ V[i]
    o = np.argsort(-s)[:a.top + 1]
    con = sqlite3.connect(DB)
    meta = {r[0]: r[1:] for r in con.execute(
        "SELECT token, scene, obst_n, blind_commit, err_occ FROM frames")}
    print(f"\nframes most similar to {a.token}")
    print(f"{'token':<34}{'scene':<13}{'cos':>7}{'obst':>7}{'blind':>8}{'err|occ':>10}")
    print("-" * 79)
    for k in o:
        t = toks[k]
        sc, nb, bc, e = meta.get(t, ("?", 0, 0, None))
        ev = "  n/a" if e is None or np.isnan(e) else f"{100*e:5.1f}%"
        tag = "  <- query" if k == i else ""
        print(f"{t:<34}{sc:<13}{s[k]:>7.4f}{nb:>7,}{bc:>8,}{ev:>10}{tag}")
    con.close()


def auroc(score, y):
    s = np.asarray(score, float); y = np.asarray(y, bool)
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    u, first, cnt = np.unique(s[o], return_index=True, return_counts=True)
    for f, c in zip(first, cnt):
        if c > 1:
            r[o[f:f + c]] = r[o[f:f + c]].mean()
    P, Nn = y.sum(), (~y).sum()
    if P == 0 or Nn == 0:
        return float("nan")
    return float((r[y].sum() - P * (P + 1) / 2) / (P * Nn))


def cmd_validate(a):
    con = sqlite3.connect(DB)
    cols = ["blind_commit", "dim_commit", "obst_dark", "low_margin",
            "mean_obs", "obst_n", "n_occ", "margin_occ",
            "err_occ", "gap_occ", "scene"]
    rs = list(con.execute(f"SELECT {','.join(cols)} FROM frames "
                          f"WHERE err_occ IS NOT NULL AND obst_n >= ?",
                          (a.min_obst,)))
    con.close()
    if not rs:
        sys.exit("nothing built yet")
    A = np.array([[r[i] for i in range(8)] for r in rs], float)
    e = np.array([r[8] for r in rs], float)
    gp = np.array([r[9] for r in rs], float)
    nocc = np.maximum(A[:, 6], 1.0)
    # RATE versions. A raw count scales with how much stuff is in the scene, so
    # it measures scene density, not model trouble. The intensive form is the
    # one that can generalise across a dense intersection and an empty highway.
    RATES = {
        "blind_commit_rate": (A[:, 0] / nocc, "blind_commit normalised by occupied voxels"),
        "dim_commit_rate":   (A[:, 1] / nocc, "dim_commit normalised by occupied voxels"),
        "obst_dark_rate":    (A[:, 2] / np.maximum(A[:, 5], 1.0), "fraction of obstacle columns with no coverage"),
        "low_margin_rate":   (A[:, 3] / nocc, "fraction of occupied voxels with an ambiguous top-2"),
        "mean_margin":       (-A[:, 7], "mean top1-top2 margin, inverted"),
    }
    SIG = {c: (-A[:, i] if c == "mean_obs" else A[:, i])
           for i, c in enumerate(cols[:6])}
    SIG.update({k: v[0] for k, v in RATES.items()})
    DESC = dict(SIGNALS)
    DESC.update({k: v[1] for k, v in RATES.items()})

    out = {}
    print(f"\n{len(rs):,} frames with >= {a.min_obst} obstacle columns")
    for name, target, expl in (
            ("WRONG", e, "frames with the highest error rate on occupied voxels"),
            ("OVERCONFIDENT", gp, "frames with the largest confidence-minus-accuracy gap")):
        cut = np.quantile(target, 1 - a.frac)
        y = target >= cut
        base = y.mean()
        print(f"\nTARGET: {name}  --  worst {100*a.frac:.0f}%, {expl}")
        print(f"        threshold {cut:+.4f}, {int(y.sum()):,} frames")
        print("=" * 80)
        print(f"{'mining signal':<20}{'AUROC':>9}{'prec@100':>11}{'lift':>8}  description")
        print("-" * 80)
        rows_ = []
        for c, sc in SIG.items():
            au = auroc(sc, y)
            p = y[np.argsort(-sc)[:100]].mean()
            rows_.append((au, c, p))
        for au, c, p in sorted(rows_, reverse=True):
            print(f"{c:<20}{au:>9.4f}{100*p:>10.0f}%{p/base:>8.2f}x  {DESC[c]}")
            out.setdefault(name, {})[c] = dict(auroc=au, prec_at_100=float(p),
                                               lift=float(p / base))
        print("-" * 80)
        print(f"{'random':<20}{0.5:>9.4f}{100*base:>10.0f}%{1.0:>8.2f}x  baseline any miner must beat")
        print("=" * 80)
    base = float((e >= np.quantile(e, 1 - a.frac)).mean())
    json.dump(dict(frames=len(rs), target_frac=a.frac, base_rate=base,
                   targets=out),
              open(os.path.join(ROOT, "outputs/artifacts/mining_validation.json"), "w"),
              indent=1)
    print("wrote outputs/artifacts/mining_validation.json")


def cmd_export(a):
    con = sqlite3.connect(DB)
    order = "ASC" if a.signal == "mean_obs" else "DESC"
    rs = list(con.execute(
        f"SELECT token, scene, {a.signal}, obst_n FROM frames WHERE obst_n >= ? "
        f"ORDER BY {a.signal} {order} LIMIT ?", (a.min_obst, a.top)))
    con.close()
    p = os.path.join(ROOT, f"outputs/artifacts/mined_{a.signal}_{a.top}.json")
    json.dump(dict(signal=a.signal, description=SIGNALS[a.signal],
                   min_obstacle_columns=a.min_obst, n=len(rs),
                   note="ranked without ground truth; reproducible from mine.sqlite",
                   frames=[dict(token=t, scene=s, value=v, obstacles=n)
                           for t, s, v, n in rs]), open(p, "w"), indent=1)
    print(f"wrote {len(rs)} frames -> {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--obs", default="data/pack/obs_max")
    b.add_argument("--chunk", type=int, default=100000); b.set_defaults(fn=cmd_build)
    r = sub.add_parser("rank"); r.add_argument("--signal", default="blind_commit",
                                               choices=list(SIGNALS))
    r.add_argument("--top", type=int, default=15); r.add_argument("--min-obst", type=int, default=200)
    r.set_defaults(fn=cmd_rank)
    s = sub.add_parser("similar"); s.add_argument("token"); s.add_argument("--top", type=int, default=10)
    s.set_defaults(fn=cmd_similar)
    v = sub.add_parser("validate"); v.add_argument("--frac", type=float, default=0.10)
    v.add_argument("--min-obst", type=int, default=200); v.set_defaults(fn=cmd_validate)
    e = sub.add_parser("export"); e.add_argument("--signal", default="blind_commit",
                                                 choices=list(SIGNALS))
    e.add_argument("--top", type=int, default=500); e.add_argument("--min-obst", type=int, default=200)
    e.set_defaults(fn=cmd_export)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
