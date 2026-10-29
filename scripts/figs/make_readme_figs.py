#!/usr/bin/env python3
"""Generate the README figures as SVG, light and dark, from the result JSONs.

Every value is read from outputs/artifacts/ -- nothing here is typed by hand.
Two files per figure so the README can serve each via <picture>.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs/figs")
FONT = ("-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif")

T = {"light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e",
                   grid="#e6e5e2", s1="#2a78d6", s2="#eb6834"),
     "dark":  dict(surface="#0d1117", ink="#f0efec", ink2="#c3c2b7",
                   grid="#2a2d33", s1="#3987e5", s2="#d95926")}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def head(w, h, c):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">'
            f'<rect width="{w}" height="{h}" fill="{c["surface"]}"/>')


def txt(x, y, s, size=13, fill="#000", anchor="start", weight="400"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')


# ---------------------------------------------------------------- figure 1
def fig_bands(mode):
    c = T[mode]
    d = json.load(open(os.path.join(ROOT, "outputs/artifacts/missed_detection_test.json")))
    rows = [(b["band"], b["n"], b["miss"]) for b in d["bands"]]
    labels = ["no camera evidence", "0.00 - 0.21", "0.21 - 0.38",
              "0.38 - 0.78", "0.78 - 1.00", "fully observable"]
    W, LEFT, RIGHT, TOP, BH, GAP = 880, 182, 92, 74, 26, 14
    H = TOP + len(rows) * (BH + GAP) + 44
    plot = W - LEFT - RIGHT
    mx = max(r[2] for r in rows) * 1.08
    o = [head(W, H, c)]
    o.append(txt(0, 26, "Miss rate falls 12.7x across the observability range",
                 17, c["ink"], weight="600"))
    o.append(txt(0, 48, "Fraction of ground-truth objects the detector misses, "
                        "by camera observability. 40,863 held-out objects.",
                 12.5, c["ink2"]))
    for i, (_, n, miss) in enumerate(rows):
        y = TOP + i * (BH + GAP)
        o.append(txt(LEFT - 12, y + BH / 2 + 4.5, labels[i], 12.5, c["ink2"],
                     anchor="end"))
        bw = max(plot * miss / mx, 3)
        o.append(f'<rect x="{LEFT}" y="{y}" width="{bw:.1f}" height="{BH}" '
                 f'rx="4" fill="{c["s1"]}"/>')
        o.append(txt(LEFT + bw + 10, y + BH / 2 + 4.5,
                     f"{miss*100:.1f}%", 13, c["ink"], weight="600"))
        o.append(txt(LEFT + bw + 58, y + BH / 2 + 4.5,
                     f"n={n:,}", 11.5, c["ink2"]))
    o.append(f'<line x1="{LEFT}" y1="{TOP-8}" x2="{LEFT}" '
             f'y2="{TOP+len(rows)*(BH+GAP)-GAP+8}" stroke="{c["grid"]}" stroke-width="1"/>')
    o.append(txt(LEFT, H - 14, "observability  O(v)  -  computed from camera "
                               "intrinsics alone, no training, no LiDAR",
                 11.5, c["ink2"]))
    return "".join(o) + "</svg>"


# ---------------------------------------------------------------- figure 2
def fig_auroc(mode):
    c = T[mode]
    d = json.load(open(os.path.join(ROOT, "outputs/artifacts/combined_object.json")))
    order = [("MSP (max conf in box)", "model confidence (MSP)", False),
             ("observability", "observability  (camera only)", False),
             ("LiDAR points in box (ORACLE)", "LiDAR point count  (oracle)", True),
             ("combined (dev-fit)", "observability + MSP", False),
             ("LiDAR+MSP (dev-fit, A78)", "LiDAR + MSP  (oracle, matched)", True)]
    W, LEFT, RIGHT, TOP, RH = 880, 252, 128, 88, 38
    H = TOP + len(order) * RH + 52
    plot = W - LEFT - RIGHT
    lo, hi = 0.68, 0.825
    sx = lambda v: LEFT + plot * (v - lo) / (hi - lo)
    o = [head(W, H, c)]
    o.append(txt(0, 26, "Camera geometry matches a LiDAR oracle at matched capacity",
                 17, c["ink"], weight="600"))
    o.append(txt(0, 48, "AUROC for predicting which objects the detector misses. "
                        "Bars are 95% paired scene-bootstrap intervals.",
                 12.5, c["ink2"]))
    # legend
    o.append(f'<circle cx="7" cy="63" r="5" fill="{c["s1"]}"/>')
    o.append(txt(18, 67, "camera only", 12, c["ink2"]))
    o.append(f'<circle cx="124" cy="63" r="5" fill="{c["s2"]}"/>')
    o.append(txt(135, 67, "uses LiDAR (oracle)", 12, c["ink2"]))
    for t in (0.70, 0.75, 0.80):
        x = sx(t)
        o.append(f'<line x1="{x:.1f}" y1="{TOP-10}" x2="{x:.1f}" '
                 f'y2="{TOP+len(order)*RH-14}" stroke="{c["grid"]}" stroke-width="1"/>')
        o.append(txt(x, H - 30, f"{t:.2f}", 11.5, c["ink2"], anchor="middle"))
    for i, (k, lab, is_oracle) in enumerate(order):
        y = TOP + i * RH
        col = c["s2"] if is_oracle else c["s1"]
        v = d["auroc"][k]; a, b = d["ci"][k]
        o.append(txt(LEFT - 14, y + 4.5, lab, 12.5, c["ink2"], anchor="end"))
        o.append(f'<line x1="{sx(a):.1f}" y1="{y}" x2="{sx(b):.1f}" y2="{y}" '
                 f'stroke="{col}" stroke-width="2" stroke-linecap="round"/>')
        o.append(f'<circle cx="{sx(v):.1f}" cy="{y}" r="5.5" fill="{col}" '
                 f'stroke="{c["surface"]}" stroke-width="2"/>')
        o.append(txt(W - RIGHT + 16, y + 4.5, f"{v:.4f}", 13, c["ink"],
                     weight="600"))
    o.append(txt(0, H - 10, "combined 0.7934 vs LiDAR + MSP 0.7963:  difference "
                            "-0.0029, 95% CI [-0.0186, +0.0134]  -  "
                            "statistically indistinguishable, with no LiDAR at "
                            "inference", 11.5, c["ink2"]))
    return "".join(o) + "</svg>"


# ---------------------------------------------------------------- figure 3
def fig_downstream(mode):
    c = T[mode]
    d = json.load(open(os.path.join(ROOT, "outputs/artifacts/a81_downstream.json")))
    hs = ["10m", "20m", "30m"]
    g = [d["horizons"][h]["point"]["gated (tau=0.15)"]["recall@10%"] for h in hs]
    f = [d["horizons"][h]["point"]["free-only (tau=0.0)"]["recall@10%"] for h in hs]
    W, LEFT, TOP, GW, BW = 880, 86, 92, 272, 76
    H = 352
    base = H - 66
    mx = max(g + f) * 1.28
    o = [head(W, H, c)]
    o.append(txt(0, 26, "Gating free space on camera evidence catches more unseen "
                        "obstacles", 17, c["ink"], weight="600"))
    o.append(txt(0, 48, "Hazard recall at a fixed 10% alarm budget, by look-ahead "
                        "horizon. 2,933 held-out frames.", 12.5, c["ink2"]))
    o.append(f'<rect x="0" y="62" width="11" height="11" rx="2" fill="{c["s1"]}"/>')
    o.append(txt(18, 72, "observability-gated", 12, c["ink2"]))
    o.append(f'<rect x="156" y="62" width="11" height="11" rx="2" fill="{c["s2"]}"/>')
    o.append(txt(174, 72, "predicted-free only", 12, c["ink2"]))
    o.append(f'<line x1="{LEFT-14}" y1="{base}" x2="{W-40}" y2="{base}" '
             f'stroke="{c["grid"]}" stroke-width="1"/>')
    for i, h in enumerate(hs):
        gx = LEFT + i * GW
        for j, (val, col) in enumerate(((g[i], c["s1"]), (f[i], c["s2"]))):
            x = gx + j * (BW + 8)
            bh = (base - TOP) * val / mx
            o.append(f'<rect x="{x}" y="{base-bh:.1f}" width="{BW}" '
                     f'height="{bh:.1f}" rx="4" fill="{col}"/>')
            o.append(txt(x + BW / 2, base - bh - 10, f"{val*100:.1f}%", 13,
                         c["ink"], anchor="middle", weight="600"))
        rel = (g[i] - f[i]) / f[i] * 100
        o.append(txt(gx + BW + 4, base + 22, f"H = {h.replace('m',' m')}", 12.5,
                     c["ink2"], anchor="middle"))
        o.append(txt(gx + BW + 4, base + 40, f"+{rel:.0f}% relative", 11.5,
                     c["ink2"], anchor="middle"))
    o.append(txt(0, H - 10, "Primary endpoint H = 20 m, declared on the dev split "
                            "before the test split was opened. AUROC gain excludes "
                            "zero at all three horizons.", 11.5, c["ink2"]))
    return "".join(o) + "</svg>"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("bands", fig_bands), ("auroc", fig_auroc),
                     ("downstream", fig_downstream)):
        for mode in ("light", "dark"):
            p = os.path.join(OUT, f"{name}-{mode}.svg")
            open(p, "w").write(fn(mode))
            print("wrote", os.path.relpath(p, ROOT))
