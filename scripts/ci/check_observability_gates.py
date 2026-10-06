#!/usr/bin/env python3
"""Release gates for the observability line of work (A12-A47).

Same principle as check_gates.py: every eval in this repo writes JSON to
outputs/artifacts/, and gating those files means a regression cannot be
committed quietly. Between A20 and A28 that principle was not being honoured --
eight results were cited in METHOD_FREEZE while their artifacts sat gitignored,
so nothing could check them. This closes that.

Each gate states the amendment it enforces and the value it was frozen at.
Thresholds are deliberately slack against the measured value: they exist to
catch a pipeline that broke, not to pin a number to four decimals.

  --self-test  feeds each gate a deliberately broken artifact and requires it
               to FAIL. A gate that cannot fail is decoration.
"""
from __future__ import annotations
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "outputs", "artifacts")
FAILED: list[str] = []


def load(name):
    p = os.path.join(ART, name)
    if not os.path.exists(p):
        FAILED.append(f"{name}: MISSING. The gate cannot run on a file that was "
                      f"never committed.")
        return None
    return json.load(open(p))


def check(cond, label, detail):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}  --  {detail}")
    if not cond:
        FAILED.append(f"{label}: {detail}")
    return cond


def gate_h2(d):
    """A26: observability must beat mask_camera by a clear margin."""
    if d is None: return
    diff = d.get("diff", d.get("difference", d.get("delta")))
    lo = (d.get("ci") or d.get("difference_ci") or [None])[0]
    check(diff is not None and diff >= 0.03, "H2 margin",
          f"observability - mask_camera = {diff} (gate >= 0.030, frozen at 0.0534)")
    if lo is not None:
        check(lo > 0, "H2 interval excludes zero", f"CI lower bound {lo}")


def gate_surf_zero(d):
    """A20: the first-hit zeroing must be geometrically justified."""
    if d is None: return
    pct = d["summary"]["pct_unexplained"]
    check(pct < 1.0, "surf_zero unexplained",
          f"{pct:.3f}% of zeroed voxels have a clear line of sight "
          f"(gate < 1.0%, measured 0.061%)")


def gate_missed(d):
    """A22: object miss rate must fall as observability rises."""
    if d is None: return
    b = d["bands"]
    check(len(b) >= 4, "missed-detection bands", f"{len(b)} observability bands")
    first, last = b[0]["no_dyn"], b[-1]["no_dyn"]
    check(first > 0.30, "blind-set miss rate",
          f"{100*first:.1f}% of objects get nothing dynamic placed when obs = 0 "
          f"(gate > 30%, frozen 48.3%)")
    check(last < 0.15, "well-seen miss rate",
          f"{100*last:.1f}% at the best-observed band (gate < 15%, frozen 5.3%)")
    check(first / max(last, 1e-9) >= 4.0, "miss-rate spread",
          f"{first/max(last,1e-9):.1f}x blind vs well seen (gate >= 4x, frozen 9.2x)")


def gate_dropout(d):
    """A21: the six-camera exposure study must cover six cameras."""
    if d is None: return
    pc = d["per_camera"]
    check(len(pc) == 6, "camera coverage", f"{len(pc)} cameras reported")
    worst = max(v["pct_obst_dark"] for v in pc.values())
    check(worst > 15.0, "worst-camera exposure",
          f"{worst:.1f}% of obstacles lose all evidence on the worst single "
          f"failure (gate > 15%, frozen 30.2%)")


def gate_formula(d):
    """A23: max must still beat the noisy-OR, and TRUST must stay inert."""
    if d is None: return
    c = d["contrasts"]
    mf = c["max_cam_minus_full"]
    check(mf["ci"][0] > 0, "max beats noisy-OR",
          f"delta {mf['delta']:.4f} CI {mf['ci']} (lower bound must exceed 0)")
    nt = c["max_noT_minus_max_cam"]
    # A23 first claimed this was EXACTLY 0.0000. It is 5e-6. Under a max, TRUST
    # is a monotone rescale and cannot change a ranking -- but the AUROC is
    # computed on a 256-bin uint8 histogram, and rescaling moves a few values
    # across bin edges. The residual is a quantisation artefact of the measure,
    # not an effect. The gate is set an order of magnitude above it.
    check(abs(nt["delta"]) < 1e-4, "TRUST is inert under max",
          f"delta {nt['delta']:.2e} (gate < 1e-4; residual is uint8 quantisation, "
          f"not a real effect)")


def gate_mining(d):
    """A28: at least one UNLABELLED signal must find the bad frames."""
    if d is None: return
    for target in ("WRONG", "OVERCONFIDENT"):
        t = d["targets"].get(target, {})
        if not t:
            FAILED.append(f"mining target {target} missing"); continue
        best = max(t.values(), key=lambda v: v["auroc"])
        name = max(t, key=lambda k: t[k]["auroc"])
        check(best["auroc"] >= 0.72, f"mining AUROC [{target}]",
              f"best signal '{name}' at {best['auroc']:.4f} (gate >= 0.72)")
        check(best["lift"] >= 2.0, f"mining lift [{target}]",
              f"{best['lift']:.2f}x over random at precision@100 (gate >= 2.0x)")



# ---------------------------------------------------------------------------
# A47. Everything from A29 onward was ungated, including the two results that
# reversed earlier conclusions. Twelve of eighteen published numbers had no
# check at all. These close that.
# ---------------------------------------------------------------------------

def gate_envelope(d):
    """A30/A38: the safety envelope. The flagged rate must not drift down
    quietly -- A38 raised it and the direction of that correction is the whole
    point of the result."""
    if d is None: return
    f = d["flagged_frac"]
    check(0.02 <= f <= 0.08, "safety envelope flagged",
          f"{100*f:.2f}% of frames (gate 2-8%, frozen 4.3%)")
    check(d["reach_median"] >= 15.0, "verified-free reach",
          f"median {d['reach_median']:.1f} m (gate >= 15, frozen 26.4)")


def gate_temporal(d):
    """A41: memory must fill the corridor and must NOT rescue the envelope.
    A run where 4 s of memory suddenly halves the flagged rate means the warp
    is leaking future evidence."""
    if d is None: return
    h = {r["seconds"]: r for r in d["horizons"]}
    a, b = h.get(0.0), h.get(4.0)
    if not (a and b): return
    check(b["corridor_evidence"] - a["corridor_evidence"] > 0.05,
          "memory adds evidence",
          f"{100*a['corridor_evidence']:.1f}% -> {100*b['corridor_evidence']:.1f}% "
          f"of corridor columns (gate > 5 pts, frozen 77.8 -> 94.0)")
    drop = a["flagged_above_10"] - b["flagged_above_10"]
    check(drop < 0.10, "memory does not rescue the envelope",
          f"flagged above 10 m/s falls {100*drop:.1f} pts (gate < 10, frozen 2.8)")


def gate_selective(d):
    """A37/A40/A47: the sensor-only contrast, the number the paper leans on
    hardest in the no-confidence regime."""
    if d is None: return
    m = d.get("boot_margin_sensor_mask_minus_obs")
    check(m is not None, "sensor-only contrast is STORED",
          "printing is not storing -- this is the A40 bug" if m is None else "present")
    if m:
        check(m[1] > 0, "sensor-only interval excludes zero",
              f"{m[0]:+.5f} [{m[1]:+.5f}, {m[2]:+.5f}] (frozen +0.0386 on the A45 split)")


def gate_recal(d):
    """A41: observability must still add over class+confidence inside the mask.
    This is the result that overturned A34, so it is the one most worth
    watching."""
    if d is None: return
    c = d["contrasts"]["observability over class+confidence, INSIDE the mask"]
    check(c[1] > 0, "recalibration gain inside the mask",
          f"{c[0]:+.6f} [{c[1]:+.6f}, {c[2]:+.6f}] (frozen +0.00304)")


def gate_h4(d):
    """A44: temporal observability at the PRE-NAMED half-life, not the best
    row of the sweep."""
    if d is None: return
    pr = str(d.get("primary_half_life", 2.0))
    r = d.get(pr) or d.get("2.0")
    check(r and r["lo"] > 0, "H4 at the pre-named half-life",
          f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] (frozen +0.0156)")


def gate_baselines(d):
    """A46: the finding that reframed the paper. Two things must stay true:
    confidence beats us per voxel (so nobody quietly re-promotes H2), and we
    beat confidence per object (the actual contribution)."""
    if d is None: return
    a = d["auroc"]
    check(a["MSP"] > a["observability"], "A46 voxel ordering is still honest",
          f"MSP {a['MSP']:.4f} vs observability {a['observability']:.4f} -- if this "
          f"ever flips, check the scoring before celebrating")
    check(a["COMBINED"] >= a["MSP"], "combined is not worse than MSP",
          f"{a['COMBINED']:.4f} vs {a['MSP']:.4f}")


def gate_baselines_obj(d):
    """A46: the object-level claim, which is now the paper's headline."""
    if d is None: return
    c = d["contrasts"]
    m = c["observability - MSP"]
    check(m[1] > 0, "object level: observability beats model confidence",
          f"{m[0]:+.4f} [{m[1]:+.4f}, {m[2]:+.4f}] (frozen +0.0430)")
    a = d["auroc"]
    check(a["observability"] >= 0.70, "object-level AUROC",
          f"{a['observability']:.4f} (gate >= 0.70, frozen 0.7623)")


def gate_placement(d):
    """A32/A45: the seventh-camera negative claim. A gate on a NEGATIVE result
    guards the opposite direction -- it fails if a candidate suddenly recovers
    a lot, which would mean the rig model changed."""
    if d is None: return
    best = max(c["obstacles_recovered_frac"] for c in d["candidates"].values())
    check(best <= 0.06, "seventh camera recovers little",
          f"best candidate {100*best:.1f}% of unseen obstacles "
          f"(gate <= 6%, frozen 2.2%)")
    check(d["frames"] >= 800, "placement sample size",
          f"{d['frames']} frames (gate >= 800, A45 reran at 1,238)")


def gate_split(_d):
    """A45: the split file must be present and its digest must match the
    constant in split.py. If the split moves, every held-out number is void."""
    sys.path.insert(0, os.path.join(ROOT, "scripts", "eval"))
    try:
        import split as SP
        d_ = SP.digest()
        check(d_ == SP.DIGEST, "split digest unchanged",
              f"{d_} (expected {SP.DIGEST})")
        check(len(SP.dev_scenes()) == 75 and len(SP.test_scenes()) == 75,
              "split is 75 dev / 75 test",
              f"{len(SP.dev_scenes())} / {len(SP.test_scenes())}")
    except Exception as e:                                   # noqa: BLE001
        check(False, "split loads", str(e))


GATES = [
    ("h2_max.json", gate_h2),
    ("surf_zero_verdict.json", gate_surf_zero),
    ("missed_detection.json", gate_missed),
    ("camera_dropout.json", gate_dropout),
    ("formula_decision.json", gate_formula),
    ("mining_validation.json", gate_mining),
    ("safety_envelope.json", gate_envelope),
    ("corridor_temporal.json", gate_temporal),
    ("selective_max_nonfree.json", gate_selective),
    ("recal_confound.json", gate_recal),
    ("temporal_observability.json", gate_h4),
    ("baselines.json", gate_baselines),
    ("baselines_object.json", gate_baselines_obj),
    ("camera_placement.json", gate_placement),
    ("val_dev_test.json", gate_split),
]


def self_test():
    """Every gate must reject a broken artifact. Otherwise it is decoration."""
    import copy
    print("SELF-TEST: each gate is fed a broken artifact and must FAIL\n")
    breaks = {
        "h2_max.json": lambda d: {**d, "diff": 0.001,
                                   "ci": [-0.01, 0.01]},
        "surf_zero_verdict.json": lambda d: {**d, "summary": {**d["summary"],
                                                              "pct_unexplained": 40.0}},
        "missed_detection.json": lambda d: {**d, "bands": [
            {**b, "no_dyn": 0.05} for b in d["bands"]]},
        "camera_dropout.json": lambda d: {**d, "per_camera": {
            k: {**v, "pct_obst_dark": 1.0} for k, v in d["per_camera"].items()}},
        "formula_decision.json": lambda d: {**d, "contrasts": {
            **d["contrasts"],
            "max_cam_minus_full": {"delta": -0.01, "ci": [-0.02, -0.001]},
            "max_noT_minus_max_cam": {"delta": 0.02, "ci": [0.01, 0.03]}}},
        "mining_validation.json": lambda d: {**d, "targets": {
            k: {kk: {**vv, "auroc": 0.51, "lift": 1.0} for kk, vv in v.items()}
            for k, v in d["targets"].items()}},
        "safety_envelope.json": lambda d: {**d, "flagged_frac": 0.001,
                                           "reach_median": 2.0},
        "corridor_temporal.json": lambda d: {**d, "horizons": [
            {**r, "corridor_evidence": 0.80,
             "flagged_above_10": 0.24 if r["seconds"] == 0.0 else 0.02}
            for r in d["horizons"]]},
        "selective_max_nonfree.json": lambda d: {
            k: v for k, v in d.items()
            if k != "boot_margin_sensor_mask_minus_obs"},
        "recal_confound.json": lambda d: {**d, "contrasts": {
            **d["contrasts"],
            "observability over class+confidence, INSIDE the mask":
                [-0.001, -0.003, 0.001]}},
        "temporal_observability.json": lambda d: {
            **d, "2.0": {**d["2.0"], "delta": -0.01, "lo": -0.02, "hi": 0.001}},
        "baselines.json": lambda d: {**d, "auroc": {
            **d["auroc"], "MSP": 0.10, "COMBINED": 0.05}},
        "baselines_object.json": lambda d: {**d,
            "contrasts": {**d["contrasts"],
                          "observability - MSP": [-0.02, -0.05, -0.001]},
            "auroc": {**d["auroc"], "observability": 0.50}},
        "camera_placement.json": lambda d: {**d, "frames": 10, "candidates": {
            k: {**v, "obstacles_recovered_frac": 0.5}
            for k, v in d["candidates"].items()}},
    }
    ok = True
    for name, fn in GATES:
        if name == "val_dev_test.json":
            continue          # guards a file identity, not a measured value
        real = load(name)
        if real is None:
            ok = False; continue
        FAILED.clear()
        fn(breaks[name](copy.deepcopy(real)))
        if FAILED:
            print(f"  [OK]   {name}: gate rejected the broken version")
        else:
            print(f"  [BAD]  {name}: gate ACCEPTED a broken artifact")
            ok = False
    FAILED.clear()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    print("Observability release gates (A12-A28)\n")
    for name, fn in GATES:
        print(f"{name}")
        fn(load(name))
        print()
    if FAILED:
        print(f"{len(FAILED)} gate(s) failed:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("All observability gates pass.")


if __name__ == "__main__":
    main()
