#!/usr/bin/env python3
"""Release gates for the observability line of work (A12-A28).

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


GATES = [
    ("h2_max.json", gate_h2),
    ("surf_zero_verdict.json", gate_surf_zero),
    ("missed_detection.json", gate_missed),
    ("camera_dropout.json", gate_dropout),
    ("formula_decision.json", gate_formula),
    ("mining_validation.json", gate_mining),
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
    }
    ok = True
    for name, fn in GATES:
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
