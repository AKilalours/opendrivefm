#!/usr/bin/env python3
"""Dump one real keyframe's integrity inputs and the Python reference maps, so
cpp/tests/test_integrity_monitor.cpp can check its port against them.

Nothing synthetic: the calibrations, the occupancy grid and the reference
coverage/integrity all come from the same code path the console renders.

  float64 header : n(int32) rng res full_px trust
  6 x camera     : quat[4] trans[3] K[9] width height     (float64)
  uint8  [n*n]   : occupancy mask at res
  float32[n*n]   : reference integrity map
  6 x float32[n*n]: reference coverage * visibility, in T.CAMERAS order
"""
from __future__ import annotations
import struct, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import odfm_geom as GE, odfm_ground as G, odfm_lidar as L, odfm_tables as T

RNG, RES_I, TRUST, FULL_PX = 54.0, 0.50, 0.795, 140.0
OUT = Path(__file__).resolve().parents[1] / "cpp/integrity_case.bin"


def main():
    tab, man = T.tables(), L.load_manifest()
    tok = next(t for toks in tab.scene_samples.values() for t in toks[6:-4] if t in man)
    pts = L.load_sweep_stack(tok, man, n_sweeps=10)
    plane = G.fit_ground_plane(pts[:, :3])
    ground = G.label_ground(pts[:, :3], plane)
    prob, _, _ = G.occupancy_logodds(pts, ground, plane, rng_m=RNG, res=RES_I)
    occ = (prob > 0.65)

    covs, recs = [], []
    for cam in T.CAMERAS:
        rec = tab.sensor_record(tok, cam)
        c, _ = GE.camera_ground_coverage(rec, rng_m=RNG, res=RES_I, soft=True,
                                         full_px=FULL_PX)
        v = G.visibility_from(occ, rec["cal_trans"][:2], rng_m=RNG, res=RES_I)
        covs.append((c * v).astype(np.float32))
        recs.append(rec)
    integ = GE.integrity_map(covs, [TRUST] * 6).astype(np.float32)
    n = integ.shape[0]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<i", n))
        f.write(struct.pack("<4d", RNG, RES_I, FULL_PX, TRUST))
        for rec in recs:
            f.write(np.asarray(rec["cal_rot"], np.float64).tobytes())
            f.write(np.asarray(rec["cal_trans"], np.float64).tobytes())
            f.write(np.asarray(rec["intrinsic"], np.float64).ravel().tobytes())
            f.write(struct.pack("<2d", float(rec["width"]), float(rec["height"])))
        f.write(occ.astype(np.uint8).tobytes())
        f.write(integ.tobytes())
        for c in covs:
            f.write(c.tobytes())
    print(f"wrote {OUT}  n={n}  {OUT.stat().st_size/1e6:.1f} MB  "
          f"mean integrity {integ.mean():.6f}  scene {tab.scene_name(tok)}")


if __name__ == "__main__":
    main()
