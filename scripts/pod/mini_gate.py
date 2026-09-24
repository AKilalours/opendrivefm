"""Twenty frames, three minutes, four questions. Run BEFORE the full export.

The previous export produced 6,019 files that looked fine -- correct shape,
correct dtype, correct count, perfect token alignment -- and were useless,
because the values inside them described the road surface rather than the
scene. Nothing in "it produced files" implies "it produced correct files",
and the only defence is to state in advance what the numbers must look like
and check them while a rerun is still cheap.

So, the four gates, with thresholds fixed here rather than after seeing the
output:

  1. SIZE     bytes per frame, extrapolated to 6,019. The pod has ~15 GB
              free. Above ~9 GB total we split the run; above ~13 GB we drop
              conf2 and rerun this gate.

  2. CLASSES  among NON-FREE voxels, the fraction belonging to the ten
              dynamic classes. KEPT AS A WEAK SANITY FLOOR ONLY, at >= 0.2%.
              A41 audited this gate and found it could not do the job it was
              written for: the collapse it was meant to catch measured 1.2%
              and the threshold was 1.0%, so the broken export PASSED. Raising
              the threshold does not fix it either -- measured on 600 frames of
              the VALIDATED export, the per-frame dynamic share runs p5 = 0.15%
              to p100 = 17.8% with a median of 1.6%, so no single threshold can
              separate a good empty street from a broken run. The quantity has
              too much natural variance to gate on. Gate 6 replaces it.

  6. STRUCTURE  the gate that can actually catch a column collapse, written
              against the failure mode instead of against a class histogram
              that resembles one. A collapse keeps exactly ONE voxel per BEV
              column by construction, so it scores exactly 1.000 here and
              100% there. Measured on 400 frames of the validated export:
                  occupied voxels per occupied column   min 3.71, median 9.45
                  columns holding exactly one voxel     median 0.29%, max 15.5%
              Gates: mean >= 2.0 voxels per occupied column (a 1.85x margin
              below the worst good frame, and unreachable by a collapse), and
              <= 50% of occupied columns holding exactly one voxel.

  3. CONF     the max-softmax distribution. It must SPREAD. If the median is
              1.000 or >50% of voxels sit at the ceiling, confidence carries
              no information and no calibration study is possible. Gate:
              median <= 0.999 AND fraction at ceiling <= 0.50.

  4. FREE     fraction of voxels predicted free. Occ3D is mostly empty air;
              anything outside 0.55-0.95 means the volume is being read
              wrongly. Gate: 0.55 <= free <= 0.95.

A gate that fails is not a reason to proceed carefully. It is a reason to
stop, because every hour after this one is spent on top of the answer.
"""
import glob
import json
import os
import sys

import numpy as np

DYNAMIC = list(range(11))     # others..truck, i.e. everything that moves + barrier
FREE = 17
N_FRAMES_FULL = 6019


def main(d):
    fs = sorted(glob.glob(os.path.join(d, '*.npz')))
    if not fs:
        sys.exit(f'no npz in {d}')
    print(f'{len(fs)} frames in {d}\n')

    nbytes = np.mean([os.path.getsize(f) for f in fs])
    total = nbytes * N_FRAMES_FULL / 1e9
    cls, conf, conf2, pfree = [], [], [], []
    for f in fs:
        z = np.load(f)
        cls.append(z['cls'].ravel())
        conf.append(z['conf'].ravel())
        conf2.append(z['conf2'].ravel())
        pfree.append(z['p_free'].ravel())
    cls = np.concatenate(cls)
    conf = np.concatenate(conf).astype(np.float32) / 255.0
    conf2 = np.concatenate(conf2).astype(np.float32) / 255.0
    pfree = np.concatenate(pfree).astype(np.float32) / 255.0

    # A41: structural check. Loaded per frame because it needs the 3D shape,
    # which the flattened arrays above have already thrown away.
    vpc, one_col = [], []
    for f in fs:
        c = np.load(f)['cls']
        col = (c != FREE).sum(-1)
        nz = col[col > 0]
        if nz.size:
            vpc.append(float(nz.mean()))
            one_col.append(float((nz == 1).mean()))
    voxels_per_col = float(np.mean(vpc)) if vpc else 0.0
    single_col_frac = float(np.mean(one_col)) if one_col else 1.0

    free_frac = float((cls == FREE).mean())
    nonfree = cls[cls != FREE]
    dyn = float(np.isin(nonfree, DYNAMIC).mean()) if nonfree.size else 0.0
    q = [1, 5, 25, 50, 75, 95, 99]
    ceil = float((conf >= 0.998).mean())
    med = float(np.median(conf))

    print(f'1 SIZE     {nbytes/1e6:.2f} MB/frame -> {total:.1f} GB for 6,019')
    print(f'2 CLASSES  free {free_frac*100:.1f}%  |  dynamic among non-free '
          f'{dyn*100:.2f}%')
    print(f'3 CONF     ' + '  '.join(f'p{p}={v:.3f}' for p, v in
                                     zip(q, np.percentile(conf, q))))
    print(f'           median {med:.4f}   at ceiling {ceil*100:.1f}%')
    print(f'  margin   ' + '  '.join(f'p{p}={v:.3f}' for p, v in
                                     zip(q, np.percentile(conf - conf2, q))))
    print(f'6 STRUCT   {voxels_per_col:.3f} occupied voxels per occupied column  |  '
          f'{single_col_frac*100:.2f}% of columns hold exactly one')
    print(f'4 P_FREE   ' + '  '.join(f'p{p}={v:.3f}' for p, v in
                                     zip(q, np.percentile(pfree, q))))
    h = np.bincount(cls, minlength=18) / cls.size
    names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'constr_veh',
             'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
             'driveable', 'other_flat', 'sidewalk', 'terrain', 'manmade',
             'vegetation', 'FREE']
    print('\n  class shares')
    for i, (n, v) in enumerate(zip(names, h)):
        if v > 0:
            print(f'   {i:>2} {n:<13} {v*100:7.3f}%')

    print()
    gates = [
        ('SIZE fits 15 GB free', total <= 13.0, f'{total:.1f} GB'),
        ('dynamic >= 0.2% (weak floor)', dyn >= 0.002, f'{dyn*100:.2f}%'),
        ('>= 2.0 voxels per occupied column', voxels_per_col >= 2.0,
         f'{voxels_per_col:.3f}'),
        ('<= 50% single-voxel columns', single_col_frac <= 0.50,
         f'{single_col_frac*100:.2f}%'),
        ('conf median <= 0.999', med <= 0.999, f'{med:.4f}'),
        ('conf ceiling <= 50%', ceil <= 0.50, f'{ceil*100:.1f}%'),
        ('free in 0.55-0.95', 0.55 <= free_frac <= 0.95, f'{free_frac:.3f}'),
    ]
    ok = True
    for name, passed, val in gates:
        print(f'  [{"PASS" if passed else "FAIL"}] {name:<32} {val}')
        ok &= passed
    print('\n' + ('ALL GATES PASS -- proceed to the full run'
                  if ok else 'GATE FAILED -- stop, do not run 6,019 frames'))
    json.dump({'mb_per_frame': nbytes / 1e6, 'gb_total': total,
               'free_frac': free_frac, 'dynamic_of_nonfree': dyn,
               'conf_median': med, 'conf_ceiling': ceil,
               'voxels_per_occupied_column': voxels_per_col,
               'single_voxel_column_frac': single_col_frac, 'pass': bool(ok)},
              open(os.path.join(d, '_gate.json'), 'w'), indent=2)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/opt/preds_mini')
