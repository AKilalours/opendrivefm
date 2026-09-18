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
              dynamic classes. The old collapse gave 1.2% of columns and that
              is what exposed it. Per voxel the honest expectation is a few
              percent -- cars are small next to road and buildings -- but it
              must not be ~0. Gate: >= 1.0% of non-free voxels are dynamic.

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
        ('dynamic >= 1.0% of non-free', dyn >= 0.010, f'{dyn*100:.2f}%'),
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
               'conf_median': med, 'conf_ceiling': ceil, 'pass': bool(ok)},
              open(os.path.join(d, '_gate.json'), 'w'), indent=2)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/opt/preds_mini')
