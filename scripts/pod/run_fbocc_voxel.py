"""Cache FB-OCC predictions per VOXEL. Replaces the column-collapse export.

Why this exists
---------------
The first export collapsed each BEV column to a single value by taking the
level with the lowest P(free) and reporting that level's class and
confidence. Two things went wrong and both were mine.

  1. p_occ = 1 - min_z P(free) saturates. A column gets sixteen chances to
     contain one low-P(free) voxel, so 59% of columns pinned at exactly 1.0
     and the value carried no information at all.

  2. Worse, the level with the lowest P(free) is almost always the road
     surface, because the road is the thing the model is surest about. So the
     collapse reported the GROUND and discarded whatever was standing on it.
     Measured over 41 frames: 40.5% driveable_surface, 20.8% manmade, 18.5%
     terrain, 14.1% sidewalk -- and 1.2% for all ten dynamic classes
     combined, in a dataset where cars are everywhere. The obstacles were
     thrown away.

The fix is to stop collapsing. Occupancy is a volume; a calibration study of
a volumetric model has to be done on the volume. This writes every voxel.

What is written, per keyframe, on the Occ3D grid (200 x 200 x 16, 0.4 m):

  cls     uint8   argmax over all 18 channels, free (17) included as a
                  legitimate prediction rather than a special case
  conf    uint8   that argmax's softmax probability, x255. This is the
                  standard confidence for ECE: max-softmax against whether
                  the argmax was right.
  conf2   uint8   the runner-up probability, x255. conf - conf2 is the margin,
                  the other standard uncertainty measure. It costs 0.64 MB a
                  frame and it is the difference between never re-running this
                  and re-running it a third time.
  p_free  uint8   P(free) itself, x255, kept separately so binary occupancy
                  can be scored without reconstructing it from cls.

2.56 MB raw per frame, roughly 15 GB raw over the validation split, and
savez_compressed takes a large bite out of that because cls is extremely
redundant. Quantising to uint8 costs 1/255 ~ 0.004 of resolution, which is an
order of magnitude finer than any calibration bin.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

FREE = 17
CFG = '/opt/FB-BEV/occupancy_configs/fb_occ/fbocc_infer.py'
CKPT = '/opt/ckpts/fbocc-r50.pth'


def q8(a):
    """Probability -> uint8. Explicit rounding, explicit clip."""
    return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cfg', default=CFG)
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = Config.fromfile(args.cfg)
    cfg.model.train_cfg = None
    ds = build_dataset(cfg.data.test)
    dl = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=2,
                          dist=False, shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.ckpt, map_location='cpu')
    model = MMDataParallel(model.cuda().eval(), device_ids=[0])

    n = len(ds) if not args.limit else min(args.limit, len(ds))
    t0 = time.time()
    done = 0
    for i, data in enumerate(dl):
        if i >= n:
            break
        tok = ds.data_infos[i]['token']
        dst = os.path.join(args.out, tok + '.npz')
        if os.path.exists(dst):
            done += 1
            continue
        with torch.no_grad():
            p = model(return_loss=False, rescale=True,
                      return_raw_occ=True, **data)[0]['pred_occupancy']
        p = np.asarray(p, np.float32)            # (200, 200, 16, 18), softmaxed
        assert p.shape[-1] == 18, p.shape

        order = np.argpartition(-p, 1, axis=-1)  # top-2 without a full sort
        cls = order[..., 0].astype(np.uint8)
        conf = np.take_along_axis(p, order[..., 0:1], -1)[..., 0]
        conf2 = np.take_along_axis(p, order[..., 1:2], -1)[..., 0]

        np.savez_compressed(dst, cls=cls, conf=q8(conf), conf2=q8(conf2),
                            p_free=q8(p[..., FREE]))
        done += 1
        if done % 200 == 0:
            el = time.time() - t0
            print(f'{done}/{n}  {el/max(done,1):.3f}s/frame  '
                  f'eta {(n-done)*el/max(done,1)/60:.1f} min', flush=True)

    meta = dict(frames=done, grid=[200, 200, 16], voxel=0.4,
                z_range=[-1.0, 5.4], classes=18, free_channel=FREE,
                quantisation='uint8 = round(255 * p)',
                fields=['cls', 'conf', 'conf2', 'p_free'],
                sec_per_frame=round((time.time() - t0) / max(done, 1), 4),
                ckpt=os.path.basename(args.ckpt))
    json.dump(meta, open(os.path.join(args.out, '_meta.json'), 'w'), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
