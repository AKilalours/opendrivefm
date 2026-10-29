# Pod runbook: the second backbone

Everything in this file is written so that the pod day is execution, not
decision-making. Read it once before starting the pod; the meter runs from the
moment it boots.

Written 2026-09-24. Nothing here has been executed yet.

---

## Why this is the one remaining blocking experiment

Every measured claim in this project rests on **one checkpoint**: FB-OCC r50,
`do_history=True`, mIoU 38.90 against a published 39.1. The claim the paper
makes is about *camera-based occupancy models*, not about FB-OCC. A reviewer's
first objection is the cheapest one to pre-empt and the most expensive one to
leave open.

It is also the only item on the plan that cannot be done on a laptop. That is
what makes it blocking and everything else scheduling.

## What "done" looks like

One number, on the same sealed test split, with the same code path:

    H2 on backbone 2:  AUROC(observability) - AUROC(mask_camera)

**Pre-registered acceptance, stated before the run:** the interval excludes
zero and the point estimate is >= +0.02. FB-OCC gave +0.0534
[+0.0462, +0.0609]. A second model at, say, +0.03 is a *stronger* paper than
one model at +0.0534, because it converts a property of a checkpoint into a
property of the class of models.

A result below +0.02, or an interval spanning zero, is a finding and gets
written up as one. It is not a reason to try a third backbone until one agrees.

**Amend `METHOD_FREEZE.md` with this acceptance line before the pod boots**, so
the record shows it was set in advance.

## Which backbone

> **SUPERSEDED 2026-10-26 by METHOD_FREEZE A62. The backbone is FlashOcc, not
> SurroundOcc.** There is no public SurroundOcc checkpoint trained on
> Occ3D-nuScenes; the released weights are 17-class on nuScenes-Occupancy and
> live on Baidu Pan, and the "37.1 on Occ3D" figure is a re-trained number from
> other papers' tables. See A62 for the full reasoning. The section below is
> kept as written because the reasons it gives are the reasons FlashOcc was
> chosen to replace it.

### What to actually run

    M2  FlashOCC-4D-Stereo  R50  256x704  2f  mIoU 37.84   <- the headline
    M1  FlashOCC            R50  256x704  1f  mIoU 32.08   <- same env, run it too

    repo    https://github.com/Yzichen/FlashOCC
    cfg     projects/configs/flashocc/flashocc-r50-4d-stereo.py   (M2)
            projects/configs/flashocc/flashocc-r50.py             (M1)
    ckpts   Google Drive. File ids read from the README model zoo on
            2026-10-26, Occ3D-nuScenes table:

              M2  12WYaCdoZA8-A6_oh6vdLgOmqyEc3PNCe   mIoU 37.84
              M1  1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B   mIoU 32.08

            gdown 12WYaCdoZA8-A6_oh6vdLgOmqyEc3PNCe -O /opt/ckpts/flashocc-m2.pth
            gdown 1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B -O /opt/ckpts/flashocc-m1.pth

            TRAPS in that README. (1) The BEVDetOCC rows are Baidu Pan, not
            Drive -- skip them. (2) There is a SECOND table below the Occ3D one,
            on a different benchmark, whose "M1: FlashOCC (1f)" row reads
            mIoU 15.41 and has a different file id. Taking that checkpoint
            silently gives you the wrong model. Match on the mIoU, not the row
            label.

Native Occ3D-nuScenes, `num_classes=18`, free at index 17, head output
`(B, Dx, Dy, Dz, C)`. No class mapping, nothing downstream changes.

Running M1 as well as M2 costs one extra inference pass and gives H2 at 1, 2
and 16 frames. A62 registers both as reportable: **both or neither.**

### The original SurroundOcc reasoning, retained

**SurroundOcc first.** Reasons, in order:

1. It is the furthest from FB-OCC architecturally that is still camera-only:
   no temporal window at all, where FB-OCC fuses 16 frames. If the measure
   holds across "16 frames of memory" and "none", it is not a fact about
   temporal fusion.
2. Published mIoU 37.1 on Occ3D-nuScenes, close enough to FB-OCC's 39.1 that a
   difference in H2 cannot be waved away as "one model is just worse".
3. Its release is a plain mmdet3d inference path. COTR (46.2) is the stronger
   model but carries a heavier custom-op surface, and the FB-BEV experience in
   `scripts/pod/setup_fbocc_pod.sh` is that custom ops are where a pod day
   dies.

**COTR only if SurroundOcc installs in under an hour.** Two extra backbones is
a nice-to-have; one extra backbone is the deliverable.

## The environment

`scripts/pod/setup_fbocc_pod.sh` documents seven install walls and two upstream
defects. SurroundOcc is a 2023 mmdet3d stack of the same generation, so
**expect the same seven walls and start from that script**, changing only the
repo and the checkpoint. Do not start from a clean image and rediscover them.

Budget: ~25 min for the shared environment (already scripted), then the
SurroundOcc-specific fights. If the total passes **90 minutes**, stop, write
down which wall it died on, and reconsider COTR or a null result. A pod hour is
not free and an unbounded install is how a day disappears.

## Exact pod configuration

Derived from what `setup_fbocc_pod.sh` installs, not from preference.

    GPU               RTX A6000 (48 GB)
    Also fine         A40, A100 40/80GB, RTX A5000, RTX 3090
    NEVER             RTX 4090, L40, L40S, H100, H200, B200
    Cloud             Secure Cloud (Community Cloud cannot mount network volumes)
    Template          RunPod PyTorch 2.8.0
    Container disk    100 GB
    Network volume    the existing one, mounted at /workspace
    Region            must match the volume's region, or it will not attach

**Why the GPU list is a hard constraint.** The environment pins
`torch==1.13.1+cu117`. That wheel ships kernels for sm_37 through sm_86 and
nothing above it. A 4090 or L40S is sm_89; an H100 is sm_90. On those cards
`torch.cuda.is_available()` returns True -- so the assertion on line 53 of the
setup script **passes** -- and then the first real kernel launch dies with
"no kernel image is available for execution on the device", deep into the run,
long after the install budget is spent. Pick from the allowed list.

GPU memory is not the binding constraint: FlashOCC-4D-Stereo at R50 256x704
with `samples_per_gpu=1` fits comfortably in 24 GB. Architecture is.

The script's own tested configuration, for reference: RunPod PyTorch 2.8.0
image, RTX A6000, CUDA 12.8 host, driver 580. A newer host CUDA and driver are
fine -- drivers are backward compatible -- which is why wall 4 exists and
installs a CUDA 11.7 toolkit inside the environment.

**Write the exports to the network volume, not the container disk.** Two
backbones at ~6.7 GB each is ~13.4 GB, and `mini_gate.py` assumes roughly
15 GB free. Container disk is lost the moment the pod stops; a stopped pod
with the export still on it is the one failure that wastes the whole session.

## Data on the pod -- the cost this runbook never counted

**Checked 2026-10-29.** The Mac holds nuScenes **v1.0-mini only**: 705 MB of
`samples/`, 32 MB of mini metadata, 4.4 GB of sweeps. The sealed test split is
75 scenes of the nuScenes *validation* set, which is trainval data. So the
FB-OCC predictions in `data/preds` were produced with nuScenes present **on the
pod**, not shipped from here.

`setup_fbocc_pod.sh` does not handle this. It prints "put nuScenes under
/opt/FB-BEV/data/nuscenes" and stops. Its "~25 min, most of it downloading"
is the conda and mmdet3d stack, nothing else. **The nuScenes download is
unbudgeted and would dwarf every install wall in that script.**

### Check this BEFORE bringing up a GPU pod

Attach the network volume to a cheap CPU pod -- not the A6000 -- and run:

    du -sh /workspace/*
    ls /workspace/*/data/nuscenes/                      # want samples/ v1.0-trainval/ maps/
    find /workspace -name "bevdetv2-nuscenes_infos_val.pkl"

    volume has nuScenes + the val info pkl   -> the 2-3 GPU-hour, $3-6 plan holds
    volume has nuScenes, no pkl              -> create_data_bevdet.py rebuilds it, minutes
    volume gone                              -> STOP. Re-plan. This is not a 90-minute job.

A network volume is also what makes the pod restartable: container disk is lost
the moment the pod stops, so without one, every interruption re-downloads
everything.

## The run, in order

    # 0. before the pod
    #    amend METHOD_FREEZE with the acceptance line above. Commit.

    # 1. pod up: RTX A6000 or better, PyTorch 2.8.0 image, network volume if
    #    one is available -- container disk is lost on stop.
    bash scripts/pod/setup_fbocc_pod.sh          # shared env, ~25 min

    # 2. FlashOCC repo + checkpoint
    git clone --depth 1 https://github.com/Yzichen/FlashOCC /opt/FlashOCC
    pip install gdown
    gdown 12WYaCdoZA8-A6_oh6vdLgOmqyEc3PNCe -O /opt/ckpts/flashocc-m2.pth
    gdown 1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B -O /opt/ckpts/flashocc-m1.pth

    # 3. SMALL export first, then the gate on its output. ALWAYS this order:
    #    mini_gate.py reads the .npz files the export writes, so it cannot run
    #    before one exists. (Earlier revisions of this runbook listed the gate
    #    as step 2, which is impossible. Corrected 2026-10-29.)
    #
    #    --backbone flashocc is REQUIRED; the default is still fbocc so the
    #    existing FB-OCC path cannot be disturbed by accident.
    python scripts/pod/run_fbocc_voxel.py /opt/preds_mini --backbone flashocc \
        --cfg /opt/FlashOCC/projects/configs/flashocc/flashocc-r50-4d-stereo.py \
        --ckpt /opt/ckpts/flashocc-m2.pth --limit 50

    python scripts/pod/mini_gate.py /opt/preds_mini    # must pass before step 4

    #    sanity on those 50 before spending the rest:
    #      - class histogram has dynamic classes above ~1%.  The Stage-2
    #        column-collapse bug reported 1.2% for all ten dynamic classes
    #        combined and cost a day. Check this number, every time.
    #      - grid orientation: the BEV transform is a[::-1, ::-1]. Verify on
    #        one frame against the same frame from FB-OCC.
    #      - mIoU in the neighbourhood of the published 37.84 for M2
    #        (32.08 for M1). NOT 37.1 -- that was the SurroundOcc figure.
    #
    #    the export path for FlashOcc was written from upstream source and has
    #    never touched a GPU. If it throws, the seam is install_flashocc_prob_hook
    #    in run_fbocc_voxel.py: it overrides simple_test_occ to return
    #    outs.softmax(-1) instead of the upstream get_occ_gpu argmax.

    # 4. full split, then pull the predictions down and STOP the pod
    python scripts/pod/run_fbocc_voxel.py <out> --backbone flashocc --cfg ... --ckpt ...

    #    then M1, same environment, one more pass:
    python scripts/pod/run_fbocc_voxel.py <out_m1> --backbone flashocc \
        --cfg /opt/FlashOCC/projects/configs/flashocc/flashocc-r50.py \
        --ckpt /opt/ckpts/flashocc-m1.pth

    # 5. everything after this is CPU work on the laptop and costs nothing:
    #    the observability maps are already built -- they are a property of the
    #    SENSOR RIG, not of the model, so they are reused unchanged. Only the
    #    predictions change.
    python scripts/eval/<the H2 script> --preds <new export>

## Cost

Two to three GPU-hours on an A6000, roughly **$3-6**. The observability maps do
not need rebuilding, which is what keeps this cheap.

Nothing else on the plan needs a GPU before 16 November. Training runs of
$50-200 stay deferred until after the submission.

## Also needs the pod, and was mis-filed as laptop work

`scripts/run_robustness_real.py` has never been executed. Checked today: the
laptop has no torch, and `all_frames.npz` (the Stage-1 mini export it loads)
does not exist anywhere in the tree. So it is pod work, not an afternoon.

It is worth an hour **if the second backbone finishes early**, because it closes
a real integrity gap: every robustness number in this project was measured
against degradations this repository rendered itself, which is close to
circular. nuScenes mini has three genuine night scenes and one after-rain
scene that have never been evaluated. Do it after the backbone, never instead.

## What to bring back from the pod

    - the per-voxel export for the sealed split
    - the 50-frame sanity numbers, written down, pass or fail
    - the install diff against setup_fbocc_pod.sh, appended to that script
    - the wall-clock and the dollar figure

The last one matters: this project quotes its own costs, and a number that was
estimated rather than measured does not get quoted.
