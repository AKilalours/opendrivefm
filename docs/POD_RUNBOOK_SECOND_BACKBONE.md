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

## The run, in order

    # 0. before the pod
    #    amend METHOD_FREEZE with the acceptance line above. Commit.

    # 1. pod up: RTX A6000 or better, PyTorch 2.8.0 image, network volume if
    #    one is available -- container disk is lost on stop.
    bash scripts/pod/setup_fbocc_pod.sh          # shared env, ~25 min

    # 2. SurroundOcc repo + checkpoint, then the smoke gate BEFORE the full run
    python scripts/pod/mini_gate.py              # must pass; it exists for this

    # 3. export per-voxel predictions for the sealed split
    python scripts/pod/run_fbocc_voxel.py <out> --cfg <surroundocc cfg> \
        --ckpt <surroundocc ckpt> --limit 50     # 50 frames first, ALWAYS

    #    sanity on those 50 before spending the rest:
    #      - class histogram has dynamic classes above ~1%.  The Stage-2
    #        column-collapse bug reported 1.2% for all ten dynamic classes
    #        combined and cost a day. Check this number, every time.
    #      - grid orientation: the BEV transform is a[::-1, ::-1]. Verify on
    #        one frame against the same frame from FB-OCC.
    #      - mIoU in the neighbourhood of the published 37.1.

    # 4. full split, then pull the predictions down and STOP the pod
    python scripts/pod/run_fbocc_voxel.py <out> --cfg ... --ckpt ...

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
