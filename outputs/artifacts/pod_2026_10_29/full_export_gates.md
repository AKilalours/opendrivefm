# Gate results on the full FlashOcc exports, 2026-10-29

`scripts/pod/mini_gate.py --frames 400` on each complete 6,019-frame export.
The runbook asks for a 50-frame smoke gate; this is 400 random frames of the
shipped export, seed 0, which is strictly stronger.

An unsampled run on all 6,019 frames was attempted first and was OOM-killed on
both exports. The gate concatenates every voxel of every frame -- 3.85 billion
values, roughly 46 GB across three float32 copies. A guard and a `--frames`
flag were added so the next person gets an explanation instead of a kill.

                                   M1 (1 frame)   M2 (2 frames)   gate
    voxels per occupied column          9.497          9.469      >= 2.0
    single-voxel columns                0.72%          1.04%      <= 50%
    conf median                        0.8667         0.8902      <= 0.999
    conf at ceiling                     12.6%          15.8%      <= 50%
    dynamic among non-free              1.85%          1.79%      >= 0.2%
    free fraction                       0.409          0.411      0.15-0.85
    size per frame                   1.26 MB        1.21 MB

    M1  ALL GATES PASS
    M2  ALL GATES PASS

## On the free fraction

0.409 and 0.411 sit essentially on FB-OCC's median of 0.411 (A66, measured on
400 random frames of the validated export: p1 0.211, median 0.411, p99 0.706).

The original window of 0.55-0.95 would have failed **both** of these exports,
as it failed 85.5% of FB-OCC's own frames. A66 recalibrated it to 0.15-0.85
using FB-OCC data alone, before these two runs existed. They are an
independent check on that decision and they land in the middle of the
corrected window.

## On the structural gate

9.497 and 9.469 voxels per occupied column, against a threshold of 2.0 and
FB-OCC's measured median of 9.45. This is the gate A41 wrote after the class
histogram failed to catch a column collapse. Both exports clear it by nearly
five times, and match the reference backbone closely.
