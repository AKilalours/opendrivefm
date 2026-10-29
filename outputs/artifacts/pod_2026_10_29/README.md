# Pod session, 2026-10-29

## flashocc_m2_gate50.json

`scripts/pod/mini_gate.py` on the 50-frame FlashOcc M2 smoke export, verbatim
as the pod wrote it. Not edited.

**`"pass": false` is from the OLD threshold and is superseded.** The run
predates A66. Five of six gates passed; the sixth was `free in 0.55-0.95`
against a measured 0.394. A66 then showed that window would have rejected
85.5% of the validated FB-OCC export's own frames, and recalibrated it to
`0.15 <= free <= 0.85` using FB-OCC data only. Under the corrected gate all
six pass.

The file is kept with the failing verdict rather than regenerated, because
the measured quantities are what matter and rewriting a recorded verdict to
agree with a later decision is the kind of tidying this project does not do.

Values that matter, none of them changed by A66:

    voxels per occupied column   9.696   gate >= 2.0, the collapse detector
    single-voxel columns         0.18%   gate <= 50%
    conf median                  0.8471  gate <= 0.999
    conf ceiling                 16.7%   gate <= 50%
    dynamic of non-free          1.36%   gate >= 0.2%
    free fraction                0.394   FB-OCC median is 0.411

## Session facts

    GPU            RTX A40, compute capability 8.6, RunPod Secure Cloud
    image          RunPod PyTorch 2.8.0, Ubuntu 24.04, CUDA 12.8 host
    storage        container disk only, no network volume
    nuScenes       32 GB, camera keyframes only, 28 min (A65)
    M2 export      6019 frames, 0.6917 s/frame, ~69 min
    M1 export      6019 frames, 0.4087 s/frame, ~41 min
    environment    scripts/pod/setup_flashocc_pod.sh + flashocc_infer_setup.sh

Wall-clock and the measured dollar figure: TO BE FILLED from RunPod billing.
The runbook requires a measured cost, not an estimated one.
