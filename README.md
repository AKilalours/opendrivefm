# SightOcc

**Geometric observability for missed objects in camera-only 3D occupancy.**

A camera-only occupancy model can assign maximal confidence to free space exactly
where the rig supplies almost no geometric evidence, because a missed object and
correctly observed empty road are both, to the model, simply *free*. SightOcc is a
learning-free measure of how much evidence the cameras actually have at a point in
space, and this repository is the harness that measures whether it helps.

Every number below is produced by a script in this tree and written to a JSON
artifact under `outputs/artifacts/`. Each row names the file it came from, and the
release gates in `.github/workflows/validation.yml` check those artifacts on every
push, so a regression cannot land quietly.

> **Method freeze.** `docs/METHOD_FREEZE.md` is a 4,300-line amendment log, A1 to
> A62. Every hypothesis and its acceptance threshold are fixed in it *before* the
> run that tests them. The test split is sealed under digest `924a44db544d4818` and
> was opened once. Three pre-registered hypotheses failed; they are reported below
> with the same weight as the ones that passed.

> **Earlier project.** This repository previously hosted a trust-aware BEV
> perception project whose numbers did not reproduce. That work, its README and the
> record of each withdrawn claim are in `archive/`. Nothing from it is reused here.

---

## The measure

    O(v) = max_i  c_i(v) · s_i(v)

      c_i(v)   projected coverage     min(1, (f_i·r / d)² / π_full)
      s_i(v)   first-hit visibility   per-pixel ray marching, no bins, no depth bias

      one parameter   π_full = 965.0 px        learned components   none

Coverage is how much image area a voxel-sized object would occupy in camera *i*;
visibility is whether the ray reaches it first. The maximum over the rig is the
strength of the best geometric evidence any camera supplies for that point. Unlike
a binary camera-observation mask it carries evidence **strength**, and it resolves
occlusion explicitly.

---

## Verified results

### The headline: the answer inverts with granularity

| Question | Model confidence | Geometric observability | Winner |
|---|---|---|---|
| Is *this voxel* wrong? | **0.8797** | 0.6321 | confidence |
| Is *this object* missed entirely? | 0.7193 | **0.7623** | geometry |
| Is *this frame* degraded? | **0.8618** | 0.8066 | confidence |

This is the finding, not a tie-break. Confidence wins on the question it is built
for and loses on the one it structurally cannot answer: a missed object is a
confident prediction of free space, and no statement about a prediction that *was
made* can flag one that was never made at all. External geometry has no such blind
spot, because it never consults the model.

### Object level (`outputs/artifacts/baselines_object.json`)

40,863 objects, 75 held-out test scenes, split digest `924a44db544d4818`.

| Signal | AUROC |
|---|---|
| LiDAR points in box (oracle) | 0.7745 |
| **Geometric observability** | **0.7623** |
| MSP, max confidence in box | 0.7193 |
| Range from ego | 0.6660 |
| Mean confidence in box | 0.5409 |
| Mean top-2 margin | 0.5202 |

    observability − MSP       +0.0432   95% CI [+0.0135, +0.0708]   excludes zero
    observability − oracle    −0.0121   95% CI [−0.0289, +0.0052]   spans zero

The second line is the one worth reading: observability comes within noise of a
LiDAR point-count oracle while using no LiDAR at all.

### Voxel level (`outputs/artifacts/baselines.json`)

Same split. Predicting per-voxel error inside `mask_camera`.

| Signal | AUROC |
|---|---|
| Confidence + observability combined | 0.8811 |
| MSP | 0.8797 |
| Top-2 margin | 0.8788 |
| `p_free` | 0.7913 |
| Geometric observability | 0.6321 |
| Range from ego | 0.6124 |
| `mask_camera` | 0.5000 |

Combining the two beats MSP alone by +0.0015 [+0.0009, +0.0022]. The interval
excludes zero and the effect is negligible. Both facts are reported.

### H2: against the dataset's own flag (`outputs/artifacts/h2_max_test.json`)

The obvious objection is that Occ3D already ships `mask_camera`. Ranking voxels by
observability beats ranking them by that flag:

    observability   0.6714
    mask_camera     0.6230
    difference     +0.0484   95% CI [+0.0390, +0.0584]      3,008 frames, 75 scenes

Pre-registered acceptance was ≥ +0.03. **Replication:** an independent z-buffer
implementation sharing no geometry code with the ray marcher gives +0.0521 against
the marcher's +0.0527 on the development split — 0.0006 apart. The signal is in the
measure, not in either implementation's bugs.

### Miss rate by observability band (`outputs/artifacts/missed_detection_test.json`)

| Band | Objects | No dynamic voxel predicted |
|---|---|---|
| `O = 0`, no camera evidence | 8,010 | **47.3%** |
| 0.000 – 0.212 | 6,368 | 25.3% |
| 0.212 – 0.384 | 6,771 | 15.5% |
| 0.384 – 0.776 | 6,506 | 11.6% |
| 0.776 – 1.000 | 2,104 | 8.5% |
| `O = 1`, fully resolved | 11,104 | **4.7%** |

A tenfold spread, monotone across every band.

### What is actually blocking the cameras (`outputs/artifacts/blind_attribution.json`)

400 frames, every blind voxel attributed to its occluder.

| Occluder | By occluded volume | By occluded road area |
|---|---|---|
| Buildings and structures | 43.9% | 14.3% |
| Vegetation | 27.9% | 5.4% |
| Vehicles | 15.7% | **45.4%** |

The ordering inverts with the measure. Volume is dominated by buildings; road
surface — where anything you could hit is sitting — is dominated by vehicles, by a
factor of three. Road figures exclude ground-class self-occlusion, which is a
ray-sampling artefact and is reported separately in the artifact.

### Cost

    direct ray marcher         1.139 s / frame
    precomputed index table    0.329 s / frame      3.46× faster
    output difference          0 of 30,720,000 voxels

Extrinsics and intrinsics are constant within a scene and the validation split
contains only two distinct rigs, so the ray-to-voxel table is built twice, in 6.9 s,
held in 832 MB, and reused across 6,019 frames. Full build: 114 min → 33 min.

---

## What failed

Reported because a pre-registration that only publishes its successes is not one.

**H1 — calibration gap is not monotone in observability.** Deciles run .030 .039
.051 .052 .051 .044 .041 .040 .027: a hump, not the predicted decrease.

**S2 — the camera mask is marginally better at selective risk** than observability
alone, despite losing on ranking.

**A55 — rebuilding the measure from predicted geometry loses the object-level
advantage.** The mechanism is not the obvious one. Observability at *missed* objects
barely moves; observability at *detected* objects collapses, because predicted
occupancy is an order of magnitude denser than the ground-truth volume and shadows
regions the rig genuinely sees. This defines where the measure applies, which is why
it is measured rather than mentioned in passing.

---

## Scope

SightOcc reads external scene geometry, so it cannot run on a vehicle that does not
already have that geometry. It is positioned as an **evaluation-time diagnostic**:
fleet-scale triage for finding where a camera-only stack is structurally blind,
before deployment rather than during it. A55 is the honest boundary of that claim.

One consequence transfers directly to hardware. At the front camera's 1266.4 px
focal length, geometric evidence saturates at 16.31 m. A loaded Class 8 tractor at
65 mph needs roughly 160 m to stop.

---

## Layout

    paper/            the CVPR submission: main.tex, cvpr.sty, figures
    src/opendrivefm/  the library
    scripts/          evaluation, mining, visualisation, pod runners
    outputs/artifacts/ every measured JSON this README cites
    docs/             METHOD_FREEZE.md (the amendment log) and the runbooks
    cpp/              C++ runtime primitives, gated under ThreadSanitizer in CI
    archive/          the retired project: its paper, its README, its claims

`data/`, `dataset/` and the model checkpoints are not in the repository.

---

## Reproduce

Each measured artifact has one producer:

    scripts/eval/baselines.py           -> baselines.json, baselines_coef.json
    scripts/eval/missed_detection.py    -> missed_detection*.json
    scripts/eval/blind_attribution.py   -> blind_attribution.json
    scripts/eval/ablate_h2.py           -> the H2 contrasts
    scripts/eval/split.py               -> the sealed dev/test split and its digest

Most take `--obs`, `--boot` and `--out`; run any of them with `-h`. They need the
packed observability maps under `data/pack/`, which are not in the repository.

Checking the committed numbers needs neither the data nor a GPU:

    python -m pytest tests/ -q
    python scripts/ci/check_gates.py
    python scripts/ci/check_observability_gates.py

Those three are what CI runs. The gates read the JSON artifacts directly, so a
changed number fails the build even when the code still passes its own tests.

`docs/METHOD_FREEZE.md` records what was run, when, against which split digest,
and what the acceptance threshold was beforehand.

---

## Known gaps

* **MC-dropout and deep ensembles are not run.** The post-hoc uncertainty family is
  complete (MSP, margin, `p_free`, range); the sampling family needs a GPU and is
  the largest single gap in the evaluation.
* **A second backbone is in progress.** H2 is measured on FB-OCC only. FlashOcc is
  registered in A62 with its acceptance fixed in advance; until it runs, the
  cross-architecture claim is unsupported.
* **Real-degradation robustness is untested.** Every robustness number in the
  archived project was measured against weather this repository rendered itself,
  which is close to circular. nuScenes mini's genuine night and after-rain scenes
  have never been evaluated.

---

## Authors

Akila Lourdes Miriyala Francis · Akilan Manivannan
Long Island University, Brooklyn
