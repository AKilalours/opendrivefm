# Validation report: camera observability as a perception-degradation monitor

**Item:** OpenDriveFM observability measure
**Function under validation:** per-cell estimation of how much camera evidence a
camera-only 3D occupancy prediction actually rests on
**Framing:** ISO 21448 (SOTIF) vocabulary
**Date:** 23 September 2026
**Evidence base:** Occ3D-nuScenes validation split, 6,019 keyframes, 150 scenes,
one frozen public checkpoint (FB-OCC r50)

---

## 0. What this document is, and what it is not

**This is not an ISO 21448 safety case and must not be cited as one.** It is a
validation report on a research prototype, written in SOTIF vocabulary because
that vocabulary is the right one for the hazard class involved. A real safety
case requires an item definition agreed with a vehicle programme, a hazard
analysis and risk assessment over a declared ODD, field data from the target
vehicle, and independent assessment. None of those exist here.

What this document does contain: a functional insufficiency named precisely, a
measurement of it on public data, pre-stated acceptance criteria, the evidence
for and against each claim, and a residual-risk section that is longer than the
results section. Every number is traceable to a script, a committed artifact and
an amendment in `docs/METHOD_FREEZE.md`, which records for each entry whether it
was decided before or after the data existed.

Claims this report explicitly does **not** make: that the measure is validated
on a production vehicle; that it has been tested in closed loop; that it
generalises beyond one model, one dataset and one sensor rig; or that any
threshold in it is a calibrated safety limit.

---

## 1. The functional insufficiency

A camera-only occupancy network emits a class and a confidence for every cell in
a volume around the vehicle. It emits them **whether or not any camera could see
that cell.** Nothing in the output distinguishes a cell that six cameras agree
on from a cell behind a parked truck that no camera has ever observed.

This is a functional insufficiency in the SOTIF sense: the system performs as
designed and is still unsafe, because the design does not represent the
difference between evidence and inference.

The motivating field event is not hypothetical. In July 2026 an operator
recalled its entire robotaxi fleet after a vehicle failed to detect heavy smoke
and drove into an active fire scene. The sensors could not see, and nothing in
the stack knew it.

## 2. The proposed mitigation

A per-cell scalar, computed from geometry alone:

    observability(v) = max_i ( coverage_i(v) * visible_i(v) )

`visible` is per-pixel ray marching with first-hit semantics against the
occupancy volume; `coverage` is the solid angle the cell subtends, clipped.
One parameter (`FULL_PX`). No learning, no training data, no LiDAR.

The measure is **available at inference on a real vehicle** from camera
extrinsics, intrinsics and the predicted occupancy. This matters and is
returned to in §6.

## 3. Acceptance criteria

Stated before the corresponding data existed, in `METHOD_FREEZE.md` amendments
A5 (pre-registration) and A21/A28/A30 (per-study). Reproduced here with outcome.

| # | Criterion | Outcome |
|---|---|---|
| AC-1 | Measure must predict per-voxel model error better than the benchmark's own visibility flag, by ≥ 0.03 AUROC | **PASS** +0.0534 [+0.0462, +0.0609] |
| AC-2 | Pipeline must reproduce a published benchmark result | **PASS** mIoU 38.90 vs published 39.1 |
| AC-3 | Zeroing must be geometrically justified: < 1% of zeroed cells may have a clear line of sight | **PASS** 0.06% |
| AC-4 | Calibration error must decline monotonically with observability | **FAIL** as pre-registered over all voxels; holds on occupied voxels only (ρ −0.865), reported as exploratory |
| AC-5 | Observability-conditioned recalibration must beat an unconditioned baseline | **PASS** after the target was corrected; see §6 |
| AC-6 | An unlabelled mining signal must find high-error frames at ≥ 2× lift | **PASS** 6.9–7.6× |

**AC-4 failed and is reported as failed.** It is not restated in a form that
passes.

## 4. Verification evidence

All figures on the Occ3D-nuScenes validation split unless stated. Amendment,
script and artifact given for each.

### 4.1 The measure separates error (A26)

| Score | AUROC |
|---|---|
| observability | 0.6748 [0.6655, 0.6839] |
| Occ3D `mask_camera` | 0.6215 [0.6136, 0.6295] |
| **difference** | **+0.0534 [+0.0462, +0.0609]** |

6,019 frames, paired scene bootstrap. Replicated across two independent
occlusion implementations that share no geometry code (+0.0521 depth-buffer,
+0.0527 ray-march, A11/A12).

### 4.2 Hazard localisation: partial sight is worse than none (A26, A14)

Confidence minus accuracy, non-free voxels, 52.4 M voxels:

| Region | Gap |
|---|---|
| out of every field of view | +0.0534 |
| occluded | +0.1745 |
| **barely observed (decile 1)** | **+0.2941** |
| well observed (decile 9) | +0.1781 |

Deciles are strictly monotone across all nine. **The system is best calibrated
where it is completely blind and worst where it can just barely see** — a factor
of 5.5. For a safety argument this inverts the usual intuition: total occlusion
is comparatively benign because the system does not commit; marginal visibility
is where it commits and is wrong.

### 4.3 Consequence: missing objects, not just wrong voxels (A22)

8,551 ground-truth dynamic objects. "Nothing dynamic placed inside the box":

| Observability | Missed entirely | Nothing dynamic |
|---|---|---|
| 0 (no evidence) | 12.24% | **48.34%** |
| 0.000–0.176 | 10.26% | 25.66% |
| 0.796–1.000 | 1.64% | **5.28%** |

Monotone across six bands, 9.2× spread. Controls, predicting "nothing dynamic
placed": observability 0.7650; **LiDAR-points-in-box oracle 0.7671**; nuScenes
human annotator visibility 0.6974; range alone 0.6607. Inside fixed range shells
observability holds 0.72–0.78, so it is not a range proxy.

A camera-only geometric quantity sits **0.0021 AUROC below a score that requires
the sensor the system does not have.**

### 4.4 Sensor-failure exposure (A21)

Recomputing the measure over five cameras instead of six. Obstacle voxels losing
all evidence on a single-camera failure:

CAM_BACK 30.20% · CAM_FRONT 22.09% · CAM_FRONT_LEFT 11.88% · CAM_FRONT_RIGHT
10.82% · CAM_BACK_LEFT 9.45% · CAM_BACK_RIGHT 8.28%

**Redundancy is uneven by 3.6×**, tracking unique angular coverage: CAM_BACK has
an 89.3° horizontal field of view, the other five 64.3–65.0°, and five 65°
cameras cannot close a 360° ring. Not traffic asymmetry — obstacles run 53.7%
ahead of the ego against 46.3% behind.

The accuracy gap between a camera's uniquely-covered region and the rest stays
within ±3.7 points and changes sign. **There is no hidden margin in the fragile
sector.**

### 4.5 Blind-spot causation (A29)

169,468,739 occluded voxels attributed to the object that occludes them:

| Occluder | Hidden volume | Hidden drivable surface |
|---|---|---|
| structures | 48.3% | 32.3% |
| vegetation | 30.7% | 12.1% |
| **vehicles** | **17.3%** | **45.4%** (131 m²/frame) |

Vehicles are 17% of what the stack cannot see and **45% of the road** it cannot
see, because a vehicle sits on the roadway at eye level and a building does not.

### 4.6 The condition is not resolved by adding hardware (A32)

76.94% of obstacle voxels have no camera coverage. A seventh camera, over six
candidate placements, recovers **at most 2.2%** of them. **The rig is
occlusion-limited, not coverage-limited.**

### 4.7 The condition is partly resolved by motion (A31)

Time until a currently-blind cell is first observed, 4 s look-ahead, ego poses:

- all blind volume: **79.5% still blind after 4 s** — mostly building interiors the vehicle never enters
- restricted to cells the ego's own future trajectory passes through: **58.3% revealed within 2 s**, median 1.90 s
- but **31.6% of the ego's own path that is blind now is still blind 4 s later**

That last figure is the operationally relevant one: space the vehicle enters
having never observed it.

### 4.8 Clearance envelope (A30)

Corridor 2.8 m wide, envelope 0.6–3.4 m, braking 4.0 m/s², no ground truth used:

| | p10 | median | p90 |
|---|---|---|---|
| near field unverifiable | 1.2 m | 1.2 m | 8.4 m |
| verified-free reach | 10.0 m | 26.4 m | 36.8 m |

Frames where verified reach is shorter than the stopping distance: **4.3%
overall, 24.5% above 10 m/s.** The corridor terminates on an obstacle 96% of the
time and on missing evidence 4%.

**This is a screening statistic, not a violation count.** Most flagged frames are
ordinary car-following at a legal gap. Its value is the tail across releases.

### 4.9 Monitoring without labels (A28)

Frame-level signals computed with no ground truth, validated against held-out
error:

| Signal | AUROC | Precision@100 | Lift |
|---|---|---|---|
| mean top-1/top-2 margin | 0.8618 | 69% | 6.9× |
| mean observability | 0.8066 | 59% | 5.9× |
| random | 0.5000 | 10% | 1.0× |

Four count-based signals are **anti-correlated** (0.17–0.22) and select easy
frames; see §7.

### 4.10 Implementation (A35)

Precomputed ray-to-voxel tables, verified **bit-identical** — 0 differing voxels
in 30,720,000 compared — at 3.46× the throughput. Full-split processing 114 min
→ 33 min. Cost: 832 MB resident.

---

## 5. Coverage of the evidence

| Dimension | Covered | Not covered |
|---|---|---|
| Frames | 6,019 keyframes, 150 scenes | non-keyframes; frames between 2 Hz samples |
| Sensor rigs | 2 nuScenes configurations | every other rig |
| Models | FB-OCC r50, frozen | every other backbone — **the principal gap** |
| Datasets | nuScenes | Waymo, Argoverse, any non-urban ODD |
| Conditions | as sampled by nuScenes | fog, heavy rain, snow, night are present but not stratified or reported separately |
| Failure modes | occlusion, field-of-view, single-camera loss | lens contamination, blooming, exposure failure, calibration drift, latency |

---

## 6. Corrected claim on calibration (A34)

Three earlier attempts concluded that observability-conditioned recalibration
fails against `mask_camera`. On 3,852,160,000 voxels with an explicit class term:

| Model | log-loss | ECE |
|---|---|---|
| class + confidence | 0.159283 | 0.002177 |
| + `mask_camera` | 0.106443 | 0.001536 |
| + observability | 0.153435 | 0.001780 |
| + both | 0.101388 | 0.001195 |

Observability beats class+confidence (+0.005820, CI excludes zero). It **loses**
to `mask_camera` (−0.046941). It **adds on top of** `mask_camera` (+0.005069, CI
excludes zero).

**Root cause of the earlier failures: `mask_camera` is not a competing
visibility measure. It is a label-validity flag** that Occ3D constructs offline
from the full sensor suite to mark where its own ground truth is trustworthy. A
flag predicting where labels are noisy will beat any perception signal at
predicting apparent error. **It also does not exist at inference on a vehicle.**

For a safety argument the relevant comparison is therefore against class and
confidence, which the measure wins, and the fact that it adds information on top
of `mask_camera` establishes that the two are not measuring the same thing.

---

## 7. Residual risk and known unknowns

**RR-1 — Single model.** Every result rests on one frozen checkpoint. The
finding may be a property of FB-OCC r50 rather than of camera geometry. This is
the largest open risk and is being addressed by a second backbone.

**RR-2 — Single dataset and ODD.** nuScenes is urban Boston and Singapore.
Nothing here speaks to highway, rural, or adverse weather as a stratified
condition.

**RR-3 — Voxel resolution.** 0.4 m voxels. A29 records a ray-sampling artefact
at range that inflates ground self-occlusion; ground classes are excluded from
that headline for this reason.

**RR-4 — Thresholds are analysis choices, not safety limits.** The 0.35
barely-observed band, the 0.70 confidence threshold, `tau = 0.10` and `0.15`,
the 2.8 m corridor and 4.0 m/s² deceleration were fixed before their runs and
not tuned, but none is calibrated against a hazard rate. No threshold in this
report may be used as an operational limit without that work.

**RR-5 — Aggregation does not transfer (A28).** A per-voxel finding does not
become a frame-level signal by summing it. Four count-based signals built from
the §4.2 danger-zone result are anti-correlated with frame error, because a
count tracks scene density and dense confident scenes are easy. The danger-zone
result is a statement about voxels and must not be used as a frame-selection
rule.

**RR-6 — No closed loop.** The measure has never gated a real decision. All
evidence is offline on recorded data. Whether surfacing it to a planner improves
or degrades behaviour is untested.

**RR-7 — Ground truth is itself camera-derived in part.** Occ3D's `mask_camera`
is built from the sensor suite; using it as an evaluation scope introduces a
dependency between the measure and its reference. §6 is the consequence.

**RR-8 — Pre-registered hypothesis failed (AC-4).** The monotone-decline
hypothesis failed over the population it was registered on. The stratified
version that holds is post-hoc and is labelled exploratory everywhere it
appears.

**RR-9 — Unmodelled degradations.** Lens contamination, direct sun, blooming,
exposure failure and calibration drift all produce a camera that is pointed at a
cell and cannot see it. The measure would score such a cell as observed. It is a
**geometric** visibility measure and treats a working camera as a seeing camera.
This is the most important gap for a real deployment.

---

## 8. Conclusion

Within the coverage of §5, the measure detects the functional insufficiency of
§1: it separates model error better than the benchmark's own visibility flag
(+0.0534), predicts whether an object will be missed at nearly the level of a
LiDAR oracle (0.7650 vs 0.7671), localises where a stack is most overconfident
(5.5× at marginal visibility), and quantifies single-sensor exposure (3.6×
uneven) and the hardware ceiling on fixing it (≤2.2%).

It does not constitute validation of a safety mechanism. RR-1, RR-6 and RR-9
each individually prevent that, and no threshold here is calibrated against a
hazard rate.

The defensible statement is narrow and worth making precisely:

> A camera-only geometric quantity, computable at inference without LiDAR,
> identifies where a camera-only occupancy stack is operating without evidence,
> and does so nearly as well as a measure that requires the sensor the stack
> does not have.

---

## 9. Traceability

| § | Claim | Amendment | Script | Artifact |
|---|---|---|---|---|
| 4.1 | +0.0534 AUROC | A26 | `eval/h2_vs_maskcamera.py` | `h2_max.json` |
| 4.2 | 5.5× danger zone | A26, A14 | `eval/kill_gate.py` | `killgate_max_nonfree.json` |
| 4.3 | 48.3% → 5.3% | A22 | `eval/missed_detection.py` | `missed_detection.json` |
| 4.4 | 30.2% → 8.3% | A21 | `eval/camera_dropout.py` | `camera_dropout.json` |
| 4.5 | vehicles 45.4% of road | A29 | `eval/blind_attribution.py` | `blind_attribution.json` |
| 4.6 | ≤2.2% from a 7th camera | A32 | `eval/camera_placement.py` | `camera_placement.json` |
| 4.7 | 31.6% path still blind | A31 | `eval/time_to_visibility.py` | `time_to_visibility.json` |
| 4.8 | 24.5% above 10 m/s | A30+A38 | `eval/safety_envelope.py` | `safety_envelope.json` |
| 4.9 | 6.9× mining lift | A28 | `mining/mine.py` | `mining_validation.json` |
| 4.10 | 3.46×, bit-identical | A35 | `perf/march_bench.py` | `march_bench.json` |
| 6 | calibration, corrected | A34 | `eval/recal_root_cause.py` | `recal_root_cause.json` |
| AC-3 | 0.06% unexplained | A20 | `eval/verify_surf_zero.py` | `surf_zero_verdict.json` |

Gates over these artifacts run in CI on every push
(`scripts/ci/check_observability_gates.py`), and each gate is itself tested
against a deliberately broken artifact.
