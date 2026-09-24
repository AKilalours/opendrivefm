# Method freeze

Written 16 September 2026. Amended only by appending, never by editing a line
above this one.

The purpose of this file is to make it impossible to quietly improve a result
after seeing it. Everything below was decided before the data it governs
existed. Where that is not true, it says so.

---

## The rule

**Unlimited iteration on dev. The test split is opened once, with the method
already frozen.**

Sealed dev/test split sha: `3e00ea450bb17507`

Anything tuned after the test split is opened is not a result, it is a
hyperparameter, and it has to be reported as one.

---

## Stage 1 — observability vs. annotator visibility

### The surviving claim

Range-stratified AUROC improvement of observability over range alone:

| band | delta AUROC | p |
|---|---|---|
| 0–10 m | +0.0543 | 0.146 |
| 10–20 m | +0.1130 | < 0.001 |
| 20–30 m | +0.1404 | < 0.001 |
| 30–45 m | +0.1088 | < 0.001 |

Paired scene bootstrap. Scenes are the independent unit because consecutive
keyframes in a scene contain the same objects.

### The dead claim

Marginal AUROC over range, unstratified: **+0.0247, p = 0.128.**

This is not significant and the claim is abandoned. It is recorded here
because it was the headline for several days and its death is part of the
result. It is not to be revived by a different bootstrap, a different
stratification, or a larger n.

### Headline evolution, with causes

| value | why it changed |
|---|---|
| 0.954 | wrong — was VLLM's number, not this measure's |
| 0.6858 | first real measurement |
| 0.6648 | LiDAR was in sensor frame, grid in ego frame |
| 0.6439 | three reference bugs in `footprint_visibility` |

Final: **0.6439, 95% CI [0.6271, 0.6613].** LiDAR-return oracle ceiling
0.7122, so the gap to an empirical upper bound is 0.068.

### Limits that must appear in the paper

- **No claim is made about observability within 2 m of a camera.** The
  bearing-bucket approximation degrades there and the acceptance test was
  never passed at that range.
- The occlusion term contributes +0.0595, p < 0.001.
- Per-camera trust is currently a constant 0.795 and contributes **zero**
  discrimination. It is in the formula for future work, not for this result.
- The grazing-angle feature was retracted: all six cameras sit within 9.5 cm
  of each other in height, so it was a pure range proxy.

---

## Stage 1b — the observability cache

Verified before use, not after.

| | ground channel | surface channel |
|---|---|---|
| coverage | 46,656 cells x 6,019 frames | every occupied cell |
| agreement vs exact test | 99.88% (100% inside 10 m) | 100.0000% |
| bias | 0.04% / 0.09% | none |

The surface channel is exact because sampling was abandoned once the occupied
set turned out to be only ~2,190 cells per frame.

---

## Stage 2 — the model predictions

### What was thrown away, and why

The first export collapsed each BEV column to the level with the lowest
P(free). Two defects, both mine:

1. `p_occ = 1 - min_z P(free)` saturated. 58.8% of columns sat at exactly
   1.0, because a column gets sixteen chances to contain one low-P(free)
   voxel.
2. More seriously, the level with the lowest P(free) is almost always the
   **road surface**, because the road is what the model is surest about. The
   collapse therefore reported the ground and discarded whatever stood on it.
   Measured: 40.5% driveable_surface, 20.8% manmade, 18.5% terrain, 14.1%
   sidewalk, and **1.2% for all ten dynamic classes combined**.

This was caught by the grid-alignment probe, which could not separate any of
the eight dihedral transforms (p_occ AUROC spread 0.613–0.640, margin 0.0015).
A probe that cannot tell a 90-degree rotation from the truth is not measuring
the scene. That failure is the only reason the defect was found before the
analysis was built on it.

**No number from the collapsed export is used anywhere.**

### The replacement

Per-voxel export on the Occ3D grid, 200 x 200 x 16 at 0.4 m: `cls` (argmax
over all 18 channels, free included as a legitimate class), `conf` (its
softmax probability), `conf2` (runner-up, so margin is available without a
third run), `p_free`. All uint8 at 1/255, an order of magnitude finer than
any calibration bin.

### Acceptance gates, fixed before any output existed

| gate | threshold |
|---|---|
| size extrapolated to 6,019 frames | <= 13 GB |
| dynamic classes as share of non-free voxels | >= 1.0% |
| conf median | <= 0.999 |
| conf at ceiling (>= 0.998) | <= 50% |
| free fraction | 0.55–0.95 |

A failed gate stops the run. It is not a reason to proceed carefully.

### Grid orientation

**UNRESOLVED as of this writing.** Occ3D is 200 x 200 x 16 at 0.4 m over
+/-40 m; the observability grid is 216 x 216 at 0.5 m over +/-54 m; both are
ego-framed, and the observability convention is flat = i*n + j with
x = (i+0.5)*res - rng and y = (j+0.5)*res - rng.

Orientation will be settled empirically against LiDAR, by scoring all eight
dihedral transforms, and NOT by trusting a documented convention. The
decision rule is fixed here: the winning transform must beat the runner-up by
a margin of at least 0.05 on the height probe. Below that margin the
orientation is not resolved and no downstream result may be produced.

---

## Stage 3 — calibration conditioned on observability (the kill gate)

### Pre-registered binning

78.7% of occupied cells have observability exactly zero, so equal-width or
equal-count bins would put four fifths of the data in one bucket. The split
is therefore:

- bin 0: observability == 0
- bins 1..10: deciles **within** observability > 0, computed on dev only

This was decided and written down **before the model predictions existed**.
That window has now closed. The split is not re-chosen.

### Confidence

Max-softmax over the 18 channels, per voxel. The column-collapsed `p_occ` is
excluded outright for the reasons above.

### Headline statistic

`gap(obs = 0) - gap(obs > 0)` where `gap = confidence - accuracy`.
Scene-level bootstrap, 2,000 resamples, 95% percentile interval.

### Ground truth

Two sources, not interchangeable:

- **LiDAR** — smoke test only. LiDAR shares most of the camera's occlusion
  geometry, so "the model is wrong where cameras cannot see" is confounded
  with "the GT is wrong where cameras cannot see". **No number from this mode
  is publishable**, and the script prints that banner when run in it.
- **Occ3D-nuScenes** — the publishable mode. Also enables the comparison that
  matters: does our continuous, resolution- and trust-weighted, noisy-OR
  observability predict model error better than Occ3D's own **binary**
  `mask_camera`.

That comparison is the paper's central experiment and must be reported
whichever way it comes out.

---

## Standing constraints

- `data/`, `dataset/` and `outputs/artifacts/*` checkpoints are never
  committed.
- Every number in the paper is measured. No number is estimated, rounded up,
  or carried over from a previous version of the pipeline.
- "Re-run until the values look better" is not available. Dev iterates; test
  opens once.

---

# Amendments

Appended, never edited above. Each entry says what changed, when, and whether
it was decided before or after seeing the data it governs.

## A1 — Gate 4 threshold was wrong (17 Sep 2026, AFTER seeing data)

The mini-gate required the free-voxel fraction to fall in 0.55-0.95. Measured:
0.369. **The threshold was wrong, not the data**, and the reasoning that set it
was wrong too: "Occ3D is mostly empty air" is true of the ground truth INSIDE
`mask_camera` and false of raw unmasked model output. FB-OCC is trained with
that mask, so every voxel behind the first visible surface is unsupervised and
gets filled with whatever -- measured as 43.7% `manmade`.

The gate was NOT relaxed to fit. It was replaced by a test that distinguishes
the two explanations, and that test was specified before it was run: if the
volume is read correctly, free should be high near the ego and high overhead,
with `manmade` climbing at range. Result:

    range     free    manmade          height        free
     0-10 m   74.0%    1.7%            z = -1.0 m     0.1%
    10-20 m   64.5%   11.5%            z = +0.2 m    17.7%
    20-30 m   43.7%   33.6%            z = +2.2 m    47.4%
    30-40 m   30.3%   52.5%            z = +3.8 m    54.5%
    40-60 m   12.4%   72.9%

Monotone in both, in the predicted direction. The volume is read correctly.

The lesson is recorded because it will recur: a gate must be written against
the quantity actually being measured, not against the benchmark it resembles.

## A2 — `--frames` took the head, not a sample (17 Sep 2026)

`kill_gate.py --frames N` sliced `index[:N]`. Consecutive keyframes share a
scene, so `--frames 300` gave **8 scenes**, and the scene bootstrap had 8
independent units. Fixed to sample. The 300-frame result (CI spanning zero) is
void; only the full-split result stands.

## A3 — Grid orientation RESOLVED: identity (17 Sep 2026)

Per-voxel 3D probe, Matthews correlation between model non-free voxels and
LiDAR-occupied voxels, r < 20 m where cameras are supervised and LiDAR dense.

    frames    identity   runner-up (flip_x)   margin
        60     +0.1564             +0.1011    +0.0554
       200     +0.1618             +0.1068    +0.0550

Margin rule was >= 0.05, fixed before the probe ran. Passes, and is stable
under a 3.3x increase in data. The other six transforms bunch at 0.068-0.074,
which is what a wrong orientation looks like. Corroborated independently by
the earlier BEV tall-class probe, where identity was the only transform with
positive separation.

Absolute MCC is low (0.16) because the model's non-free set includes the
unsupervised fill, which LiDAR has no returns for. That does not affect the
ranking.

## A4 — LiDAR smoke test result (17 Sep 2026). NOT PUBLISHABLE.

Full split, 6,019 frames, 150 scenes, 3.85 billion voxels:

    gap(obs=0) - gap(obs>0) = +0.0362   95% CI [+0.0231, +0.0493]

Direction as predicted. **This is not evidence for the paper's claim.** Gaps
run 0.23-0.53, which is not a property of a published model but of ground
truth that calls the model wrong wherever LiDAR got no return. The +0.036
effect sits on a +0.42 baseline that is mostly artifact, and LiDAR shares the
camera's occlusion geometry, so the effect cannot be separated from GT error.
What it establishes is that the pipeline runs: binning, scene bootstrap, and
0.038 s/frame.

Observed but NOT pre-registered, and therefore exploratory: within obs > 0 the
gap falls almost monotonically across deciles, +0.5227 / +0.5331 / +0.4600 /
+0.4303 / +0.4069 / +0.3790 / +0.3364 / +0.2920 / +0.2670 / +0.2305 -- nine
strict decreases with one reversal between deciles 1 and 2.

## A5 — PRE-REGISTRATION for the Occ3D run (17 Sep 2026, BEFORE the data exists)

Written while Occ3D-nuScenes GT is not yet downloaded to this machine. The
window for pre-registering these is therefore still open, and closes the
moment that download completes.

**Scoring scope.** All Occ3D results are computed inside `mask_camera` only.
Outside it the training loss never supervised the model, so scoring there
measures the unsupervised fill rather than the model. This also removes most
of the region where the LiDAR confound lived.

**H1 (primary, replaces the zero-vs-nonzero headline inside the mask).**
Within `mask_camera`, the calibration gap (confidence - accuracy) decreases
monotonically with observability decile. Tested by Spearman correlation
between decile index and gap, scene-bootstrapped. Pre-registered direction:
negative. `mask_camera` keeps roughly the obs > 0 region, so the zero bin
largely disappears under masking and the original headline has little left to
measure there; it is still reported, for whatever zero-observability voxels
survive the mask.

**H2 (the comparison that matters).** Our continuous observability predicts
model error better than Occ3D's own binary `mask_camera`. Operationalised as:
AUROC of per-voxel error, observability as the score, versus `mask_camera` as
the score, on the same voxels, paired scene bootstrap. This is the experiment
the paper turns on, and it is reported whichever way it comes out. A null or
negative result here is a finding about the contribution's size, not a reason
to look for another statistic.

**What would falsify the contribution.** If H2 shows observability does not
beat the binary mask, the continuous measure is not earning its complexity
and the paper says so.

## A6 — Occ3D results (17 Sep 2026, AFTER the data existed)

Pipeline validated first: FB-OCC r50 reproduces **mIoU 37.66** against a
published high-30s, with identity orientation beating every other transform by
4x (next best flip_x at 9.50). That confirms export, orientation, class
indexing, z-convention and mask handling simultaneously, and puts A3 on far
firmer ground than the LiDAR probe's 0.055 margin did.

**H1 HOLDS.** Inside mask_camera, gap by observability decile:

    .1081 .0984 .0869 .0838 .0815 .0701 .0604 .0522 .0497 .0347

Strictly monotone, ten of ten, no reversals, direction as pre-registered.
Spread 3.1x. Gaps are now plausible (0.035-0.108) where LiDAR GT gave
0.23-0.53, confirming those were artifact.

Old headline, retained for completeness: gap(obs=0) - gap(obs>0) = +0.0061,
CI [+0.0032, +0.0093]. Significant, and six times smaller than the decile
spread. The zero bin sits mid-pack between deciles 6 and 7.

**H2 FAILS.** Predicting per-voxel error, full volume:

    observability   AUROC 0.5491  [0.5440, 0.5544]
    mask_camera     AUROC 0.6215  [0.6136, 0.6295]
    difference     -0.0724  95% CI [-0.0791, -0.0652]

Inside mask_camera, observability alone: **AUROC 0.5280**.

Per A5's falsification clause: the continuous measure does not currently earn
its complexity. The binary mask that ships with the benchmark is better.

**The tension that must be stated in any writeup.** H1's decile table is
aggregate: each bin averages tens of millions of voxels, so bin means separate
cleanly while the per-voxel distributions overlap almost entirely (AUROC
0.528). A monotone bin-mean trend is NOT evidence of per-voxel discrimination.
Presenting the decile table without the AUROC would be misleading, and is
forbidden.

**Known cause, identified BEFORE H2 was run.** Observability is a BEV
ground-cell quantity broadcast across all 16 height levels, so a voxel of air
above an occluded ground patch inherits zero. 65% of camera-visible voxels
(350M of 536M) get observability exactly 0 under this broadcast. We are
comparing a 2D measure against a true 3D mask on a 3D task.

**Next step, and its honest prior.** Rebuild observability per voxel -- ray
from lens to THAT voxel, not to the ground column beneath it. This is a
methodological fix, not a search for a better number, and it was named before
H2 ran. It may not rescue H2. If per-voxel observability still loses to
mask_camera, the contribution as framed is dead and the paper reports that.

## A7 — Per-voxel observability, and the mixed verdict (17 Sep 2026)

Observability rebuilt per voxel: z-buffer occlusion against the dense Occ3D
volume, coverage evaluated at each voxel's own depth. Two defects fixed, both
named BEFORE the re-run, neither found by hunting for a better number:

1. **Column broadcast.** A BEV ground value applied to all 16 heights, so air
   above occluded pavement inherited zero. 65% of camera-visible voxels were
   wrongly zeroed.
2. **Saturated coverage.** FULL_PX = 140 was calibrated for a 0.5 m GROUND
   patch, whose projected area falls as 1/r^3 from foreshortening. A voxel face
   falls as 1/Z^2, so the same constant put saturation at 42.8 m -- outside the
   grid. Coverage was 1 everywhere and the measure was two-valued: p50 = p90 =
   0.796 = trust. Re-derived by preserving the validated 16.3 m saturation
   range: FULL_PX = 965.

**Consequence, stated not buried.** Occluders are now the dense Occ3D volume,
because single-sweep LiDAR left the z-buffer full of holes (93% of all voxels
came back visible against Occ3D's 13.9%). Observability is therefore an
OFFLINE quantity requiring ground-truth geometry. The runtime-signal framing
is withdrawn until a proxy exists. The validation, sensor-placement and
safety-case uses are unaffected, and the H2 comparison becomes cleaner:
identical geometry, ours continuous, theirs binary.

### H2 PASSES. Full split, 6,019 frames, 150 scenes.

    observability   AUROC 0.6736  [0.6663, 0.6804]
    mask_camera     AUROC 0.6215  [0.6136, 0.6295]
    difference           +0.0521  95% CI [+0.0429, +0.0609]
    inside mask_camera: observability 0.6343

Swing from the 2D measure: +0.1245 AUROC. The contribution is supported: a
continuous measure beats the benchmark's own binary mask at predicting model
error, and discriminates inside the region that mask calls uniformly visible,
which a binary flag cannot do by construction.

### H1 FAILS.

    .0328 .0528 .0537 .0542 .0540 .0520 .0503 .0464 .0384

A hump, not the pre-registered monotone decrease. H1 is recorded as FAILED and
is not replaced by whichever statistic happens to survive.

### The demoted headline recovered.

gap(obs=0) - gap(obs>0) = +0.0592, CI [+0.0544, +0.0644], zero bin +0.1037 and
clearly the worst. A5 demoted this in favour of H1 and got it backwards.

### Open question, to be tested not narrated

Decile 1 has the HIGHEST confidence (0.9746) and HIGHEST accuracy (0.9418) of
deciles 1-8. Barely-visible voxels being best-calibrated is not a visibility
story. Most likely a CLASS-COMPOSITION confound: extreme deciles dominated by
free space (easy), middle deciles holding hard surface voxels. If so the
deciles are confounded with their contents and calibration must be conditioned
within class, or restricted to non-free GT voxels. Until tested, the hump is
an open question in the writeup, not a finding.

### Reporting rule for this result

H2 passed, H1 failed, and the statistic A5 demoted is the one that recovered.
All three are reported together. Reporting only the winners is the same class
of error as showing a decile table without its AUROC.

## A8 — The occlusion test is not yet correct (17 Sep 2026)

The composition column exposed the confound A7 suspected, and worse.

    bin        %GTfree
    obs = 0       54.6
    decile 1      95.5
    decile 9      84.8

The zero bin is half surface voxels; every positive decile is ~90% empty air.
So +0.0592 was substantially measuring bin CONTENTS, not visibility.

Restricted to non-free GT voxels, the headline REVERSES:

    gap(obs=0) - gap(obs>0) = -0.0688   CI [-0.0751, -0.0624]

i.e. surface voxels our measure calls invisible are BETTER calibrated
(accuracy 0.729) than ones it calls barely visible (0.529). That is not a fact
about the world; it is a symptom, and the cause is in our measure.

**Diagnosis, tested not asserted.** 66% of non-free voxels inside mask_camera
get observability 0. Occluders are the GT non-free volume, which contains the
surface voxel under test, and the 3x3 buffer dilation drags a nearer
neighbour's depth into that voxel's own bin. Surfaces sit at depth
discontinuities, so they shadow themselves -- the 3D form of the
self-occlusion bug already fixed once in the 2D per-box measure.

Measured over 20 frames, share of voxels zeroed:

    build                         surface   free   ratio
    point splat, no dilation        0.333  0.079    4.19
    point splat, 3x3 dilation       0.681  0.296    2.30
    isotropic depth-aware splat     0.855  0.601    1.42

All three are wrong, each differently. The isotropic splat over-occludes
because it gives a voxel a SQUARE image footprint: a road voxel 2 m ahead gets
a ~126 px radius and smears its 2 m depth from the bottom of the frame up to
the horizon, shadowing the distant scene.

**The fix, not yet implemented:** project each occluder voxel's eight corners
and splat its true 2D bounding box. That respects perspective and obliquity,
closes wall gaps without spreading depth sideways, and keeps near ground where
it belongs in the image.

**Status of the A7 numbers.** H2's +0.0521 was computed on the dilated maps,
which zero 68% of surface voxels. It is NOT final. The direction is well
supported -- a swing of +0.1245 AUROC from the 2D measure is far too large to
be noise -- but the magnitude will move once occlusion is correct, and it
could move either way. No number in A7 is to be quoted until this is redone.

**Process note.** Three occlusion attempts in one session, each fixing the
previous failure and introducing a new one, and the third was worse than the
second. Work stopped deliberately rather than attempting a fourth variant of
the same routine while tired. The failure modes are getting more plausible,
which makes them harder to catch, not easier.

## A9 — Five occlusion attempts, all failed. Stopping. (17 Sep 2026)

    build                        surf_zero  free_zero  ratio  agree
    point splat, no dilation         0.333      0.079   4.19  0.375
    point splat + 3x3 dilation       0.681      0.296   2.30  0.672
    isotropic depth-aware splat      0.855      0.601   1.42  0.770
    corner bounding-box splat        0.926      0.623   1.49  0.765
    bbox + slope-scaled bias         0.780      0.610   1.28  0.751

Gates (fixed before any attempt): ratio 0.7-1.3, surf_zero < 0.25,
agree 0.75-0.90, spread >= 0.4.

The slope-scaled bias -- bias(r) = BIN*r^2/(f*h), derived from grazing-angle
geometry, no free parameter -- did what it was meant to: the surface/free
asymmetry fell 2.30 -> 1.28, so self-shadowing on oblique surfaces is largely
solved. But absolute over-occlusion is not: 78% of surface and 61% of free
voxels are zeroed. The bbox splat with nearest-corner depth occludes far too
much everywhere.

**A gate was badly designed.** `spread` returned exactly 0.796 for all five
builds, because it is p90 - p10 where p10 is always 0 and p90 is always the
0.796 ceiling. It measured the presence of zeros, not gradation. Must be
recomputed over NON-ZERO values. A diagnostic that cannot distinguish five
different algorithms is not a diagnostic.

**Diagnosis.** A z-buffer is the wrong tool here. It answers "what is nearest
along this pixel" when the question is "is anything between the lens and THIS
voxel". On a voxel grid with oblique surfaces those differ badly, and every
patch that fixed one symptom created another.

**The correct algorithm, for next session.** Per-voxel ray marching: sample the
segment from lens to voxel centre, test the GT occupancy volume at each sample,
occluded if any sample before the target is occupied, excluding the target's
own voxel. Rejected earlier as too slow at 23M rays/frame -- but the grid and
the cameras are BOTH ego-fixed, so the sample indices for a given voxel and
camera are identical in every frame. Precompute them once per camera; per frame
it is a gather and an any(). Estimated ~1.5 s/frame, so run on a
scene-stratified 1,500-frame subsample rather than all 6,019, which still
gives 150 scenes and ample power.

**Everything in A7 remains provisional.** No 3D number is quotable.

**Process.** Five attempts in one session, each fixing the previous symptom and
producing a new one. The correct response is a different algorithm, not a sixth
variant of the wrong one.

## A10 — Z-buffer abandoned. Ray marching is the method. (17 Sep 2026)

Continued past A9 at the user's request. Four further attempts, all measured
against gates fixed beforehand (surf_zero < 0.25, ratio 0.7-1.3, agree
0.75-0.90, spread_nz >= 0.4):

    build                              surf_zero  free_zero  ratio  agree
    bbox + BIN=2                           0.793      0.482   1.65  0.751
    bbox + BIN=1                           0.715      0.307   2.33  0.670
    far-face depth                         0.735      0.475   1.55  0.750
    off-screen discard + cap 512           0.795      0.619   1.28  0.759
    bias = res*r/h                         0.569      0.552   1.03  0.721

### Real bugs found and fixed along the way

- **Phantom occluders.** np.clip squashed off-screen occluders onto the image
  border instead of discarding them: 60.7% of the occluder set (12,606 of
  20,766) was geometry the camera cannot see, stamping small depths along the
  edges. Clipping is not intersection.
- **Bbox cap.** Capped at 24 bins while real near occluders reach 443, so
  genuine occlusion was under-covered while fake occlusion was injected --
  two errors in opposite directions, which is why they cancelled and the
  metric did not move when the first was fixed alone.
- **Wrong bias length scale.** Bias was derived from BIN; it must come from the
  occluder's own projected height, bias(r) = res*r/h. This took the
  surface/free ratio from 1.28 to **1.03**, i.e. self-occlusion on grazing
  surfaces is SOLVED.
- **A useless gate.** `spread` was p90-p10 including zeros, returning 0.796 for
  every build. Fixed to use non-zero values only.

### Hypotheses tested and REJECTED by measurement

- Frustum/projection error: 97.4% of voxels fall in >= 1 frustum with occlusion
  off, and surf_zero is 0.027. Projection is correct.
- Ego vehicle self-occlusion: only 32 occluders within 2 m of any lens; the
  ego box is 93% free. Not the cause.
- Bin resolution: 0.793 at 2 px, 0.715 at 1 px. Resolution was never the cause.

### What remains

Over-occlusion is now uniform across classes and GROWS WITH RANGE: zeroed
voxels sit at median 25.3 m, visible ones at 16.0 m. 53% of surface voxels that
Occ3D marks camera-visible are still zeroed, and the blocking occluder has not
been identified.

### Decision

A z-buffer answers "what is nearest along this pixel". The question is "is
anything between the lens and THIS voxel". On a voxel grid with grazing
surfaces those differ, and nine attempts have each fixed one symptom and
revealed another. Every remaining failure is an artefact of rasterising boxes
into bins -- a class of bug ray marching cannot have, because it never
rasterises anything.

**Next session: ray marching.** Sample the segment lens -> voxel, test the GT
occupancy volume at each sample, occluded if any sample strictly before the
target is occupied. Affordable because grid and cameras are both ego-fixed, so
sample indices are identical every frame: precompute once, then gather and
any() per frame. No bins, no bboxes, no bias, no clipping.

All A7 numbers remain provisional. Nothing 3D is quotable.

## A11 — Ray marching works. H2 replicates on a second algorithm. (17 Sep 2026)

The z-buffer line was abandoned in A10. Replacement: one ray per PIXEL, first
-hit semantics -- march from the lens, mark every free voxel reached plus the
first occupied one, stop. No bins, no bounding boxes, no depth bias, no splat
shape. The visibility term now has ZERO free parameters.

Two bugs found and fixed inside the marcher itself:
  * ray-per-VOXEL is the wrong formulation. The ray to a road voxel at 30 m
    passes through the road voxel at 29.6 m, which is also occupied, so the
    ground slab occludes itself end to end (95.4% of surface voxels zeroed).
    Occ3D casts per pixel, and each road voxel is the first hit of whichever
    pixel points at it.
  * the recursive shell march needed shells ONE step wide and a first-shell
    seed; at two steps wide a voxel's predecessor falls in its own shell and
    the cascade collapses to "nothing visible".

Frame-mismatch hypothesis TESTED AND REJECTED: all eight dihedral transforms
give agree ~0.748 and recall ~0.44. Not an orientation problem.

### Results, 364 frames / 38 scenes

    FULL VOLUME, predicting per-voxel error
      observability   AUROC 0.6871  [0.6680, 0.7050]
      mask_camera     AUROC 0.6289  [0.6138, 0.6435]
      difference           +0.0582  95% CI [+0.0408, +0.0763]
    INSIDE mask_camera
      observability   AUROC 0.6395
    CALIBRATION
      gap(obs=0) - gap(obs>0) = +0.0513  CI [+0.0421, +0.0599]

**H2 PASSES, and REPLICATES.** The z-buffer build gave +0.0521 on maps that
zeroed 68% of surface voxels; it was flagged as possibly artefactual. A
completely different algorithm gives +0.0582 with overlapping intervals. Two
independent implementations agreeing is far stronger than either alone: the
signal is in the measure, not in the bug. Acceptance bar was >= 0.03.

### Still open, not to be glossed

* surf_zero 0.704 against a < 0.25 gate. The marcher's strict first-hit
  semantics are stricter than mask_camera. Defensible as a definitional
  difference, NOT yet verified as intended. This gate stays failed.
* H1 still fails: deciles .030 .039 .051 .052 .051 .044 .041 .040 .027 -- a
  hump, not the pre-registered monotone decrease.
* Composition confound persists: %GTfree runs 92.2% (decile 1) to 85.1%
  (decile 9), so some decile structure is bin contents, not visibility.
* 364 of 1,500 frames used: the build took the FIRST 1,500 while the analysis
  samples randomly across all 6,019. Enough for a read, not for a final value.

### Next

Build all 6,019, re-run both, and report on the full split.

## A12 — FINAL VALUES. Full split, ray-marched measure. (17 Sep 2026)

6,019 frames, 150 scenes, 0 skipped, 536,229,736 voxels inside mask_camera.
Visibility by per-pixel ray marching with first-hit semantics: no bins, no
bounding boxes, no depth bias, no splat shape. ZERO free parameters in the
visibility term.

### H2 — the deciding experiment. PASSES.

    observability   AUROC 0.6742  [0.6649, 0.6833]
    mask_camera     AUROC 0.6215  [0.6136, 0.6295]
    difference           +0.0527  95% CI [+0.0455, +0.0602]
    inside mask_camera:  0.6354

Acceptance bar was >= +0.03. Achieved +0.0527.

**REPLICATION.** The z-buffer build gave +0.0521 on maps that wrongly zeroed
68% of surface voxels. The ray marcher -- a different algorithm sharing no
geometry code -- gives +0.0527. Two independent implementations within 0.0006
of each other. The signal is in the measure, not in either implementation's
bugs. This is the strongest evidence in the project.

### Calibration conditioned on observability

    gap(obs = 0) - gap(obs > 0) = +0.0537   95% CI [+0.0488, +0.0588]

    bin        n            conf     acc      gap    %GTfree
    obs = 0    185,546,662  0.9139   0.8147   +0.0992   60.3
    decile 1    24,887,987  0.9662   0.9223   +0.0440   91.7
    decile 2    18,242,447  0.9609   0.9048   +0.0561   88.0
    decile 3    21,344,183  0.9526   0.8923   +0.0602   87.9
    decile 4    24,494,177  0.9563   0.8981   +0.0582   87.3
    decile 5    27,180,327  0.9572   0.8989   +0.0583   86.7
    decile 6    30,561,515  0.9602   0.9053   +0.0549   86.2
    decile 7    36,821,989  0.9627   0.9109   +0.0517   85.7
    decile 8    26,018,839  0.9655   0.9194   +0.0461   85.4
    decile 9   141,131,610  0.9756   0.9418   +0.0338   85.2

Accuracy 0.9418 where cameras see well vs 0.8147 where they do not, while
confidence falls only 0.9756 -> 0.9139. The model loses 12.7 points of accuracy
and gives up 6.2 points of confidence: it does not discount enough for what it
cannot see. That gap IS the result.

### Stability across sample size

    frames    H2 diff    calib gap
       364    +0.0582     +0.0513
     1,532    +0.0564     +0.0502
     6,019    +0.0527     +0.0537

### STILL FAILING / OPEN -- reported with the wins, per A7's rule

* **H1 FAILS.** Deciles .0440 .0561 .0602 .0582 .0583 .0549 .0517 .0461 .0338
  are a hump, not the pre-registered monotone decrease. Recorded as failed.
* **Composition confound persists.** %GTfree runs 91.7 -> 85.2 across deciles
  and is only 60.3 at obs = 0, so part of the decile structure is bin contents.
  The non-free-only analysis must be re-run on the ray-marched maps before the
  calibration numbers are used in the paper.
* **surf_zero gate still fails** (0.704 vs a < 0.25 target). The marcher's
  first-hit semantics are stricter than mask_camera. Defensible as a
  definitional difference; NOT yet verified as intended behaviour.

### Quotable as of now

mIoU 37.66 (reproduces published FB-OCC r50). H2 +0.0527 [+0.0455, +0.0602].
Observability AUROC 0.6742 vs mask_camera 0.6215. Inside-mask 0.6354.
Calibration gap +0.0537 [+0.0488, +0.0588]. All on 6,019 frames / 150 scenes.

## A13 — Composition confound confirmed; recalibration FAILS. (17 Sep 2026)

### Composition confound: the calibration headline does not survive

Non-free voxels only, ray-marched measure, 6,019 frames / 150 scenes:

    gap(obs=0) - gap(obs>0) = -0.0293   95% CI [-0.0349, -0.0239]

The A12 headline of +0.0537 was driven by bin COMPOSITION (%GTfree 60.3 at
obs = 0 versus 85-92 in the deciles), not by visibility. On surfaces alone the
sign flips. The +0.0537 figure is withdrawn as a visibility result.

What DOES survive on surfaces is the decile trend, in H1's predicted direction:

    .2882 .2198 .2213 .2119 .2019 .1986 .1954 .1951 .1758

Calibration error falls from 0.288 to 0.176 as observability rises -- one
trivial reversal. Among voxels the cameras see at all, the monotone story
holds. The obs = 0 bin is anomalous (0.1659, better than every decile) because
our strict first-hit measure puts "behind the first surface" there: car
interiors and wall backs, which the model predicts well by continuation rather
than by sight. That heterogeneity is the measure's central weakness.

### Recalibration: DOES NOT WORK

                        full volume   surface only
    A  none               0.39744       0.23191
    B  global             0.00472       0.00370
    C  mask_camera        0.00362       0.00578
    D  observability      0.00559       0.00430

**D loses to B in BOTH scopes.** Conditioning on observability is worse than
not conditioning at all. On surfaces D beats C only because C is itself worse
than doing nothing there.

**My acceptance criteria were incomplete.** They required D > C and D > A, and
never required D > B, the do-nothing-extra baseline. The script therefore
printed "BOTH ACCEPTANCE CRITERIA MET" for a result where the conditioning
actively hurts. Any future scheme comparison MUST include the unconditioned
map as a mandatory baseline.

**A second flaw, caught before it could mislead.** The first run scored only
inside mask_camera, which made the mask grouping constant and silently
collapsed C into B -- they printed identical ECE to five decimals (0.00315).
Had that not been checked, the recalibration result would have been reported
as a win.

**Why it fails.** A single global histogram map removes ~98% of ECE. Splitting
into ten observability groups then fragments the data and adds variance with
no bias left to remove.

### Standing after today

* **H2 STANDS.** +0.0527 [+0.0455, +0.0602], replicated across two independent
  occlusion implementations (z-buffer +0.0521, ray march +0.0527).
* **H1 FAILS** on all voxels; holds directionally on surfaces only.
* **Calibration headline WITHDRAWN** as a visibility result (composition).
* **Recalibration FAILS** against the unconditioned baseline.

One positive result, three negatives, all measured. The paper is narrower than
hoped and much harder to attack.

## A14 — Zero bin split. A real finding, and my prediction was wrong. (17 Sep 2026)

The zero bin was split into OUT OF VIEW (no camera frustum contains the voxel)
and OCCLUDED (in view, behind the first surface). This changes REPORTING only:
observability itself is untouched, no new parameter, no new derivation.

**Prediction on record before running: occluded would be well calibrated and
out-of-view badly calibrated. The opposite is true.**

    FULL VOLUME                n            gap
      out of view        10,575,796      +0.0208   <- BEST of all bins
      occluded          174,970,866      +0.1039   <- worst
      deciles 1 -> 9                     +0.0440 -> +0.0338

    SURFACES ONLY
      out of view         3,370,114      +0.0521
      occluded           70,328,350      +0.1713
      decile 1            2,053,265      +0.2882   <- worst of everything
      decile 9           20,832,577      +0.1758

### The finding

**The model is most miscalibrated where it can BARELY see, not where it cannot
see at all.**

Out-of-view voxels are the best calibrated in the dataset: nothing was observed,
the model falls back on priors, and it is appropriately unsure. Marginal
visibility is the danger zone -- decile 1 on surfaces has a gap of 0.2882,
WORSE than having no information whatsoever (0.0521). The model commits on weak
evidence.

This also explains H1's hump without special pleading: the gap rises out of
decile 1 as evidence strengthens past the danger zone, then falls as it becomes
decisive. H1's pre-registered monotone form stays FAILED; the true structure is
non-monotone with a mechanism.

For AV this inverts the usual intuition. No visibility is comparatively safe,
because the system knows it is blind. Partial visibility is where it fools
itself. That is a statement about when to distrust a perception stack, and it
is the most useful thing the measure has produced.

### Recalibration: still fails, and the split made it worse

    A none 0.39744 | B global 0.00472 | C mask_camera 0.00362 | D obs 0.00605

D loses to the mandatory B baseline by 28.3%. Conditioning on observability
does not help calibration. Closed; not to be reopened by a further regrouping.

### Standing

* H2 STANDS: +0.0527 [+0.0455, +0.0602], replicated across two independent
  occlusion implementations.
* NEW: the barely-visible danger zone, measured on 6,019 frames / 150 scenes.
* H1 FAILED as pre-registered (monotone); real structure is non-monotone.
* Calibration headline WITHDRAWN as a visibility result (composition).
* Recalibration FAILED against an unconditioned baseline.

## A15 — Ablation. Occlusion carries it; it is NOT a range proxy. (17 Sep 2026)

1,500 frames, 150 scenes, paired scene bootstrap. All variants scored
identically against per-voxel error.

    variant       AUROC     95% CI              vs full
    full         0.6754   [0.6658, 0.6849]
    no_occ       0.5851   [0.5800, 0.5903]     +0.0903
    range_only   0.5920   [0.5867, 0.5974]     +0.0834
    no_cov       0.6622   [0.6526, 0.6717]     +0.0132
    n_vis        0.6622   [0.6526, 0.6717]     +0.0132
    max_cam      0.6761   [0.6664, 0.6855]     -0.0007

### The measure is NOT a range proxy

range_only -- coverage alone, a pure function of distance -- scores **0.5920**,
which is BELOW mask_camera's 0.6215. Range alone does not even beat the
baseline, so it cannot explain the +0.0527. Earlier in this project a
grazing-angle feature was retracted for being exactly this; the concern was
legitimate and is now answered with a number rather than an argument.

### Occlusion is the engine

Removing it costs **0.0903**, the largest drop by a wide margin. The nine
failed occlusion attempts were spent on the component that actually carries
the result.

### Two simplifications the ablation demands

**no_cov and n_vis are identical to four decimals, with identical intervals.**
That is correct, not a bug: with constant trust the noisy-OR reduces to
1 - (1-T)^n, a function of the camera COUNT alone, and AUROC is invariant under
monotone transforms. A free internal consistency check.

**max_cam beats full by +0.0007, CI [-0.0008, -0.0006], excluding zero.** Taking
the single best camera is marginally BETTER than combining them by noisy-OR.
The multi-camera combination does not earn its complexity. The paper reports
this and prefers the simpler form; TRUST, already known to contribute no
discrimination, collapses out with it.

### Component budget, for the writeup

    occlusion (ray-marched visibility)   +0.0903
    resolution-weighted coverage         +0.0132
    multi-camera noisy-OR                -0.0007  (harmful, drop it)

Paper-acceptance criterion "honest ablation" is now MET.

## A16 — Selective prediction. The contribution boundary, measured. (17 Sep 2026)

Two changes from A13's failed recalibration, each from its diagnosis:

**Smooth instead of fragmented.** Scheme D fit ten independent tables (10 obs
bins x 50 conf bins); its sparse cells generalised badly, which is why it lost
to a single global map. Replaced by ONE logistic model with observability as a
continuous feature -- four parameters, fit by weighted IRLS on dev cells:

    P(correct) = sigmoid(a + b*logit(conf) + c*obs + d*obs*logit(conf))

This WORKS as a fix. ECE, full volume: global logistic 0.03263 ->
+observability 0.02319, a 28.9% reduction. Conditioning now helps rather than
hurts. The fragmentation diagnosis was correct.

**Selective prediction instead of ECE.** ECE after recalibration had almost no
headroom (residual 0.0047), so it was the wrong target. The AV question is
which voxels to distrust given an abstention budget -- a RANKING question, which
is what H2 already showed observability is good at.

### Results, 6,019 frames, 150 scenes, 75 dev / 75 test

    AURC (lower better)            full volume    surfaces
      confidence alone               0.41180      0.27065
      conf + mask_camera             0.34330      0.21899
      conf + observability           0.34087      0.21755

    Scene bootstrap, dev fit fixed, 2,000 resamples over 75 test scenes:
      AURC(conf) - AURC(obs) = +0.07089  CI [+0.05994, +0.08127]   REAL
      AURC(mask) - AURC(obs) = +0.00239  CI [-0.00292, +0.00829]   NO DIFFERENCE
      (surfaces:              +0.00143  CI [-0.00110, +0.00399]   NO DIFFERENCE)

### The contribution boundary

**Visibility information is worth +0.071 AURC over confidence alone** -- a 17%
improvement in selective prediction, interval far from zero. Knowing what the
cameras could see genuinely helps decide where to distrust the model.

**Knowing HOW WELL they saw it adds nothing measurable on top of that.** The
0.7% point margin over mask_camera is noise; both CIs span zero.

This sits in real tension with H2, where observability ALONE beats mask_camera
ALONE by +0.0527 with a solid interval. Both are true, and the resolution is
the finding: the continuous measure is a strong STANDALONE error predictor and
a WEAK COMPLEMENT to the model's own confidence, which already encodes much of
the same information. The paper states this rather than leaving a reviewer to
find it.

### Third false banner of the session

selective.py printed "observability HELPS the decision" by comparing point
estimates with no interval. Same failure mode as A13's two: a pass/fail banner
evaluated without uncertainty, wrong in the optimistic direction. **Rule: no
accept/reject banner may be printed from point estimates. Every one requires
its interval.**

### Final standing

    H2 (standalone ranking)          PASS   +0.0527 [+0.0455, +0.0602], replicated
    Danger zone (barely-visible)     PASS   0.288 vs 0.052, 5.5x
    Ablation (occlusion, not range)  PASS   occlusion +0.0903; range alone 0.5920
    Base model reproduction          PASS   mIoU 37.66
    Visibility aids selection        PASS   +0.071 AURC [+0.060, +0.081]
    Continuous beats binary downstream  NO   CI spans zero
    H1 (monotone)                    FAIL   pre-registered, stays failed
    Calibration headline             WITHDRAWN (composition)
    Recalibration vs global          FIXED by smoothing (+28.9%), but ties mask

## A17 — Sensor-only regime. The downstream result PASSES, scoped. (17 Sep 2026)

Pre-registered before running: observability alone must beat mask_camera alone
at the abstention decision. Motivation: A16 handed the model's confidence to
every method, where confidence already encodes much of what visibility knows.
But sensor-geometry gating -- placement studies, fleet coverage, safety-case
arguments -- happens WITHOUT model confidence, and that is the regime the
measure is meant for. Same comparison H2 made as a ranking, made here as a
decision.

    SENSOR-ONLY, AURC, scene bootstrap over 75 test scenes
      full volume     mask - obs = -0.00550  CI [-0.01108, +0.00040]  no difference
      surfaces only   mask - obs = +0.04301  CI [+0.03557, +0.05033]  OBSERVABILITY WINS

**On occupied voxels, observability alone beats the binary mask by 0.043 AURC**,
interval clear of zero. Same direction and regime as H2's +0.0527 ranking
result, computed by a different metric. Over the full volume, dominated by
free space, there is no difference.

**The claim is scoped to occupied voxels and must be stated that way.** Both
scopes have been run side by side all session, so this is not a scope picked
after seeing which won; the full-volume null is reported with it.

### The contribution boundary, now complete and measured

1. WITH model confidence: a binary flag suffices; the continuous measure adds
   nothing (A16, CI spans zero in both scopes).
2. WITHOUT model confidence, on occupied voxels: the continuous measure beats
   the binary flag, +0.043 AURC (A17).
3. Visibility information of either kind is worth +0.071 AURC over confidence
   alone (A16).

This is a better result than the original recalibration target, because it says
WHEN the method helps rather than asserting that it always does.

### Outstanding defect

selective.py still prints "observability HELPS the decision" from point
estimates with no interval -- the banner flaw A16 wrote a rule against. It is
WRONG on the full-volume run. Fix before any further use of that script.

## A18 — Full-split mIoU: 38.90 against a published 39.1. (18 Sep 2026)

The 37.66 in A12 was a 200-frame sample. Scored on all 6,019 frames:

    mIoU 38.90   (NVlabs publishes 39.1 for this checkpoint)

**Gap 0.2.** The discrepancy was sampling noise, not a systematic effect from
the dropped auxiliary head or from test-time augmentation. Per-class values
line up with the benchmark: driveable 80.25, free 89.16, car 48.99,
sidewalk 49.53, terrain 54.94, barrier 44.07.

**37.66 is superseded. Quote 38.90.** A12's other numbers are unaffected --
every result there is an internal comparison over the same predictions, and
none of them depends on the absolute mIoU.

This also settles the reproduction criterion more strongly than before: the
pipeline does not merely land in the right range, it reproduces a published
result to within two tenths of a point.

### Quotable set, current

    mIoU                      38.90            (published 39.1)
    H2, ranking               +0.0527  [+0.0455, +0.0602]  replicated
    Sensor-only, occupied     +0.0430  [+0.0356, +0.0503]
    Visibility vs confidence  +0.0709  [+0.0599, +0.0813]
    Danger zone               0.288 vs 0.052   (5.5x)
    Ablation, occlusion       +0.0903          range alone 0.5920

---

## A19 -- per-frame visualiser (2026-09-18)

Decided **after** seeing data. Presentation only. **No new claim, no new
metric, no change to any frozen number.** Recorded here so that nobody later
mistakes a figure for a result.

`scripts/viz/render_frames.py` renders one 1600x900 figure per keyframe:

    A  forward camera strip  (CAM_FRONT_LEFT / CAM_FRONT / CAM_FRONT_RIGHT)
    B  predicted occupancy, BEV, obstacles over ground and static structure
    C  camera observability of the topmost occupied voxel per column
    D  the same obstacles recoloured by how well the cameras saw them

Sources, all read, none recomputed:
`data/preds/preds_voxel/<token>.npz` (cls, conf), `data/pack/obs_ray/<token>.npy`,
`data/occ3d/.../gts/<scene>/<token>/labels.npz` (semantics, mask_camera).

**Grid orientation was verified, not assumed.** index0 = x forward,
index1 = y left. Checked three ways on scene-0553 frame 0: CAM_FRONT optical
axis in ego is [1.00, 0.01, 0.01]; the nearest GT car sits at x=-0.2, y=+2.6
and appears immediately left of the ego in CAM_FRONT_LEFT; the GT truck at
x=-10.2..-3.8, y=+2.2..4.6 is behind-left, and the bicycle at x=14.6, y=-5.0
is ahead-right in CAM_FRONT. BEV transform is therefore `a[::-1, ::-1]`.
A first attempt used `rot90(a, 1)` and was wrong by 90 degrees.

**The accuracy printed on each frame** is per-voxel class agreement with
Occ3D on voxels that are GT-occupied AND inside `mask_camera`, split by
observability band. It is a per-frame diagnostic, not a benchmark figure; the
split-level numbers in A12 and the quotable set remain the only quotable ones.

An earlier draft printed the accuracy of the *drawn surface voxel* against GT
at the same height and reported ~2.8%. That figure was meaningless -- the
topmost predicted voxel and the topmost GT voxel rarely share a height -- and
was discarded rather than published. Noted because it is exactly the kind of
number that could have ended up on a slide.

Thresholds on panel D (`--lo 0.35` barely-observed band, `--conf 0.70`
asserted) are display choices. They were picked to make the band visible, they
are exposed as flags, and no claim rests on them.

Rendered: scene-0553, scene-0103, scene-0916, scene-0796 -- the four val
scenes for which all six camera images are present locally. 162 frames.
Outputs under `outputs/viz/` (gitignored).

---

## A20 -- the surf_zero gate is RESOLVED. The gate was wrong, not the measure. (18 Sep 2026)

Decided **after** seeing data, as an audit of a standing failure. This closes
the last open acceptance gate. It adds no new headline number.

### The standing failure

Since A11 the gate has read: of the ground-truth occupied voxels that Occ3D
marks camera-visible, fewer than 25% should get observability exactly 0. The
ray marcher gives **70.4%**. A11 and A12 both recorded this as failed and
flagged it as "defensible as a definitional difference, NOT yet verified".
It has been carried as an open failure for two sessions. Verifying it is
overdue.

### The test

`scripts/eval/verify_surf_zero.py`. It does not use the marcher at all --
different geometry, different code path, so a shared bug cannot hide. For each
zeroed voxel it walks the straight segment from each camera's optical centre to
that voxel and asks the ground-truth occupancy volume whether anything sits
strictly in between, stopping 1.05 voxels short so a voxel can never occlude
itself. 420 samples per segment, which is under 0.175 m spacing at the longest
range present -- finer than the 0.4 m voxel, so a blocker cannot be stepped
over.

Each zeroed voxel lands in exactly one bucket:

    out_of_fov    no camera has it inside the image at positive depth
    occluded      inside at least one image, and EVERY such camera is blocked
                  by a ground-truth occupied voxel before reaching it
    unexplained   some camera has a clear line of sight  ->  a real bug

### Result. 60 frames, 290,897 zeroed voxels tested.

    out of every field of view      14,389    4.95%
    occluded by GT geometry        276,331   94.99%
    UNEXPLAINED (clear line)           177    0.06%

    unexplained, range p50 36.0 m, p90 46.4 m

**99.94% of the zeroing is geometrically correct.** The residual 0.06% sits at
long range, which is exactly where 0.4 m sampling can occasionally thread a gap
between two voxels that the marcher's one-voxel step did not; it is a sampling
artefact at the measure's resolution limit, not a systematic error.

### Verdict

The gate was written against the wrong quantity. It compared a strict first-hit
measure to `mask_camera`, and `mask_camera` is not a first-hit mask -- it marks
a voxel visible on looser terms. A camera sees the front face of an object, not
its interior and not its far side, so a measure that zeroes the back of every
car and the inside of every building is behaving correctly. 70.4% is the right
answer to the question the measure asks.

**The gate is retired, not passed.** It is replaced by the test above, which is
the question it was trying to ask:

    new gate: fewer than 1% of zeroed GT-occupied voxels inside mask_camera
    may have a clear line of sight from any camera.        MEASURED 0.06%. PASS.

No frozen number changes. H2, the ablation, the sensor-only regime and the
zero-bin split were all computed on these maps and are unaffected -- this
audit confirms the maps, it does not alter them.

### What this does to the paper

The 70.4% figure stops being an apology and becomes a statement of what the
measure is: it is the first quantity in this literature that reports the back
of a car as unseen. That is the reason it separates from `mask_camera` at all.

---

## A21 -- single-camera failure exposure. NEW RESULT. (18 Sep 2026)

Decided **after** seeing data. Not pre-registered. Reported as an exploratory
deployment measurement, not as a confirmatory hypothesis test.

### What it is, and what it is NOT

**It is a coverage study.** The noisy-OR is recomputed over five cameras
instead of six, so every voxel whose only camera evidence came from the failed
unit drops to observability exactly zero.

**It is NOT a robustness study.** The model was not re-run with a camera
removed. There is one published FB-OCC r50 checkpoint and its predictions are
fixed. Any sentence of the form "accuracy drops when the camera fails" is NOT
supported by this experiment and must not be written.

What it does support: *which part of the scene the stack would be asserting
with no camera evidence at all, and how good the model was in exactly that
region while the camera still worked.*

`scripts/eval/camera_dropout.py`. One ray march per camera per frame, per-camera
factors kept rather than collapsed, so all seven configurations come out of one
pass. Evaluated on voxels that are GT-occupied, inside `mask_camera`, and had
observability > 0 with all six cameras. 450 frames, 143 scenes, chunked and
cached. Intervals are a paired scene bootstrap, 4,000 resamples.

### Result

    camera failed     field dark  obstacles dark   acc dark   acc kept   gap (kept - dark)
    CAM_BACK              26.88%          30.20%      69.1%      71.4%   +2.3 [+1.3, +3.4]
    CAM_FRONT             19.59%          22.09%      72.6%      70.4%   -2.2 [-3.5, -0.9]
    CAM_FRONT_RIGHT       12.44%          10.82%      67.9%      71.2%   +3.3 [+1.5, +5.1]
    CAM_BACK_LEFT         11.28%           9.45%      74.1%      70.4%   -3.7 [-5.5, -1.9]
    CAM_FRONT_LEFT        11.07%          11.88%      72.4%      70.6%   -1.7 [-3.5, +0.2]
    CAM_BACK_RIGHT        10.69%           8.28%      70.0%      70.9%   +0.9 [-1.2, +3.1]

### Two findings

**1. Single-camera redundancy on this rig is uneven by 3.6x.** Losing CAM_BACK
takes 30.20% of obstacle voxels to zero camera evidence; losing CAM_BACK_RIGHT
takes 8.28%. The ordering tracks unique angular coverage, measured from the
intrinsics rather than assumed: CAM_BACK has an 89.3 degree horizontal field of
view, every other camera 64.3 to 65.0. Five 65-degree cameras cannot close a
360 degree ring, and the back camera is alone in the gap.

It is not traffic asymmetry. Obstacle voxels are near-symmetric front to rear,
53.7% ahead of the ego against 46.3% behind, over 120 frames. The exposure
asymmetry is geometry.

**2. The exposure is not offset by model quality.** Across all six cameras the
accuracy gap between the uniquely-covered region and the rest is between -3.7
and +3.3 points, and it changes sign. The region a camera uniquely covers is
neither a region the model is unusually good at nor one it is unusually bad at.

Finding 2 is the useful negative. There is no hidden margin: you cannot argue
that the fragile sector is one the model happens to handle well. Redundancy
planning on this rig has to be driven by geometry, and the front/back axis is
where it is thin.

### Limits, to appear wherever this is quoted

* One checkpoint, one rig. This is a statement about the nuScenes camera
  geometry and this model, not about camera rigs in general.
* `mask_camera` was itself built with all six cameras, so the evaluation set is
  defined by the intact rig. That is the right control for "what did I lose"
  and the wrong one for "what would I have evaluated on".
* 450 frames of 6,019. Scene-level intervals are given; the point estimates
  would move by tenths on the full split, not by points.
* Exploratory. Not in the A5 pre-registration.

---

## A22 -- Stage 4. Observability predicts MISSED OBJECTS, and my prediction was wrong. (18 Sep 2026)

Decided **after** seeing data. Exploratory, not in the A5 pre-registration.
This is the step that connects a voxel metric to something a planner would
drive into.

### Setup

`scripts/eval/missed_detection.py`. For each ground-truth dynamic object
(car, truck, bus, trailer, construction vehicle, motorcycle, bicycle,
pedestrian) with at least one LiDAR point:

    missed entirely    no voxel inside the box carries ANY predicted class
    no dynamic class   voxels are placed, but none of a class that moves
    obs                max camera observability over the GT-occupied voxels
                       inside that box

570 frames, 8,551 objects, scene-paired bootstrap, 3,000 resamples.

**The global-to-ego transform was verified before use, not assumed.** Sweeping
a z offset, agreement between box interiors and same-class ground truth peaks
at 100.0% at -0.4 m and is 99.3% at 0.0 m -- inside one voxel. 0.0 was used,
because the convention implies it and choosing -0.4 post hoc to buy three boxes
out of 442 is precisely the kind of fudge this document exists to prevent.

### Result

    observability band          objects   missed entirely   no dynamic class
    obs = 0 (no evidence)         1,773            12.24%             48.34%
    obs 0.000 - 0.176             1,325            10.26%             25.66%
    obs 0.176 - 0.345             1,373             3.35%             14.79%
    obs 0.345 - 0.686             1,360             2.35%             11.99%
    obs 0.686 - 0.796               278             1.80%              6.83%
    obs 0.796 - 1.000             2,442             1.64%              5.28%

    no evidence vs best seen   +43.05 pts  [+39.64, +46.52]
    barely seen vs best seen   +20.38 pts  [+17.70, +23.21]
    barely seen vs no evidence -22.68 pts  [-26.99, -18.70]

**A 9.2x spread, monotone across all six bands.** Where the cameras have no
evidence, the model fails to place anything that moves half the time.

### MY PREDICTION FAILED, and it is instructive

The script's own docstring predicted, from the A14 danger-zone result, that
misses would PEAK in the barely-seen band rather than at zero. They do not.
Misses are worst at zero evidence and fall monotonically. Recorded as a failed
prediction, not quietly deleted.

The failure is the finding. The two effects are different failure modes:

* **No camera evidence -> the object is ABSENT.** 48.3% of the time nothing
  dynamic is placed at all. The planner has no obstacle to avoid.
* **Barely seen -> the object is PRESENT but the model is overconfident about
  it.** This is where the calibration gap peaks at 0.288 (A14) while the miss
  rate has already dropped to 25.7%.

One measure separates a *missing* obstacle from a *misjudged* one. That is the
"so what" this project has been missing, and it is a sentence a safety engineer
can act on.

Note also: the monotone structure H1 pre-registered and failed to find at the
voxel level is present at the OBJECT level. That is an observation made after
the fact and is NOT a rescue of H1. H1 stays failed.

### The controls a reviewer will demand. All four run.

Predicting "no dynamic class was placed here", per object, AUROC:

    observability (ours)                    0.7650
    LiDAR points in box  (ORACLE)           0.7671
    nuScenes annotator visibility           0.6974
    range from ego alone                    0.6607

    observability inside fixed range shells, range held constant
      0-10 m    n=  930   0.7810
     10-20 m    n=2,441   0.7405
     20-30 m    n=2,432   0.7314
     30-55 m    n=2,745   0.7181

* **Not a range proxy.** Range alone scores 0.6607; observability holds 0.72 to
  0.78 inside every fixed range shell. A15 showed this for voxel errors; it now
  holds for object misses.
* **It beats the dataset's own human visibility label** by +0.068. A geometric,
  camera-only quantity predicts model failure better than the four-bucket
  judgement a human annotator recorded.
* **It ties a LiDAR oracle.** Counting the actual LiDAR returns inside the box
  scores 0.7671 against our 0.7650 -- a gap of 0.0021. The camera-only measure
  is within a fifth of a point of a score that requires the sensor we refuse
  to use.

### Limits

570 of 6,019 frames. Dynamic classes only. Detection here means "the occupancy
field contains something", which is weaker than a detector's notion of a true
positive; a box-level IoU criterion would be stricter and is not run.

---

## A23 -- the formula is settled: MAX single camera, no TRUST constant. (18 Sep 2026)

Decided **after** seeing data, as a direct response to A15, where the project's
own ablation scored `max_cam` above the formula the project uses. A paper whose
ablation argues against its own method is a paper a reviewer takes apart.

`scripts/eval/formula_decision.py`. Five scores accumulated in ONE march per
frame, so they are perfectly paired. 675 frames, **145 of 150 scenes**,
paired scene bootstrap.

    score        AUROC     vs mask_camera            95% CI
    full        0.6748           +0.0568   [+0.0458, +0.0678]
    max_cam     0.6755           +0.0575   [+0.0465, +0.0685]
    max_noT     0.6755           +0.0575   [+0.0465, +0.0685]
    max_novis   0.5908           -0.0272   [-0.0375, -0.0160]
    mask_cam    0.6180

    max_cam - full        +0.0007   [+0.0006, +0.0008]
    max_noT - max_cam     +0.0000   [+0.0000, +0.0000]
    max_cam - max_novis   +0.0847   [+0.0740, +0.0953]

### Decision

**The measure becomes** `observability(v) = max_i ( coverage_i(v) * visible_i(v) )`.

Three things are established, not argued:

1. **The noisy-OR is dropped.** Max beats it by +0.0007 with the interval
   excluding zero, replicating A15's +0.0007 on an independent sample. The
   effect is tiny, but the direction is reliable and the simpler formula is the
   one that wins. Combining six cameras multiplicatively was modelling
   redundancy that does not exist: a voxel is seen by the camera that sees it
   best, and the others add noise.
2. **TRUST = 0.795 is removed, and the removal is EXACTLY free.**
   `max_noT - max_cam = +0.0000` with a bootstrap interval of [0.0000, 0.0000].
   Under a max, a positive constant is a monotone rescale and cannot change a
   ranking. This was predicted analytically and then measured rather than
   asserted. The measure now has **one** parameter, `FULL_PX`, inside the
   coverage term.
3. **Occlusion is still the engine.** Removing it costs 0.0847 and drops the
   score below `mask_camera`. Consistent with A15's 0.0903.

### Bug found and fixed during this run, same class as A2

The chunked cache took the head of an index-sorted sample, so the first partial
report covered **23 scenes**, not 150, and its scene bootstrap was meaningless
(it gave +0.0630 with a CI three times too wide). The sample is now shuffled
deterministically before slicing. A2 recorded this exact failure for
`--frames`; it reappeared in new code. Noted so the next chunked script is
written with it in mind.

### What this does NOT yet change

Every frozen full-split number -- H2 +0.0527, the ablation, the sensor-only
regime, the zero-bin split, A21, A22 -- was computed on noisy-OR maps.
**They stay quotable as they are, under the noisy-OR, and are labelled as such.**
Switching the headline to max requires rebuilding all 6,019 maps under the new
formula and re-running H2, the selective study and the kill gate on them. That
is roughly two hours of CPU and no money.

The expected movement is +0.0007, i.e. H2 goes from +0.0527 to about +0.0534.
Nothing about the paper's argument depends on it. **The reason to do it is
that the method section should describe the formula that was actually used,
and that formula should be the one with fewer parameters.**

    STATUS: decision made, full-split rebuild PENDING.
    Command:  python3 scripts/eval/build_observability_ray.py \
                  --out data/pack/obs_max --measure max
    (the --measure flag does not exist yet; it is one branch in the camera loop)

### The switch is narrower than it looks, and this was checked

`--measure max` is implemented in `build_observability_ray.py` and verified on
three frames against the existing maps:

    max mean 0.4821   noisy-OR mean 0.3905   over 154,563 nonzero cells
    zero pattern identical: TRUE

**A voxel is zero under max exactly when it is zero under the noisy-OR** -- both
are zero iff no camera sees it, which is a property of the visibility term, not
of the combination rule. So every result that turns on the zero set is
*invariant* to this change:

* A14, the danger-zone split into out-of-view and occluded
* A21, which voxels go dark when a camera fails
* A22, the obs = 0 row (48.34% no dynamic class) and the whole monotone ordering

Only the RANKING among nonzero cells moves, by the measured +0.0007. The
rebuild is therefore about the method section reading correctly, not about any
headline being at risk.

---

## A24 -- H1's root cause found. It was scored on the wrong population. (18 Sep 2026)

Decided **after** seeing data. **H1 REMAINS FAILED.** H1 was pre-registered over
all voxels inside `mask_camera`, it was tested that way, and it lost. Nothing
below changes that. What follows is a post-hoc diagnosis, labelled exploratory,
and it is not a rescue.

`scripts/eval/root_cause_h1.py`. 900 frames, 149 scenes, 78,652,377 voxels.
Reads the stored maps; no marching. Decile edges frozen on the first 60 frames
before any gap was computed.

### The table

    bin          ALL voxels    GT FREE   GT OCCUPIED    %free        voxels
    obs = 0          0.1005     0.0570        0.1655    59.9%    27,133,227
    decile 1         0.0415     0.0197        0.2874    91.9%     3,092,390
    decile 2         0.0545     0.0317        0.2276    88.4%     2,990,094
    decile 3         0.0597     0.0364        0.2259    87.7%     3,319,313
    decile 4         0.0588     0.0360        0.2127    87.1%     3,733,355
    decile 5         0.0596     0.0365        0.2059    86.4%     4,156,255
    decile 6         0.0546     0.0304        0.2044    86.1%     4,811,496
    decile 7         0.0495     0.0246        0.1999    85.8%     5,657,188
    decile 8         0.0454     0.0195        0.1994    85.6%     2,953,116
    decile 9              --         --            --       --             0
    decile 10        0.0337     0.0086        0.1781    85.2%    20,805,943

    rank correlation with decile, paired scene bootstrap x3000
      ALL voxels        rho -0.418  [-0.630, -0.161]
      GT FREE only      rho -0.520  [-0.700, -0.281]
      GT OCCUPIED only  rho -0.865  [-0.924, -0.772]

### The finding

**On ground-truth OCCUPIED voxels the pre-registered decline holds exactly.**
All nine live deciles are strictly decreasing, 0.2874 down to 0.1781, rho
-0.865. Not approximately monotone -- every consecutive pair.

**On ground-truth FREE voxels it is a hump**: 0.0197 up to 0.0365 at decile 5,
then down to 0.0086.

Free space is 85-92% of every decile, so the aggregate follows the free
stratum, and that is why H1 failed. H1 was a claim about how well the model
knows what is THERE, scored on a population that is overwhelmingly what is NOT
there.

### MY OWN PREDICTION IN THIS SCRIPT WAS WRONG

The docstring predicted, before printing, that the hump came from COMPOSITION:
%free falling across deciles mixing an easy stratum out. That is falsified.
%free moves only 91.9% to 85.2%, far too little to produce the hump, and the
hump sits INSIDE the free stratum itself. The cause is not mixing. The cause is
that the hypothesis is simply false for free space and true for occupied space.
Recorded as a failed prediction.

Why free space humps is not established. A plausible mechanism -- at near-zero
observability free space is trivially free and confidently so; a little
evidence brings surface confusion; a lot resolves it -- is a HYPOTHESIS and is
not tested here. It is not to be written as if it were measured.

### Two things that fall out of the same table

1. **The danger zone appears again, cleanly, on occupied voxels.** Blind
   (obs = 0) sits at 0.1655 while barely-seen (decile 1) sits at 0.2874. Same
   direction as A14, on a different stratification and a different sample.
2. **Decile 9 is EMPTY, and that is an artefact of TRUST.** The constant 0.795
   caps the measure, a large mass lands on exactly 0.7961, and two quantile
   edges coincide. An empty bin silently contributed a spurious 0.0 to the
   first monotonicity statistic (it read -0.561 / -0.655 / -0.681 before the bin
   was dropped). This is an independent argument for A23: under
   `max_i(cov_i * vis_i)` there is no TRUST constant and no saturation mass, so
   the deciles cannot collapse this way.

### Reporting rule

Quote as: "H1, as pre-registered over all voxels, failed. A post-hoc
stratification shows the predicted monotone decline holds on ground-truth
occupied voxels (rho -0.865, nine of nine deciles strictly decreasing) and does
not hold on free space." Never quote the occupied column as if H1 passed.

---

## A25 -- temporal observability. The broad claim FAILS, the narrow one is strong. (18 Sep 2026)

Decided **after** seeing data. Exploratory. New measurement axis, camera-only,
no re-inference.

### Motivation

Every observability number in this project is present tense. But FB-OCC r50 runs
with `do_history=True` over 16 frames, so a voxel occluded now may have been in
plain view two seconds ago and the model is entitled to remember it. Present-
tense observability calls both cases blind; they are not the same case.

`scripts/eval/temporal_observability.py`. Past maps are warped into the current
ego frame through the recorded ego poses, so the question is asked about a place
in the world, not an index in a grid. 8 keyframes of history = 4 s at 2 Hz.
tau = 0.10 counts as "seen". 240 frames, 1,733,259 sampled voxels.

    age(v) = seconds since ANY camera last had observability > tau at the
             world location this voxel currently occupies

### H-T, as stated in the script before the numbers existed

"Error rises with age, AND age carries information that present-tense
observability does not." **The second half FAILS.**

    error rate by age          all voxels   occupied only
    seen right now (59.5%)          8.02%          28.84%
    0.5 s                          11.71%          26.58%
    1.0 s                          11.39%          26.67%
    1.5 - 2.0 s                    12.87%          27.32%
    2.5 - 4.0 s                    15.67%          26.95%
    never in 4 s (15.5%)           23.74%          31.03%

    AUROC for a wrong voxel
      model confidence (inverted)       0.8806
      present observability (inverted)  0.6319
      age since last seen               0.6277

    age INSIDE fixed bands of present observability
      obs = 0        n=598,466   0.5812
      obs 0 - 0.25   n=328,339   0.4680
      obs 0.25 - 0.6 n=273,276   0.5000   DEGENERATE
      obs > 0.6      n=533,178   0.5000   DEGENERATE

The last two bands are 0.5000 **by construction, not by measurement**: age is
defined as 0 whenever present observability exceeds tau, so it is constant
there. That is a definitional artefact and must never be quoted as a result. In
the one non-degenerate band above zero, obs 0-0.25, age scores 0.4680 -- BELOW
chance. Age is not a general-purpose second axis.

### The narrow claim, which is strong

Inside the blind set, where present observability is exactly 0 and age is the
ONLY evidence axis available:

    age since last seen        voxels    error rate   occupied only
    0.5 s                     109,341        13.63%          25.66%
    1.0 s                      69,809        14.06%          25.28%
    1.5 - 2.0 s                83,627        14.86%          26.33%
    2.5 - 4.0 s                72,425        16.43%          26.39%
    never in 4 s              262,567        23.83%          30.97%

    blind AND never seen  minus  blind but seen within 4 s
        +9.20 pts   [+8.08, +10.34]   paired scene bootstrap

**The blind set is not one population, it is two.** A cell the cameras cannot
see now but saw two seconds ago behaves almost like a seen cell (13.6% error).
A cell no camera has seen in four seconds is where the model actually fails
(23.8%). This also measures something nobody in this project had measured: the
model's temporal memory demonstrably works, and its benefit decays with age.

### The deployable number

**13.95% of all voxels the model asserts at p >= 0.70 have not been seen by any
camera in the last 4 seconds. Their error rate is 17.43% against 7.25% for the
rest.** That is a fleet-monitorable quantity computable from geometry and ego
pose alone, with no ground truth and no LiDAR.

### Limits

240 frames, voxels subsampled 1 in 13. tau = 0.10 is a choice and the split
point between "seen" and "not" moves with it; it was fixed once, before the
run, and not tuned. 4 s horizon is set by MAXH = 8 and is shorter than the
model's own 16-frame window in wall-clock terms only if keyframes are 2 Hz,
which they are. Age is computed against keyframes, so sub-500 ms resolution is
not available.

---

## A26 -- full split rebuilt under the A23 formula. Prediction confirmed. (18 Sep 2026)

All 6,019 maps rebuilt with `--measure max`, i.e.
`observability(v) = max_i ( coverage_i(v) * visible_i(v) )`. 3.7 GB,
`data/pack/obs_max`, meta records `measure: max`. One parameter, `FULL_PX`.

### H2 on the full split, new formula

    observability   AUROC 0.6748   [0.6655, 0.6839]
    mask_camera     AUROC 0.6215   [0.6136, 0.6295]
    difference          +0.0534   [+0.0462, +0.0609]
    inside mask_camera      0.6373

A23 predicted "+0.0527 -> about +0.0534" before the rebuild ran. Measured
**+0.0534**. The prediction is confirmed to the fourth decimal, which is a
check on the A23 reasoning, not a new claim.

### The kill gate, non-free voxels, zero bin split

2,600 frames, 150 scenes, 52,418,690 non-free voxels.

    bin                    n          conf      acc       gap
    out of view    1,460,180        0.9712   0.9178   +0.0534
    occluded      30,515,543        0.8717   0.6971   +0.1745
    decile 1         793,194        0.8713   0.5773   +0.2941
    decile 2       1,019,997        0.8851   0.6641   +0.2210
    decile 3       1,103,403        0.8802   0.6594   +0.2208
    decile 4       1,242,425        0.8863   0.6732   +0.2131
    decile 5       1,525,334        0.8931   0.6852   +0.2079
    decile 6       1,777,344        0.8990   0.6915   +0.2075
    decile 7       2,324,627        0.9039   0.7002   +0.2038
    decile 8       1,856,379        0.9084   0.7062   +0.2022
    decile 9       8,800,264        0.9266   0.7485   +0.1781

    gap(obs=0) - gap(obs>0) = -0.0300  [-0.0360, -0.0240]

Three things, all of which were predicted in earlier amendments and are here
confirmed rather than discovered:

1. **The deciles are strictly monotone decreasing, nine of nine**, 0.2941 down
   to 0.1781. A24 predicted exactly this on occupied voxels. It now reproduces
   in the kill-gate pipeline itself, on 52.4M voxels, under the new formula.
   **This still does NOT rescue H1.** H1 was pre-registered over ALL voxels
   inside `mask_camera` and it failed there. Non-free-only is a post-hoc
   stratification and is reported as exploratory, exactly as A24 requires.
2. **The decile collapse is gone.** Under the noisy-OR a mass sat at exactly
   0.7961 (the TRUST cap) and two quantile edges coincided, emptying decile 9.
   The new edges run to 1.000 and all nine deciles are populated. A24 predicted
   the removal of TRUST would fix this. It did.
3. **The danger zone survives the formula change**: barely seen +0.2941 against
   out of view +0.0534, a factor of 5.5, matching A14's 0.288 / 0.052.
   The composition-corrected headline is -0.0300, replicating A13's -0.0293.

### Quotable set, current (full split, 6,019 frames, 150 scenes)

    mIoU                             38.90        (published 39.1)
    H2, ranking                     +0.0534  [+0.0462, +0.0609]   max formula
    H2, ranking (noisy-OR)          +0.0527  [+0.0455, +0.0602]   superseded
    Danger zone, non-free           0.2941 vs 0.0534  (5.5x)
    Calibration, composition-corr.  -0.0300  [-0.0360, -0.0240]
    Missed objects, nothing dynamic 48.3% -> 5.3%, monotone (A22)
    Object miss vs LiDAR oracle     0.7650 vs 0.7671 (A22)
    Blind set split by staleness    +9.20 pts [+8.08, +10.34] (A25)
    Camera dropout, obstacles dark  30.2% CAM_BACK ... 8.3% CAM_BACK_RIGHT (A21)

### STILL PENDING

`selective.py` on `obs_max` has not been run on the full split -- the scan is
about 200 s and exceeds the remote shell's limit. Command:

    python3 scripts/eval/selective.py --obs data/pack/obs_max --boot 2000 \
        --out outputs/artifacts/selective_max.json

Until it runs, the sensor-only AURC numbers (+0.0430, +0.0709) stay quoted from
the noisy-OR maps and are labelled as such.

---

## A27 -- recalibration REOPENED with staleness. Improved, still FAILS. (18 Sep 2026)

Decided **after** seeing data. `scripts/eval/recalibrate_v2.py`.

A13 closed Stage 5: observability-conditioned recalibration lost to a global map,
and the smooth replacement only tied `mask_camera`. It is reopened for one
specific reason: A25 showed the blind set is two populations (13.63% error if
seen 0.5 s ago, 23.83% if never seen in 4 s), and every earlier scheme was blind
to that split because age did not exist as a feature. Observability assigns both
groups exactly 0.

Platt maps on l = logit(confidence), fitted on 61 dev scenes, scored on 61
held-out scenes, 917,612 test voxels. **Acceptance was written into the script
before any number printed: E must beat A, B AND C.**

    scheme                ECE     vs C      Brier   ECE|blind   ECE|seen
    A  none           0.06321  +1639.1%   0.08330     0.09674    0.04529
    B  global         0.00525    +44.3%   0.07415     0.01162    0.00449
    C  mask_camera    0.00363      0.0%   0.07413     0.00521    0.00522
    D  observability  0.00411    +13.1%   0.07414     0.00658    0.00453
    E  + staleness    0.00336     -7.5%   0.07414     0.00554    0.00470

    E vs A  -0.05985  [-0.06354, -0.05310]   E WINS
    E vs B  -0.00188  [-0.00209, -0.00045]   E WINS
    E vs D  -0.00075  [-0.00085, -0.00000]   E WINS
    E vs C  -0.00027  [-0.00041, +0.00009]   TIE

### Verdict: ACCEPTANCE NOT MET. Recorded as a failure.

Staleness is a real improvement over observability alone -- E beats D with the
interval excluding zero, which is the first time any conditioning scheme has
moved. But E only TIES the binary `mask_camera` baseline, which is what the
acceptance criterion forbade. Stage 5 stays closed. It is not reframed as a
win, and the 7.5% point estimate is not to be quoted without the interval that
crosses zero.

### What this settles for the paper

Three schemes have now tied `mask_camera` on calibration while observability
beats it by +0.0534 on RANKING. That is not a contradiction, it is the
boundary of the contribution, and it should be stated as one:

**Observability answers WHICH cells are likely wrong. It does not improve HOW
WRONG the model says it is.** Ranking and calibration are different tasks; this
measure wins the first and does not win the second. Claiming both would be
false, and a reviewer who runs the calibration check would find it.

---

## A28 -- hard-case mining. Two signals work, four are ANTI-correlated. (19 Sep 2026)

Decided **after** seeing data. Exploratory. `scripts/mining/mine.py`.

The question a data engine answers is not "where was the model wrong" -- that
needs labels, and fleet data has none. It is whether an UNLABELLED signal,
computable from geometry and the model's own output, finds the frames where the
model is wrong. So the tool is built in two halves that never touch:

    MINING SIGNALS   predictions + observability only. No ground truth.
    VALIDATION       per-frame error and calibration gap from Occ3D, used ONLY
                     to score the signals. Never an input to one.

6,019 frames indexed into SQLite plus an 84-dimension descriptor per frame
(class histogram, observability histogram, obstacle range and azimuth profile,
height profile, confidence profile). Retrieval is an exact inner product:
6,019 x 84 floats is 2 MB and sub-millisecond. An ANN index would take longer
to build than to skip; FAISS earns its place at millions of vectors, not
thousands, and adding it here would be resume-driven engineering.

### Result. 5,817 frames with >= 200 obstacle columns.

Target: the worst 10% of frames. Two targets, because they are not the same
question -- A14 established that error rate and calibration gap differ.

    signal              AUROC(wrong)  AUROC(overconfident)   lift@100
    mean_margin               0.8618              0.8150       7.6x / 6.9x
    low_margin_rate           0.8192              0.7709       2.5x / 2.2x
    low_margin                0.8118              0.7495       4.2x / 3.0x
    mean_obs                  0.8066              0.7807       5.9x / 6.0x
    blind_commit              0.5322              0.5002       0.2x / 0.5x
    obst_dark_rate            0.4293              0.4489       2.4x / 2.7x
    blind_commit_rate         0.3753              0.3997       0.0x / 0.1x
    obst_dark                 0.1964              0.2201       0.0x / 0.0x
    dim_commit                0.1921              0.2136       0.0x / 0.3x
    obst_n                    0.1864              0.2104       0.0x / 0.0x
    dim_commit_rate           0.1698              0.2065       0.0x / 0.3x
    random                    0.5000              0.5000       1.0x

**Two signals work.** Mean top1-top2 margin (inverted) and mean observability.
Best precision@100 is 69% against a 10% base rate, a 6.9x lift.

### TWO PREDICTIONS OF MINE FAILED. Both are recorded.

**Prediction 1: counts are confounded by scene density; normalising will fix
them.** FALSE. `blind_commit_rate` 0.3753 and `dim_commit_rate` 0.1698 are no
better than the raw counts. Normalisation was not the problem.

**Prediction 2: `dim_commit` will win on the OVERCONFIDENT target, because A14
showed barely-seen voxels carry the largest calibration gap.** FALSE. It scores
0.2136 there -- still strongly anti-correlated.

### What the failures mean, stated as a limit and not as a save

**A per-voxel finding does not become a frame-level signal by summing it.**
A14 is about the calibration of individual voxels and it holds. Aggregating it
into a per-frame count produces something that tracks how much confident
occupied volume a scene contains, and dense confident scenes are the ones the
model handles well. Four signals built that way select EASY frames, reliably
enough to be useful inverted.

The signals that survive are intensive and per-voxel-averaged, not extensive.
That is the transferable lesson and it is the one to state in the paper: the
danger-zone result is a statement about voxels and must not be quoted as a
frame-selection rule.

### Retrieval works across scenes, which is the point

Nearest neighbours of the worst-observed frame (scene-0921, err 52.7%) are
scene-0921 at 61.5%, then **scene-0272** at 39.9%, **scene-0272** 38.8%,
**scene-0920** 37.0%, **scene-0015** 47.4% -- all well above the 10% base rate
and drawn from four different scenes. The descriptor finds analogous
situations across the fleet rather than adjacent frames in time.

### Committed

`mining_validation.json` and `mined_mean_obs_500.json`. The SQLite store and the
descriptor matrix are 2.1 MB each, rebuildable in four minutes, and stay out of
git -- derived state does not belong there.

---

## A23 CORRECTION -- "exactly inert" was an overclaim. (19 Sep 2026)

A23 stated that removing the TRUST constant under a max is **exactly** free,
quoting `max_noT - max_cam = +0.0000` with a bootstrap interval of
`[0.0000, 0.0000]`.

The stored value is **5e-6**, not 0. The printed table rounded to four decimals
and the amendment repeated the rounded figure as if it were exact.

The reasoning was still right: under a max, a positive constant is a monotone
rescale and cannot change a ranking. The residual is a **uint8 quantisation
artefact** -- the AUROC is computed on a 256-bin histogram, and rescaling moves
a handful of values across bin edges.

**Corrected wording:** TRUST is inert to within the measure's quantisation,
5e-6, which is four orders of magnitude below the effect being measured. Not
"exactly zero".

Found by the CI gate in `scripts/ci/check_observability_gates.py`, which asserts
the value is below 1e-4, on its first run. Recorded because a gate that catches
the author is the only evidence that the gate is not decoration.

---

## A29 -- blind-spot attribution. Vehicles are 17% of what you cannot see and 45% of the road you cannot see. (19 Sep 2026)

Decided **after** seeing data. Exploratory. `scripts/eval/blind_attribution.py`.
Week 7 figure, pulled forward.

Every occlusion result up to here says a cell is unseen. None says WHO made it
unseen. The same ray march answers it: a ray stops at its first occupied voxel,
so every cell that ray would have reached afterwards is hidden BY that voxel.
Rays no longer terminate; the first hit is recorded as the occluder and the
remainder of the ray is credited to it. Cost is about 2x the terminating march.
Ground truth supplies the occluder's CLASS only; the geometry is the same march
that produced every frozen map, so this is consistent with A12-A28 by
construction.

400 frames, 169,468,739 occluded voxels attributed.

### Result, structures and objects only

    occluder        hidden volume   share    hidden drivable   share
    manmade            74,320,658   48.3%       93.1 m2/frame   32.3%
    vegetation         47,200,523   30.7%       34.9 m2/frame   12.1%
    car                17,606,247   11.5%      106.6 m2/frame   37.0%
    truck               5,951,473    3.9%       18.5 m2/frame    6.4%
    pedestrian          2,647,132    1.7%        8.5 m2/frame    2.9%
    bus                 2,026,584    1.3%        3.0 m2/frame    1.0%
    barrier             1,166,753    0.8%        6.7 m2/frame    2.3%

    all vehicles       17.3% of hidden VOLUME     45.4% of hidden DRIVABLE SURFACE
    worst single occluder in a frame is manmade in 42% of frames

### The finding

**Buildings dominate the hidden volume; vehicles dominate the hidden road.**
Structures account for nearly half of everything a camera-only stack cannot
see, but most of that volume is sky, facade and interior -- space no vehicle
will ever occupy. Cars are 11.5% of blind volume and **37.0% of blind drivable
surface**: roughly three times more road hidden per unit of blind volume,
because a car sits on the road at eye level and a building does not.

The one-line version, which no occupancy paper currently states:

    vehicles are 17% of what a camera-only stack cannot see,
    and 45% of the ROAD it cannot see.

That is the number a rig designer and a safety case both want, and it is only
computable because the measure is per-voxel and occlusion-aware.

### LIMITATION, stated before the headline and not after it

The raw table, before ground classes are separated out, credits **47.8% of
hidden drivable surface to `driveable` itself** -- the road as its own
occluder. That is a **ray-sampling artefact, not geometry**. At range the pixel
grid under-samples the ground plane, some road cells fall between rays at
stride 4 and are reached by none, and the attribution then credits them to the
nearest road cell a ray did hit.

Cameras do lose angular resolution with range and that part is real -- the
coverage term already models it -- but calling the road its own occluder is
misleading, so ground classes (11-14) are reported separately and excluded from
the headline. A finer stride would shrink the artefact and would not change the
object-occluder ordering, which depends on solid angle rather than sampling.

This is A10's lesson in a new costume: in a voxel world, a grazing surface
looks like an occluder. It was caught here because 47.8% for the road was too
large to be geometry.

---

## A30 -- the safety envelope. Can the cameras see far enough to stop? (19 Sep 2026)

Decided **after** seeing data. Exploratory, deployment-facing.
`scripts/eval/safety_envelope.py`. Full split, 5,869 frames with a measurable
speed. No ground truth used at all.

A benchmark asks how much of the scene was labelled correctly. A safety case
asks how far ahead the stack has POSITIVELY VERIFIED the road is clear, and
whether that is further than the distance it needs to stop. "Verified" means
both halves: predicted free AND carrying camera evidence. A cell the model calls
free with no camera on it is a guess, and a guess is not a clearance.

Corridor 2.8 m wide, envelope 0.6 to 3.4 m above ground, obs >= 0.15 counts as
observed, braking 4.0 m/s^2, d_stop = v^2 / 2a.

    quantity                             p10    median     p90
    near field unverifiable (m)          1.2       1.2     12.8
    verified-free reach (m)             10.8      26.4     36.8
    speed (m/s)                          0.0       5.6      9.8
    stopping distance needed (m)         0.0       4.0     11.9

    frames where reach < d_stop            2.7%   (160 of 5,869)
    frames with no verified corridor       0.3%
    corridor ends on an obstacle            95%
    corridor ends on missing evidence        5%

    by speed          frames   med reach   med d_stop   flagged
    0-2 m/s            1,371        25.6          0.0      0.0%
    2-5 m/s            1,197        20.8          1.8      0.1%
    5-10 m/s           2,804        29.6          6.8      3.4%
    10-15 m/s            497        31.2         14.8     13.1%

**13.1% of frames above 10 m/s have less camera-verified clear road ahead than
the distance the vehicle needs to stop.** One number, trackable per release,
computable with no labels.

### Two bugs found on the way, both from tuning discipline rather than luck

**1. The envelope started inside the road.** `clear_from = 0.4 m` puts the first
z level at k=3, and the measured height profile shows k=0,1,2 are 100% occupied
(that is the road) and k=3 is still 54% road bleed. So the test was asking the
ROAD to be free and reported **no verified corridor in 51.5% of frames**. The
envelope now runs k=4..10, z = +0.6 to +3.4 m: above the surface, below
overhead structure. Fixed after measuring the profile, not by sweeping the
parameter until the answer improved.

**2. First run instead of longest run.** With the envelope fixed, the median
reach was still only 2.4 m while the corridor is ~95% verified out to 22 m.
Cause: the run was taken from the first verified cell, which sits in the
near-field band where coverage is marginal, so it broke immediately and never
restarted. `render_hd.py` had already made this exact correction and it was not
carried across. Longest run gives median 26.4 m.

Both were caught by asking why a number was implausible, which is the rule this
document exists to enforce. Neither was caught by a gate.

### Limit, stated with the headline

The corridor ends on an obstacle 95% of the time, which is correct behaviour --
you stop before the vehicle in front. So `reach < d_stop` is a **screening**
statistic, not a violation count: many flagged frames are ordinary car
following at a legal gap. What it is good for is tracking the tail across
releases, and for finding the frames where the ego is closing on something it
cannot positively see past.

---

## A31 -- time to visibility. Occlusion with a clock on it. (19 Sep 2026)

Decided **after** seeing data. Exploratory. New axis, camera-only, no
re-inference. `scripts/eval/time_to_visibility.py`.

Every occlusion number so far is a snapshot. A planner cannot act on "this cell
is unseen"; it can act on how long that lasts, because the ego's own motion
resolves most occlusions without anyone doing anything. Future observability
maps are warped into the current ego frame through the recorded poses, so the
question is about a place in the world, not an index in a grid. 8 keyframes of
look-ahead = 4.0 s at 2 Hz. 600 frames, 260,954,085 blind voxels.

Crucially the "on the ego's path" set is computed from the **actual future
trajectory**, not from a straight-ahead assumption: a blind cell counts if a
future ego pose puts the vehicle box on it.

    time to visibility        all blind   share    on ego path   share
    within 0.5 s             15,552,480    6.0%         86,747   17.6%
    by 1.0 s                 10,352,880    4.0%        106,770   21.7%
    by 2.0 s                 13,281,949    5.1%         93,744   19.0%
    by 3.0 s                  8,434,347    3.2%         31,369    6.4%
    by 4.0 s                  5,983,825    2.3%         18,177    3.7%
    STILL BLIND after 4 s   207,348,604   79.5%        155,628   31.6%

    per-frame median: 4.50 s over all blind cells
                      1.90 s over cells the ego drives through

### The finding

**The blind volume splits cleanly by whether the ego is going there.** 79.5% of
everything unseen is still unseen four seconds later -- that is building
interiors, the far side of walls, space behind parked cars the ego never
enters, and it does not matter. Restricted to cells the ego's own trajectory
passes through, the picture inverts: **58.3% is revealed within two seconds**
and the median is 1.90 s.

But **31.6% of the ego's own path that is blind now is still blind four seconds
later.** That is the number worth reporting. It is not resolved by driving; it
is space the vehicle enters without ever having seen it.

That sentence is a planning input, not a perception metric, and it is the form
in which an occlusion measure becomes usable in a safety case: not "x% of the
scene is occluded" but "x% of where I am about to be, I will never have seen".

### Limits

600 frames of 6,019. 4 s horizon is set by the 8-keyframe look-ahead at 2 Hz;
sub-500 ms resolution is not available. tau = 0.10 was fixed before the run and
not tuned. The ego box test is a 4.8 x 2.8 x 2.5 m volume around each future
pose, which is the vehicle's own footprint and not a planned corridor with
margin.

---

## A32 -- where does the 7th camera go? Nowhere useful. (23 Sep 2026)

Decided **after** seeing data. Exploratory. Completes Week 4's missing half and
Week 7's placement figure. `scripts/eval/camera_placement.py`.

A21 asked what the rig loses when a camera fails. This asks the design
question: given the observability field over real driving, where would an
ADDITIONAL camera buy the most? The measure makes this answerable analytically
-- a candidate camera is just another term in the max, so place it, march it,
recombine. No retraining, no simulator. You cannot ask a trained occupancy
network what a camera it has never seen would contribute; you can ask the
geometry.

164 frames. Baseline: **287,248 of 373,345 obstacle voxels (76.94%) have no
camera coverage at all**, mean observability on obstacles 0.1550.

    add a 7th camera   obstacles recovered   road recovered   mean obs gain
    roof_high  90 deg        6,200   2.2%     52,381   6.8%        +0.0076
    rear_tele  45 deg        3,977   1.4%     58,998   7.7%        +0.0287
    front_wide 120 deg       2,281   0.8%     31,191   4.1%        +0.0024
    bumper_low 100 deg       2,225   0.8%     15,078   2.0%        +0.0020
    side_right  90 deg       1,306   0.5%     11,689   1.5%        +0.0020
    side_left   90 deg       1,136   0.4%      9,064   1.2%        +0.0020

### The finding, and it is a negative one worth more than a positive

**Adding a seventh camera anywhere recovers at most 2.2% of the obstacle voxels
the six-camera rig cannot see.** The best placement by road recovered
(rear_tele, 7.7%) is a telephoto pointed backwards, which is a narrow fix for a
specific gap rather than a rig improvement.

The reason is the point: **this rig is not coverage-limited, it is
occlusion-limited.** The 76.94% of obstacle voxels with no evidence are behind
things, not outside anyone's field of view. A29 already said what those things
are -- structures 48.3% of hidden volume, vehicles 45.4% of hidden drivable
surface. More lenses do not see through a parked truck.

### Where this leaves the three studies together

    A21  losing one camera costs 8.3% to 30.2% of obstacles
    A29  what makes cells blind is objects, not field-of-view gaps
    A32  adding a camera buys back at most 2.2%
    A31  but 58.3% of the blind volume ON THE EGO'S PATH clears within 2 s

So the rig is well designed for coverage, the residual blindness is physical,
and the thing that actually resolves it is **motion plus memory**, not
hardware. For a camera-only stack the return on a seventh camera is near zero
and the return on knowing what you cannot see is large. That is the argument
this whole project exists to make, and it now has a hardware-side number.

### Limits

164 frames, six hand-chosen placements rather than a dense sweep -- so this
bounds the gain, it does not find the optimum. Candidate intrinsics are ideal
pinholes at 1600x900; a real lens has distortion and a real mount has
occlusion by the vehicle itself, both of which would lower the gains further,
not raise them.

---

## A33 -- selective prediction on the max maps. A RESULT CHANGED. (23 Sep 2026)

Decided **after** seeing data. `scripts/eval/selective.py --obs data/pack/obs_max`.
2,000 frames, 150 scenes, dev/test split by scene, 400 scene bootstrap
resamples, dev fit fixed.

This was the last analysis still quoted from the superseded noisy-OR maps.
Re-running it on the A23/A26 maps did not merely move a decimal.

    AURC, lower is better                      value
    confidence alone                         0.39278 (implied)
    conf + mask_camera                       0.34418 (+12.4% vs conf alone)
    conf + observability                     0.32797 (+16.5% vs conf alone)
    observability alone                      0.38708
    mask_camera alone                        0.39235

    AURC(mask) - AURC(obs), with confidence  +0.01606  [+0.01041, +0.02169]
    AURC(conf) - AURC(obs)                   +0.06481  [+0.05502, +0.07538]
    sensor-only, mask - obs                  +0.00502  [-0.00088, +0.01093]

    ECE                                        value
    raw confidence                           0.39830
    global logistic                          0.03182
    + mask_camera                            0.02335
    + observability (smooth)                 0.02829

### What changed, stated plainly

On the noisy-OR maps the reported position was: **with** model confidence in
hand the binary flag is sufficient and the continuous measure adds nothing (CI
spanned zero), while the continuous measure won in the **sensor-only** regime
(+0.0430 on occupied voxels).

On the max maps that **inverts**:

* **With confidence, observability now BEATS mask_camera, +0.01606, CI excludes
  zero.** The old null is gone.
* **Sensor-only, the difference is now +0.00502 with the CI spanning zero.** The
  old win is gone.

### This is NOT yet quotable, and here is why

Three things differ from the frozen A12-era run at once: the measure (max
instead of noisy-OR), the sample (2,000 frames instead of 6,019), and for the
sensor-only line the scope (full volume here, occupied voxels there). A result
that reverses under three simultaneous changes has not been isolated to any of
them.

    ACTION REQUIRED before either number is used:
      python3 scripts/eval/selective.py --obs data/pack/obs_max --boot 2000
      python3 scripts/eval/selective.py --obs data/pack/obs_max --nonfree --boot 2000
    on the FULL split. The scan is ~380 s and exceeds the remote shell's limit,
    so it runs locally.

Until then the paper quotes neither, and the sensor-only claim in the Field
Brief (+0.0430) stays labelled as measured on the superseded maps.

### What did NOT change, and it matters

**mask_camera still wins on ECE** -- 0.02335 against 0.02829. Three separate
attempts (A13, A27, and now this) have failed to beat the binary flag on
calibration while the continuous measure wins on ranking. That boundary is now
the most replicated statement in the project:

    observability answers WHICH cells are likely wrong.
    it does not improve HOW WRONG the model says it is.

---

## A34 -- recalibration: ROOT CAUSE FOUND. The target was wrong, not the model. (23 Sep 2026)

Decided **after** seeing data. `scripts/eval/recal_root_cause.py`. Full split,
**3,852,160,000 voxels**, 150 scenes, held out 75, exact sufficient statistics
binned by (predicted class, confidence, observability, mask_camera). No
sampling.

Three attempts had tied or lost to `mask_camera` on calibration (A13, A27,
A33). Each fixed a modelling problem and each still failed. This asks the
question none of them asked: **conditional on the predicted class and the
model's own confidence, does accuracy still depend on observability?**

    model for p(correct)        log-loss        ECE    vs baseline
    class + confidence          0.159283   0.002177
    + mask_camera               0.106443   0.001536       +33.17%
    + observability             0.153435   0.001780        +3.67%
    + both                      0.101388   0.001195       +36.35%

    scene bootstrap, log-loss reduction
    observability vs class+confidence   +0.005820  [+0.005400, +0.006258]  WINS
    observability vs mask_camera        -0.046941  [-0.049693, -0.043659]  LOSES
    both         vs mask_camera         +0.005069  [+0.004592, +0.005484]  WINS

    model-free, inside a fixed (class, confidence) cell, 157 cells
      mean accuracy spread across observability   0.0995
      mean slope                                 +0.0618 per unit observability
      cells where accuracy rises with it          85.1% by voxel weight

### Three findings, and together they close the question

**1. The signal is real.** Conditional on class AND confidence, accuracy still
moves 0.0995 across observability with a +0.0618 slope, and observability beats
the class+confidence baseline with the interval excluding zero. Every earlier
attempt that concluded "no signal" was wrong about that.

**2. It loses to mask_camera by a wide margin, not a tie.** -0.046941, and the
interval is nowhere near zero. A13, A27 and A33 all reported a tie; on the full
population it is a clear loss. The tie was an artefact of evaluating INSIDE
mask_camera, where the flag is constant and cannot show its advantage.

**3. But it adds on top of mask_camera.** "Both" beats mask_camera alone,
+0.005069, interval excludes zero. The two are not measuring the same thing.

### THE ROOT CAUSE

**`mask_camera` is not a competing visibility measure. It is a label-validity
flag.** Occ3D builds it during dataset construction from the full sensor suite
to mark where its own ground truth is trustworthy. Outside it, the labels are
themselves unreliable, so "correct" is partly noise -- and a flag that predicts
where the labels are noisy will beat any perception signal at predicting
apparent error.

Three attempts were therefore aimed at the wrong target. Beating `mask_camera`
on calibration was never the right bar, because part of what it predicts is
whether the benchmark can be evaluated at all.

**And it does not exist at deployment.** `mask_camera` is computed offline,
once, from LiDAR and the full rig. A vehicle on the road has observability and
does not have `mask_camera`. So the operationally meaningful comparison is
observability against class and confidence -- **which observability wins** --
and the fact that it still adds on top of `mask_camera` proves the two carry
different information.

### Status change

    Stage 5 (recalibration) moves from FAILED to RESOLVED, with a corrected
    claim rather than a rescued one:

      observability improves calibration over a class-and-confidence baseline,
      by 3.67% log-loss and 18.2% ECE, on 3.85 billion voxels.
      It does not beat mask_camera, which is a label-validity flag unavailable
      at inference, and it adds information on top of it.

The old claim -- "observability-conditioned recalibration fails" -- is
**withdrawn as mis-specified**, not as wrong: it was true against the target it
chose, and that target was the wrong one.

### A trap walked into again, and caught

The first version of this script filtered to voxels inside `mask_camera`, which
makes `mask_camera` **constant** in the sample -- so the baseline the previous
three attempts lost to could not even be expressed, and the script would have
reported "observability helps" against a baseline missing its only real
competitor. A13 recorded this exact trap ("--scope mask made the mask grouping
constant"). It was walked into a second time and caught only because the first
result looked too good. The file now keeps all voxels and carries a comment
naming the trap.

---

## A35 -- the marcher is 3.46x faster, and the answer is bit-identical. (23 Sep 2026)

Decided **after** seeing data. Stage 7, performance. `scripts/perf/march_bench.py`.

`build_observability_ray.py` has carried this line in its own docstring since
A11:

    "Affordable because grid and cameras are both ego-fixed, so sample indices
     are identical every frame: precompute once, then gather and any() per
     frame."

**That optimisation was described and never implemented.** The shipped marcher
recomputes every ray direction and every voxel index on every frame, for every
camera, for all 6,019 frames.

The precondition is in fact stronger than the docstring claims. Measured across
all 150 scenes: camera extrinsics and intrinsics are **constant within a scene**
(max variation exactly 0.0) and the whole validation split contains only **two
distinct rigs**. So the ray-to-voxel index table can be built twice and reused
6,019 times.

    table[ray, step] -> flat voxel index, plus a validity mask
    per frame: gather occupancy, find each ray's first hit, mark up to and
               including it. No trigonometry, no projection, no per-step
               index arithmetic.

### Result

    verification            0 differing voxels out of 30,720,000 compared
    shipped                 1.139 s / frame
    precomputed table       0.329 s / frame
    speedup                 3.46x
    table build             6.91 s once, 832 MB resident for 12 camera tables
    full 6,019-frame build  114 min  ->  33 min, table build included

**Bit-identical, not approximately identical.** A speedup that changes the
answer is worthless, so the benchmark refuses to print a timing number until
the equality check passes. Every frozen observability map in A12-A34 can be
reproduced by the fast path without re-deriving any result.

### Cost, stated with the win

832 MB of resident tables for 12 cameras. That is the trade: memory for
arithmetic. It is affordable on a laptop and would not be on an embedded
target, where the per-frame projection is the right implementation. The fast
path is for offline fleet-scale processing, which is what this pipeline is.

### Why it matters beyond the clock

Every experiment in this project that needed a rebuild -- A23's formula
decision, A26's full-split rebuild -- cost about two hours. At 33 minutes the
rebuild stops being a decision and becomes a step. A24's decile collapse and
A33's three-way confound both went unexamined longer than they should have
partly because re-running was expensive.

---

## A36 -- the renderer was showing a corridor computed with the bug A30 fixed. (23 Sep 2026)

Presentation fix, no new claim, recorded because it is the third instance of the
same failure mode and the pattern now matters more than the instance.

`render_hd.py` computes a verified-free corridor per frame and draws it in green.
It shipped with `clear_from = 0.4 m`, which A30 later measured to be **inside the
road surface** -- levels k=0,1,2 are 100% occupied by the road and k=3 is still
54% road bleed, so the test was asking the road to be free. A30 corrected the
envelope to k=4..10 (z = +0.6 to +3.4 m) in `safety_envelope.py` and **the fix
was never carried back to the renderer.**

Effect on what was displayed, scene-0916 frame 21:

    before  corridor 20.4 - 23.2 m   (2.8 m of verified clearance)
    after   corridor  3.2 - 40.0 m   (36.8 m)

The published median from A30 is 26.4 m, so the videos were understating the
corridor by roughly an order of magnitude. All 162 frames across four scenes
were re-rendered.

### The pattern, stated once

Three times now a correction has been made in one file and not carried to
another that computes the same quantity:

1. A30: longest-run vs first-run. `render_hd.py` had it right; `safety_envelope.py`
   did not.
2. A30 again: the height envelope. `safety_envelope.py` got it right; `render_hd.py`
   did not -- the same two files, in the opposite direction.
3. A34: the mask_camera scoping trap, recorded in A13 and walked into again.

Each was caught by a number looking implausible, never by a test. The corridor
computation now exists in two places with the same constants and no shared
implementation, which is the actual defect. **Action for the writeup phase:
`verified_free` moves to one module and both callers import it.** Until then
the constants are duplicated with a comment in each naming the other file.

---

## A37 -- selective prediction, FULL SPLIT. A33's flag lifted, and A33 was partly my error. (24 Sep 2026)

A33 recorded the selective-prediction result as **NOT QUOTABLE** and named a
required action: run it on the full split. That action is done, and the reason
it had not been done was itself a defect worth naming.

### Root cause of the delay, and the fix

The scan was single-threaded at ~0.063 s/frame, so 6,019 frames took ~380 s --
longer than the 180 s a remote shell call allows. A33 was therefore run on a
2,000-frame subsample and flagged. The scan is a pure per-frame histogram
accumulation with no cross-frame state, so it parallelises exactly.
`selective.py` now takes `--jobs`; **verified identical output on 300 frames
with one worker and with many** before the full run. 380 s becomes 91 s.

This is the root fix for "the full split cannot be run", not a workaround.

### Result. 6,019 frames, 150 scenes, 75 dev / 75 test, 2,000 scene bootstrap.

    scope          regime            AURC(mask) - AURC(obs)          verdict
    full volume    with confidence   +0.01544 [+0.01028, +0.02098]   obs WINS
    full volume    sensor-only       +0.00551 [-0.00040, +0.01220]   tie
    occupied only  with confidence   -0.00327 [-0.00485, -0.00174]   mask WINS
    occupied only  sensor-only       +0.04215 [+0.03475, +0.04970]   obs WINS

    ECE, full volume   + mask_camera 0.02384   + observability 0.02664
    ECE, occupied      + mask_camera 0.02221   + observability 0.02378

### A33 WAS PARTLY WRONG, AND IT WAS MY ERROR

A33 stated that "the old +0.0430 sensor-only win is gone", citing +0.00502 with
an interval spanning zero. **That compared the wrong things.** The frozen
+0.0430 was measured on OCCUPIED voxels; the +0.00502 I set against it was the
FULL VOLUME line. Different populations.

On the matching scope the original claim is confirmed, not lost:

    frozen (noisy-OR maps, occupied)   +0.0430  [+0.0356, +0.0503]
    now    (max maps, occupied)        +0.04215 [+0.03475, +0.04970]

A33 listed "the scope" as one of three simultaneous changes and then failed to
control for it in its own comparison. Recorded as an error of mine, not a
finding.

### What actually changed, stated correctly

**One thing changed and one thing was confirmed.**

* **Changed:** on the full volume WITH confidence, observability now beats
  mask_camera (+0.01544, interval excludes zero) where the noisy-OR maps gave a
  null. The 2,000-frame A33 estimate (+0.01606) and the full split (+0.01544)
  agree, so the reversal is real and is attributable to the A23 formula change.
* **Confirmed:** the sensor-only advantage on occupied voxels, +0.04215 against
  a frozen +0.0430.

### The shape this leaves, and it is the paper's scoping statement

    on OCCUPIED voxels -- the cells a planner cares about --
      without model confidence, observability wins decisively (+0.04215)
      with model confidence, the binary flag is slightly better (-0.00327)

    on the FULL VOLUME -- dominated by free space --
      with confidence, observability wins (+0.01544)
      without, they tie

And across every scope, `mask_camera` remains better on ECE. That is the same
boundary A34 established from a different direction: **observability answers
WHICH cells are likely wrong; it does not improve HOW WRONG the model says it
is.** Four independent analyses now agree on that sentence.

### Status

A33's NOT QUOTABLE flag is **lifted**. Both numbers are full-split and may be
used, with their scope stated every time -- the scope is the whole finding.
The Field Brief's +0.0430 stands and becomes +0.04215 on the max maps.

---

## A38 -- One corridor implementation, fourteen regression tests, and the bug
## they found on their first run

**Date:** 2026-09-24. **Decided:** after seeing data, on the dev split and on
synthetic scenes. **Affects:** A30 (safety envelope) and A36 (HD renders).
**Supersedes the A30 figures below; the A30 method is unchanged.**

### The root cause, first

Three separate bugs in this project share one cause, and it is not arithmetic:

| # | Bug | Caught by |
|---|---|---|
| A30 | `clear_from=0.4` bled the road surface into the corridor | an implausible 51.5% no-corridor rate |
| A30 | first-run instead of longest-run gave a 2.4 m median | an implausible median |
| A36 | the `clear_from` fix never reached `render_hd.py` | a render that disagreed with the table by 10x |

Each time, the same physical quantity -- the verified-free corridor -- was
computed by two files, a fix landed in one, and nothing checked that the other
agreed. Every one of the three was caught by a number that looked wrong to a
human. That is not a control. It is luck, and it only works on quantities whose
plausible range I happen to know.

**Fix:** `scripts/eval/corridor.py` is now the ONLY implementation.
`safety_envelope.py` and `render_hd.py` both import `verified_free` from it.
`tests/test_observability_geometry.py` contains a test,
`test_one_corridor_implementation`, that compares the `__code__` objects of the
function each module holds and fails if a second copy ever appears.

### The tests

Fourteen, all synthetic -- no dataset, no checkpoint, they run in 1.2 s in CI.
Each exists because a real bug got through:

    test_run_beyond_an_obstacle_is_not_credited      A38 (below)
    test_longest_run_not_first                       A30 bug 2
    test_envelope_starts_above_the_road_surface      A30 bug 1
    test_envelope_has_an_upper_bound                 overhead structure
    test_one_corridor_implementation                 A36
    test_road_surface_alone_does_not_block           A30 bug 1, converse
    test_unobserved_lane_yields_no_corridor          the evidence half
    test_marcher_is_blocked_by_a_wall                first-hit semantics
    test_marcher_marks_the_wall_itself               off-by-one at the hit
    test_fast_path_is_bit_identical                  A35
    (+4 supporting)

### The bug they found

`longest_run` returned the longest contiguous verified-free run **anywhere** in
the 40 m lane. On a synthetic scene with a wall at 8 m and clear observed road
behind it, it returned a corridor starting at 18.8 m. A clear stretch on the FAR
SIDE of an obstacle is not clearance the ego has. The published metric was
crediting exactly that whenever a near obstacle was followed by open road.

`MAX_START = 15.0` m now bounds where a credited run may begin. The bound is not
free-floating: A30 measured the near-blind distribution at median 1.2 m and p90
8.4 m, so 15 m sits well beyond any legitimate start and only excludes runs that
begin past an obstacle. The p90 near-blind figure itself falls from 12.8 m to
8.4 m under the bound, which is the same bug seen from the other side: those
long "near-blind" stretches were the distance to an obstacle, not to the start
of vision.

**This is the first bug in this project found by a test rather than by a number
that looked wrong.** It is the only one of the four that I would not have
noticed, because a 26 m median reach is exactly what I expected to see.

### Effect on the A30 figures -- they get worse

Full split, 5,869 frames, re-run after the fix:

    quantity                          p10    median      p90
    near field unverifiable (m)       1.2       1.2       8.4      p90 12.8 -> 8.4
    verified-free reach (m)          10.0      26.4      36.8      p10 10.8 -> 10.0

    frames reach < d_stop        2.7%  ->  4.3%   (160 -> 252 of 5,869)
    frames with no corridor      0.27% ->  0.5%
    corridor ends on an obstacle  95%  ->  96%

    by speed, flagged      2-5 m/s   0.1%  (was 0.1%)
                          5-10 m/s   4.6%  (was 3.4%)
                         10-15 m/s  24.5%  (was 13.1%)

The headline sentence nearly doubles: **24.5%, not 13.1%, of frames above
10 m/s have less camera-verified clear road ahead than the vehicle needs to
stop.** The error ran in the safe-looking direction -- the metric was
understating the exposure it exists to measure -- which is the direction a
safety metric must never be wrong in, and the reason this amendment exists
rather than a quiet edit.

Superseded everywhere: `docs/SOTIF_VALIDATION_REPORT.md` (rows 4.8 and the
envelope table), the Field Brief, and any slide carrying 2.7% / 13.1%.

### What did NOT change

The A30 method, its parameters (`tau=0.15`, 2.8 m corridor, 0.6-3.4 m envelope,
`occ_frac=0.8`, `BUMPER=2.4`), and the sealed test split. `MAX_START` is a
correction to an implementation that did not match the written definition
("the contiguous verified-free run beyond the near-blind zone"), not a new
parameter choice tuned against a result. The definition always said "beyond";
the code did not enforce it.

---

## A39 -- A verification image, and what was and was not verified about it

**Date:** 2026-09-24. **Affects:** reproducibility claims only. No measured
number changes.

`Dockerfile` + `docker/verify.sh` build one image that runs every check this
repository can run **without a GPU and without the nuScenes export**:

    ctest cpp/build          C++ runtime primitives, release
    ctest cpp/build-tsan     the same, under ThreadSanitizer
    pytest tests/            57 tests, including the 14 from A38
    check_gates.py           release gates on the committed detection artifact
    check_gates.py (inverted) must FAIL on the known-bad v11 baseline
    check_observability_gates.py            21 gates
    check_observability_gates.py --self-test each gate fed a broken artifact

The image is built from the committed tree (`git archive HEAD | docker build
-t odfm -`), so it is a function of a commit hash. `.dockerignore` reproduces
that from a dirty tree.

### What it deliberately cannot do

It does not reproduce the measurements. The observability maps need the 26 GB
packed export, which is not redistributable, and FB-OCC needs a GPU. Both are
stated in the Dockerfile header with the mount that makes the eval scripts run
against a local copy. An image that implied otherwise would be a worse artifact
than none.

### Verification status -- read this before quoting it

**The checks were run and pass. The container was not built.**

Docker Hub is unreachable from this session's sandbox (`registry-1.docker.io`
returns 403 through the egress proxy), so `docker build` could not be executed.
What was executed instead, on a clean `git archive HEAD` extract under
Python 3.11.15:

    pytest tests/ --collect-only        57 collected
    pytest tests/                       57 passed
    cmake + ctest, release              3/3 passed
    cmake + ctest, TSan                 4/4 passed
    docker/verify.sh gates              exit 0, 21 PASS, all self-tests OK
    sh -n docker/verify.sh              syntax clean

So the *contents* of the image are verified against the committed tree; the
`FROM`/`COPY`/`pip` layering is not. `docker build` on a machine with registry
access is an outstanding one-command check. `verify.sh` now honours
`ODFM_HOME`, which is how the above was run outside a container and is how it
can be re-run without Docker at all.

This distinction is recorded rather than glossed because "we have a Dockerfile"
and "the image builds and passes" are different claims, and only the second one
is worth anything to a reviewer.

---

## A40 -- audit: the most-quoted number was printed, never stored

**Date:** 2026-09-24. **Affects:** provenance only. No value changes.

A plan-versus-delivered audit of all 39 amendments checked every quoted figure
against the artifact the script actually wrote. One failed.

The sensor-only contrast -- `AURC(mask_camera alone) - AURC(observability
alone)`, the A17/A37 headline and the number the paper leans on hardest --
was **printed to stdout and never written to `selective_max_*.json`**. It could
be quoted only from a terminal log, which is exactly the provenance this
project refuses everywhere else.

`selective.py` now stores `boot_margin_sensor_mask_minus_obs`. Both full-split
runs were re-executed and reproduce the quoted values exactly:

    occupied / non-free   +0.042151  [+0.034755, +0.049705]
    full volume           +0.005512  [-0.000405, +0.012200]

Every other figure in A1-A39 was traced to a committed artifact and matched.

The lesson is the same one A38 records from the other direction: a number is
not verified because it was measured, it is verified because it can be read
back. Printing is not storing.

---

## A41 -- root causes behind the recorded failures, and what each one actually
## fixes

**Date:** 2026-09-24. **Decided:** after seeing data, except the two
pre-registrations in section 6, whose window is open because the data they
judge does not exist yet.

The instruction behind this amendment was to stop recording failures and find
their causes. Four of the five had a fixable cause. One did not, and saying so
is part of the work: **a hypothesis that was pre-registered and lost stays
lost.** No verdict below is reversed by re-specifying it after the fact.

---

### 1. Stage 5 recalibration -- the target was 86% unsupervised, and the
### baseline was mislabelled

Two separate defects, both real, and together they change the verdict.

**Defect one: the baseline in `recalibrate_v2.py` was never `mask_camera`.**
The scheme table calls row C "mask_camera" and A27 quotes it that way. The code
computes `m = (obs > 0)`. That script never loads Occ3D's mask at all. So A27's
"only ties the binary mask_camera baseline" was really **"adds nothing over its
own binarisation"** -- true, materially weaker, and it has been quoted wrongly
in the Field Brief and in three amendments. The row is renamed in code and the
artifact re-run; no value moves, only the name and what it licenses.

**Defect two, the substantive one: the evaluation target.** Error is
`predicted class != Occ3D ground-truth class`, and Occ3D builds `mask_camera`
offline from the full sensor suite to mark **which voxels the label is valid
for**. A34 knew this and still scored over all voxels, departing from A5's
pre-registered scoping rule -- *"All Occ3D results are computed inside
`mask_camera` only. Outside it the training loss never supervised the model"* --
which was fixed on 17 Sep before the data existed.

The size of what that departure buys, measured:

    stratum                          voxels          share    accuracy
    mask_camera = 1  labels valid   536,229,736      13.9%      0.8842
    mask_camera = 0  unsupervised 3,315,930,264      86.1%      0.3821
                                                   difference  +0.5022

A feature that flags the valid stratum separates a population the model gets
88% right from one it gets 38% right. That is not a visibility measure winning
a fair contest. **It is a feature leaking the construction of the label**, and
it accounts for the whole of mask_camera's +0.046955 [+0.043601, +0.049932]
log-loss advantage.

`scripts/eval/recal_confound.py` re-slices A34's own sufficient statistics --
no new pass over the data -- and asks A34's question inside the pre-registered
scope, where `mask_camera` is constant and cannot compete. The comparison there
is observability against **nothing extra**, which is fully expressible, so this
is not the A13/A34 constant-mask trap.

    inside mask_camera == 1, held-out scenes, 269,451,701 voxels
      class + confidence        log-loss 0.239721   ECE 0.004100
      + observability           log-loss 0.236462   ECE 0.002837
      gain             +0.003134 log-loss  [+0.002455, +0.003901]

**The interval excludes zero, and ECE falls 30.8%.** A34's conclusion -- "there
is no residual to model, the question is closed" -- **is wrong on the scope the
project pre-registered**. Conditional on predicted class and the model's own
confidence, accuracy still depends on observability wherever the ground truth
is valid.

What this does and does not license:

* It **does** overturn A34's closure and reopen Stage 5 on valid labels.
* It **does not** retroactively pass A13, A27 or A33. Those ran a different
  model family on 240 frames and their recorded verdicts stand as run.
* It **does not** claim observability beats `mask_camera`. Inside the mask that
  comparison is undefined, and outside it the comparison is contaminated. The
  honest statement is that **`mask_camera` is not a fair competitor for this
  target at all**, and it is unavailable at deployment regardless.

---

### 2. The export gate -- it could not catch its own failure, and no threshold
### on that quantity could

The gate read "dynamic classes >= 1.0% of non-free voxels". The column collapse
it was written for measured 1.2% and passed. The instinct is to raise the
threshold; the measurement says that cannot work. On 600 frames of the
**validated** export:

    per-frame dynamic share of non-free voxels
      p1 0.036%   p5 0.150%   p50 1.635%   p100 17.779%    pooled 2.274%

An empty street legitimately scores below the broken run. The quantity's own
variance is larger than the effect the gate is trying to detect, so no single
threshold separates them. The gate was written against a class histogram that
*resembles* the failure instead of against the failure.

**The failure is structural and so is the replacement.** A column collapse
keeps exactly one voxel per BEV column by construction. On 400 frames of the
validated export:

    occupied voxels per occupied BEV column   min 3.706   median 9.447
    occupied columns holding exactly one      median 0.29%   max 15.54%
    a column collapse, by construction              1.000        100%

New gates: **>= 2.0 voxels per occupied column** (a 1.85x margin below the
worst good frame, and arithmetically unreachable by a collapse) and **<= 50% of
occupied columns holding exactly one voxel**. The dynamic-class check drops to a
weak floor at 0.2%.

Three tests in `tests/test_observability_geometry.py` pin this, including
`test_the_old_dynamic_gate_could_not_have_caught_it`, which builds a collapsed
export, shows it clearing the old 1.0% threshold, and shows the new gate
rejecting it.

---

### 3. The safety envelope -- the strongest objection to it, tested, and it
### does not hold

A38 made the flagged rate worse and the obvious objection is that the metric is
too conservative: it is single-frame, while FB-OCC fuses sixteen frames and the
vehicle is moving. A25 measured that memory is real (never-seen 23.83% error
against 13.63% for seen 0.5 s ago). So the corridor should be a **bracket**, not
a point.

`scripts/eval/corridor_temporal.py` warps **past** observability maps into the
current ego frame through the recorded poses -- past only; a deployed stack has
memory, not prophecy -- and takes the max. 4,819 frames, all with a full
eight-keyframe history.

    memory      evidence   med reach   p10    flagged   >10 m/s   no corridor
     0.0 s        77.8%       26.4     10.0      4.2%     24.1%        0.4%
     0.5 s        83.5%       27.2     11.2      3.6%     21.3%        0.2%
     1.0 s        87.2%       27.2     11.2      3.6%     21.3%        0.2%
     2.0 s        92.0%       27.2     11.2      3.6%     21.3%        0.2%
     4.0 s        94.0%       27.2     11.2      3.6%     21.3%        0.2%

Two things, and the second is the finding.

**The T = 0 row independently reproduces A38** (4.2% against 4.3%, 24.1%
against 24.5%, on a different frame subset through different code). A38's
correction is confirmed by a second implementation.

**Memory fills the corridor and the envelope does not move.** Evidence rises
from 77.8% to 94.0% of corridor columns -- the warp is working, and the
`corridor_evidence` column exists precisely so a null cannot be confused with a
no-op -- while the flagged rate above 10 m/s falls only 24.1% to 21.3% and is
flat after the first half second.

The reason is already in A38: **the corridor terminates on an obstacle 96% of
the time, not on missing evidence.** Memory adds evidence. Evidence behind an
obstacle is not clearance in front of one, and `MAX_START` correctly refuses to
credit it.

So the bracket is narrow and the pessimistic end is nearly the whole story:
**24.5% is real exposure, not single-frame conservatism.** A38's worse number
now stands because it survived the strongest objection available, rather than
because nobody raised one. This also sharpens A32: the residual blindness is
occlusion-limited, so neither a seventh camera (<= 2.2%) nor four seconds of
memory (2.8 points) buys much, and the return is on knowing what you cannot see.

Reported as an upper bound and labelled memory-perfect: a patch of road seen two
seconds ago can hold a cyclist now.

---

### 4. The mining failures -- two of the four are usable inverted, two are not

A28 recorded `blind_commit`, `dim_commit`, `obst_dark` and `obst_n` as failures
at AUROC 0.17-0.20 against WRONG, with the correct diagnosis: they measure how
much confident occupied volume a scene holds, and dense confident scenes are
the ones the model handles well. A signal that ranks the worst frames last
ranks the easiest frames first, so an EASY target was added and **measured
rather than inferred from a flipped sign**:

    target EASY -- the best 10% of frames by error rate
      dim_commit         AUROC 0.7862   prec@100 54%   5.40x
      dim_commit_rate    AUROC 0.7762   prec@100 60%   6.00x
      obst_n             AUROC 0.7251   prec@100 50%   5.00x
      obst_dark          AUROC 0.7033   prec@100 40%   4.00x
      blind_commit       AUROC 0.4449                  1.40x
      blind_commit_rate  AUROC 0.4247                  1.70x

Two of the four become a validated down-sampling and triage selector at 5-6x
lift over the base rate. **Two do not**, and that is the reason for measuring:
inverting a poor ranker for one target does not produce a good ranker for its
complement, because the two targets are different label sets rather than
complements. `blind_commit` at 0.5322 on WRONG and 0.4449 on EASY is simply
uninformative in both directions.

A28's transferable lesson is unchanged and is now stated with its exception: a
per-voxel finding does not become a frame-level signal by summing it, and the
extensive signals that result are useful only where the question is "which
frames are boring".

---

### 5. What is NOT fixed, and will not be

**H1 stays failed.** It was pre-registered over all voxels, it lost, and A24
already found the cause: free space is 85-92% of every decile so the aggregate
followed the stratum the hypothesis was not about. Re-specifying it over
occupied voxels after seeing that result would be choosing the population by
the answer, which is the single thing the pre-registration apparatus exists to
prevent. The corrected form is registered in section 6 and tested on data that
does not exist yet. Until then, H1 is reported as failed in every writeup.

**A22, A24, A28 and A25's five predictions stay failed.** A prediction is a
statement about the world that the measurement refuted; there is nothing to
repair. Their causes are understood -- two failure modes rather than one,
composition versus a genuinely false hypothesis for free space, extensive
versus intensive signals, degeneracy above the seen threshold -- and the causes
are what earn their place in the paper. A22's failure produced the project's
headline finding.

**The free-space hump remains unexplained.** A24 called it plausible but not
established and that is still true. It is a real open question, not a failure,
and it is not being dressed up as either.

---

### 6. PRE-REGISTRATION for the second backbone (written before that data exists)

The SurroundOcc export has not been run. This window closes the moment it lands.
Both entries below are corrected forms of hypotheses that failed, and the only
legitimate repair for a pre-registered failure is a correctly specified
successor tested out of sample.

**H1' (replaces H1).** On ground-truth OCCUPIED voxels inside `mask_camera`,
the calibration gap declines monotonically with observability decile.
Pre-registered direction: negative, by Spearman correlation on decile index,
scene-bootstrapped. Population chosen because the claim H1 was trying to encode
-- the model knows less where it sees less -- is a statement about cells that
contain something, and because A5 fixed the scoring scope inside the mask on
17 Sep. **Acceptance: strictly monotone on at least eight of nine live deciles
AND rank correlation <= -0.5.** FB-OCC gives -0.865; a second architecture at
-0.3 would mean this is a property of one checkpoint and H1' fails.

**H3 (replaces A25's broad staleness claim).** A25 failed because age was
defined as time since `obs > 0`, which is zero by construction above the seen
threshold and therefore degenerate outside the blind set. Redefined: age is
**time since observability last exceeded tau = 0.15**, which varies everywhere.
Hypothesis: conditional on observability and the model's confidence, error
increases with that age. **Acceptance: positive, interval excluding zero, on
held-out scenes.** A null closes the staleness line for good rather than
prompting a third definition.

**Stage 5 reopened (from section 1).** On the second backbone, inside
`mask_camera`, observability must beat class + confidence on held-out ECE with
an interval excluding zero. FB-OCC gives +0.003134 [+0.002455, +0.003901]. A
null on a second architecture means the FB-OCC result was a property of that
model and Stage 5 closes properly this time.

Each of these was written with the outcome unknown and is reported whichever
way it lands.

---

## A42 -- the free-space hump, explained; and a correction to the result that
## survived A24

**Date:** 2026-09-24. **Decided:** after seeing data. **EXPLORATORY.**
`scripts/eval/freespace_hump.py`. Full split, 536,229,736 voxels inside
`mask_camera` (A5's pre-registered scope), 150 scenes, exact sufficient
statistics binned by (GT free/occupied, observability decile, height band,
range ring, proximity to structure, confidence bin).

A24 located H1's failure in the free stratum and recorded that the CAUSE "is
not established". Two of this project's failed predictions trace to that one
unexplained phenomenon, so it is worth closing. It is now closed, and closing
it cost the project one of its own favourable numbers.

### 1. The hump is in ACCURACY, not confidence

    GT FREE      voxels        confidence   accuracy      gap
    obs = 0     111,848,198      0.9375      0.8823     0.0552
    decile 1     22,617,552      0.9746      0.9528     0.0219   <- the anomaly
    decile 3     18,929,992      0.9607      0.9200     0.0407   <- the peak
    decile 5     24,612,781      0.9664      0.9291     0.0373
    decile 8     19,522,889      0.9765      0.9576     0.0189
    decile 10   117,598,612      0.9844      0.9756     0.0088

Both terms dip in the middle; accuracy dips further. So the model is genuinely
worse at mid observability on free space and its confidence does not follow it
down. This is a real overconfidence pocket, not a confidence artefact, and the
distinction was not previously available because nobody had split the gap.

The anomaly is not the peak, it is **decile 1**: barely-observed free cells are
the second best calibrated of all, beaten only by the fully observed.

### 2. Three composition hypotheses tested. All three FAIL.

**MY PREDICTION FAILED, and this is the sixth recorded.** I predicted the hump
was Simpson's paradox along geometry -- that deciles mix heights and ranges
unevenly, air at 4 m being trivially free while the 40 cm above the road is
contested. Rank correlation of gap against decile, the statistic H1 was
registered on:

    GT FREE   aggregate                                   -0.550
              voxel-weighted mean, 7 cells >= 5M voxels   +0.581

It reverses. Geometry composition does not cause the hump -- **it MASKS it.**
Inside fixed height and range the rising limb is steeper and longer than in the
aggregate. The aggregate's apparent decline on free space is partly manufactured
by the height mix shifting across deciles.

Proximity to real structure fails as an explanation too, though it produces the
mechanism in section 3: the hump is present with the same shape in all three
shells.

    gap by decile, GT FREE
      touching a surface   .1112 .1297 .1315 .1423 .1469 .1419 .1265 .1155 .0865
      one cell away        .0632 .0734 .0828 .0888 .0890 .0780 .0643 .0518 .0231
      open air (2+ cells)  .0136 .0268 .0316 .0287 .0257 .0185 .0134 .0092 .0017

Rise then fall, every time. **The hump is robust to height, to range and to
proximity. It is a property of the observability axis, not of what the deciles
happen to contain.**

### 3. The mechanism: it is a false-positive occupancy curve

On a ground-truth FREE voxel every error is, by construction, the model
asserting something occupied. So free-space accuracy IS one minus the
false-positive occupancy rate, and the hump is a hump in false positives.

Where those false positives live:

    GT FREE, by distance to the nearest GT-occupied voxel
      touching a surface    29,116,975    7.0%   gap 0.1194
      one cell away         46,593,757   11.2%   gap 0.0627
      open air (2+ cells)  339,193,947   81.8%   gap 0.0190

A **6.3x gradient**. The model's free-space miscalibration is concentrated at
object boundaries: it smears occupancy outward from real surfaces rather than
inventing objects in empty space.

Put beside the occupied stratum, the two halves finally make one statement. On
OCCUPIED voxels the gap peaks at decile 1 (0.2889) and declines. On FREE voxels
decile 1 is the best and the peak moves to deciles 3-5. Those are different
physical situations:

* **barely seen and occupied** -- the model half-sees an object and
  overcommits. This is A14's danger zone.
* **barely seen and free** -- there is nothing to assert, so the model leaves it
  empty and is right. Easy.
* **moderately seen and free** -- enough evidence to start placing a boundary,
  not enough to place it correctly. **This is where the false positives are**,
  and it is why the free-space peak sits at mid observability rather than at
  the bottom of the scale.

So the hump is not a defect in the measure. It is the measure resolving a
second, distinct failure mode that the aggregate H1 test could not see because
it summed the two strata together.

### 4. The correction, and it goes against us

A24's surviving claim is that the pre-registered decline "holds exactly" on
ground-truth occupied voxels at rank correlation -0.865 (here -0.967). Applying
the identical confound test to that stratum -- which A24 did not do, and which
it would have been selective not to do here:

    GT OCCUPIED   aggregate                                   -0.967
                  within-cell mean, 12 cells >= 1M voxels     -0.301

    by cell, the largest band (vehicle 0.6-1.8 m)
      10-20 m  -0.086      20-30 m  -0.333      30-57 m  -0.486

**A large part of the occupied decline is also geometry composition.** The
direction survives everywhere outside the road-surface band, so the sign of
A24's finding stands, but its strength does not: -0.967 is an aggregate figure
and the within-geometry figure is about a third of it. Every future quote of the
occupied decline carries both numbers.

### 5. A defect in the binning H1 was tested on

Observability saturates. Measured on 76,800,000 sampled voxels:

    obs > 0                     32.3% of all voxels
    obs == 1.0000                7.5% of all, and 23.2% of the obs > 0 population

Almost a quarter of the non-zero population sits at exactly 1.0, so the top two
deciles cannot be separated: decile 9 comes out **empty** and decile 10 holds
117.6M of 414.9M free voxels. A26's "all nine deciles are populated" was true of
its own binning and is not true of an equal-mass decile cut on the full split.

A monotone-across-deciles test in which one bin is empty and the top bin holds
28% of the stratum is weaker than it appears. This does not change H1's verdict
-- H1 failed -- but it means the test H1 failed was a blunter instrument than
the amendment describing it implies, and the second backbone's H1' must state
its binning against the saturation rather than assuming ten equal bins exist.

### 6. What this does NOT do

It does not rescue H1. H1 was pre-registered over all voxels and lost, and a
mechanism for why a stratum behaves oddly is an explanation, not a pass. It
does not rescue A24's failed composition prediction, and it adds a sixth failed
prediction of mine on top of it. What it does is convert "we do not know why the
measure is non-monotone on 86% of the volume" -- the first question a reviewer
asks -- into a mechanism with a 6.3x gradient behind it.

---

## A43 -- PRE-REGISTRATION: H1 is re-tested once, on a repaired binning

**Date:** 2026-09-24. **Written BEFORE the re-test is run.** The numbers do not
exist at the time of writing and this entry is committed before the script that
produces them.

### Why a re-test is legitimate here, and where the line is

H1 failed and that verdict is not in dispute. What A42 section 5 established is
that **the instrument was broken**: observability saturates, 23.2% of the
non-zero population sits at exactly 1.0, so an equal-mass decile cut cannot
separate the top of the scale. On the full split, decile 9 comes out EMPTY and
decile 10 holds 117.6M of 414.9M free voxels. H1 was specified as a monotone
trend "with observability decile". A cut in which one decile is empty and
another holds 28% of the stratum is not a decile cut.

So the distinction this amendment turns on:

* re-running with a **different population** chosen after seeing which
  population gives the answer you want -- forbidden, and refused in A41
  section 5;
* re-running with the **same population and the same statistic** on a binning
  that actually satisfies the definition already written down -- a repair.

This is the second. The population is unchanged: all voxels inside
`mask_camera`, exactly as A5 fixed on 17 September. The statistic is unchanged.
Only the bin edges change, and they change for a defect found independently,
while investigating something else, and recorded before this re-test was
contemplated.

**This is the only re-test. It runs once. Whatever it prints is the result.**

### The repaired binning, fixed now

Saturation is not noise, it is a distinct population -- cells a camera sees
completely -- so it gets its own bin instead of being smeared across two empty
ones.

    bin 0      observability == 0
    bins 1-8   equal-mass octiles of the observability in (0, 1) population,
               edges computed on a 120-frame sample and then FROZEN
    bin 9      observability == 1.0 exactly

Nine live bins (1-9) for the trend test, as against the nine live deciles A6
reported. Bin 0 is reported and, as in A5, excluded from the monotone test.

### Acceptance, fixed now

H1 as originally stated: the calibration gap declines monotonically with
observability, over all voxels inside `mask_camera`, pre-registered direction
negative.

    HOLDS   strictly decreasing across all nine live bins, no reversals,
            AND Spearman rank correlation <= -0.5
    FAILS   anything else

Both figures are reported either way. A single reversal is a failure: that is
the standard A6 applied when it declared H1 held at ten of ten, and applying a
looser one now would be choosing the threshold by the answer.

### Stated prior, so it cannot be claimed afterwards

**I expect this to fail.** Free space is 77.4% of the voxels inside the mask
(414.9M of 536.2M) and A42 measured the free-space hump surviving inside every
height band, every range ring and every proximity shell. A binning repair does
not remove a hump that is robust to three stratifications. The re-test is run
because the instrument was demonstrably broken and the fair thing is to give
the hypothesis the test it was written for, not because the outcome is expected
to change.

If it fails, H1 is reported as failed on a repaired binning, which is a
STRONGER negative result than failing on a broken one, and the matter is closed
for FB-OCC. If it holds, A7's verdict is superseded and the reason is recorded
as an instrument defect rather than as a new analysis.

---

## A44 -- PRE-REGISTRATION: A25's failure, repaired properly this time

**Date:** 2026-09-24. **Written BEFORE the measurement.** Committed before the
script that produces it.

### First, a defect in my own A41 pre-registration

A41 section 6 registered H3 as "age = time since observability last exceeded
tau = 0.15, which varies everywhere". **It does not.** Any cell whose
observability is above tau right now has age zero by construction, which is
exactly the degeneracy that killed A25. I wrote a repair that reproduces the
original defect. Recorded as an error and superseded here before it was run.

### What A25 actually got wrong

A25 asked "how long since this cell was last seen" and could only ask it of
cells that are blind NOW. That splits the volume into a set where the variable
is informative (+9.20 pts [+8.08, +10.34] between never-seen and recently-seen)
and a set where it is identically zero. A variable defined on a subset cannot
be tested for adding information everywhere, so the broad claim was untestable
rather than false.

### The repair: one continuous quantity defined everywhere

Replace age with **temporal observability**, evidence discounted by how old it
is:

    obs_T(v) = max over the current and past 8 keyframes of
               [ obs_t(v) * lambda^(seconds since t) ]

Past frames only. A cell seen well now scores ~obs_now; a cell seen well two
seconds ago scores a discounted value; a cell no camera has ever seen scores 0.
It is continuous, defined on every voxel, and it collapses to plain
observability at lambda = 0, so the comparison against A26's measure is nested.

**lambda is fixed now, before the run:** half-life **2.0 seconds**, so
lambda = 0.5^(1/2) per second. Justified from A25's own measured decay -- error
inside the blind set climbs 13.63% at 0.5 s to 23.83% at never-seen, with the
midpoint of that climb near 2 s. Other half-lives are reported as a sensitivity
sweep; **the primary result is the 2.0 s row and it is named here so the sweep
cannot be mined for the best one.**

### Hypothesis and acceptance, fixed now

H4: temporal observability predicts per-voxel model error better than
instantaneous observability. AUROC of per-voxel error, inside `mask_camera`
(A5's scope), paired scene bootstrap.

    HOLDS   AUROC(obs_T) - AUROC(obs) > 0 with a 95% interval excluding zero,
            at the pre-named half-life of 2.0 s
    FAILS   anything else

### Stated prior

**Genuinely uncertain, and leaning positive.** A25's narrow result is strong
and A41's corridor study showed warped past evidence reaching 94% of corridor
columns, so the information exists. Against that, FB-OCC already fuses sixteen
frames internally, so the model may have internalised most of it -- which is
the same argument that closed Stage 5 before A41 reopened it. A null here is a
real finding about how much temporal information survives in the model's own
output, and closes the staleness line for good rather than prompting a third
definition.
