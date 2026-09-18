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
