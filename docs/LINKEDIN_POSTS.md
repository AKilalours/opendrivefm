# OpenDriveFM / SightOcc - 6 LinkedIn posts + matching image prompts

Every number below is measured on the sealed 75-scene test split
(3,008 frames, 40,863 dynamic objects, FB-OCC r50 on Occ3D-nuScenes).
Nothing is estimated. No post mentions a conference submission.

---

## SHARED IMAGE STYLE

House style for all five, matching the reference infographics:

> Editorial infographic, warm off-white background #F7F3EC, deep navy
> #1C2A4A for titles and rules, muted slate blue #7D93B2 and warm tan
> #C9A87C as the only two accent fills. Serif display title at the top,
> italic serif subtitle beneath it. Content divided into labelled panels by
> thin 1pt rounded rectangles with small-caps panel headers. Thin solid
> arrows with small heads. Flat vector, no gradients, no drop shadows, no
> 3D, no photographs. All numbers set in tabular serif. One short italic
> takeaway line centred at the bottom. Small attribution text bottom right.
> 4:3 aspect ratio.

---

# 0. THE CORE IDEA (lead post for the series)

### Prompt

> Title: "Observability". Subtitle: "A camera either resolved that space,
> or it did not. The network does not get a vote."
> Three panels left to right, connected by thin arrows.
>
> PANEL 1 "The question nobody asks": a plan-view line-art car with six
> camera cones radiating outward in warm tan, overlapping near the vehicle.
> Three simple obstacle blocks labelled building, hedge, truck each throw a
> grey wedge-shaped shadow outward. Inside the largest shadow, a small
> dashed outline of a bus with the label "predicted free, p = 1.00". Panel
> caption underneath: "every rig has a blind volume; the model reports the
> same answer inside it and outside it".
>
> PANEL 2 "The measure": two small rounded sub-boxes side by side joined by
> a circled multiply symbol. Left sub-box "coverage", with a tiny diagram
> of a voxel cube projecting onto an image plane as a shrinking square and
> the note "how much image area, falls off as 1/d squared". Right sub-box
> "visibility", with a tiny diagram of a ray from a camera origin stopping
> at the first filled cell, the cells behind it greyed, and the note "does
> the ray get there first". Below both, a single boxed line in larger
> serif: "O = max over cameras of coverage x visibility". Three small tags
> beneath it: "no training", "one parameter", "never reads the model".
>
> PANEL 3 "What it buys": a clean slope chart with two lines across three
> x-positions labelled per voxel, per frame, per object. Tan line labelled
> "model confidence" at 0.880, 0.862, 0.719. Slate line labelled
> "observability" at 0.632, 0.807, 0.762. Circle the crossing between frame
> and object. Beneath the chart, two stacked stat rows: "beats the dataset
> camera mask by +0.048 AUROC" and "within noise of a LiDAR point-count
> oracle, 0.762 against 0.774".
>
> Bottom line: "Confidence rates the prediction. Geometry rates the
> evidence."
> Small caption in the lower left: "40,863 objects, 3,008 frames, 75 sealed
> test scenes."

### Post

Every self-driving camera rig has a volume of space it cannot see, and the
model gives you the same answer inside it as outside it.

That is the whole project in one sentence.

A camera-only occupancy model predicts a class for every voxel around the
car. Where it has no evidence, it does not abstain. It predicts free space,
and it does so at maximum confidence. Across 40,863 objects in my test
split the median softmax value is exactly 1.00.

So I stopped asking the network and asked the rig instead.

Observability is two things multiplied. How much image area a voxel
subtends in a camera, which falls off as one over distance squared. And
whether the ray from that camera actually reaches it before a building, a
hedge or a truck gets in the way. Take the best any single camera manages.

No training. One parameter. It never reads the model's output, which is
exactly why it can see what the model's confidence cannot.

The result is a crossover. Ranking whether an individual voxel is wrong,
confidence wins by a wide margin, 0.880 against 0.632. Ranking whether a
whole object was missed entirely, geometry wins, 0.762 against 0.719, and
lands within noise of a LiDAR point-count oracle at 0.774.

Confidence is a statement about a prediction that exists. A missed object
is the absence of one.

Two different questions. Most stacks only instrument the first.

Which one is your monitoring actually answering?

#AutonomousDriving #ComputerVision #Perception #3DOccupancy #MLSafety

---

# 1. The granularity inversion

### Prompt

> Title: "Confidence". Subtitle: "It can only rate a prediction that
> exists."
> Three panels left to right.
> PANEL 1 "The two failures": two stacked rows. Top row, a simple line-art
> car inside a wireframe voxel column with the label "wrong prediction,
> p = 0.55" and a half-filled bar. Bottom row, the identical car and column
> with "no prediction, p = 1.00" and a completely filled bar.
> PANEL 2 "Two scorers": two rounded boxes, "model confidence, reads the
> network" in tan and "geometric observability, reads the rig" in slate,
> with a small eye icon on the first and a small lens-and-ray icon on the
> second.
> PANEL 3 "The inversion": a clean slope chart, two lines across three
> x-positions labelled "per voxel", "per frame", "per object". Tan line at
> 0.880, 0.862, 0.719. Slate line at 0.632, 0.807, 0.762. Mark the crossing
> between frame and object with a small circle. Y axis labelled AUROC.
> Bottom line: "Softmax cannot be unsure about a prediction it never made."

### Post

A camera-only occupancy model can be 100% confident about a bus it never saw.

Not 99%. The median maximum-softmax value across all 40,863 objects in my
test split is exactly 1.00.

Here is the part that surprised me.

I built a learning-free measure of geometric observability: how much image
area a voxel subtends in each camera, multiplied by whether the ray reaches
it before anything else blocks it. No network. One parameter.

Then I raced it against the model's own confidence at three granularities.

Per voxel, confidence wins easily. 0.880 AUROC against 0.632.

Per object, meaning "did the model miss this thing entirely", the ordering
flips. Geometry 0.762, confidence 0.719.

The reason is structural, not empirical. Confidence is a statement about a
prediction the network made. A missed object is the absence of a
prediction. There is nothing for softmax to be unsure about, so it reports
a confident vote for free space.

Geometry never consults the model, so it does not share the blind spot.

If your failure detector is a function of the model's output, what happens
when the model produces no output at all?

#AutonomousDriving #ComputerVision #Perception #MLSafety #3DOccupancy

---

# 2. The blind spot

### Prompt

> Title: "The blind spot". Subtitle: "Confidence and evidence are different
> axes."
> Left two thirds: a large 2x2 matrix. X axis "was there evidence?" with
> ticks low and high. Y axis "is the prediction reliable?" with ticks low
> and high. Cells shaded in increasing tan by miss rate, each carrying a
> large numeral and a small caption: top-left "23.6% missed, BLIND SPOT",
> top-right "3.7% missed, trusted", bottom-left "39.8% missed, unknown
> space", bottom-right "26.7% missed, model error". A thin double-headed
> arrow spans the top row labelled "same confidence, 6.3x the miss rate".
> Right third, stacked: a small horizontal bar chart titled "what blocks
> the view" with bars man-made 43.9, vegetation 27.9, other vehicles 10.4,
> drivable 4.3, percentages at the ends. Below it a two-bar comparison
> titled "objects never predicted", lowest observability band 47.3%,
> highest band 4.7%.
> Bottom line: "The geometry knows before the network runs."

### Post

Split 40,863 driving objects by two questions and one cell should scare you.

Question one: is the model confident?
Question two: did the cameras actually have evidence?

High confidence, good evidence: 3.7% of those objects are missed entirely.

High confidence, no evidence: 23.6% missed.

Same confidence. 6.3 times the miss rate.

That gap is the whole argument. Confidence and observability are not
measuring the same thing, so a stack that monitors only the first is blind
to the second.

I also looked at where the unobserved volume actually sits. It is not
exotic. 43.9% of it is behind man-made structure, 27.9% behind vegetation,
10.4% behind other vehicles.

Buildings, bushes and trucks. The ordinary furniture of a city street.

One more number. Among objects in the lowest observability band, 47.3% are
never predicted at all. In the highest band, 4.7%.

The geometry knows which is which before the network runs, because it is
computed from the calibration and the scene, not from the model.

When your system reports high confidence, do you know whether it had
anything to be confident about?

#AutonomousVehicles #Perception #SensorFusion #SafetyCritical #AIResearch

---

# 3. The result I did not want

### Prompt

> Title: "A negative result". Subtitle: "The story was elegant. The
> measurement disagreed."
> Three panels.
> PANEL 1 "The hypothesis": a small diagram of a ray passing cleanly
> through a transparent car outline toward a camera, with the caption "a
> missed object stops occluding, so observability is falsely high", and a
> thin diagonal strike-through line across the panel in muted red #B3463A.
> PANEL 2 "What actually moved": a slopegraph with two lines between
> x-labels "ground-truth geometry" and "predicted geometry". A flat line at
> 0.1987 to 0.2017 labelled "missed objects, unchanged". A falling line
> from 0.5402 to 0.4586 labelled "detected objects, collapse". Vertical
> double-headed arrows showing the gap shrinking from 0.342 to 0.257, with
> "separation shrinks 25%" beside them.
> PANEL 3 "The cost": three horizontal bars, object-level AUROC, ground
> truth geometry 0.765, model confidence 0.718, predicted geometry 0.671,
> with a dashed baseline at the confidence value.
> Bottom line: "Evaluation-time diagnostic, not a runtime monitor. Yet."

### Post

The most useful experiment I ran this year is the one that failed.

My observability measure needs a geometry field to decide what occludes
what. In evaluation I use ground-truth occupancy, which a vehicle does not
have. So the obvious question is whether it survives when you substitute
the model's own predicted occupancy.

It does not.

Object-level AUROC: 0.765 with ground-truth geometry, 0.718 for plain
confidence, 0.671 with predicted geometry. The deployable version lands
below the baseline it was supposed to beat.

I had a tidy story ready. A missed object is predicted as free, so it stops
occluding, so the ray passes through and observability is falsely high.
Elegant, and wrong.

I measured it. On missed objects observability barely moves, 0.1987 to
0.2017. On detected objects it collapses, 0.5402 to 0.4586.

The real cause is that predicted occupancy calls far more of the volume
occupied than the ground truth does, on one frame 79.0% against 5.4%. It
over-occludes regions the cameras genuinely see. The separation between
seen and unseen shrinks by a quarter and the ranking degrades with it.

So the honest claim is narrower than I wanted. This is an evaluation-time
diagnostic, not a runtime monitor, until the geometry comes from a source
independent of the occupancy model.

Writing that sentence was harder than running the experiment.

What is a negative result you kept in the paper instead of burying?

#ResearchIntegrity #MachineLearning #AutonomousDriving #PhDLife #OpenScience

---

# 4. One bit is not enough

### Prompt

> Title: "One bit is not enough". Subtitle: "A mask is not a measure."
> Two columns.
> LEFT COLUMN "mask_camera": a bird's-eye square split into pure black and
> pure white with a hard edge, a tiny car glyph at the centre. Beneath it a
> histogram with exactly two spikes, at 0 and at 1. Three bullet labels:
> "binary", "built offline with LiDAR", "constant inside the mask".
> RIGHT COLUMN "O(v;G)": the same bird's-eye square rendered as a smooth
> gradient from dark to light with wedge-shaped shadows radiating behind
> obstacles. Beneath it a smooth continuous density curve across 0 to 1.
> Three bullet labels: "continuous", "camera geometry only", "resolves
> occlusion".
> A full-width strip across the bottom titled "inside mask_camera = 1":
> ten dots on a gentle curve showing per-voxel error falling from 0.158 to
> 0.060 with small error bars, annotated "error still varies 2.6x where the
> mask is flat".
> Bottom line: "The mask decides where labels are valid. That is a
> different job."

### Post

"Occ3D already ships a camera mask. Why build anything?"

Fair, and it was the first thing I tested.

mask_camera marks whether a voxel is observed. Three things make it a
different object from what I am measuring.

It is binary. Inside the mask it is constant, so it cannot rank anything,
and ranking is the entire task.

It is built offline from the full sensor suite, including the LiDAR the
camera-only model never receives. It describes the annotation pipeline, not
what the cameras resolved.

It marks observation, not evidence strength. A voxel grazed by one camera
at 40 metres and a voxel filling half an image both read 1.

Then the measurement. Restrict to voxels the mask already calls observed,
and the per-voxel error rate still varies by 2.6 times across observability
deciles. The mask is flat there. Geometry is not.

Head to head on ranking per-voxel error, continuous observability beats the
binary mask by 0.0484 AUROC, interval [0.0390, 0.0584] on a split that was
sealed before any of this was developed.

None of which means the mask fails at its job. Its job is deciding where
labels are valid, and it does that well.

It is just not a reliability signal, and it was never built to be one.

What is a dataset artifact you have seen quietly repurposed as a metric?

#ComputerVision #Datasets #Benchmarking #AutonomousDriving #MLEngineering

---

# 5. Two camps, one blind volume

### Prompt

> Title: "Two camps". Subtitle: "Both have a blind volume. Only one kind of
> system knows it."
> Three panels.
> PANEL 1 "Pure vision": a plan-view car with six camera cones radiating
> out in tan, overlapping near the vehicle, with grey wedge shadows thrown
> behind three simple obstacle blocks labelled building, hedge, truck.
> PANEL 2 "Camera plus LiDAR": the same plan-view car with the camera cones
> plus a slate 360-degree ring, the shadow wedges visibly smaller but still
> present behind the same three obstacles. A caption between the panels
> reads "a second modality shrinks the blind volume".
> PANEL 3 "What neither gives you for free": a rounded box labelled
> "O(v;G)" over a small continuous heat strip from dark to light, with
> three stacked statistics: "blind volume: man-made 43.9%, vegetation
> 27.9%, vehicles 10.4%", "high confidence, no evidence: 23.6% missed",
> "high confidence, good evidence: 3.7% missed".
> Bottom line: "The question is not whether the blind volume exists. It is
> whether the stack knows it is in one."

### Post

The industry split on LiDAR is usually argued as a cost question. I think
that framing misses the interesting part.

Tesla runs pure vision. Waymo, Zoox, Motional, Plus AI and Bot Auto all
carry LiDAR. Both camps ship real systems, so neither position is naive.

Here is what my measurements say the argument is actually about.

Every camera rig has a blind volume. Not a bug, a geometric fact of where
the lenses are and what is standing in front of them. On my test split that
volume is dominated by ordinary occluders: buildings 43.9%, vegetation
27.9%, other vehicles 10.4%.

The question is not whether the blind volume exists. It does for a LiDAR
rig too. The question is whether the stack knows it is in one.

A camera-only occupancy model does not. It predicts free space there, at
maximum confidence, and nothing in its output distinguishes "empty road"
from "no evidence either way". I measured 23.6% of high-confidence objects
in low-observability volume as missed entirely, against 3.7% where the
cameras had a view.

Adding a second modality shrinks the blind volume. It does not by itself
tell you where the remaining one is.

That seems like the more useful thing to compute, whichever camp you are
in, and it costs one geometric calculation rather than a sensor.

For the perception people here: does your stack have an explicit
representation of where it cannot see?

#AutonomousVehicles #Robotics #Perception #LiDAR #ComputerVision

---

---
