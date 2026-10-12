# OpenDriveFM / SightOcc - 7 LinkedIn posts + matching image prompts

Every number below is measured on the sealed 75-scene test split
(3,008 frames, 40,863 dynamic objects, FB-OCC r50 on Occ3D-nuScenes).
Nothing is estimated. No post mentions a conference submission.

---

## SHARED IMAGE STYLE

House style for all six, matching the reference infographics. Paste this
block in front of each numbered prompt below.

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

That is OpenDriveFM in one sentence.

I did not get here alone. The project runs on nuScenes, which Motional gave
the field. Tesla's occupancy networks are why camera-only occupancy is a
serious direction at all. Waymo set the bar for what rigorous AV evaluation
looks like. Zoox and Plus AI keep proving the hard part on real roads.

And the company I keep coming back to is Bot Auto.

Xiaodi Hou founded it in 2023 after co-founding TuSimple, and the team is
taking driverless Class 8 freight down I-45 between Houston and Dallas on a
200 mile overnight lane, with Ryan Transportation moving the freight. They
raised a 20 million dollar Series A and went after driver-out with a fraction
of the capital the field assumed it required. That discipline is the thing I
tried to copy in how I ran this research.

Here is the gap I went after.

A camera-only occupancy model predicts a class for every voxel. Where it has
no evidence it does not abstain. It predicts free space at maximum
confidence. Across 40,863 objects in my sealed test split the median softmax
value is exactly 1.00.

So I stopped asking the network and asked the rig.

Observability is the image area a voxel subtends, times whether the ray
reaches it before a building or a trailer does. No training. One parameter.
It never reads the model, which is exactly why it catches what confidence
cannot.

Per voxel confidence wins, 0.880 to 0.632. Per object, meaning the thing was
missed entirely, geometry wins, 0.762 to 0.719.

Which question is your monitoring actually answering?

#AutonomousDriving #AutonomousTrucking #ComputerVision #Perception #MLSafety

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

A camera-only occupancy model can be 100 percent confident about a bus it
never saw.

Not 99. The median maximum-softmax value across all 40,863 objects in my
sealed test split is exactly 1.00.

Tesla's occupancy network work convinced a lot of us that dense volumetric
prediction from cameras was the right output format. I agree. What I wanted
to know is how such a model fails, and whether its own confidence tells you.

Bot Auto is part of why I framed it that way.

Xiaodi Hou has argued publicly that miles per intervention is the wrong
yardstick, because "every company defines intervention differently" and at
scale it becomes "a statistical rarity, essentially noise." He pushes cost
per mile instead, on the grounds that "CPM is a benchmark metric. It's
honest. You can't game it."

That critique stuck with me, because softmax confidence has the same
disease.

It looks like a reliability number. It is reported everywhere. And it is
quietly unfalsifiable in the one case that matters most, because a model
that predicts nothing reports no doubt.

So I built a learning-free measure of geometric observability. Image area a
voxel subtends in each camera, times whether the ray reaches it before
anything blocks it. One parameter. No network.

Per voxel, confidence wins easily, 0.880 AUROC to 0.632.

Per object, meaning did the model miss this thing entirely, it flips.
Geometry 0.762, confidence 0.719.

What makes this different from the uncertainty literature is one design
choice. Every baseline I compared against is a function of the model's
output. Mine is the only signal that never reads it.

Confidence is a statement about a prediction that was made. A missed object
is the absence of one.

If your failure detector is a function of the model's output, what happens
when there is no output?

#AutonomousTrucking #ComputerVision #Perception #MLSafety #3DOccupancy

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

Split 10,790 driving objects by two questions and one cell should stop you.

Is the model confident? Did the cameras actually have evidence?

High confidence with evidence: 3.7 percent missed entirely.
High confidence without evidence: 23.6 percent missed.

Same confidence. 6.3 times the miss rate.

I kept thinking about I-45 while I measured this.

Bot Auto runs an overnight lane between Houston and Dallas, roughly 200
miles of interstate, at night, in Texas weather, with Ryan Transportation's
freight on the deck. A loaded Class 8 truck needs a lot of road to stop.
The occluder ahead of you is a 53 foot trailer, and the thing that matters
is well past it.

In that domain, the gap between "I see nothing there" and "I cannot see
there" is not an academic distinction. It is stopping distance.

So I went looking for what actually does the occluding. Over 400 test
frames, 43.9 percent of the unobserved volume sits behind man-made
structure, 27.9 percent behind vegetation, 10.4 percent behind cars, 15.7
percent behind vehicles of all classes.

Nothing exotic. Buildings, bushes and traffic.

To be straight about scope: nuScenes is urban, not highway. I cannot claim
these exact fractions transfer to a Texas interstate. What transfers is the
method. The geometry comes from calibration and the scene, so any team can
run it on their own rig and get their own numbers before the network ever
runs.

One more, this time across all 40,863 annotated objects. In the lowest
observability band, 47.3 percent are never predicted at all. In the highest
band, 4.7 percent.

The model is not slightly less sure in those places. It is confidently
wrong, and it cannot tell you which it is.

When your stack reports high confidence, do you know whether it had
anything to be confident about?

#AutonomousTrucking #Freight #Perception #SafetyCritical #AutonomousVehicles

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
what. In evaluation I use ground-truth occupancy. A truck does not have
that. So the honest question is whether it survives when you substitute the
model's own predicted occupancy.

It does not.

Object-level AUROC: 0.765 with ground-truth geometry, 0.718 for plain
confidence, 0.671 with predicted geometry. The deployable version lands
below the baseline it was meant to beat, by 0.048 with a 95 percent
interval of 0.014 to 0.083. Not noise.

I had a tidy story ready. A missed object is predicted free, so it stops
occluding, so the ray passes through and observability is falsely high.
Elegant. Wrong.

I measured it. On missed objects observability barely moves, 0.1987 to
0.2017. On detected objects it collapses, 0.5402 to 0.4586. The real cause
is that predicted occupancy calls far more of the volume occupied than the
truth does, on one frame 79.0 percent against 5.4 percent. It over-occludes
regions the cameras genuinely see.

So the claim got narrower. Evaluation-time diagnostic, not a runtime
monitor, until the geometry comes from somewhere independent of the model.
That sentence is in the paper, with the interval next to it.

Why I did not bury it.

On 29 April 2026 a Bot Auto truck left Houston overnight with nobody in it
and arrived south of Dallas at 4:57 in the morning, 230 miles, a real Ryan
Transportation load, no safety driver and no in-cab observer. Xiaodi Hou's
line afterwards was not a victory lap. It was "Houston to Dallas is mile
one."

That is the posture. You state what you did, precisely, and you make the
remaining distance visible.

If I want a measurement taken seriously, it gets held to the same standard,
including when the answer is no.

What is a negative result you kept instead of burying?

#ResearchIntegrity #AutonomousTrucking #MachineLearning #OpenScience #AIResearch

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

Fair, and it was the first thing I tested. That mask exists because the
nuScenes and Occ3D authors did careful work, and I use it constantly.

Three things make it a different object from what I measure.

It is binary. Inside the mask it is constant, so it cannot rank anything,
and ranking is the whole task.

It is built offline from the full sensor suite, including the LiDAR a
camera-only model never receives. It describes the annotation pipeline, not
what the cameras resolved.

It marks observation, not evidence strength. Here is what that costs you.
With the nuScenes front camera, a 0.4 metre voxel is fully resolved out to
16.3 metres. At 20 metres its coverage is 0.67. At 40 metres, 0.17. At 70
metres, 0.05. The mask reads 1 for every one of them.

Then the measurement. Restrict to voxels the mask already calls observed
and the per-voxel error rate still varies by 2.6 times across observability
deciles. The mask is flat there. Geometry is not.

Head to head on ranking per-voxel error, continuous observability beats the
binary mask by 0.0484 AUROC, interval 0.0390 to 0.0584, on a split that was
sealed before any of this was developed.

This matters most to teams running long sensing ranges. On the 230 mile
Houston to Dallas run Bot Auto completed in April with nobody in the cab,
the far field is where the decisions get made, and that is exactly where a
binary mask stops telling you anything.

None of which means the mask fails at its job. Its job is deciding where
labels are valid and it does that well. It was simply never built to be a
reliability signal, and the field has been quietly using it as one.

What is a dataset artifact you have seen repurposed as a metric?

#ComputerVision #Benchmarking #AutonomousTrucking #Datasets #MLEngineering

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

The LiDAR debate is usually argued as a cost question. I think that framing
misses the interesting part.

Tesla runs pure vision. Waymo, Zoox, Motional and Plus AI carry LiDAR.
Bot Auto is doing something that interests me more than the sensor list,
which is building a driverless freight business that has to pencil out per
mile on one real lane before it scales to a second.

Every one of them is shipping something real. Nobody here is naive.

Here is what my measurements say the argument is actually about.

Every rig has a blind volume. Not a defect, a geometric fact of where the
lenses sit and what is standing in front of them. On my test split that
volume is dominated by ordinary occluders: buildings 43.9 percent,
vegetation 27.9 percent, other vehicles 10.4 percent.

The question is not whether the blind volume exists. It does for a LiDAR rig
too, just smaller. The question is whether the stack knows it is inside one.

A camera-only occupancy model does not. It predicts free space there at
maximum confidence, and nothing in its output separates empty road from no
evidence either way. I measured 23.6 percent of high-confidence objects in
low-observability volume as missed entirely, against 3.7 percent where the
cameras had a view.

Adding a modality shrinks the blind volume. It does not tell you where the
remaining one is. That is a separate computation, it is cheap, it runs
before the network, and it is the same computation on a robotaxi in Las
Vegas or a Class 8 truck on I-45.

Does your stack carry an explicit representation of where it cannot see?

#AutonomousVehicles #AutonomousTrucking #Perception #LiDAR #Robotics

---

# 6. MASTER POSTER (the whole project in one image)

### Prompt

> A single large editorial infographic poster, 16:9, titled "Geometric
> Observability", subtitle "A camera either resolved that space, or it did
> not. The network does not get a vote."
>
> LIGHT COLOUR SCHEME. Warm off-white page #F7F3EC. Every panel is a white
> #FFFFFF rounded rectangle with a 1pt pale grey #DDD8CE border and a very
> soft shadow. All body text deep navy #1C2A4A. Five desaturated pastel
> accents used strictly by category: sage green #6E9A78 for cameras and
> evidence, slate blue #5E7FA8 for geometry and rays, warm tan #C2A06B for
> the model and its confidence, dusty rose #B5726B for failures and blind
> volume, soft violet #8E7BA8 for results. Flat vector illustration, clean
> thin strokes, small rounded icons, no photographs, no gradients, no neon,
> no dark background. Serif display title, clean sans body text, all
> numbers in tabular figures.
>
> LAYOUT: a narrow left legend column, a large central illustration, a
> narrow right flow column, and a full-width strip across the bottom.
>
> LEFT COLUMN, three stacked cards.
> Card A "WHAT GOES IN", four rows each with a small icon: "Six cameras,
> images only"; "Rig calibration, intrinsics and extrinsics"; "Scene
> geometry field G"; "No LiDAR, no training, no model output".
> Card B "THE TWO INGREDIENTS", two rows: "COVERAGE c(v), how much image
> area a voxel subtends, falls off as one over distance squared"; "FIRST
> HIT VISIBILITY s(v), does the ray reach the voxel before anything else".
> Card C "THE TWO FAILURES", two rows in dusty rose: "VOXEL ERROR, the
> class is wrong"; "MISSED OBJECT, nothing was predicted there at all".
>
> CENTRE, the hero illustration on a soft off-white road plan view.
> A clean top-down car at the centre with six labelled camera frusta
> radiating outward in translucent sage green, overlapping near the
> vehicle. Three obstacles cast long pale grey wedge shadows outward: a
> building block at the upper left, a hedge at the lower left, a parked
> truck at the right. Each wedge is labelled with its share of the blind
> volume: "man-made 43.9%", "vegetation 27.9%", "other vehicles 10.4%".
> Inside the truck's shadow, draw a dashed rose outline of a bus with a
> leader line to a white callout box reading "ground truth: bus, 43 m" and
> underneath in two coloured rows "model says: free, p = 1.00" in tan and
> "observability: O = 0.00" in slate.
> Under the car, a boxed equation in larger serif: "O(v;G) = max over
> cameras of c(v) x s(v)", with three small pill tags beneath it: "no
> training", "one parameter", "never reads the model".
> A small continuous colour strip from deep slate to pale yellow sits in
> the lower corner of the illustration labelled "observability 0 to 1".
>
> RIGHT COLUMN, a numbered vertical flow of six small cards joined by thin
> downward arrows, each with an icon and two lines of text.
> 1 CALIBRATE, "rig geometry fixes what each camera can resolve".
> 2 PROJECT, "coverage per voxel per camera".
> 3 MARCH, "ray casting gives first-hit visibility".
> 4 COMBINE, "take the best any single camera manages".
> 5 RANK, "score every voxel, frame and object for failure".
> 6 READ, "evaluation-time diagnostic, geometry source stated".
>
> BOTTOM STRIP, full width, titled "WHERE IT WINS AND WHERE IT DOES NOT",
> split into four equal cells.
> Cell 1 "PER VOXEL": two small bars, confidence 0.880 in tan, observability
> 0.632 in slate, caption "confidence wins".
> Cell 2 "PER FRAME": confidence 0.862, observability 0.807, caption
> "close".
> Cell 3 "PER OBJECT": observability 0.762 in slate, confidence 0.719 in
> tan, caption "geometry wins, within noise of a LiDAR oracle at 0.774",
> with a small highlight ring around this cell.
> Cell 4 "THE LIMIT": a single line of text in dusty rose, "swap in the
> model's own predicted occupancy and it drops to 0.671, below confidence.
> Over-occlusion, not circularity."
>
> Bottom rule line, centred and italic: "Confidence rates the prediction.
> Geometry rates the evidence."
> Lower left caption in small grey: "40,863 objects, 3,008 frames, 75
> sealed test scenes, Occ3D-nuScenes." Lower right: author name.

### Post

A self-driving camera rig has a volume of space it physically cannot
resolve, and the model reports the same thing inside it as outside it.

I spent this year turning that sentence into a number.

Here is the whole project on one page.

The input is deliberately thin. Six camera images, the rig calibration, and
a scene geometry field. No LiDAR. No training. It never touches the model's
output, which turns out to be the entire point.

Observability is two things multiplied. How much image area a voxel
subtends in a camera, which falls off as one over distance squared. And
whether the ray from that camera actually reaches it before a building, a
hedge or a trailer gets in the way. Take the best any single camera
manages. One parameter, fixed once.

What it buys you is a crossover.

Ranking whether an individual voxel is wrong, the model's own confidence
wins comfortably, 0.880 against 0.632. Ranking whether a whole object was
missed entirely, geometry wins, 0.762 against 0.719, landing within noise
of a LiDAR point-count oracle at 0.774.

The reason is structural. Confidence is a statement about a prediction that
exists. A missed object is the absence of one, so softmax votes free space
at maximum confidence and means nothing by it.

The honest limit is on the page too. Substitute the model's own predicted
occupancy as the geometry and it falls to 0.671, under the baseline, from
over-occlusion rather than circularity. That bounds this to evaluation time
until the geometry comes from somewhere independent.

Credit where it is due. This runs on nuScenes from Motional. Tesla made
camera-only occupancy a serious direction. Waymo set the evaluation bar.
Zoox and Plus AI keep proving it on road. And Bot Auto, Xiaodi Hou's team
running driverless Class 8 freight down I-45 between Houston and Dallas on
a 200 mile overnight lane, is the clearest reminder that none of this
matters until it holds up per mile on a real route.

Does your stack have an explicit representation of where it cannot see?

#AutonomousDriving #AutonomousTrucking #ComputerVision #Perception #MLSafety

---
