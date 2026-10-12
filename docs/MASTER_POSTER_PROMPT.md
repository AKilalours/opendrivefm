# OpenDriveFM / SightOcc - master poster

One dense, fully-labelled infographic in the style of a technical AV
explainer poster, but in a light palette. Many callouts, each only a few
words. Every number is measured; do not change any of them.

---

## THE PROMPT

A large, highly detailed technical infographic poster, 16:9, in a LIGHT
colour scheme. Dense and fully annotated like a professional engineering
explainer, with many small labelled callout cards joined to the artwork by
thin leader lines. Every label is short, two to seven words. No
paragraphs.

PALETTE AND STYLE. Page background warm off-white #F4F1EA with a faint
blueprint grid. All cards are white #FFFFFF rounded rectangles with a 1pt
coloured left edge, a pale grey #DCD7CD outline and a very soft shadow.
Body text deep navy #1C2A4A, label headings in small-caps. Six desaturated
accents, each used consistently by meaning: sage green #6E9A78 cameras and
evidence, slate blue #5E7FA8 geometry and rays, warm tan #C2A06B the model
and its confidence, dusty rose #B5726B failure and blind volume, soft
violet #8E7BA8 results and metrics, muted teal #4F8A8B scale and data.
Clean flat vector illustration with crisp thin strokes and simple rounded
line icons. No photographs, no gradients, no neon, no dark background.
Numbers in tabular figures.

HEADER, top centre. Serif title "GEOMETRIC OBSERVABILITY". One italic
subtitle: "A camera either resolved that space, or it did not."

LEFT COLUMN, three stacked legend blocks, narrow.

Block 1, heading "WHAT IT IS BUILT FROM", four rows, each a small icon
plus a short label and a one-line note:
  camera icon, sage: "6 cameras" / "images only"
  crosshair icon, sage: "calibration" / "intrinsics, extrinsics"
  cube-grid icon, slate: "geometry field G" / "what blocks what"
  crossed-out brain icon, grey: "no training" / "no model output"

Block 2, heading "THE TWO INGREDIENTS", two rows with small diagrams:
  a cube projecting to a shrinking square on an image plane, slate:
    "COVERAGE c(v)" / "image area, falls as 1/d squared"
  a ray stopping at the first filled cell with the cells behind it greyed,
  slate: "VISIBILITY s(v)" / "does the ray arrive first"

Block 3, heading "THE TWO FAILURES", two rows, dusty rose:
  "VOXEL ERROR" / "the class is wrong"
  "MISSED OBJECT" / "nothing predicted at all"

CENTRE, the main illustration, occupying the largest area. A clean
top-down plan view of a car on a pale road with lane markings, a kerb, a
crosswalk and two parked cars. Six translucent sage-green camera frusta
radiate from the vehicle and overlap near it, each with a tiny label:
front, front left, front right, rear, rear left, rear right.

Three obstacles cast long pale grey shadow wedges outward from the car: a
building block upper left, a hedge lower left, a parked truck to the
right. Each wedge carries a small rose-edged callout with a leader line:
  "MAN-MADE" / "43.9% of blind volume"
  "VEGETATION" / "27.9%"
  "OTHER VEHICLES" / "10.4%"

Inside the truck's shadow, a dashed dusty-rose outline of a bus, with a
leader line to a white card titled "THE FAILURE" holding three short rows:
  "ground truth: bus, 43 m"
  "model: free, p = 1.00"
  "observability: 0.00"

Four further small callouts with leader lines around the scene:
  pointing at a voxel close to the car, slate: "NEAR FIELD" / "c = 1.00"
  pointing at a voxel at the far edge, slate: "FAR FIELD" / "c = 0.06"
  pointing at a thin dashed outline that encloses most of the scene, tan:
    "mask_camera" / "one bit, flat inside"
  pointing at the ego vehicle, teal: "EGO" / "6-camera rig, no LiDAR"

Directly beneath the car, one boxed equation in serif:
  "O(v;G) = max over cameras of  c(v) x s(v)"
with three small pill tags under it: "learning-free", "one parameter",
"never reads the model".

RIGHT COLUMN, a numbered vertical pipeline of six cards joined by thin
downward arrows, each with a line icon, a bold label and one short line:
  1 CALIBRATE, slate: "rig fixes what is resolvable"
  2 PROJECT, slate: "coverage per voxel, per camera"
  3 MARCH, slate: "ray cast for first hit"
  4 COMBINE, violet: "best single camera wins"
  5 RANK, violet: "score voxels, frames, objects"
  6 READ, tan: "evaluation-time diagnostic"

BOTTOM STRIP, full width, heading "OBSERVABILITY BANDS", styled as a
horizontal spectrum bar running left to right from deep slate to pale
yellow, divided into five segments with a small arrow above it labelled
"more camera evidence". Under each segment, the percentage of objects the
model never predicts at all:
  band 0 : 47.3%
  band .21 : 25.1%
  band .38 : 15.6%
  band .78 : 11.9%
  band 1 : 4.7%
Caption under the strip, small: "share of annotated objects with no
prediction anywhere inside the box".

BOTTOM RIGHT, a small boxed key titled "WHERE EACH SIGNAL WINS", three
compact rows with tiny paired bars, tan for confidence and slate for
observability:
  "per voxel"  0.880 / 0.632
  "per frame"  0.862 / 0.807
  "per object" 0.719 / 0.762   with a small ring highlighting this row
One rose line beneath: "predicted-geometry occluder: 0.671".

FOOTER, centred italic: "Confidence rates the prediction. Geometry rates
the evidence."
Small grey caption lower left: "40,863 objects, 3,008 frames, 75 sealed
test scenes, Occ3D-nuScenes."

---

## NUMBERS TO VERIFY BEFORE POSTING

Image models garble small digits at poster scale. Check every one.

  per voxel           confidence 0.880   observability 0.632
  per frame           confidence 0.862   observability 0.807
  per object          confidence 0.719   observability 0.762
  predicted geometry  0.671
  blind volume        man-made 43.9%   vegetation 27.9%   vehicles 10.4%
  bands               47.3 / 25.1 / 15.6 / 11.9 / 4.7 percent
  the missed bus      p = 1.00   O = 0.00   43 m
  scale               40,863 objects, 3,008 frames, 75 sealed test scenes

If any come out wrong, regenerate with those slots blank and set the type
yourself.

---

## THE DESCRIPTION

A self-driving camera rig has a volume of space it physically cannot
resolve, and the model reports the same thing inside it as outside it.

This poster is a year of work on that one sentence.

The inputs are deliberately thin. Six camera images, the rig calibration,
and a field describing what blocks what. No LiDAR. No training. It never
reads the model's output, which turns out to be the whole point.

Observability multiplies two quantities. How much image area a voxel
subtends, which falls off as one over distance squared. And whether the
ray from that camera reaches it before a building, a hedge or a trailer
does. Take the best any single camera manages.

Then the crossover. Ranking whether a voxel is wrong, the model's own
confidence wins, 0.880 to 0.632. Ranking whether a whole object was missed
entirely, geometry wins, 0.762 to 0.719, within noise of a LiDAR
point-count oracle.

Confidence is a statement about a prediction that exists. A missed object
is the absence of one.

The limit is on the poster too. Swap in the model's own predicted
occupancy as the geometry and it falls to 0.671, under the baseline, from
over-occlusion rather than circularity.

Credit where it is due. This runs on nuScenes from Motional. Tesla made
camera-only occupancy a serious direction. Waymo set the evaluation bar.
Zoox and Plus AI keep proving it on road. And Bot Auto, Xiaodi Hou's team
running driverless Class 8 freight down I-45 between Houston and Dallas,
is the reminder that none of it counts until it holds up per mile on a
real lane.

Does your stack know where it cannot see?

#AutonomousDriving #AutonomousTrucking #ComputerVision #Perception #MLSafety
