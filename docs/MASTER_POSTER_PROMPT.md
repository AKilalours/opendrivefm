# OpenDriveFM / SightOcc - master poster prompt

One image, light palette, picture-led. The illustration carries the idea;
text is labels only. Every number below is measured, do not alter them.

---

## THE PROMPT

Wide editorial infographic poster, 16:9, light and airy, picture-led with
minimal text.

STYLE. Warm off-white page #F7F3EC. Panels are white #FFFFFF rounded
rectangles with a 1pt pale grey #DDD8CE edge and a very soft shadow. Text
deep navy #1C2A4A. Five desaturated accents, used only by meaning: sage
green #6E9A78 cameras, slate blue #5E7FA8 geometry, warm tan #C2A06B the
model, dusty rose #B5726B failure, soft violet #8E7BA8 results. Flat
vector, clean thin strokes, simple rounded icons, generous white space. No
photographs, no gradients, no neon, no dark background, no clutter.

TITLE, top centre, serif: "Geometric Observability".
One italic line beneath it: "A camera either resolved that space, or it did
not."

CENTRE, filling roughly two thirds of the poster. A large top-down plan
view of a car on a pale road. Six translucent sage-green camera frusta
radiate outward and overlap near the vehicle. Three obstacles throw long
pale grey shadow wedges outward: a building upper left, a hedge lower left,
a parked truck right. Label each wedge with only a percentage and one word:
"43.9% man-made", "27.9% vegetation", "10.4% vehicles".

Inside the truck's shadow, a dashed dusty-rose outline of a bus with one
leader line to a small white card holding three short rows:
"ground truth: bus"
"model: free, p = 1.00"
"observability: 0.00"

Under the car, one boxed equation in serif, nothing else:
"O = max over cameras of coverage x visibility"
Three small pill tags beneath it: "no training", "one parameter", "never
reads the model".

LEFT EDGE, two small icon cards stacked, two or three words each.
Card 1, a lens icon plus a shrinking square: "coverage, 1/d squared".
Card 2, a ray hitting a cube with the cubes behind it greyed: "first hit".

RIGHT EDGE, a slim vertical strip of six tiny numbered icons joined by thin
arrows, one word under each: calibrate, project, march, combine, rank,
read.

BOTTOM, a single clean slope chart spanning the width, no panel border.
X axis three ticks: per voxel, per frame, per object.
Tan line labelled "confidence" at 0.880, 0.862, 0.719.
Slate line labelled "observability" at 0.632, 0.807, 0.762.
Circle the crossing between frame and object.
One short dusty-rose note to the right of the chart: "predicted geometry:
0.671".

Bottom rule, centred italic: "Confidence rates the prediction. Geometry
rates the evidence."
Small grey caption lower left: "40,863 objects, 75 sealed test scenes,
Occ3D-nuScenes."

---

## NUMBERS TO VERIFY IN THE OUTPUT

Image models drop and garble small numerals. Check each one before posting,
and if any are wrong, regenerate with the slots empty and set the type
yourself.

  per voxel        confidence 0.880   observability 0.632
  per frame        confidence 0.862   observability 0.807
  per object       confidence 0.719   observability 0.762
  predicted geometry occluder          0.671
  blind volume     man-made 43.9%  vegetation 27.9%  vehicles 10.4%
  missed bus       p = 1.00   O = 0.00
  scale            40,863 objects, 3,008 frames, 75 sealed test scenes
