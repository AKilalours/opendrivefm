# Figure generation prompts — SightOcc (CVPR submission)

Style target for all three: the MAD / DriveLaW / DLWM CVPR 2026 figures.
Flat vector blocks with 8px corner radius, 1pt dark-grey outlines, pastel
fills (lavender #CDB4E8 for encoders/decoders, salmon #F2A7A0 for the frozen
core, pale blue #BBD3F0 for trainable parts, cream #FAF3DC for the outer
container), thin black arrows with small solid heads, Helvetica/Arial labels
at 2 sizes only (block labels 11pt, panel captions 10pt bold), real
photographic insets with square corners and a 1pt grey border, white
background, no drop shadows, no gradients, no 3D. Panel letters as
"(a)", "(b)" centred under each panel.

---

## FIGURE 1 — Method pipeline (replaces the current hero)

A single-row left-to-right pipeline, full page width, aspect ratio about
3.2:1. Five stages separated by thin black arrows.

STAGE 1 (left, stacked vertically, each a square photo inset with a 1pt grey
border and a small label underneath): six nuScenes surround-view camera
frames labelled CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK,
CAM_BACK_LEFT, CAM_BACK_RIGHT. Under the stack, a small lavender rounded box
labelled "Rig calibration  K_i, T_i". Group caption: "Multi-camera rig".

STAGE 2: a lavender rounded box labelled "Scene geometry field G" with a
small inset above it showing a grey voxel volume. A dashed grey line splits
the box label into two sub-rows: "G_GT  (evaluation)" in solid text and
"G_pred  (deployable)" in grey.

STAGE 3 (the centre, largest, inside a cream rounded container labelled
"Geometric Observability  O(v;G)  —  learning-free, one parameter"): two
pale-blue rounded boxes side by side, joined by a small circled "x" between
them.
  - Left box: "Projected coverage  c_i(v)" with a tiny diagram above it of a
    voxel cube projecting onto an image plane as a shrinking square, and the
    formula c_i(v) = min(1, (f_i r / d_i(v))^2 / pi_full) in small type.
  - Right box: "First-hit visibility  s_i(v;G)" with a tiny diagram above it
    of a ray from a camera origin stopping at the first occupied voxel, the
    voxels behind it shaded grey.
  Under the circled x, centred: "O(v;G) = max_i  c_i(v) · s_i(v;G)" in
  slightly larger type, boxed in white.
  A small snowflake icon in the container's top-left corner, with the words
  "no training, no learned parameters" beside it.

STAGE 4: a salmon rounded box labelled "Observability volume" with a BEV
heatmap inset above it (viridis colormap, dark = unobserved, bright =
observed, ego vehicle as a small white triangle at centre).

STAGE 5 (right, two stacked outputs, each with a small photo inset):
  - Top: "Voxel-level ranking" with a tiny ROC curve inset.
  - Bottom: "Object-level missed-detection ranking" with a camera frame inset
    showing one green ground-truth 3D box and the word "MISSED" in small red
    type beside it.

Across the top of stages 3–5, a thin horizontal bracket labelled
"evaluation-time diagnostic".

---

## FIGURE 3 — Qualitative, one real frame (must NOT resemble Figure 1)

Four panels in a 1x4 row, each a real nuScenes CAM_FRONT image at identical
crop, 16:9, with a bold 10pt caption underneath. No block diagram elements
anywhere — this figure is photographic only.

(a) "Input" — the raw camera frame, unannotated.
(b) "Ground truth" — same frame with 3D boxes projected: dynamic objects in
    green 2pt, a single pedestrian at the far right in thicker green 3pt.
(c) "Prediction" — same frame with the model's predicted occupancy projected
    as a semi-transparent colour overlay (nuScenes class palette, 45%
    opacity); the far-right pedestrian region is left completely uncovered,
    and a thin red dashed rectangle marks that gap with the label
    "predicted free, conf 1.00".
(d) "Observability" — same frame with the observability field projected as a
    viridis heatmap at 55% opacity; the same far-right region is the darkest
    area in the frame, marked with a thin white rectangle labelled
    "O = 0.00".

A single horizontal colourbar under (d) only, labelled "O(v;G)", ticks at
0.0 and 1.0.

---

## FIGURE 4 — Why not mask_camera

Replace the current four-quadrant layout. Two columns, two rows, no tick or
cross marks and no evaluative words in the image.

Row 1, left: a BEV plot of the binary mask_camera field — pure black and
white, hard edge. Label underneath: "mask_camera: binary, no occlusion".
Row 1, right: a BEV plot of O(v;G) over the same frame — viridis, continuous.
Label underneath: "O(v;G): continuous, occlusion-resolved".

Row 2, left: a histogram with two overlaid distributions on a shared x axis
0 to 1 — mask_camera as two spikes at 0 and 1, O(v;G) as a smooth density.
Label: "support".
Row 2, right: a scatter or 2D density of O(v;G) (x) against per-voxel error
rate (y) inside the region where mask_camera is constant at 1, with a fitted
monotone trend line. Label: "inside mask_camera = 1, error still varies with O".

All judgement goes in the caption text, not in the artwork.
