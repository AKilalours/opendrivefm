# Figure plan — SightOcc (CVPR submission)
# Revision 2. Supersedes revision 1.

## 0. Duplication audit (measured from the figure scripts)

Four of the six figures are built from the SAME frame,
token `dc03fe4e0a374592af7701c66d5276c4`.

| Asset | Appears in | Count |
|---|---|---|
| Camera JPEG of token dc03fe4e | Fig 1 (a), Fig 3 (a), Fig 4 top-left | 3x |
| BEV observability heatmap of token dc03fe4e | Fig 1 (c), Fig 3 (d), Fig 4 top-right, Fig 5 (a) | 4x |
| Granularity line plot (confidence vs geometry) | Fig 1 (d), Fig 2 (a) | 2x |

This is why Figure 1 and Figure 3 read as the same picture. Fix, applied in code,
not by prompting: one distinct sealed-test token per figure, and the granularity
line plot deleted from Figure 1 so it lives only in Figure 2.

---

## 1. The rule that decides what can be generated

A figure panel is either SCHEMATIC or DATA.

* SCHEMATIC = boxes, arrows, formulas, a cartoon of a ray hitting a voxel.
  Claims nothing about a measurement. Safe to generate.
* DATA = any camera frame, any BEV map, any heatmap, any curve, any scatter,
  any number. Claims a measurement. Must be rendered from
  `data/pack/` and `outputs/artifacts/`. Never generated.

Revision 1 broke this rule and the generated images contained an invented
AUC of 0.93 (the real voxel AUROC is 0.6321) and an invented monotone
error curve (the real H1 result is rho = -0.433 with two reversals).

So: prompts below produce SCHEMATIC artwork with EMPTY LABELLED SLOTS.
The slots get filled with real renders.

---

## PROMPT A — Figure 1, method pipeline (schematic shell)

Use the revision-1 image you already made as the visual target. Keep the
palette, the rounded blocks, the arrows, the formulas. Change only this:

> Produce the same pipeline figure, but every photographic inset, every map
> and every plot is replaced by an EMPTY PLACEHOLDER: a plain light-grey
> rectangle (#E8E8E8) with a 1pt dashed mid-grey border and a centred
> monospace slot code in 9pt dark grey. No imagery, no axes, no numbers, no
> curves inside any placeholder. Six placeholders, with these exact codes and
> aspect ratios:
> - SLOT-A1 .. SLOT-A6, in the Multi-camera rig block, six 16:9 rectangles in
>   a 2x3 grid, keeping the labels CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
>   CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT underneath.
> - SLOT-B, in the Scene geometry field block, 1:1.
> - SLOT-C, in the Observability volume block, 1:1.
> - SLOT-D, in the Voxel-level ranking block, 4:3.
> - SLOT-E, in the Object-level missed-detection block, 16:9.
>
> Keep, exactly as drawn: the two pale-blue sub-blocks "Projected coverage
> c_i(v)" and "First-hit visibility s_i(v;G)", their two small line-art
> diagrams (the voxel projecting onto an image plane; the ray stopping at the
> first occupied voxel with the voxels behind it greyed), the circled
> multiply between them, the boxed equation O(v;G) = max_i c_i(v) . s_i(v;G),
> the cream container labelled "Geometric Observability O(v;G) — learning-free,
> one parameter", and the top bracket "evaluation-time diagnostic". Those are
> schematic and they are correct.
>
> Remove the "AUC=0.93" text and any other number that is not part of a
> formula. Formulas keep their symbols. Nothing numeric otherwise.

I then fill SLOT-A1..E with real renders from a single sealed-test token and
composite them in.

---

## PROMPT B — NEW Figure 2, "what confidence cannot see" (fully schematic, no slots)

This is the one genuinely new figure worth adding, and it is 100% concept,
so it can be generated outright with nothing to fill in.

> A clean two-row conceptual diagram, same palette and line style as the
> pipeline figure, aspect ratio about 2.4:1, white background.
>
> Row 1, labelled "A wrong prediction": three stages left to right joined by
> thin arrows. Stage 1, a cartoon voxel column with a small grey car icon in
> it, labelled "object present". Stage 2, a rounded salmon box labelled
> "model predicts: car, p = 0.55", with a small horizontal probability bar
> underneath that is about half filled. Stage 3, a rounded pale-blue box
> labelled "confidence is low -> flagged", with a small upward arrow icon.
>
> Row 2, labelled "A missing prediction": same three stages. Stage 1,
> identical voxel column with the same car icon, labelled "object present".
> Stage 2, a rounded salmon box labelled "model predicts: free, p = 1.00",
> with a fully filled probability bar underneath. Stage 3, a rounded
> pale-blue box labelled "confidence is maximal -> not flagged", with a small
> downward arrow icon.
>
> To the right of both rows, spanning them, one lavender rounded box labelled
> "O(v;G) does not consult the model" with two thin arrows pointing into it,
> one from each row, and a short caption inside: "same geometry either way".
>
> No photographs, no heatmaps, no plots, no AUROC values, no dataset names.
> Only the two probabilities 0.55 and 1.00, which are illustrative and are
> labelled as such in the figure with the word "illustrative" in 8pt grey
> under row 2.

---

## NOT PROMPTS — Figures 3, 4, 5, 6 are data figures

These cannot be generated at all. Their layouts are fixed as follows and I
render them from the artifacts. Listed here so the design is agreed before I
build.

**Figure 3, qualitative.** Your revision-1 1x4 layout is adopted verbatim:
(a) Input, (b) Ground truth, (c) Prediction, (d) Observability, bold captions
underneath, red dashed box with "predicted free, conf 1.00" on (c), white
dashed box with "O = 0.00" on (d), one colourbar under (d) only. Rendered
from a real token that contains a genuinely missed object, chosen by script.
Distinct token from every other figure.

**Figure 4, why not mask_camera.** Your revision-1 2x2 layout is adopted, with
one change forced by the data: the bottom-right panel plots the measured
error rate per observability decile as points with bootstrap error bars, and
the fitted line is NOT smooth, because the real relationship has two
reversals (rho = -0.433). Drawing it monotone would be reporting a result we
did not get. Distinct token for the two BEV panels.

**Figure 5, deployability.** Unchanged in content. Gets its own token.

**Figure 6, quadrant.** Unchanged, no imagery, no duplication risk.

---

## Token assignment (removes the 4x repeat)

| Figure | Token | Selection rule |
|---|---|---|
| Fig 1 slots | T1 | highest contrast between observed and blind volume |
| Fig 3 | T2 | contains at least one strict-miss object with O < 0.05 |
| Fig 4 | T3 | large region where mask_camera = 1 but O varies widely |
| Fig 5 | T4 | largest gap between occ_gt and occ_pred occupied fraction |

All four drawn from the 75 sealed test scenes, chosen by script, recorded in
METHOD_FREEZE so the choice is reproducible and not cherry-picked by eye.
