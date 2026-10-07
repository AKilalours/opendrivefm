# Manuscript spec: format, figures, and what the four CVPR 2026 papers do

Source material: Rahimi et al. MAD; Tan et al. LCDrive; Xia et al. DriveLaW;
Zhu et al. DLWM. All four are CVPR 2026 camera-ready, all four use the same
template, and the conventions below are consistent across all of them unless
noted.

This file is a spec, not prose. `opendrivefm_paper.tex` is still 918 lines of
the retired project; this is what replaces it.

---

## 1. Template and typography

Use the official `cvpr.sty` for the target year. Do not restyle anything; the
four papers are visually identical because nobody touched the template.

    document       two-column, US Letter, 8 pages + unlimited references
    body font      Times, 10pt / 11pt leading
    title          ~14pt bold, centred, capitalised as a sentence
    authors        11pt, superscript affiliation marks, emails in \texttt
    ABSTRACT       SINGLE column, left, ITALIC, ~9pt
                   -- this is CVPR-specific and every one of the four does it
    sections       bold, numbered, ~12pt     "3. Method"
    subsections    bold, ~11pt               "3.2. DriveLaW-Video"
    captions       9pt, bold lead-in phrase then a full explanatory paragraph
    page numbers   proceedings numbers, not 1..8
    header         the CVF open-access watermark block

**Cross-references are coloured links.** `Fig. 1`, `Sec. 3.2`, `Tab. 1`,
`[11]` all render in colour. This is `hyperref` with the template's colour
set, and it is load-bearing for readability: DLWM's method section points
back to `Fig. 2` nine times.

**Bold run-in paragraph heads are used heavily.** Not optional styling; it is
how these papers make a dense method section scannable:

    \paragraph{Spatiotemporal VAE.}   DriveLaW 3.3
    \paragraph{Noise Reinjection.}    DriveLaW 3.3
    \paragraph{Gaussian Flow Prediction.}  DLWM 3.2
    \paragraph{Future Latent Supervision.} DLWM 3.2

Our method section gets the same treatment: **Definition.**, **The one
parameter.**, **First-hit semantics.**, **Why not `mask_camera`.**,
**The sealed split.**

---

## 2. Structure, section by section

All four follow this skeleton. Deviating from it costs reviewer goodwill for
no gain.

    1. Introduction        ~1.25 cols
       - opens with an analogy or a concrete failure, not with "Recently, ..."
         MAD opens with how professional animators draw an animatic before
         rendering. That sentence does more work than a paragraph of citations.
       - the gap, stated as a sentence a reviewer could disagree with
       - "We therefore propose X", then a NUMBERED list of the components
       - Contributions: 3-4 bullets, each one sentence, each falsifiable

    2. Related Work        ~1 col, 3 subsections
       - the last subsection is always the CLOSEST competitors, and it ends
         with an explicit delta: "To the best of our knowledge, we are the
         first to fully separate X from Y" (MAD 2.3)

    3. Method              ~2 cols
       - 3.1 is Motivation or Preliminaries, and it sets notation
       - one subsection per component, bold run-ins inside
       - equations numbered and referenced

    4. Experiments         ~2.5 cols
       - 4.1 Setup: dataset, preprocessing, metrics, training details
       - 4.2 proof of concept at small scale
       - 4.3 the main result
       - 4.4 Ablations
       - key findings as a NUMBERED BOLD list:
         "1. MAD-LTX outperforms all previous open-source driving models."

    5. Conclusion          one paragraph, no new claims

---

## 3. The figures, which is where these papers actually win

### Figure 1 is the whole paper in one picture, on page 1, right column

Every one of the four puts Figure 1 beside the abstract. It is not decoration
and it is not an architecture diagram. It is the claim.

* **MAD Fig. 1** -- top: the two-stage pipeline as three thumbnails
  (first frame -> skeleton pose video -> rendered RGB). Bottom: a scatter of
  ELO against compute budget on a log x-axis, with their model circled and an
  arrow to "Proprietary models". Method and headline result in one frame.
* **DLWM Fig. 1** -- top: the pipeline. Bottom: three small panels, one per
  downstream task, with **the delta numbers printed inside the figure**
  (`IoU 2.84`, `mIoU 1.02`, `L2 16%`, `Col. 21%`). The reader has the result
  before reading a word of the body.
* **Tan Fig. 1** -- the problem and the fix side by side: text CoT at 90
  tokens above, latent CoT at 38 tokens below, each with its decoded
  trajectory. The comparison IS the figure.

**Rule to copy: Figure 1 carries numbers, not just boxes.**

### Figure 2 is the full architecture, two-column width

Shared conventions:

* colour encodes ROLE and stays consistent (DriveLaW: green = video model,
  blue = action model, and the two run as visibly parallel tracks)
* frozen vs trainable marked with a snowflake and a flame icon (DLWM, MAD)
* **real data thumbnails embedded in the diagram** -- actual camera frames,
  actual rendered semantics, actual pose skeletons, not grey placeholder boxes
* panel labels (a) / (b) when there are two trainable stages
* a legend inside the figure box
* the caption is a 5-8 line PARAGRAPH that walks the flow and points at
  section numbers

### Figure 3 is the design-choice justification, and it is cheap

MAD Fig. 3 compares three candidate intermediate representations and
annotates them **directly on the images** with red crosses and green ticks:

    HDMap          x Poor Pedestrian Detail   x Ambiguous 3D Orientation
    Panoptic seg   x Weak Visual Correlation  x Scalability Bottleneck
    pose (ours)    + Preserves Detail and 3D Orientation
                   + Strong Visual Correlation  + Highly Scalable

Four hours of work, and it pre-empts the "why not just use X" review comment
entirely. We need exactly this figure for `mask_camera`.

### Figure 4 teaches the reader how to READ a representation

MAD Fig. 4 shows the ego-motion encoding across three timesteps and annotates
what to look at: blue dots are the trajectory, the checkerboard's apparent
rotation encodes yaw, the parallax of the dust particles encodes speed. It
does not assume the reader decodes the picture unaided.

### Results figures

Diverging stacked bars with an explicit legend and a winner marker (MAD
Figs. 5-6). Not a bar chart of raw metrics.

### Tables

`booktabs`: no vertical rules, `\toprule \midrule \bottomrule`, `\cmidrule`
under grouped headers. Arrows in the header (`minADE6 down`, `APD6 up`). Best
value bold. Relative deltas in parentheses: `4.88 (+10%)`.

---

## 4. What OpenDriveFM's figures must be

Our paper is a MEASUREMENT paper. There is no trained module, so there is no
architecture diagram with flames and snowflakes, and imitating one would be
dishonest. The figure budget goes somewhere else.

**Fig. 1 (teaser, page 1, beside the abstract). The granularity inversion.**
Left: one real frame where the model is confidently predicting free space
exactly where an object is, with the observability map beside it flagging the
same cell. Right: the three-row table rendered as a figure, numbers printed
in the panel the way DLWM does it:

    per voxel    confidence 0.8797   geometry 0.6321   confidence +0.248
    per frame    confidence 0.8618   geometry 0.8066   confidence +0.055
    per OBJECT   confidence 0.7193   geometry 0.7623   GEOMETRY   +0.043

One caption sentence: confidence cannot flag what the model never considered.
That is the paper, and it is visible in four seconds.

**Fig. 2. What observability is.** Ray-marching schematic with the definition
inline, `obs(v) = max_i coverage_i(v) * visible_i(v)`, the single parameter
`FULL_PX = 965.0` called out, and a real observability heatmap over a BEV
beside it. Real data, not a cartoon.

**Fig. 3. Observability vs `mask_camera`, the MAD Fig. 3 treatment.** Same
frame, two maps, annotated directly:

    mask_camera    x binary        x built offline from the full sensor suite
                   x label-validity flag, not a visibility measure
    observability  + continuous    + camera-only, available at inference
                   + ranks error inside the mask

This kills the single most likely rejection comment.

**Fig. 4. The missed-object curve.** 47.32% -> 4.69% across six observability
bins, monotone 6/6, with bootstrap CIs, n = 40,863 objects.

**Fig. 5. Risk-coverage.** Sensor-only vs full volume, with the null result
for full volume shown rather than hidden.

**Fig. 6. Qualitative gallery.** Frames where geometry flags and confidence
does not, and the honest converse: frames where it fails.

**Tab. 1** main H2 result with CIs, both backbones once SurroundOcc lands.
**Tab. 2** per class, including the three where we lose.
**Tab. 3** baselines x granularity: MSP, margin, entropy, logistic, LiDAR
oracle.

**Tab. 4 is the one no other paper in the session has.** The pre-registration
ledger: hypothesis, direction registered in advance, result, verdict. Seven
predictions failed. Six numbers were corrected downward. Print it.

    H     registered         result                   verdict
    H1    monotone, r<=-0.5  2 reversals, r=-0.283    FAILS
    H2    >0, CI excl. 0     +0.0484 [+0.0390,...]    HOLDS
    ...

A reviewer who sees a table of their own failures stops looking for the ones
you hid. This is the paper's strongest rhetorical asset and it is free,
because the record already exists in METHOD_FREEZE.md.

---

## 5. Order of work

Figures before prose. All four of these papers were clearly built around
their figures, and the text is caption expansion. Draft Fig. 1 and Tab. 4
first; if those two do not carry the argument on their own, the argument is
not ready to write up.
