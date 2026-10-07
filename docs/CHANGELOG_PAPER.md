# OpenDriveFM paper: what changed, 2026-10-07

Generated from git. Range: 21a85a6..HEAD

## Commits

- **c27e15e** A53: pre-register H2' for the second backbone before the pod boots
- **c3fe0da** docs: manuscript spec derived from four CVPR 2026 driving papers
- **d5825de** paper: Figure 1 and Table 4, both generated from artifacts
- **cbc5b58** paper: manuscript scaffold on the CVPR structure, compiles
- **10a7519** paper: Figure 2, a real four-panel qualitative frame; author emails
- **08ac4a4** paper: Figure 3 and Table 1 at the density the accepted papers use
- **01c209f** paper: related work, the remaining method, 4.2 through 5, and a loss we had mislabelled
- **1105b71** paper: fix duplicate Ablations section, fill the last two ablations
- **02660ce** paper: the two remaining introduction paragraphs
- **44c907a** A54: withdraw the inference-time claim; it was false
- **e13ca7a** A55 RESULT: the deployable occluder fails, and the failure explains itself
- **2f428f2** paper: propagate A55 into the figure, the caption and the conclusion
- **e8e08bf** A55b: the mechanism written for A55 was wrong; measured and corrected
- **a8be9ea** track the A55 artifact: it is cited in the paper
- **3bf41bf** paper: new title, rebuilt abstract, explicit geometry field, Figure 6
- **f2db3ea** paper: correct the mask_camera characterisation, reposition, three contributions
- **06124cd** paper: hero figure, and one stale claim the audit caught
- **cf15293** A56: pi_full is close to the wrong question, and the baselines table
- **75f0fd2** paper: title to concept-first; abstract and contributions tightened

## Files that did not exist before this work

- `docs/PAPER_SPEC.md`
- `outputs/artifacts/a55_pred_occluder.json`
- `outputs/artifacts/a56_fullpx.json`
- `outputs/figures/fig0_hero.pdf`
- `outputs/figures/fig1_granularity.pdf`
- `outputs/figures/fig2_qualitative.pdf`
- `outputs/figures/fig3_whynotmask.pdf`
- `outputs/figures/fig5_deployability.pdf`
- `outputs/figures/fig6_quadrant.pdf`
- `outputs/tables/table1_perclass.tex`
- `outputs/tables/table4_ledger.tex`
- `outputs/tables/table4_ledger.txt`
- `paper/main.tex`
- `scripts/eval/a55_pred_occluder.py`
- `scripts/eval/a56_fullpx.py`
- `scripts/paper/table1_perclass.py`
- `scripts/paper/table4_ledger.py`
- `scripts/viz/fig0_hero.py`
- `scripts/viz/fig1_granularity.py`
- `scripts/viz/fig2_qualitative.py`
- `scripts/viz/fig3_whynotmask.py`
- `scripts/viz/fig5_deployability.py`
- `scripts/viz/fig6_quadrant.py`

## Paper

- `paper/main.tex`  854 lines, 11 pages, 6 figures, 6 tables, 16 references
- title: Geometric Observability for Missed Objects in Camera-Only 3D Occupancy

## Experiments run in this window

| id | question | answer |
|---|---|---|
| A55 | does the object result survive a deployable occluder? | no: 0.6708 vs MSP 0.7184, FAILS |
| A55b | why? | over-occlusion, not evidence erasure. predicted occupancy 11x denser |
| A56 | why pi_full = 965? | near-invariant by construction; 0.0067 spread over a 4x range |

## Corrections made to our own earlier claims

| was | now | found by |
|---|---|---|
| "camera-only, available before the model runs" | evaluation-time; occ() is Occ3D ground truth | external review |
| "mask_camera is not a visibility measure at all" | binary camera-observation mask; three precise differences | external review |
| Figure 1 footer "6,016 frames" | 3,008; the field counts frames scanned | our own audit |
| Table 1 aggregate miss rate 10.5% (hand-typed) | 19.35%, computed | our own audit |
| A55 mechanism: "the miss erases its own evidence" | measured and refuted; over-occlusion | our own measurement |
| selective rows read from an ambiguously named key | two clearly named contrasts; surfaced a FAILS we had hidden | our own audit |
| "first work to..." | no priority claim | external review |
