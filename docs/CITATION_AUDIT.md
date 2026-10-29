# Citation audit, 2026-10-29

CVPR 2027 states that **papers with inaccurate or non-existent citations may be
rejected**. This records what was checked, how, and what was found. It
distinguishes references verified against a primary source today from those
accepted on recognition, because conflating the two would defeat the purpose.

Starting point: 66 entries in `opendrivefm_refs.bib`, 64 cited by `main.tex`.
After this audit: 65 entries, 63 cited.

## Found and fixed

| Entry | Defect | Action |
|---|---|---|
| `occludednuscenes2025` | `author={Anonymous and others}` | real authors added (Kumar, Brophy, Mohandas, Grua, Sistu, Donzella, Eising) |
| `lightocc2024` | `author={Anonymous and others}` | real authors added (Zhang, Zhang, Cai, Liu, Wang) |
| `occworldsemi2025` | `author={Anonymous and others}` | real authors added (Li, Li, Zheng, Sun, Wang, Chen) |
| `depthanyprior2025` | `author={Anonymous and others}` | real authors added (Wang, Chen, Yang, Wang, Zhang, Zhao, Zhao) |
| `wang2026ngocc` | **could not be verified to exist** | **entry and citation removed** |
| `li2023fbocc` | `journal={arXiv preprint}`, no identifier | arXiv:2307.01492 added |
| `chen2024vadv2` | `booktitle={arXiv}`, no identifier, wrong type | `@article`, arXiv:2402.13243 |
| `iso21448` | empty journal field, BibTeX warning | `@techreport` with institution and number |
| `kim2025protoocc` | volume and number both set, BibTeX warning | number removed |

### On NGOcc

`wang2026ngocc` claimed "NGOcc: Normal-Guided 3D Occupancy Prediction with
Observability-aware Time Fusion", CVPR Workshops 2026, with the author field
literally `{Wang and others}`. Three separate searches returned nothing: no CVF
workshop page, no arXiv entry, no project page.

It was cited once, in the Related Work paragraph whose explicit purpose is to
say we claim no priority over visibility. An unverifiable citation is bad
anywhere; in that paragraph it is self-defeating. Removed rather than kept
pending, because the submission cannot carry a reference nobody can find.

If the paper does exist and a PDF turns up, restoring it is one line.

## Verified today against a primary source

CVF open-access pages, arXiv abstract pages or publisher records were opened
for each of these.

    rahimi2026mad          CVPR 2026, CVF open access
    xia2026drivelaw        CVPR 2026, CVF open access
    zhu2026dlwm            CVPR 2026, CVF open access
    tan2026lcdrive         CVPR 2026, CVF open access (also arXiv 2512.10226)
    kalble2025evocc        CVPR 2025, CVF open access
    li2024gafusion         CVPR 2024, CVF open access (also arXiv 2411.00340)
    occludednuscenes2025   arXiv 2510.18552
    lightocc2024           arXiv 2412.05976
    occworldsemi2025       arXiv 2502.07309
    depthanyprior2025      arXiv 2505.10565
    borger2022sensorvisibility  arXiv 2211.06308
    occ2024alpha           arXiv 2406.11021
    zheng2024monoocc       arXiv 2403.08766
    li2023fbocc            arXiv 2307.01492
    chen2024vadv2          arXiv 2402.13243

The five 2026-dated entries were the highest risk in the file -- recent enough
that they could plausibly have been written from expectation. All four CVPR
2026 papers are real and have CVF PDFs. The fifth, NGOcc, was not.

## Accepted on recognition, not individually re-verified

The remaining entries are canonical works whose titles, venues and years are
standard and were checked by inspection of the entry rather than by opening a
source: Hanley & McNeil 1982, Efron & Tibshirani 1986, El-Yaniv & Wiener 2010,
Naeini 2015, Gal & Ghahramani 2016, Hendrycks & Gimpel 2017, Guo 2017,
Lakshminarayanan 2017, Geifman & El-Yaniv 2017 and 2019, Kendall & Gal 2017,
Liang 2018, Lee 2018, Jiang 2018, Nosek 2018, Corbiere 2019, Behley 2019,
Blum 2021, Philion & Fidler 2020, Caesar 2020, Liu 2020, Sun 2020, Huang 2021,
Granese 2021, Angelopoulos & Bates 2021, Li 2022 (BEVFormer), Cao 2022,
ISO 21448, Harley 2023, Hu 2023, Tian 2023 (Occ3D), Wei 2023 (SurroundOcc),
Huang 2023 (TPVFormer), Li 2023 (VoxFormer, BEVDepth), Zhang 2023 (OccFormer),
Wang 2023 (OpenOccupancy), Liu 2023 (SparseOcc), de Jorge 2023, Dreissig 2023,
Chambon 2024, Wang 2024 (OccWorld), Zhu 2024, Bochkovskii 2024, Hu 2024,
Dauner 2024.

**This is a weaker standard and is recorded as such.** Every one is a widely
cited paper that a reviewer will recognise, and none carries the markers that
flagged the five defects above -- no placeholder authors, no missing
identifiers, no unverifiable venue. If time allows before 16 November, opening
a source for each would close the gap.

## Known soft spots, not errors

* `occ2024alpha` is cited as an arXiv preprint. It appears to have since been
  published in TMLR. Citing the preprint is accurate, not wrong, but the
  published venue would be better.
* `blum2019fishyscapes` has a 2019 key and a 2021 year. The entry is correct
  (IJCV 2021); only the key is misleading, and keys are not rendered.
* `derooij2023segreliability` has a key naming de Rooij while the first author
  is de Jorge. Again key-only, not rendered.
* Two entries are uncited and harmless: `radford2019gpt2`, `hu2023uniadcvpr`.
  BibTeX omits them from the printed list.

## Build state after the audit

    0 LaTeX errors, 0 undefined citations, 0 BibTeX warnings
    63 bibitems, 13 pages total
