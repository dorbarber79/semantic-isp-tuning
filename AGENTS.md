# Session Handoff

## Communication

Communicate with the user in English. Keep the distinction between verified
results, historical experiments, and proposed work explicit.

## Start Here

1. Read [`README.md`](README.md) for the repository map.
2. Read [`experiments/semantic_isp/CURRENT_METHOD.json`](experiments/semantic_isp/CURRENT_METHOD.json)
   and [`experiments/semantic_isp/README.md`](experiments/semantic_isp/README.md)
   before changing the active ISP.
3. Read [`docs/REPRODUCING.md`](docs/REPRODUCING.md) and
   [`datasets/README.md`](datasets/README.md) before running image experiments.
4. Read [`papers/README.md`](papers/README.md) before changing the manuscript
   or its figures.

The active implementation is the six-control Exp81 CLIP calibration under
`experiments/semantic_isp/`. The module has a five-axis legacy fallback; the
bundled Exp81 config declares six axes and contains the learned calibration.
Trust the config and `CURRENT_METHOD.json`, not the legacy default `AXES` tuple.

## Task Entry Points

- Fit one FiveK image against photographer targets:
  `experiments/semantic_parameter_learning/fivek_photographer_fit.py`.
- Run/resume the FiveK batch:
  `experiments/semantic_parameter_learning/fivek_batch.py`.
- Review and validate visual annotations:
  `experiments/semantic_parameter_learning/FIVEK_FIT_ANNOTATION_PROTOCOL.md`.
- Build an immutable dataset release:
  `experiments/semantic_parameter_learning/package_fivek_dataset.py`.
- Generate the PSNR-trimmed statistics:
  `experiments/semantic_parameter_learning/fivek_dataset_psnr_statistics.py`.
- Rebuild Exp81 poles from saved Exp80 phrase-image records:
  `experiments/semantic_isp/exp81_descriptor_tail_calibration.py`.
- Probe request-to-ISP mapping:
  `experiments/semantic_isp/exp80_word_probe.py`.

Each task has a separate script. Keep raw source images and generated results in
the external dataset/results roots; use a fresh output directory for each run.
Do not silently regenerate or overwrite immutable releases.

## Important Locations

- Current ISP source, tests, descriptors, and frozen calibration:
  `experiments/semantic_isp/`.
- Superseded numbered studies:
  `experiments/archive/numbered/`.
- FiveK scripts and annotation source documents:
  `experiments/semantic_parameter_learning/`.
- Current manuscript and the one canonical paper-figure directory:
  `papers/` and `papers/figures/`.
- Current FiveK release pointer on the original workstation:
  `/data/data1/FiveK/datasets/FiveK-ISPVA/latest.json`.
- At this handoff it points to v5: 145 scenes, 725 pairs, zero pending reviews;
  the v5 staged full-checksum verification passed.
- Latest additional 20-scene pilot, fully fitted and manually reviewed:
  `/data/data1/FiveK/C_data/results/isp_match_20_pilot_v2/`.
- FiveK reports, annotations, source masters, and renders remain in that
  external immutable release; do not make a second copy in this repo.

The v5 dataset manifest says redistribution review is pending. Do not publish
FiveK RAW/TIFF images or the complete dataset without explicit rights clearance.
The historical research file `docs/SEMANTIC_ISP_RESEARCH.md` contains Exp63-73
notes and is not the active method specification.