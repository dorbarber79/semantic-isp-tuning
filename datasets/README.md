# Dataset Index

## FiveK-ISPVA

The canonical dataset is managed outside Git at
`/data/data1/FiveK/datasets/FiveK-ISPVA/`. On the original workstation,
`latest.json` currently points to release `2026-11-15-v5`. Always read that
pointer instead of hard-coding an old release number.

The v5 manifest records 145 scenes, 725 expert-fit pairs, 145 visually reviewed
scenes, zero pending scenes, and approximately 51.6 GB of unique source assets.
It combines the original 105 scenes and two distinct 20-scene fit/annotation
pilots. The v5 staged release passed full SHA-256 verification before activation;
its post-publication structural verification also passes. Redistribution rights
still require review. Its report bundle is under
`releases/2026-11-15-v5/reports/latest/`; the combined gallery is at
`releases/2026-11-15-v5/screen_report/index.html`.

Verify a release before analysis:

```bash
python experiments/semantic_parameter_learning/package_fivek_dataset.py \
  --verify-release /data/data1/FiveK/datasets/FiveK-ISPVA/releases/2026-11-15-v5 \
  --full-checksums
```

The previous v4 release remains preserved. The original MIT-Adobe FiveK DNG/TIFF
masters, fitted renders, and derived
reports are not copied into this repository. Obtain and use them only under the
dataset's terms. Build releases with `package_fivek_dataset.py`; it creates
immutable snapshots and refuses to overwrite an existing release. Preserve
annotation-source files and per-scene provenance.

## Image Organization

The manuscript has one tracked image directory: [`../papers/figures/`](../papers/figures/).
All source photographs and experiment images belong in the external dataset or
result release, not in additional Git folders. Do not copy the same image into
the paper directory; make a publication montage there only when its source and
license are recorded.