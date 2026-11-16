# Active Semantic ISP Pipeline

## Start Here

**Current renderer:** `semantic_isp.render_linear` with `compose_parameters`. This is the six-axis direct-control ISP. Do not use `axis_isp.MoranAxisISP` as the active ISP; that class is the retained five-axis cinematic seed implementation.

**Current descriptors:** `isp_control_pole_descriptors.json`, revision `moran_style_v14_extreme_brightness_4x4` (16 phrases per pole). The brightness phrases explicitly target very, much, and extreme dark/bright appearances. The old five-axis seed table and SigLIP comparison tools are under [`legacy/`](legacy/README.md).

**Current protection policy:** gray/white and skin protection applies to palette only. Warmth and saturation remain unprotected; warmth both signs, brightness both signs, and positive saturation were freshly retuned with CLIP.

**Which config should be used:** use the bundled Exp81 CLIP calibration at [`calibration/exp81/direct_soft_calibration.json`](calibration/exp81/direct_soft_calibration.json). Its SHA-256 is recorded in [`CURRENT_METHOD.json`](CURRENT_METHOD.json). It uses v14 brightness descriptors and applies gray/skin protection to palette only. Contrast applies an additive sRGB-luminance shift equally to RGB channels, then compresses gamut while preserving luminance. CLIP ViT-B/32 is the selected model.

Each pole endpoint and its centroid now use the same autotune tail. For every descriptor, take the median physical control setting across the seven autotune images; select the direction-aligned q90 across the 16 descriptor medians; then average the two most extreme descriptor embeddings for the centroid. Each selected text embedding is normalized before averaging, the centroid is normalized afterward, and query embeddings are normalized before projection. Score scales and relevance gates are recalibrated from these selected centroids.

Machine-readable source of truth: [`CURRENT_METHOD.json`](CURRENT_METHOD.json). The older research record is a historical calibration log; use [`docs/REPRODUCING.md`](../../docs/REPRODUCING.md) for current commands and data requirements.

The current implementation is frozen as the **V1 baseline** for ISP debugging.
[`v2/`](v2/README.md) is an isolated byte-identical starting copy; do not change
V1 while evaluating candidate processing decisions in V2.
V2 currently includes a report-serialization repair only; its ISP math is pixel-
equivalent to V1 on the first parity fixture.

Older numbered experiments remain in `../` for provenance; the superseded Exp64 mean aggregator is in [`../archive/`](../archive/README.md).

## Active Files

- `semantic_isp.py`: text centroids, projection, control composition, and current renderer.
- `axis_isp.py`: low-level color, detail, and gamut operators used by the renderer.
- `isp_control_pole_descriptors.json`: active direct-control descriptor table.
- `exp38_learn_poles.py`, `exp42_phrase_pole_calibration.py`, `exp47_relevance_calibration.py`, `exp49_direct_isp_comparison.py`: optimizer, phrase statistics, q90 aggregation and relevance gate.
- `exp55_large_kernel_detail_retune.py`: one-axis retuner used for v14 brightness and contrast poles.
- `exp81_descriptor_tail_calibration.py`: rebuilds all pole endpoints and centroids from saved phrase-image autotune records.
- `exp82_pole_latent_mds.py`: plots selected pole centroids in a metric-MDS projection of normalized CLIP embeddings.
- `legacy/`: completed SigLIP comparison tools, earlier protection discovery scripts, and the five-axis seed descriptors.
- `render_pole_centroids.py`: isolated endpoints and projected text-centroid galleries.
- `differentiable_isp.py`: shared RAW front-end and legacy ISP utilities; not the active semantic renderer.
- `test_*.py`: active renderer, projection, gate, and direct-calibration tests.

## Commands

Historical Exp73 v8 strength retune:

```bash
python experiments/semantic_isp/retune_clip_strength.py \
  --out /path/to/new/exp73_clip_extreme_descriptors \
  --iters 100 --device cuda:0
```

Use a fresh output directory. This historical run performs 560 fresh CLIP phrase/image optimizations. Add `--smoke` for a one-image orchestration check.

Rebuild the selected Exp81 calibration from the saved Exp80 phrase-image tuning records:

```bash
python experiments/semantic_isp/exp81_descriptor_tail_calibration.py --base /data/data1/semantic_isp/results/exp80_v14_additive_contrast/contrast --out /path/to/new/exp81_joint_descriptor_tail_final --images 2 33 384 --device cuda:0
```

The rebuild reuses 1,344 saved phrase-image tuning records (16 descriptors per pole x 7 images x 12 poles). It selects two descriptors per pole; both the q90 endpoint and centroid use this descriptor-level distribution. Endpoints are listed in `pole_quantile_statistics.csv`, and the isolated gallery uses the normalized selected centroids.

Plot one axis in two dimensions using metric MDS:

```bash
python experiments/semantic_isp/exp82_pole_latent_mds.py --axis warmth
```

The plot uses Euclidean distances among all normalized CLIP descriptor embeddings and all 12 selected pole centroids. It reports MDS stress and pole-distance retention; 2D cannot preserve every distance from the 512D space.

Terminology: **descriptors** are all 16 phrases per pole, each autotuned and used to rank per-phrase medians and fit the relevance gate. **Selected phrases** are the top two descriptors per pole by tuned magnitude; only these two embeddings are averaged to make that pole's centroid.

Evaluate unseen corpus words on held-out images, with each word-image pair in its own page:

```bash
python experiments/semantic_isp/exp80_word_probe.py --unseen-count 10 --images 2 33 384 --skip-sheets --out /data/data1/semantic_isp/results/exp81_joint_descriptor_tail_final/unseen_words10_heldout
```

The probe uses only `unseen_test_words` and the held-out image split. It records per-word axis coordinates and ISP controls in `word_mapping.csv`, with per-image cosine deltas in `image_scores.csv`.

Compare four direct semantic words with three less-specific words across five images:

```bash
python experiments/semantic_isp/exp80_word_probe.py --words vivid bright dark warm dreamy retro angelic --images 2 33 384 4 6 --out /data/data1/semantic_isp/results/exp81_joint_descriptor_tail_final/words7_images5
```

Images 2, 33, and 384 are the registered held-out split. Images 4 and 6 were not used for autotuning but are outside that fixed three-image evaluation split. Each image gets its own neutral-plus-words sheet, and each individual pair remains on a separate page.

Historical Exp73 render example:

```bash
python experiments/semantic_isp/semantic_isp.py \
  --config /data/data1/semantic_isp/results/exp73_clip_extreme_descriptors/03_positive_saturation/direct_soft_calibration.json \
  --out /path/to/new/render --images 2 33 384
```

Historical SigLIP comparison: `/data/data1/semantic_isp/results/exp70_clip_vs_siglip_all_axes/index.html`. Historical easy/hard inference comparison: `/data/data1/semantic_isp/results/exp72_backbone_inference_easy_hard/index.html`.

Root-level compatibility modules remain for older experiment scripts. New code and commands should use this directory.
