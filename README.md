# Semantic ISP Tuning

Python experiments for language-guided camera ISP research. The active semantic-pole
study uses OpenAI CLIP text centroids and six bounded direct ISP controls. The repository
also retains an earlier SigLIP per-image optimizer as a separate historical experiment.

## Code

- [Active semantic ISP pipeline](experiments/semantic_isp/README.md): current
  source, descriptors, tests, and reproduction commands.
- [Legacy differentiable ISP](experiments/differentiable_isp.py): compatibility
  import for older five-control experiments.
- [Legacy per-image experiment](experiments/archive/numbered/exp07_emovis_diff.py): SigLIP optimization
  with semantic, realism, and beauty losses; separate from the active pole study.
- [Historical multi-image runner](experiments/archive/numbered/exp14_ten_images.sh): runs the legacy experiment
  across a selected image set.
- [Regression tests](experiments/test_differentiable_isp.py): gamut protection,
  luminance preservation, and gradient checks.

The supported semantic-pole implementation is isolated in
[`experiments/semantic_isp/`](experiments/semantic_isp/). Superseded numbered
studies and their tests live in [`experiments/archive/numbered/`](experiments/archive/numbered/).
The few root-level ISP modules are compatibility shims for those archives.

## Semantic-axis ISP

[Canonical research record](docs/SEMANTIC_ISP_RESEARCH.md) documents the active
six-axis method, descriptors, text centroids, single-control autotuning, q90 endpoints,
results, provenance, related work, and limitations.

[Current-method manifest](experiments/semantic_isp/CURRENT_METHOD.json) names the
renderer, descriptor revision, autotune entry point, protection status, and latest result root.

[Current manuscript](papers/semantic_isp_tip.tex) and its figures are grouped in
[`papers/`](papers/README.md). It describes the Exp81 CLIP calibration and makes
no human-preference claim.

[Experiments index](experiments/README.md) separates the active source bundle from
historical and supporting experiment scripts.

## Environment and data

The tested runtime and reproduction steps are in
[`docs/REPRODUCING.md`](docs/REPRODUCING.md). The frozen Exp81 calibration JSON
is bundled under `experiments/semantic_isp/calibration/`; raw images and model
weights are not.

The FiveK release is centrally managed outside Git to avoid copies of licensed
RAW/TIFF and render assets. See [`datasets/README.md`](datasets/README.md) for
the authoritative release pointer, checksums, and access requirements.

## Tests

The active semantic-pole tests use the standard-library `unittest` runner and do
not require the image dataset or a GPU:

```bash
python -m unittest discover -s experiments/semantic_isp -p 'test_*.py'
```

Historical SigLIP, per-image optimization, and numbered calibration studies are
preserved under `experiments/archive/numbered/`; they are not current entry points.

## Repository scope

This repository contains source, tests, the current manuscript, its figures,
and the canonical research/reproduction notes. Licensed source imagery, model
weights, generated results, reference papers, previous iterations, and caches
remain outside Git.

## License

[MIT](LICENSE).
