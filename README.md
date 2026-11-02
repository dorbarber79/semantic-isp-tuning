# Semantic ISP Tuning

Python experiment code for text-guided optimization of camera ISP parameters.

The current differentiable pipeline controls white balance, saturation, brightness,
contrast, and sharpness. It uses SigLIP ViT-B/16 for image-text scoring and
luminance-preserving chroma compression to protect RGB channel headroom.

## Code

- [Differentiable ISP](experiments/differentiable_isp.py): RAW preprocessing and
  the five-parameter PyTorch pipeline.
- [Main experiment](experiments/exp07_emovis_diff.py): per-image optimization with
  semantic, realism, and beauty losses.
- [Multi-image runner](experiments/exp14_ten_images.sh): runs the main experiment
  across a selected image set.
- [Regression tests](experiments/test_differentiable_isp.py): gamut protection,
  luminance preservation, and gradient checks.

Other scripts in [experiments](experiments/) are individual experiments. Historical
scripts can use different models and loss configurations; the main experiment
contains the current configuration.

## Environment and data

The main pipeline uses Python with PyTorch, NumPy, rawpy, OpenCV, open_clip,
Pillow, and Matplotlib. Optimization scripts expect a CUDA GPU and SigLIP model
weights.

Datasets, the style-word corpus, model weights, and experiment outputs are external
to this repository. Update the dataset and output paths in the experiment scripts
for your environment before running them.

## Tests

The regression tests use the standard-library `unittest` runner and do not require
the image dataset, SigLIP weights, or a GPU:

```bash
python experiments/test_differentiable_isp.py -v
```

## Repository scope

This repository contains source code and tests. Local research documents,
reference papers, the previous iteration, virtual environments, caches, and
generated data/results are intentionally excluded.

## License

[MIT](LICENSE).
