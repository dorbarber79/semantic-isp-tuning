# Reproducing the Current Results

## Supported Method

The supported method is the six-control Exp81 semantic-pole ISP using OpenAI
CLIP ViT-B/32. The frozen config is
[`experiments/semantic_isp/calibration/exp81/direct_soft_calibration.json`](../experiments/semantic_isp/calibration/exp81/direct_soft_calibration.json).
Its SHA-256 is recorded in
[`experiments/semantic_isp/CURRENT_METHOD.json`](../experiments/semantic_isp/CURRENT_METHOD.json).
Do not use the archived numbered experiments as current entry points.

## Tested Environment

The active source was tested with Python 3.12.9, PyTorch 2.7.1 and CUDA 12.6,
NumPy 2.2.5, rawpy 0.25.1, OpenCV 4.11.0.86, open_clip_torch 3.3.0, Pillow
11.2.1, Matplotlib 3.10.3, SciPy 1.16.2, and scikit-learn 1.8.0. Direct
Python package pins are in [`requirements-active.txt`](../requirements-active.txt).
Use the CUDA 12.6 PyTorch wheel for GPU runs:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-active.txt
```

The exact Python, PyTorch, CUDA, and library versions matter. GPU kernels and
driver differences may still cause small floating-point variation; the frozen
calibration file makes request-time outputs independent of rerunning autotuning.

## Checks and Runs

Run the active, data-free regression suite:

```bash
python -m unittest discover -s experiments/semantic_isp -p 'test_*.py'
```

Render the registered held-out images using the frozen Exp81 calibration. First
make the licensed RAW files available at the paths expected by the renderer and
ensure OpenAI CLIP weights can be loaded by `open_clip`:

```bash
python experiments/semantic_isp/semantic_isp.py \
  --config experiments/semantic_isp/calibration/exp81/direct_soft_calibration.json \
  --out /path/to/new/exp81_render --images 2 33 384
```

Rebuild the Exp81 endpoints/centroids only when the original Exp80 tuning records
and calibration images are available. This aggregation reuses 1,344 saved
phrase-image records; it does not recreate missing records:

```bash
python experiments/semantic_isp/exp81_descriptor_tail_calibration.py \
  --base /path/to/exp80_v14_additive_contrast/contrast \
  --out /path/to/new/exp81_rebuild --images 2 33 384 --device cuda:0
```

For FiveK fitting, use the dedicated scripts documented in
[`experiments/semantic_parameter_learning/README.md`](../experiments/semantic_parameter_learning/README.md).
`fivek_photographer_fit.py` fits one source image; `fivek_batch.py` runs or
resumes the multi-image workflow; `package_fivek_dataset.py` creates immutable
dataset releases. The original FiveK masters and the 45 GB v3 release are not
included in Git. See [`datasets/README.md`](../datasets/README.md) for access,
verification, and licensing requirements.

Every run should use a new output directory. Keep source checksums, config,
model identifier, optimizer settings, image split, and output manifest with the
results. Do not claim bitwise reproduction across different GPU/runtime stacks.