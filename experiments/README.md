# Experiments

## Active semantic-pole work

Current code and descriptors are grouped in [`semantic_isp/`](semantic_isp/README.md). Use [`semantic_isp/CURRENT_METHOD.json`](semantic_isp/CURRENT_METHOD.json) for the machine-readable source of truth, and [`semantic_isp/autotune.py`](semantic_isp/autotune.py) for the current four-mode autotune. Root-level modules with matching names are compatibility shims for earlier imports; do not edit those shims as the source of truth.

## Historical and supporting work

The numbered scripts and their tests are under [`archive/numbered/`](archive/numbered/).
They document earlier ISP, SigLIP, cinematic-axis, corpus-evaluation, and diagnostic
iterations; they are retained for provenance and are not supported current entry
points. The active renderer, Exp81 calibration, descriptors, and focused tests are
under [`semantic_isp/`](semantic_isp/README.md).

The repository-level [current-method manifest](semantic_isp/CURRENT_METHOD.json)
and [reproduction guide](../docs/REPRODUCING.md) are the current source of truth.
The older research record is explicitly labeled as historical. Experiment outputs,
RAW imagery, and model weights remain external under their documented roots.
