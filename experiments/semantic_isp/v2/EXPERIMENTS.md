# V2 ISP Experiment Plan

V2 begins from the byte-identical V1 baseline in `BASELINE_MANIFEST.json`. Keep
V1 frozen. Every experiment gets a fresh output directory and records input
hashes, metadata, source/config hashes, resolution, precision, and processing
parameters. Change one factor per experiment; do not compare runs with different
inputs or crops.

## Decision Rules

- **E0 Baseline parity:** same prompt, RAW, config, software, device and output
  dimensions must reproduce identical projected coordinates and control values.
  Compare rendered pixels bitwise first; if a backend prevents that, record max,
  mean and p99 absolute error and the reason.
- **E1 Exposure normalization:** compare current p99 gain, no gain, and bounded
  alternative statistics on a fixed stratified RAW set. Do not use shutter/ISO
  alone as scene brightness correction. Report gain in stops, mean/median display
  luma, p99, black fraction, highlight clip fraction, and exposure metadata.
- **E2 Metadata handling:** inventory EXIF/DNG/XMP fields; separately test
  camera-as-shot WB, available RAW camera matrices, and explicit XMP develop
  settings where an independent renderer can apply them. Never treat a saved
  snapshot as the active develop state without verifying it.
- **E3 RAW decode/highlights:** compare current AHD, camera WB, no-auto-bright,
  and highlight-clip choices one at a time. Record decoded levels, clip fraction,
  color statistics, and render differences.
- **E4 Precision/export:** compare uint16 intermediates, float32 intermediates,
  16-bit PNG/TIFF, and current 8-bit sRGB PNG. Measure round-trip max/mean error,
  PSNR, clipped samples and file size. Label PNG compression lossless but
  quantization separately; label JPEG quality and chroma subsampling.
- **E5 Resolution/crop:** compare native, 512 square center crop, and documented
  aspect-preserving alternatives. Record source/output dimensions and resize
  kernel. Evaluate crop and full-frame metrics separately.
- **E6 Denoise/detail scale:** compare the current bilateral settings and
  resolution-scaled Gaussian detail at fixed inputs/resolutions. Measure edge
  contrast and noise/detail proxies; do not infer full-resolution sharpness from
  reduced sheets.
- **E7 Color/contrast:** compare the current order and per-operation gamut
  compressor with candidate alternatives using fixed controls. Track luminance,
  chroma, clipping, and color difference independently.

## Required Per-Run Record

| Field | Record |
|---|---|
| Experiment ID / hypothesis | One testable change and expected observation |
| Input | File path, checksum, camera, capture metadata, RAW dimensions |
| Code/config | V2 source hashes and complete calibration/config hash |
| Pipeline | Decode flags, metadata fields consumed, gain, denoise, resize/crop |
| ISP | Prompt, CLIP model/weights, coordinates, physical controls and bounds |
| Export | Pixel dimensions, color space, channel count, bit depth, file format |
| Measures | p99/median luma, clipping, color error, PSNR, file size, runtime |
| Decision | Keep/reject/defer with observed evidence; never overwrite V1 |

## Run Log

| ID | Status | Change | Evidence / decision |
|---|---|---|---|
| E0 | Passed | Copy V1; repair V2 report serialization only | 36 V2 tests pass. On DNG 1 / `warm`, input tensor, coordinates and 3040x2014 uint8 output match exactly; max pixel difference 0. V1 CLI fails before render when JSON serialization reaches dict-valued protection metadata. |
| E1-E7 | Not run | No change selected | Await measured results. |
| E8 | Exploratory, completed | Rank crop-only photographer fits by joint native-size PSNR/SSIM, then test Tint and Vibrance individually and together | From 75 pairs across 15 completed scenes, the lowest 20 combined-percentile-rank pairs span 7 scenes. Mean deltas vs six-control baseline: Tint +0.325 dB / +0.00605 SSIM; Vibrance +0.0666 dB / +0.00135; both +0.436 dB / +0.00769. Joint-vs-Tint marginal gain is +0.111 dB mean, +0.0046 dB median; two pairs drive most of that gain and Vibrance reaches its bound on both. This hard set is selected and optimized in-sample; results are exploratory, not held-out generalization. Optional renderer parameters are not in the six-control semantic calibration and are not adopted. See external `analysis_results/hard_fit_crop_only_v1/`, `analysis_results/tint_dof_hard_fit_v1/`, `analysis_results/vibrance_dof_hard_fit_v1/`, and `analysis_results/tint_vibrance_dof_hard_fit_v1/`. |