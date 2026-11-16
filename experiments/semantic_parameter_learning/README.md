# Semantic Parameter Learning

This experiment is isolated from the active pole pipeline. It imports the existing
six-control differentiable ISP without editing that pipeline, and writes generated
data and models under a separate results directory. The active calibration is
used only as the renderer and control-bound configuration; this experiment does
not replace its text mapping or pole settings.

## Data Generation

`sentences_agent_estimates.json` is a separate proposal for the next dataset:
72 sentences, 18 for each Moran D1-D4 group, with direct, low, medium, and high
ambiguity levels. Each sentence contains six agent-estimated adjustments in
[-1, 1]; 0 means no adjustment is inferred from the text alone. These are
subjective semantic estimates, not physical ISP parameters, human annotations,
or autotune results. In particular, brightness is a direction and strength;
the physical `tone_gain` control is neutral at 1, not 0. The palette score
specifically represents magenta-green versus teal-orange attraction, not general
saturation or warmth. Ambiguous phrases can have nonzero estimates as plausible
interpretations, but need independent review; zero is also used when no specific
adjustment can be inferred. No train/validation/test assignment is claimed for
this proposal: future splitting must keep paraphrase families disjoint.

This proposal is not the generator input below. A few of its sentences were
autotuned in the separate FiveK portrait experiments; the complete 72-sentence
table has not been autotuned.

For a quick visual check, `fivek_preview.py` selects five sentences from the
agent-estimate table and optimizes seven sentence-image pairs across three FiveK
images. It uses preprocessed 16-bit linear RGB PNGs, not the original RAW files.
It optimizes a 256-pixel short-side image, renders the resulting controls on the
720p source, and saves two PNG sheets: one image with five sentences, and one
sentence on three images. To repeat into a fresh output directory:

```bash
python experiments/semantic_parameter_learning/fivek_preview.py \
  --iters 20 --out /path/to/new/fivek_preview
```

This is a quick visual preview, not a converged dataset or a held-out evaluation.
The saved `results.json` contains the exact phrases, controls, selected restarts,
loss components, and input paths. Agent-estimated ratings are recorded for
comparison only; they are not used in the optimization objective.

For convergence-monitored portrait comparisons, run:

```bash
python experiments/semantic_parameter_learning/fivek_portraits.py \
  --semantic-backbone siglip --out /path/to/new/fivek_portraits
```

The portrait experiment optimizes normalized P90 control coordinates: -1 is the
negative calibrated pole, 0 is neutral, and +1 is the positive calibrated pole.
Each side is mapped linearly to its corresponding physical ISP endpoint; physical
and normalized values are both retained. The P90 endpoints come from the supplied
active calibration and need not be symmetric around neutral. Raw optimizer
coordinates are passed through `tanh`, so each setting stays within the pole
range. Regularization in this mode is measured in raw normalized coordinates,
and therefore differs from the older physical-bound pilot's regularization.

The annotated portrait PNGs show the sentence, descriptor group, ambiguity,
agent estimate, normalized result, physical ISP setting, objective components,
quality proxy scores, and convergence status. Stability criteria and full
per-restart traces are saved in the result. A converged proxy objective does not
guarantee that the image matches the request. In the first P90 SigLIP run, the
request for a much brighter image still produced negative normalized brightness.

Generate separate three-image comparisons of neutral, agent estimate, and saved
autotune results without rerunning optimization:

```bash
python experiments/semantic_parameter_learning/fivek_agent_comparison.py \
  --out /path/to/new/fivek_agent_comparisons
```

The agent image applies the original agent-estimated P90 coordinates directly;
it is not autotuned. The report shows the agent and autotune controls in both P90
and physical units. The renderer reproduces and checks the saved neutral and
autotuned images before making the comparison. A compatible extended result can
replace a weaker original result if its selected restart converged and has a
lower objective; the source and alternative-restart convergence remain recorded.

To inspect individual loss components for one image and one sentence:

```bash
python experiments/semantic_parameter_learning/loss_weight_study.py --out /path/to/new/loss_study
python experiments/semantic_parameter_learning/refine_quality_loss.py \
  --base /path/to/new/loss_study --out /path/to/new/refinement
```

The study compares quality weights 0, 1, and 3 with identical P90 bounds and
SigLIP prompts. Each convergence PNG shows total, semantic, realism, beauty,
and regularization traces for all restarts. Semantic, realism, and beauty are
shown unweighted; regularization already includes its coefficient. The stopping
test checks the combined objective, gradient, and control stability, not each
individual component. A last-window range for each component is saved in each
summary, and a capped run is marked nonconverged.

To run only the original weight-3 procedure on a different image and sentence,
without warm-start refinement:

```bash
python experiments/semantic_parameter_learning/loss_weight_study.py \
  --conditions stronger_quality --image-id a0008 --sentence-id D1_008 \
  --out /path/to/new/weight3_example
```

To compare the selected MIT-licensed FiveK examples against all five expert
TIFFs and fit the nearest six-control ISP rendering to each target:

```bash
python experiments/semantic_parameter_learning/fivek_photographer_fit.py \
  --out /data/data1/FiveK/C_data/results/isp_match_v2
```

For the resumable 105-image study (the five original images plus 100 randomly
selected licensed images), run `python experiments/semantic_parameter_learning/fivek_batch.py`.
It downloads the DNG and five original TIFFs for each image, saves each image as
an independently validated checkpoint under `results/isp_match_105_v3/images/`,
and maintains `progress.json`, `index.json`, and combined control and descriptor
CSVs. Repeating the same command resumes completed images, checks source hashes,
and retries failures. Use a fresh output directory when changing the code or
settings. `tmux` can keep this process running after disconnecting.

The saved expert, default ISP, and fitted previews have identical dimensions;
the original TIFFs remain at their native resolution. The older v3 fit output
contains control-derived D1-D4 templates for diagnostics. These are not the
current visual descriptions and are not used in the fit-description report.

For human-language descriptors and compact comparisons, run:

```bash
python experiments/semantic_parameter_learning/fivek_semantic_descriptions.py --watch
```

The separate `results/isp_match_105_semantic_v1/` directory contains an HTML
index, one figure per image showing Default ISP once and each expert beside its
fit, and a CSV with four phrases per D1-D4 group. D4 candidates are ranked using
the earlier word-to-control spreadsheet, with clear opposite directions filtered
out. These phrases are subjective, control-derived suggestions, not independent
visual assessments, expert-authored descriptions, or validated training labels.
The watcher follows new completed checkpoints without changing the fitting run.

The current landscape report compares Default ISP with each fitted render.
Reviewed fit descriptions are manually authored from those renders and stored
in `fivek_fit_visual_descriptions.json`; unreviewed fits show a pending notice
rather than generated text. See
[`FIVEK_FIT_ANNOTATION_PROTOCOL.md`](FIVEK_FIT_ANNOTATION_PROTOCOL.md) for the
annotator provenance, reconstructed guidance, visual-input caveat, and checks.
Current reviewed and pending counts are recorded in the JSON metadata. The
separate filter names are approximate look references, not verified matches to
any app preset.

The named dataset and provenance manifest is
[`fivek_isp_visual_annotation_dataset.json`](fivek_isp_visual_annotation_dataset.json).
FiveK-ISPVA-105 contains D1-D4 descriptions for all 105 scenes / 525 fits;
assistant-authored style names, approximate Instagram/app references, and
expert-versus-Default prompt annotations cover only 11 scenes / 55 fits.

Refresh the existing screen report in place with:

```bash
python experiments/semantic_parameter_learning/fivek_screen_comparisons.py \
  --source /data/data1/FiveK/C_data/results/isp_match_105_v4 \
  --out /data/data1/FiveK/C_data/results/isp_match_105_screen_v22
```

## Centralized Dataset Releases

[`package_fivek_dataset.py`](package_fivek_dataset.py) builds immutable,
self-contained snapshots under `/data/data1/FiveK/datasets/FiveK-ISPVA/`.
The initial release consolidates the original 105 scenes and the 20-scene pilot.
`latest.json` identifies the most recent completed release. Existing source data
are neither removed nor overwritten, and partial builds are not advertised as
completed releases.

```bash
python experiments/semantic_parameter_learning/package_fivek_dataset.py \
  --release 2026-11-14-v1
```

After additional annotations, build a new explicitly named release. Existing
releases refuse overwrites.

Verify a completed release and all content checksums with:

```bash
python experiments/semantic_parameter_learning/package_fivek_dataset.py \
  --verify-release /data/data1/FiveK/datasets/FiveK-ISPVA/releases/2026-11-14-v1 \
  --full-checksums
```

The release structure is:

```text
releases/<release>/
  dataset_manifest.json
  assets.json
  source_path_map.json
  SHA256SUMS
  scenes/<image_id>/original/raw.dng
  scenes/<image_id>/default/{linear_720,display_720}.png
  scenes/<image_id>/experts/<A-E>/{original.tif,display_720.png,fit_720.png}
  annotations/source_documents/
  annotations/fit_visual_descriptions.json
  annotations/normalized_pairs.json
  calibrations/
  provenance/code/
  archives/
  statistics/
  screen_report/index.html
```

The authoritative pointer currently resolves to `2026-11-15-v5` through
`/data/data1/FiveK/datasets/FiveK-ISPVA/latest.json`. Its manifest records 145
annotated scenes, 725 fits, 2,900 D1-D4 texts, and zero pending scenes. It also
consolidates the latest analysis reports under `reports/latest/`. The v5 release
passed full staged SHA-256 verification before activation. It remains an internal
candidate pending licensing review. Do not edit the pointer by hand;
build immutable releases through this packager.

Releases v2 through v4 remain preserved as prior snapshots. The
unified description JSON retains source/global and original scene provenance
rather than inventing missing AI metadata. Source annotation files remain
unchanged for audit.

Primary pair records retain full fit metadata, normalized and physical controls,
optimization losses/restarts, ICC checks, source checksums, image statistics,
all matching annotation layers, and per-scene annotation provenance. Source text
documents and historical result artifacts are copied without rewriting their
contents, so generated template descriptions, visual observations, semantic
words, original look names, approximate app references, and predictions remain
available with their distinct roles. Legacy absolute paths are mapped to release
paths by `source_path_map.json`.

Original RAW/DNG and full-resolution expert TIFF masters are retained. The
collected default/fit renders are 720 pixels high (not uniformly a 720-pixel
short side); linguistic annotation used these and smaller comparison sheets.
Full-resolution default RGB and fitted outputs were not collected, and no
upscaled image is called an original. Sharpness/detail phrases are not validated
full-resolution sharpness labels. The original AI model identifier was unknown;
the GPT-5.6 Luna screenshot is only reported verification evidence. Pilot model
metadata and any visual uncertainty are preserved verbatim.

The statistics directory contains per-pair controls and fit errors, mean/median,
population standard deviation and quantiles in normalized and physical domains,
expert-specific summaries, fixed-bin histograms, paired A-E control differences,
an expert-distance heatmap, and annotation coverage. It describes fitted values,
not photographers' original editing parameters or human quality judgments.

Files are independent copies of source bytes; content-identical files are
hardlinked only within the release. Do not edit release object files in place.
Licensing and redistribution clearance, reference verification, and independent
annotation validation remain required before publishing with the paper.

The central dataset index and the current SVM comparison are available from
`/data/data1/FiveK/datasets/FiveK-ISPVA/index.html` and
`/data/data1/FiveK/datasets/FiveK-ISPVA/releases/2026-11-15-v5/reports/latest/d1_text_linear_svm/index.html`.
The combined 145-scene visual report is at
`/data/data1/FiveK/datasets/FiveK-ISPVA/releases/2026-11-15-v5/screen_report/index.html`.
The complete release is intentionally not duplicated in this Git repository.

## Bounded FiveK Collection and Visual-Review Agent

The user-authorized pilot is 20 new scenes, excluding the original 105 scenes.
[`fivek_collection_worker.py`](fivek_collection_worker.py) reuses the licensed
MIT list, existing Default ISP renders, validated DNG/TIFF downloads, and the
500-iteration five-expert fitter. It preserves completed results and interrupted
directories, checks free disk space, and locks against duplicate worker processes.
It does not generate annotations from controls or call an automatic VLM.

```bash
python experiments/semantic_parameter_learning/fivek_collection_worker.py --prepare-only
python experiments/semantic_parameter_learning/fivek_collection_worker.py --watch
```

The worker remains active after finishing the authorized 20 fits, refreshing the
visual-review queue when annotations change. It never expands the 20-scene limit
on its own. Restart with the same arguments to resume; `--retry-failed` explicitly
retries failed jobs. Stop gracefully using the PID in `worker_pid.json` and
`kill -TERM PID`, or create a `STOP` file in the pilot directory (checked between
jobs). Do not force-kill while fitting. Files are stored under
`/data/data1/FiveK/C_data/results/isp_match_20_pilot_v1/`:
`collection_manifest.json`, `collection_state.json`, `annotation_queue.json`,
`fit_visual_descriptions.json`, per-scene fits, and `screen_report/`.

The workspace agent
[FiveK Visual Annotator](../../.github/agents/fivek-visual-annotator.agent.md)
reviews queued fits when invoked in chat. It inspects actual Default/fit renders,
writes A-E D1-D4 descriptions, records per-scene provenance and uncertainty, and
can add explicitly approximate style/app references. It cannot annotate while
no chat invocation is active. The queue does not mark a scene reviewed without
complete descriptors and provenance matching the source-result hash.

The pilot uses a separate annotation JSON and report; the original 105-scene
annotations, dataset manifest, and reports are not modified. The optional
`--descriptions` argument of `fivek_screen_comparisons.py` selects the pilot
annotation document when refreshing its report.

## Image-Conditioned CLIP + SVR Baseline

[`fivek_clip_svr.py`](fivek_clip_svr.py) selects one expert per scene, using a
seeded balanced assignment (21 scenes for each of A-E in the 105-scene run).
Frozen OpenAI CLIP ViT-B/32 encodes only the Default ISP display image and the
selected fit's descriptions. Image and text embeddings are separately
L2-normalized and concatenated, following the official OpenAI CLIP similarity
example. There is no additional feature standardization or CLIP fine-tuning.

Six RBF SVRs predict the P90-normalized controls jointly through a multi-output
wrapper; each scalar regressor has fixed `C=10`, `epsilon=0.05`, and
`gamma="scale"`. These settings were not tuned on held-out targets. Training
uses the 105 selected pairs, each with five text variants: D1, D2, D3, D4, and
their concatenation. The combined text also includes existing fit-style names
and approximate app references where available. Historical control-derived
drafts and Expert-versus-Default prompt annotations are not used as training text.

Five-fold evaluation keeps every text variant of a scene together. Each
prediction comes from a model trained on the other 84 scenes; no expert or fit
image is embedded. The separate all-data model is saved for future inference
and is not used to report these results.

```bash
HF_HUB_OFFLINE=1 python experiments/semantic_parameter_learning/fivek_clip_svr.py \
  --out /data/data1/FiveK/C_data/results/isp_match_105_clip_svr_v1
```

The command refuses a nonempty output directory. Remove `HF_HUB_OFFLINE=1` if
the OpenAI checkpoint has not been cached. Use `--limit 5` and a different output
directory for a pipeline smoke check, not a reported accuracy experiment.

The full run saves 105 existing-format 1800-by-1100 comparison figures, 525
predicted renders, source records and selected experts, raw CLIP embeddings and
fused features, five fold models and one all-data model, fold assignments,
per-input predictions and errors, an HTML index, and a protocol/metrics manifest.

Each comparison panel shows the saved expert ISP fit as the GT target (not
the original expert TIFF), the SVR prediction, the exact descriptor text used,
and both GT and predicted six-control P90 vectors. To redraw only the existing
figures without changing models, predictions, or metrics:

```bash
python experiments/semantic_parameter_learning/fivek_clip_svr.py \
  --refresh-figures-only \
  --out /data/data1/FiveK/C_data/results/isp_match_105_clip_svr_v1
```

| Text input | Held-out P90 MAE | Display RGB MAE vs saved ISP fit |
| --- | ---: | ---: |
| D1 | 0.2424 | 0.0806 |
| D2 | 0.2477 | 0.0801 |
| D3 | 0.2452 | 0.0806 |
| D4 | 0.2412 | 0.0779 |
| Combined | 0.2350 | 0.0779 |

The training-fold median-control baseline has P90 MAE 0.2143, so this initial
SVR does not outperform it. The combined input is best among the tested text
variants but not a successful generalization result. All 525 text inputs fit
within CLIP's context without truncation. Descriptions remain assistant-authored
appearance notes, not independently collected user requests; CLIP image inputs
use its native resizing and center cropping, not full-resolution detail analysis.

### D1 Brightness-Only Study

[`fivek_brightness_svr.py`](fivek_brightness_svr.py) reuses the same selected
105 scene/expert pairs and cached CLIP features, but predicts only normalized
`tone_gain` from D1. Text-only and image-plus-text linear/RBF SVRs use five
scene-disjoint outer folds. Three inner scene folds choose `C`, `epsilon`, and
RBF `gamma` without examining the outer test targets.

```bash
python experiments/semantic_parameter_learning/fivek_brightness_svr.py
```

Results are saved under
`/data/data1/FiveK/C_data/results/isp_match_105_d1_brightness_svr_v1/`, including
105 comparison figures, a pooled held-out trend plot, predictions, twenty fold
models, fold-specific hyperparameters, and protocol/metrics records. Figure
targets and predictions change brightness only; all other controls are neutral.
The full original fit is retained in each scene's metadata, not used as the
isolated brightness target image.

| Method | Brightness P90 MAE | R-squared | Pearson correlation |
| --- | ---: | ---: | ---: |
| Text linear SVR | 0.3563 | -0.0674 | 0.0499 |
| Text RBF SVR | 0.3580 | -0.0471 | 0.0867 |
| Image + text linear SVR | 0.3863 | -0.2003 | 0.0798 |
| Image + text RBF SVR | 0.3680 | -0.0956 | 0.0402 |
| Training-fold median | 0.3582 | -0.0397 | -0.1696 |

No useful held-out trend in brightness magnitude was established. Text-only
linear regression is effectively tied with the median baseline. Targets include
76 positive, 15 negative, and 14 near-neutral values (threshold 0.1). The median
baseline already achieves 83.5% sign accuracy among non-neutral targets; sign
accuracy alone is therefore misleading. Targets were originally jointly fitted
with other controls, and many D1 descriptions do not explicitly mention brightness.

### All Controls: D1 Text-Only Linear SVR

The same runner can repeat the nested scene-held-out study for all six controls,
using only normalized D1 CLIP text embeddings and a linear SVR kernel:

```bash
python experiments/semantic_parameter_learning/fivek_brightness_svr.py \
  --all-controls --text-linear-only \
  --out /data/data1/FiveK/C_data/results/isp_match_105_d1_text_linear_all_controls_v1
```

Each control is trained independently on the same selected 105 scene/expert pairs.
The outer folds and inner hyperparameter search match the brightness study. No
image embeddings enter these models. Each subdirectory saves five fold models,
predictions, metrics, a trend plot, and 105 comparison figures isolating the
selected control. The root index summarizes all six studies (630 figures total).

| Control | Linear SVR P90 MAE | Median baseline MAE | R-squared | Pearson correlation |
| --- | ---: | ---: | ---: | ---: |
| Warmth | 0.3077 | 0.3787 | 0.2686 | 0.5368 |
| Saturation | 0.2942 | 0.2994 | 0.1066 | 0.3318 |
| Brightness | 0.3563 | 0.3582 | -0.0674 | 0.0499 |
| Contrast | 0.1417 | 0.1406 | -0.0011 | 0.1029 |
| Sharpness | 0.0556 | 0.0533 | 0.0055 | 0.1101 |
| Palette | 0.0592 | 0.0554 | -0.0313 | -0.0683 |

Warmth exhibits the clearest held-out association and approximately 18.7% lower
MAE than the median baseline, although extreme values remain poorly predicted.
Saturation shows a weaker association and only a small MAE improvement. The
remaining controls do not establish a useful improvement over the baseline.
These are fitted-control prediction diagnostics, not perceptual quality results;
small sharpness/palette errors can reflect mostly near-neutral targets.

### Composed D1 Linear Prediction Screen Report

[`fivek_text_linear_report.py`](fivek_text_linear_report.py) applies the six
saved independent text-only linear SVRs to each expert's D1 description, then
composes all six predicted controls into one ISP render. It performs no training
and does not embed the image. Every scene uses its held-out fold model for all
six controls; the models were trained on one selected expert per training scene.

```bash
HF_HUB_OFFLINE=1 python experiments/semantic_parameter_learning/fivek_text_linear_report.py
```

Output is saved under
`/data/data1/FiveK/C_data/results/isp_match_105_screen_d1_text_linear_v1/`.
The original v22 report is unchanged. The same 1800-by-1100 layout shows Default
ISP once and five expert panels, each containing Expert TIFF, target ISP fit,
and predicted ISP render. Panels show the exact D1 input, GT and predicted
P90 controls, parameter MAE, image MAE on saved 8-bit sRGB pixels, and PSNR.
Image scores use the fit target, not the expert TIFF. Lower MAE and higher PSNR
are better; these are numerical errors, not human preference scores.

The report includes 105 comparison figures and 525 predictions. Aggregate
parameter MAE is 0.2054, image MAE is 18.45/255, and mean per-image PSNR is
22.96 dB. For the original selected 105 pairs, parameter MAE is 0.2024 and
image MAE is 19.02/255. JSON/CSV data retain all controls, input texts, source
fits, fold IDs, scores, and model hashes. D2-D4 and filter references are never
used as prediction inputs in this report.

### Matched D1 Versus D1-D4 Linear SVR Comparison

[`fivek_all_descriptors_svr.py`](fivek_all_descriptors_svr.py) trains six
text-only linear SVRs with exact concatenated D1-D4 descriptions, without filter
names or app-reference additions. It uses the same selected 105 training pairs,
five outer scene folds, and three-fold inner `C`/`epsilon` search as D1. All
525 held-out scene/expert pairs are compared against the saved D1 predictions.

```bash
HF_HUB_OFFLINE=1 python experiments/semantic_parameter_learning/fivek_all_descriptors_svr.py
```

Outputs are saved under
`/data/data1/FiveK/C_data/results/isp_match_105_d1_vs_all_d_linear_v1/`: thirty
fold models, embeddings, exact input texts, paired predictions, metrics, fold
parameters, provenance hashes, and paired scene-bootstrap error intervals.
No image embedding is used, and no input is truncated (maximum 56 text tokens).

| Control | D1 MAE | D1-D4 MAE | D1 within 0.1 | D1-D4 within 0.1 |
| --- | ---: | ---: | ---: | ---: |
| Warmth | 0.3253 | 0.3589 | 27.0% | 21.7% |
| Saturation | 0.2916 | 0.2910 | 28.2% | 29.5% |
| Brightness | 0.3439 | 0.3520 | 21.1% | 17.0% |
| Contrast | 0.1493 | 0.1524 | 46.7% | 44.2% |
| Sharpness | 0.0678 | 0.0686 | 81.7% | 81.0% |
| Palette | 0.0548 | 0.0559 | 87.2% | 87.8% |

On all 525 pairs, D1-D4 increases warmth MAE by 10.3%; the grouped-scene
bootstrap interval for the absolute MAE increase is [0.0183, 0.0486]. Other
per-control MAE differences are small, with intervals spanning zero. These
intervals resample scenes with their five experts, without model refitting or
multiple-comparison adjustment; they are conditional diagnostics, not a full
training-uncertainty estimate. The threshold 0.1 is an unvalidated diagnostic
tolerance, not a human-perceptual definition of a good output.

The original selected 105-pair subset is reported separately in the metrics
JSON. In that subset, D1-D4 reduces palette MAE by 5.8%, but this benefit does
not hold across all 525 pairs; warmth MAE increases by 15.7%. More descriptor
text therefore does not establish a general improvement for this setup.

For the five images with assistant-authored visual comparisons, run
`python experiments/semantic_parameter_learning/fivek_visual_report.py`.
The separate `results/isp_match_visual_filters_v1/` report includes proposed
Instagram-style filter names, an overview for each image, and a large figure per
expert showing all four phrases in each D1-D4 group. The names are original
suggestions, not Instagram presets or verified matches to existing filters; the
visual and semantic descriptions are subjective and limited to the five reviewed
images. This report does not automatically annotate later FiveK downloads.

Six additional independently reviewed images are in
`results/isp_match_visual_filters_additional_v5/`. Their observations and
proposed filter names are recorded separately in
`fivek_visual_annotations_additional.json` and
`fivek_visual_filter_names_additional.json`.

`fivek_vlm_probe.py` tests a local Qwen2.5-VL-3B model on Default ISP/expert
image pairs without using fitted controls as input. On the beach test, it
incorrectly described the darker expert A edit as brighter with both a combined
comparison image and separate image inputs. It also missed the strong warmth of
expert E and sometimes invented details. These probes are not annotations and
the model is not approved to generate training labels without further validation.

The runner converts 16-bit ProPhoto TIFF targets to linear sRGB with an ICC-
validated transform, then fits bounded P90 controls against the existing
Default ISP linear render. It writes one Default-ISP-versus-experts sheet,
per-image expert-versus-ISP-fit sheets, and JSON/CSV values including physical ISP
controls and a short suggested prompt for each photographer. The fit is the
best global six-control approximation under its pixel/low-frequency objective;
it cannot reproduce local masks or recover the photographers' actual editing
parameters. The selected sample is from the dataset's MIT-licensed subset; its
license and source manifest are stored under
`/data/data1/FiveK/C_data/`.

For the recorded brightness example, the original weight-3 run produced a
brighter local minimum. However, the weight-1 solution had a lower objective when
rescored at weight 3. Refinement includes both prior solutions as warm starts;
its lowest weight-3 solution was dark. Different weighted totals are not directly
comparable across objectives, and prompt-pair quality scores are model proxies.

`sentences.json` contains 18 engineering-authored style phrases with separate
train, validation, and test splits. For every sentence and image, the generator
jointly tunes warmth, saturation, tone gain, contrast, sharpness, and palette.
The saved record includes the phrase, image, physical and raw ISP controls,
optimization iteration, restart, semantic similarity, quality scores, and loss
components. New runs default to P90-normalized controls and also save the
physical values and exact pole endpoints. Unit-normalized text embeddings are
stored separately by ID. The 18-sentence generator uses EmotionISP RAW inputs;
the FiveK portrait runner above uses FiveK linear PNGs.

The objective follows the earlier two-sided quality-loss formulation:

```text
loss = -semantic_image_text_cosine
     + (neutral_realism_score - rendered_realism_score)
     + (neutral_beauty_score - rendered_beauty_score)
     + 0.003 * squared_raw_distance_from_neutral
```

Each quality score is the difference of SigLIP image-to-text cosines for the
positive and negative prompt pair used by Exp07. New generation defaults to
SigLIP B/16 for both semantic and quality scores, with image mean/std 0.5;
`--semantic-backbone clip` selects the previous CLIP semantic model. These are
learned proxies, not
guarantees of real-world realism, beauty, or human preference. The two-sided
quality losses reward improvements as well as penalize degradation.

The existing palette operator has zero gradient at its neutral coordinate. The
new optimizer uses three starting palette coordinates: 0, -0.02, and +0.02. It
keeps the minimum-objective iterate across all starts, including neutral.
Source and configuration hashes are recorded; a partial run resumes only when
its manifest matches. Use a fresh output directory for changed settings.

## Run

The following is a small pipeline pilot, not a converged training dataset:

```bash
python experiments/semantic_parameter_learning/generate_dataset.py \
  --out /path/to/new/p90_pilot \
  --iters 8 --size 256

python experiments/semantic_parameter_learning/benchmark.py \
  --dataset /path/to/new/p90_pilot \
  --out /path/to/new/benchmark

python experiments/semantic_parameter_learning/test_experiment.py -v
```

The historical eight-step CLIP/physical-unit pilot contains 18 sentences x 3 images = 54 pairs. The 12 training sentences
and two training images produce 24 training pairs. The other phrases and image
are explicitly held out in separate sentence, image, and joint test groups.
Model selection uses only validation sentences on training images. The pilot has
only three pairs with both a test sentence and held-out image. In the completed
pilot, 51 of 54 best iterates were at the final eighth update, so label
convergence is not established.

## Mapping Candidates

- **Ridge regression:** a simple text-only linear baseline from normalized text
  embeddings to standardized ISP controls.
- **Support-vector regression:** an RBF SVR per control predicts continuous ISP
  settings. This is regression rather than SVM classification; fixed pilot
  hyperparameters are compared on validation data. Predictions are clamped to
  the target control bounds (P90 normalized for new datasets).
- **Image-conditioned regression:** concatenate CLIP text and neutral-image
  embeddings, optionally with image statistics, and compare ridge, SVR, or a
  small regularized MLP. This is important because per-image autotuning can
  produce different targets for the same text. Fit preprocessing only on the
  training split, and hold out entire image identities and sentence families.
- **Nearest-neighbor or kernel regression:** interpolate among training text or
  text-image embeddings; use a neutral fallback for unsupported requests and
  report uncertainty from distance to the training set.

The current benchmark implements the two text-only regressors and neutral and
training-mean baselines. It reports mean absolute parameter error normalized by
each control's half-range, plus per-control errors in the dataset's target units.
For P90 data, it uses normalized controls and verifies their consistency with
physical controls; historical datasets remain in physical units. Parameter
agreement with the autotuner is not perceptual validation: different ISP settings
may produce similar images. A larger run should check convergence, use more
diverse images and sentences, and evaluate rendered outputs with independent
metrics and human judgments before making any mapping active.