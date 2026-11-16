# Semantic ISP Research Notes: Historical Calibration Log

**Status:** Historical research record; not the source of truth for the active method  
**Updated:** November 14, 2026

This file preserves notes from the Exp63-73 development period. Its tables,
commands, and result claims are historical and must not be treated as the current
calibration. The active method is Exp81: OpenAI CLIP ViT-B/32, six direct controls,
descriptor revision `moran_style_v14_extreme_brightness_4x4`, joint descriptor-tail
endpoint/centroid selection, and palette-only gray/skin protection. The current
source of truth is [`experiments/semantic_isp/CURRENT_METHOD.json`](../experiments/semantic_isp/CURRENT_METHOD.json);
reproduction inputs and commands are in [`REPRODUCING.md`](REPRODUCING.md).

## 1. Research Question and Claim Boundary

The research question is whether a natural-language request can be represented by signed semantic pole coordinates and mapped to a small, bounded set of interpretable ISP controls. The current prototype uses frozen OpenAI CLIP text embeddings to route requests and phrase-level RAW-image optimization to calibrate pole strengths.

The system is not a generative image editor. Calibration uses an image encoder offline, but request-time mapping needs text encoding and a parameter transform; it does not need repeated image-side optimization. The available evidence does **not** establish human preference, professional colorist validity, broad open-vocabulary generalization for the final Exp81 calibration, statistical significance, real-time performance, hardware deployment, camera transfer, or temporal stability.

The current manuscript is [`papers/semantic_isp_tip.tex`](../papers/semantic_isp_tip.tex). The earlier SigLIP per-image optimizer was a separate project and is not the active method.

## 2. Historical Method Notes (Exp63-73)

### Six bipolar axes and physical controls

Each semantic axis maps directly to one physical control. A positive or negative request coordinate changes only its corresponding physical control; all other controls stay at neutral for isolated pole calibration and rendering.

| Axis | Negative pole | Positive pole | Physical control | Neutral | Bounds |
| --- | --- | --- | --- | ---: | ---: |
| Warmth | Cooler white balance | Warmer white balance | Red/blue white-balance warmth | 0 | [-0.3, +0.3] |
| Saturation | Muted / bleached | Vivid | Linear-light saturation | 0 | [-1, +1] |
| Brightness | Dark | Bright | Median-pivot tone gain | 1 | [0.05, 5.0] |
| Contrast | Low contrast | High contrast | Median-pivot contrast | 1 | [0.2, 3.0] |
| Sharpness | Blur | Sharpen | Detail amount | 0 | [-0.9, +0.9] |
| Palette | Magenta-Green | Teal-Orange | Nearest-swatch palette coordinate | 0 | [-0.4, +0.4] |

These are semantic control axes, not a claim that the text embedding axes are orthogonal. Free text can activate several axes, and the optional relevance gate is estimated from weak phrase labels.

### Text centroids and request mapping

The text model is OpenAI CLIP ViT-B/32 with the OpenAI checkpoint and QuickGELU. For axis $k$ and sign $s$, the descriptor table supplies sixteen phrases $d_{k,s,n}$. Each phrase is independently tokenized and encoded. The pole's **text centroid** is the L2-normalized mean:

$$
\mathbf{c}_{k,s} = \frac{\sum_{n=1}^{16} f_T(d_{k,s,n})}{\left\|\sum_{n=1}^{16} f_T(d_{k,s,n})\right\|_2}.
$$

For normalized request embedding $\mathbf{z}_q$, the signed raw score is

$$
r_k(q)=\mathbf{z}_q^\top\mathbf{c}_{k,+}-\mathbf{z}_q^\top\mathbf{c}_{k,-}.
$$

A generic `a photo` prompt supplies the score origin. Positive and negative scales are fitted separately from the pole-description scores, with a 90th-percentile scale and minimum floor. The centered, sign-scaled score is clipped and mapped into the legal semantic coordinate range [-0.4, +0.4]. It is a model-derived coordinate, not a probability or perceptually calibrated distance.

The optional per-axis relevance gate multiplies the projected coordinate. Its features include pole similarity and pole-axis geometry. Gate classifiers are trained on D1-D3 plus unrelated nouns; D4 and auxiliary prompts select the smooth activation interval. D4 also contributes to the text centroids, so this is not an independent descriptor holdout. Other axes may remain active simultaneously.

**Terminology:** a text centroid is the normalized mean of descriptor embeddings. A physical pole endpoint is the calibrated ISP setting. In isolated gallery mode, each axis is set to its signed coordinate limit while all other axes are zero; these rendered tiles are not the free-text projection of a centroid. The renderer's `results.json` stores both projected and isolated coordinates.

### Descriptor design

The active descriptor source is
[`experiments/semantic_isp/isp_control_pole_descriptors.json`](../experiments/semantic_isp/isp_control_pole_descriptors.json),
revision `moran_style_v7_moderate_palette_4x4`. Each of twelve poles has sixteen distinct
phrases arranged in four groups: D1 technical/visual, D2 perceptual, D3 technical/visual,
and D4 cultural/associative, four phrases per group. The non-palette banks are inherited
from the accepted direct-control calibration. Exp67 revises only the two palette banks to
remove excessive intensity terms while preserving each pole's color identity.

The exact current palette phrase table is also exported in the Exp67 result directory. Its phrases are engineering-authored and have not been validated by colorists or users.

### ISP rendering

The differentiable renderer operates on linear RGB and maps raw optimizer values through tanh to the bounded controls. It implements linear red/blue warmth, linear-light saturation, median-pivot tone gain, median-pivot contrast, a resolution-aware detail operator, and the palette operator. Gamut handling follows color-changing stages.

The accepted detail definition is inherited from Exp57: a resolution-aware Gaussian with sigma 8 at 512-pixel reference short side, support truncated at three sigma, and RGB blending for negative sharpness. Positive sharpening uses a separate 3-pixel reference sigma, gain 2.4, and detail limit 0.06. The six controls remain the only trainable physical degrees of freedom.

The active palette operator is the per-pixel nearest-swatch method from Exp63. It converts pixels and the selected pole's two swatches to YIQ, uses a softmax over negative squared opponent-chroma distances (softness ratio 0.18), and blends chroma toward the weighted pair while preserving luminance before gamut handling. The coordinate scales blend magnitude; positive teal-orange uses mix 0.85 and negative magenta-green uses mix 0.75. This is not hue rotation, luminance split-toning, or a global chroma-axis shift.

## 3. Phrase Autotuning

### Data and optimization

The seven calibration images are DNG IDs 1, 28, 236, 194, 277, 263, and 103. Images are preprocessed and center-cropped at 512 pixels on the short side. The three output/evaluation scenes are IDs 2, 33, and 384. Image and model data are external to this repository.

Every descriptor phrase is optimized separately against its own CLIP text embedding. For phrase $d$ and image $R_i$, the base objective is

$$
L_{\mathrm{CLIP}}(u)=-\cos\left(f_I(T_{\theta(u)}(R_i)), f_T(d)\right).
$$

The CLIP encoders remain frozen. Adam starts from neutral physical controls, uses learning rate 0.03 with cosine decay, and runs 100 updates. The best objective iterate across the initialization and updates is retained. For direct axes, Adam receives one scalar only: the matching physical control. All five inactive controls are fixed at neutral throughout every forward pass. No realism, beauty, human-rating, or generic regularization term is used in the direct phrase objective.

### Palette-specific strength prior

The palette phrases showed a consistent tendency to push CLIP-tuned settings outward. Exp67 therefore adds a sign-aware quadratic prior while retaining exact phrase-level CLIP similarity. With $p$ the physical palette coordinate, $s\in\{-1,+1\}$ the requested pole direction, legal half-span $h=0.4$, target magnitude $t=0.17$, and weight $\lambda=0.6$:

$$
L_{\mathrm{palette}}(p,d)=-\cos(I_p,d)+\lambda\left(\frac{t}{h}-s\frac{p}{h}\right)^2.
$$

The prior target is a tuning target, not the selected pole setting. It does not replace the CLIP loss or set the final endpoint directly.

### Endpoint aggregation

For each pole, the 16 phrases x 7 images produce 112 one-control physical settings. The selected endpoint is the 90th percentile of signed displacement from neutral in the intended direction. Positive poles use numerical q90; negative poles use numerical q10. If the selected tail does not support the requested direction, it falls back to neutral. The chosen value is mapped back through the bounded tanh transform and assigned directly to the matching physical coefficient. There is no mean aggregation, regression shrinkage, or post-tuning manual scaling in the active calibration.

Exp67 freshly tunes the 32 palette phrases (224 runs). The other 160 phrase banks and ten non-palette endpoint rows are inherited from the accepted direct calibration. Thus the active calibration has records for 192 phrases across seven images, but the latest run does not claim that all 1,344 optimizations were repeated in Exp67.

## 4. Final Calibration

The selected endpoints in the Exp67 isolated endpoint table are:

| Control | Negative pole | Neutral | Positive pole |
| --- | ---: | ---: | ---: |
| Warmth | -0.267773 | 0 | +0.269057 |
| Saturation | -0.906885 | 0 | +0.884081 |
| Tone gain | 0.438405 | 1 | 1.813333 |
| Contrast | 0.236760 | 1 | 2.685622 |
| Sharpness | -0.819512 | 0 | +0.774859 |
| Palette coordinate | -0.204298 | 0 | +0.202091 |

Each isolated endpoint changes only its listed control; table values for other controls are neutral. The palette medians across the 112 phrase/image settings are -0.188435 and +0.182181, while direction-aligned q90 endpoints are -0.204298 and +0.202091. These final endpoints are within 0.005 of the requested magnitudes.

### Palette calibration progression

| Experiment | Change | Magenta-Green | Teal-Orange | Status |
| --- | --- | ---: | ---: | --- |
| Exp63 | Nearest-swatch operator; strong descriptors; q90; prior toward full legal endpoint | -0.323222 | +0.332589 | Superseded strength/text setting; operator retained |
| Exp64 | Reaggregate saved Exp63 settings by direction-aligned mean | -0.296040 | +0.303076 | Superseded; no new optimization |
| Exp65 | 17-point [-0.4,+0.4] sweep on scenes 2, 33, 384 | Diagnostic | Diagnostic | CLIP phrase cosine peaks at +0.4 and -0.35 on average; not preference evidence |
| Exp66 | Moderate descriptors; target 0.20, prior weight 0.60; q90 | -0.241714 | +0.226108 | Intermediate full run |
| Exp67 | Target 0.17, prior weight 0.60; same descriptors and q90 rule | -0.204298 | +0.202091 | Superseded |

Exp65's score maxima are same-model CLIP self-scores. They explain why a CLIP-only objective did not naturally yield the desired moderate strength, but do not indicate which render people prefer.

### Exp68: Color-Protection Alternatives

[`experiments/semantic_isp/exp68_color_protection_sweep.py`](../experiments/semantic_isp/exp68_color_protection_sweep.py)
renders four opt-in methods for the two warmth poles, the positive saturation pole, and both
palette poles: no protection, gray/white only, skin only, and both. The camera white balance
from the RAW decode remains the baseline; warmth is applied afterward as a protected per-pixel
delta. Saturation protection applies only when saturation increases. Palette protection applies
to both signs. The default ISP configuration remains unchanged unless a `color_protection`
block is explicitly supplied.

Protection uses normalized linear-RGB chromaticity
$(r,g,b)/(r+g+b)$, so reference distance is independent of brightness and contains no luma
term. Gray/white reference chromaticity is $(1/3,1/3,1/3)$. The provisional mean-skin reference
is sRGB $(0.76,0.57,0.47)$, linearized before chromaticity comparison. Gaussian proximity has
sigma 0.07 for gray/white and 0.10 for skin; at each pixel the larger selected proximity scales
the operation delta by $1-\mathrm{proximity}$. Protection strength is 1.0. These fixed choices
are comparison settings, not a validated skin detector; the average skin reference may not fit
all complexions or lighting.

The full comparison on scenes 2, 33 and 384 contains 60 operation/mode metric rows, 60 protected
renders plus three neutrals, and three contact sheets under
`/data/data1/semantic_isp/results/exp68_color_protection_sweep/`. Across those scenes, mean total
RGB effect remaining relative to no protection is 0.44 for negative warmth, 0.46 for positive
warmth, 0.45 for positive saturation, 0.58 for negative palette and 0.55 for positive palette
when both protections are enabled. These are whole-image pixel-change measurements, not
perceptual-quality scores. The side-by-side native renders are intended for selecting which
protection option, if any, should be evaluated further; none is adopted as the default ISP yet.

### Exp69: Protection-Aware Color Pole Autotuning

Because protection reduces color-operation effect, Exp69 reruns phrase autotuning independently
for each Exp68 protection mode. Each mode performs three sequential, fresh retunes: both warmth
poles, positive saturation only, and both palette poles. Every selected pole still uses its
direction-aligned q90 from 112 phrase/image settings; other controls and unselected poles are
inherited. Negative saturation is not retuned or protected because the requested protection
applies only to positive saturation. The palette strength prior and operator settings remain
fixed; warmth and saturation have no palette strength prior. Exp69 was a protection-choice
study; the subsequent Exp70 comparison adopts combined gray/white and skin protection while
comparing CLIP with SigLIP.

The selected settings are:

| Protection | Warmth - | Warmth + | Saturation + | Palette - | Palette + |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | -0.26808 | +0.27007 | +0.88408 | -0.20496 | +0.20235 |
| Gray/white | -0.27355 | +0.27650 | +0.88081 | -0.20464 | +0.19781 |
| Skin | -0.26915 | +0.27414 | +0.88993 | -0.20637 | +0.20212 |
| Gray/white + skin | -0.27333 | +0.27674 | +0.88409 | -0.20565 | +0.19749 |

The complete run performs 2,240 fresh optimizations (560 per mode) and records 226,240
initial-plus-100-update trace rows. It produces four independently calibrated configs and
four 39-image galleries under
`/data/data1/semantic_isp/results/exp69_color_protection_retune/`; `index.html` links each
mode and `color_endpoint_statistics.csv` lists the selected endpoints. The comparison data
do not establish that protection improves quality; they let a human choose which effect profile
to evaluate further. The skin reference remains a provisional single chromaticity and is not a
validated segmentation model.

### Exp70: Full-Axis CLIP versus SigLIP Autotuning

Exp70 adopts combined gray/white and skin protection and repeats the complete six-axis direct
phrase-autotune process for both backbones. Each model starts from the same Exp69 combined-
protection config and active v7 descriptor table, but gets freshly constructed text centroids,
generic-photo score origin, signed distance scales, and relevance gate. It then independently
optimizes all 192 exact phrases on the same seven RAW training images, one matching physical
control at a time, for 100 Adam updates at learning rate 0.03. Each pole uses the direction-aligned
q90 endpoint (numeric q90 for positive, q10 for negative). No phrase optimum is shared between
models.

CLIP is OpenAI ViT-B/32 with its OpenAI checkpoint, QuickGELU, and image normalization
mean/std `[0.48145466, 0.4578275, 0.40821073]` / `[0.26862954, 0.26130258, 0.27577711]`.
SigLIP is ViT-B/16-SigLIP with WebLI weights, its native activation, and mean/std `[0.5,0.5,0.5]`
for both. Both use the same RAW preprocessing, linear-light ISP, descriptors, image split,
combined color protection, palette prior/operator, optimizer, and endpoint statistic. Similarity
values from the two models are not directly comparable because their embedding scales/objectives
differ; compare physical endpoints and renders instead.

The run performed 1,344 fresh optimizations per model (2,688 total). It compares all twelve
positive/negative physical endpoints and measures CLIP reproduction against the existing Exp69
combined-protection endpoints. The CLIP run reproduces those endpoints with maximum absolute
difference 0.004414 (positive contrast); palette differences are at most 3e-7.

| Axis/control | CLIP negative | SigLIP negative | CLIP positive | SigLIP positive | SigLIP minus CLIP, positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warmth | -0.27333 | -0.27409 | +0.27674 | +0.27264 | -0.00410 |
| Saturation | -0.90688 | -0.89313 | +0.88409 | +0.85469 | -0.02940 |
| Brightness / tone gain | 0.43884 | 0.44025 | 1.80995 | 1.75857 | -0.05138 |
| Contrast | 0.23715 | 0.24645 | 2.69004 | 2.51945 | -0.17059 |
| Sharpness | -0.81867 | -0.83047 | +0.77485 | +0.74155 | -0.03330 |
| Palette | -0.20565 | -0.20962 | +0.19749 | +0.21068 | +0.01319 |

The largest physical endpoint difference is positive contrast, where SigLIP is lower by 0.17059.
Warmth endpoints differ by at most 0.00410, and palette endpoints by at most 0.01319. The raw
CLIP/SigLIP cosine scales are not compared. Results and matched neutral/CLIP/SigLIP pole sheets
are under `/data/data1/semantic_isp/results/exp70_clip_vs_siglip_all_axes/`. The directory
includes each model's full traces, endpoint statistics, model-specific calibrations/gates, 39
native pole images per model, three SBS sheets, and an HTML index. The one-image smoke output is
not used as final evidence.

### Exp71: Palette-Only Protection Retune

After the broad Exp69/Exp70 protection experiments, the adopted policy restricts gray/white and
skin chromaticity masks to the palette operator. Warmth and positive saturation are freshly
retuned without protection; both palette poles are freshly retuned with combined gray/white and
skin protection. Negative saturation and brightness, contrast, and sharpness endpoints inherit
Exp70 values. The model-specific centroids, score scales, and gates are unchanged from Exp70.

| Model | Warmth - | Warmth + | Saturation + | Palette - | Palette + |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIP | -0.268077 | +0.270075 | +0.884081 | -0.205648 | +0.197487 |
| SigLIP | -0.263304 | +0.269273 | +0.887079 | -0.209246 | +0.210676 |

Exp71 includes 1,120 fresh optimizations (560 per model), direction-aligned q90 endpoint
statistics, final configs, and per-model galleries under
`/data/data1/semantic_isp/results/exp71_palette_only_protection/`. The final config explicitly
sets `color_protection.operations=["palette"]`.

### Exp72: Easy/Hard Request Inference Comparison

Exp72 uses the final palette-only Exp71 configs and prompt template `a {word} photo`. The easy
group is twelve clear control words from the anchors/seen corpus: `warm`, `cool`, `bright`, `dark`,
`saturated`, `muted`, `sharp`, `monochrome`, `retro`, `golden`, `faded`, and `crisp`. The hard
group is twelve subjective held-out corpus words: `angelic`, `ancient`, `ethereal`, `noir`,
`tropical`, `gothic`, `melancholic`, `futuristic`, `pastoral`, `mystical`, `whimsical`, and
`bohemian`. These labels define the comparison strata, not semantic ground truth.

On scenes 2, 33, and 384, the runner records each model's raw pole scores, relevance gates, six
coordinates, physical controls, and native render. This produces 144 mapping rows (24 prompts x
2 models x 3 images), three neutral/CLIP/SigLIP SBS sheets, and an HTML index. Across words, the
mean maximum-axis coordinate gap is 0.133 for easy and 0.089 for hard prompts. Mean rendered
sRGB RGB MAE between models is 6.89/255 for easy and 6.58/255 for hard prompts. Largest coordinate
gaps include `faded` on sharpness (+0.400 SigLIP minus CLIP), `saturated` on saturation (-0.240),
`muted` on saturation (+0.222), and `angelic` on brightness (-0.207). The largest mean render
difference is `melancholic` at 23.30/255, then `muted` at 17.64/255 and `faded` at 14.96/255.
These are model mapping and pixel-difference diagnostics, not image-quality or preference scores.
Results are under `/data/data1/semantic_isp/results/exp72_backbone_inference_easy_hard/`.

Exp72 is a historical inference comparison using the Exp71 v7 calibrations. Its SigLIP mapping
and renders are retained for audit only; CLIP is now the selected model for active development.

### Exp73: Stronger CLIP Warmth, Brightness, and Saturation Descriptors

The active descriptor table is now revision
`moran_style_v8_extreme_warmth_brightness_vivid_4x4`. It changes only five banks: both warmth
poles, both brightness poles, and positive saturation. The palette and all other descriptor banks
are byte-for-byte unchanged from the prior table. The v8 phrases intensify the requested color or
tone explicitly; they do not alter physical bounds, renderer operators, protection masks, optimizer,
or the direction-aligned q90 endpoint rule.

Using OpenAI CLIP ViT-B/32, the retune freshly optimized those five poles on all seven calibration
images, 100 updates per phrase, one control at a time (560 optimizations; 56,560 trace rows). The
selected q90 endpoints are:

| Pole | Exp71 endpoint | Exp73 v8 endpoint |
| --- | ---: | ---: |
| Warmth, negative | -0.268077 | -0.269376 |
| Warmth, positive | +0.270075 | +0.270106 |
| Brightness, negative tone gain | 0.438405 | 0.436543 |
| Brightness, positive tone gain | 1.813333 | 1.827876 |
| Saturation, positive | +0.884081 | +0.890006 |

Thus the new language made only small numeric changes: warmth endpoints are effectively unchanged;
positive saturation rises by 0.005925; bright tone gain rises by 0.014544; dark tone gain moves
0.001862 closer to its lower bound. The selected settings were already near their physical bounds,
which limits how much descriptor-only changes can increase the output. No bounds or operator
strengths were manually changed. The negative saturation pole and palette endpoints are preserved
from Exp71, and palette remains the only protected operation. The final CLIP config and gallery are
under `/data/data1/semantic_isp/results/exp73_clip_extreme_descriptors/`; the index and endpoint
CSV report the five retuned poles.

## 5. Evidence and Evaluation Status

The final Exp67 run includes 224 new phrase/image optimizations, each with a 101-row initial-plus-100-update trace (22,624 trace rows), 112 samples per palette pole, an updated palette text-score calibration and relevance-gate row, q90 statistics, and 39 native 8-bit RGB pole/reference images across three scenes. All ten non-palette physical endpoint rows are inherited unchanged. The gallery and run artifacts are under:

`/data/data1/semantic_isp/results/exp67_moderate_palette_q90_pm020/`

The Exp65 palette sweep contains 51 renders (17 coordinates x 3 scenes), contact sheets, a metric chart, CSV, summary, and HTML index. The endpoint and coordinate records are under:

`/data/data1/semantic_isp/results/exp65_palette_sweep/`

Exp72 evaluates a curated 24-word easy/hard subset on three scenes for model-to-model mapping comparison; it is not a corpus-wide generalization study. No full prompt-corpus evaluation was run with the final Exp71 palette-only configs. Earlier 90-pair and 1,360-pair scores in older manuscripts correspond to earlier descriptor/calibration versions and must not be presented as Exp71/Exp72 results. No human preference evaluation was run.

### Limitations

- Each model supplies its own tuning objective and text-pole geometry. CLIP and SigLIP results may reflect model-specific language and image biases; their raw cosine scales are not cross-comparable.
- The 7-image calibration set is small and selected; the three output scenes are also a small selected set.
- Descriptors and gate labels are engineering-authored; D4 contributes to centroids and to gate threshold validation.
- The direct controls do not establish perceptual independence. Free text may activate several axes.
- The Exp72 prompt subset is small and deliberately stratified, not statistically representative. No blinded user/colorist study, independent VLM evaluation, uncertainty analysis, temporal test, device transfer test, or runtime/power measurement has been done.
- RAW data and model weights are external. Redistribution rights and public reproducibility have not been resolved.

## 6. Related Work and Positioning

The paper should position this work as a calibration study, not as the first use of language or CLIP to tune image processing. Relevant precedents include CLIPtone (learned text-conditioned tone adjustment), Mayer et al.'s language-based differentiable color ISP tuning (the closest direct optimization precedent), T2ONet (language-to-editing operations), CRISP (controllable ISP codes without a text interface), DynamicISP (task-feedback ISP adaptation), and CLIP-IQA (positive/negative-prompt image assessment). CLIP supplies the shared image/text embedding space; EMOVIS is engineering context for expressive ISP operations. ConSE is a structural analogy for embedding combinations, not a text-to-ISP method.

The current distinction is a signed multi-pole text representation, one-to-one semantic-axis/physical-control constraints during phrase autotuning, data-derived directional q90 endpoints, and an explicitly controlled nearest-swatch palette operator. Novelty claims require a refreshed literature review before submission. The retired broad survey used an unimplemented anchor-interpolation premise and is not an authoritative survey for this work.

## 7. Reproduction and Files

- Machine-readable current method and mode selection: [`experiments/semantic_isp/CURRENT_METHOD.json`](../experiments/semantic_isp/CURRENT_METHOD.json)
- Active pipeline index: [`experiments/semantic_isp/README.md`](../experiments/semantic_isp/README.md)
- Active descriptor table: [`experiments/semantic_isp/isp_control_pole_descriptors.json`](../experiments/semantic_isp/isp_control_pole_descriptors.json)
- Text projection and inference: [`experiments/semantic_isp/moran_semantic_isp.py`](../experiments/semantic_isp/moran_semantic_isp.py)
- Differentiable ISP operators: [`experiments/semantic_isp/axis_isp.py`](../experiments/semantic_isp/axis_isp.py)
- Phrase-level optimizer: [`experiments/semantic_isp/exp38_learn_poles.py`](../experiments/semantic_isp/exp38_learn_poles.py)
- Direct pole calibration and q90 aggregation: [`experiments/semantic_isp/exp49_direct_isp_comparison.py`](../experiments/semantic_isp/exp49_direct_isp_comparison.py)
- Targeted descriptor retuner: [`experiments/semantic_isp/exp55_large_kernel_detail_retune.py`](../experiments/semantic_isp/exp55_large_kernel_detail_retune.py)
- Isolated/projected pole renderer: [`experiments/semantic_isp/render_pole_centroids.py`](../experiments/semantic_isp/render_pole_centroids.py)
- Palette sweep: [`experiments/semantic_isp/exp65_palette_sweep.py`](../experiments/semantic_isp/exp65_palette_sweep.py)
- Color-protection comparison: [`experiments/semantic_isp/exp68_color_protection_sweep.py`](../experiments/semantic_isp/exp68_color_protection_sweep.py)
- Four-mode protection autotuner: [`experiments/semantic_isp/autotune.py`](../experiments/semantic_isp/autotune.py)
- Autotune coordinator implementation: [`experiments/semantic_isp/exp69_protection_retune.py`](../experiments/semantic_isp/exp69_protection_retune.py)
- Easy/hard CLIP/SigLIP inference comparison: [`experiments/semantic_isp/compare_backbone_inference.py`](../experiments/semantic_isp/compare_backbone_inference.py)
- Focused tests: `experiments/semantic_isp/test_exp49_direct_isp_comparison.py`, `experiments/semantic_isp/test_moran_semantic_isp.py`, and `experiments/semantic_isp/test_axis_isp.py`
- Current manuscript: [`papers/semantic_isp_tip.tex`](../papers/semantic_isp_tip.tex)

Historical Exp69 four-mode autotune command:

```bash
python experiments/semantic_isp/autotune.py \
  --base /data/data1/semantic_isp/results/exp67_moderate_palette_q90_pm020 \
  --out /path/to/new/exp70_protection_retune \
  --iters 100 --device cuda:0
```

The output manifest records source hashes, image splits, model/operator settings, and optimization counts. Existing result directories are provenance artifacts and should not be overwritten with a changed source or configuration.

## 8. Paper Requirements

For current claims, consult the Exp81 manifest and the manuscript. This historical
checklist describes Exp67 and must not be used to update the current method without
checking the frozen Exp81 configuration and results.

The manuscript is currently in IEEE TIP format, but the venue is not selected. The older journal ranking was based on the superseded anchor-interpolation framing and should not be used as a submission recommendation. Before submission: update and verify references; settle author details; document image licensing; add independent request-level evaluation and a blinded preference study; report uncertainty and convergence; and evaluate runtime on target hardware. Until then, the manuscript remains an internal technical draft.
