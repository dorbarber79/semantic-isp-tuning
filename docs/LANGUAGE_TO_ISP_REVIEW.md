# Language-to-ISP: Literature Review and Model Plan

**Review date:** 2026-11-13  
**Scope:** Papers and research materials present in this repository, plus recent
closely related work. This is a methods review, not a claim that one method is
universally state of the art.

## Main Finding

There are three different problems often grouped under “text-guided image
editing”:

1. **Prompt-to-render optimization:** optimize parameters for one image and one
   prompt at inference time using a frozen VLM score. This is the closest match
   to Sony’s paper and the current isolated loss experiment.
2. **Learned instruction-conditioned processing:** train a reusable model on
   image, instruction, and target-output examples. PixTalk and InstructIR are
   examples; they learn task-conditioned behavior rather than running a fresh
   optimizer for every request.
3. **Generative image editing:** produce or alter pixels with a diffusion or
   other image generator. This offers broader edits, but its output is not
   constrained to be a faithful ISP transform.

For this project, the target should be **an instruction- and image-conditioned
policy over bounded ISP controls**, not a generic image generator and not
unconstrained CLIP cosine maximization. Frozen CLIP or another encoder can
provide features, but should not be treated as a calibrated action policy or a
human-quality oracle.

## What The Papers Do

| Work | Conditioning and output | Training signal | Relevance and limitation |
| --- | --- | --- | --- |
| [Language-based Color ISP Tuning](2509.10765v1.pdf) (Sony, CIC 2025) | A differentiable 3x3 color matrix is optimized for the current image and prompt. Matrix rows sum to one to preserve the white point; coefficients are bounded. A two-prompt variant tunes toward a requested mixture of two prompt scores. | Frozen CLIP image/text cosine, differentiated through the color matrix; no trained text-to-parameter model. | Closest direct precedent. It supports explicit color prompts, but the result is per-image optimization and the paper reports abstract-emotion prompts as less consistent. It does not learn reusable physical ISP settings. |
| PixTalk (ICCV 2025, [official paper](https://openaccess.thecvf.com/content/ICCV2025/html/Conde_PixTalk_Controlling_Photorealistic_Image_Processing_and_Editing_with_Language_ICCV_2025_paper.html)) | A BGE-micro-v2 text encoder plus learned projection conditions a language controller. Sigmoid mixture-of-experts gates select among more than 40 operations; individual branches include NILUT tone mapping, white-balance choices, color matrices, denoising, and sharpening. Some color operations also use image features. | About 100K professionally edited image pairs, with manually written and LLM-expanded prompts. Task classification is an auxiliary loss; image transformations are trained from paired examples. | The closest learned, multi-operation photo-processing system. Its operation routing and paired supervision are more relevant to a camera than an unconstrained pixel generator. The paper’s output is still a neural processing model, not a six-value physical-control API. |
| [InstructIR](https://arxiv.org/abs/2401.16468) (ECCV 2024) | Frozen BGE-micro-v2 sentence embedding plus learned projection; instruction-condition blocks gate image-feature channels in a NAFNet restoration network. | GPT-4 expands prompts for seven restoration tasks; paired degraded/clean training data supervise the model, with an intent-classification loss shaping text features. | Strong precedent for explicit task labels and learned routing. Its prompts mostly describe degradation/removal actions, not creative global style, and it predicts restored pixels rather than ISP controls. |
| CLIPtone ([2404.01123v1.pdf](2404.01123v1.pdf)) | A CLIP text adapter converts a target-versus-neutral text direction into modulation of an image-adaptive 3D LUT. | Unpaired source images and color-name text; directional CLIP loss, image content loss, and LUT smoothness/regularization. | Shows a reusable text-conditioned tone model can be learned without paired retouches. Its text-to-image direction is still a proxy; it is global tone adjustment and not calibrated to hardware ISP controls. |
| [InstantRetouch](https://arxiv.org/abs/2606.05071) (CVPR 2026) | A low-resolution instruction-conditioned diffusion branch predicts a bilateral grid of local affine transforms, applied to the full-resolution image. | A multi-step diffusion teacher is distilled with variational score distillation, prompt-alignment loss, paired data supervision, and bilateral-grid regularization. | A useful fidelity/efficiency bridge between diffusion and classical processing. It predicts a flexible image-space transform, not interpretable camera ISP controls. |
| PICE ([icip2026_text_guided_3d_luts.pdf](../reference_papers/icip2026_text_guided_3d_luts.pdf)) | A text/style-conditioned compact network predicts color processing; the submission describes portrait-specific local feed-forward blocks. | LUT-transformed target images, pixel and perceptual losses, and 181-way style classification. | Efficient LUT supervision is relevant. The described labels are a finite set of named looks, so this is not evidence of arbitrary language-to-control generalization. The local copy is anonymized and its publication status should be verified before treating its claims as settled. |
| CLIP-IQA ([2207.12396v2.pdf](2207.12396v2.pdf)) | Scores image attributes using paired antonym prompts, e.g. good versus bad. | Evaluated against human-rated image-quality datasets; prompt-pair softmax improves correlation over a single prompt. | Supports paired prompts as a better assessment baseline than a lone quality prompt. It does not make the score a reliable reward for instruction compliance or ISP control. |
| EMOVIS ([2605.03131v1.pdf](2605.03131v1.pdf)) | Four discrete emotions map to six interpretable ISP modifiers in linear, high-precision processing. | Interactive human calibration and a separate A/B preference study; 24 participants, 384 comparisons. | The strongest local example that subjective processing should be calibrated with people. It is a small closed vocabulary, not open-vocabulary prompt understanding. The same paper is duplicated in `reference_papers/`. |
| CRISP ([Learning_Controllable_ISP_for_Image_Enhancement.pdf](Learning_Controllable_ISP_for_Image_Enhancement.pdf); also [2206.08488v1.pdf](2206.08488v1.pdf)) | An encoder-decoder maps image and expert-retouch information to a low-dimensional style code and then to 19 differentiable ISP parameters. | Paired MIT-Adobe FiveK default/retouched images, with image reconstruction and style-control objectives. | Shows compact latent controls can span expert looks efficiently; no natural-language interface or semantic action labels. |
| DynamicISP ([2211.01146v3.pdf](2211.01146v3.pdf)) | Image/intermediate task features control a classical ISP inside a downstream recognition pipeline. | End-to-end task loss and task-recognition performance. | A precedent for image-conditioned ISP parameters, not aesthetic intent or language control. |
| InstructPix2Pix ([2211.09800v2.pdf](2211.09800v2.pdf)) and MGIE ([2309.17102v2.pdf](2309.17102v2.pdf)) | Instruction-conditioned diffusion; MGIE adds an MLLM that rewrites the request into an expressive edit instruction. | Synthetic instruction/image pairs from language and image-generation models; diffusion training. | General and expressive, but can change image content. That is a poor default for a camera whose output should remain a faithful capture. |
| EmoEdit ([2405.12661v3.pdf](2405.12661v3.pdf)) | Emotion adapter conditions a diffusion editor to evoke a target emotion. | A 40K-pair emotion-edit dataset assembled from image collections and generated edits, filtered with VLM/aesthetic scores and human review. | Covers abstract affect beyond simple color words, but remains generative and its automatic filters are not substitutes for user preference. |
| Semantic anchor patent ([patent_idf_semantic_isp.docx](../reference_papers/patent_idf_semantic_isp.docx)) and project spreadsheet ([dof scale per word.xlsx](dof%20scale%20per%20word.xlsx)) | Semantic anchors carry coordinated ISP vectors; a word embedding interpolates among them. The spreadsheet gives hand-authored direction/strength vectors for style words. | Hand-authored anchors and ratings; no learned image-conditioned policy or human study documented in these files. | A compact, bounded, interpretable baseline. It is a useful source of weak supervision, not ground-truth preference. The patent says seven selected anchors but lists eight; reconcile that count before reusing its example table. |

## Recent Direction And Backbone Risk

PixTalk is the strongest supplied precedent for a learned language controller over
multiple real photo operations. Its notable choices are explicit operation
routing, image-conditioned submodules where needed, and paired training data.
Sony’s method is the most direct precedent for optimizing a camera-like ISP from
a VLM score, but that is an inference-time search baseline rather than a learned
policy. InstantRetouch shows a current middle ground: distill a generative teacher
into a constrained, low-dimensional bilateral transform. These methods solve
different tasks and report different metrics; there is no fair single leaderboard
that establishes one universal winner for prompt-to-physical-ISP control.

A 2026 [Language-Guided Invariance Probing study](https://arxiv.org/abs/2511.13494)
tests paraphrase consistency and sensitivity to changed object/color/count words.
It reports stronger robustness for EVA02-CLIP and larger OpenCLIP variants than
for SigLIP/SigLIP2 on its benchmark. This is a warning, not proof that those
models will rank best for our ISP tasks. It makes a direct, controlled backbone
audit necessary; the recent SigLIP brightness and white-balance reversals in this
project are consistent with that risk.

The local corpus contains 12 PDF files and 11 unique works; the EMOVIS PDF is
duplicated. I also reviewed the local research memo, manuscript draft, previous
iteration report, patent, presentation, and degree-of-freedom spreadsheet. The
active pole system described in `SEMANTIC_ISP_RESEARCH.md` remains a separate
CLIP-centroid method; the SigLIP per-image optimizer in
`experiments/semantic_parameter_learning/` is an experiment, not the active
inference system.

## Evidence From Our Recent Comparisons

The five additional weight-3 examples reinforce that convergence is not the
same as following the request:

- On `a0002`, “Cool the white balance a little” has an agent estimate of warmth
  `-0.250`; the converged optimizer chose `+0.254` and also changed five other
  controls.
- On `a0008`, “Make the entire image much brighter” has agent brightness `+0.800`
  and zeros on other controls; the optimizer chose brightness `+0.042`, warmth
  `+0.812`, saturation `-0.631`, and contrast `-0.863` (selected and all starts
  converged).
- On `a0015`, the same brightness request produced brightness `-0.151`. The
  selected restart converged, though not all starts did.

These are narrow examples from one dataset and one optimizer objective. The
agent estimates are subjective and not human-validated. Still, they falsify the
idea that increasing the realism/beauty weights or waiting for the optimizer to
converge is enough to create a reliable instruction-to-ISP mapping.

## Recommended Model

Start with a small deterministic **image + instruction → bounded control**
model:

```text
text instruction ── frozen text encoder ── intent/action features ─┐
                                                                  ├─ small fusion/controller ─ P90 ISP controls
input image ─────── compact image encoder + exposure/color stats ──┘
```

Use the six existing controls as the first output space: warmth, saturation,
tone gain, contrast, sharpness, and palette. Emit normalized P90 values in
`[-1, 1]`, then map them to the calibrated physical endpoints. For direct
requests such as “cool” or “brighter”, include an explicit action/direction head
and enforce that the primary axis cannot flip sign. Let image state adjust the
magnitude for clipping, exposure, or content risk; do not let it reverse the
requested direction. For compound style requests, allow multiple axes and train
that behavior separately from direct controls.

The image encoder should help decide *how much* and *where it is safe* to apply
an edit, not redefine what the user asked for. Start global. Add spatial or
bilateral controls only if global ISP knobs demonstrably cannot express the
desired correction.

## Training Plan

1. **Create groundable examples.** Cross the existing sentence/action table with
   varied FiveK images, rendering exact P90 control targets through the existing
   differentiable ISP. Add paraphrases and explicit negations. Keep paraphrase
   families and images disjoint across train/validation/test. Treat agent ratings
   as weak labels with provenance, not human truth.
2. **Add real preference supervision.** For a balanced subset, show neutral,
   agent-estimated, model-predicted, and alternative-magnitude renders. Collect
   pairwise choices for instruction compliance, realism, and preservation. Fit
   the scales and any quality model to those responses. Do not use the current
   SigLIP “beauty” and “realism” prompt scores as the sole target.
3. **Train in stages.** First fit text-to-action on direct, one-axis language.
   Then add image-conditioned magnitude and style compositions. Compare frozen
   CLIP, SigLIP, BGE sentence embeddings, and at least one robust OpenCLIP/EVA
   candidate on the same splits. Keep the text encoder frozen initially; train a
   small projection/controller, following InstructIR/PixTalk.
4. **Use a multi-term objective.** Regress trusted control targets with Huber
   loss; add sign/direction classification, paraphrase consistency, off-axis
   sparsity for direct requests, and bounded-control regularization. Use an
   independent VLM directional score only as an auxiliary loss. Add human
   pairwise preference loss when labels are available.
5. **Compare fair baselines.** Evaluate direct agent settings, text-only ridge
   and SVR, image+text MLP/controller, Sony-style per-image CLIP optimization,
   and the current weight-3 optimizer. Keep identical images, P90 calibration,
   sentence sets, and output rendering.

Report per-axis sign accuracy, magnitude error, off-axis leakage, paraphrase
variance, image/prompt held-out generalization, highlight/skin preservation,
human instruction-preference rate, latency, and control stability. Use an
independent evaluator backbone from the one used for training. A prompt score or
training loss alone is not a sufficient success metric.

## Figure Convention

Recent comparison figures now display the agent estimate and each autotuned
result in the same P90 coordinates, with all six controls shown. The agent result
is applied directly and is not an autotune target. The original, weight-1, and
refined weight-3 sheet has been regenerated from saved data; no optimizer rerun
was needed. Keep this table on future agent-versus-autotune figures so visual
differences can be tied to the actual settings.