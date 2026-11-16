# FiveK Fit-Description Annotation Protocol

## Provenance

- Annotator surface: GitHub Copilot Chat acting as a multimodal assistant.
- Underlying model identifier/version: unavailable in the captured session metadata; do not infer or invent one for citation.
- User-reported model label: GPT-5.6 Luna, visible in a user-provided screenshot labeling a report-verification execution. This does not independently establish the model used during the original annotation turns.
- These descriptions are assistant-authored visual notes, not human annotations, consensus labels, or validated training ground truth.
- Current reviewed and pending counts are stored in `annotation_process.snapshot_counts` in the JSON file and updated after each batch.
- No batch vision-language model generated the committed descriptions. A separate Qwen2.5-VL-3B-Instruct probe was tested on Default-versus-expert pairs, reversed a known brightness difference, missed warmth, and was rejected.

## Visual Inputs

The fit-comparison sheets displayed three panels: Expert TIFF, Default ISP, and the fitted ISP render. The description target was the fitted output relative to Default ISP. The Expert TIFF was visible as context, so this was not a blinded study; the text is intended to characterize the fit rather than describe the expert edit.

Each scene contains five fitted outputs, A-E. The fits were produced by the 500-iteration run recorded in `isp_match_105_v4`.

## Prompt and Guidance

There was no fixed per-image model prompt or command. Descriptions were written interactively in response to user instructions. The following is a reconstructed annotation specification, not a verbatim prompt:

> Compare the fitted ISP render with Default ISP. Describe the fitted output in short, natural, image-specific language. D1 notes the visible processing effect; D2 gives the perceptual mood; D3 mentions concrete scene elements as they appear in the fit; D4 gives a cautious style association. Do not write commands, infer hidden camera settings or photographer intent, name expert letters, or compare the fit with another expert. Avoid repeating wording across different fits in the same scene.

## Descriptor Groups

- D1: visible fit effects such as light, warmth, color, contrast, or detail.
- D2: the perceived mood of the fitted output.
- D3: concrete scene elements in the fitted output.
- D4: a cautious style or cinematic association.

## Validation and Reproduction

`fivek_fit_visual_descriptions.json` stores the exact text. The loader checks five fits per scene, nonempty D1-D4 entries, and no repeated phrase within a scene and group. Tests enforce concise D1/D3 text and the pending state for unreviewed fits. The report renderer uses this text verbatim for reviewed fits and does not generate replacement prose for the remaining fits.

Refresh the existing report in place. Per-image hashes redraw only scenes whose descriptions changed; a different fit source is rejected:

```bash
cd experiments/semantic_parameter_learning
python fivek_screen_comparisons.py \
  --source /data/data1/FiveK/C_data/results/isp_match_105_v4 \
  --out /data/data1/FiveK/C_data/results/isp_match_105_screen_v22
```

For publication, describe this dataset as assistant-authored subjective fit notes. Human validation and inter-rater agreement were not recorded. The GPT-5.6 Luna label was supplied from a screenshot of a verification execution and is not independently tied to the original annotation turns.