"""Retune selected detail poles while preserving every unselected pole."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from differentiable_isp import load_and_preprocess_raw
from exp38_learn_poles import optimize_controls, save_csv, save_json
from exp42_phrase_pole_calibration import TRAIN_IMAGES, TEST_IMAGES, descriptor_rows, save_descriptor_stats
from exp49_direct_isp_comparison import direct_quantile_calibration, recalibrate_semantic_mapping
from moran_semantic_isp import CONTROL_BOUNDS, bounded_controls, neutral_raw_controls
from render_pole_centroids import run as render_poles


@torch.no_grad()
def encoded_phrases(model, tokenizer, texts, device):
    return F.normalize(model.encode_text(tokenizer(texts).to(device)).float(), dim=-1)


def run(args):
    import open_clip

    torch.manual_seed(0)
    torch.set_num_threads(2)
    base_path = args.base / "direct_soft_calibration.json"
    original = json.loads(base_path.read_text())
    config = copy.deepcopy(original)
    sides = ("negative", "positive") if args.side == "both" else (args.side,)
    descriptor_table_path = Path(__file__).with_name("isp_control_pole_descriptors.json")
    descriptor_table = json.loads(descriptor_table_path.read_text())
    merged_descriptor_table = copy.deepcopy(original["descriptor_table"])
    for key in ("schema_version", "rationale", "descriptor_revision", "strength_revision",
                "sentences_per_descriptor_type", "guideline_source"):
        if key in descriptor_table:
            merged_descriptor_table[key] = descriptor_table[key]
    revisions_by_axis = dict(merged_descriptor_table.get("revisions_by_axis", {}))
    revisions_by_axis[args.axis] = descriptor_table["descriptor_revision"]
    merged_descriptor_table["revisions_by_axis"] = revisions_by_axis
    merged_descriptor_table["axes"][args.axis] = descriptor_table["axes"][args.axis]
    config["descriptor_table"] = merged_descriptor_table
    for axis in config["axes"]:
        for side in ("negative", "positive"):
            config["poles"][axis][side]["texts"] = [
                entry["text"] for entry in merged_descriptor_table["axes"][axis][side]]
    if args.side == "both":
        if "detail_filter" not in original or args.sigma != original["detail_filter"]["sigma_at_reference"]:
            config["detail_filter"] = {"sigma_at_reference": args.sigma,
                "reference_short_side": 512., "truncate": 3., "implementation": "separable Gaussian luminance filter",
                "scope": "Gaussian sigma/support change; luma blend retained"}
    elif "detail_filter" not in original or args.sigma != original["detail_filter"]["sigma_at_reference"]:
        raise ValueError("Single-side retuning requires the base detail filter and unchanged sigma")
    if args.blur_color_mode != "preserve":
        config["detail_filter"]["blur_color_mode"] = args.blur_color_mode
        config["detail_filter"]["scope"] = "Negative branch color processing changed; Gaussian geometry and positive sharpening preserved"
    if args.sharpen_gain is not None:
        config["positive_sharpening"] = {"gain": args.sharpen_gain, "detail_limit": args.sharpen_limit,
            "scope": "Positive branch only; negative blur and spatial filter unchanged"}
    if args.sharpen_sigma is not None:
        if args.axis != "sharpness" or "positive" not in sides:
            raise ValueError("Separate sharpening sigma applies only to the positive sharpness pole")
        config["detail_filter"]["sharpen_sigma_at_reference"] = args.sharpen_sigma
        config["detail_filter"]["sharpen_sigma_scope"] = "Positive detail signal only; negative RGB blur keeps the full configured blur sigma"
    if args.contrast_range is not None:
        if args.axis != "contrast" or set(sides) != {"negative", "positive"}:
            raise ValueError("Contrast range changes require retuning both contrast poles")
        if len(args.contrast_range) != 2 or not (0 < args.contrast_range[0] < 1 < args.contrast_range[1]):
            raise ValueError("Contrast range must straddle the neutral setting 1")
        bounds = [list(pair) for pair in config.get("control_bounds",
            config["learned_calibration"].get("control_bounds", CONTROL_BOUNDS))]
        bounds[3] = list(args.contrast_range)
        config["control_bounds"] = bounds
        config["learned_calibration"]["control_bounds"] = bounds
        config["learned_calibration"]["raw_neutral"] = neutral_raw_controls(bounds=bounds).tolist()
    tone_gain_range = getattr(args, "tone_gain_range", None)
    if tone_gain_range is not None:
        if args.axis != "brightness" or set(sides) != {"negative", "positive"}:
            raise ValueError("Tone-gain range changes require retuning both brightness poles")
        if len(tone_gain_range) != 2 or not (0 < tone_gain_range[0] < 1 < tone_gain_range[1]):
            raise ValueError("Tone-gain range must be positive and straddle the neutral setting 1")
        bounds = [list(pair) for pair in config.get("control_bounds",
            config["learned_calibration"].get("control_bounds", CONTROL_BOUNDS))]
        bounds[2] = list(tone_gain_range)
        config["control_bounds"] = bounds
        config["learned_calibration"]["control_bounds"] = bounds
        config["learned_calibration"]["raw_neutral"] = neutral_raw_controls(bounds=bounds).tolist()
    palette_strength_weight = args.palette_strength_weight if args.axis == "palette" else 0.
    if args.axis == "palette":
        config["palette_split_tone"] = {"coordinate_limit": float(config["coordinate_limit"]),
            "positive_mix": args.palette_positive_mix, "negative_mix": args.palette_negative_mix,
            "method": "nearest_swatch"}
        config["palette_nearest_swatch"] = {"softness_ratio": args.palette_softness_ratio,
            "classification": "softmax negative squared YIQ chroma distance to two swatches for selected pole"}
        config["palette_tuning_objective"] = {
            "clip_target": "exact pole sentence embedding",
            "directional_strength_weight": palette_strength_weight,
            "directional_strength_target_coordinate": float(args.palette_strength_target),
            "directional_strength_prior": "quadratic reward toward the requested palette coordinate magnitude",
            "operator_strengths": config["palette_split_tone"],
        }
        if not 0 <= args.palette_strength_target <= config["coordinate_limit"]:
            raise ValueError("Palette strength target must be within the configured coordinate range")
    protection_mode = getattr(args, "color_protection_mode", None)
    if protection_mode is not None:
        if protection_mode not in {"none", "gray_white", "skin", "gray_white_and_skin"}:
            raise ValueError(f"Unknown color-protection mode: {protection_mode}")
        protection = {
            "mode": protection_mode,
            "strength": 1.,
            "gray_sigma": .07,
            "skin_sigma": .10,
            "skin_reference_srgb": [.76, .57, .47],
            "saturation_positive_only": True,
        }
        operations = getattr(args, "color_protection_operations", None)
        if operations is not None:
            if not set(operations).issubset({"warmth", "saturation", "palette"}):
                raise ValueError("Color-protection operations may only include warmth, saturation, palette")
            protection["operations"] = list(operations)
        config["color_protection"] = protection
    rows = descriptor_rows(config)
    if args.axis not in config["axes"]:
        raise ValueError(f"Unknown direct ISP axis: {args.axis}")
    changed_descriptor_sides = [side for side in sides
        if config["descriptor_table"]["axes"][args.axis][side] !=
        original["descriptor_table"]["axes"][args.axis][side]]
    control = config["autotune_policy"]["axis_controls"][args.axis]
    selected = [row for row in rows if row["axis"] == args.axis and row["sign"] in sides]
    if len(selected) != 16 * len(sides):
        raise ValueError("Expected sixteen descriptions per selected sharpness pole")
    training_images = TRAIN_IMAGES[:1] if args.smoke else TRAIN_IMAGES
    output_images = TEST_IMAGES[:1] if args.smoke else TEST_IMAGES
    sources = (Path(__file__), Path(__file__).with_name("moran_semantic_isp.py"),
               Path(__file__).with_name("exp38_learn_poles.py"), Path(__file__).with_name("exp49_direct_isp_comparison.py"),
               Path(__file__).with_name("axis_isp.py"), descriptor_table_path)
    manifest = {"base": str(args.base), "base_config_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "iterations": args.iters, "learning_rate": .03, "training_images": training_images,
        "output_images": output_images, "smoke": args.smoke, "detail_filter": config["detail_filter"],
        "retuned_sides": list(sides), "positive_sharpening": config.get("positive_sharpening"),
        "control_bounds": config.get("control_bounds", config["learned_calibration"].get("control_bounds", CONTROL_BOUNDS)),
        "contrast_range": args.contrast_range, "sharpen_sigma": args.sharpen_sigma,
        "tone_gain_range": tone_gain_range,
        "palette_tuning_objective": config.get("palette_tuning_objective"),
        "palette_nearest_swatch": config.get("palette_nearest_swatch"),
        "color_protection": config.get("color_protection"),
        "skip_gallery": getattr(args, "skip_gallery", False),
        "retuned_axes": [args.axis], "trainable_controls": [control],
        "new_optimizations": len(training_images) * len(selected),
        "changed_descriptor_sides": changed_descriptor_sides,
        "preservation": f"Unselected pole rows and non-target axis text/gate calibration unchanged; selected axis recalibrated if its descriptors changed",
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "experiment.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing manifest differs; use a new output directory")
    elif any(args.out.iterdir()):
        raise FileExistsError("Output is not empty and has no matching manifest")
    else:
        save_json(manifest_path, manifest)
    save_json(args.out / "descriptor_table.json", config["descriptor_table"])
    save_csv(args.out / "descriptor_table.csv", rows)
    model, _, _ = open_clip.create_model_and_transforms(config["model"], pretrained=config["pretrained"],
        force_quick_gelu=config.get("force_quick_gelu", config["pretrained"] == "openai"),
        device=args.device)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])

    @torch.no_grad()
    def encode_text(texts):
        return model.encode_text(tokenizer(texts).to(args.device)).float()

    targets = encoded_phrases(model, tokenizer, [row["text"] for row in selected], args.device)
    fresh_records = []
    for image_id in training_images:
        image = load_and_preprocess_raw(f"/data/data1/EmotionISP_dataset/dng_filtered/{image_id}.dng", 512).to(args.device)
        folder = args.out / f"train_img_{image_id}"
        folder.mkdir(exist_ok=True)
        for index, row in enumerate(selected):
            path = folder / f"{row['axis']}_{row['sign']}_{row['id']}.json"
            if path.exists():
                record = json.loads(path.read_text())
            else:
                record, trace = optimize_controls(image, targets[index:index + 1], model,
                    config, args.iters, .03, active_control=control,
                    strength_target_sign=(-1 if row["sign"] == "negative" else 1)
                    if args.axis == "palette" else None,
                    strength_weight=palette_strength_weight)
                record.update(image=image_id, **row)
                save_csv(path.with_name(path.stem + "_trace.csv"), trace)
                save_json(path, record)
                if not getattr(args, "quiet", False):
                    print(image_id, row["sign"], row["id"],
                          f"gain={record['best_similarity'] - record['initial_similarity']:+.5f}", flush=True)
            fresh_records.append(record)
        del image
    inheritance_roots = [args.base]
    ancestor = original
    while ancestor.get("detail_retuning", {}).get("base"):
        ancestor_root = Path(ancestor["detail_retuning"]["base"])
        if ancestor_root in inheritance_roots:
            raise ValueError("Circular inherited-record provenance")
        inheritance_roots.append(ancestor_root)
        ancestor = json.loads((ancestor_root / "direct_soft_calibration.json").read_text())
    inherited = []
    for image_id in training_images:
        for row in rows:
            if row["axis"] == args.axis and row["sign"] in sides:
                continue
            relative = f"train_img_{image_id}/{row['axis']}_{row['sign']}_{row['id']}.json"
            path = next((root / relative for root in inheritance_roots if (root / relative).exists()), None)
            if path is None:
                raise FileNotFoundError(f"Missing inherited record: {relative}")
            record = json.loads(path.read_text())
            if record["text"] != row["text"]:
                raise ValueError(f"Inherited descriptor does not match: {path}")
            inherited.append(record)
    all_records = inherited + fresh_records
    save_descriptor_stats(all_records, rows, args.out)
    calibrated, statistics = direct_quantile_calibration(config, all_records)
    calibrated, mapping_split = recalibrate_semantic_mapping(calibrated, rows, encode_text)
    if changed_descriptor_sides:
        calibrated["descriptor_recalibration"] = {"axis": args.axis,
            "changed_sides": changed_descriptor_sides,
            "text_score_calibration": "Recomputed all axes using selected normalized descriptor-tail centroids",
            "relevance_gate": "Recomputed all axes using selected normalized descriptor-tail centroids",
            "gate_diagnostics": mapping_split["diagnostics"]}
    before = np.asarray(original["learned_calibration"]["pole_raw_deltas"])
    after = np.asarray(calibrated["learned_calibration"]["pole_raw_deltas"])
    axis_index = config["axes"].index(args.axis)
    updated_rows = sorted(axis_index + (len(config["axes"]) if sign == "negative" else 0)
                          for sign in sides)
    frozen_rows = [index for index in range(len(before)) if index not in updated_rows]
    old_bounds = original.get("control_bounds",
        original["learned_calibration"].get("control_bounds", CONTROL_BOUNDS))
    new_bounds = config.get("control_bounds",
        config["learned_calibration"].get("control_bounds", CONTROL_BOUNDS))
    old_neutral = torch.tensor(original["learned_calibration"]["raw_neutral"])
    new_neutral = torch.tensor(calibrated["learned_calibration"]["raw_neutral"])
    recalibrated_coefficients = np.asarray(calibrated["learned_calibration"]["pole_raw_deltas"], dtype=float)
    for row_index in frozen_rows:
        old_setting = bounded_controls(old_neutral + torch.tensor(before[row_index]), old_bounds)
        new_bounds_tensor = old_setting.new_tensor(new_bounds)
        midpoint = new_bounds_tensor.mean(dim=1)
        half_range = (new_bounds_tensor[:, 1] - new_bounds_tensor[:, 0]) / 2
        normalized = ((old_setting - midpoint) / half_range).clamp(-1 + 1e-7, 1 - 1e-7)
        physical_raw = torch.atanh(normalized)
        recalibrated_coefficients[row_index] = (physical_raw - new_neutral).numpy()
        new_setting = bounded_controls(new_neutral + torch.tensor(recalibrated_coefficients[row_index]), new_bounds)
        torch.testing.assert_close(new_setting, old_setting, atol=1e-6, rtol=0)
    calibrated["learned_calibration"]["pole_raw_deltas"] = recalibrated_coefficients.tolist()
    expected_neutral = neutral_raw_controls(bounds=config.get("control_bounds",
        config["learned_calibration"].get("control_bounds", CONTROL_BOUNDS))).tolist()
    np.testing.assert_allclose(calibrated["learned_calibration"]["raw_neutral"], expected_neutral, atol=1e-12)
    for axis in config["axes"]:
        for sign in ("negative", "positive"):
            if axis != args.axis or sign not in sides:
                assert calibrated["descriptor_table"]["axes"][axis][sign] == original["descriptor_table"]["axes"][axis][sign]
    calibrated["status"] = f"Only {args.axis} {', '.join(sides)} retuned; all {12 - len(sides)} other pole settings inherited unchanged"
    calibrated["detail_retuning"] = {"base": str(args.base), "fresh_optimization_runs": len(fresh_records),
        "new_training_images": training_images, "inherited_other_pole_records": len(inherited),
        "updated_coefficient_rows": updated_rows, "preserved_coefficient_rows": frozen_rows,
        "preservation_unit": "Physical ISP settings; raw tanh coefficients may change when configured bounds change",
        "descriptor_revision": descriptor_table["descriptor_revision"],
        "retuned_descriptor_texts": [row["text"] for row in selected],
        "inherited_record_roots": [str(root) for root in inheritance_roots],
        "rendering_note": "All pole outputs use the new filter; non-detail isolated poles have zero detail amount"}
    save_json(args.out / "calibrated_poles.json", calibrated)
    save_json(args.out / "direct_soft_calibration.json", calibrated)
    save_csv(args.out / "pole_quantile_statistics.csv", statistics)
    save_json(args.out / "summary.json", {"fresh_optimization_runs": len(fresh_records),
        "inherited_other_pole_records": len(inherited), "preserved_other_poles": 12 - len(sides),
        "retuned_sides": list(sides), "positive_sharpening": config.get("positive_sharpening"),
        "iterations": args.iters, "smoke": args.smoke,
        "reference_kernel_size": 2 * math.ceil(3 * args.sigma) + 1,
        "reference_sigma": args.sigma, "reference_short_side": 512,
        "contrast_range": args.contrast_range, "sharpen_sigma": args.sharpen_sigma,
        "palette_tuning_objective": config.get("palette_tuning_objective"),
        "palette_nearest_swatch": config.get("palette_nearest_swatch"),
        "retuned_poles": [row for row in statistics if row["axis"] == args.axis]})
    del targets, model
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    gallery = None
    if not getattr(args, "skip_gallery", False):
        render_poles(args.out / "calibrated_poles.json", args.out / "pole_results", output_images, args.device, isolated=True)
        gallery = str(args.out / "pole_results/index.html")
    if not getattr(args, "quiet", False):
        print(json.dumps({"output": str(args.out), "new_runs": len(fresh_records),
            "unchanged_other_poles": 12 - len(sides), "all_pole_gallery": gallery}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/data/data1/semantic_isp/results/exp53_strong_words_one_dof_q90"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=8.)
    parser.add_argument("--side", choices=("both", "positive", "negative"), default="both")
    parser.add_argument("--axis", default="sharpness")
    parser.add_argument("--contrast-range", nargs=2, type=float)
    parser.add_argument("--tone-gain-range", nargs=2, type=float,
        help="Override tone-gain bounds while retuning both brightness poles")
    parser.add_argument("--sharpen-sigma", type=float)
    parser.add_argument("--palette-strength-weight", type=float, default=.6)
    parser.add_argument("--palette-strength-target", type=float, default=.17,
        help="Target absolute physical palette coordinate for the directional tuning prior")
    parser.add_argument("--palette-positive-mix", type=float, default=.65)
    parser.add_argument("--palette-negative-mix", type=float, default=.50)
    parser.add_argument("--palette-softness-ratio", type=float, default=.18)
    parser.add_argument("--color-protection-mode",
        choices=("none", "gray_white", "skin", "gray_white_and_skin"))
    parser.add_argument("--color-protection-operations", nargs="+",
        choices=("warmth", "saturation", "palette"))
    parser.add_argument("--sharpen-gain", type=float)
    parser.add_argument("--sharpen-limit", type=float)
    parser.add_argument("--blur-color-mode", choices=("preserve", "luminance", "rgb"), default="preserve")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-gallery", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.iters < 1 or not math.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("Iterations and sigma must be positive and finite")
    if (args.sharpen_gain is None) != (args.sharpen_limit is None):
        parser.error("Specify both --sharpen-gain and --sharpen-limit")
    if args.sharpen_gain is not None and (args.side != "positive" or any(
            not math.isfinite(value) or value <= 0 for value in (args.sharpen_gain, args.sharpen_limit))):
        parser.error("Positive capacity changes require --side positive and finite positive gain/limit")
    if args.blur_color_mode != "preserve" and args.side != "negative":
        parser.error("Blur color-mode changes require --side negative")
    if args.sharpen_sigma is not None and (not math.isfinite(args.sharpen_sigma) or args.sharpen_sigma <= 0):
        parser.error("Sharpening sigma must be positive and finite")
    palette_values = (args.palette_strength_weight, args.palette_positive_mix, args.palette_negative_mix)
    if any(not math.isfinite(value) or value < 0 for value in palette_values):
        parser.error("Palette strengths and prior weight must be finite and nonnegative")
    if args.palette_strength_weight and args.axis != "palette":
        parser.error("The directional strength prior is only available for palette retuning")
    if not 0 < args.palette_softness_ratio <= 1:
        parser.error("Palette softness ratio must be in (0, 1]")
    run(args)