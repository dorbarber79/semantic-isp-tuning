"""Retune and compare all direct semantic ISP poles with CLIP and SigLIP."""

import argparse
import csv
import html
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont
import torch

from exp38_learn_poles import fit_score_calibration, save_json
from exp42_phrase_pole_calibration import descriptor_rows
from exp49_direct_isp_comparison import direct_quantile_calibration, fit_soft_gate
from exp55_large_kernel_detail_retune import run as retune_axis
from semantic_isp import SemanticProjection


MODEL_SPECS = {
    "clip": {
        "model": "ViT-B-32", "pretrained": "openai", "force_quick_gelu": True,
        "image_mean": [.48145466, .4578275, .40821073],
        "image_std": [.26862954, .26130258, .27577711],
    },
    "siglip": {
        "model": "ViT-B-16-SigLIP", "pretrained": "webli", "force_quick_gelu": False,
        "image_mean": [.5, .5, .5], "image_std": [.5, .5, .5],
    },
}
PROTECTION_MODE = "gray_white_and_skin"
AXES = ("warmth", "saturation", "brightness", "contrast", "sharpness", "palette")
SIGNS = ("negative", "positive")
REFERENCE_CALIBRATION = Path(
    "/data/data1/semantic_isp/results/exp69_color_protection_retune/"
    "gray_white_and_skin/03_palette/direct_soft_calibration.json")


def prepare_model_base(source_path, output, model_key, device):
    import open_clip

    config = json.loads(source_path.read_text())
    config.update(MODEL_SPECS[model_key])
    config["color_protection"] = {
        "mode": PROTECTION_MODE, "strength": 1., "gray_sigma": .07,
        "skin_sigma": .10, "skin_reference_srgb": [.76, .57, .47],
        "saturation_positive_only": True,
    }
    config["palette_tuning_objective"] = {
        "clip_target": "exact pole sentence embedding",
        "directional_strength_weight": .6,
        "directional_strength_target_coordinate": .17,
        "directional_strength_prior": "quadratic reward toward the requested palette coordinate magnitude",
        "operator_strengths": config["palette_split_tone"],
    }
    config["backbone_comparison"] = {
        "model_key": model_key, "model": config["model"], "pretrained": config["pretrained"],
        "image_mean": config["image_mean"], "image_std": config["image_std"],
        "text_score_calibration": "freshly recomputed for this model",
        "relevance_gate": "freshly recomputed for this model",
        "physical_endpoints": "all six axes are freshly retuned in later stages",
    }

    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=device,
        force_quick_gelu=config["force_quick_gelu"])
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])

    @torch.no_grad()
    def encode_text(texts):
        return model.encode_text(tokenizer(texts).to(device)).float()

    rows = descriptor_rows(config)
    projection = SemanticProjection(encode_text, config)
    raw_scores, _ = projection.project([row["text"] for row in rows])
    generic_origin, _ = projection.project(["a photo"])
    calibrated = fit_score_calibration(raw_scores.cpu().numpy(),
                                       generic_origin[0].cpu().numpy(), config)
    config["learned_calibration"] = calibrated["learned_calibration"]
    gate, split = fit_soft_gate(config, encode_text)
    config["relevance_gate"] = gate
    config["backbone_comparison"]["gate_diagnostics"] = split["diagnostics"]
    config["status"] = f"Fresh {model_key} six-axis initialization before all-pole q90 autotuning"
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "direct_soft_calibration.json", config)
    save_json(output / "calibration_metadata.json", {
        "source_config": str(source_path), "model_key": model_key,
        "model": config["model"], "pretrained": config["pretrained"],
        "force_quick_gelu": config["force_quick_gelu"],
        "image_mean": config["image_mean"], "image_std": config["image_std"],
        "descriptor_revision": config["descriptor_table"]["descriptor_revision"],
        "relevance_gate_diagnostics": split["diagnostics"],
    })


def retune_model(model_key, model_root, source_config, args):
    base = model_root / "model_base"
    prepare_model_base(source_config, base, model_key, args.device)
    current_base = base
    axes = AXES
    for index, axis in enumerate(axes):
        output = model_root / f"axis_{index + 1:02d}_{axis}"
        options = SimpleNamespace(
            base=current_base, out=output, iters=args.iters, sigma=8., side="both", axis=axis,
            contrast_range=None, sharpen_sigma=None,
            palette_strength_weight=args.palette_strength_weight,
            palette_strength_target=args.palette_strength_target,
            palette_positive_mix=.85, palette_negative_mix=.75,
            palette_softness_ratio=.18, sharpen_gain=None, sharpen_limit=None,
            blur_color_mode="preserve", device=args.device, smoke=args.smoke,
            color_protection_mode=PROTECTION_MODE,
            skip_gallery=index != len(axes) - 1, quiet=True,
        )
        log_path = model_root / f"axis_{index + 1:02d}_{axis}.log"
        with log_path.open("w") as log:
            with redirect_stdout(log):
                retune_axis(options)
        current_base = output
        print(f"completed {model_key}/{axis}", flush=True)

    calibration_path = current_base / "direct_soft_calibration.json"
    config = json.loads(calibration_path.read_text())
    config["status"] = f"Fresh {model_key} all-six-axis one-DOF calibration; q90 endpoints; combined color protection"
    config["backbone_comparison"].update({
        "all_axes_freshly_retuned": list(axes),
        "fresh_phrase_image_optimizations": 192 if args.smoke else 1344,
        "protection_mode": PROTECTION_MODE,
        "endpoint_rule": "direction-aligned q90; positive numerical q90, negative numerical q10",
    })
    save_json(calibration_path, config)
    save_json(current_base / "calibrated_poles.json", config)
    return {
        "root": str(model_root), "final_stage": str(current_base),
        "config": str(calibration_path),
        "gallery": str(current_base / "pole_results/index.html"),
        "endpoint_statistics": str(current_base / "pole_quantile_statistics.csv"),
        "fresh_optimizations": 192 if args.smoke else 1344,
    }


def read_endpoint_rows(path):
    with Path(path).open(newline="") as stream:
        return {(row["axis"], row["sign"]): row for row in csv.DictReader(stream)}


def save_endpoint_comparison(path, clip_rows, siglip_rows, reference_rows):
    fields = ["axis", "sign", "control", "clip_selected", "siglip_selected", "siglip_minus_clip",
              "existing_protected_clip_selected", "fresh_clip_minus_existing"]
    rows = []
    for key in sorted(clip_rows):
        clip_value = float(clip_rows[key]["selected_setting"])
        siglip_value = float(siglip_rows[key]["selected_setting"])
        existing_value = float(reference_rows[key]["selected_setting"])
        rows.append({"axis": key[0], "sign": key[1], "control": clip_rows[key]["control"],
            "clip_selected": clip_value, "siglip_selected": siglip_value,
            "siglip_minus_clip": siglip_value - clip_value,
            "existing_protected_clip_selected": existing_value,
            "fresh_clip_minus_existing": clip_value - existing_value})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_sbs_sheet(output, image_id, clip_root, siglip_root):
    width, height, header = 300, 235, 44
    rows, columns = len(AXES) * 2 + 1, 4
    sheet = Image.new("RGB", (columns * width, rows * (height + header)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    labels = ["Neutral", "CLIP", "SigLIP"]
    for column, label in enumerate(labels, start=1):
        draw.text((column * width + 8, 8), label, fill="black", font=font)
    pairs = [(axis, sign) for axis in AXES for sign in SIGNS]
    for row_index, (axis, sign) in enumerate(pairs, start=1):
        y = row_index * (height + header)
        draw.text((6, y + 12), f"{axis} {sign}", fill="black", font=font)
        filename = f"pole_{axis}_{sign}.png"
        for column, root in ((1, clip_root), (2, clip_root), (3, siglip_root)):
            source_path = root / f"img_{image_id}" / ("neutral.png" if column == 1 else filename)
            with Image.open(source_path) as source:
                source.thumbnail((width - 10, height - 8), Image.Resampling.LANCZOS)
                sheet.paste(source, (column * width + (width - source.width) // 2,
                                     y + header + (height - source.height) // 2))
    path = output / f"side_by_side_image_{image_id}.jpg"
    sheet.save(path, quality=94)
    return path


def run(args):
    import open_clip

    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {args.device}")
    args.out.mkdir(parents=True)
    source_config = Path(args.base_config)
    base_config = json.loads(source_config.read_text())
    if base_config.get("descriptor_table", {}).get("descriptor_revision") != "moran_style_v7_moderate_palette_4x4":
        raise ValueError("Backbone comparison requires the current v7 direct-control descriptor table")
    if base_config.get("color_protection", {}).get("mode") != PROTECTION_MODE:
        raise ValueError("Backbone comparison requires both gray/white and skin protections")

    models = {}
    for model_key in ("clip", "siglip"):
        model_root = args.out / model_key
        model_root.mkdir()
        models[model_key] = retune_model(model_key, model_root, source_config, args)

    clip_rows = read_endpoint_rows(models["clip"]["endpoint_statistics"])
    siglip_rows = read_endpoint_rows(models["siglip"]["endpoint_statistics"])
    reference_stats = source_config.parent / "pole_quantile_statistics.csv"
    reference_rows = read_endpoint_rows(reference_stats)
    endpoint_rows = save_endpoint_comparison(args.out / "endpoint_comparison.csv",
                                             clip_rows, siglip_rows, reference_rows)

    clip_gallery = Path(models["clip"]["gallery"]).parent
    siglip_gallery = Path(models["siglip"]["gallery"]).parent
    output_scenes = [2] if args.smoke else [2, 33, 384]
    sheets = [str(make_sbs_sheet(args.out, image_id, clip_gallery, siglip_gallery))
              for image_id in output_scenes]
    summary = {
        "protection_mode": PROTECTION_MODE,
        "descriptor_revision": base_config["descriptor_table"]["descriptor_revision"],
        "training_images": [1] if args.smoke else [1, 28, 236, 194, 277, 263, 103],
        "output_images": output_scenes,
        "iterations": args.iters,
        "fresh_runs_per_model": models["clip"]["fresh_optimizations"],
        "total_fresh_runs": sum(result["fresh_optimizations"] for result in models.values()),
        "models": {
            key: {**value, "model": MODEL_SPECS[key]["model"], "pretrained": MODEL_SPECS[key]["pretrained"],
                  "image_mean": MODEL_SPECS[key]["image_mean"], "image_std": MODEL_SPECS[key]["image_std"]}
            for key, value in models.items()},
        "endpoint_comparison_csv": str(args.out / "endpoint_comparison.csv"),
        "side_by_side_sheets": sheets,
        "comparison_note": "CLIP and SigLIP share data, descriptors, optimizer, protection, and q90 endpoint rule. Score magnitudes are not cross-model comparable.",
    }
    save_json(args.out / "summary.json", summary)
    document = ["<!doctype html><meta charset='utf-8'><title>CLIP vs SigLIP ISP poles</title>",
        "<h1>CLIP vs SigLIP: protection-aware direct ISP pole calibration</h1>",
        "<p>Both models use the same six-axis descriptors, seven training images, one-control "
        "phrase autotuning, combined gray/white and skin protections, and direction-aligned q90 "
        "physical endpoints. Cosine magnitudes are model-specific and are not compared directly.</p>",
        "<p><a href='endpoint_comparison.csv'>Endpoint comparison</a> | "
        "<a href='summary.json'>Run metadata</a></p>"]
    for image_id, sheet in zip(output_scenes, sheets):
        relative = Path(sheet).name
        document.append(f"<h2>Scene {image_id}</h2><a href='{relative}'>"
                        f"<img style='max-width:100%' src='{relative}'></a>")
    for model_key, value in models.items():
        document.append(f"<h2>{html.escape(model_key.upper())}</h2>"
                        f"<p><a href='{html.escape(str(Path(value['gallery']).relative_to(args.out)))}'>"
                        "Open full pole gallery</a></p>")
    (args.out / "index.html").write_text("\n".join(document))
    print(json.dumps({"output": str(args.out), "endpoint_rows": len(endpoint_rows),
                      "side_by_side_sheets": sheets}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=REFERENCE_CALIBRATION)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--palette-strength-weight", type=float, default=.6)
    parser.add_argument("--palette-strength-target", type=float, default=.17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
