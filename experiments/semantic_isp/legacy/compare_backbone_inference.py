"""Run CLIP/SigLIP text mapping and render selected corpus words side by side."""

import argparse
import csv
import html
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont
import torch

from differentiable_isp import load_and_preprocess_raw
from exp38_learn_poles import save_json
from semantic_isp import (CONTROL_NAMES, SemanticProjection, compose_parameters,
                                linear_to_srgb8, render_linear)


PROMPT_GROUPS = {
    "easy": [
        "warm", "cool", "bright", "dark", "saturated", "muted",
        "sharp", "monochrome", "retro", "golden", "faded", "crisp",
    ],
    "hard": [
        "angelic", "ancient", "ethereal", "noir", "tropical", "gothic",
        "melancholic", "futuristic", "pastoral", "mystical", "whimsical", "bohemian",
    ],
}
SCENES = (2, 33, 384)
MODEL_KEYS = ("clip", "siglip")
MODEL_PATHS = {
    "clip": Path("/data/data1/semantic_isp/results/exp71_palette_only_protection/clip/03_palette_protected/direct_soft_calibration.json"),
    "siglip": Path("/data/data1/semantic_isp/results/exp71_palette_only_protection/siglip/03_palette_protected/direct_soft_calibration.json"),
}
DATA = Path("/data/data1/EmotionISP_dataset/dng_filtered")


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def save_rgb(path, image):
    pixels = linear_to_srgb8(image)[0].permute(1, 2, 0).detach().cpu().numpy()
    Image.fromarray(pixels, mode="RGB").save(path)


def to_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def mapped_physical_controls(parameters):
    return {
        "warmth": float(parameters["wb"][0] - 1),
        "saturation": float(parameters.get("saturation", 0.)),
        "tone_gain": float(parameters["tone_gain"]),
        "contrast": float(parameters["contrast"]),
        "sharpness": float(parameters["sharpness"]),
        "palette_coordinate": float(parameters["palette_coordinate"]),
    }


def make_sheet(folder, image_id, prompts):
    tile_w, tile_h, header, label_w = 260, 210, 46, 160
    columns = 4
    rows = len(prompts)
    sheet = Image.new("RGB", (label_w + columns * tile_w, rows * (tile_h + header)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for column, title in enumerate(("Neutral", "CLIP", "SigLIP")):
        draw.text((label_w + column * tile_w + 8, 8), title, fill="black", font=font)
    for row_index, (category, prompt) in enumerate(prompts):
        top = row_index * (tile_h + header)
        draw.text((6, top + 8), category, fill="#555", font=font)
        draw.text((6, top + 26), prompt[:20], fill="black", font=font)
        names = ["neutral.png", f"{slug(prompt)}_clip.png", f"{slug(prompt)}_siglip.png"]
        for column, filename in enumerate(names):
            with Image.open(folder / filename) as source:
                source.thumbnail((tile_w - 8, tile_h - 8), Image.Resampling.LANCZOS)
                sheet.paste(source, (label_w + column * tile_w + (tile_w - source.width) // 2,
                                     top + header + (tile_h - source.height) // 2))
    output = folder / f"side_by_side_image_{image_id}.jpg"
    sheet.save(output, quality=94)
    return output


def run(args):
    import open_clip

    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {args.device}")
    args.out.mkdir(parents=True)
    prompts = [(category, word) for category, words in PROMPT_GROUPS.items() for word in words]
    for _, word in prompts:
        if word not in args.dataset_words:
            raise ValueError(f"Prompt is not in the style corpus: {word}")

    projections = {}
    models = {}
    tokenizers = {}
    configs = {}
    for model_key in MODEL_KEYS:
        config_path = Path(args.configs[model_key])
        config = json.loads(config_path.read_text())
        if config["color_protection"].get("operations") != ["palette"]:
            raise ValueError(f"{model_key} config does not scope protection to palette only")
        model, _, _ = open_clip.create_model_and_transforms(
            config["model"], pretrained=config["pretrained"], device=args.device,
            force_quick_gelu=config.get("force_quick_gelu", config["pretrained"] == "openai"))
        model.eval().requires_grad_(False)
        tokenizer = open_clip.get_tokenizer(config["model"])

        @torch.no_grad()
        def encode_text(texts, model=model, tokenizer=tokenizer):
            return model.encode_text(tokenizer(texts).to(args.device)).float()

        projections[model_key] = SemanticProjection(encode_text, config)
        models[model_key] = model
        tokenizers[model_key] = tokenizer
        configs[model_key] = config

    output_rows = []
    page = ["<!doctype html><meta charset='utf-8'><title>CLIP vs SigLIP inference comparison</title>",
        "<h1>CLIP vs SigLIP text mapping and ISP inference</h1>",
        "<p>Rows are prompts from clear control words and held-out subjective styles. "
        "Each sheet shows the same neutral image and the independently mapped CLIP and SigLIP outputs. "
        "Coordinates and control values are recorded per model; cosine magnitudes are not cross-model comparable.</p>",
        "<p><a href='inference_mapping.csv'>Coordinates and physical controls</a> | "
        "<a href='prompt_selection.json'>Prompt groups</a></p>"]
    for image_id in args.images:
        image = load_and_preprocess_raw(str(DATA / f"{image_id}.dng"), None).to(args.device)
        neutral = render_linear(image, compose_parameters(image.new_zeros(6), configs["clip"]))
        directory = args.out / f"img_{image_id}"
        directory.mkdir()
        save_rgb(directory / "neutral.png", neutral)
        sheet_prompts = []
        for category, prompt in prompts:
            formatted = f"a {prompt} photo"
            sheet_prompts.append((category, prompt))
            for model_key in MODEL_KEYS:
                projection = projections[model_key]
                raw_scores, coordinates = projection.project([formatted])
                probabilities, gates = projection.relevance_weights(
                    models[model_key].encode_text(tokenizers[model_key](formatted).to(args.device)).float())
                setting = compose_parameters(coordinates[0], configs[model_key])
                rendered = render_linear(image, setting)
                filename = f"{slug(prompt)}_{model_key}.png"
                save_rgb(directory / filename, rendered)
                output_rows.append({
                    "image": image_id, "category": category, "word": prompt,
                    "formatted_prompt": formatted, "model": model_key,
                    "raw_scores_json": json.dumps(dict(zip(projection.axes, raw_scores[0].cpu().tolist()))),
                    "coordinates_json": json.dumps(dict(zip(projection.axes, coordinates[0].cpu().tolist()))),
                    "gate_probabilities_json": json.dumps(dict(zip(projection.axes, probabilities[0].cpu().tolist()))),
                    "gate_weights_json": json.dumps(dict(zip(projection.axes, gates[0].cpu().tolist()))),
                    "physical_controls_json": json.dumps(mapped_physical_controls(setting)),
                    "protection_mode": configs[model_key]["color_protection"]["mode"],
                    "protected_operations": ",".join(configs[model_key]["color_protection"].get("operations", [])),
                })
        sheet = make_sheet(directory, image_id, sheet_prompts)
        page.append(f"<h2>Scene {image_id}</h2><a href='img_{image_id}/{sheet.name}'>"
                    f"<img style='max-width:100%' src='img_{image_id}/{sheet.name}'></a>")
        print(f"rendered scene {image_id}: {len(prompts)} prompts x 2 models", flush=True)

    fields = list(output_rows[0])
    with (args.out / "inference_mapping.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    prompt_info = {category: [{"word": word, "in_corpus": True} for group, word in prompts if group == category]
                   for category in PROMPT_GROUPS}
    save_json(args.out / "prompt_selection.json", prompt_info)
    summary = {
        "models": {key: {"model": configs[key]["model"], "pretrained": configs[key]["pretrained"],
                          "config": str(args.configs[key])} for key in MODEL_KEYS},
        "images": args.images, "prompts_per_group": {key: len(value) for key, value in PROMPT_GROUPS.items()},
        "rows": len(output_rows), "protection_policy": "gray/white and skin protection applies to palette only",
        "score_note": "CLIP and SigLIP coordinates use independent model-specific centroids, scales, and gates; cosine values are not compared directly.",
    }
    save_json(args.out / "summary.json", summary)
    (args.out / "index.html").write_text("\n".join(page))
    print(json.dumps({"output": str(args.out), "rows": len(output_rows),
                      "sheets": len(args.images)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=int, nargs="+", default=list(SCENES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--configs", type=json.loads,
        default=json.loads(json.dumps({
            "clip": str(MODEL_PATHS["clip"]), "siglip": str(MODEL_PATHS["siglip"])})))
    corpus = json.loads(Path("/data/data1/semantic_isp/datasets/style_words/style_corpus.json").read_text())
    parser.set_defaults(dataset_words=set(corpus["anchors"] + corpus["seen_words"] + corpus["unseen_test_words"] + corpus["compound_phrases"]))
    run(parser.parse_args())
