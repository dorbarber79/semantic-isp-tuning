"""Render two clear requests per learned Moran pole on RAW image 2."""

import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

import open_clip

from differentiable_isp import load_and_preprocess_raw
from exp38_learn_poles import encode_image
from moran_semantic_isp import AXES, SemanticProjection, compose_parameters, render_linear, linear_to_srgb8

CALIBRATION = Path("/data/data1/semantic_isp/results/exp42_phrase_pole_clip_100/calibrated_poles.json")
IMAGE = Path("/data/data1/EmotionISP_dataset/dng_filtered/2.dng")
REQUESTS = [
    ("warm", "thermal"), ("cold", "thermal"),
    ("dreamy", "fidelity"), ("documentary", "fidelity"),
    ("chiaroscuro", "moral"), ("high key", "moral"),
    ("vivid", "vitality"), ("bleached", "vitality"),
    ("retro", "hue_shift"), ("blockbuster", "hue_shift"),
]


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def save_png(path, image):
    pixels = linear_to_srgb8(image)[0].detach().cpu().permute(1, 2, 0).numpy()
    if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write {path}")


def make_sheet(folder, words):
    tile_w, tile_h, columns, header = 480, 320, 3, 42
    rows = (len(words) + columns) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + header)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, word in enumerate(words):
        x, y = (index % columns) * tile_w, (index // columns) * (tile_h + header)
        draw.text((x + 8, y + 8), word, fill="black", font=font(20))
        with Image.open(folder / f"{word.replace(' ', '_')}.png") as image:
            image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
            sheet.paste(image, (x + (tile_w - image.width) // 2,
                                y + header + (tile_h - image.height) // 2))
    sheet.save(folder / "comparison.jpg", quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--image", type=Path, default=IMAGE)
    parser.add_argument("--out", type=Path, default=Path("/data/data1/semantic_isp/results/exp44_image2_clear_poles"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.out}")

    config = json.loads(args.calibration.read_text())
    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=args.device,
        force_quick_gelu=config.get("force_quick_gelu", config["pretrained"] == "openai"))
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])

    @torch.no_grad()
    def encode_text(texts):
        return model.encode_text(tokenizer(texts).to(args.device)).float()

    prompts = [f"a {word} photo" for word, _ in REQUESTS]
    text_features = F.normalize(encode_text(prompts), dim=-1)
    projection = SemanticProjection(encode_text, config)
    _, coordinates = projection.project(prompts)
    image = load_and_preprocess_raw(str(args.image), None).to(args.device)
    neutral_params = compose_parameters(torch.zeros(5, device=args.device), config)
    mean = image.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
    std = image.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]

    args.out.mkdir(parents=True, exist_ok=True)
    neutral = render_linear(image, neutral_params)
    save_png(args.out / "neutral.png", neutral)
    neutral_feature = encode_image(model, neutral)
    rows = []
    report = {"image": args.image.name, "calibration": str(args.calibration), "requests": []}
    for index, (word, target_axis) in enumerate(REQUESTS):
        coordinate = coordinates[index]
        parameters = compose_parameters(coordinate, config)
        rendered = render_linear(image, parameters)
        filename = f"{word.replace(' ', '_')}.png"
        save_png(args.out / filename, rendered)
        feature = encode_image(model, rendered)
        score = float((feature @ text_features[index:index + 1].T).mean())
        neutral_score = float((neutral_feature @ text_features[index:index + 1].T).mean())
        axis_values = {axis: float(coordinate[axis_index]) for axis_index, axis in enumerate(AXES)}
        active_poles = {}
        for axis in AXES:
            coordinate = axis_values[axis]
            if coordinate > 0:
                active_poles[axis] = config["descriptor_table"]["axes"][axis]["positive_pole"]
            elif coordinate < 0:
                active_poles[axis] = config["descriptor_table"]["axes"][axis]["negative_pole"]
            else:
                active_poles[axis] = "Neutral"
        effective = {"warmth": float(parameters["wb"][0] - 1),
                     "wb_red_gain": float(parameters["wb"][0]),
                     "wb_green_gain": float(parameters["wb"][1]),
                     "wb_blue_gain": float(parameters["wb"][2]),
                     "saturation": float(parameters["saturation"]),
                     "tone_gain": float(parameters["tone_gain"]),
                     "contrast": float(parameters["contrast"]),
                     "sharpness": float(parameters["sharpness"]),
                     "sharpness_radius": float(parameters["radius"]),
                     "palette_coordinate": float(parameters["palette_coordinate"]),
                     **{f"ccm_{r}{c}": float(parameters["ccm"][r, c])
                        for r in range(3) for c in range(3)}}
        rows.append({"word": word, "target_axis": target_axis,
                     **{f"pole_{axis}": active_poles[axis] for axis in AXES},
                     **{f"axis_{axis}": axis_values[axis] for axis in AXES},
                     **effective, "neutral_cosine": neutral_score, "output_cosine": score,
                     "cosine_gain": score - neutral_score, "image": str(filename)})
        report["requests"].append({"word": word, "clip_prompt": prompts[index],
                                   "target_axis": target_axis, "active_poles": active_poles,
                                   "axis_values": axis_values, "isp_degrees_of_freedom": effective,
                                   "neutral_cosine": neutral_score, "output_cosine": score,
                                   "cosine_gain": score - neutral_score, "image_file": filename})
        print(word, {axis: round(value, 3) for axis, value in axis_values.items()},
              f"cosine_gain={score-neutral_score:+.4f}", flush=True)

    with (args.out / "per_word_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.out / "values.json").write_text(json.dumps(report, indent=2))
    make_sheet(args.out, [word for word, _ in REQUESTS])
    html_rows = []
    for row in rows:
        html_rows.append("<tr><td>" + html.escape(row["word"]) + "</td>"
                         + "".join(f"<td>{html.escape(str(row[f'pole_{axis}']))}</td>"
                                  f"<td>{row[f'axis_{axis}']:+.3f}</td>"
                                  for axis in AXES)
                         + f"<td>{row['warmth']:+.3f}</td><td>{row['saturation']:+.3f}</td>"
                         + f"<td>{row['tone_gain']:.3f}</td><td>{row['contrast']:.3f}</td>"
                         + f"<td>{row['sharpness']:+.3f}</td><td>{row['sharpness_radius']:.2f}</td>"
                         + f"<td>{row['palette_coordinate']:+.3f}</td>"
                         + f"<td>{row['cosine_gain']:+.4f}</td></tr>")
    page = ["<!doctype html><meta charset='utf-8'><title>Image 2 clear pole examples</title>",
            "<style>body{font:13px sans-serif;margin:20px}table{border-collapse:collapse;font-size:11px}td,th{border:1px solid #ccc;padding:4px}img{max-width:100%}th{position:sticky;top:0;background:#eee}</style>",
            "<h1>Image 2: one request for each pole</h1>",
            "<p>Coordinates and every ISP control are recorded per word. CLIP cosine gain is a self-score, not human preference.</p>",
            "<img src='comparison.jpg'><table><thead><tr><th>Word</th>"
            + "".join(f"<th>{axis} pole</th><th>{axis}</th>" for axis in AXES)
            + "<th>warmth</th><th>saturation</th><th>tone gain</th><th>contrast</th><th>sharpness</th><th>radius</th><th>palette</th><th>CLIP gain</th></tr></thead><tbody>"
            + "\n".join(html_rows) + "</tbody></table>"]
    (args.out / "index.html").write_text("\n".join(page))
    print("gallery", args.out / "index.html")


if __name__ == "__main__":
    main()
