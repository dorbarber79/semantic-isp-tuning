"""Render projected CLIP pole centroids or isolated calibrated axis endpoints."""

import argparse
import csv
import html
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import open_clip

from differentiable_isp import load_and_preprocess_raw
from semantic_isp import (
    SemanticProjection, compose_parameters, render_linear, linear_to_srgb8,
)

DATA = Path("/data/data1/EmotionISP_dataset/dng_filtered")
DEFAULT_CONFIG = Path("/data/data1/semantic_isp/results/exp38_clip_learned_poles_100/calibrated_poles.json")
DEFAULT_IMAGES = [1, 2, 33]
POLE_NAMES = {"positive": {"thermal": "Golden", "fidelity": "Documentary",
                            "moral": "High key", "vitality": "Vivid",
                            "hue_shift": "Teal-Orange blockbuster"},
              "negative": {"thermal": "Cyan", "fidelity": "Ethereal",
                            "moral": "Chiaroscuro", "vitality": "Bleached",
                            "hue_shift": "Magenta-Green synthwave retro"}}


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def save_png(path, image):
    pixels = linear_to_srgb8(image)[0].permute(1, 2, 0).detach().cpu().numpy()
    if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write {path}")


def contact_sheet(directory, labels, filenames):
    columns, tile_w, tile_h, header = 4, 420, 300, 58
    rows = (len(labels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + header)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, filename) in enumerate(zip(labels, filenames)):
        x = (index % columns) * tile_w
        y = (index // columns) * (tile_h + header)
        label_font = font(18)
        lines, line = [], ""
        for word in label.split():
            candidate = f"{line} {word}".strip()
            if line and draw.textlength(candidate, font=label_font) > tile_w - 16:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        draw.multiline_text((x + 8, y + 6), "\n".join(lines), fill="black", font=label_font, spacing=3)
        with Image.open(directory / filename) as image:
            image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
            sheet.paste(image, (x + (tile_w - image.width) // 2,
                                y + header + (tile_h - image.height) // 2))
    sheet.save(directory / "pole_centroids.jpg", quality=94)


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def run(config_path, output, images, device, isolated=False):
    config = json.loads(config_path.read_text())
    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=device,
        force_quick_gelu=config.get("force_quick_gelu", config["pretrained"] == "openai"))
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])

    @torch.no_grad()
    def encode_text(texts):
        return model.encode_text(tokenizer(texts).to(device)).float()

    projection = SemanticProjection(encode_text, config)
    axes = projection.axes
    centroids = projection.centroids.reshape(len(axes) * 2, -1)
    raw_scores, projected_coordinates = projection.project_features(centroids)
    coordinates = projected_coordinates.clone()
    if isolated:
        coordinates.zero_()
        for axis_index in range(len(axes)):
            coordinates[axis_index * 2, axis_index] = -float(config["coordinate_limit"])
            coordinates[axis_index * 2 + 1, axis_index] = float(config["coordinate_limit"])
    zero = torch.zeros(len(axes), device=device)
    neutral_parameters = compose_parameters(zero, config)
    labels, filenames, records = ["Neutral"], ["neutral.png"], []
    for axis_index, axis in enumerate(axes):
        for sign_index, sign in enumerate(("negative", "positive")):
            index = axis_index * 2 + sign_index
            pole_label = config.get("descriptor_table", {}).get("axes", {}).get(axis, {}).get(
                f"{sign}_pole", POLE_NAMES[sign].get(axis, axis))
            labels.append(f"{axis}: {pole_label}")
            filenames.append(f"pole_{axis}_{sign}.png")
            records.append({
                "axis": axis, "pole": pole_label, "sign": sign,
                "raw_cosine_differences": dict(zip(axes, raw_scores[index].cpu().tolist())),
                "learned_coordinates": dict(zip(axes, coordinates[index].cpu().tolist())),
                "projected_centroid_coordinates": dict(zip(axes, projected_coordinates[index].cpu().tolist())),
                "parameters": {key: json_value(value)
                               for key, value in compose_parameters(coordinates[index], config).items()},
                "descriptor_texts": config["poles"][axis][sign]["texts"],
            })

    output.mkdir(parents=True, exist_ok=True)
    descriptor_rows = []
    descriptor_axes = config.get("descriptor_table", {}).get("axes", {})
    for axis in axes:
        for sign in ("negative", "positive"):
            entries = descriptor_axes.get(axis, {}).get(sign, [])
            if entries:
                descriptor_rows.extend({"axis": axis,
                    "pole": descriptor_axes[axis].get(f"{sign}_pole", axis),
                    "sign": sign, **entry} for entry in entries)
            else:
                descriptor_rows.extend({"axis": axis, "pole": POLE_NAMES[sign].get(axis, axis),
                    "sign": sign, "id": f"D{index + 1}", "group": f"D{index + 1}",
                    "type": "unknown", "text": text}
                    for index, text in enumerate(config["poles"][axis][sign]["texts"]))
    save_csv(output / "pole_descriptors.csv", descriptor_rows)
    endpoint_rows = []
    for record in records:
        parameters = record["parameters"]
        endpoint_rows.append({"axis": record["axis"], "pole": record["pole"],
            "sign": record["sign"], "warmth": float(parameters["wb"][0]) - 1,
            "saturation": parameters["saturation"], "tone_gain": parameters["tone_gain"],
            "contrast": parameters["contrast"], "sharpness": parameters["sharpness"],
            "palette_coordinate": parameters["palette_coordinate"],
            "coordinates_json": json.dumps(record["learned_coordinates"]),
            "descriptor_texts": " | ".join(record["descriptor_texts"])})
    save_csv(output / "pole_endpoint_settings.csv", endpoint_rows)
    for image_id in images:
        image = load_and_preprocess_raw(str(DATA / f"{image_id}.dng"), None).to(device)
        directory = output / f"img_{image_id}"
        directory.mkdir(exist_ok=True)
        with torch.no_grad():
            save_png(directory / "neutral.png", render_linear(image, neutral_parameters))
            for index, record in enumerate(records):
                setting = compose_parameters(coordinates[index], config)
                save_png(directory / filenames[index + 1], render_linear(image, setting))
        contact_sheet(directory, labels, filenames)
        del image
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"rendered centroids on image {image_id}", flush=True)

    interpretation = (
        "Isolated calibrated endpoints: one semantic axis at its signed coordinate limit, all other axes zero. "
        "Direct single-DOF mappings change only the matching physical control. These are not projected centroid queries."
        if isolated else
        "Each pole centroid is projected independently through the learned axis calibration; other projections can be nonzero because the semantic poles are not perfectly orthogonal.")
    report = {"config_file": str(config_path), "model": config["model"],
              "pretrained": config["pretrained"], "images": images,
              "centroids": records,
              "render_mode": "isolated_axis_endpoints" if isolated else "projected_centroids",
              "interpretation": interpretation}
    (output / "results.json").write_text(json.dumps(report, indent=2))
    document = ["<!doctype html><meta charset='utf-8'><title>Pole results</title>",
                "<style>body{font:15px sans-serif;max-width:1600px;margin:2rem auto;padding:0 1rem}"
                "table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #bbb;padding:.45rem;vertical-align:top}"
                "th{background:#eee}td li{margin:.25rem 0}img{max-width:100%;height:auto}figure{margin:1rem 0}</style>",
                f"<h1>{'Isolated direct ISP pole results' if isolated else 'Projected text-pole centroid results'}</h1>",
                "<p>" + html.escape(interpretation) + "</p>"]
    document.append("<p><a href='pole_descriptors.csv'>Download all pole words</a> | "
                    "<a href='pole_endpoint_settings.csv'>Download physical endpoint settings</a> | "
                    "<a href='results.json'>Full coordinates and parameters</a></p>")
    document.append("<h2>Pole description table</h2><table><thead><tr><th>Axis / pole</th>"
                    "<th>D1 technical</th><th>D2 perceptual</th><th>D3 technical</th>"
                    "<th>D4 associative</th></tr></thead><tbody>")
    for axis in axes:
        for sign in ("negative", "positive"):
            entries = descriptor_axes.get(axis, {}).get(sign, [])
            label = descriptor_axes.get(axis, {}).get(f"{sign}_pole", axis)
            grouped = {group: [] for group in ("D1", "D2", "D3", "D4")}
            if entries:
                for entry in entries:
                    grouped[entry.get("group", entry["id"])].append(entry["text"])
            else:
                for index, text in enumerate(config["poles"][axis][sign]["texts"]):
                    grouped[f"D{index + 1}"].append(text)
            cells = "".join("<td><ol>" + "".join(f"<li>{html.escape(text)}</li>" for text in grouped[group])
                            + "</ol></td>" for group in grouped)
            document.append(f"<tr><th>{html.escape(axis)} / {html.escape(label)} ({sign})</th>{cells}</tr>")
    document.append("</tbody></table><h2>Rendered pole images</h2>")
    for image_id in images:
        document.append(f"<h3>Image {image_id}</h3><figure><a href='img_{image_id}/pole_centroids.jpg'>"
                        f"<img src='img_{image_id}/pole_centroids.jpg'></a></figure><ul>")
        for record in records:
            filename = f"pole_{record['axis']}_{record['sign']}.png"
            document.append(f"<li><a href='img_{image_id}/{filename}'>"
                            f"{html.escape(record['axis'])}: {html.escape(record['pole'])} ({record['sign']})</a></li>")
        document.append("</ul>")
    document.append("<h2>Projected coordinates and parameters</h2><pre>"
                    + html.escape(json.dumps(records, indent=2)) + "</pre>")
    (output / "index.html").write_text("\n".join(document))
    print(json.dumps({"images": len(images), "poles": len(records),
                      "renders": len(images) * (len(records) + 1), "output": str(output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=int, nargs="+", default=DEFAULT_IMAGES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--isolated", action="store_true", help="Render one axis at a time instead of the full text-centroid projection")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    run(args.config, args.out, args.images, args.device, isolated=args.isolated)


if __name__ == "__main__":
    main()
