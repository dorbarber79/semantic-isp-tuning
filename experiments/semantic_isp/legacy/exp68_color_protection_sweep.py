"""Compare gray/white and skin chromaticity protection for color ISP controls."""

import argparse
import csv
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from differentiable_isp import load_and_preprocess_raw
from axis_isp import chromaticity_proximity, linear_to_srgb8
from semantic_isp import CONTROL_NAMES, compose_parameters, render_linear


PROTECTION_MODES = (
    ("none", "No protection"),
    ("gray_white", "Gray/white only"),
    ("skin", "Skin only"),
    ("gray_white_and_skin", "Gray/white + skin"),
)
OPERATIONS = (
    ("warmth_negative", "warmth", -1),
    ("warmth_positive", "warmth", 1),
    ("saturation_positive", "saturation", 1),
    ("palette_negative", "palette", -1),
    ("palette_positive", "palette", 1),
)
GRAY_REFERENCE = (1 / 3, 1 / 3, 1 / 3)
SKIN_REFERENCE_SRGB = (.76, .57, .47)
GRAY_SIGMA = .07
SKIN_SIGMA = .10


def save_rgb(path, image):
    pixels = linear_to_srgb8(image)[0].permute(1, 2, 0).detach().cpu().numpy()
    Image.fromarray(pixels, mode="RGB").save(path)


def make_contact_sheet(directory, image_id):
    width, height, header = 280, 220, 50
    columns = len(PROTECTION_MODES) + 1
    rows = len(OPERATIONS)
    sheet = Image.new("RGB", (columns * width, rows * (height + header)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    column_labels = ["Neutral", *[label for _, label in PROTECTION_MODES]]
    for row_index, (case_name, _, _) in enumerate(OPERATIONS):
        top = row_index * (height + header)
        draw.text((8, top + 5), case_name.replace("_", " "), fill="black", font=font)
        for column, label in enumerate(column_labels):
            left = column * width
            draw.text((left + 6, top + 26), label, fill="black", font=font)
            filename = "neutral.png" if column == 0 else f"{case_name}_{PROTECTION_MODES[column - 1][0]}.png"
            with Image.open(directory / filename) as source:
                source.thumbnail((width - 10, height - 8), Image.Resampling.LANCZOS)
                sheet.paste(source, (left + (width - source.width) // 2,
                                     top + header + (height - source.height) // 2))
    output = directory / "color_protection_comparison.jpg"
    sheet.save(output, quality=94)
    return output


def effect_metrics(output, neutral, gray_mask, skin_mask, baseline_output):
    output_srgb = linear_to_srgb8(output).to(torch.float32)
    neutral_srgb = linear_to_srgb8(neutral).to(torch.float32)
    baseline_srgb = linear_to_srgb8(baseline_output).to(torch.float32)
    delta = (output_srgb - neutral_srgb).abs().mean(dim=1, keepdim=True)
    reference_delta = (output_srgb - baseline_srgb).abs().mean(dim=1, keepdim=True)

    def masked_mean(values, mask):
        if not bool(mask.any()):
            return float("nan")
        return float(values.masked_select(mask).mean())

    return {
        "effect_mae_from_neutral_0_255": float(delta.mean()),
        "gray_region_effect_mae_0_255": masked_mean(delta, gray_mask),
        "skin_region_effect_mae_0_255": masked_mean(delta, skin_mask),
        "mae_from_no_protection_0_255": float(reference_delta.mean()),
    }


def run(args):
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    config = json.loads(args.config.read_text())
    axes = tuple(config["axes"])
    limit = float(config["coordinate_limit"])
    if not {"warmth", "saturation", "palette"}.issubset(axes):
        raise ValueError("Calibration must include warmth, saturation, and palette axes")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")
    torch.set_num_threads(2)
    args.out.mkdir(parents=True)
    rows = []
    page = ["<!doctype html><meta charset='utf-8'><title>Color protection comparison</title>",
            "<h1>Color-operation protection comparison</h1>",
            "<p>Columns compare no protection, gray/white only, skin only, and both. "
            "Warmth starts from the camera-WB RAW decode; saturation is protected only "
            "for its positive pole; palette protection applies to both signs. The score "
            "uses normalized linear-RGB chromaticity, not luminance.</p>",
            "<p><a href='protection_metrics.csv'>Metrics CSV</a> | "
            "<a href='summary.json'>Configuration</a></p>"]
    for image_id in args.images:
        image = load_and_preprocess_raw(
            f"/data/data1/EmotionISP_dataset/dng_filtered/{image_id}.dng", None).to(args.device)
        neutral_coordinates = image.new_zeros(len(axes))
        neutral = render_linear(image, compose_parameters(neutral_coordinates, config))
        directory = args.out / f"img_{image_id}"
        directory.mkdir()
        save_rgb(directory / "neutral.png", neutral)

        gray_proximity = chromaticity_proximity(image, GRAY_REFERENCE, GRAY_SIGMA)
        skin_proximity = chromaticity_proximity(image, SKIN_REFERENCE_SRGB, SKIN_SIGMA,
                                                reference_is_srgb=True)
        gray_mask = gray_proximity >= .5
        skin_mask = skin_proximity >= .5

        for case_name, axis, sign in OPERATIONS:
            coordinates = image.new_zeros(len(axes))
            coordinates[axes.index(axis)] = sign * limit
            baseline_config = dict(config, color_protection={"mode": "none"})
            baseline_output = render_linear(image, compose_parameters(coordinates, baseline_config))
            save_rgb(directory / f"{case_name}_none.png", baseline_output)
            for mode, _ in PROTECTION_MODES[1:]:
                protection_config = dict(config, color_protection={
                    "mode": mode,
                    "strength": 1.,
                    "gray_sigma": GRAY_SIGMA,
                    "skin_sigma": SKIN_SIGMA,
                    "skin_reference_srgb": SKIN_REFERENCE_SRGB,
                    "saturation_positive_only": True,
                })
                output = render_linear(image, compose_parameters(coordinates, protection_config))
                save_rgb(directory / f"{case_name}_{mode}.png", output)
                metrics = effect_metrics(output, neutral, gray_mask, skin_mask, baseline_output)
                rows.append({"image": image_id, "operation": case_name, "axis": axis,
                    "pole_sign": sign, "protection_mode": mode, **metrics})
            baseline_metrics = effect_metrics(baseline_output, neutral, gray_mask, skin_mask,
                                              baseline_output)
            rows.append({"image": image_id, "operation": case_name, "axis": axis,
                "pole_sign": sign, "protection_mode": "none", **baseline_metrics})
        sheet = make_contact_sheet(directory, image_id)
        page.append(f"<h2>Scene {image_id}</h2><a href='img_{image_id}/{sheet.name}'>"
                    f"<img style='max-width:100%' src='img_{image_id}/{sheet.name}'></a>")
        print(f"rendered image {image_id}: {len(OPERATIONS)} operations x {len(PROTECTION_MODES)} modes",
              flush=True)

    csv_path = args.out / "protection_metrics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "config": str(args.config), "images": args.images,
        "protection_modes": [{"mode": key, "label": label} for key, label in PROTECTION_MODES],
        "operations": [{"name": name, "axis": axis, "sign": sign} for name, axis, sign in OPERATIONS],
        "color_metric": "Euclidean distance between normalized linear-RGB chromaticities; no luminance term",
        "gray_white_reference_chromaticity": GRAY_REFERENCE,
        "skin_reference_srgb": SKIN_REFERENCE_SRGB,
        "gray_sigma": GRAY_SIGMA, "skin_sigma": SKIN_SIGMA,
        "protection_strength": 1.,
        "note": "Skin reference and chromaticity radii are provisional comparison settings for user selection, not validated skin segmentation.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "index.html").write_text("\n".join(page))
    print(json.dumps({"output": str(args.out), "metric_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
        default=Path("/data/data1/semantic_isp/results/exp67_moderate_palette_q90_pm020/direct_soft_calibration.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=int, nargs="+", default=[2, 33, 384])
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
