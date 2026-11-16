"""Sweep the physical palette coordinate with all other ISP controls neutral."""

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from differentiable_isp import load_and_preprocess_raw
from exp38_learn_poles import encode_image, save_csv, save_json, save_render, srgb
from moran_semantic_isp import CONTROL_NEUTRAL, CONTROL_NAMES, parameters_from_controls, render_linear


def slug(value):
    return "z0.00" if abs(value) < 1e-8 else f"{'p' if value > 0 else 'm'}{abs(value):.2f}"


def contact_sheet(folder, amounts):
    columns, tile_w, tile_h, header = 5, 280, 270, 34
    rows = (len(amounts) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + header)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    for index, amount in enumerate(amounts):
        left, top = (index % columns) * tile_w, (index // columns) * (tile_h + header)
        draw.text((left + 8, top + 8), f"Palette coordinate {amount:+.2f}", fill="black", font=font)
        with Image.open(folder / f"palette_{slug(amount)}.png") as image:
            image.thumbnail((tile_w - 16, tile_h - 12), Image.Resampling.LANCZOS)
            sheet.paste(image, (left + (tile_w - image.width) // 2,
                                top + header + (tile_h - image.height) // 2))
    sheet.save(folder / "palette_sweep.jpg", quality=94)


def run(args):
    import open_clip

    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    config = json.loads(args.config.read_text())
    if "palette_nearest_swatch" not in config or config.get("palette_split_tone", {}).get("method") != "nearest_swatch":
        raise ValueError("Sweep requires a calibrated nearest-swatch palette config")
    if args.steps < 3 or args.steps % 2 == 0:
        raise ValueError("Use an odd number of at least three sweep steps so neutral is included")
    limit = float(config["coordinate_limit"])
    amounts = np.linspace(-limit, limit, args.steps).tolist()
    index = CONTROL_NAMES.index("palette")
    bounds = config.get("control_bounds", config["learned_calibration"]["control_bounds"])
    if amounts[0] < bounds[index][0] or amounts[-1] > bounds[index][1]:
        raise ValueError("Palette sweep exceeds the calibrated physical control bounds")
    torch.set_num_threads(2)
    model, _, _ = open_clip.create_model_and_transforms(config["model"], pretrained=config["pretrained"],
        force_quick_gelu=True, device=args.device)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])
    with torch.no_grad():
        targets = []
        for side in ("negative", "positive"):
            text = config["poles"]["palette"][side]["texts"]
            features = model.encode_text(tokenizer(text).to(args.device)).float()
            targets.append(F.normalize(features.mean(dim=0), dim=0))
        targets = torch.stack(targets)
    rgb_to_yiq = torch.tensor(config["palette_projection"]["rgb_to_yiq"], device=args.device)
    swatches = config["palette_projection"]["swatches"]
    palette_pairs = torch.tensor([swatches["retro"], swatches["blockbuster"]], device=args.device)
    endpoint_chroma = torch.einsum("ij,pkj->pki", rgb_to_yiq, palette_pairs)[:, :, 1:3]
    args.out.mkdir(parents=True)
    rows = []
    page = ["<!doctype html><meta charset='utf-8'><title>Palette coordinate sweep</title>",
        "<h1>Nearest-swatch palette sweep</h1>",
        "<p>Only the physical palette coordinate varies. All other ISP controls are neutral. "
        "Contact sheets show native renders. CLIP scores use the same 512-pixel center crop and "
        "224-pixel encoder resize at every step.</p>",
        "<p><a href='palette_sweep.csv'>Metrics CSV</a> | <a href='summary.json'>Summary</a></p>"]
    for image_id in args.images:
        native = load_and_preprocess_raw(
            f"/data/data1/EmotionISP_dataset/dng_filtered/{image_id}.dng", None).to(args.device)
        crop = load_and_preprocess_raw(
            f"/data/data1/EmotionISP_dataset/dng_filtered/{image_id}.dng", 512).to(args.device)
        folder = args.out / f"img_{image_id}"
        folder.mkdir()
        neutral_controls = native.new_tensor(CONTROL_NEUTRAL)
        neutral = render_linear(native, parameters_from_controls(neutral_controls, config))
        neutral_feature = encode_image(model, render_linear(crop,
            parameters_from_controls(crop.new_tensor(CONTROL_NEUTRAL), config)))
        for amount in amounts:
            controls_native = native.new_tensor(CONTROL_NEUTRAL)
            controls_native[index] = amount
            native_output = render_linear(native, parameters_from_controls(controls_native, config))
            filename = f"palette_{slug(amount)}.png"
            save_render(folder / filename, native_output)
            controls_crop = crop.new_tensor(CONTROL_NEUTRAL)
            controls_crop[index] = amount
            crop_output = render_linear(crop, parameters_from_controls(controls_crop, config))
            image_feature = encode_image(model, crop_output)
            cosine = (image_feature @ targets.T)[0]
            yiq = torch.einsum("ij,bjhw->bihw", rgb_to_yiq, native_output)
            chroma = yiq[:, 1:3].permute(0, 2, 3, 1)
            palette_index = 1 if amount >= 0 else 0
            selected_endpoints = endpoint_chroma[palette_index]
            distances = torch.linalg.vector_norm(chroma.unsqueeze(-2) - selected_endpoints.view(1, 1, 1, 2, 2), dim=-1)
            softness = (config["palette_nearest_swatch"]["softness_ratio"]
                        * torch.linalg.vector_norm(selected_endpoints[0] - selected_endpoints[1]))
            assignments = torch.softmax(-distances.square() / (2 * softness.square()), dim=-1)
            pair_names = ("retro", "blockbuster")[palette_index]
            pixel_change = float((srgb(native_output) - srgb(neutral)).abs().mean() * 255)
            rows.append({"image": image_id, "palette_coordinate": amount,
                "negative_palette_phrase_cosine": float(cosine[0]),
                "positive_palette_phrase_cosine": float(cosine[1]),
                "native_rgb_mae_from_neutral_0_255": pixel_change,
                "mean_distance_to_nearest_swatch_yiq": float(distances.min(dim=-1).values.mean()),
                "endpoint_0_name": pair_names + ":0", "endpoint_0_assignment_fraction": float(assignments[..., 0].mean()),
                "endpoint_1_name": pair_names + ":1", "endpoint_1_assignment_fraction": float(assignments[..., 1].mean())})
        contact_sheet(folder, amounts)
        page.append(f"<h2>Scene {image_id}</h2><a href='img_{image_id}/palette_sweep.jpg'>"
                    f"<img style='max-width:100%' src='img_{image_id}/palette_sweep.jpg'></a>")
        print(f"swept image {image_id}: {len(amounts)} palette values", flush=True)
    save_csv(args.out / "palette_sweep.csv", rows)
    palette_settings = {}
    for row in csv.DictReader((args.config.parent / "palette_average_settings.csv").open()):
        palette_settings[row["sign"]] = float(row["selected_endpoint"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for image_id in args.images:
        subset = [row for row in rows if row["image"] == image_id]
        x = [row["palette_coordinate"] for row in subset]
        axes[0].plot(x, [row["positive_palette_phrase_cosine"] for row in subset], marker="o", label=f"Scene {image_id}")
        axes[1].plot(x, [row["negative_palette_phrase_cosine"] for row in subset], marker="o", label=f"Scene {image_id}")
        axes[2].plot(x, [row["native_rgb_mae_from_neutral_0_255"] for row in subset], marker="o", label=f"Scene {image_id}")
    axes[0].set_title("CLIP cosine to teal-orange phrases")
    axes[1].set_title("CLIP cosine to magenta-green phrases")
    axes[2].set_title("Native RGB change from neutral")
    for ax in axes:
        ax.axvline(0, color="#555", linewidth=.8)
        ax.grid(True, alpha=.25)
        ax.set_xlabel("Palette coordinate")
    axes[0].axvline(palette_settings["positive"], color="#c54b39", linestyle="--", label="Mean-tuned +")
    axes[1].axvline(palette_settings["negative"], color="#2d8b63", linestyle="--", label="Mean-tuned -")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "palette_sweep_metrics.png", dpi=150)
    plt.close(fig)
    summary = {"config": str(args.config), "images": args.images, "steps": args.steps,
        "palette_coordinate_range": [-limit, limit], "palette_mix": config["palette_split_tone"],
        "nearest_swatch": config["palette_nearest_swatch"], "mean_tuned_settings": palette_settings,
        "other_controls": "fixed at neutral", "metric_note": "CLIP is the calibration model; curves are self-evaluation, not human preference"}
    save_json(args.out / "summary.json", summary)
    page.insert(2, "<p><a href='palette_sweep_metrics.png'>Sweep curves</a></p>")
    (args.out / "index.html").write_text("\n".join(page))
    print(json.dumps({"output": str(args.out), "settings": palette_settings, "steps": len(amounts)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/data/data1/semantic_isp/results/exp64_moderate_palette_mean/direct_soft_calibration.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=int, nargs="+", default=[2, 33, 384])
    parser.add_argument("--steps", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())