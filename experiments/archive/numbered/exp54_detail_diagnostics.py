"""Diagnose the fixed detail operator and tuning/export scale without changing the ISP."""

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from differentiable_isp import _luminance, load_and_preprocess_raw
from exp38_learn_poles import encode_image, save_csv, save_json, save_render, srgb
from moran_semantic_isp import CONTROL_BOUNDS, CONTROL_NEUTRAL, parameters_from_controls, render_linear


def setting(image, config, amount):
    controls = image.new_tensor(CONTROL_NEUTRAL)
    controls[4] = amount
    return parameters_from_controls(controls, config)


def gaussian_rgb(image):
    positions = torch.arange(-3, 4, device=image.device, dtype=image.dtype)
    weights = torch.exp(-positions.square() / 8)
    weights /= weights.sum()
    kernel = (weights[:, None] * weights[None, :])[None, None].expand(3, 1, 7, 7)
    return F.conv2d(F.pad(image, (3, 3, 3, 3), mode="replicate"), kernel, groups=3)


def matched_view(image):
    pixels = image[0].permute(1, 2, 0).cpu().numpy()
    height, width = pixels.shape[:2]
    scale = 512 / min(height, width)
    pixels = cv2.resize(pixels, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    center_y, center_x = pixels.shape[0] // 2, pixels.shape[1] // 2
    pixels = pixels[center_y - 256:center_y + 256, center_x - 256:center_x + 256].copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(image.device)


def synthetic_checks(config, device):
    grayscale = torch.full((1, 3, 64, 64), .2, device=device)
    grayscale[:, :, :, 32:] = .6
    blurred = render_linear(grayscale, setting(grayscale, config, -.9))
    expected = .1 * grayscale + .9 * gaussian_rgb(grayscale)
    gray_error = float((blurred - expected).abs().max())
    assert gray_error < 2e-6
    color = grayscale.clone()
    left = color.new_tensor([.6, .15, .15])
    luma = float((left * color.new_tensor([.299, .587, .114])).sum())
    right = color.new_tensor([.15, (luma - .299 * .15 - .114 * .15) / .587, .15])
    color[:, :, :, :32] = left.view(1, 3, 1, 1)
    color[:, :, :, 32:] = right.view(1, 3, 1, 1)
    current = render_linear(color, setting(color, config, -.9))
    rgb_reference = .1 * color + .9 * gaussian_rgb(color)
    preserved_error = float((current - color).abs().max())
    rgb_edge_change = float((rgb_reference - color)[:, :, :, 29:35].abs().mean())
    assert preserved_error < 2e-6 and rgb_edge_change > .01
    positions = torch.arange(-3, 4, dtype=torch.float64)
    weights = torch.exp(-positions.square() / 8)
    weights /= weights.sum()
    result = {"kernel_size": 7, "configured_sigma_pixels": 2.,
        "effective_sigma_pixels": float((weights * positions.square()).sum().sqrt()),
        "grayscale_blend_max_error": gray_error,
        "isoluminant_edge_current_max_change": preserved_error,
        "isoluminant_edge_rgb_reference_mean_change": rgb_edge_change,
        "max_legal_sharpen_luma_increment": .8 * CONTROL_BOUNDS[4][1] * .0285,
        "interpretation": "Negative sharpness really blurs luminance, but preserves an equal-luminance color edge; it is not full RGB Gaussian blur."}
    return result


def comparison_sheet(folder, amounts):
    names = list(amounts)
    tile_width, tile_height = 300, 330
    sheet = Image.new("RGB", (tile_width * len(names), tile_height * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for row, mode in enumerate(("tuning_512", "native_matched", "native_100pct")):
        for column, name in enumerate(names):
            path = folder / f"{'tuning_512' if mode == 'tuning_512' else 'native'}_{name}.png"
            with Image.open(path) as image:
                if mode == "native_matched":
                    scale = 512 / min(image.size)
                    image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.BOX)
                    center_x, center_y = image.width // 2, image.height // 2
                    image = image.crop((center_x - 256, center_y - 256, center_x + 256, center_y + 256))
                elif mode == "native_100pct":
                    center_x, center_y = image.width // 2, image.height // 2
                    image = image.crop((center_x - 144, center_y - 144, center_x + 144, center_y + 144))
                image.thumbnail((288, 288), Image.Resampling.LANCZOS)
                left, top = column * tile_width, row * tile_height
                sheet.paste(image, (left + (tile_width - image.width) // 2, top + 36))
                draw.text((left + 5, top + 5), f"{mode}: {name}", fill="black")
                draw.text((left + 5, top + 19), f"amount {amounts[name]:+.4f}", fill="black")
    sheet.save(folder / "detail_comparison.jpg", quality=95)


@torch.no_grad()
def run(args):
    import open_clip

    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    torch.set_num_threads(2)
    config = json.loads(args.config.read_text())
    coefficient = config["learned_calibration"]["pole_raw_deltas"]
    raw_neutral = config["learned_calibration"]["raw_neutral"]
    from moran_semantic_isp import bounded_controls

    index = config["axes"].index("sharpness")
    positive = bounded_controls(torch.tensor(raw_neutral) + torch.tensor(coefficient[index]))[4].item()
    negative = bounded_controls(torch.tensor(raw_neutral) + torch.tensor(coefficient[index + 6]))[4].item()
    amounts = {"neutral": 0., "tuned_blur": negative, "max_blur": CONTROL_BOUNDS[4][0],
               "tuned_sharpen": positive, "max_sharpen": CONTROL_BOUNDS[4][1]}
    model, _, _ = open_clip.create_model_and_transforms(config["model"], pretrained=config["pretrained"],
        force_quick_gelu=True, device=args.device)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])
    targets = []
    for side in ("negative", "positive"):
        embeddings = model.encode_text(tokenizer(config["poles"]["sharpness"][side]["texts"]).to(args.device)).float()
        targets.append(F.normalize(embeddings.mean(dim=0), dim=0))
    targets = torch.stack(targets)
    args.out.mkdir(parents=True)
    checks = synthetic_checks(config, args.device)
    save_json(args.out / "operator_checks.json", checks)
    records, dimensions = [], []
    for image_id in args.images:
        native = load_and_preprocess_raw(f"/data/data1/EmotionISP_dataset/dng_filtered/{image_id}.dng", None).to(args.device)
        small = matched_view(native)
        folder = args.out / f"img_{image_id}"
        folder.mkdir()
        scale = min(native.shape[-2:]) / 512
        dimensions.append({"image": image_id, "native_height": native.shape[-2], "native_width": native.shape[-1],
            "native_to_tuning_scale": scale, "sigma_native_pixels": 2.,
            "sigma_native_in_tuning_pixels": 2 / scale, "sigma_tuning_in_clip_pixels": 2 * 224 / 512})
        for mode, source in (("tuning_512", small), ("native", native)):
            neutral_view = small if mode == "tuning_512" else matched_view(native)
            neutral_feature = encode_image(model, neutral_view)
            for name, amount in amounts.items():
                output = render_linear(source, setting(source, config, amount))
                assert torch.isfinite(output).all()
                save_render(folder / f"{mode}_{name}.png", output)
                view = output if mode == "tuning_512" else matched_view(output)
                feature = encode_image(model, view)
                scores = (feature @ targets.T)[0]
                mae = float((srgb(view) - srgb(neutral_view)).abs().mean() * 255)
                records.append({"image": image_id, "render_resolution": mode, "setting": name,
                    "sharpness": amount, "matched_view_mae_0_255": mae,
                    "clip_cosine_to_neutral": float((feature * neutral_feature).sum()),
                    "blur_centroid_cosine": float(scores[0]), "sharp_centroid_cosine": float(scores[1])})
        comparison_sheet(folder, amounts)
        print(f"diagnosed image {image_id}: native {native.shape[-1]}x{native.shape[-2]}, scale {scale:.3f}", flush=True)
    save_csv(args.out / "detail_metrics.csv", records)
    save_csv(args.out / "resolution_summary.csv", dimensions)
    save_json(args.out / "manifest.json", {"config": str(args.config), "images": args.images,
        "amounts": amounts, "checks": checks, "dimensions": dimensions,
        "scope": "Diagnostics only; existing ISP operators, bounds, descriptors and calibrations unchanged",
        "comparison": "Metrics use the same full linear frame area-resized and center-cropped in float32. Sheet rows: tuning 512, native render resized to the same field of view in sRGB, native 100-percent center crop."})
    page = ["<!doctype html><meta charset='utf-8'><title>Detail operator diagnostics</title>",
        "<h1>Blur and sharpening diagnostics</h1><p>No ISP changes or retraining. "
        "Top row: 512-pixel tuning render. Middle: native render resized to the same field of view in sRGB. "
        "Bottom: native 288-pixel center crops at 100%, with a tighter field of view. "
        "The CSV uses content-matched float32 linear resampling before the sRGB transfer.</p>",
        "<a href='detail_metrics.csv'>Metrics</a> | <a href='resolution_summary.csv'>Resolutions</a>",
        "<pre>" + html.escape(json.dumps(checks, indent=2)) + "</pre>"]
    for image_id in args.images:
        page.append(f"<h2>Image {image_id}</h2><a href='img_{image_id}/detail_comparison.jpg'>"
                    f"<img style='max-width:100%' src='img_{image_id}/detail_comparison.jpg'></a>")
    (args.out / "index.html").write_text("\n".join(page))
    print(json.dumps({"output": str(args.out), "checks": checks, "dimensions": dimensions}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/data/data1/semantic_isp/results/exp53_strong_words_one_dof_q90/direct_soft_calibration.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=int, nargs="+", default=[2, 33, 384])
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())