"""Fit the six-control ISP to MIT-Adobe FiveK photographer renderings."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont
import tifffile
import torch
import torch.nn.functional as F

EXPERIMENT_DIR = Path(__file__).resolve().parent
SEMANTIC_ISP_DIR = EXPERIMENT_DIR.parent / "semantic_isp"
if str(SEMANTIC_ISP_DIR) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_ISP_DIR))

from moran_semantic_isp import (CONTROL_NAMES, linear_to_srgb, parameters_from_controls,
                                render_linear)
from pole_normalization import normalized_to_physical, physical_to_normalized, pole_endpoints
from fivek_preview import load_linear_png, write_render


DEFAULT_DATA = Path("/data/data1/FiveK/C_data")
DEFAULT_FIVEK = Path("/data/data1/FiveK")
DEFAULT_CONFIG = Path("/data/data1/semantic_isp/results/exp81_joint_descriptor_tail_final/direct_soft_calibration.json")
DEFAULT_OUT = DEFAULT_DATA / "results/isp_match_v1"
IMAGE_IDS = ("a0015-DSC_0081", "a0017-050710_031618__MG_3496",
             "a0865-20080515_075226__MG_9983", "a2456-IMG_1681", "a3768-IMG_1641")
PHOTOGRAPHERS = ("A", "B", "C", "D", "E")
P90_NAMES = ("warmth", "saturation", "tone_gain", "contrast", "sharpness", "palette")

PROPHOTO_TO_XYZ_D50 = np.array([
    [0.7976749, 0.1351917, 0.0313534],
    [0.2880402, 0.7118741, 0.0000857],
    [0.0, 0.0, 0.8252100],
], dtype=np.float64)
D50_TO_D65 = np.array([
    [0.9555766, -0.0230393, 0.0631636],
    [-0.0282895, 1.0099416, 0.0210077],
    [0.0122982, -0.0204830, 1.3299098],
], dtype=np.float64)
XYZ_D65_TO_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
], dtype=np.float64)
PROPHOTO_TO_LINEAR_SRGB = XYZ_D65_TO_SRGB @ D50_TO_D65 @ PROPHOTO_TO_XYZ_D50


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def linear_to_display(linear):
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * np.maximum(linear, 0.0031308) ** (1 / 2.4) - 0.055)


def prophoto_tiff_to_linear_srgb(path, verify_icc=True):
    path = Path(path)
    with tifffile.TiffFile(path) as source:
        page = source.pages[0]
        profile_tag = page.tags.get("InterColorProfile")
        if profile_tag is None:
            raise ValueError(f"Missing embedded ICC profile: {path}")
        profile_bytes = profile_tag.value
        encoded = page.asarray()
    if encoded.dtype != np.uint16 or encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError(f"Expected a 16-bit RGB TIFF: {path}")
    profile = ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes))
    description = ImageCms.getProfileDescription(profile).strip()
    if "ProPhoto RGB" not in description:
        raise ValueError(f"Expected ProPhoto RGB, got {description!r}: {path}")
    encoded_float = encoded.astype(np.float64) / 65535.0
    prophoto_linear = np.where(encoded_float <= 16 / 512,
                               encoded_float / 16.0,
                               np.maximum(encoded_float, 0.0) ** 1.8)
    srgb_linear_unclipped = np.einsum("ij,hwj->hwi", PROPHOTO_TO_LINEAR_SRGB, prophoto_linear)
    out_of_gamut_fraction = float(np.mean((srgb_linear_unclipped < 0) | (srgb_linear_unclipped > 1)))
    srgb_linear = np.clip(srgb_linear_unclipped, 0.0, 1.0).astype(np.float32)
    validation = None
    if verify_icc:
        sample = np.rint(encoded_float[::16, ::16] * 255).astype(np.uint8)
        source_8bit = Image.fromarray(sample, "RGB")
        srgb_profile = ImageCms.createProfile("sRGB")
        reference = np.asarray(ImageCms.profileToProfile(
            source_8bit, profile, srgb_profile, outputMode="RGB", renderingIntent=1), dtype=np.int16)
        manual_encoded = np.rint(np.clip(linear_to_display(srgb_linear[::16, ::16]), 0, 1) * 255).astype(np.int16)
        difference = np.abs(reference - manual_encoded)
        validation = {"mean_abs_8bit": float(difference.mean()),
                      "p95_abs_8bit": float(np.percentile(difference, 95)),
                      "max_abs_8bit": int(difference.max())}
        if validation["mean_abs_8bit"] > 2.0 or validation["p95_abs_8bit"] > 3.0:
            raise ValueError(f"ProPhoto transform does not match ICC conversion: {validation}")
    return srgb_linear, {"profile": description, "shape": list(encoded.shape),
                         "dtype": str(encoded.dtype), "out_of_gamut_fraction": out_of_gamut_fraction,
                         "icc_check": validation}


def load_neutral(path, device):
    image = load_linear_png(path).to(device=device, dtype=torch.float32)
    return image.clamp(0, 1)


def resize_target(target, dimensions, device):
    tensor = torch.from_numpy(np.ascontiguousarray(target)).permute(2, 0, 1).unsqueeze(0).to(device)
    original_ratio = tensor.shape[-1] / tensor.shape[-2]
    target_ratio = dimensions[1] / dimensions[0]
    if abs(original_ratio - target_ratio) > 0.002:
        raise ValueError(f"Neutral/photographer aspect ratios differ: {original_ratio} vs {target_ratio}")
    return F.interpolate(tensor, size=dimensions, mode="area").clamp(0, 1)


def image_statistics(image):
    display = linear_to_srgb(image.clamp(0, 1))
    luma = (display[:, 0:1] * 0.2126 + display[:, 1:2] * 0.7152 + display[:, 2:3] * 0.0722)
    maximum = display.amax(dim=1, keepdim=True)
    minimum = display.amin(dim=1, keepdim=True)
    saturation = (maximum - minimum) / maximum.clamp_min(1e-5)
    return {"mean_display_rgb": display.mean(dim=(0, 2, 3)).detach().cpu().tolist(),
            "mean_display_luma": float(luma.mean()), "mean_display_saturation": float(saturation.mean())}


def objective(rendered, target, normalized):
    rendered_display = linear_to_srgb(rendered.clamp(0, 1))
    target_display = linear_to_srgb(target.clamp(0, 1))
    pixel = F.smooth_l1_loss(rendered_display, target_display, beta=0.025)
    low_rendered = F.avg_pool2d(rendered_display, 5, 5)
    low_target = F.avg_pool2d(target_display, 5, 5)
    low_frequency = F.smooth_l1_loss(low_rendered, low_target, beta=0.025)
    regularization = normalized.square().mean()
    total = pixel + 0.25 * low_frequency + 0.002 * regularization
    return total, {"pixel": float(pixel.detach()), "low_frequency": float(low_frequency.detach()),
                   "regularization": float(regularization.detach())}


def fit_controls(neutral, target, config, endpoints, iterations, learning_rate):
    fit_image = F.interpolate(neutral, size=(256, round(256 * neutral.shape[-1] / neutral.shape[-2])), mode="area")
    fit_target = F.interpolate(target, size=fit_image.shape[-2:], mode="area")
    palette_index = CONTROL_NAMES.index("palette")
    best = None
    restarts = []
    for palette_seed in (0.0, -0.12, 0.12):
        raw_start = fit_image.new_zeros(len(CONTROL_NAMES))
        raw_start[palette_index] = palette_seed
        raw = torch.nn.Parameter(raw_start)
        optimizer = torch.optim.Adam([raw], lr=learning_rate)
        local_best = float("inf")
        local_result = None
        for iteration in range(iterations + 1):
            normalized = torch.tanh(raw)
            physical = normalized_to_physical(normalized, endpoints)
            rendered = render_linear(fit_image, parameters_from_controls(physical, config))
            loss, terms = objective(rendered, fit_target, normalized)
            value = float(loss.detach())
            if value < local_best:
                local_best = value
                local_result = {"loss": value, "iteration": iteration,
                    "normalized_controls": dict(zip(CONTROL_NAMES, normalized.detach().cpu().tolist())),
                    "physical_controls": dict(zip(CONTROL_NAMES, physical.detach().cpu().tolist())),
                    "loss_terms": terms}
            if best is None or value < best["loss"]:
                best = {**local_result, "palette_seed": palette_seed}
            if iteration < iterations:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        restarts.append({"palette_seed": palette_seed, "best_loss": local_best,
                         "best_iteration": local_result["iteration"]})
    best["restarts"] = restarts
    best["parameterization"] = "P90 normalized [-1,1] mapped to calibrated physical endpoints"
    return best


def phrase_from_controls(values):
    descriptors = []
    for name, positive, negative, threshold in (
            ("warmth", "warm", "cool", 0.13),
            ("saturation", "vivid", "muted", 0.18),
            ("tone_gain", "bright", "dark", 0.15),
            ("contrast", "high-contrast", "soft-contrast", 0.18),
            ("sharpness", "crisp", "soft", 0.2),
            ("palette", "teal-and-orange", "magenta-and-green", 0.2)):
        value = values[name]
        if name == "tone_gain":
            value -= 0.0
        if value >= threshold:
            descriptors.append((abs(value), positive))
        elif value <= -threshold:
            descriptors.append((abs(value), negative))
    selected = [label for _, label in sorted(descriptors, reverse=True)[:3]]
    if not selected:
        return "Keep the colors natural and the tones balanced."
    if len(selected) == 1:
        return f"Give the image a {selected[0]} look."
    return "Make it " + ", ".join(selected[:-1]) + " and " + selected[-1] + "."


def descriptor_bundle(values):
    settings = []
    definitions = (
        ("warmth", "warm", "cool", "white balance"),
        ("saturation", "vivid", "muted", "color saturation"),
        ("tone_gain", "bright", "dark", "tone gain"),
        ("contrast", "high-contrast", "soft-contrast", "contrast"),
        ("sharpness", "crisp", "soft", "sharpness"),
        ("palette", "teal-and-orange", "magenta-and-green", "palette"),
    )
    for name, positive, negative, technical_name in definitions:
        value = float(values[name])
        threshold = 0.12 if name in ("warmth", "tone_gain") else 0.16
        if abs(value) < threshold:
            continue
        direction = positive if value > 0 else negative
        strength = "slightly" if abs(value) < 0.30 else "noticeably" if abs(value) < 0.65 else "strongly"
        settings.append({"control": name, "technical_name": technical_name,
                         "direction": direction, "strength": strength, "p90_value": value})
    settings.sort(key=lambda row: abs(row["p90_value"]), reverse=True)
    selected = settings[:4]
    if not selected:
        direct = "Keep the colors natural and the tones balanced."
        perceptual = "Give the image a natural, balanced look."
        technical = "Keep the white balance, tone, color, contrast, and detail near neutral."
        association = "Give it a clean, natural photographic feel."
    else:
        directions = [row["direction"] for row in selected]
        direct_actions = []
        technical_actions = []
        for row in selected:
            if row["control"] == "warmth":
                direct_actions.append(f"shift the white balance {row['strength']} {row['direction']}")
                technical_actions.append(f"move white balance {row['strength']} toward {row['direction']}")
            elif row["control"] == "saturation":
                direct_actions.append(f"make the colors {row['strength']} more {row['direction']}")
                technical_actions.append(f"adjust saturation {row['strength']} toward {row['direction']}")
            elif row["control"] == "tone_gain":
                direct_actions.append(f"make the image {row['strength']} {row['direction']}")
                technical_actions.append(f"move tone gain {row['strength']} toward {row['direction']}")
            elif row["control"] == "contrast":
                direct_actions.append(f"make contrast {row['strength']} more {row['direction']}")
                technical_actions.append(f"adjust contrast {row['strength']} toward {row['direction']}")
            elif row["control"] == "sharpness":
                direct_actions.append(f"make detail {row['strength']} more {row['direction']}")
                technical_actions.append(f"adjust detail {row['strength']} toward {row['direction']}")
            else:
                direct_actions.append(f"shift the palette {row['strength']} toward {row['direction']}")
                technical_actions.append(f"move palette control {row['strength']} toward {row['direction']}")
        direct = "Adjust the image to " + "; ".join(direct_actions) + "."
        perceptual = "Give the image a " + ", ".join(directions[:3]) + " look."
        technical = "For the edit, " + "; ".join(technical_actions[:3]) + "."
        if "warm" in directions and "bright" in directions:
            style = "sunlit golden-hour"
        elif "cool" in directions and "vivid" in directions:
            style = "cool, vivid editorial"
        elif "soft-contrast" in directions and "bright" in directions:
            style = "soft, luminous"
        elif "high-contrast" in directions and "muted" in directions:
            style = "dramatic film"
        elif "vivid" in directions and "bright" in directions:
            style = "bright, vivid editorial"
        elif "warm" in directions and "vivid" in directions:
            style = "rich, warm film"
        else:
            style = ", ".join(directions[:2]) + " photographic"
        association = f"Give the scene a {style} mood."
    base = {"D1": ("direct technical/visual", direct), "D2": ("perceptual", perceptual),
            "D3": ("technical/visual", technical), "D4": ("cultural/associative", association)}
    prefixes = {
        "D1": ("", "For the whole image: ", "Globally, ", "For the photograph: "),
        "D2": ("", "Visually, ", "For the overall appearance: ", "In the final image: "),
        "D3": ("", "Globally: ", "For the overall rendering: ", "For the full frame: "),
        "D4": ("", "As a photographic style: ", "For the overall mood: ", "In the final rendering: "),
    }
    output = {}
    for group, (kind, text) in base.items():
        variants = [text if not prefix else prefix + text[0].lower() + text[1:]
                    for prefix in prefixes[group]]
        output[group] = {"type": kind, "text": text, "variants": variants}
    return {
        **output,
        "source": "Rule-based paraphrases derived from the fitted P90 controls; not photographer-authored captions.",
        "active_control_directions": selected,
    }


def font(size):
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default()


def paste_thumbnail(sheet, image_path, box):
    left, top, width, height = box
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        sheet.paste(image, (left + (width - image.width) // 2, top + (height - image.height) // 2))


def make_reference_sheet(path, records):
    ids = list(records)
    photographers = list(records[ids[0]]["targets"])
    tile_width, image_height, header_height = 320, 220, 54
    sheet = Image.new("RGB", ((1 + len(photographers)) * tile_width,
                               header_height + len(ids) * (image_height + 36)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "FiveK: default ISP versus five professional edits", fill="#18252e", font=font(22))
    labels = ("Default ISP", *(f"Expert {name}" for name in photographers))
    for column, label in enumerate(labels):
        draw.text((column * tile_width + 10, 34), label, fill="#34444c", font=font(15))
    for row_index, image_id in enumerate(ids):
        row_top = header_height + row_index * (image_height + 36)
        draw.text((10, row_top + 4), image_id, fill="#34444c", font=font(14))
        paste_thumbnail(sheet, records[image_id]["neutral_display"], (0, row_top + 26, tile_width, image_height))
        for column, photographer in enumerate(photographers, 1):
            paste_thumbnail(sheet, records[image_id]["targets"][photographer]["display_path"],
                            (column * tile_width, row_top + 26, tile_width, image_height))
    sheet.save(path)


def make_fit_sheet(path, image_id, record):
    photographers = list(record["targets"])
    tile_width, image_height, header_height, row_gap = 360, 165, 122, 34
    sheet = Image.new("RGB", (3 * tile_width, header_height + len(photographers) * (image_height + row_gap)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"{image_id}: expert TIFF versus best six-control ISP approximation",
              fill="#18252e", font=font(20))
    for column, label in enumerate(("Expert TIFF", "Default ISP", "ISP fit")):
        draw.text((column * tile_width + 8, 92), label, fill="#34444c", font=font(14))
    for row_index, photographer in enumerate(photographers):
        top = header_height + row_index * (image_height + row_gap)
        fit = record["targets"][photographer]["fit"]
        controls = fit["normalized_controls"]
        descriptors = fit["descriptors"]
        draw.text((8, top), f"Expert {photographer} | {descriptors['D2']['text']}", fill="#18252e", font=font(13))
        draw.text((8, top + 17), f"P90: W {controls['warmth']:+.2f}, S {controls['saturation']:+.2f}, "
                  f"T {controls['tone_gain']:+.2f}, C {controls['contrast']:+.2f}, "
                  f"Sh {controls['sharpness']:+.2f}, P {controls['palette']:+.2f}", fill="#34444c", font=font(11))
        paste_thumbnail(sheet, record["targets"][photographer]["display_path"], (4, top + 34, tile_width - 8, image_height))
        paste_thumbnail(sheet, record["neutral_display"], (tile_width + 4, top + 34, tile_width - 8, image_height))
        paste_thumbnail(sheet, record["targets"][photographer]["fit_display_path"], (2 * tile_width + 4, top + 34, tile_width - 8, image_height))
    sheet.save(path)


def main(args):
    image_ids = tuple(args.image_ids)
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("Image IDs must be nonempty and unique")
    if not set(args.photographers) <= set(PHOTOGRAPHERS) or not args.photographers:
        raise ValueError("Photographers must be selected from A-E")
    if args.iterations < 1 or args.learning_rate <= 0:
        raise ValueError("Invalid optimization settings")
    data_root, fivek_root = args.data_root, args.fivek_root
    config = json.loads(args.config.read_text())
    config["image_mean"], config["image_std"] = [0.5] * 3, [0.5] * 3
    endpoints = pole_endpoints(config)
    source_info = {}
    for image_id in image_ids:
        neutral_path = fivek_root / "default_isp_linear_720p" / f"{image_id}.png"
        if not neutral_path.is_file():
            raise FileNotFoundError(neutral_path)
        raw_paths = list((data_root / "raw").glob(image_id + ".dng"))
        if len(raw_paths) != 1:
            raise ValueError(f"Expected one downloaded DNG for {image_id}")
        source_info[image_id] = {"neutral_path": neutral_path, "raw_path": raw_paths[0]}
        for photographer in args.photographers:
            tiff = data_root / "photographers" / photographer / f"{image_id}.tif"
            if not tiff.is_file():
                raise FileNotFoundError(tiff)
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.out}; pass --overwrite to replace this run")
    if args.overwrite:
        for child in args.out.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                import shutil
                shutil.rmtree(child)
    render_dir, target_dir, figure_dir = (args.out / "renders", args.out / "targets", args.out / "figures")
    for directory in (render_dir, target_dir, figure_dir):
        directory.mkdir(exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    records = {}
    for image_id in image_ids:
        neutral = load_neutral(source_info[image_id]["neutral_path"], device)
        target_records = {}
        with torch.no_grad():
            neutral_display_path = target_dir / f"{image_id}_neutral.png"
            write_render(neutral_display_path, neutral)
        baseline_stats = image_statistics(neutral)
        for photographer in args.photographers:
            tiff_path = data_root / "photographers" / photographer / f"{image_id}.tif"
            target_array, profile_info = prophoto_tiff_to_linear_srgb(tiff_path)
            target = resize_target(target_array, neutral.shape[-2:], device)
            fit = fit_controls(neutral, target, config, endpoints, args.iterations, args.learning_rate)
            controls = torch.tensor([fit["physical_controls"][name] for name in CONTROL_NAMES],
                                    device=device, dtype=neutral.dtype)
            full_resolution_fit = render_linear(neutral, parameters_from_controls(controls, config))
            target_display_path = target_dir / f"{image_id}_expert_{photographer}.png"
            fitted_display_path = render_dir / f"{image_id}_expert_{photographer}_isp_fit.png"
            write_render(target_display_path, target)
            write_render(fitted_display_path, full_resolution_fit)
            fit["image_id"] = image_id
            fit["photographer"] = photographer
            fit["agent_prompt"] = phrase_from_controls(fit["normalized_controls"])
            fit["descriptors"] = descriptor_bundle(fit["normalized_controls"])
            fit["expert_original_dimensions"] = profile_info["shape"][:2]
            fit["comparison_dimensions"] = [int(neutral.shape[-2]), int(neutral.shape[-1])]
            fit["source_tiff"] = str(tiff_path)
            fit["source_tiff_sha256"] = sha256(tiff_path)
            fit["raw_source"] = str(source_info[image_id]["raw_path"])
            fit["raw_source_sha256"] = sha256(source_info[image_id]["raw_path"])
            fit["neutral_source"] = str(source_info[image_id]["neutral_path"])
            fit["neutral_source_sha256"] = sha256(source_info[image_id]["neutral_path"])
            fit["color_profile"] = profile_info
            fit["neutral_statistics"] = baseline_stats
            fit["default_isp_statistics"] = baseline_stats
            expert_stats = image_statistics(target)
            fitted_stats = image_statistics(full_resolution_fit)
            fit["expert_statistics"] = expert_stats
            fit["fit_statistics"] = fitted_stats
            fit["expert_minus_neutral"] = {
                "mean_display_rgb": [expert_stats["mean_display_rgb"][index] - baseline_stats["mean_display_rgb"][index]
                                     for index in range(3)],
                "mean_display_luma": expert_stats["mean_display_luma"] - baseline_stats["mean_display_luma"],
                "mean_display_saturation": expert_stats["mean_display_saturation"] - baseline_stats["mean_display_saturation"]}
            fit["neutral_target_mae_display"] = float(F.l1_loss(
                linear_to_srgb(neutral), linear_to_srgb(target)).detach())
            fit["default_isp_target_mae_display"] = fit["neutral_target_mae_display"]
            fit["fit_target_mae_display"] = float(F.l1_loss(
                linear_to_srgb(full_resolution_fit), linear_to_srgb(target)).detach())
            fit["relative_mae_reduction_vs_neutral"] = float(
                (fit["neutral_target_mae_display"] - fit["fit_target_mae_display"])
                / max(fit["neutral_target_mae_display"], 1e-8))
            fit["relative_mae_reduction_vs_default_isp"] = fit["relative_mae_reduction_vs_neutral"]
            fit["neutral_display_path"] = str(neutral_display_path)
            fit["display_path"] = str(target_display_path)
            fit["fit_display_path"] = str(fitted_display_path)
            target_records[photographer] = {"display_path": target_display_path,
                "fit_display_path": fitted_display_path, "fit": fit}
            print(f"{image_id} expert {photographer}: normalized={fit['normalized_controls']} "
                  f"fit_MAE={fit['fit_target_mae_display']:.4f} ", flush=True)
            del target, full_resolution_fit
            if device.type == "cuda":
                torch.cuda.empty_cache()
        records[image_id] = {"neutral_display": neutral_display_path, "targets": target_records}
    aggregate = {}
    for photographer in args.photographers:
        normalized = {name: float(np.median([
            records[image_id]["targets"][photographer]["fit"]["normalized_controls"][name]
            for image_id in image_ids])) for name in CONTROL_NAMES}
        physical = {name: float(np.median([
            records[image_id]["targets"][photographer]["fit"]["physical_controls"][name]
            for image_id in image_ids])) for name in CONTROL_NAMES}
        aggregate[photographer] = {"normalized_controls": normalized,
            "physical_controls": physical, "suggested_prompt": phrase_from_controls(normalized),
            "images": list(image_ids),
            "note": "Median of per-image best six-control pixel fits; not an original photographer preset or a measured camera setting."}
    for image_id in image_ids:
        make_fit_sheet(figure_dir / f"{image_id}_isp_fit_comparison.png", image_id, records[image_id])
    make_reference_sheet(figure_dir / "expert_retouch_contact_sheet.png", records)
    summary = {"dataset": f"MIT-Adobe FiveK, {len(image_ids)} selected images from the MIT-licensed subset",
        "license_file": str(data_root / "LicenseAdobeMIT.txt"),
        "images": list(image_ids), "photographers": list(args.photographers),
        "default_isp_source": "Existing 16-bit linear RGB default-ISP renders, 720-pixel short side; downloaded DNGs are retained for future reprocessing.",
        "photographer_source": "Original 16-bit ProPhoto RGB TIFF masters, converted to linear sRGB with an ICC-validated matrix transform.",
        "calibration_config": str(args.config), "calibration_config_sha256": sha256(args.config),
        "fit_objective": "sRGB-display smooth-L1 pixel loss + 0.25 low-frequency loss + 0.002 normalized-control L2 regularization; best of three palette starts.",
        "optimization": {"iterations": args.iterations, "learning_rate": args.learning_rate,
                         "fit_short_side": 256, "device": str(device)},
        "descriptor_types": {"D1": "direct technical/visual", "D2": "perceptual",
                     "D3": "technical/visual", "D4": "cultural/associative"},
        "limits": ["A six-global-control ISP cannot reproduce local masks, retouching, crop, or nonlinear custom curves.",
                   "The fitted values are best approximations under this objective, not the photographers' actual Lightroom or Photoshop settings.",
               "The generated D1-D4 phrases describe the fitted global effect and are not original photographer instructions.",
               "Expert TIFF comparison previews and fitted ISP renders are saved at the same Default-ISP dimensions; full-resolution TIFF masters remain under photographers/."],
        "photographer_profiles": aggregate,
        "per_image": {image_id: {photographer: records[image_id]["targets"][photographer]["fit"]
            for photographer in args.photographers} for image_id in image_ids},
        "figures": [str(figure_dir / "expert_retouch_contact_sheet.png"),
                    *[str(figure_dir / f"{image_id}_isp_fit_comparison.png") for image_id in image_ids]]}
    result_path = args.out / "results.json"
    result_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    fields = ["image_id", "photographer", *[f"{name}_p90" for name in CONTROL_NAMES],
              *[f"{name}_physical" for name in CONTROL_NAMES], "suggested_prompt",
              "default_isp_target_mae_display", "neutral_target_mae_display", "fit_target_mae_display"]
    with (args.out / "controls.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image_id in image_ids:
            for photographer in args.photographers:
                fit = records[image_id]["targets"][photographer]["fit"]
                writer.writerow({"image_id": image_id, "photographer": photographer,
                    **{f"{name}_p90": fit["normalized_controls"][name] for name in CONTROL_NAMES},
                    **{f"{name}_physical": fit["physical_controls"][name] for name in CONTROL_NAMES},
                    "suggested_prompt": fit["agent_prompt"],
                    "default_isp_target_mae_display": fit["default_isp_target_mae_display"],
                    "neutral_target_mae_display": fit["neutral_target_mae_display"],
                    "fit_target_mae_display": fit["fit_target_mae_display"]})
    descriptor_fields = ["image_id", "photographer", *[f"{group}_type" for group in ("D1", "D2", "D3", "D4")],
                         *[f"{group}_{index}" for group in ("D1", "D2", "D3", "D4") for index in range(1, 5)],
                         "p90_controls_json"]
    with (args.out / "expert_descriptors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=descriptor_fields)
        writer.writeheader()
        for image_id in image_ids:
            for photographer in args.photographers:
                fit = records[image_id]["targets"][photographer]["fit"]
                writer.writerow({"image_id": image_id, "photographer": photographer,
                          **{f"{group}_{index}": text for group in ("D1", "D2", "D3", "D4")
                              for index, text in enumerate(fit["descriptors"][group]["variants"], 1)},
                    **{f"{key}_type": fit["descriptors"][key]["type"] for key in ("D1", "D2", "D3", "D4")},
                    "p90_controls_json": json.dumps(fit["normalized_controls"], sort_keys=True)})
    print(json.dumps({"output": str(args.out), "results": str(result_path),
                      "figures": summary["figures"], "photographer_profiles": aggregate}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fivek-root", type=Path, default=DEFAULT_FIVEK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--image-ids", nargs="+", default=list(IMAGE_IDS))
    parser.add_argument("--photographers", nargs="+", choices=PHOTOGRAPHERS, default=list(PHOTOGRAPHERS))
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())