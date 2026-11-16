"""Moran deck: text pole centroids -> axis coordinates -> composed linear ISP."""

import argparse
import copy
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from axis_isp import (MoranAxisISP, apply_palette_projection, apply_nearest_swatch_palette,
                      apply_split_palette_projection, color_protection_factor, linear_to_srgb8)
from differentiable_isp import _compress_gamut, _luminance, load_and_preprocess_raw


AXES = ("thermal", "fidelity", "moral", "vitality", "hue_shift")
POLE_DESCRIPTOR_PATH = Path(__file__).with_name("legacy") / "moran_pole_descriptors.json"
CONTROL_NAMES = ("warmth", "saturation", "tone_gain", "contrast", "sharpness", "palette")
CONTROL_BOUNDS = ((-0.3, 0.3), (-1.0, 1.0), (0.05, 5.0),
                  (0.2, 3.0), (-0.9, 0.9), (-0.4, 0.4))
CONTROL_NEUTRAL = (0.0048, 0.2637, 0.9755, 1.2032, 0.0391, -0.0097)
TINT_DEFAULT = 0.0041
SEMANTIC_ENDPOINTS = {
    "warmth":    {"default": 0.0048, "neg": -0.3000, "pos": 0.3000, "scale": "linear"},
    "saturation":{"default": 0.2637, "neg": -0.7337, "pos": 1.0000, "scale": "linear"},
    "tone_gain": {"default": 0.9755, "neg": 0.0500,  "pos": 5.0000, "scale": "log"},
    "contrast":  {"default": 1.2032, "neg": 0.6795,  "pos": 2.1967, "scale": "linear"},
    "sharpness": {"default": 0.0391, "neg": -0.0991, "pos": 0.4111, "scale": "linear"},
    "palette":   {"default": 0.0000, "neg": -0.2090, "pos": 0.0921, "scale": "linear"},
    "tint":      {"default": 0.0000, "neg": -0.0981, "pos": 0.1278, "scale": "linear"},
}


def linear_to_srgb(image):
    return torch.where(image <= 0.0031308, 12.92 * image,
        1.055 * image.clamp_min(0.0031308).pow(1 / 2.4) - 0.055)


def srgb_to_linear(image):
    return torch.where(image <= 0.04045, image / 12.92,
        ((image + 0.055) / 1.055).clamp_min(0).pow(2.4))


def load_pole_descriptors(path=POLE_DESCRIPTOR_PATH):
    table = json.loads(Path(path).read_text())
    axes = table.get("axes", {})
    if set(axes) != set(AXES):
        raise ValueError(f"Descriptor table must define exactly these axes: {AXES}")
    for axis in AXES:
        for sign in ("negative", "positive"):
            rows = axes[axis].get(sign, [])
            if len(rows) != 4 or {row["id"] for row in rows} != {"D1", "D2", "D3", "D4"}:
                raise ValueError(f"{axis}/{sign} must have exactly D1-D4 descriptors")
            if any(not row["text"].strip() for row in rows):
                raise ValueError(f"Empty descriptor text in {axis}/{sign}")
    return table


def saturation_matrix(multiplier):
    weights = np.array([0.299, 0.587, 0.114])
    return (multiplier * np.eye(3) + (1 - multiplier) * np.tile(weights, (3, 1))).tolist()


def seed_config():
    descriptor_table = load_pole_descriptors()
    poles = {}
    for axis in AXES:
        poles[axis] = {}
        for sign in ("negative", "positive"):
            descriptor_texts = [row["text"] for row in descriptor_table["axes"][axis][sign]]
            poles[axis][sign] = {
                "texts": descriptor_texts, "wb": [1.0, 1.0, 1.0],
                "ccm_delta": np.zeros((3, 3)).tolist(), "tone_gain": 1.0,
                "contrast": 1.0,
                "sharpness": 0.0, "radius": 2.0,
            }
    poles["thermal"]["negative"]["wb"] = [0.82, 1.0, 1.18]
    poles["thermal"]["positive"]["wb"] = [1.18, 1.0, 0.82]
    poles["fidelity"]["negative"].update(sharpness=-0.8, contrast=0.7)
    poles["fidelity"]["positive"].update(sharpness=0.8, contrast=1.5)
    poles["moral"]["negative"].update(tone_gain=0.65, contrast=1.3)
    poles["moral"]["positive"].update(tone_gain=1.3, contrast=0.85)
    for sign, multiplier in (("negative", 0.0), ("positive", 1.6)):
        poles["vitality"][sign]["ccm_delta"] = (
            np.array(saturation_matrix(multiplier)) - np.eye(3)
        ).tolist()
    palette = MoranAxisISP("cpu")
    return {
        "status": "Engineering seed calibration, not professional colorist validation",
        "descriptor_provenance": "legacy/moran_pole_descriptors.json; five-axis cinematic seed only",
        "descriptor_table": descriptor_table,
        "model": "ViT-B-32", "pretrained": "openai", "alpha": 2.0,
        "force_quick_gelu": True,
        "coordinate_limit": 0.4,
        "hue_method": "Soft palette-axis projection in linear YIQ; no hue CCM contribution",
        "palette_projection": {
            "rgb_to_yiq": palette.rgb_to_yiq.tolist(),
            "yiq_to_rgb": palette.yiq_to_rgb.tolist(),
            "points": palette.palette_axis_points.tolist(),
            "swatches": palette.PALETTE_SWATCHES,
            "max_attraction": palette.HUE_AXIS_MAX_ATTRACTION,
            "shift_fraction": palette.HUE_AXIS_SHIFT_FRACTION,
        },
        "sharpening_rule": "Largest absolute weighted strength; signed blur extension to deck max rule",
        "poles": poles,
    }


def relevance_features(features, centroids):
    features = F.normalize(features, dim=-1)
    similarities = torch.einsum("bd,ksd->bks", features, centroids)
    average = similarities.mean(dim=-1)
    maximum = similarities.amax(dim=-1)
    difference = (similarities[:, :, 1] - similarities[:, :, 0]).abs()
    eye = torch.eye(centroids.shape[0], dtype=torch.bool, device=features.device)
    alternatives = average[:, None, :].expand(-1, centroids.shape[0], -1)
    other_maximum = alternatives.masked_fill(eye[None], -torch.inf).amax(dim=-1)
    midpoint = centroids.mean(dim=1)
    direction = F.normalize(centroids[:, 1] - centroids[:, 0], dim=-1)
    delta = features[:, None, :] - midpoint[None]
    along = (delta * direction[None]).sum(dim=-1, keepdim=True)
    perpendicular = (delta - along * direction[None]).norm(dim=-1)
    return torch.stack((average, maximum, average - other_maximum, perpendicular, difference), dim=-1)


class SemanticProjection:
    def __init__(self, encode_text, config):
        self.encode_text = encode_text
        self.config = copy.deepcopy(config)
        self.axes = tuple(config.get("axes", AXES))
        alpha, limit = float(config["alpha"]), float(config["coordinate_limit"])
        if not np.isfinite(alpha) or alpha <= 0 or not 0 < limit <= 1:
            raise ValueError("Require finite positive alpha and coordinate_limit in (0, 1]")
        centroids = []
        learned_centroids = config.get("learned_calibration", {}).get("pole_centroid_texts", {})
        with torch.no_grad():
            for axis in self.axes:
                pair = []
                for sign in ("negative", "positive"):
                    texts = learned_centroids.get(axis, {}).get(
                        sign, config["poles"][axis][sign]["texts"])
                    if not texts or any(not text.strip() for text in texts):
                        raise ValueError(f"Empty descriptor bank: {axis}/{sign}")
                    features = F.normalize(encode_text(texts), dim=-1)
                    centroid = F.normalize(features.mean(dim=0), dim=0)
                    pair.append(centroid)
                centroids.append(torch.stack(pair))
        self.centroids = torch.stack(centroids)

    @torch.no_grad()
    def relevance_weights(self, features):
        if features.ndim == 1:
            features = features.unsqueeze(0)
        gate = self.config.get("relevance_gate")
        if gate is None:
            ones = features.new_ones((len(features), len(self.axes)))
            return ones, ones
        measurements = relevance_features(features, self.centroids)
        standardized = ((measurements - measurements.new_tensor(gate["feature_mean"]))
                        / measurements.new_tensor(gate["feature_scale"]))
        logits = (standardized * measurements.new_tensor(gate["coefficients"])).sum(dim=-1)
        probabilities = torch.sigmoid(logits + measurements.new_tensor(gate["intercepts"]))
        threshold = measurements.new_tensor(gate["thresholds"])
        upper = measurements.new_tensor(gate.get("full_weight_at", [1.] * len(self.axes)))
        progress = ((probabilities - threshold) / (upper - threshold).clamp_min(1e-6)).clamp(0, 1)
        weights = progress.square() * (3 - 2 * progress)
        return probabilities, weights

    @torch.no_grad()
    def project_features(self, features):
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2 or not torch.isfinite(features).all():
            raise ValueError("Expected finite text features with shape (batch, embedding_dim)")
        features = F.normalize(features, dim=-1)
        similarities = torch.einsum("bd,ksd->bks", features, self.centroids)
        raw_scores = similarities[:, :, 1] - similarities[:, :, 0]
        limit = float(self.config["coordinate_limit"])
        if "learned_calibration" in self.config:
            learned = self.config["learned_calibration"]
            centered = raw_scores - raw_scores.new_tensor(learned["score_origin"])
            scale = torch.where(centered >= 0,
                                centered.new_tensor(learned["positive_scale"]),
                                centered.new_tensor(learned["negative_scale"]))
            coordinates = (centered / scale).clamp(-1, 1) * limit
        else:
            coordinates = (raw_scores * float(self.config["alpha"])).clamp(-limit, limit)
        if "relevance_gate" in self.config:
            _, weights = self.relevance_weights(features)
            coordinates = coordinates * weights
        return raw_scores, coordinates

    @torch.no_grad()
    def project(self, prompts):
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("Provide nonempty text requests")
        return self.project_features(self.encode_text(prompts))


def bounded_controls(raw_controls, bounds=CONTROL_BOUNDS):
    bounds = raw_controls.new_tensor(bounds)
    if bounds.shape != (len(CONTROL_NAMES), 2) or not torch.isfinite(bounds).all() or (bounds[:, 0] >= bounds[:, 1]).any():
        raise ValueError("Control bounds must be finite increasing pairs for every ISP control")
    return bounds.mean(dim=1) + torch.tanh(raw_controls) * (bounds[:, 1] - bounds[:, 0]) / 2


def neutral_raw_controls(device="cpu", dtype=torch.float32, bounds=CONTROL_BOUNDS):
    bounds = torch.tensor(bounds, device=device, dtype=dtype)
    neutral = bounds.new_tensor(CONTROL_NEUTRAL)
    return torch.atanh((neutral - bounds.mean(dim=1)) / ((bounds[:, 1] - bounds[:, 0]) / 2))


def parameters_from_controls(controls, config):
    warmth, saturation, gain, contrast, sharpness, palette = controls.unbind(dim=0)
    weights = controls.new_tensor([0.299, 0.587, 0.114]).expand(3, -1)
    identity = torch.eye(3, device=controls.device, dtype=controls.dtype)
    wb = torch.stack((1 + warmth, torch.ones_like(warmth), 1 - warmth))
    return {
        "wb": wb, "saturation": saturation,
        "ccm": (1 + saturation) * identity - saturation * weights,
        "tone_gain": gain, "contrast": contrast, "sharpness": sharpness,
        "radius": controls.new_tensor(config.get("detail_filter", {}).get("sigma_at_reference", 2.0)),
        "palette_coordinate": palette,
        "color_protection": config.get("color_protection"),
        **({"detail_reference_short_side": controls.new_tensor(config["detail_filter"]["reference_short_side"]),
            "detail_truncate": controls.new_tensor(config["detail_filter"].get("truncate", 3.0))}
           if "detail_filter" in config else {}),
            **({"sharpen_radius": controls.new_tensor(config["detail_filter"]["sharpen_sigma_at_reference"])}
               if config.get("detail_filter", {}).get("sharpen_sigma_at_reference") is not None else {}),
          **({"blur_rgb": controls.new_tensor(1.)}
              if config.get("detail_filter", {}).get("blur_color_mode") == "rgb" else {}),
        **({"sharpen_gain": controls.new_tensor(config["positive_sharpening"]["gain"]),
            "sharpen_detail_limit": controls.new_tensor(config["positive_sharpening"]["detail_limit"])}
           if "positive_sharpening" in config else {}),
        **({"palette_split_swatches": controls.new_tensor([
                config["palette_projection"]["swatches"]["retro"],
                config["palette_projection"]["swatches"]["blockbuster"]]),
            "palette_split_coordinate_limit": controls.new_tensor(config["palette_split_tone"]["coordinate_limit"]),
            "palette_split_positive_mix": controls.new_tensor(config["palette_split_tone"]["positive_mix"]),
            "palette_split_negative_mix": controls.new_tensor(config["palette_split_tone"]["negative_mix"]),
            "palette_shadow_quantile": config["palette_split_tone"].get("shadow_quantile", .15),
            "palette_highlight_quantile": config["palette_split_tone"].get("highlight_quantile", .85),
            "palette_split_method": config["palette_split_tone"].get("method", "luminance_split")}
           if "palette_split_tone" in config else {}),
        **({"palette_nearest_swatches": controls.new_tensor([
                config["palette_projection"]["swatches"]["retro"],
                config["palette_projection"]["swatches"]["blockbuster"]]),
            "palette_nearest_softness": controls.new_tensor(config["palette_nearest_swatch"]["softness_ratio"])}
           if "palette_nearest_swatch" in config else {}),
        **{f"palette_{key}": controls.new_tensor(value)
           for key, value in config["palette_projection"].items() if key != "swatches"},
    }


def compose_parameters(coordinates, config):
    axes = tuple(config.get("axes", AXES))
    if coordinates.shape != (len(axes),) or not torch.isfinite(coordinates).all():
        raise ValueError(f"Expected {len(axes)} finite coordinates")
    if (coordinates.abs() > 1).any():
        raise ValueError("Coordinates must be in [-1, 1]")
    if "learned_calibration" in config:
        learned = config["learned_calibration"]
        normalized = (coordinates / float(config["coordinate_limit"])).clamp(-1, 1)
        features = torch.cat((normalized.clamp_min(0), (-normalized).clamp_min(0)))
        raw = coordinates.new_tensor(learned["raw_neutral"]) + (
            features @ coordinates.new_tensor(learned["pole_raw_deltas"])
        )
        bounds = config.get("control_bounds", learned.get("control_bounds", CONTROL_BOUNDS))
        return parameters_from_controls(bounded_controls(raw, bounds), config)
    wb = coordinates.new_ones(3)
    ccm = torch.eye(3, dtype=coordinates.dtype, device=coordinates.device)
    gain = coordinates.new_ones(())
    contrast = coordinates.new_ones(())
    strengths, radii = [], []
    for index, axis in enumerate(AXES):
        coordinate = coordinates[index]
        strength = coordinate.abs()
        negative = config["poles"][axis]["negative"]
        positive = config["poles"][axis]["positive"]

        def selected(name):
            return torch.where(coordinate >= 0, coordinates.new_tensor(positive[name]),
                               coordinates.new_tensor(negative[name]))

        wb = wb * (1 + strength * (selected("wb") - 1))
        if axis != "hue_shift":
            ccm = ccm + strength * selected("ccm_delta")
        gain = gain * (1 + strength * (selected("tone_gain") - 1))
        contrast = contrast * (1 + strength * (selected("contrast") - 1))
        strengths.append(strength * selected("sharpness"))
        radii.append(selected("radius"))
    weighted = torch.stack(strengths)
    winner = weighted.abs().argmax()
    return {"wb": wb, "ccm": ccm, "tone_gain": gain, "contrast": contrast,
            "sharpness": weighted[winner],
            "radius": torch.stack(radii)[winner],
            "palette_coordinate": coordinates[4],
            **{f"palette_{key}": coordinates.new_tensor(value)
               for key, value in config["palette_projection"].items() if key != "swatches"}}


def detail_gaussian(luma, parameters, radius=None):
    sigma = (parameters["radius"] if radius is None else radius).clamp_min(.2)
    channels = luma.shape[1]
    if "detail_reference_short_side" not in parameters:
        positions = torch.arange(-3, 4, dtype=luma.dtype, device=luma.device)
        kernel = torch.exp(-positions.square() / (2 * sigma.square()))
        kernel = kernel / kernel.sum()
        kernel = (kernel[:, None] * kernel[None, :])[None, None]
        return F.conv2d(F.pad(luma, (3, 3, 3, 3), mode="replicate"),
                kernel.expand(channels, 1, 7, 7), groups=channels)
    reference = float(parameters["detail_reference_short_side"])
    truncate = float(parameters["detail_truncate"])
    if not math.isfinite(reference) or reference <= 0 or not math.isfinite(truncate) or truncate <= 0:
        raise ValueError("Detail reference resolution and Gaussian truncation must be positive and finite")
    sigma = sigma * min(luma.shape[-2:]) / reference
    half_width = max(1, math.ceil(truncate * float(sigma.detach())))
    positions = torch.arange(-half_width, half_width + 1, dtype=luma.dtype, device=luma.device)
    kernel = torch.exp(-positions.square() / (2 * sigma.square()))
    kernel = kernel / kernel.sum()
    horizontal = F.conv2d(F.pad(luma, (half_width, half_width, 0, 0), mode="replicate"),
                         kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
    return F.conv2d(F.pad(horizontal, (0, 0, half_width, half_width), mode="replicate"),
                    kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)


def render_linear(image, parameters):
    if not image.is_floating_point() or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected floating-point BCHW linear RGB")
    color_reference = image
    protection = parameters.get("color_protection")
    warmth_amount = parameters["wb"][0] - 1
    warmth_scale = color_protection_factor(color_reference, protection, "warmth", warmth_amount)
    full_balance = image * parameters["wb"].view(1, 3, 1, 1)
    balanced = image + warmth_scale * (full_balance - image)
    image = _compress_gamut(balanced, image)
    palette_coordinate = parameters["palette_coordinate"]
    positive = palette_coordinate >= 0
    attraction = torch.where(positive,
        parameters.get("palette_max_attraction_positive", parameters["palette_max_attraction"]),
        parameters.get("palette_max_attraction_negative", parameters["palette_max_attraction"]))
    shift = torch.where(positive,
        parameters.get("palette_shift_fraction_positive", parameters["palette_shift_fraction"]),
        parameters.get("palette_shift_fraction_negative", parameters["palette_shift_fraction"]))
    if parameters.get("palette_split_method") == "nearest_swatch":
        projected = apply_nearest_swatch_palette(image, palette_coordinate,
            parameters["palette_rgb_to_yiq"], parameters["palette_yiq_to_rgb"],
            parameters["palette_split_swatches"], parameters["palette_split_coordinate_limit"],
            parameters["palette_split_positive_mix"], parameters["palette_split_negative_mix"],
            parameters["palette_nearest_softness"])
    elif "palette_split_swatches" in parameters:
        projected = apply_split_palette_projection(image, palette_coordinate,
            parameters["palette_rgb_to_yiq"], parameters["palette_yiq_to_rgb"],
            parameters["palette_split_swatches"], parameters["palette_split_coordinate_limit"],
            parameters["palette_split_positive_mix"], parameters["palette_split_negative_mix"],
            parameters["palette_shadow_quantile"], parameters["palette_highlight_quantile"])
    else:
        projected = apply_palette_projection(
            image, palette_coordinate, parameters["palette_rgb_to_yiq"],
            parameters["palette_yiq_to_rgb"], parameters["palette_points"], attraction, shift,
        )
    palette_scale = color_protection_factor(color_reference, protection, "palette", palette_coordinate)
    image = _compress_gamut(image + palette_scale * (projected - image), image)
    colored = torch.einsum("ij,bjhw->bihw", parameters["ccm"], image)
    saturation_amount = parameters.get("saturation")
    if saturation_amount is None:
        saturation_amount = -parameters["ccm"][0, 1] / .587
    saturation_scale = color_protection_factor(color_reference, protection, "saturation",
                                                saturation_amount)
    image = _compress_gamut(image + saturation_scale * (colored - image), image)
    luma = _luminance(image)
    safe = luma.clamp(1e-6, 1 - 1e-6)
    tone_mapped_luma = torch.sigmoid(torch.logit(safe) + torch.log(parameters["tone_gain"]))
    image = _compress_gamut(image * (tone_mapped_luma / safe), image)

    encoded = linear_to_srgb(image.clamp(0, 1))
    contrast_luma = _luminance(encoded).clamp(1e-5, 1 - 1e-5)
    pivot = contrast_luma.flatten(1).median(dim=1).values[:, None, None, None]
    contrast_target = torch.sigmoid(
        parameters["contrast"] * (torch.logit(contrast_luma) - torch.logit(pivot))
        + torch.logit(pivot))
    contrast_shift = contrast_target - contrast_luma
    encoded = _compress_gamut(encoded + contrast_shift, encoded)
    image = srgb_to_linear(encoded)
    amount = parameters["sharpness"].clamp(-1, 1)
    if parameters.get("blur_rgb", False) and float(amount.detach()) < 0:
        blurred_rgb = detail_gaussian(image, parameters)
        return (image * (1 + amount) - blurred_rgb * amount).clamp(0, 1)
    luma = _luminance(image)
    blur = detail_gaussian(luma, parameters, parameters.get("sharpen_radius"))
    detail_limit = parameters.get("sharpen_detail_limit", .0285)
    gain = parameters.get("sharpen_gain", .8)
    detail = torch.tanh((luma - blur) / detail_limit) * detail_limit
    target = torch.where(amount >= 0, luma + gain * amount * detail,
                         luma * (1 + amount) - blur * amount).clamp(0, 1)
    return _compress_gamut(image * (target / luma.clamp_min(1e-6)), image)


def write_gallery(root, report):
    axes = tuple(report["config"].get("axes", AXES))
    labels = ["neutral", *[entry["prompt"] for entry in report["prompts"]]]
    files = ["neutral", *[f"prompt_{index:02d}" for index in range(len(report["prompts"]))]]
    document = ["<!doctype html><meta charset='utf-8'><title>Moran semantic ISP</title>",
                "<h1>Text-to-ISP prototype</h1>",
                "<p>Engineering seed calibration. Coordinates are computed from text, not a word lookup table. "
                "Linear processing through sharpening; final 8-bit sRGB export.</p>",
                "<table border='1'><tr><th>Request</th>"
                + "".join(f"<th>{axis}</th>" for axis in axes) + "</tr>"]
    for entry in report["prompts"]:
        document.append("<tr><td>" + html.escape(entry["prompt"]) + "</td>"
                        + "".join(f"<td>{entry['coordinates'][axis]:+.3f}</td>" for axis in axes)
                        + "</tr>")
    document.append("</table>")
    for image_id in report["images"]:
        directory = root / f"img_{image_id}"
        rows = (len(files) + 3) // 4
        sheet = Image.new("RGB", (4 * 280, rows * 220), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (label, filename) in enumerate(zip(labels, files)):
            left, top = (index % 4) * 280, (index // 4) * 220
            with Image.open(directory / f"{filename}.png") as source:
                source.thumbnail((280, 190), Image.Resampling.LANCZOS)
                sheet.paste(source, (left + (280 - source.width) // 2,
                                     top + 25 + (190 - source.height) // 2))
            draw.text((left + 6, top + 5), label, fill="black")
        sheet.save(directory / "comparison.jpg", quality=92)
        document.append(f"<h2>Image {image_id}</h2>"
                        f"<img style='max-width:100%' src='img_{image_id}/comparison.jpg'>")
        document.append("<p>" + " | ".join(
            f"<a href='img_{image_id}/{filename}.png'>{html.escape(label)}</a>"
            for label, filename in zip(labels, files)) + "</p>")
    (root / "index.html").write_text("\n".join(document))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", nargs="+", type=int, default=[1, 2, 33])
    parser.add_argument("--prompts", nargs="+", default=["warm", "cold", "dreamy", "dramatic",
                                                       "vivid", "retro", "warm but muted"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    config = json.loads(args.config.read_text()) if args.config else seed_config()
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=args.device,
        force_quick_gelu=config.get("force_quick_gelu", config["pretrained"] == "openai"))
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])

    @torch.no_grad()
    def encode_text(texts):
        return model.encode_text(tokenizer(texts).to(args.device))

    projection = SemanticProjection(encode_text, config)
    raw_scores, coordinates = projection.project(args.prompts)
    parameters = [compose_parameters(coordinate, config) for coordinate in coordinates]
    neutral = compose_parameters(coordinates.new_zeros(len(projection.axes)), config)
    args.out.mkdir(parents=True)
    (args.out / "calibration.json").write_text(json.dumps(config, indent=2))
    torch.save(projection.centroids.cpu(), args.out / "pole_centroids.pt")
    report = {"config": config, "images": args.images, "prompts": []}
    for index, prompt in enumerate(args.prompts):
        report["prompts"].append({"prompt": prompt,
            "raw_scores": dict(zip(projection.axes, raw_scores[index].cpu().tolist())),
            "coordinates": dict(zip(projection.axes, coordinates[index].cpu().tolist())),
            "parameters": {key: value.detach().cpu().tolist() for key, value in parameters[index].items()}})
    torch.set_num_threads(2)
    with torch.no_grad():
        for image_id in args.images:
            image = load_and_preprocess_raw(
                f"/data/data1/EmotionISP_dataset/dng_filtered/{image_id}.dng", None).to(args.device)
            directory = args.out / f"img_{image_id}"
            directory.mkdir()
            for label, setting in [("neutral", neutral),
                                   *[(f"prompt_{index:02d}", setting)
                                     for index, setting in enumerate(parameters)]]:
                output = render_linear(image, setting)
                if not torch.isfinite(output).all():
                    raise ValueError(f"Nonfinite render: {image_id}/{label}")
                pixels = linear_to_srgb8(output)[0].permute(1, 2, 0).cpu().numpy()
                if not cv2.imwrite(str(directory / f"{label}.png"),
                                   cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"Failed export: {directory}/{label}")
            print(f"rendered image {image_id}", flush=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=2))
    write_gallery(args.out, report)
    print(json.dumps({"output": str(args.out), "images": len(args.images),
                      "prompts": len(args.prompts)}, indent=2))


if __name__ == "__main__":
    main()