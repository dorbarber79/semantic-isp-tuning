"""Prototype differentiable linear-light ISP controlled by Moran's five semantic axes."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from differentiable_isp import DifferentiableEMOVIS, _compress_gamut, _luminance


def chromaticity_proximity(image, reference_rgb, sigma, reference_is_srgb=False):
    """Return Gaussian proximity to a color in brightness-independent RGB chromaticity."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected BCHW RGB image")
    if not math.isfinite(float(sigma)) or sigma <= 0:
        raise ValueError("Chromaticity sigma must be positive and finite")
    color = image.clamp_min(0)
    total = color.sum(dim=1, keepdim=True)
    neutral = color.new_full((1, 3, 1, 1), 1 / 3)
    chromaticity = torch.where(total > 1e-8, color / total.clamp_min(1e-8), neutral)
    reference = torch.as_tensor(reference_rgb, dtype=color.dtype, device=color.device).view(1, 3, 1, 1)
    if reference.shape[1] != 3 or not torch.isfinite(reference).all():
        raise ValueError("Reference RGB must contain three finite channels")
    if reference_is_srgb:
        reference = torch.where(reference <= .04045, reference / 12.92,
            ((reference + .055) / 1.055).clamp_min(0).pow(2.4))
    reference = reference.clamp_min(0)
    reference = reference / reference.sum(dim=1, keepdim=True).clamp_min(1e-8)
    distance = (chromaticity - reference).square().sum(dim=1, keepdim=True).sqrt()
    return torch.exp(-.5 * (distance / sigma).square())


def color_protection_factor(image, protection, operation, amount=0.):
    """Return the per-pixel fraction of a color-operation delta to retain."""
    ones = image.new_ones((image.shape[0], 1, image.shape[2], image.shape[3]))
    if not protection or protection.get("mode", "none") == "none":
        return ones
    operations = protection.get("operations")
    if operations is not None and operation not in operations:
        return ones
    if operation == "saturation" and protection.get("saturation_positive_only", True) and float(amount) <= 0:
        return ones
    mode = protection.get("mode")
    if mode not in {"gray_white", "skin", "gray_white_and_skin"}:
        raise ValueError(f"Unknown color-protection mode: {mode}")
    scores = []
    if mode in {"gray_white", "gray_white_and_skin"}:
        scores.append(chromaticity_proximity(image, (1 / 3, 1 / 3, 1 / 3),
            protection.get("gray_sigma", .07)))
    if mode in {"skin", "gray_white_and_skin"}:
        scores.append(chromaticity_proximity(image,
            protection.get("skin_reference_srgb", (.76, .57, .47)),
            protection.get("skin_sigma", .10), reference_is_srgb=True))
    closeness = torch.stack(scores).amax(dim=0)
    strength = float(protection.get("strength", 1.))
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Color-protection strength must be in [0, 1]")
    return 1 - strength * closeness


def linear_to_srgb8(image_linear):
    """Apply the output transfer and quantize only after all linear ISP stages."""
    if not image_linear.is_floating_point():
        raise TypeError("Expected floating-point linear RGB")
    linear = image_linear.clamp(0, 1)
    srgb = torch.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * linear.clamp_min(0.0031308).pow(1.0 / 2.4) - 0.055,
    )
    return torch.round(srgb.clamp(0, 1) * 255).to(torch.uint8)


def apply_palette_projection(image, amount, rgb_to_yiq, yiq_to_rgb, points,
                             max_attraction, shift_fraction):
    yiq = torch.einsum("ij,bjhw->bihw", rgb_to_yiq, image)
    chroma = yiq[:, 1:3]
    negative, positive = points.unbind(dim=0)
    direction = positive - negative
    axis_length = direction.norm().clamp_min(1e-8)
    unit = (direction / axis_length).view(1, 2, 1, 1)
    origin = negative.view(1, 2, 1, 1)
    projection = origin + ((chroma - origin) * unit).sum(dim=1, keepdim=True) * unit
    perpendicular = chroma - projection
    adjusted_chroma = (projection + (1 - max_attraction * amount.square()) * perpendicular
                       + shift_fraction * amount * axis_length * unit)
    adjusted_yiq = torch.cat((yiq[:, :1], adjusted_chroma), dim=1)
    return torch.einsum("ij,bjhw->bihw", yiq_to_rgb, adjusted_yiq)


def apply_split_palette_projection(image, amount, rgb_to_yiq, yiq_to_rgb, swatches,
                                   coordinate_limit, positive_mix, negative_mix,
                                   shadow_quantile=.15, highlight_quantile=.85):
    """Blend chroma toward the requested shadow/highlight swatches as luminance changes."""
    if not 0 <= shadow_quantile < highlight_quantile <= 1:
        raise ValueError("Palette shadow/highlight quantiles must be ordered in [0, 1]")
    yiq = torch.einsum("ij,bjhw->bihw", rgb_to_yiq, image)
    neutral = yiq[:, :1]
    swatch_yiq = torch.einsum("ij,pkj->pki", rgb_to_yiq, swatches)
    endpoint_chroma = swatch_yiq[:, :, 1:3]
    negative_chroma, positive_chroma = endpoint_chroma.unbind(dim=0)
    selected = torch.where(amount >= 0, positive_chroma, negative_chroma)
    shadow = selected[0].view(1, 2, 1, 1)
    highlight = selected[1].view(1, 2, 1, 1)
    quantiles = neutral.new_tensor([shadow_quantile, highlight_quantile])
    thresholds = torch.quantile(neutral.flatten(2), quantiles, dim=-1)
    low = thresholds[0].view(-1, 1, 1, 1)
    high = thresholds[1].view(-1, 1, 1, 1)
    progress = ((neutral - low) / (high - low).clamp_min(1e-4)).clamp(0, 1)
    luminance_weight = progress.square() * (3 - 2 * progress)
    target = shadow * (1 - luminance_weight) + highlight * luminance_weight
    maximum_mix = torch.where(amount >= 0, positive_mix, negative_mix)
    mix = (amount.abs() / coordinate_limit).clamp(0, 1) * maximum_mix
    target_rgb = torch.einsum("ij,bjhw->bihw", yiq_to_rgb,
                              torch.cat((neutral, target), dim=1))
    return image + mix * (target_rgb - image)


def apply_nearest_swatch_palette(image, amount, rgb_to_yiq, yiq_to_rgb, swatches,
                                 coordinate_limit, positive_mix, negative_mix,
                                 softness_ratio=.18):
    """Classify each pixel by opponent-chroma distance and pull it toward a palette swatch."""
    if not 0 < softness_ratio <= 1:
        raise ValueError("Palette softness ratio must be in (0, 1]")
    yiq = torch.einsum("ij,bjhw->bihw", rgb_to_yiq, image)
    luma = yiq[:, :1]
    chroma = yiq[:, 1:3]
    swatch_yiq = torch.einsum("ij,pkj->pki", rgb_to_yiq, swatches)
    endpoints = swatch_yiq[:, :, 1:3]
    endpoint_pair = torch.where(amount >= 0, endpoints[1], endpoints[0])
    gap = (endpoint_pair[1] - endpoint_pair[0]).norm().clamp_min(1e-6)
    softness = gap * softness_ratio
    distances = (chroma.unsqueeze(1) - endpoint_pair.view(1, 2, 2, 1, 1)).square().sum(dim=2)
    assignments = torch.softmax(-distances / (2 * softness.square()), dim=1)
    target_chroma = (assignments.unsqueeze(2) * endpoint_pair.view(1, 2, 2, 1, 1)).sum(dim=1)
    target_rgb = torch.einsum("ij,bjhw->bihw", yiq_to_rgb,
                              torch.cat((luma, target_chroma), dim=1))
    mix_limit = torch.where(amount >= 0, positive_mix, negative_mix)
    mix = (amount.abs() / coordinate_limit).clamp(0, 1) * mix_limit
    return image + mix * (target_rgb - image)


class SemanticAxisISP(nn.Module):
    """Linear-RGB ISP with five bounded bipolar axis controls.

    Positive poles are Golden, Documentary, High key, Vivid, and Teal-Orange.
    Pole strengths below are an initial engineering calibration, not measured
    perceptual settings from the proposal deck.
    """

    AXES = ("thermal", "fidelity", "moral", "vitality", "hue_shift")
    POLES = {
        "thermal": ("Cyan", "Golden"),
        "fidelity": ("Ethereal", "Documentary"),
        "moral": ("Chiaroscuro", "High key"),
        "vitality": ("Bleached", "Vivid"),
        "hue_shift": ("Magenta-Green (synthware / retro)", "Teal-Orange (blockbuster)"),
    }
    AXIS_STEPS = (-0.4, -0.8 / 3.0, -0.4 / 3.0, 0.0,
                  0.4 / 3.0, 0.8 / 3.0, 0.4)

    # Mapped control displacement per unit axis coordinate.
    AXIS_DELTAS = {
        "thermal": {"wb_warmth": 0.22},
        "fidelity": {"sharpness": 1.15, "contrast": 0.75},
        "moral": {"contrast": -0.20},
        "vitality": {"brightness": 0.02},
    }
    MORAL_DARKNESS_ENDPOINT = -0.08
    VITALITY_MONOCHROME_ENDPOINT = -1.0
    VITALITY_SATURATION_DELTA = 0.62
    PALETTE_SWATCHES = {
        "retro": ((0.76, 0.12, 0.66), (0.15, 0.62, 0.24)),
        "blockbuster": ((0.06, 0.50, 0.58), (0.92, 0.44, 0.13)),
    }
    PALETTE_POINT_RADIUS = 0.18
    HUE_AXIS_MAX_ATTRACTION = 0.45
    HUE_AXIS_SHIFT_FRACTION = 0.12

    def __init__(self, device="cuda", defaults=None):
        super().__init__()
        self.raw_axes = nn.Parameter(torch.zeros(len(self.AXES), device=device, dtype=torch.float32))
        defaults = defaults or DifferentiableEMOVIS.DEFAULT_PARAMS
        self.register_buffer(
            "defaults",
            torch.tensor([defaults[k]
                          for k in DifferentiableEMOVIS.PARAM_NAMES], device=device, dtype=torch.float32),
        )
        yiq = torch.tensor(
            [[0.299, 0.587, 0.114],
             [0.596, -0.274, -0.322],
             [0.211, -0.523, 0.312]],
            device=device, dtype=torch.float32,
        )
        self.register_buffer("rgb_to_yiq", yiq)
        self.register_buffer("yiq_to_rgb", torch.linalg.inv(yiq))
        palette_points = []
        for first, second in (self.PALETTE_SWATCHES["retro"],
                              self.PALETTE_SWATCHES["blockbuster"]):
            rgb_delta = torch.tensor(second, device=device) - torch.tensor(first, device=device)
            opponent_delta = self.rgb_to_yiq @ rgb_delta
            chroma_delta = opponent_delta[1:]
            palette_points.append(
                chroma_delta / chroma_delta.norm().clamp_min(1e-8) * self.PALETTE_POINT_RADIUS
            )
        self.register_buffer("palette_axis_points", torch.stack(palette_points))

        kernel_size, sigma = 7, 2.0
        coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-coords.square() / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :])[None, None]
        self.register_buffer("sharp_kernel", kernel_2d)

    def axis_values(self):
        return torch.tanh(self.raw_axes)

    def set_axes(self, values):
        """Set mapped axis coordinates in [-1, 1], preserving differentiable storage."""
        if set(values) - set(self.AXES):
            raise ValueError(f"Unknown axes: {set(values) - set(self.AXES)}")
        with torch.no_grad():
            self.raw_axes.zero_()
            for index, name in enumerate(self.AXES):
                value = float(values.get(name, 0.0))
                if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                    raise ValueError(f"{name} must be finite and in [-1, 1]")
                self.raw_axes[index] = math.atanh(max(-0.999999, min(0.999999, value)))

    def mapped_controls(self):
        axes = self.axis_values()
        controls = {name: self.defaults[index] for index, name in enumerate(DifferentiableEMOVIS.PARAM_NAMES)}
        for axis_index, axis in enumerate(self.AXES[:4]):
            for name, strength in self.AXIS_DELTAS[axis].items():
                controls[name] = controls[name] + strength * axes[axis_index]

        moral = axes[2]
        moral_negative_progress = (-moral / 0.4).clamp(0, 1)
        moral_negative_blend = moral_negative_progress.square() * (3 - 2 * moral_negative_progress)
        moral_default = self.defaults[DifferentiableEMOVIS.PARAM_NAMES.index("brightness")]
        moral_positive_brightness = moral_default + 0.30 * moral.clamp_min(0)
        moral_negative_brightness = moral_default + (
            self.MORAL_DARKNESS_ENDPOINT - moral_default
        ) * moral_negative_blend
        controls["brightness"] = torch.where(moral < 0, moral_negative_brightness,
                                              moral_positive_brightness)

        vitality = axes[3]
        saturation_default = self.defaults[DifferentiableEMOVIS.PARAM_NAMES.index("saturation")]
        vitality_negative_progress = ((vitality + 0.4) / 0.4).clamp(0, 1)
        vitality_negative_blend = (
            vitality_negative_progress.square() * (3 - 2 * vitality_negative_progress)
        )
        vitality_positive_progress = (vitality / 0.4).clamp(0, 1)
        vitality_positive_blend = (
            vitality_positive_progress.square() * (3 - 2 * vitality_positive_progress)
        )
        monochrome_saturation = self.VITALITY_MONOCHROME_ENDPOINT + (
            saturation_default - self.VITALITY_MONOCHROME_ENDPOINT
        ) * vitality_negative_blend
        vivid_saturation = saturation_default + (
            self.VITALITY_SATURATION_DELTA * 0.4 * vitality_positive_blend
        )
        controls["saturation"] = torch.where(vitality < 0, monochrome_saturation,
                                              vivid_saturation)

        for name, value in controls.items():
            lo, hi = DifferentiableEMOVIS.PARAM_RANGES[name]
            controls[name] = value.clamp(lo, hi)
        return controls

    def get_params(self):
        return {name: float(value.detach()) for name, value in self.mapped_controls().items()}

    def _apply_sharpness(self, image, sharpness):
        sharp_luma = _luminance(image)
        blurred_luma = F.conv2d(sharp_luma, self.sharp_kernel, padding=3)
        detail = torch.tanh((sharp_luma - blurred_luma) / (0.03 * 0.95)) * (0.03 * 0.95)
        protection = ((sharp_luma - 0.1) / 0.1).clamp(0, 1)
        sharpened_luma = torch.where(
            sharpness > 0,
            (sharp_luma + sharpness * 0.8 * detail * protection).clamp(0, 1),
            (sharp_luma * (1.0 + sharpness) + blurred_luma * (-sharpness)).clamp(0, 1),
        )
        final = image * (sharpened_luma / sharp_luma.clamp_min(1e-6))
        return _compress_gamut(final, image).clamp(0, 1)

    def _attract_to_palette_axis(self, image, amount):
        """Partially contract opponent chroma toward the retro/blockbuster axis."""
        return apply_palette_projection(
            image, amount, self.rgb_to_yiq, self.yiq_to_rgb, self.palette_axis_points,
            self.HUE_AXIS_MAX_ATTRACTION, self.HUE_AXIS_SHIFT_FRACTION,
        )

    def forward(self, image_linear):
        """Process linear RGB in float32; return linear RGB without intermediate quantization."""
        if image_linear.ndim != 4 or image_linear.shape[1] != 3:
            raise ValueError("Expected a BCHW RGB tensor")
        if not image_linear.is_floating_point():
            raise TypeError("Input must be floating point; use normalized 16-bit linear RGB")

        controls = self.mapped_controls()
        axes = self.axis_values()
        image = image_linear

        warmth = controls["wb_warmth"]
        balanced = torch.cat((image[:, 0:1] * (1.0 + warmth), image[:, 1:2],
                              image[:, 2:3] * (1.0 - warmth)), dim=1)
        image = _compress_gamut(balanced, image)

        hue_changed = self._attract_to_palette_axis(image, axes[4])
        image = _compress_gamut(hue_changed, image)

        saturation = controls["saturation"]
        luma = _luminance(image)
        chroma = image - luma
        sat_multiplier = 1.0 + saturation
        changed_chroma = sat_multiplier * chroma
        original_magnitude = torch.sqrt(chroma.square().sum(dim=1, keepdim=True) + 1e-8)
        changed_magnitude = torch.sqrt(changed_chroma.square().sum(dim=1, keepdim=True) + 1e-8)
        maximum = original_magnitude + 0.5 * (1.0 - original_magnitude)
        scale = torch.where((saturation > 0) & (changed_magnitude > maximum),
                            maximum / (changed_magnitude + 1e-8), torch.ones_like(maximum))
        saturated = luma + changed_chroma * scale
        image = _compress_gamut(saturated, image)

        brightness = controls["brightness"]
        target_srgb = (0.50 + brightness * 0.5).clamp(0.05, 0.95)
        target_linear = target_srgb.pow(2.2)
        median_luma = _luminance(image).flatten(1).median(dim=1).values[:, None, None, None]
        gamma = (torch.log(target_linear + 1e-8) / torch.log(median_luma + 1e-8)).clamp(0.3, 3.0)
        safe_luma = _luminance(image).clamp(1e-8, 1.0)
        tone_mapped = image * (safe_luma.pow(gamma) / safe_luma)
        image = _compress_gamut(tone_mapped, image)

        contrast = controls["contrast"]
        contrast_luma = _luminance(image).clamp(1e-5, 1.0 - 1e-5)
        pivot = contrast_luma.flatten(1).median(dim=1).values[:, None, None, None]
        slope = torch.pow(2.0, contrast)
        contrasted_luma = torch.sigmoid(
            slope * torch.logit(contrast_luma) + (1.0 - slope) * torch.logit(pivot)
        )
        contrasted = image * (contrasted_luma / contrast_luma)
        image = _compress_gamut(contrasted, image)

        return self._apply_sharpness(image, controls["sharpness"])
