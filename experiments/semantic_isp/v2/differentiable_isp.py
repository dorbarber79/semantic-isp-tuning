"""
Differentiable ISP matching EMOVIS — for CLIP-guided backpropagation.

Architecture:
    ┌─────────────────────────────────────────────┐
    │  NON-DIFFERENTIABLE (rawpy + OpenCV)        │
    │  RAW DNG → Demosaic (AHD) → WB → Denoise   │
    │  Output: 16-bit linear RGB as float32       │
    └─────────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────────┐
    │  DIFFERENTIABLE (PyTorch)                   │
    │  All stages below are differentiable        │
    │  Input/output in float32, 16-bit precision  │
    │                                             │
    │  1. WB fine-tune (warm ↔ cool), linear      │
    │  2. Saturation, linear                     │
    │  3. Brightness: luma gamma → gain map,      │
    │     then sRGB encoding                      │
    │  3b. Contrast: logit-linear S-curve on sRGB │
    │     luma, pivot = median (gain map)         │
    │  4. Sharpening: luminance unsharp mask      │
    │  All stages: luma-preserving RGB gamut     │
    │  compression instead of channel clipping  │
    │                                             │
    │  → 8-bit output (for the VLM)               │
    └─────────────────────────────────────────────┘

Learnable parameters (5):
    wb_warmth, saturation, brightness, sharpness, contrast
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rawpy
import cv2


# ==============================================================================
# Non-differentiable front-end: RAW → demosaic → WB → denoise
# ==============================================================================

def load_and_preprocess_raw(dng_path, target_size: int | None = 512):
    """Load RAW DNG, demosaic, WB, denoise. Returns float32 tensor [0,1] in 16-bit precision.

    This part is NOT differentiable — runs once per image.
    target_size=None retains the native post-demosaic dimensions without cropping.
    """
    # Demosaic + WB via rawpy
    with rawpy.imread(dng_path) as raw:
        rgb_16bit = raw.postprocess(
            use_camera_wb=True,
            gamma=(1, 1),              # linear output
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            use_auto_wb=False,
            half_size=False,
            highlight_mode=rawpy.HighlightMode.Clip,
        )

    img_float = rgb_16bit.astype(np.float32) / 65535.0

    # Denoise (bilateral filter, same as EMOVIS)
    denoised = cv2.bilateralFilter(img_float, 5, 0.05, 35)
    black_mask = img_float < 1e-5
    denoised[black_mask] = 0.0
    rgb_16bit = (denoised * 65535.0).astype(np.uint16)

    cropped = rgb_16bit
    if target_size is not None:
        h, w = rgb_16bit.shape[:2]
        scale = target_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(rgb_16bit, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cy, cx = new_h // 2, new_w // 2
        half = target_size // 2
        cropped = resized[cy - half:cy + half, cx - half:cx + half]

    # Convert to float32 tensor [0, 1]
    tensor = torch.from_numpy(cropped.astype(np.float32) / 65535.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    return tensor


# ==============================================================================
# Differentiable ISP — matches EMOVIS stages
# ==============================================================================

def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def _gamut_extent(chroma: torch.Tensor, luma: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(chroma.dtype).eps
    upper = chroma.amax(dim=1, keepdim=True) / (1.0 - luma).clamp_min(eps)
    lower = -chroma.amin(dim=1, keepdim=True) / luma.clamp_min(eps)
    return torch.maximum(upper, lower)


def _compress_gamut(rgb: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Keep feasible BT.601 luma and RGB chroma direction, compressing near gamut limits.

    Gamut extent 1 means a channel reaches 0 or 1. Limit new extent to 99%,
    retaining higher reference extent without increasing it. This keeps identity
    operations unchanged and prevents successive gains from consuming all headroom.
    The limiter is piecewise differentiable, like the ISP's parameter clamps.
    Infeasible luma targets become neutral black/white; roundoff alone is clamped.
    """
    luma = _luminance(rgb)
    chroma = rgb - luma
    extent = _gamut_extent(chroma, luma)
    reference_luma = _luminance(reference)
    reference_extent = _gamut_extent(reference - reference_luma, reference_luma)
    limit = reference_extent.clamp(0, 1).clamp_min(0.99)
    eps = torch.finfo(rgb.dtype).eps
    scale = limit / extent.clamp_min(eps)
    compressed = luma + chroma * scale
    out = torch.where(extent > limit, compressed, rgb)
    out = torch.where(luma <= 0, torch.zeros_like(out), out)
    out = torch.where(luma >= 1, torch.ones_like(out), out)
    return torch.clamp(out, 0, 1)


class DifferentiableEMOVIS(nn.Module):
    """Differentiable ISP matching EMOVIS stages.

    5 learnable parameters, initialized to DEFAULT_PARAMS (neutral).
    Processing in float32 throughout (16-bit precision from input).
    """

    # 'contrast' is appended last so indices of the original 4 params are unchanged.
    PARAM_NAMES = ['wb_warmth', 'saturation', 'brightness', 'sharpness', 'contrast']

    # Parameter ranges
    PARAM_RANGES = {
        'wb_warmth':        (-0.3, 0.3),   # cool(-) ↔ warm(+), applied in linear domain
        'saturation':       (-1.0, 1.0),
        'brightness':       (-0.5, 0.5),
        'sharpness':        (-1.0, 1.0),
        'contrast':         (-1.0, 1.0),   # slope at pivot k = 2^contrast  → [0.5x, 2x]
    }

    # Default (neutral, "no semantics") value of each parameter, in mapped units.
    # The ISP is initialized here, optimization starts here, and regularization pulls here.
    # Defaults = mean optimized value over all Exp 19 word-runs (10 images) whose word scores
    # level 0 ("no change") for that DoF in word_dof_expectations.csv, after the outlier
    # removal that makes presets strictly monotone in level (Exp 22, Nov 4).
    # Previous defaults: all 0 except contrast +0.40 (Exp 16).
    DEFAULT_PARAMS = {
        'wb_warmth': 0.022,
        'saturation': 0.205,
        'brightness': 0.116,
        'sharpness': -0.055,
        'contrast': 0.273,
    }

    def __init__(self, device="cuda"):
        super().__init__()
        # Raw unconstrained parameters — mapped to ranges via tanh scaling
        raw_default = torch.tensor([self._to_raw(n, self.DEFAULT_PARAMS[n]) for n in self.PARAM_NAMES],
                                   device=device, dtype=torch.float32)
        self.register_buffer('raw_default', raw_default)
        self.raw_params = nn.Parameter(raw_default.clone())

        # Gaussian kernel for sharpening (7x7, sigma=2.0, luminance-only)
        ks = 7
        sigma = 2.0
        x = torch.arange(ks, device=device, dtype=torch.float32) - ks // 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        k2d = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sharp_kernel', k2d)  # (1, 1, 7, 7)
        self.sharp_pad = ks // 2

    @classmethod
    def _to_raw(cls, name, value):
        """Inverse of the tanh mapping (mapped value → raw parameter)."""
        lo, hi = cls.PARAM_RANGES[name]
        v = (value - (hi + lo) / 2.0) / ((hi - lo) / 2.0)
        return float(np.arctanh(max(-0.999, min(0.999, v))))

    def set_params(self, params):
        """Set mapped values; parameters missing from `params` get their default."""
        with torch.no_grad():
            for i, name in enumerate(self.PARAM_NAMES):
                self.raw_params[i] = self._to_raw(name, params.get(name, self.DEFAULT_PARAMS[name]))

    def get_params(self):
        """Get mapped parameter values as dict."""
        mapped = self._map_params()
        return {name: float(mapped[i]) for i, name in enumerate(self.PARAM_NAMES)}

    def _map_params(self):
        """Map raw params to valid ranges using tanh scaling."""
        mapped = []
        for i, name in enumerate(self.PARAM_NAMES):
            lo, hi = self.PARAM_RANGES[name]
            # tanh maps to (-1, 1), scale to (lo, hi)
            val = torch.tanh(self.raw_params[i]) * (hi - lo) / 2.0 + (hi + lo) / 2.0
            mapped.append(val)
        return mapped

    def forward(self, img_linear):
        """Full differentiable ISP.

        Input:  (B, 3, H, W) float32 [0, 1], linear domain (post-demosaic, post-denoise)
        Output: (B, 3, H, W) float32 [0, 1], sRGB domain (ready for 8-bit conversion)
        """
        params = self._map_params()
        wb_warmth  = params[0]
        sat_adj    = params[1]
        brightness = params[2]
        sharpness  = params[3]
        contrast   = params[4]

        img = img_linear
        B, C, H, W = img.shape

        # ==== Stage 1: White Balance fine-tune (cool ↔ warm) ====
        # Applied in LINEAR domain, after rawpy WB.
        # Adjusts the red/blue ratio: warm(+) boosts red & reduces blue,
        # cool(-) boosts blue & reduces red. Green unchanged.
        # Uses correlated color temperature (CCT) style: move along the
        # Planckian locus approximation.
        #
        # wb_warmth in [-0.3, +0.3]:
        #   +0.3 = warm (sunset, golden hour)
        #   -0.3 = cool (shade, overcast, blue hour)
        #    0.0 = no change (keep rawpy camera WB)

        r_gain = 1.0 + wb_warmth      # warm → more red
        b_gain = 1.0 - wb_warmth      # warm → less blue
        # Green stays at 1.0 (anchor channel)
        balanced = torch.cat([
            img[:, 0:1] * r_gain,
            img[:, 1:2],
            img[:, 2:3] * b_gain,
        ], dim=1)
        img = _compress_gamut(balanced, img)

        # ==== Stage 2: Saturation with adaptive limiting and RGB gamut compression ====

        sat_mult = 1.0 + sat_adj
        gray = _luminance(img)
        chroma = img - gray

        orig_sat = torch.sqrt((chroma ** 2).sum(dim=1, keepdim=True) + 1e-8)
        new_chroma = sat_mult * chroma
        new_sat = torch.sqrt((new_chroma ** 2).sum(dim=1, keepdim=True) + 1e-8)

        # Retain EMOVIS's adaptive chroma ceiling; it is not an RGB gamut bound.
        max_sat = orig_sat + 0.5 * (1.0 - orig_sat)

        # Scale down if over the ceiling (only when boosting)
        scale = torch.where(
            (sat_adj > 0) & (new_sat > max_sat),
            max_sat / (new_sat + 1e-8),
            torch.ones_like(new_sat)
        )
        new_chroma = new_chroma * scale

        img = _compress_gamut(gray + new_chroma, img)

        # ==== Stage 2: Tone Mapping (gamma curve on luminance, applied as gain map) ====
        #
        # 1. Compute luminance
        # 2. Compute gamma so median luminance meets target brightness
        # 3. Apply gamma to luminance: lum_out = lum_in ^ gamma
        # 4. Compute per-pixel gain map: gain = lum_out / lum_in
        # 5. Apply gain to RGB: rgb_out = rgb_in * gain
        # 6. Apply sRGB gamma (linear → perceptual)
        #
        # Color ratios are preserved unless chroma must be compressed to stay in gamut.

        # Target brightness in sRGB [0,1] perceptual space
        # brightness=0 → target=0.50 (neutral mid-gray in sRGB)
        # brightness=+0.5 → target=0.75 (brighter)
        # brightness=-0.5 → target=0.25 (darker)
        target_srgb = 0.50 + brightness * 0.5
        target_srgb = torch.clamp(target_srgb, 0.05, 0.95)

        # Current luminance in linear domain
        lum = _luminance(img)
        median_lum = lum.reshape(B, -1).median(dim=1)[0].reshape(B, 1, 1, 1)

        # Convert target from sRGB to linear: target_linear = target_srgb^2.2
        target_linear = torch.pow(target_srgb, 2.2)

        # Calculate gamma: median_lum^gamma = target_linear
        # gamma = log(target_linear) / log(median_lum)
        log_target = torch.log(target_linear + 1e-8)
        log_median = torch.log(median_lum + 1e-8)
        gamma = torch.clamp(log_target / (log_median + 1e-8), 0.3, 3.0)

        # Apply gamma to luminance only
        lum_safe = torch.clamp(lum, 1e-8, 1.0)
        lum_out = torch.pow(lum_safe, gamma)

        # Gain map: per-pixel ratio
        gain_map = lum_out / lum_safe   # (B, 1, H, W)

        img = _compress_gamut(img * gain_map, img)

        # Now apply sRGB gamma (linear → sRGB perceptual)
        img = torch.where(img <= 0.0031308,
                         12.92 * img,
                         1.055 * torch.pow(torch.clamp(img, 0.0031308, 1.0), 1.0/2.4) - 0.055)
        img = torch.clamp(img, 0, 1)

        # ==== Stage 3b: Global contrast (log-odds / logit-linear curve) ====
        # Applied after brightness, on sRGB luma, via a gain map (same as tone mapping).
        #   logit(L_out) = logit(p) + k * (logit(L_in) - logit(p)),   k = 2^contrast
        #   ⇔ L_out = sigmoid(k * logit(L_in) + (1 - k) * logit(p))
        # p = median sRGB luma after brightness, so the brightness target is kept:
        #   L(p) = p, L(0) = 0, L(1) = 1, monotone, dL/dx at p = k.
        # k > 1: S-curve (more contrast), k < 1: flatter (less), k = 1: identity.
        # One formula for ±contrast; inverse is the same curve with 1/k.
        if contrast.abs() > 0 or contrast.requires_grad:
            eps = 1e-5
            lum_c = _luminance(img)
            lum_c = torch.clamp(lum_c, eps, 1.0 - eps)
            pivot = lum_c.reshape(B, -1).median(dim=1)[0].reshape(B, 1, 1, 1)
            k = torch.pow(2.0, contrast)
            lum_ct = torch.sigmoid(k * torch.logit(lum_c) + (1.0 - k) * torch.logit(pivot))
            img = _compress_gamut(img * (lum_ct / lum_c), img)

        # ==== Stage 4: Sharpening (matching EMOVIS apply_sharpening) ====
        # Luminance-domain, dark-pixel protection, soft-clip
        lum_s = _luminance(img)
        lum_safe2 = torch.clamp(lum_s, 1e-6, 1.0)

        lum_blurred = F.conv2d(lum_s, self.sharp_kernel, padding=self.sharp_pad)
        detail_raw = lum_s - lum_blurred

        # Soft clip (matching EMOVIS _soft_clip with threshold=0.03)
        threshold = 0.03
        softness = 0.95
        normalized = detail_raw / (threshold + 1e-10)
        detail = torch.tanh(normalized / softness) * softness * threshold

        # Dark-pixel protection
        protection = torch.clamp((lum_s - 0.1) / 0.1, 0, 1)

        # Amount proportional to sharpness (matching EMOVIS: amount = sigma * 0.8)
        amount = sharpness * 0.8

        # For negative sharpness → blur instead
        lum_sharp = torch.where(
            sharpness > 0,
            torch.clamp(lum_s + amount * detail * protection, 0, 1),
            # Blur: blend toward blurred
            torch.clamp(lum_s * (1.0 + sharpness) + lum_blurred * (-sharpness), 0, 1)
        )

        ratio_s = lum_sharp / lum_safe2
        img = _compress_gamut(img * ratio_s, img)

        return img
