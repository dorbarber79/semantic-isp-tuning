"""
Experiment 05: Proper Differentiable ISP from RAW (16-bit)

Loads RAW DNG → demosaic at 16-bit → differentiable ISP matching EMOVIS stages.
6 learnable parameters, proper implementations.

Pipeline (all in float32 from 16-bit source):
    RAW DNG → demosaic (rawpy, 16-bit) → [Linear float32]
    → Color Bias (saturation-weighted WB) → Hue Shift
    → Saturation (adaptive limiting) → Gamma+Brightness (sRGB)
    → Contrast (S-curve) → Sharpen (luminance-domain unsharp mask)
    → [sRGB output] → CLIP
"""

import os, json, warnings, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', message="Can't initialize NVML")
import clip
import rawpy

DNG_DIR = "/data/data1/EmotionISP_dataset/dng_filtered"
RESULTS_DIR = "/data/data1/semantic_isp/results/exp05_proper_isp_raw"
DEVICE = "cuda"

WORDS = ['colorful', 'dark', 'bright', 'warm', 'cool',
         'monochrome', 'saturated', 'faded', 'vivid', 'muted']


def load_raw_16bit(dng_path, target_size=512):
    """Load RAW DNG and demosaic at 16-bit linear precision."""
    with rawpy.imread(dng_path) as raw:
        rgb_16bit = raw.postprocess(
            use_camera_wb=True,
            gamma=(1, 1),          # linear output
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            use_auto_wb=False,
            half_size=False,
            highlight_mode=rawpy.HighlightMode.Clip,
        )
    # Convert to float32 [0, 1] preserving 16-bit precision
    img_float = rgb_16bit.astype(np.float32) / 65535.0

    # Resize
    h, w = img_float.shape[:2]
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_pil = Image.fromarray((img_float * 255).astype(np.uint8))  # for resize only
    # Use float resize
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    img_tensor = F.interpolate(img_tensor, size=(target_size, target_size),
                               mode='bicubic', align_corners=False)
    img_tensor = torch.clamp(img_tensor, 0, 1)
    return img_tensor


class ProperDifferentiableISP(nn.Module):
    """Differentiable ISP matching EMOVIS stages more faithfully.

    6 learnable parameters:
        0: saturation      [-1, 1]   Chroma scaling with adaptive limiting
        1: brightness       [-1, 1]   Gamma-based with gain control
        2: sharpness        [-1, 1]   Luminance-domain unsharp mask
        3: contrast         [-1, 1]   S-curve on luminance
        4: color_bias       [-1, 1]   Warm/cool WB with neutral preservation
        5: hue_shift        [-1, 1]   → [-60, 60] degrees

    All processing in float32 from 16-bit source. No quantization until output.
    """

    PARAM_NAMES = ['saturation', 'brightness', 'sharpness',
                   'contrast', 'color_bias', 'hue_shift']

    def __init__(self, device="cuda"):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(6, device=device))

        # Gaussian kernel for sharpening (luminance-domain)
        ks = 7
        sigma = 2.0
        x = torch.arange(ks, device=device, dtype=torch.float32) - ks // 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        k2d = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        # Single-channel kernel for luminance
        self.register_buffer('blur_kernel_1ch', k2d)
        self.blur_pad = ks // 2

    def get_named_params(self):
        p = self.params.detach().cpu()
        return {name: float(torch.tanh(p[i])) for i, name in enumerate(self.PARAM_NAMES)}

    def forward(self, img_linear):
        """Full pipeline. Input: linear float32 [0,1]. Output: sRGB [0,1]."""
        sat      = torch.tanh(self.params[0])
        bright   = torch.tanh(self.params[1])
        sharp    = torch.tanh(self.params[2])
        contrast = torch.tanh(self.params[3])
        wb       = torch.tanh(self.params[4])
        hue      = torch.tanh(self.params[5]) * 60.0

        img = img_linear
        B, C, H, W = img.shape

        # ---- Stage 1: Color Bias with neutral-pixel preservation ----
        # Matching EMOVIS: saturation-weighted WB so neutral pixels stay neutral
        max_rgb = img.max(dim=1, keepdim=True)[0]
        min_rgb = img.min(dim=1, keepdim=True)[0]
        pixel_sat = (max_rgb - min_rgb) / (max_rgb + 1e-6)
        sat_weight = torch.pow(pixel_sat, 0.5)  # sqrt for gentle transition

        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        # Apply WB gains
        r_biased = r * (1.0 + wb * 0.4)
        b_biased = b * (1.0 - wb * 0.4)
        img_biased = torch.cat([r_biased, g, b_biased], dim=1)
        # Blend: neutral pixels keep original, saturated pixels get bias
        img = img * (1.0 - sat_weight) + img_biased * sat_weight
        img = torch.clamp(img, 0, 1)

        # ---- Stage 2: Hue Shift ----
        angle = hue * (3.14159265 / 180.0)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        inv3, s3 = 1.0 / 3.0, 1.7320508
        rot = torch.stack([
            cos_a + inv3*(1-cos_a), inv3*(1-cos_a) - sin_a/s3, inv3*(1-cos_a) + sin_a/s3,
            inv3*(1-cos_a) + sin_a/s3, cos_a + inv3*(1-cos_a), inv3*(1-cos_a) - sin_a/s3,
            inv3*(1-cos_a) - sin_a/s3, inv3*(1-cos_a) + sin_a/s3, cos_a + inv3*(1-cos_a),
        ]).reshape(3, 3)
        pixels = img.reshape(B, 3, -1)
        img = torch.clamp(torch.matmul(rot, pixels).reshape(B, 3, H, W), 0, 1)

        # ---- Stage 3: Saturation with adaptive limiting ----
        # Matching EMOVIS: limit chroma boost to avoid over-saturation
        gray = 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]
        chroma = img - gray
        sat_mult = 1.0 + sat

        # Adaptive limit: don't let chroma exceed headroom
        orig_chroma_mag = torch.sqrt((chroma ** 2).sum(dim=1, keepdim=True) + 1e-8)
        new_chroma = chroma * sat_mult
        new_chroma_mag = torch.sqrt((new_chroma ** 2).sum(dim=1, keepdim=True) + 1e-8)

        boost = 0.5 if sat.item() > 0 else 1.0
        max_chroma = orig_chroma_mag + (1.0 - orig_chroma_mag) * boost
        scale = torch.where(new_chroma_mag > max_chroma,
                           max_chroma / (new_chroma_mag + 1e-8),
                           torch.ones_like(new_chroma_mag))
        new_chroma = new_chroma * scale

        img = torch.clamp(gray + new_chroma, 0, 1)

        # ---- Stage 4: Gamma + Brightness ----
        # Matching EMOVIS: gain-based brightness + sRGB gamma
        brightness_mult = 1.0 + bright * 1.5  # [0.25, 2.5]

        # Apply gain before gamma (like EMOVIS face-aware gain)
        lum = 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]
        median_lum = lum.median()
        target_lum = 0.18 * brightness_mult  # 0.18 = mid-gray in linear
        gain = torch.clamp(target_lum / (median_lum + 1e-6), 0.2, 4.0)
        img = torch.clamp(img * gain, 0, 1)

        # sRGB gamma (proper piecewise)
        img = torch.where(img <= 0.0031308,
                         12.92 * img,
                         1.055 * torch.pow(torch.clamp(img, 0.0031308, 1.0), 1.0/2.4) - 0.055)
        img = torch.clamp(img, 0, 1)

        # ---- Stage 5: Contrast (S-curve on luminance) ----
        # Matching EMOVIS: luminance-preserving contrast
        lum_gamma = 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]
        lum_safe = torch.clamp(lum_gamma, 1e-6, 1.0)

        strength = contrast * 1.5
        lum_adj = torch.clamp((lum_gamma - 0.5) * (1.0 + strength) + 0.5, 0, 1)

        # Apply via luminance ratio to preserve color
        ratio = lum_adj / lum_safe
        img = torch.clamp(img * ratio, 0, 1)

        # ---- Stage 6: Sharpening in luminance domain ----
        # Matching EMOVIS: luminance-only unsharp mask with dark-pixel protection
        lum_sharp = 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]
        lum_safe2 = torch.clamp(lum_sharp, 1e-6, 1.0)

        lum_blurred = F.conv2d(lum_sharp, self.blur_kernel_1ch,
                               padding=self.blur_pad)
        detail = lum_sharp - lum_blurred

        # Dark-pixel protection (like EMOVIS)
        protection = torch.clamp((lum_sharp - 0.1) / 0.1, 0, 1)

        amount = sharp * 1.5
        lum_sharpened = torch.clamp(lum_sharp + amount * detail * protection, 0, 1)

        # Apply via luminance ratio
        ratio_s = lum_sharpened / lum_safe2
        img = torch.clamp(img * ratio_s, 0, 1)

        return img


def optimize_for_word(word, img_tensor, clip_model, clip_mean, clip_std,
                      num_iters=500, lr=0.02, reg_weight=0.005):
    tok = clip.tokenize([f"a {word} photo"]).to(DEVICE)
    neutral_tok = clip.tokenize(["a photo"]).to(DEVICE)
    with torch.no_grad():
        text_feat = clip_model.encode_text(tok)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        neutral_feat = clip_model.encode_text(neutral_tok)
        neutral_feat = neutral_feat / neutral_feat.norm(dim=-1, keepdim=True)

    isp = ProperDifferentiableISP(device=DEVICE)
    optimizer = torch.optim.Adam(isp.parameters(), lr=lr)

    loss_history = []
    sim_history = []
    best_sim = -1
    best_params = None

    for it in range(num_iters):
        optimizer.zero_grad()
        processed = isp(img_tensor)

        resized = F.interpolate(processed, size=(224, 224), mode='bicubic', align_corners=False)
        normed = (resized - clip_mean) / clip_std
        img_feat = clip_model.encode_image(normed)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sim_target = (img_feat @ text_feat.T).mean()
        sim_neutral = (img_feat @ neutral_feat.T).mean()
        clip_loss = -(sim_target - 0.5 * sim_neutral)
        reg_loss = reg_weight * (isp.params ** 2).sum()
        total_loss = clip_loss + reg_loss

        total_loss.backward()
        optimizer.step()

        loss_history.append(total_loss.item())
        sim_history.append(sim_target.item())

        if sim_target.item() > best_sim:
            best_sim = sim_target.item()
            best_params = isp.get_named_params()

    return best_params, best_sim, loss_history, sim_history


def render_grid(img_tensor, results, sorted_words, out_path):
    n_cols = 6
    n_rows = 2
    thumb = (256, 256)
    label_h = 24
    cell_w, cell_h = thumb

    canvas = Image.new('RGB', (n_cols * cell_w, n_rows * (cell_h + label_h)), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 11)
    except:
        font = ImageFont.load_default()

    isp = ProperDifferentiableISP(device=DEVICE)

    with torch.no_grad():
        orig_proc = isp(img_tensor)  # neutral ISP (all params=0)
        orig_np = (orig_proc[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    canvas.paste(Image.fromarray(orig_np).resize(thumb, Image.LANCZOS), (0, label_h))
    draw.text((cell_w // 2, 4), "Original", fill=(255, 255, 255), font=font, anchor='mt')

    for idx, word in enumerate(sorted_words):
        pos = idx + 1
        col = pos % n_cols
        row = pos // n_cols
        x = col * cell_w
        y = row * (cell_h + label_h)

        p = results[word]['params']
        with torch.no_grad():
            isp.params.copy_(torch.tensor([
                p['saturation'], p['brightness'], p['sharpness'],
                p['contrast'], p['color_bias'], p['hue_shift']
            ], device=DEVICE))
            processed = isp(img_tensor)
            proc_np = (processed[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        canvas.paste(Image.fromarray(proc_np).resize(thumb, Image.LANCZOS), (x, y + label_h))
        sim = results[word]['clip_sim']
        draw.text((x + cell_w // 2, y + 4), f"{word} ({sim:.3f})",
                  fill=(100, 255, 100), font=font, anchor='mt')

    canvas.save(out_path, quality=95)
    return canvas


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Experiment 05: Proper ISP from RAW (16-bit)")
    print("6 DoF: sat, bright, sharp, contrast, WB, hue")
    print("Proper EMOVIS-style stages, RAW DNG input")
    print("=" * 60)

    # Find DNG files
    all_dngs = sorted([f for f in os.listdir(DNG_DIR) if f.endswith('.dng')])
    print(f"Available DNGs: {len(all_dngs)}")

    # Pick 4 diverse images
    pick_indices = [0, 20, 50, 75]
    selected_dngs = [all_dngs[i] for i in pick_indices]

    clip_model, _ = clip.load('ViT-B/32', device=DEVICE)
    clip_model.eval()
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=DEVICE).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=DEVICE).view(1, 3, 1, 1)

    all_grids = []
    all_results = {}

    for dng_name in selected_dngs:
        dng_path = os.path.join(DNG_DIR, dng_name)
        print(f"\n{'='*60}")
        print(f"RAW: {dng_name}")
        print(f"{'='*60}")

        # Load RAW at 16-bit, convert to float32
        img_tensor = load_raw_16bit(dng_path, target_size=512).to(DEVICE)
        print(f"  Loaded: {img_tensor.shape}, range [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")

        results = {}
        all_loss = {}
        all_sim = {}

        for i, word in enumerate(WORDS):
            print(f"  [{i+1}/{len(WORDS)}] '{word}'...", end=" ", flush=True)
            params, sim, loss_h, sim_h = optimize_for_word(
                word, img_tensor, clip_model, clip_mean, clip_std,
                num_iters=500, lr=0.02, reg_weight=0.005
            )
            results[word] = {'params': params, 'clip_sim': sim}
            all_loss[word] = loss_h
            all_sim[word] = sim_h
            p = params
            print(f"sim={sim:.4f}  sat={p['saturation']:+.3f} br={p['brightness']:+.3f} "
                  f"sh={p['sharpness']:+.3f} co={p['contrast']:+.3f} "
                  f"wb={p['color_bias']:+.3f} hue={p['hue_shift']:+.3f}")

        sorted_words = sorted(WORDS, key=lambda w: -results[w]['clip_sim'])

        # Grid
        grid_path = os.path.join(RESULTS_DIR, f'grid_{dng_name.replace(".dng","")}.png')
        grid = render_grid(img_tensor, results, sorted_words, grid_path)
        all_grids.append((dng_name, grid))
        all_results[dng_name] = results

        # Convergence
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for idx_w, word in enumerate(WORDS):
            ax = axes.flatten()[idx_w]
            ax.plot(all_loss[word], 'b-', alpha=0.7, linewidth=0.8)
            ax2 = ax.twinx()
            ax2.plot(all_sim[word], 'r-', alpha=0.5, linewidth=0.8)
            ax2.set_ylabel('sim', color='red', fontsize=8)
            ax.set_title(f"'{word}' ({results[word]['clip_sim']:.3f})", fontsize=10)
            ax.grid(True, alpha=0.3)
        plt.suptitle(f'Convergence — {dng_name} — Proper ISP from RAW', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'conv_{dng_name.replace(".dng","")}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()

    # Combined grid
    total_h = sum(g.height for _, g in all_grids) + 30 * len(all_grids)
    combined = Image.new('RGB', (all_grids[0][1].width, total_h), (15, 15, 15))
    draw_c = ImageDraw.Draw(combined)
    try:
        font_t = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    except:
        font_t = ImageFont.load_default()

    y = 0
    for dng_name, grid in all_grids:
        draw_c.text((10, y + 3), f"RAW: {dng_name}", fill=(255, 200, 100), font=font_t)
        y += 22
        combined.paste(grid, (0, y))
        y += grid.height + 8

    combined_path = os.path.join(RESULTS_DIR, 'all_images.png')
    combined.save(combined_path, quality=95)
    print(f"\nCombined: {combined_path}")

    # Save
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump({
            'description': 'Proper differentiable ISP from RAW DNG (16-bit input)',
            'isp_stages': [
                '1. Color Bias (saturation-weighted WB, neutral preservation)',
                '2. Hue Shift (Rodrigues rotation)',
                '3. Saturation (adaptive chroma limiting)',
                '4. Gamma+Brightness (gain + proper sRGB piecewise gamma)',
                '5. Contrast (luminance-preserving S-curve)',
                '6. Sharpening (luminance-domain unsharp mask, dark-pixel protection)',
            ],
            'params': '6 DoF: saturation, brightness, sharpness, contrast, color_bias, hue_shift',
            'input': 'RAW DNG → rawpy demosaic (AHD, 16-bit) → float32',
            'images': {name: res for name, res in all_results.items()},
        }, f, indent=2)

    # Summary
    print(f"\n{'Word':<12}", end="")
    for name, _ in all_grids:
        print(f"  {name[:8]:>8}", end="")
    print()
    print("-" * 55)
    for word in WORDS:
        print(f"{word:<12}", end="")
        for name, _ in all_grids:
            sim = all_results[name][word]['clip_sim']
            print(f"    {sim:.3f}", end="")
        print()

    print("\nDone!")


if __name__ == "__main__":
    main()
