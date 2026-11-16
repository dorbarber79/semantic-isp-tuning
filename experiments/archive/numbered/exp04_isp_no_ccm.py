"""
Experiment 04: ISP without CCM — 6 learnable parameters

Learns: saturation, brightness, sharpness, contrast, color_bias (WB), hue_shift
Does NOT learn: CCM (fixed as identity)

This tests whether interpretable ISP knobs work better than a CCM matrix
for CLIP-guided style optimization.

Pipeline:
    Linear image → Color Bias (WB) → Hue Shift → Saturation
    → Gamma/Brightness → Contrast → Sharpen → sRGB → CLIP

6 DoF, all with clear visual meaning.
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

FIVEK_LINEAR = "/data/data1/FiveK/default_isp_linear_720p"
RESULTS_DIR = "/data/data1/semantic_isp/results/exp04_isp_no_ccm"
DEVICE = "cuda"

WORDS = ['colorful', 'dark', 'bright', 'warm', 'cool',
         'monochrome', 'saturated', 'faded', 'vivid', 'muted']

IMAGE_INDICES = [0, 500, 1500, 1800, 2200, 3500, 4500]  # All images used across experiments


class DifferentiableISP(nn.Module):
    """6-parameter ISP. No CCM.

    Params:
        0: saturation      [-1, 1]
        1: brightness       [-1, 1]
        2: sharpness        [-1, 1]
        3: contrast         [-1, 1]
        4: color_bias       [-1, 1]  warm(+) ↔ cool(-)   (red-blue axis)
        5: hue_shift        [-1, 1]  mapped to [-60, 60] degrees
    """

    PARAM_NAMES = ['saturation', 'brightness', 'sharpness',
                   'contrast', 'color_bias', 'hue_shift']

    def __init__(self, device="cuda"):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(6, device=device))
        # Gaussian kernel for sharpening
        ks = 5
        sigma = 1.5
        x = torch.arange(ks, device=device, dtype=torch.float32) - ks // 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        k2d = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('blur_kernel', k2d.expand(3, 1, -1, -1))
        self.blur_pad = ks // 2

    def get_named_params(self):
        p = self.params.detach().cpu()
        return {name: float(torch.tanh(p[i])) for i, name in enumerate(self.PARAM_NAMES)}

    def forward(self, img_linear):
        sat      = torch.tanh(self.params[0])
        bright   = torch.tanh(self.params[1])
        sharp    = torch.tanh(self.params[2])
        contrast = torch.tanh(self.params[3])
        wb       = torch.tanh(self.params[4])
        hue      = torch.tanh(self.params[5]) * 60.0  # degrees

        img = img_linear
        B, C, H, W = img.shape

        # 1. Color Bias / White Balance — in linear domain
        #    wb > 0 → warm (red up, blue down)
        #    wb < 0 → cool (blue up, red down)
        r_gain = 1.0 + wb * 0.4
        g_gain = 1.0
        b_gain = 1.0 - wb * 0.4
        img = torch.clamp(torch.cat([
            img[:, 0:1] * r_gain,
            img[:, 1:2] * g_gain,
            img[:, 2:3] * b_gain
        ], dim=1), 0, 1)

        # 2. Hue Shift — Rodrigues rotation around (1,1,1)/sqrt(3)
        angle = hue * (3.14159265 / 180.0)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        inv3 = 1.0 / 3.0
        s3 = 1.7320508
        rot = torch.stack([
            cos_a + inv3*(1-cos_a), inv3*(1-cos_a) - sin_a/s3, inv3*(1-cos_a) + sin_a/s3,
            inv3*(1-cos_a) + sin_a/s3, cos_a + inv3*(1-cos_a), inv3*(1-cos_a) - sin_a/s3,
            inv3*(1-cos_a) - sin_a/s3, inv3*(1-cos_a) + sin_a/s3, cos_a + inv3*(1-cos_a),
        ]).reshape(3, 3)
        pixels = img.reshape(B, 3, -1)
        img = torch.clamp(torch.matmul(rot, pixels).reshape(B, 3, H, W), 0, 1)

        # 3. Saturation — in linear domain
        gray = 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]
        factor = 1.0 + sat  # [0, 2]
        img = torch.clamp(gray + factor * (img - gray), 0, 1)

        # 4. Gamma + Brightness — linear → sRGB
        gamma_exp = torch.clamp(1.0 / 2.2 - bright * 0.15, 0.15, 0.85)
        img = torch.pow(torch.clamp(img, 1e-8, 1.0), gamma_exp)

        # 5. Contrast — S-curve in sRGB domain
        strength = contrast * 3.0
        img = torch.clamp(img + strength * img * (1.0 - img) * (2.0 * img - 1.0), 0, 1)

        # 6. Sharpening — in sRGB domain
        blurred = F.conv2d(img, self.blur_kernel, padding=self.blur_pad, groups=3)
        img = torch.clamp(img + sharp * 1.5 * (img - blurred), 0, 1)

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

    isp = DifferentiableISP(device=DEVICE)
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

    isp = DifferentiableISP(device=DEVICE)

    # Original
    with torch.no_grad():
        orig_srgb = torch.pow(img_tensor + 1e-8, 1.0 / 2.2)
        orig_np = (orig_srgb[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
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
    print("Experiment 04: ISP without CCM (6 DoF)")
    print("Params: saturation, brightness, sharpness,")
    print("        contrast, color_bias (WB), hue_shift")
    print("=" * 60)

    all_imgs = sorted([f for f in os.listdir(FIVEK_LINEAR) if f.endswith('.png')])

    clip_model, _ = clip.load('ViT-B/32', device=DEVICE)
    clip_model.eval()
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=DEVICE).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=DEVICE).view(1, 3, 1, 1)

    all_grids = []
    all_results = {}

    for img_idx in IMAGE_INDICES:
        img_name = all_imgs[img_idx]
        print(f"\n{'='*60}")
        print(f"Image [{img_idx}]: {img_name}")
        print(f"{'='*60}")

        img = Image.open(os.path.join(FIVEK_LINEAR, img_name)).convert('RGB')
        img = img.resize((512, 512), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

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
        grid_path = os.path.join(RESULTS_DIR, f'grid_{img_idx}.png')
        grid = render_grid(img_tensor, results, sorted_words, grid_path)
        all_grids.append((img_idx, img_name, grid))
        all_results[img_name] = results

        # Convergence plot for this image
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        for idx_w, word in enumerate(WORDS):
            ax = axes[idx_w]
            ax.plot(all_loss[word], 'b-', alpha=0.7, linewidth=0.8)
            ax2 = ax.twinx()
            ax2.plot(all_sim[word], 'r-', alpha=0.5, linewidth=0.8)
            ax2.set_ylabel('CLIP sim', color='red', fontsize=8)
            ax.set_title(f"'{word}' ({results[word]['clip_sim']:.3f})", fontsize=10)
            ax.set_xlabel('iter', fontsize=8)
            ax.set_ylabel('loss', fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.suptitle(f'Convergence — {img_name} — ISP 6 DoF (no CCM)', fontsize=13)
        plt.tight_layout()
        conv_path = os.path.join(RESULTS_DIR, f'convergence_{img_idx}.png')
        plt.savefig(conv_path, dpi=120, bbox_inches='tight')
        plt.close()

    # Combined grid
    total_h = sum(g.height for _, _, g in all_grids) + 30 * len(all_grids)
    combined = Image.new('RGB', (all_grids[0][2].width, total_h), (15, 15, 15))
    draw_c = ImageDraw.Draw(combined)
    try:
        font_t = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    except:
        font_t = ImageFont.load_default()

    y = 0
    for img_idx, img_name, grid in all_grids:
        draw_c.text((10, y + 3), f"[{img_idx}] {img_name}", fill=(255, 200, 100), font=font_t)
        y += 22
        combined.paste(grid, (0, y))
        y += grid.height + 8

    combined_path = os.path.join(RESULTS_DIR, 'all_images.png')
    combined.save(combined_path, quality=95)
    print(f"\nCombined: {combined_path}")

    # Save
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'Word':<12} ", end="")
    for img_idx, img_name, _ in all_grids:
        print(f" [{img_idx:>4}]", end="")
    print()
    print("-" * 45)
    for word in WORDS:
        print(f"{word:<12} ", end="")
        for _, img_name, _ in all_grids:
            sim = all_results[img_name][word]['clip_sim']
            print(f" {sim:.3f}", end="")
        print()

    print("\nDone!")


if __name__ == "__main__":
    main()
