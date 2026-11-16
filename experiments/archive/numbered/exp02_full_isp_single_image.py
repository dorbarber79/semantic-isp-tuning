"""
Experiment 02: CLIP-Guided Full ISP Optimization (Single Image)

Uses differentiable approximations of EMOVIS ISP stages to optimize
7 ISP parameters per style word via backpropagation (Mayer et al. approach).

Pipeline (differentiable):
    Linear image → Color Bias → Hue Shift → Saturation → Brightness/Gamma
    → Contrast → Sharpen → Output sRGB → CLIP

Usage:
    python exp02_full_isp_single_image.py
"""

import os, json, warnings, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
warnings.filterwarnings('ignore', message="Can't initialize NVML")
import clip

FIVEK_LINEAR = "/data/data1/FiveK/default_isp_linear_720p"
STYLE_CORPUS = "/data/data1/semantic_isp/datasets/style_words/style_corpus.json"
RESULTS_DIR = "/data/data1/semantic_isp/results/exp02_full_isp"
DEVICE = "cuda"


# ==============================================================================
# Differentiable ISP Stages (PyTorch, matching EMOVIS semantics)
# ==============================================================================

class DifferentiableISP(nn.Module):
    """Differentiable ISP with 7 parameters matching EMOVIS.

    All operations are fully differentiable — no early returns, no
    non-differentiable branches. Gradients always flow.
    """

    PARAM_NAMES = ['saturation', 'brightness', 'sharpness',
                   'red_green_bias', 'yellow_blue_bias', 'contrast', 'hue_shift']

    def __init__(self, device="cuda"):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(7, device=device))
        # Pre-compute Gaussian kernel for sharpening (fixed, not learned)
        self._build_blur_kernel(device)

    def _build_blur_kernel(self, device):
        """Pre-build Gaussian kernel for sharpening/blur."""
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
        """Apply full ISP. Input: linear [0,1]. Output: sRGB [0,1]."""
        # Map raw params to bounded ranges via tanh
        sat       = torch.tanh(self.params[0])            # [-1, 1]
        bright    = torch.tanh(self.params[1])            # [-1, 1]
        sharp     = torch.tanh(self.params[2])            # [-1, 1]
        rg_bias   = torch.tanh(self.params[3])            # [-1, 1]
        yb_bias   = torch.tanh(self.params[4])            # [-1, 1]
        contrast  = torch.tanh(self.params[5])            # [-1, 1]
        hue       = torch.tanh(self.params[6]) * 60.0     # [-60, 60] degrees

        img = img_linear

        # 1. Color Bias — in linear domain (white balance shift)
        img = self._color_bias(img, rg_bias, yb_bias)

        # 2. Hue Shift — in linear domain
        img = self._hue_shift(img, hue)

        # 3. Saturation — in linear domain
        img = self._saturation(img, sat)

        # 4. Gamma + Brightness — linear → sRGB
        img = self._brightness_gamma(img, bright)

        # 5. Contrast — in sRGB (perceptual) domain
        img = self._contrast(img, contrast)

        # 6. Sharpening — in sRGB domain
        img = self._sharpening(img, sharp)

        return torch.clamp(img, 0, 1)

    def _color_bias(self, img, rg_bias, yb_bias):
        """White balance / color temperature shift.

        Symmetric: positive rg = warm (red up, blue down),
                   positive yb = yellow (red+green up, blue down).
        Applied as additive offset on log-domain for symmetry.
        """
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]

        # Red-green axis: warm ↔ cool
        r_gain = 1.0 + rg_bias * 0.25
        g_gain = 1.0
        b_gain = 1.0 - rg_bias * 0.25

        # Yellow-blue axis: warm ↔ cool
        r_gain = r_gain * (1.0 + yb_bias * 0.15)
        g_gain = g_gain * (1.0 + yb_bias * 0.10)
        b_gain = b_gain * (1.0 - yb_bias * 0.20)

        out = torch.cat([r * r_gain, g * g_gain, b * b_gain], dim=1)
        return torch.clamp(out, 0, 1)

    def _hue_shift(self, img, degrees):
        """Hue rotation via Rodrigues rotation around (1,1,1)/sqrt(3) in RGB."""
        angle = degrees * (3.14159265 / 180.0)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        inv3 = 1.0 / 3.0
        s3 = 1.7320508  # sqrt(3)

        # Rotation matrix around (1,1,1)/sqrt(3)
        rot = torch.stack([
            cos_a + inv3 * (1 - cos_a),
            inv3 * (1 - cos_a) - sin_a / s3,
            inv3 * (1 - cos_a) + sin_a / s3,
            inv3 * (1 - cos_a) + sin_a / s3,
            cos_a + inv3 * (1 - cos_a),
            inv3 * (1 - cos_a) - sin_a / s3,
            inv3 * (1 - cos_a) - sin_a / s3,
            inv3 * (1 - cos_a) + sin_a / s3,
            cos_a + inv3 * (1 - cos_a),
        ]).reshape(3, 3)

        B, C, H, W = img.shape
        pixels = img.reshape(B, 3, -1)
        rotated = torch.matmul(rot, pixels)
        return torch.clamp(rotated.reshape(B, 3, H, W), 0, 1)

    def _saturation(self, img, sat_adjust):
        """Saturation: interpolate between gray and original.
        sat=-1 → grayscale, sat=0 → unchanged, sat=+1 → 2x chroma.
        """
        gray = 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]
        factor = 1.0 + sat_adjust  # [0, 2]
        return torch.clamp(gray + factor * (img - gray), 0, 1)

    def _brightness_gamma(self, img, brightness):
        """Linear → sRGB with brightness-adjusted gamma.
        brightness=0 → standard gamma 1/2.2
        brightness>0 → brighter (lower gamma exponent)
        brightness<0 → darker (higher gamma exponent)
        """
        # gamma exponent: 0.45 (neutral) adjusted by brightness
        gamma = 1.0 / 2.2 - brightness * 0.15
        gamma = torch.clamp(gamma, 0.15, 0.85)
        return torch.pow(torch.clamp(img, 1e-8, 1.0), gamma)

    def _contrast(self, img, contrast):
        """S-curve contrast. Always differentiable, no branches.
        contrast>0 → increase contrast (push away from 0.5)
        contrast<0 → decrease contrast (push toward 0.5)
        """
        # Smooth S-curve using sigmoid reparametrization
        strength = contrast * 3.0  # scale for visible effect
        # S-curve: sigmoid-based remapping centered at 0.5
        # At strength=0 this is identity
        x = img
        # Soft S-curve: x' = sigmoid(strength * (x - 0.5)) remapped to [0,1]
        # When strength→0, sigmoid(0*(x-0.5))=0.5, but we want identity.
        # Use: x' = x + strength * x * (1-x) * (x - 0.5)
        # This is always differentiable and identity at strength=0.
        adjusted = x + strength * x * (1.0 - x) * (2.0 * x - 1.0)
        return torch.clamp(adjusted, 0, 1)

    def _sharpening(self, img, sharp):
        """Unsharp mask / blur. Always applied (no early return).
        sharp>0 → sharpen, sharp<0 → blur, sharp=0 → identity.
        """
        blurred = F.conv2d(img, self.blur_kernel, padding=self.blur_pad, groups=3)
        detail = img - blurred
        # sharp in [-1, 1]: positive adds detail, negative removes it
        amount = sharp * 1.5
        return torch.clamp(img + amount * detail, 0, 1)


# ==============================================================================
# Optimization
# ==============================================================================

def optimize_isp_for_word(word, image, clip_model, clip_mean, clip_std,
                          num_iters=800, lr=0.02, reg_weight=0.008):
    """Optimize 7 ISP params for a single (word, image) pair."""
    tok = clip.tokenize([f"a {word} photo"]).to(DEVICE)
    neutral_tok = clip.tokenize(["a photo"]).to(DEVICE)
    with torch.no_grad():
        text_feat = clip_model.encode_text(tok)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        neutral_feat = clip_model.encode_text(neutral_tok)
        neutral_feat = neutral_feat / neutral_feat.norm(dim=-1, keepdim=True)

    isp = DifferentiableISP(device=DEVICE)
    optimizer = torch.optim.Adam(isp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=lr * 0.1)

    best_sim = -1
    best_params = None

    for it in range(num_iters):
        optimizer.zero_grad()
        processed = isp(image)

        resized = F.interpolate(processed, size=(224, 224),
                                mode='bicubic', align_corners=False)
        normed = (resized - clip_mean) / clip_std
        img_feat = clip_model.encode_image(normed)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sim_target = (img_feat @ text_feat.T).mean()
        sim_neutral = (img_feat @ neutral_feat.T).mean()

        clip_loss = -(sim_target - 0.5 * sim_neutral)
        reg_loss = reg_weight * (isp.params ** 2).sum()
        loss = clip_loss + reg_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        if sim_target.item() > best_sim:
            best_sim = sim_target.item()
            best_params = isp.get_named_params()

    return best_params, best_sim


# ==============================================================================
# Main
# ==============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load FULL style corpus
    with open(STYLE_CORPUS) as f:
        corpus = json.load(f)

    # All single words (no compound phrases or robustness tests)
    all_words = corpus['anchors'] + corpus['seen_words'] + corpus['unseen_test_words']
    seen = set()
    words = [w.lower() for w in all_words if not (w.lower() in seen or seen.add(w.lower()))]

    print(f"Words to optimize: {len(words)}")

    # Load ONE image (linear domain)
    all_imgs = sorted([f for f in os.listdir(FIVEK_LINEAR) if f.endswith('.png')])
    img_name = all_imgs[0]  # a0001 - Monument Valley
    img_path = os.path.join(FIVEK_LINEAR, img_name)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((512, 512), Image.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    print(f"Image: {img_name} (512x512, linear domain)")

    # Load CLIP
    clip_model, _ = clip.load('ViT-B/32', device=DEVICE)
    clip_model.eval()
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=DEVICE).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=DEVICE).view(1, 3, 1, 1)

    # Optimize per word
    all_results = {}
    t0 = time.time()

    for i, word in enumerate(words):
        params, sim = optimize_isp_for_word(
            word, img_tensor, clip_model, clip_mean, clip_std,
            num_iters=800, lr=0.02, reg_weight=0.008
        )
        all_results[word] = {'params': params, 'clip_sim': sim}
        p = params
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1:3d}/{len(words)}] '{word}'  sim={sim:.4f}  "
                  f"sat={p['saturation']:+.3f} br={p['brightness']:+.3f} "
                  f"sh={p['sharpness']:+.3f} rg={p['red_green_bias']:+.3f} "
                  f"yb={p['yellow_blue_bias']:+.3f} co={p['contrast']:+.3f} "
                  f"hu={p['hue_shift']:+.3f}")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/len(words):.1f}s/word)")

    # Save results
    with open(os.path.join(RESULTS_DIR, 'results_all_words.json'), 'w') as f:
        json.dump({'image': img_name, 'num_words': len(words),
                   'words': all_results, 'total_time_sec': elapsed}, f, indent=2)

    # =========================================================================
    # Generate visualization grid — show ALL words
    # =========================================================================
    print("\nGenerating visualization grid...")

    isp_module = DifferentiableISP(device=DEVICE)
    n_total = len(words) + 1  # +1 for original
    n_cols = 10
    n_rows = (n_total + n_cols - 1) // n_cols
    thumb = (200, 200)
    label_h = 20
    cell_w, cell_h = thumb

    canvas = Image.new('RGB', (n_cols * cell_w, n_rows * (cell_h + label_h)), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
    except:
        font = ImageFont.load_default()

    # Original
    with torch.no_grad():
        orig_srgb = torch.pow(img_tensor + 1e-8, 1.0 / 2.2)
        orig_np = (orig_srgb[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    orig_pil = Image.fromarray(orig_np).resize(thumb, Image.LANCZOS)
    canvas.paste(orig_pil, (0, label_h))
    draw.text((cell_w // 2, 2), "Original", fill=(255, 255, 255), font=font, anchor='mt')

    for idx, word in enumerate(words):
        pos = idx + 1
        col = pos % n_cols
        row = pos // n_cols
        x = col * cell_w
        y = row * (cell_h + label_h)

        p = all_results[word]['params']
        with torch.no_grad():
            isp_module.params.copy_(torch.tensor([
                p['saturation'], p['brightness'], p['sharpness'],
                p['red_green_bias'], p['yellow_blue_bias'],
                p['contrast'], p['hue_shift']
            ], device=DEVICE))
            processed = isp_module(img_tensor)
            proc_np = (processed[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        proc_pil = Image.fromarray(proc_np).resize(thumb, Image.LANCZOS)
        canvas.paste(proc_pil, (x, y + label_h))

        sim = all_results[word]['clip_sim']
        draw.text((x + cell_w // 2, y + 2), f"{word}", fill=(100, 255, 100),
                  font=font, anchor='mt')

    out_path = os.path.join(RESULTS_DIR, 'isp_grid_all_words.png')
    canvas.save(out_path, quality=95)
    print(f"Grid: {out_path} ({canvas.size[0]}x{canvas.size[1]})")

    # =========================================================================
    # Summary table
    # =========================================================================
    print(f"\n{'Word':<16} {'CLIP':>5} {'Sat':>6} {'Br':>6} {'Sh':>6} "
          f"{'R/G':>6} {'Y/B':>6} {'Co':>6} {'Hue':>6}")
    print("-" * 85)
    for word in sorted(all_results.keys(), key=lambda w: -all_results[w]['clip_sim']):
        p = all_results[word]['params']
        s = all_results[word]['clip_sim']
        print(f"{word:<16} {s:.3f} {p['saturation']:+.3f} {p['brightness']:+.3f} "
              f"{p['sharpness']:+.3f} {p['red_green_bias']:+.3f} "
              f"{p['yellow_blue_bias']:+.3f} {p['contrast']:+.3f} "
              f"{p['hue_shift']:+.3f}")

    print(f"\nDone! Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
