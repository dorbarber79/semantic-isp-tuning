"""
Experiment 02b: CLIP-Guided ISP + CCM Optimization (Single Image)

Extends exp02 by adding a 3x3 CCM (with row-sum=1 constraint) to the
7 EMOVIS ISP parameters. Total: 7 ISP knobs + 9 CCM entries = 16 params
(13 effective DoF after row-sum constraints).

Pipeline:
    Linear → CCM (row-sum=1) → Color Bias → Hue → Saturation
    → Gamma/Brightness → Contrast → Sharpen → sRGB → CLIP
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
RESULTS_DIR = "/data/data1/semantic_isp/results/exp02b_isp_ccm"
DEVICE = "cuda"


class DifferentiableISP_CCM(nn.Module):
    """Differentiable ISP with 7 EMOVIS params + 3x3 CCM (row-sum=1).

    Total: 16 raw parameters (13 effective DoF).

    Pipeline order (matching real ISP):
        Linear → CCM → Color Bias → Hue Shift → Saturation
        → Gamma/Brightness → Contrast → Sharpen → sRGB output
    """

    PARAM_NAMES = ['saturation', 'brightness', 'sharpness', 'contrast']

    def __init__(self, device="cuda"):
        super().__init__()
        # 4 ISP knobs (color handled by CCM, no redundant WB/hue params)
        self.isp_params = nn.Parameter(torch.zeros(4, device=device))
        # 9 CCM offsets from identity (constrained to row-sum=1)
        # CCM handles: white balance, color bias, hue shift, color mixing
        self.ccm_raw = nn.Parameter(torch.zeros(3, 3, device=device))
        # Gaussian kernel for sharpening
        self._build_blur_kernel(device)

    def _build_blur_kernel(self, device):
        ks = 5
        sigma = 1.5
        x = torch.arange(ks, device=device, dtype=torch.float32) - ks // 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        k2d = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('blur_kernel', k2d.expand(3, 1, -1, -1))
        self.blur_pad = ks // 2

    def get_constrained_ccm(self):
        """Build CCM with row-sum=1 constraint."""
        ccm = torch.eye(3, device=self.ccm_raw.device) + self.ccm_raw
        row_sums = ccm.sum(dim=1, keepdim=True)
        return ccm - (row_sums - 1.0) / 3.0

    def get_all_params(self):
        """Return all params as a dict."""
        p = self.isp_params.detach().cpu()
        ccm = self.get_constrained_ccm().detach().cpu().numpy()
        result = {name: float(torch.tanh(p[i]))
                  for i, name in enumerate(self.PARAM_NAMES)}
        result['ccm'] = ccm.tolist()
        result['ccm_diag'] = [float(ccm[i, i]) for i in range(3)]
        result['num_isp_params'] = len(self.PARAM_NAMES)
        result['num_ccm_params'] = 9
        result['total_params'] = len(self.PARAM_NAMES) + 9
        result['effective_dof'] = len(self.PARAM_NAMES) + 6  # 3 row-sum constraints
        return result

    def forward(self, img_linear):
        # Map ISP params (4 knobs)
        sat       = torch.tanh(self.isp_params[0])     # [-1, 1]
        bright    = torch.tanh(self.isp_params[1])      # [-1, 1]
        sharp     = torch.tanh(self.isp_params[2])      # [-1, 1]
        contrast  = torch.tanh(self.isp_params[3])      # [-1, 1]

        img = img_linear

        # 0. CCM in linear domain (handles all color: WB, hue, color mixing)
        ccm = self.get_constrained_ccm()
        B, C, H, W = img.shape
        pixels = img.reshape(B, 3, -1)
        img = torch.clamp(torch.matmul(ccm, pixels).reshape(B, 3, H, W), 0, 1)

        # 1. Saturation
        gray = 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]
        img = torch.clamp(gray + (1.0 + sat) * (img - gray), 0, 1)

        # 4. Gamma + Brightness (linear → sRGB)
        gamma = torch.clamp(1.0 / 2.2 - bright * 0.15, 0.15, 0.85)
        img = torch.pow(torch.clamp(img, 1e-8, 1.0), gamma)

        # 5. Contrast (S-curve, always differentiable)
        strength = contrast * 3.0
        img = torch.clamp(img + strength * img * (1.0 - img) * (2.0 * img - 1.0), 0, 1)

        # 6. Sharpening
        blurred = F.conv2d(img, self.blur_kernel, padding=self.blur_pad, groups=3)
        img = torch.clamp(img + sharp * 1.5 * (img - blurred), 0, 1)

        return img


def optimize_for_word(word, image, clip_model, clip_mean, clip_std,
                      num_iters=1000, lr=0.015, reg_isp=0.008, reg_ccm=0.015):
    tok = clip.tokenize([f"a {word} photo"]).to(DEVICE)
    neutral_tok = clip.tokenize(["a photo"]).to(DEVICE)
    with torch.no_grad():
        text_feat = clip_model.encode_text(tok)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        neutral_feat = clip_model.encode_text(neutral_tok)
        neutral_feat = neutral_feat / neutral_feat.norm(dim=-1, keepdim=True)

    isp = DifferentiableISP_CCM(device=DEVICE)
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
        reg_loss = reg_isp * (isp.isp_params ** 2).sum() + \
                   reg_ccm * (isp.ccm_raw ** 2).sum()
        loss = clip_loss + reg_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        if sim_target.item() > best_sim:
            best_sim = sim_target.item()
            best_params = isp.get_all_params()

    return best_params, best_sim


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(STYLE_CORPUS) as f:
        corpus = json.load(f)

    all_words = corpus['anchors'] + corpus['seen_words'] + corpus['unseen_test_words']
    seen = set()
    words = [w.lower() for w in all_words if not (w.lower() in seen or seen.add(w.lower()))]
    print(f"Words: {len(words)}")

    # Load image
    all_imgs = sorted([f for f in os.listdir(FIVEK_LINEAR) if f.endswith('.png')])
    # Pick image by index: 0=Monument Valley, 1499=BBQ person, etc.
    img_index = int(os.environ.get('IMG_INDEX', '0'))
    img_name = all_imgs[img_index]
    img = Image.open(os.path.join(FIVEK_LINEAR, img_name)).convert('RGB')
    img = img.resize((512, 512), Image.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    print(f"Image: {img_name} (512x512, linear)")
    print(f"Parameters: 4 ISP (sat, bright, sharp, contrast) + 9 CCM (row-sum=1) = 13 params (10 effective DoF)")

    # CLIP
    clip_model, _ = clip.load('ViT-B/32', device=DEVICE)
    clip_model.eval()
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=DEVICE).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=DEVICE).view(1, 3, 1, 1)

    # Optimize
    all_results = {}
    t0 = time.time()
    for i, word in enumerate(words):
        params, sim = optimize_for_word(
            word, img_tensor, clip_model, clip_mean, clip_std,
            num_iters=1000, lr=0.015, reg_isp=0.008, reg_ccm=0.015
        )
        all_results[word] = {'params': params, 'clip_sim': sim}
        if (i + 1) % 10 == 0 or i == 0:
            p = params
            cd = p['ccm_diag']
            print(f"[{i+1:3d}/{len(words)}] '{word}' sim={sim:.4f}  "
                  f"sat={p['saturation']:+.3f} br={p['brightness']:+.3f} "
                  f"sh={p['sharpness']:+.3f} co={p['contrast']:+.3f} "
                  f"ccm=[{cd[0]:.2f},{cd[1]:.2f},{cd[2]:.2f}]")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/len(words):.1f}s/word)")

    # Save
    results_data = {'image': img_name, 'num_words': len(words),
                    'params_description': '4 ISP (sat,bright,sharp,contrast) + 9 CCM (row-sum=1) = 13 params, 10 effective DoF',
                    'words': all_results, 'total_time_sec': elapsed}
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results_data, f, indent=2)

    # =====================================================================
    # Visualization — sorted by CLIP similarity
    # =====================================================================
    print("\nGenerating grid (sorted by CLIP similarity)...")

    sorted_words = sorted(words, key=lambda w: -all_results[w]['clip_sim'])

    isp_module = DifferentiableISP_CCM(device=DEVICE)
    n_total = len(sorted_words) + 1
    n_cols = 10
    n_rows = (n_total + n_cols - 1) // n_cols
    thumb = (200, 200)
    label_h = 22
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
    canvas.paste(Image.fromarray(orig_np).resize(thumb, Image.LANCZOS), (0, label_h))
    draw.text((cell_w // 2, 3), "Original", fill=(255, 255, 255), font=font, anchor='mt')

    for idx, word in enumerate(sorted_words):
        pos = idx + 1
        col = pos % n_cols
        row = pos // n_cols
        x = col * cell_w
        y = row * (cell_h + label_h)

        p = all_results[word]['params']
        with torch.no_grad():
            isp_module.isp_params.copy_(torch.tensor([
                p['saturation'], p['brightness'], p['sharpness'],
                p['contrast']
            ], device=DEVICE))
            isp_module.ccm_raw.copy_(
                torch.tensor(p['ccm'], device=DEVICE) - torch.eye(3, device=DEVICE))
            processed = isp_module(img_tensor)
            proc_np = (processed[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        canvas.paste(Image.fromarray(proc_np).resize(thumb, Image.LANCZOS), (x, y + label_h))

        sim = all_results[word]['clip_sim']
        draw.text((x + cell_w // 2, y + 3), f"{word} {sim:.3f}",
                  fill=(100, 255, 100), font=font, anchor='mt')

    out_path = os.path.join(RESULTS_DIR, 'grid_sorted_by_sim.png')
    canvas.save(out_path, quality=95)
    print(f"Grid: {out_path} ({canvas.size[0]}x{canvas.size[1]})")

    # Summary table sorted by similarity
    print(f"\n{'Word':<16} {'Sim':>5} {'Sat':>6} {'Br':>6} {'Sh':>6} "
          f"{'Co':>6} {'CCM diag'}")
    print("-" * 80)
    for word in sorted_words:
        p = all_results[word]['params']
        s = all_results[word]['clip_sim']
        cd = p['ccm_diag']
        print(f"{word:<16} {s:.3f} {p['saturation']:+.3f} {p['brightness']:+.3f} "
              f"{p['sharpness']:+.3f} {p['contrast']:+.3f} "
              f"[{cd[0]:.2f},{cd[1]:.2f},{cd[2]:.2f}]")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
