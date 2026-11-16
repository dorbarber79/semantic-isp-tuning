"""
Experiment 10b: Three Loss Functions Compared with SigLIP-B/16

A) SEMANTIC: L = -cos(text, processed)
   "Does this image look like a {word} photo?"

B) DELTA: L = -[cos(text, processed) - cos(text, original)]
   "Did the image become MORE similar to {word} than the original?"
   Note: gradient is identical to Semantic (original is constant),
   but the score metric is different (measures improvement).

C) DIRECTIONAL (CLIPtone-style): L = -cos(processed-original, text_word-text_antonym)
   "Does the image CHANGE direction match the word-antonym text direction?"
   This is what CLIPtone (CVPR 2024) uses.

All three include beauty + realism guards + regularization.
"""

import os, sys, json, warnings, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
warnings.filterwarnings('ignore', message="Can't initialize NVML")
import open_clip

sys.path.insert(0, os.path.dirname(__file__))
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw

RESULTS_DIR = "/data/data1/semantic_isp/results/exp10b_three_losses"
DEVICE = "cuda"

WORD_PAIRS = [
    ('colorful', 'dull'),
    ('dark', 'bright'),
    ('bright', 'dim'),
    ('warm', 'cool'),
    ('cool', 'warm'),
    ('monochrome', 'colorful'),
    ('saturated', 'desaturated'),
    ('faded', 'vivid'),
    ('vivid', 'faded'),
    ('muted', 'vibrant'),
]


class SigLIPWrapper:
    def __init__(self, device):
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16-SigLIP', pretrained='webli', device=device)
        self.tokenize = open_clip.get_tokenizer('ViT-B-16-SigLIP')
        self.model.eval()
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)

    def encode_text(self, text):
        tokens = self.tokenize([text]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            return feat / feat.norm(dim=-1, keepdim=True)

    def encode_image(self, img_tensor):
        resized = F.interpolate(img_tensor, size=(224, 224), mode='bicubic', align_corners=False)
        normed = (resized - self.mean) / self.std
        feat = self.model.encode_image(normed)
        return feat / feat.norm(dim=-1, keepdim=True)


def optimize(word, antonym, img_tensor, vlm, orig_feat, beauty_feat, realism_feat,
             orig_beauty, orig_realism, orig_word_sim, text_feat, text_dir,
             mode='semantic', num_iters=500, lr=0.03):

    isp = DifferentiableEMOVIS(device=DEVICE)
    optimizer = torch.optim.Adam(isp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=lr*0.1)

    best_loss = float('inf')
    best_params = isp.get_params()
    best_delta = 0.0

    for it in range(num_iters):
        optimizer.zero_grad()
        processed = isp(img_tensor)
        proc_feat = vlm.encode_image(processed)

        proc_word_sim = (proc_feat @ text_feat.T).mean()

        if mode == 'semantic':
            # A) Maximize absolute similarity
            L_semantic = -proc_word_sim
        elif mode == 'delta':
            # B) Maximize improvement over original
            # gradient is same as semantic (orig_word_sim is constant)
            # but we track the delta as our metric
            L_semantic = -(proc_word_sim - orig_word_sim)
        elif mode == 'directional':
            # C) CLIPtone-style: direction alignment
            img_dir = proc_feat - orig_feat
            L_semantic = -(img_dir @ text_dir.T).mean()

        # Guards
        proc_beauty = (proc_feat @ beauty_feat.T).mean()
        L_beauty = F.relu(orig_beauty - proc_beauty)
        proc_realism = (proc_feat @ realism_feat.T).mean()
        L_realism = F.relu(orig_realism - proc_realism)
        L_reg = 0.003 * (isp.raw_params ** 2).sum()

        total_loss = L_semantic + 0.5 * L_beauty + 0.5 * L_realism + L_reg
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        delta = (proc_word_sim - orig_word_sim).item()
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_params = isp.get_params()
            best_delta = delta

    return best_params, best_delta


def render(img_tensor, params):
    isp = DifferentiableEMOVIS(device=DEVICE)
    with torch.no_grad():
        for pi, name in enumerate(DifferentiableEMOVIS.PARAM_NAMES):
            lo, hi = DifferentiableEMOVIS.PARAM_RANGES[name]
            mid, half = (hi+lo)/2.0, (hi-lo)/2.0
            norm = (params[name]-mid)/half
            norm = max(-0.999, min(0.999, norm))
            isp.raw_params.data[pi] = np.arctanh(norm)
        out = isp(img_tensor)
        return (out[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Exp 10b: Three Losses — SigLIP-B/16")
    print("A) Semantic  B) Delta  C) Directional (CLIPtone)")
    print("=" * 60)

    dng_path = '/data/data1/EmotionISP_dataset/dng/happy/8.dng'
    img_tensor = load_and_preprocess_raw(dng_path, target_size=512).to(DEVICE)
    vlm = SigLIPWrapper(DEVICE)

    isp_neutral = DifferentiableEMOVIS(device=DEVICE)
    with torch.no_grad():
        orig = isp_neutral(img_tensor)
        orig_feat = vlm.encode_image(orig)
        beauty_feat = vlm.encode_text("a beautiful high quality photo")
        realism_feat = vlm.encode_text("a realistic photo")
        orig_beauty = (orig_feat @ beauty_feat.T).mean()
        orig_realism = (orig_feat @ realism_feat.T).mean()

    words = [w for w, _ in WORD_PAIRS]
    modes = ['semantic', 'delta', 'directional']
    all_results = {m: {} for m in modes}

    for i, (word, antonym) in enumerate(WORD_PAIRS):
        print(f"\n[{i+1}/{len(WORD_PAIRS)}] '{word}' (vs '{antonym}')")

        # Pre-compute text features
        text_feat = vlm.encode_text(f"a {word} photo")
        anti_feat = vlm.encode_text(f"a {antonym} photo")
        with torch.no_grad():
            text_dir = text_feat - anti_feat
            text_dir = text_dir / (text_dir.norm(dim=-1, keepdim=True) + 1e-8)
            orig_word_sim = (orig_feat @ text_feat.T).mean()

        for mode in modes:
            params, delta = optimize(
                word, antonym, img_tensor, vlm, orig_feat,
                beauty_feat, realism_feat, orig_beauty, orig_realism,
                orig_word_sim, text_feat, text_dir, mode=mode)
            all_results[mode][word] = params
            p = params
            print(f"  {mode:12s}: Δsim={delta:+.4f}  wb={p['wb_warmth']:+.3f} "
                  f"sat={p['saturation']:+.3f} br={p['brightness']:+.3f} sh={p['sharpness']:+.3f}")

    # Grid: 3 rows × 11 cols
    thumb = (150, 150)
    label_h = 30
    row_label_w = 100
    cell_w, cell_h = thumb
    n_cols = len(words) + 1
    mode_labels = [
        ('Semantic', (255, 150, 150)),
        ('Delta', (255, 255, 150)),
        ('Directional', (150, 255, 150)),
    ]

    canvas = Image.new('RGB', (row_label_w + n_cols*cell_w, 3*(cell_h+label_h)), (25,25,25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 9)
        font_big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 10)
    except:
        font = font_big = ImageFont.load_default()

    orig_np = render(img_tensor, {n: 0 for n in DifferentiableEMOVIS.PARAM_NAMES})

    for row, (mode, (label, color)) in enumerate(zip(modes, mode_labels)):
        y = row * (cell_h + label_h)
        draw.text((5, y + cell_h//2 + label_h), label, fill=color, font=font_big, anchor='lm')

        canvas.paste(Image.fromarray(orig_np).resize(thumb, Image.LANCZOS),
                     (row_label_w, y + label_h))
        if row == 0:
            draw.text((row_label_w + cell_w//2, y + 3), "Original", fill=(255,255,255), font=font, anchor='mt')

        for col, word in enumerate(words):
            x = row_label_w + (col+1) * cell_w
            p = all_results[mode][word]
            img_np = render(img_tensor, p)
            canvas.paste(Image.fromarray(img_np).resize(thumb, Image.LANCZOS), (x, y + label_h))

            if row == 0:
                draw.text((x + cell_w//2, y + 3), word, fill=(200,200,200), font=font, anchor='mt')
            ps = f"s={p['saturation']:+.2f} b={p['brightness']:+.2f}"
            draw.text((x + cell_w//2, y + 14), ps, fill=(150,150,150), font=font, anchor='mt')

    out_path = os.path.join(RESULTS_DIR, 'comparison.png')
    canvas.save(out_path, quality=95)
    print(f"\nGrid: {out_path}")

    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
