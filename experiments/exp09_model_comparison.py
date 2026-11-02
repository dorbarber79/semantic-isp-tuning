"""
Experiment 09: Foundation Model Comparison for Text-Guided ISP

Compares 5 VLM models on the same image + 10 words:
1. CLIP ViT-B/32 (OpenAI) — our baseline
2. CLIP ViT-L/14 (OpenAI) — larger CLIP
3. OpenCLIP ViT-L/14 (LAION-2B) — open data, larger
4. SigLIP ViT-B/16 (WebLI) — Google's improved loss
5. SigLIP ViT-L/16 (WebLI) — larger SigLIP

Same ISP (4 params: wb, sat, bright, sharp), same loss function,
same 10 words. Only the VLM changes.
"""

import os, sys, json, warnings, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
warnings.filterwarnings('ignore', message="Can't initialize NVML")

sys.path.insert(0, os.path.dirname(__file__))
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw

import clip
import open_clip

RESULTS_DIR = "/data/data1/semantic_isp/results/exp09_models"
DEVICE = "cuda"

WORDS = ['colorful', 'dark', 'bright', 'warm', 'cool',
         'monochrome', 'saturated', 'faded', 'vivid', 'muted']

# Models to compare
MODELS = [
    ('CLIP-B/32',    'clip',      'ViT-B/32',           None),
    ('CLIP-L/14',    'clip',      'ViT-L/14',           None),
    ('OpenCLIP-L/14','open_clip', 'ViT-L-14',           'laion2b_s32b_b82k'),
    ('SigLIP-B/16',  'open_clip', 'ViT-B-16-SigLIP',    'webli'),
    ('SigLIP-L/16',  'open_clip', 'ViT-L-16-SigLIP-256','webli'),
]


class VLMWrapper:
    """Unified interface for CLIP and OpenCLIP models."""

    def __init__(self, name, backend, model_name, pretrained, device):
        self.name = name
        self.device = device

        if backend == 'clip':
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.tokenize = clip.tokenize
            self.input_res = 224
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device)
            self.tokenize = open_clip.get_tokenizer(model_name)
            # Get input resolution from model
            self.input_res = 256 if '256' in model_name else (384 if '384' in model_name else 224)

        self.model.eval()

        # Pre-compute text features
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)

    def encode_text(self, text):
        tokens = self.tokenize([text]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            return feat / feat.norm(dim=-1, keepdim=True)

    def encode_image(self, img_tensor):
        """Encode image tensor (B,3,H,W) in [0,1]."""
        resized = F.interpolate(img_tensor, size=(self.input_res, self.input_res),
                                mode='bicubic', align_corners=False)
        normed = (resized - self.clip_mean) / self.clip_std
        feat = self.model.encode_image(normed)
        return feat / feat.norm(dim=-1, keepdim=True)


def optimize_for_word(word, img_tensor, vlm, num_iters=500, lr=0.03,
                      reg_weight=0.003, quality_weight=0.5, realism_weight=0.5):
    text_feat = vlm.encode_text(f"a {word} photo")
    beauty_feat = vlm.encode_text("a beautiful high quality photo")
    realism_feat = vlm.encode_text("a realistic photo")

    isp = DifferentiableEMOVIS(device=DEVICE)

    # Original features
    with torch.no_grad():
        orig = isp(img_tensor)
        orig_feat = vlm.encode_image(orig)
        orig_beauty = (orig_feat @ beauty_feat.T).mean()
        orig_realism = (orig_feat @ realism_feat.T).mean()

    optimizer = torch.optim.Adam(isp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=lr*0.1)

    best_sim = -1
    best_params = isp.get_params()

    for it in range(num_iters):
        optimizer.zero_grad()
        processed = isp(img_tensor)
        img_feat = vlm.encode_image(processed)

        sim_target = (img_feat @ text_feat.T).mean()
        L_semantic = -sim_target

        proc_beauty = (img_feat @ beauty_feat.T).mean()
        L_beauty = F.relu(orig_beauty - proc_beauty)

        proc_realism = (img_feat @ realism_feat.T).mean()
        L_realism = F.relu(orig_realism - proc_realism)

        L_reg = reg_weight * (isp.raw_params ** 2).sum()

        total_loss = L_semantic + quality_weight * L_beauty + realism_weight * L_realism + L_reg
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if sim_target.item() > best_sim:
            best_sim = sim_target.item()
            best_params = isp.get_params()

    return best_params, best_sim


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Exp 09: Foundation Model Comparison")
    print("5 VLMs × 10 words × 1 image")
    print("=" * 60)

    # Load image
    dng_path = '/data/data1/EmotionISP_dataset/dng/happy/8.dng'
    img_tensor = load_and_preprocess_raw(dng_path, target_size=512).to(DEVICE)
    print(f"Image: 8.dng (child portrait)")

    all_results = {}

    for model_label, backend, model_name, pretrained in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_label}")
        print(f"{'='*60}")

        vlm = VLMWrapper(model_label, backend, model_name, pretrained, DEVICE)

        results = {}
        t0 = time.time()
        for i, word in enumerate(WORDS):
            params, sim = optimize_for_word(word, img_tensor, vlm)
            results[word] = {'params': params, 'clip_sim': sim}
            p = params
            print(f"  [{i+1:2d}] {word:<12} sim={sim:.4f}  wb={p['wb_warmth']:+.3f} "
                  f"sat={p['saturation']:+.3f} br={p['brightness']:+.3f} sh={p['sharpness']:+.3f}")
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.0f}s")

        all_results[model_label] = results

        # Free GPU memory
        del vlm
        torch.cuda.empty_cache()

    # Build comparison grid: rows = models, cols = words
    isp = DifferentiableEMOVIS(device=DEVICE)
    thumb = (140, 140)
    label_h = 18
    row_label_w = 100
    cell_w, cell_h = thumb
    n_cols = len(WORDS) + 1  # +1 for original
    n_rows = len(MODELS)

    canvas = Image.new('RGB', (row_label_w + n_cols*cell_w, n_rows*(cell_h+label_h)), (25,25,25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 9)
        font_big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 10)
    except:
        font = font_big = ImageFont.load_default()

    # Column headers (words)
    for col, word in enumerate(['Original'] + WORDS):
        x = row_label_w + col * cell_w
        draw.text((x + cell_w//2, 2), word, fill=(200,200,200), font=font, anchor='mt')

    for row, (model_label, _, _, _) in enumerate(MODELS):
        y = row * (cell_h + label_h)

        # Row label
        draw.text((5, y + cell_h//2 + label_h), model_label, fill=(255,200,100), font=font_big, anchor='lm')

        # Original
        with torch.no_grad():
            isp.raw_params.zero_()
            orig = isp(img_tensor)
            orig_np = (orig[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
        canvas.paste(Image.fromarray(orig_np).resize(thumb, Image.LANCZOS),
                     (row_label_w, y + label_h))

        # Each word
        for col, word in enumerate(WORDS):
            x = row_label_w + (col+1) * cell_w
            p = all_results[model_label][word]['params']
            with torch.no_grad():
                for pi, name in enumerate(DifferentiableEMOVIS.PARAM_NAMES):
                    lo, hi = DifferentiableEMOVIS.PARAM_RANGES[name]
                    mid, half = (hi+lo)/2.0, (hi-lo)/2.0
                    norm = (p[name]-mid)/half
                    norm = max(-0.999, min(0.999, norm))
                    isp.raw_params.data[pi] = np.arctanh(norm)
                processed = isp(img_tensor)
                proc_np = (processed[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
            canvas.paste(Image.fromarray(proc_np).resize(thumb, Image.LANCZOS), (x, y+label_h))

    out_path = os.path.join(RESULTS_DIR, 'model_comparison.png')
    canvas.save(out_path, quality=95)
    print(f"\nGrid: {out_path}")

    # Save
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'Model':<16}", end="")
    for word in WORDS:
        print(f" {word[:6]:>6}", end="")
    print(f"   {'avg':>5}")
    print("-" * (16 + 7*len(WORDS) + 8))
    for model_label, _, _, _ in MODELS:
        print(f"{model_label:<16}", end="")
        sims = []
        for word in WORDS:
            sim = all_results[model_label][word]['clip_sim']
            sims.append(sim)
            print(f" {sim:.3f}", end="")
        print(f"   {np.mean(sims):.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
