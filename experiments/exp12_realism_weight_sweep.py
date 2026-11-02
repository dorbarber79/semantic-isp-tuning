"""
Experiment 12: Realism guard weight sweep (with SigLIP-B/16)

Compare semantic loss + realism guard at different weights.
No beauty guard. 10 words, child portrait.

Variants:
  A) realism=0.0 (no guard)
  B) realism=0.5
  C) realism=1.0
  D) realism=2.0
  E) realism=5.0
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

RESULTS_DIR = "/data/data1/semantic_isp/results/exp12_realism_sweep"
DEVICE = "cuda"

WORDS = ['colorful', 'dark', 'bright', 'warm', 'cool',
         'monochrome', 'saturated', 'faded', 'vivid', 'muted']

REALISM_WEIGHTS = [0.0, 0.5, 1.0, 2.0, 5.0]


class SigLIPModel:
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


def optimize(word, img_tensor, vlm, realism_feat, orig_realism,
             realism_weight=1.0, num_iters=500, lr=0.03):

    text_feat = vlm.encode_text(f"a {word} photo")

    isp = DifferentiableEMOVIS(device=DEVICE)
    optimizer = torch.optim.Adam(isp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=lr*0.1)

    best_sim = -1
    best_params = isp.get_params()

    for it in range(num_iters):
        optimizer.zero_grad()
        processed = isp(img_tensor)
        img_feat = vlm.encode_image(processed)

        # Semantic loss (weight=1.0 always)
        sim = (img_feat @ text_feat.T).mean()
        L = -sim

        # Realism guard (variable weight)
        if realism_weight > 0:
            proc_realism = (img_feat @ realism_feat.T).mean()
            L = L + realism_weight * F.relu(orig_realism - proc_realism)

        L = L + 0.003 * (isp.raw_params ** 2).sum()

        L.backward()
        optimizer.step()
        scheduler.step()

        if sim.item() > best_sim:
            best_sim = sim.item()
            best_params = isp.get_params()

    return best_params


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
    print("Exp 12: Realism weight sweep (SigLIP-B/16)")
    print(f"Weights: {REALISM_WEIGHTS}")
    print("=" * 60)

    dng_path = '/data/data1/EmotionISP_dataset/dng/happy/8.dng'
    img_tensor = load_and_preprocess_raw(dng_path, target_size=512).to(DEVICE)

    vlm = SigLIPModel(DEVICE)

    isp_neutral = DifferentiableEMOVIS(device=DEVICE)
    with torch.no_grad():
        orig = isp_neutral(img_tensor)
        orig_feat = vlm.encode_image(orig)
        realism_feat = vlm.encode_text("a realistic photo")
        orig_realism = (orig_feat @ realism_feat.T).mean()
        print(f"Original realism score: {orig_realism.item():.4f}")

    all_results = {}

    for rw in REALISM_WEIGHTS:
        label = f"r={rw:.1f}"
        print(f"\n=== {label} ===")
        results = {}
        for i, word in enumerate(WORDS):
            p = optimize(word, img_tensor, vlm, realism_feat, orig_realism,
                         realism_weight=rw)
            results[word] = p
            print(f"  [{i+1:2d}] {word:<12} wb={p['wb_warmth']:+.3f} sat={p['saturation']:+.3f} "
                  f"br={p['brightness']:+.3f} sh={p['sharpness']:+.3f}")
        all_results[label] = results

    # Grid: rows = weights, cols = Original + 10 words
    thumb = (140, 140)
    label_h = 22
    row_label_w = 60
    cell_w, cell_h = thumb
    n_cols = len(WORDS) + 1
    n_rows = len(REALISM_WEIGHTS)

    canvas = Image.new('RGB', (row_label_w + n_cols*cell_w, n_rows*(cell_h+label_h)), (25,25,25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 9)
        font_big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 10)
    except:
        font = font_big = ImageFont.load_default()

    orig_np = render(img_tensor, {n: 0 for n in DifferentiableEMOVIS.PARAM_NAMES})

    colors = [(255,100,100), (255,180,100), (255,255,100), (100,200,255), (100,255,100)]

    for row, (rw, color) in enumerate(zip(REALISM_WEIGHTS, colors)):
        label = f"r={rw:.1f}"
        y = row * (cell_h + label_h)
        draw.text((5, y + cell_h//2 + label_h), label, fill=color, font=font_big, anchor='lm')

        # Original
        canvas.paste(Image.fromarray(orig_np).resize(thumb, Image.LANCZOS),
                     (row_label_w, y + label_h))
        if row == 0:
            draw.text((row_label_w + cell_w//2, y + 3), "Original",
                      fill=(255,255,255), font=font, anchor='mt')

        for col, word in enumerate(WORDS):
            x = row_label_w + (col+1) * cell_w
            p = all_results[label][word]
            img_np = render(img_tensor, p)
            canvas.paste(Image.fromarray(img_np).resize(thumb, Image.LANCZOS), (x, y + label_h))
            if row == 0:
                draw.text((x + cell_w//2, y + 3), word, fill=(200,200,200), font=font, anchor='mt')

    out_path = os.path.join(RESULTS_DIR, 'sweep.png')
    canvas.save(out_path, quality=95)
    print(f"\nGrid: {out_path}")

    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'Word':<12}", end="")
    for rw in REALISM_WEIGHTS:
        print(f"  r={rw:.1f}  ", end="")
    print()
    print("-" * (12 + 9*len(REALISM_WEIGHTS)))
    for word in WORDS:
        print(f"{word:<12}", end="")
        for rw in REALISM_WEIGHTS:
            label = f"r={rw:.1f}"
            sat = all_results[label][word]['saturation']
            print(f"  s={sat:+.2f}", end="")
        print()

    print("\nDone!")


if __name__ == "__main__":
    main()
