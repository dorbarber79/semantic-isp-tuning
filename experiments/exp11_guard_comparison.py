"""
Experiment 11: Compare guard combinations

4 variants, all with semantic loss (weight=1.0):
A) Semantic only           — no guards
B) Semantic + Beauty       — beauty guard only
C) Semantic + Realism      — realism guard only
D) Semantic + Both         — beauty + realism guards

10 words, 1 image (child portrait), CLIP ViT-B/32.
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

RESULTS_DIR = "/data/data1/semantic_isp/results/exp11_guards"
DEVICE = "cuda"

WORDS = ['colorful', 'dark', 'bright', 'warm', 'cool',
         'monochrome', 'saturated', 'faded', 'vivid', 'muted']


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


def optimize(word, img_tensor, vlm,
             orig_feat, beauty_feat, realism_feat, orig_beauty, orig_realism,
             use_beauty=True, use_realism=True,
             num_iters=500, lr=0.03):

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

        sim = (img_feat @ text_feat.T).mean()
        L = -sim

        if use_beauty:
            proc_beauty = (img_feat @ beauty_feat.T).mean()
            L = L + F.relu(orig_beauty - proc_beauty)

        if use_realism:
            proc_realism = (img_feat @ realism_feat.T).mean()
            L = L + F.relu(orig_realism - proc_realism)

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
    print("Exp 11: Guard Comparison")
    print("A) No guards  B) +Beauty  C) +Realism  D) +Both")
    print("=" * 60)

    dng_path = '/data/data1/EmotionISP_dataset/dng/happy/8.dng'
    img_tensor = load_and_preprocess_raw(dng_path, target_size=512).to(DEVICE)

    vlm = SigLIPModel(DEVICE)
    print("Model: SigLIP ViT-B/16 (WebLI)")

    isp_neutral = DifferentiableEMOVIS(device=DEVICE)
    with torch.no_grad():
        orig = isp_neutral(img_tensor)
        orig_feat = vlm.encode_image(orig)
        beauty_feat = vlm.encode_text("a beautiful high quality photo")
        realism_feat = vlm.encode_text("a realistic photo")
        orig_beauty = (orig_feat @ beauty_feat.T).mean()
        orig_realism = (orig_feat @ realism_feat.T).mean()
        print(f"Original beauty: {orig_beauty.item():.4f}, realism: {orig_realism.item():.4f}")

    variants = [
        ('No guards',   False, False),
        ('+Beauty',     True,  False),
        ('+Realism',    False, True),
        ('+Both',       True,  True),
    ]

    all_results = {}
    for vname, use_b, use_r in variants:
        print(f"\n=== {vname} ===")
        results = {}
        for i, word in enumerate(WORDS):
            p = optimize(word, img_tensor, vlm,
                         orig_feat, beauty_feat, realism_feat, orig_beauty, orig_realism,
                         use_beauty=use_b, use_realism=use_r)
            results[word] = p
            print(f"  [{i+1}] {word:<12} wb={p['wb_warmth']:+.3f} sat={p['saturation']:+.3f} "
                  f"br={p['brightness']:+.3f} sh={p['sharpness']:+.3f}")
        all_results[vname] = results

    # Grid: 4 rows × 11 cols (Original + 10 words)
    thumb = (150, 150)
    label_h = 22
    row_label_w = 80
    cell_w, cell_h = thumb
    n_cols = len(WORDS) + 1

    canvas = Image.new('RGB', (row_label_w + n_cols*cell_w, 4*(cell_h+label_h)), (25,25,25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 9)
        font_big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 10)
    except:
        font = font_big = ImageFont.load_default()

    orig_np = render(img_tensor, {n: 0 for n in DifferentiableEMOVIS.PARAM_NAMES})
    colors = [(255,100,100), (255,200,100), (100,200,255), (100,255,100)]

    for row, ((vname, _, _), color) in enumerate(zip(variants, colors)):
        y = row * (cell_h + label_h)
        draw.text((5, y + cell_h//2 + label_h), vname, fill=color, font=font_big, anchor='lm')

        # Original
        canvas.paste(Image.fromarray(orig_np).resize(thumb, Image.LANCZOS),
                     (row_label_w, y + label_h))
        if row == 0:
            draw.text((row_label_w + cell_w//2, y + 3), "Original",
                      fill=(255,255,255), font=font, anchor='mt')

        for col, word in enumerate(WORDS):
            x = row_label_w + (col+1) * cell_w
            p = all_results[vname][word]
            img_np = render(img_tensor, p)
            canvas.paste(Image.fromarray(img_np).resize(thumb, Image.LANCZOS), (x, y + label_h))
            if row == 0:
                draw.text((x + cell_w//2, y + 3), word, fill=(200,200,200), font=font, anchor='mt')

    out_path = os.path.join(RESULTS_DIR, 'comparison.png')
    canvas.save(out_path, quality=95)
    print(f"\nGrid: {out_path}")

    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
