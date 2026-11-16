"""
Experiment 10: Semantic vs Directional Loss — revisited with SigLIP-B/16

Two loss functions, both with beauty + realism guards:

A) SEMANTIC LOSS:
   Maximize absolute CLIP similarity between processed image and text.
   L = -cos(VLM(processed), VLM("a {word} photo"))
   → Asks: "Does this image look like a {word} photo?"

B) DIRECTIONAL LOSS:
   Maximize the alignment between the image CHANGE direction and the
   text direction from antonym to target word.
   L = -dot(VLM(processed) - VLM(original), VLM("{word}") - VLM("{antonym}"))
   → Asks: "Did the image CHANGE in the direction of {word} vs {antonym}?"

Both include:
   + 0.5 * max(0, beauty_orig - beauty_proc)    # beauty guard
   + 0.5 * max(0, realism_orig - realism_proc)  # realism guard
   + 0.003 * ||params||²                         # regularization

Using SigLIP-B/16 (WebLI) as the foundation model.
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

RESULTS_DIR = "/data/data1/semantic_isp/results/exp10_loss_siglip"
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
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16-SigLIP', pretrained='webli', device=device)
        self.tokenize = open_clip.get_tokenizer('ViT-B-16-SigLIP')
        self.model.eval()
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)

    def encode_text(self, text):
        tokens = self.tokenize([text]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            return feat / feat.norm(dim=-1, keepdim=True)

    def encode_image(self, img_tensor):
        resized = F.interpolate(img_tensor, size=(224, 224), mode='bicubic', align_corners=False)
        normed = (resized - self.clip_mean) / self.clip_std
        feat = self.model.encode_image(normed)
        return feat / feat.norm(dim=-1, keepdim=True)


def optimize(word, antonym, img_tensor, vlm, orig_feat, beauty_feat, realism_feat,
             orig_beauty, orig_realism, mode='semantic',
             num_iters=500, lr=0.03, reg_weight=0.003,
             quality_weight=0.5, realism_weight=0.5):

    text_feat = vlm.encode_text(f"a {word} photo")

    if mode == 'directional':
        anti_feat = vlm.encode_text(f"a {antonym} photo")
        with torch.no_grad():
            text_dir = text_feat - anti_feat
            text_dir = text_dir / (text_dir.norm(dim=-1, keepdim=True) + 1e-8)

    isp = DifferentiableEMOVIS(device=DEVICE)
    optimizer = torch.optim.Adam(isp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=lr*0.1)

    best_loss = float('inf')
    best_params = isp.get_params()

    for it in range(num_iters):
        optimizer.zero_grad()
        processed = isp(img_tensor)
        proc_feat = vlm.encode_image(processed)

        if mode == 'semantic':
            L_semantic = -(proc_feat @ text_feat.T).mean()
        else:
            img_dir = proc_feat - orig_feat
            L_semantic = -(img_dir @ text_dir.T).mean()

        proc_beauty = (proc_feat @ beauty_feat.T).mean()
        L_beauty = F.relu(orig_beauty - proc_beauty)

        proc_realism = (proc_feat @ realism_feat.T).mean()
        L_realism = F.relu(orig_realism - proc_realism)

        L_reg = reg_weight * (isp.raw_params ** 2).sum()

        total_loss = L_semantic + quality_weight * L_beauty + realism_weight * L_realism + L_reg
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
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
    print("Exp 10: Semantic vs Directional — SigLIP-B/16")
    print("=" * 60)

    dng_path = '/data/data1/EmotionISP_dataset/dng/happy/8.dng'
    img_tensor = load_and_preprocess_raw(dng_path, target_size=512).to(DEVICE)

    vlm = SigLIPWrapper(DEVICE)

    # Pre-compute
    isp_neutral = DifferentiableEMOVIS(device=DEVICE)
    with torch.no_grad():
        orig = isp_neutral(img_tensor)
        orig_feat = vlm.encode_image(orig)
        beauty_feat = vlm.encode_text("a beautiful high quality photo")
        realism_feat = vlm.encode_text("a realistic photo")
        orig_beauty = (orig_feat @ beauty_feat.T).mean()
        orig_realism = (orig_feat @ realism_feat.T).mean()

    results_sem = {}
    results_dir = {}
    words = [w for w, _ in WORD_PAIRS]

    for i, (word, antonym) in enumerate(WORD_PAIRS):
        print(f"\n[{i+1}/{len(WORD_PAIRS)}] '{word}' (antonym: '{antonym}')")

        p_sem = optimize(word, antonym, img_tensor, vlm, orig_feat,
                         beauty_feat, realism_feat, orig_beauty, orig_realism,
                         mode='semantic')
        results_sem[word] = p_sem
        print(f"  Semantic:    wb={p_sem['wb_warmth']:+.3f} sat={p_sem['saturation']:+.3f} "
              f"br={p_sem['brightness']:+.3f} sh={p_sem['sharpness']:+.3f}")

        p_dir = optimize(word, antonym, img_tensor, vlm, orig_feat,
                         beauty_feat, realism_feat, orig_beauty, orig_realism,
                         mode='directional')
        results_dir[word] = p_dir
        print(f"  Directional: wb={p_dir['wb_warmth']:+.3f} sat={p_dir['saturation']:+.3f} "
              f"br={p_dir['brightness']:+.3f} sh={p_dir['sharpness']:+.3f}")

    # Grid: 2 rows (Semantic, Directional) × 11 columns (Original + 10 words)
    thumb = (150, 150)
    label_h = 30
    row_label_w = 90
    cell_w, cell_h = thumb
    n_cols = len(words) + 1

    canvas = Image.new('RGB', (row_label_w + n_cols*cell_w, 2*(cell_h+label_h)), (25,25,25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 9)
        font_big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 10)
    except:
        font = font_big = ImageFont.load_default()

    orig_np = render(img_tensor, {n: 0 for n in DifferentiableEMOVIS.PARAM_NAMES})

    for row, (label, results, color) in enumerate([
        ('Semantic', results_sem, (255, 150, 150)),
        ('Directional', results_dir, (150, 255, 150)),
    ]):
        y = row * (cell_h + label_h)

        # Row label
        draw.text((5, y + cell_h//2 + label_h), label, fill=color, font=font_big, anchor='lm')

        # Original
        canvas.paste(Image.fromarray(orig_np).resize(thumb, Image.LANCZOS),
                     (row_label_w, y + label_h))
        if row == 0:
            draw.text((row_label_w + cell_w//2, y + 3), "Original", fill=(255,255,255), font=font, anchor='mt')

        # Words
        for col, word in enumerate(words):
            x = row_label_w + (col+1) * cell_w
            p = results[word]
            img_np = render(img_tensor, p)
            canvas.paste(Image.fromarray(img_np).resize(thumb, Image.LANCZOS), (x, y + label_h))

            if row == 0:
                draw.text((x + cell_w//2, y + 3), word, fill=(200,200,200), font=font, anchor='mt')

            # Show params below word name
            ps = f"w={p['wb_warmth']:+.2f} s={p['saturation']:+.2f}"
            draw.text((x + cell_w//2, y + 14), ps, fill=(150,150,150), font=font, anchor='mt')

    out_path = os.path.join(RESULTS_DIR, 'comparison.png')
    canvas.save(out_path, quality=95)
    print(f"\nGrid: {out_path}")

    # Save
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump({'semantic': results_sem, 'directional': results_dir}, f, indent=2)

    # Summary
    print(f"\n{'Word':<12} {'--- Semantic ---':^30} {'--- Directional ---':^30}")
    print(f"{'':12} {'WB':>6} {'Sat':>6} {'Br':>6} {'Sh':>6}   {'WB':>6} {'Sat':>6} {'Br':>6} {'Sh':>6}")
    print("-" * 75)
    for word in words:
        ps = results_sem[word]
        pd = results_dir[word]
        print(f"{word:<12} {ps['wb_warmth']:+.3f} {ps['saturation']:+.3f} {ps['brightness']:+.3f} {ps['sharpness']:+.3f}"
              f"   {pd['wb_warmth']:+.3f} {pd['saturation']:+.3f} {pd['brightness']:+.3f} {pd['sharpness']:+.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
