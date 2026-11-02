"""
Experiment 08b: Side-by-side comparison of two CLIP loss functions

A) Semantic loss:     -cos(CLIP_image(processed), CLIP_text("a {word} photo"))
                      + beauty preservation + regularization

B) Directional loss:  -dot(CLIP_image(processed) - CLIP_image(original),
                           CLIP_text("{word}") - CLIP_text("{antonym}"))
                      + beauty preservation + regularization

Same ISP (3 params: sat, bright, sharp), same image, same words.
Output: side-by-side grid comparing A vs B.
"""

import os, sys, json, warnings, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
warnings.filterwarnings('ignore', message="Can't initialize NVML")
import clip

sys.path.insert(0, os.path.dirname(__file__))
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw

DNG_DIR = "/data/data1/EmotionISP_dataset/dng_filtered"
RESULTS_DIR = "/data/data1/semantic_isp/results/exp08b_comparison"
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


def encode_clip_image(img_tensor, clip_model, clip_mean, clip_std):
    resized = F.interpolate(img_tensor, size=(224, 224), mode='bicubic', align_corners=False)
    normed = (resized - clip_mean) / clip_std
    feat = clip_model.encode_image(normed)
    return feat / feat.norm(dim=-1, keepdim=True)


def optimize(word, antonym, img_tensor, orig_feat, beauty_feat, orig_beauty,
             clip_model, clip_mean, clip_std, mode='directional',
             num_iters=500, lr=0.03, reg_weight=0.003, quality_weight=0.5):
    """
    mode='directional': directional CLIP loss with antonym pair
    mode='semantic':    absolute CLIP similarity loss (original exp07 style)
    """
    tok_word = clip.tokenize([f"a {word} photo"]).to(DEVICE)
    with torch.no_grad():
        feat_word = clip_model.encode_text(tok_word)
        feat_word = feat_word / feat_word.norm(dim=-1, keepdim=True)

    if mode == 'directional':
        tok_anti = clip.tokenize([f"a {antonym} photo"]).to(DEVICE)
        with torch.no_grad():
            feat_anti = clip_model.encode_text(tok_anti)
            feat_anti = feat_anti / feat_anti.norm(dim=-1, keepdim=True)
            text_dir = feat_word - feat_anti
            text_dir = text_dir / (text_dir.norm(dim=-1, keepdim=True) + 1e-8)

    isp = DifferentiableEMOVIS(device=DEVICE)
    optimizer = torch.optim.Adam(isp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters,
                                                            eta_min=lr * 0.1)
    best_loss = float('inf')
    best_params = isp.get_params()

    for it in range(num_iters):
        optimizer.zero_grad()
        processed = isp(img_tensor)
        proc_feat = encode_clip_image(processed, clip_model, clip_mean, clip_std)

        # Semantic loss
        if mode == 'directional':
            img_dir = proc_feat - orig_feat
            L_semantic = -(img_dir @ text_dir.T).mean()
        else:  # 'semantic'
            sim = (proc_feat @ feat_word.T).mean()
            L_semantic = -sim

        # Quality
        proc_beauty = (proc_feat @ beauty_feat.T).mean()
        L_quality = F.relu(orig_beauty - proc_beauty)

        # Reg
        L_reg = reg_weight * (isp.raw_params ** 2).sum()

        total_loss = L_semantic + quality_weight * L_quality + L_reg
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_params = isp.get_params()

    return best_params


def render_image(img_tensor, params):
    isp = DifferentiableEMOVIS(device=DEVICE)
    with torch.no_grad():
        for pi, name in enumerate(DifferentiableEMOVIS.PARAM_NAMES):
            lo, hi = DifferentiableEMOVIS.PARAM_RANGES[name]
            mid, half = (hi + lo) / 2.0, (hi - lo) / 2.0
            norm = (params[name] - mid) / half
            norm = max(-0.999, min(0.999, norm))
            isp.raw_params.data[pi] = np.arctanh(norm)
        processed = isp(img_tensor)
        return (processed[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Exp 08b: Semantic vs Directional CLIP loss")
    print("Both with beauty preservation + regularization")
    print("=" * 60)

    all_dngs = sorted([f for f in os.listdir(DNG_DIR) if f.endswith('.dng')])
    dng_path = os.path.join(DNG_DIR, all_dngs[0])
    img_tensor = load_and_preprocess_raw(dng_path, target_size=512).to(DEVICE)

    clip_model, _ = clip.load('ViT-B/32', device=DEVICE)
    clip_model.eval()
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(1, 3, 1, 1)

    isp_neutral = DifferentiableEMOVIS(device=DEVICE)
    with torch.no_grad():
        orig_processed = isp_neutral(img_tensor)
        orig_feat = encode_clip_image(orig_processed, clip_model, clip_mean, clip_std)
        beauty_tok = clip.tokenize(["a beautiful high quality photo"]).to(DEVICE)
        beauty_feat = clip_model.encode_text(beauty_tok)
        beauty_feat = beauty_feat / beauty_feat.norm(dim=-1, keepdim=True)
        orig_beauty = (orig_feat @ beauty_feat.T).mean()

    results_sem = {}
    results_dir = {}

    for i, (word, antonym) in enumerate(WORD_PAIRS):
        print(f"\n[{i + 1}/{len(WORD_PAIRS)}] '{word}'")

        # Semantic loss
        p_sem = optimize(word, antonym, img_tensor, orig_feat, beauty_feat, orig_beauty,
                         clip_model, clip_mean, clip_std, mode='semantic')
        results_sem[word] = p_sem
        print(f"  Semantic:    sat={p_sem['saturation']:+.3f} br={p_sem['brightness']:+.3f} sh={p_sem['sharpness']:+.3f}")

        # Directional loss
        p_dir = optimize(word, antonym, img_tensor, orig_feat, beauty_feat, orig_beauty,
                         clip_model, clip_mean, clip_std, mode='directional')
        results_dir[word] = p_dir
        print(f"  Directional: sat={p_dir['saturation']:+.3f} br={p_dir['brightness']:+.3f} sh={p_dir['sharpness']:+.3f}")

    # Build comparison grid: 2 rows per word block, words as columns
    words = [w for w, _ in WORD_PAIRS]
    thumb = (180, 180)
    label_h = 38
    cell_w, cell_h = thumb
    n_cols = len(words) + 1  # +1 for original

    canvas_w = n_cols * cell_w
    canvas_h = 2 * (cell_h + label_h) + 30  # 2 rows + header
    canvas = Image.new('RGB', (canvas_w, canvas_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 10)
        font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 9)
    except:
        font = font_small = ImageFont.load_default()

    # Row labels
    row0_y = 0
    row1_y = cell_h + label_h + 10

    draw.text((5, row0_y + cell_h // 2 + label_h), "", fill=(255, 200, 100), font=font)
    draw.text((cell_w // 2, row0_y + 2), "Original", fill=(255, 255, 255), font=font, anchor='mt')

    # Original in both rows
    orig_np = render_image(img_tensor, {n: 0 for n in DifferentiableEMOVIS.PARAM_NAMES})
    orig_pil = Image.fromarray(orig_np).resize(thumb, Image.LANCZOS)
    canvas.paste(orig_pil, (0, row0_y + label_h))
    canvas.paste(orig_pil, (0, row1_y + label_h))

    draw.text((cell_w // 2, row0_y + 14), "Semantic", fill=(255, 150, 150), font=font_small, anchor='mt')
    draw.text((cell_w // 2, row1_y + 2), "Original", fill=(255, 255, 255), font=font, anchor='mt')
    draw.text((cell_w // 2, row1_y + 14), "Directional", fill=(150, 255, 150), font=font_small, anchor='mt')

    for col, word in enumerate(words):
        x = (col + 1) * cell_w

        # Semantic (top row)
        p = results_sem[word]
        img_np = render_image(img_tensor, p)
        canvas.paste(Image.fromarray(img_np).resize(thumb, Image.LANCZOS), (x, row0_y + label_h))
        draw.text((x + cell_w // 2, row0_y + 2), word, fill=(255, 150, 150), font=font, anchor='mt')
        ps = f"s={p['saturation']:+.2f} b={p['brightness']:+.2f} h={p['sharpness']:+.2f}"
        draw.text((x + cell_w // 2, row0_y + 14), ps, fill=(180, 180, 180), font=font_small, anchor='mt')

        # Directional (bottom row)
        p = results_dir[word]
        img_np = render_image(img_tensor, p)
        canvas.paste(Image.fromarray(img_np).resize(thumb, Image.LANCZOS), (x, row1_y + label_h))
        draw.text((x + cell_w // 2, row1_y + 2), word, fill=(150, 255, 150), font=font, anchor='mt')
        ps = f"s={p['saturation']:+.2f} b={p['brightness']:+.2f} h={p['sharpness']:+.2f}"
        draw.text((x + cell_w // 2, row1_y + 14), ps, fill=(180, 180, 180), font=font_small, anchor='mt')

    out_path = os.path.join(RESULTS_DIR, 'comparison.png')
    canvas.save(out_path, quality=95)
    print(f"\nComparison: {out_path}")

    # Save
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump({'semantic': {w: results_sem[w] for w in words},
                   'directional': {w: results_dir[w] for w in words}}, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
