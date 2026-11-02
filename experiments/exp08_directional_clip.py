"""
Experiment 08: Directional CLIP Loss + Quality Preservation

Improvements over exp07:
1. Directional CLIP loss: measures the DIRECTION of change from original→processed
   aligned with the text direction from antonym→word. Prevents CLIP bias.
2. Quality preservation: penalizes if "beauty" score drops vs original.
3. Same 3-param ISP from RAW (saturation, brightness, sharpness).

Loss = L_semantic + λ_quality * L_quality + λ_reg * ||params||²
"""

import os, sys, json, warnings, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', message="Can't initialize NVML")
import clip

sys.path.insert(0, os.path.dirname(__file__))
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw

DNG_DIR = "/data/data1/EmotionISP_dataset/dng_filtered"
RESULTS_DIR = "/data/data1/semantic_isp/results/exp08_directional"
DEVICE = "cuda"

# Words with antonym pairs
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
    """Encode image through CLIP (differentiable)."""
    resized = F.interpolate(img_tensor, size=(224, 224), mode='bicubic', align_corners=False)
    normed = (resized - clip_mean) / clip_std
    feat = clip_model.encode_image(normed)
    return feat / feat.norm(dim=-1, keepdim=True)


def optimize_for_word(word, antonym, img_tensor, orig_feat, beauty_feat,
                      orig_beauty_score, clip_model, clip_mean, clip_std,
                      num_iters=500, lr=0.03, reg_weight=0.003,
                      quality_weight=0.5):
    # Text direction: antonym → word
    tok_word = clip.tokenize([f"a {word} photo"]).to(DEVICE)
    tok_anti = clip.tokenize([f"a {antonym} photo"]).to(DEVICE)
    with torch.no_grad():
        feat_word = clip_model.encode_text(tok_word)
        feat_word = feat_word / feat_word.norm(dim=-1, keepdim=True)
        feat_anti = clip_model.encode_text(tok_anti)
        feat_anti = feat_anti / feat_anti.norm(dim=-1, keepdim=True)
        # Text direction
        text_dir = feat_word - feat_anti
        text_dir = text_dir / (text_dir.norm(dim=-1, keepdim=True) + 1e-8)

    isp = DifferentiableEMOVIS(device=DEVICE)
    optimizer = torch.optim.Adam(isp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters,
                                                            eta_min=lr * 0.1)

    loss_hist, sem_hist, qual_hist = [], [], []
    best_loss = float('inf')
    best_dir_sim = 0.0
    best_params = isp.get_params()  # start with neutral

    for it in range(num_iters):
        optimizer.zero_grad()
        processed = isp(img_tensor)

        # Encode processed image
        proc_feat = encode_clip_image(processed, clip_model, clip_mean, clip_std)

        # === Loss 1: Directional CLIP (semantic) ===
        # Image direction: original → processed (unnormalized dot product)
        img_dir = proc_feat - orig_feat
        # Use dot product with text direction (not cosine — avoids zero-norm issue)
        directional_sim = (img_dir @ text_dir.T).mean()
        L_semantic = -directional_sim

        # === Loss 2: Quality preservation ===
        # How beautiful is the processed image vs original?
        proc_beauty = (proc_feat @ beauty_feat.T).mean()
        # Penalize only if beauty drops
        beauty_drop = F.relu(orig_beauty_score - proc_beauty)
        L_quality = beauty_drop

        # === Loss 3: Regularization ===
        L_reg = reg_weight * (isp.raw_params ** 2).sum()

        total_loss = L_semantic + quality_weight * L_quality + L_reg

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_hist.append(total_loss.item())
        sem_hist.append(directional_sim.item())
        qual_hist.append(proc_beauty.item())

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_params = isp.get_params()
            best_dir_sim = directional_sim.item()

    return best_params, best_dir_sim, loss_hist, sem_hist, qual_hist


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Exp 08: Directional CLIP + Quality Preservation")
    print("Loss = L_semantic(directional) + λ*L_quality(beauty) + reg")
    print("3 params: saturation, brightness, sharpness")
    print("=" * 60)

    # Load RAW
    all_dngs = sorted([f for f in os.listdir(DNG_DIR) if f.endswith('.dng')])
    dng_name = all_dngs[0]
    dng_path = os.path.join(DNG_DIR, dng_name)
    print(f"\nRAW: {dng_name}")

    img_tensor = load_and_preprocess_raw(dng_path, target_size=512).to(DEVICE)
    print(f"Loaded: {img_tensor.shape}")

    # CLIP
    clip_model, _ = clip.load('ViT-B/32', device=DEVICE)
    clip_model.eval()
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(1, 3, 1, 1)

    # Pre-compute original image CLIP features and beauty score
    isp_neutral = DifferentiableEMOVIS(device=DEVICE)
    with torch.no_grad():
        orig_processed = isp_neutral(img_tensor)
        orig_feat = encode_clip_image(orig_processed, clip_model, clip_mean, clip_std)

        beauty_tok = clip.tokenize(["a beautiful high quality photo"]).to(DEVICE)
        beauty_feat = clip_model.encode_text(beauty_tok)
        beauty_feat = beauty_feat / beauty_feat.norm(dim=-1, keepdim=True)

        orig_beauty_score = (orig_feat @ beauty_feat.T).mean()
        print(f"Original beauty score: {orig_beauty_score.item():.4f}")

    # Optimize
    results = {}
    t0 = time.time()

    for i, (word, antonym) in enumerate(WORD_PAIRS):
        print(f"\n[{i + 1}/{len(WORD_PAIRS)}] '{word}' (vs '{antonym}')...", flush=True)
        params, score, loss_h, sem_h, qual_h = optimize_for_word(
            word, antonym, img_tensor, orig_feat, beauty_feat,
            orig_beauty_score, clip_model, clip_mean, clip_std,
            num_iters=500, lr=0.03, reg_weight=0.003, quality_weight=0.5
        )
        results[word] = {
            'params': params, 'directional_sim': score,
            'antonym': antonym,
            'final_beauty': qual_h[-1] if qual_h else 0,
        }
        p = params
        print(f"  dir_sim={score:.4f}  beauty={qual_h[-1]:.4f}  "
              f"sat={p['saturation']:+.3f} br={p['brightness']:+.3f} sh={p['sharpness']:+.3f}")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed / len(WORD_PAIRS):.1f}s/word)")

    # Grid
    sorted_words = [w for w, _ in WORD_PAIRS]
    isp = DifferentiableEMOVIS(device=DEVICE)

    n_cols = 6
    n_rows = 2
    thumb = (256, 256)
    label_h = 42
    cell_w, cell_h = thumb

    canvas = Image.new('RGB', (n_cols * cell_w, n_rows * (cell_h + label_h)), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 11)
        font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
    except:
        font = font_small = ImageFont.load_default()

    # Original
    with torch.no_grad():
        orig = isp(img_tensor)
        orig_np = (orig[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    canvas.paste(Image.fromarray(orig_np).resize(thumb, Image.LANCZOS), (0, label_h))
    draw.text((cell_w // 2, 4), "Original", fill=(255, 255, 255), font=font, anchor='mt')
    draw.text((cell_w // 2, 20), "sat=0  br=0  sh=0", fill=(180, 180, 180), font=font_small, anchor='mt')

    # Sort by directional similarity
    sorted_by_sim = sorted(results.keys(), key=lambda w: -results[w]['directional_sim'])

    for idx, word in enumerate(sorted_by_sim):
        pos = idx + 1
        col = pos % n_cols
        row = pos // n_cols
        x = col * cell_w
        y = row * (cell_h + label_h)

        p = results[word]['params']
        with torch.no_grad():
            isp.raw_params.zero_()
            for pi, name in enumerate(DifferentiableEMOVIS.PARAM_NAMES):
                lo, hi = DifferentiableEMOVIS.PARAM_RANGES[name]
                mid, half = (hi + lo) / 2.0, (hi - lo) / 2.0
                norm = (p[name] - mid) / half
                norm = max(-0.999, min(0.999, norm))
                isp.raw_params.data[pi] = np.arctanh(norm)
            processed = isp(img_tensor)
            proc_np = (processed[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        canvas.paste(Image.fromarray(proc_np).resize(thumb, Image.LANCZOS), (x, y + label_h))
        sim = results[word]['directional_sim']
        draw.text((x + cell_w // 2, y + 4), f"{word} ({sim:.3f})",
                  fill=(100, 255, 100), font=font, anchor='mt')
        param_str = f"sat={p['saturation']:+.2f}  br={p['brightness']:+.2f}  sh={p['sharpness']:+.2f}"
        draw.text((x + cell_w // 2, y + 20), param_str,
                  fill=(200, 200, 200), font=font_small, anchor='mt')

    grid_path = os.path.join(RESULTS_DIR, 'grid.png')
    canvas.save(grid_path, quality=95)
    print(f"Grid: {grid_path}")

    # Save
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump({
            'description': 'Directional CLIP + Quality preservation',
            'loss': 'L_semantic(directional) + 0.5*L_quality(beauty_drop) + 0.003*L_reg',
            'dng': dng_name,
            'orig_beauty': orig_beauty_score.item(),
            'words': results,
            'time_sec': elapsed,
        }, f, indent=2)

    # Summary
    print(f"\n{'Word':<12} {'DirSim':>7} {'Beauty':>7} {'Sat':>6} {'Br':>6} {'Sh':>6}")
    print("-" * 50)
    for word in sorted_by_sim:
        p = results[word]['params']
        s = results[word]['directional_sim']
        b = results[word]['final_beauty']
        print(f"{word:<12} {s:+.4f} {b:.4f} {p['saturation']:+.3f} "
              f"{p['brightness']:+.3f} {p['sharpness']:+.3f}")

    print(f"\nOriginal beauty: {orig_beauty_score.item():.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
