"""
Experiment 03b: CCM-only on 3 different images

Same setup as exp03 (10 clear words, CCM 6 DoF, 500 iters)
but run on 3 diverse images to see if the pink bias is image-specific.

Images:
    - a0501: Urban street (Paris, cars)
    - a1501: Child portrait (indoor)
    - a3501: Flowers (yellow, close-up)
"""

import os, json, warnings, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', message="Can't initialize NVML")
import clip

FIVEK_LINEAR = "/data/data1/FiveK/default_isp_linear_720p"
RESULTS_DIR = "/data/data1/semantic_isp/results/exp03b_multi_image"
DEVICE = "cuda"

WORDS = ['colorful', 'dark', 'bright', 'warm', 'cool',
         'monochrome', 'saturated', 'faded', 'vivid', 'muted']

IMAGE_INDICES = [2200, 1800, 4500]


def optimize_ccm(word, img_tensor, clip_model, clip_mean, clip_std,
                 num_iters=500, lr=0.01, reg_weight=0.01):
    tok = clip.tokenize([f"a {word} photo"]).to(DEVICE)
    neutral_tok = clip.tokenize(["a photo"]).to(DEVICE)
    with torch.no_grad():
        text_feat = clip_model.encode_text(tok)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        neutral_feat = clip_model.encode_text(neutral_tok)
        neutral_feat = neutral_feat / neutral_feat.norm(dim=-1, keepdim=True)

    ccm_raw = torch.nn.Parameter(torch.zeros(3, 3, device=DEVICE))
    optimizer = torch.optim.Adam([ccm_raw], lr=lr)

    loss_history = []
    sim_history = []
    best_sim = -1
    best_ccm = None
    B, C, H, W = img_tensor.shape

    for it in range(num_iters):
        optimizer.zero_grad()
        ccm = torch.eye(3, device=DEVICE) + ccm_raw
        row_sums = ccm.sum(dim=1, keepdim=True)
        ccm = ccm - (row_sums - 1.0) / 3.0

        pixels = img_tensor.reshape(B, 3, -1)
        transformed = torch.clamp(torch.matmul(ccm, pixels).reshape(B, 3, H, W), 0, 1)
        srgb = torch.pow(transformed + 1e-8, 1.0 / 2.2)

        resized = F.interpolate(srgb, size=(224, 224), mode='bicubic', align_corners=False)
        normed = (resized - clip_mean) / clip_std
        img_feat = clip_model.encode_image(normed)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sim_target = (img_feat @ text_feat.T).mean()
        sim_neutral = (img_feat @ neutral_feat.T).mean()
        clip_loss = -(sim_target - 0.5 * sim_neutral)
        reg_loss = reg_weight * (ccm_raw ** 2).sum()
        total_loss = clip_loss + reg_loss

        total_loss.backward()
        optimizer.step()

        loss_history.append(total_loss.item())
        sim_history.append(sim_target.item())

        if sim_target.item() > best_sim:
            best_sim = sim_target.item()
            with torch.no_grad():
                c = torch.eye(3, device=DEVICE) + ccm_raw
                rs = c.sum(dim=1, keepdim=True)
                best_ccm = (c - (rs - 1.0) / 3.0).cpu().numpy().copy()

    return best_ccm, best_sim, loss_history, sim_history


def render_grid(img_tensor, results, sorted_words, title, out_path):
    """Render grid for one image."""
    n_cols = 6
    n_rows = 2
    thumb = (256, 256)
    label_h = 24
    cell_w, cell_h = thumb

    canvas = Image.new('RGB', (n_cols * cell_w, n_rows * (cell_h + label_h)), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 12)
    except:
        font = ImageFont.load_default()

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

        ccm = torch.tensor(results[word]['ccm'], device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            B, C, H, W = img_tensor.shape
            pixels = img_tensor.reshape(B, 3, -1)
            transformed = torch.clamp(torch.matmul(ccm, pixels).reshape(B, 3, H, W), 0, 1)
            srgb = torch.pow(transformed + 1e-8, 1.0 / 2.2)
            proc_np = (srgb[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        canvas.paste(Image.fromarray(proc_np).resize(thumb, Image.LANCZOS), (x, y + label_h))
        sim = results[word]['clip_sim']
        draw.text((x + cell_w // 2, y + 4), f"{word} ({sim:.3f})",
                  fill=(100, 255, 100), font=font, anchor='mt')

    canvas.save(out_path, quality=95)
    return canvas


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Experiment 03b: CCM-only on 3 diverse images")
    print("=" * 60)

    all_imgs = sorted([f for f in os.listdir(FIVEK_LINEAR) if f.endswith('.png')])

    clip_model, _ = clip.load('ViT-B/32', device=DEVICE)
    clip_model.eval()
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=DEVICE).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=DEVICE).view(1, 3, 1, 1)

    all_grids = []
    all_image_results = {}

    for img_idx in IMAGE_INDICES:
        img_name = all_imgs[img_idx]
        print(f"\n{'='*60}")
        print(f"Image: {img_name} (index {img_idx})")
        print(f"{'='*60}")

        img = Image.open(os.path.join(FIVEK_LINEAR, img_name)).convert('RGB')
        img = img.resize((512, 512), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        results = {}
        for i, word in enumerate(WORDS):
            print(f"  [{i+1}/{len(WORDS)}] '{word}'...", end=" ", flush=True)
            ccm, sim, _, _ = optimize_ccm(
                word, img_tensor, clip_model, clip_mean, clip_std,
                num_iters=500, lr=0.01, reg_weight=0.01
            )
            results[word] = {
                'ccm': ccm.tolist(),
                'ccm_diag': [float(ccm[j, j]) for j in range(3)],
                'clip_sim': sim,
            }
            d = np.diag(ccm)
            print(f"sim={sim:.4f}  diag=[{d[0]:.3f},{d[1]:.3f},{d[2]:.3f}]")

        sorted_words = sorted(WORDS, key=lambda w: -results[w]['clip_sim'])

        # Grid for this image
        grid_path = os.path.join(RESULTS_DIR, f'grid_{img_idx}_{img_name.split("-")[0]}.png')
        grid = render_grid(img_tensor, results, sorted_words, img_name, grid_path)
        all_grids.append(grid)
        all_image_results[img_name] = results
        print(f"  Grid: {grid_path}")

    # Stitch all 3 grids vertically
    total_h = sum(g.height for g in all_grids) + 30 * len(all_grids)
    combined = Image.new('RGB', (all_grids[0].width, total_h), (15, 15, 15))
    draw_c = ImageDraw.Draw(combined)
    try:
        font_title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    except:
        font_title = ImageFont.load_default()

    y = 0
    for i, (img_idx, grid) in enumerate(zip(IMAGE_INDICES, all_grids)):
        img_name = all_imgs[img_idx]
        draw_c.text((10, y + 5), f"Image: {img_name}", fill=(255, 200, 100), font=font_title)
        y += 22
        combined.paste(grid, (0, y))
        y += grid.height + 8

    combined_path = os.path.join(RESULTS_DIR, 'all_images_combined.png')
    combined.save(combined_path, quality=95)
    print(f"\nCombined grid: {combined_path} ({combined.size[0]}x{combined.size[1]})")

    # Save results
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(all_image_results, f, indent=2)

    # Print comparison table
    print(f"\n{'Word':<14}", end="")
    for img_idx in IMAGE_INDICES:
        print(f"  img{img_idx:>5}", end="")
    print()
    print("-" * 50)
    for word in WORDS:
        print(f"{word:<14}", end="")
        for img_idx in IMAGE_INDICES:
            img_name = all_imgs[img_idx]
            sim = all_image_results[img_name][word]['clip_sim']
            print(f"  {sim:.4f}", end="")
        print()

    print("\nDone!")


if __name__ == "__main__":
    main()
