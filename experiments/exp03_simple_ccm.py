"""
Experiment 03: Simplified CCM-only optimization

Single image, 10 clear words, CCM with row-sum=1 (6 effective DoF).
No other ISP params. Includes convergence plots.

Pipeline:
    Linear image → CCM (row-sum=1) → gamma (1/2.2) → sRGB → CLIP

Words chosen for clear visual meaning in image processing:
    colorful, dark, bright, warm, cool, monochrome, saturated, faded, sharp, soft
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
RESULTS_DIR = "/data/data1/semantic_isp/results/exp03_simple_ccm"
DEVICE = "cuda"

# 10 words with clear, unambiguous image-processing meaning
WORDS = ['colorful', 'dark', 'bright', 'warm', 'cool',
         'monochrome', 'saturated', 'faded', 'vivid', 'muted']


def optimize_ccm(word, img_tensor, clip_model, clip_mean, clip_std,
                 num_iters=500, lr=0.01, reg_weight=0.01):
    """Optimize 3x3 CCM (row-sum=1) for one word on one image.

    Returns: ccm (3x3), best_sim, loss_history, sim_history
    """
    # Text encoding
    tok = clip.tokenize([f"a {word} photo"]).to(DEVICE)
    neutral_tok = clip.tokenize(["a photo"]).to(DEVICE)
    with torch.no_grad():
        text_feat = clip_model.encode_text(tok)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        neutral_feat = clip_model.encode_text(neutral_tok)
        neutral_feat = neutral_feat / neutral_feat.norm(dim=-1, keepdim=True)

    # CCM: parameterized as offsets from identity
    ccm_raw = torch.nn.Parameter(torch.zeros(3, 3, device=DEVICE))
    optimizer = torch.optim.Adam([ccm_raw], lr=lr)

    loss_history = []
    sim_history = []
    best_sim = -1
    best_ccm = None

    B, C, H, W = img_tensor.shape

    for it in range(num_iters):
        optimizer.zero_grad()

        # Build constrained CCM (row-sum = 1)
        ccm = torch.eye(3, device=DEVICE) + ccm_raw
        row_sums = ccm.sum(dim=1, keepdim=True)
        ccm = ccm - (row_sums - 1.0) / 3.0

        # Apply CCM in linear domain
        pixels = img_tensor.reshape(B, 3, -1)
        transformed = torch.clamp(torch.matmul(ccm, pixels).reshape(B, 3, H, W), 0, 1)

        # Apply gamma (linear → sRGB)
        srgb = torch.pow(transformed + 1e-8, 1.0 / 2.2)

        # CLIP
        resized = F.interpolate(srgb, size=(224, 224), mode='bicubic', align_corners=False)
        normed = (resized - clip_mean) / clip_std
        img_feat = clip_model.encode_image(normed)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sim_target = (img_feat @ text_feat.T).mean()
        sim_neutral = (img_feat @ neutral_feat.T).mean()

        # Loss
        clip_loss = -(sim_target - 0.5 * sim_neutral)
        reg_loss = reg_weight * (ccm_raw ** 2).sum()
        total_loss = clip_loss + reg_loss

        total_loss.backward()
        optimizer.step()

        loss_val = total_loss.item()
        sim_val = sim_target.item()
        loss_history.append(loss_val)
        sim_history.append(sim_val)

        if sim_val > best_sim:
            best_sim = sim_val
            with torch.no_grad():
                best_ccm_t = torch.eye(3, device=DEVICE) + ccm_raw
                rs = best_ccm_t.sum(dim=1, keepdim=True)
                best_ccm = (best_ccm_t - (rs - 1.0) / 3.0).cpu().numpy().copy()

    return best_ccm, best_sim, loss_history, sim_history


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Experiment 03: Simplified CCM-only (6 DoF)")
    print("=" * 60)

    # Load image (linear domain)
    all_imgs = sorted([f for f in os.listdir(FIVEK_LINEAR) if f.endswith('.png')])
    img_name = all_imgs[0]  # Monument Valley
    img = Image.open(os.path.join(FIVEK_LINEAR, img_name)).convert('RGB')
    img = img.resize((512, 512), Image.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    print(f"Image: {img_name} (512x512, linear domain)")
    print(f"Words: {WORDS}")
    print(f"Params: 3x3 CCM with row-sum=1 constraint (6 effective DoF)")
    print(f"Iters: 500, lr=0.01, reg=0.01")
    print()

    # CLIP
    clip_model, _ = clip.load('ViT-B/32', device=DEVICE)
    clip_model.eval()
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=DEVICE).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=DEVICE).view(1, 3, 1, 1)

    # Optimize each word
    results = {}
    all_loss_histories = {}
    all_sim_histories = {}

    for i, word in enumerate(WORDS):
        print(f"[{i+1}/{len(WORDS)}] '{word}'...", end=" ", flush=True)
        ccm, sim, loss_hist, sim_hist = optimize_ccm(
            word, img_tensor, clip_model, clip_mean, clip_std,
            num_iters=500, lr=0.01, reg_weight=0.01
        )
        results[word] = {
            'ccm': ccm.tolist(),
            'ccm_diag': [float(ccm[i, i]) for i in range(3)],
            'row_sums': ccm.sum(axis=1).tolist(),
            'clip_sim': sim,
        }
        all_loss_histories[word] = loss_hist
        all_sim_histories[word] = sim_hist

        d = np.diag(ccm)
        print(f"sim={sim:.4f}  diag=[{d[0]:.3f},{d[1]:.3f},{d[2]:.3f}]  "
              f"loss: {loss_hist[0]:.4f}→{loss_hist[-1]:.4f}")

    # ==================================================================
    # Plot 1: Loss convergence per word
    # ==================================================================
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for idx, word in enumerate(WORDS):
        ax = axes[idx]
        ax.plot(all_loss_histories[word], 'b-', alpha=0.7, linewidth=0.8, label='loss')
        ax.set_title(f"'{word}' (sim={results[word]['clip_sim']:.4f})", fontsize=11)
        ax.set_xlabel('iteration')
        ax.set_ylabel('loss')
        ax.grid(True, alpha=0.3)

        # Add sim on secondary axis
        ax2 = ax.twinx()
        ax2.plot(all_sim_histories[word], 'r-', alpha=0.5, linewidth=0.8, label='CLIP sim')
        ax2.set_ylabel('CLIP sim', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    plt.suptitle('Loss & CLIP Similarity Convergence — CCM only (6 DoF), 500 iters, lr=0.01',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    conv_path = os.path.join(RESULTS_DIR, 'convergence.png')
    plt.savefig(conv_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConvergence plot: {conv_path}")

    # ==================================================================
    # Plot 2: Visual results grid
    # ==================================================================
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

    # Render original
    with torch.no_grad():
        orig_srgb = torch.pow(img_tensor + 1e-8, 1.0 / 2.2)
        orig_np = (orig_srgb[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    canvas.paste(Image.fromarray(orig_np).resize(thumb, Image.LANCZOS), (0, label_h))
    draw.text((cell_w // 2, 4), "Original", fill=(255, 255, 255), font=font, anchor='mt')

    # Sort by similarity for display
    sorted_words = sorted(WORDS, key=lambda w: -results[w]['clip_sim'])

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

    grid_path = os.path.join(RESULTS_DIR, 'grid.png')
    canvas.save(grid_path, quality=95)
    print(f"Grid: {grid_path}")

    # ==================================================================
    # Summary table
    # ==================================================================
    print(f"\n{'Word':<14} {'Sim':>6} {'Diag[R,G,B]':<24} {'Row Sums'}")
    print("-" * 65)
    for word in sorted_words:
        r = results[word]
        d = r['ccm_diag']
        rs = r['row_sums']
        print(f"{word:<14} {r['clip_sim']:.4f} [{d[0]:.3f},{d[1]:.3f},{d[2]:.3f}]    "
              f"[{rs[0]:.3f},{rs[1]:.3f},{rs[2]:.3f}]")

    # Save
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump({'image': img_name, 'words': results,
                   'config': {'num_iters': 500, 'lr': 0.01, 'reg_weight': 0.01,
                              'params': 'CCM 3x3 row-sum=1 (6 DoF)',
                              'pipeline': 'linear → CCM → gamma(1/2.2) → sRGB → CLIP'}
                   }, f, indent=2)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
