"""
PCA analysis of per-(word, image) CCMs from Experiment 01.

Optimizes a CCM per (word, image) pair in LINEAR domain with gamma,
then uses PCA to visualize variance across images for each word.

Usage:
    python exp01_pca_analysis.py [--num_images 15] [--num_iters 300]
"""

import os, json, warnings, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
warnings.filterwarnings('ignore', message="Can't initialize NVML")
import clip
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIVEK_LINEAR = "/data/data1/FiveK/default_isp_linear_720p"
RESULTS_DIR = "/data/data1/semantic_isp/results/exp01_anchor_discovery"
DEVICE = "cuda"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=15)
    parser.add_argument("--num_iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--reg_weight", type=float, default=0.015)
    parser.add_argument("--image_size", type=int, default=448)
    args = parser.parse_args()

    WORDS = ['sunset', 'noir', 'dreamy', 'vivid', 'monochrome',
             'modern', 'vintage', 'golden']

    # Load CLIP
    model, _ = clip.load('ViT-B/32', device=DEVICE)
    model.eval()
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=DEVICE).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=DEVICE).view(1, 3, 1, 1)

    # Load linear-domain images
    all_imgs = sorted([f for f in os.listdir(FIVEK_LINEAR)
                       if f.endswith(('.png', '.jpg'))])
    indices = np.linspace(0, len(all_imgs) - 1, args.num_images, dtype=int)
    selected = [all_imgs[i] for i in indices]

    image_tensors = []
    for fname in selected:
        img = Image.open(os.path.join(FIVEK_LINEAR, fname)).convert('RGB')
        img = img.resize((args.image_size, args.image_size), Image.LANCZOS)
        t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        image_tensors.append(t)
    images = torch.stack(image_tensors).to(DEVICE)
    print(f"Loaded {args.num_images} LINEAR-domain images")

    # Pre-encode text
    neutral_tok = clip.tokenize(["a photo"]).to(DEVICE)
    with torch.no_grad():
        neutral_feat = model.encode_text(neutral_tok)
        neutral_feat = neutral_feat / neutral_feat.norm(dim=-1, keepdim=True)

    # Optimize per (word, image)
    results = []
    total = len(WORDS) * args.num_images
    done = 0
    t0 = time.time()

    for word in WORDS:
        tok = clip.tokenize([f"a {word} photo"]).to(DEVICE)
        with torch.no_grad():
            text_feat = model.encode_text(tok)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        for img_idx in range(args.num_images):
            img_batch = images[img_idx:img_idx + 1]

            raw = torch.nn.Parameter(torch.zeros(3, 3, device=DEVICE))
            optimizer = torch.optim.Adam([raw], lr=args.lr)

            best_sim = -1
            best_ccm = None

            for it in range(args.num_iters):
                optimizer.zero_grad()

                # Build constrained CCM
                identity = torch.eye(3, device=DEVICE)
                ccm = identity + raw
                row_sums = ccm.sum(dim=1, keepdim=True)
                ccm = ccm - (row_sums - 1.0) / 3.0

                # Apply CCM in LINEAR domain
                pixels = img_batch.view(1, 3, -1)
                transformed = torch.matmul(ccm, pixels)
                transformed = transformed.view(1, 3, args.image_size, args.image_size)
                transformed = torch.clamp(transformed, 0.0, 1.0)

                # Apply GAMMA to convert to sRGB
                transformed_srgb = torch.pow(transformed + 1e-8, 1.0 / 2.2)

                # CLIP preprocessing
                resized = F.interpolate(transformed_srgb, size=(224, 224),
                                        mode='bicubic', align_corners=False)
                normed = (resized - clip_mean) / clip_std
                img_feat = model.encode_image(normed)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                sim_target = (img_feat @ text_feat.T).mean()
                sim_neutral = (img_feat @ neutral_feat.T).mean()
                loss = -(sim_target - 0.5 * sim_neutral) + \
                       args.reg_weight * (raw ** 2).sum()
                loss.backward()
                optimizer.step()

                if sim_target.item() > best_sim:
                    best_sim = sim_target.item()
                    with torch.no_grad():
                        ccm_final = identity + raw
                        rs = ccm_final.sum(dim=1, keepdim=True)
                        ccm_final = ccm_final - (rs - 1.0) / 3.0
                        best_ccm = ccm_final.cpu().numpy().copy()

            results.append({
                'word': word,
                'image_idx': img_idx,
                'ccm_flat': best_ccm.flatten(),
                'sim': best_sim,
            })
            done += 1

        elapsed = time.time() - t0
        eta = elapsed / done * (total - done)
        print(f"  {word}: done ({done}/{total}, ETA {eta:.0f}s)")

    # PCA
    all_ccms = np.array([r['ccm_flat'] for r in results])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_ccms)
    ev = pca.explained_variance_ratio_
    print(f"PCA explained variance: {ev[0]:.3f}, {ev[1]:.3f} (total: {ev.sum():.3f})")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    colors_map = plt.cm.Set1(np.linspace(0, 1, len(WORDS)))

    for i, word in enumerate(WORDS):
        mask = np.array([r['word'] == word for r in results])
        xy = coords[mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=[colors_map[i]], s=80, alpha=0.7,
                   label=word, edgecolors='white', linewidth=0.5)
        cx, cy = xy.mean(axis=0)
        ax.scatter(cx, cy, c=[colors_map[i]], s=250, marker='*',
                   edgecolors='black', linewidth=1.5, zorder=5)

    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% variance)', fontsize=13)
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% variance)', fontsize=13)
    ax.set_title('CCM per (word, image) — PCA projection [LINEAR domain + gamma]\n'
                 'Dots = individual images, Stars = word centroid', fontsize=14)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, 'pca_ccm_per_image_v3_linear.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")

    # Per-word variance
    print(f"\n{'Word':<15} {'Std(PC1)':<10} {'Std(PC2)':<10} {'Total Std':<10}")
    print('-' * 45)
    for word in WORDS:
        mask = np.array([r['word'] == word for r in results])
        xy = coords[mask]
        s1, s2 = xy[:, 0].std(), xy[:, 1].std()
        print(f"{word:<15} {s1:<10.4f} {s2:<10.4f} {np.sqrt(s1**2+s2**2):<10.4f}")


if __name__ == "__main__":
    main()
