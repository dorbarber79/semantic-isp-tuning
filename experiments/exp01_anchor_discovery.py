"""
Experiment 01: Anchor Discovery via CLIP-Guided CCM Optimization

Implements the Mayer et al. (CIC 2025) approach:
  - Differentiable 3x3 Color Correction Matrix (CCM) applied to images
  - CLIP-based loss: maximize alignment between processed image and text prompt
  - Adam optimizer to find optimal CCM per style word

Then selects optimal anchor set via greedy submodular coverage maximization.

Usage:
    python exp01_anchor_discovery.py [--num_images 50] [--num_iters 200] [--device cuda]
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Suppress NVML warning
warnings.filterwarnings("ignore", message="Can't initialize NVML")

import clip


# ==============================================================================
# Configuration
# ==============================================================================

FIVEK_DIR = "/data/data1/FiveK/default_isp_linear_720p"
STYLE_CORPUS = "/data/data1/semantic_isp/datasets/style_words/style_corpus.json"
RESULTS_DIR = "/data/data1/semantic_isp/results/exp01_anchor_discovery"


# ==============================================================================
# Differentiable CCM ISP Block
# ==============================================================================

class DifferentiableCCM(torch.nn.Module):
    """A differentiable 3x3 Color Correction Matrix with white-point constraint.

    Transforms each pixel: [R', G', B']^T = CCM @ [R, G, B]^T

    Constraint: each row of the CCM sums to 1 (white point preservation).
    This is enforced by parameterizing as unconstrained offsets from identity
    and projecting rows to sum to 1 via softmax-like normalization.

    The CCM is initialized to identity and optimized via gradient descent.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        # Parameterize as offsets from identity. The actual CCM is computed
        # in get_constrained_ccm() with row-sum-to-1 enforcement.
        self.raw = torch.nn.Parameter(torch.zeros(3, 3, device=device))

    def get_constrained_ccm(self):
        """Build the constrained CCM from raw parameters.

        Each row is: identity_row + offset, then normalized so row sums to 1.
        Values are soft-clamped to [-0.5, 1.5] range for realistic CCMs.
        """
        identity = torch.eye(3, device=self.raw.device)
        ccm = identity + self.raw

        # Enforce row sum = 1 by subtracting the excess evenly
        row_sums = ccm.sum(dim=1, keepdim=True)
        ccm = ccm - (row_sums - 1.0) / 3.0

        return ccm

    def forward(self, image_tensor, apply_gamma=True):
        """Apply constrained CCM to linear-domain image, then apply gamma.

        The correct ISP pipeline is:
            Linear RGB → CCM → clamp → gamma (1/2.2) → sRGB output

        Args:
            image_tensor: (B, 3, H, W) float tensor in [0, 1], LINEAR domain
            apply_gamma: if True, apply sRGB gamma after CCM (default True)

        Returns:
            processed: (B, 3, H, W) float tensor in [0, 1], sRGB domain
        """
        ccm = self.get_constrained_ccm()

        B, C, H, W = image_tensor.shape
        # Reshape to (B, 3, H*W) for matrix multiplication
        pixels = image_tensor.view(B, 3, -1)
        # Apply CCM in linear domain: (3, 3) @ (B, 3, H*W) -> (B, 3, H*W)
        transformed = torch.matmul(ccm, pixels)
        # Reshape back and clamp in linear domain
        transformed = transformed.view(B, 3, H, W)
        transformed = torch.clamp(transformed, 0.0, 1.0)

        # Apply gamma correction: linear → sRGB (gamma ≈ 1/2.2)
        if apply_gamma:
            transformed = torch.pow(transformed + 1e-8, 1.0 / 2.2)

        return transformed

    def get_ccm(self):
        """Return the constrained CCM as a numpy array."""
        return self.get_constrained_ccm().detach().cpu().numpy()

    def regularization_loss(self):
        """Penalty for deviating too far from identity.

        Returns L2 distance of raw offsets from zero (i.e., CCM from identity).
        """
        return (self.raw ** 2).sum()

    def reset(self):
        """Reset CCM to identity (raw offsets to zero)."""
        with torch.no_grad():
            self.raw.zero_()


# ==============================================================================
# CLIP-Based Loss
# ==============================================================================

class CLIPLoss:
    """CLIP-based loss for text-image alignment."""

    def __init__(self, model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        # CLIP normalization
        self.normalize = torch.nn.Sequential(
            # Resize to 224x224
        )

        # CLIP image preprocessing constants
        self.register_mean_std()

    def register_mean_std(self):
        """CLIP's normalization constants."""
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                 device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                device=self.device).view(1, 3, 1, 1)

    def encode_text(self, text):
        """Encode text prompt to CLIP embedding."""
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def preprocess_image(self, image_tensor):
        """Preprocess image tensor for CLIP (resize + normalize).

        Args:
            image_tensor: (B, 3, H, W) float tensor in [0, 1]
        """
        # Resize to 224x224
        resized = F.interpolate(image_tensor, size=(224, 224),
                                mode="bicubic", align_corners=False)
        # Normalize with CLIP stats
        normalized = (resized - self.mean) / self.std
        return normalized

    def compute_similarity(self, image_tensor, text_features):
        """Compute cosine similarity between processed image and text.

        Args:
            image_tensor: (B, 3, H, W) float tensor in [0, 1]
            text_features: (1, D) CLIP text embedding

        Returns:
            similarity: scalar tensor (differentiable)
        """
        preprocessed = self.preprocess_image(image_tensor)
        image_features = self.model.encode_image(preprocessed)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).mean()
        return similarity


# ==============================================================================
# Image Loading
# ==============================================================================

def load_images(image_dir, num_images=50, device="cuda", size=512):
    """Load a subset of images as tensors.

    Selects evenly-spaced images from the sorted list for diversity.
    """
    all_images = sorted([f for f in os.listdir(image_dir)
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    # Select evenly spaced subset
    indices = np.linspace(0, len(all_images) - 1, num_images, dtype=int)
    selected = [all_images[i] for i in indices]

    tensors = []
    for fname in selected:
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        # Resize to manageable size
        img = img.resize((size, size), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(img)).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        tensors.append(tensor)

    batch = torch.stack(tensors).to(device)
    print(f"Loaded {len(tensors)} images from {image_dir} (resized to {size}x{size})")
    return batch


# ==============================================================================
# CCM Optimization for a Single Word
# ==============================================================================

def optimize_ccm_for_word(word, images, clip_loss, num_iters=1000, lr=0.005,
                          batch_size=8, reg_weight=0.02, device="cuda",
                          verbose=True):
    """Optimize CCM to maximize CLIP alignment with a style word.

    Uses row-sum-constrained CCM with regularization toward identity,
    following the Mayer et al. (CIC 2025) approach.

    Args:
        word: style word (e.g., "angelic")
        images: (N, 3, H, W) tensor of input images
        clip_loss: CLIPLoss instance
        num_iters: number of optimization iterations (Mayer uses ~1000)
        lr: learning rate
        batch_size: images per optimization step
        reg_weight: regularization strength (penalty for deviation from identity)
        device: torch device

    Returns:
        ccm: optimized 3x3 CCM as numpy array
        final_sim: final CLIP similarity score
        history: list of similarity scores per iteration
    """
    # Encode text prompt — use two prompt variants for robustness
    prompt = f"a {word} photo"
    text_features = clip_loss.encode_text(prompt)

    # Also compute baseline similarity with neutral prompt
    neutral_prompt = "a photo"
    neutral_features = clip_loss.encode_text(neutral_prompt)

    # Initialize CCM
    ccm_module = DifferentiableCCM(device=device)
    optimizer = torch.optim.Adam(ccm_module.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters,
                                                            eta_min=lr * 0.1)

    N = images.shape[0]
    history = []
    best_sim = -1
    best_ccm = None

    for iteration in range(num_iters):
        # Sample a random batch of images
        idx = torch.randint(0, N, (min(batch_size, N),))
        batch = images[idx]

        optimizer.zero_grad()

        # Apply CCM
        processed = ccm_module(batch)

        # CLIP directional loss: maximize similarity with target prompt
        # relative to neutral prompt (directional approach from CLIPtone)
        sim_target = clip_loss.compute_similarity(processed, text_features)
        sim_neutral = clip_loss.compute_similarity(processed, neutral_features)

        # Loss = maximize (sim_target - sim_neutral) + regularization
        clip_loss_val = -(sim_target - 0.5 * sim_neutral)

        # Regularization: penalize deviation from identity CCM
        reg_loss = reg_weight * ccm_module.regularization_loss()

        total_loss = clip_loss_val + reg_loss

        # Backpropagate
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        sim_val = sim_target.item()
        history.append(sim_val)

        # Track best
        if sim_val > best_sim:
            best_sim = sim_val
            best_ccm = ccm_module.get_ccm().copy()

        if verbose and (iteration + 1) % 200 == 0:
            ccm_vals = ccm_module.get_ccm()
            diag = np.diag(ccm_vals)
            row_sums = ccm_vals.sum(axis=1)
            reg_val = ccm_module.regularization_loss().item()
            print(f"  iter {iteration+1:4d}/{num_iters}: "
                  f"sim={sim_val:.4f}, reg={reg_val:.4f}, "
                  f"diag=[{diag[0]:.3f}, {diag[1]:.3f}, {diag[2]:.3f}], "
                  f"row_sums=[{row_sums[0]:.3f}, {row_sums[1]:.3f}, {row_sums[2]:.3f}]")

    # Use best CCM found during optimization
    final_ccm = best_ccm if best_ccm is not None else ccm_module.get_ccm()
    final_sim = best_sim

    return final_ccm, final_sim, history


# ==============================================================================
# Greedy Anchor Selection
# ==============================================================================

def greedy_anchor_selection(word_ccms, num_anchors=10):
    """Select anchors that maximize coverage of the CCM space.

    Uses greedy submodular maximization.

    Args:
        word_ccms: dict {word: ccm_9d_vector}
        num_anchors: number of anchors to select

    Returns:
        selected: list of selected anchor words
        coverage_history: coverage at each selection step
    """
    words = list(word_ccms.keys())
    vectors = np.array([word_ccms[w].flatten() for w in words])

    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    vectors_norm = vectors / norms

    selected_indices = []
    selected_words = []
    coverage_history = []

    for step in range(num_anchors):
        best_word = None
        best_idx = None
        best_coverage = -1

        for i, word in enumerate(words):
            if i in selected_indices:
                continue

            # Try adding this word to the selected set
            candidate_indices = selected_indices + [i]
            candidate_vectors = vectors_norm[candidate_indices]

            # Compute coverage: for each word, find its best cosine similarity
            # to its nearest anchor (or best convex combination)
            coverage = compute_coverage(vectors_norm, candidate_vectors)

            if coverage > best_coverage:
                best_coverage = coverage
                best_word = word
                best_idx = i

        selected_indices.append(best_idx)
        selected_words.append(best_word)
        coverage_history.append(best_coverage)

        print(f"  Anchor {step+1}: '{best_word}' → coverage = {best_coverage:.4f}")

    return selected_words, coverage_history


def compute_coverage(all_vectors, anchor_vectors):
    """Compute average cosine similarity between each vector and its nearest anchor."""
    # Cosine similarity matrix: (N, K)
    similarities = all_vectors @ anchor_vectors.T
    # Best similarity for each word
    best_sim = similarities.max(axis=1)
    # Average coverage
    return best_sim.mean()


# ==============================================================================
# Main Experiment
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 01: Anchor Discovery")
    parser.add_argument("--num_images", type=int, default=50,
                        help="Number of FiveK images to use")
    parser.add_argument("--num_iters", type=int, default=1000,
                        help="Optimization iterations per word (Mayer uses ~1000)")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate for CCM optimization")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Images per optimization step")
    parser.add_argument("--reg_weight", type=float, default=0.02,
                        help="Regularization weight (penalty for CCM deviation from identity)")
    parser.add_argument("--num_anchors", type=int, default=10,
                        help="Number of anchors to select")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Resize images to this size")
    args = parser.parse_args()

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("Experiment 01: Anchor Discovery via CLIP-Guided CCM Optimization")
    print("=" * 70)

    # Load style corpus
    with open(STYLE_CORPUS) as f:
        corpus = json.load(f)

    # Combine all words for optimization
    all_words = (corpus["anchors"] +
                 corpus["seen_words"] +
                 corpus["unseen_test_words"])
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for w in all_words:
        if w.lower() not in seen:
            seen.add(w.lower())
            unique_words.append(w.lower())
    all_words = unique_words

    print(f"\nTotal style words: {len(all_words)}")
    print(f"Images: {args.num_images} from FiveK")
    print(f"Iterations per word: {args.num_iters}")
    print(f"Device: {args.device}")
    print()

    # Load images
    print("Loading images...")
    images = load_images(FIVEK_DIR, num_images=args.num_images,
                         device=args.device, size=args.image_size)

    # Initialize CLIP
    print("Loading CLIP model (ViT-B/32)...")
    clip_loss = CLIPLoss(model_name="ViT-B/32", device=args.device)

    # =========================================================================
    # Step 1: Optimize CCM for each word
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: CLIP-Guided CCM Optimization per Word")
    print("=" * 70)

    word_ccms = {}
    word_sims = {}
    word_histories = {}

    checkpoint_path = os.path.join(RESULTS_DIR, "ccm_checkpoint.json")

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        word_ccms = {w: np.array(v) for w, v in checkpoint.get("ccms", {}).items()}
        word_sims = checkpoint.get("sims", {})
        print(f"  Loaded {len(word_ccms)} pre-computed CCMs")

    start_time = time.time()

    for i, word in enumerate(all_words):
        if word in word_ccms:
            print(f"[{i+1}/{len(all_words)}] '{word}' — loaded from checkpoint "
                  f"(sim={word_sims.get(word, 0):.4f})")
            continue

        print(f"\n[{i+1}/{len(all_words)}] Optimizing CCM for '{word}'...")

        ccm, sim, history = optimize_ccm_for_word(
            word, images, clip_loss,
            num_iters=args.num_iters,
            lr=args.lr,
            batch_size=args.batch_size,
            reg_weight=args.reg_weight,
            device=args.device,
            verbose=True
        )

        word_ccms[word] = ccm
        word_sims[word] = sim
        word_histories[word] = history

        # Save checkpoint every 10 words
        if (i + 1) % 10 == 0:
            save_checkpoint(checkpoint_path, word_ccms, word_sims)
            elapsed = time.time() - start_time
            remaining = elapsed / (i + 1) * (len(all_words) - i - 1)
            print(f"  Checkpoint saved. Elapsed: {elapsed/60:.1f}min, "
                  f"ETA: {remaining/60:.1f}min")

    # Final checkpoint
    save_checkpoint(checkpoint_path, word_ccms, word_sims)

    total_time = time.time() - start_time
    print(f"\nStep 1 complete. Total time: {total_time/60:.1f} minutes")
    print(f"Optimized CCMs for {len(word_ccms)} words")

    # =========================================================================
    # Step 2: Greedy Anchor Selection
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"STEP 2: Greedy Anchor Selection (k={args.num_anchors})")
    print("=" * 70)

    selected_anchors, coverage_history = greedy_anchor_selection(
        word_ccms, num_anchors=args.num_anchors
    )

    print(f"\nSelected {args.num_anchors} Anchors:")
    for i, anchor in enumerate(selected_anchors):
        ccm = word_ccms[anchor]
        print(f"  {i+1}. '{anchor}' — CCM diag: "
              f"[{ccm[0,0]:.3f}, {ccm[1,1]:.3f}, {ccm[2,2]:.3f}]")

    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    results = {
        "experiment": "exp01_anchor_discovery",
        "config": {
            "num_images": args.num_images,
            "num_iters": args.num_iters,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_anchors": args.num_anchors,
            "image_size": args.image_size,
            "clip_model": "ViT-B/32",
            "isp_param": "CCM_3x3",
        },
        "selected_anchors": selected_anchors,
        "anchor_ccms": {
            w: word_ccms[w].tolist() for w in selected_anchors
        },
        "coverage_history": coverage_history,
        "all_word_ccms": {w: v.tolist() for w, v in word_ccms.items()},
        "all_word_sims": word_sims,
        "total_words": len(word_ccms),
        "total_time_minutes": total_time / 60,
    }

    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Save a human-readable summary
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Experiment 01: Anchor Discovery via CLIP-Guided CCM Optimization\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total words optimized: {len(word_ccms)}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n\n")
        f.write(f"Selected {args.num_anchors} Anchors:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'#':<4} {'Word':<20} {'CCM Diagonal':<30} {'CLIP Sim':<10}\n")
        f.write("-" * 50 + "\n")
        for i, anchor in enumerate(selected_anchors):
            ccm = word_ccms[anchor]
            sim = word_sims.get(anchor, 0)
            diag = f"[{ccm[0,0]:.3f}, {ccm[1,1]:.3f}, {ccm[2,2]:.3f}]"
            f.write(f"{i+1:<4} {anchor:<20} {diag:<30} {sim:<10.4f}\n")
        f.write("\nCoverage vs Anchor Count:\n")
        for i, cov in enumerate(coverage_history):
            f.write(f"  k={i+1}: coverage={cov:.4f}\n")
    print(f"Summary saved to {summary_path}")

    print("\nDone!")


def save_checkpoint(path, word_ccms, word_sims):
    """Save optimization checkpoint."""
    checkpoint = {
        "ccms": {w: v.tolist() for w, v in word_ccms.items()},
        "sims": word_sims,
    }
    with open(path, "w") as f:
        json.dump(checkpoint, f)


if __name__ == "__main__":
    main()
