"""
Experiment 06: Real EMOVIS ISP + CMA-ES Optimization

Uses the ACTUAL EMOVIS ISP (NumPy/OpenCV, 16-bit, 11 stages) as a black box.
Optimizes 7 ISP parameters per word using CMA-ES (gradient-free).

CLIP evaluates the output image and CMA-ES searches for the best parameters.

Pipeline:
    RAW DNG → EMOVIS ISP (full 16-bit, 11 stages) → JPEG → CLIP
    CMA-ES optimizes the 7 ISP params to maximize CLIP alignment with text.
"""

import os, sys, json, warnings, time
import numpy as np
from PIL import Image
from pathlib import Path
import cma

warnings.filterwarnings('ignore', message="Can't initialize NVML")

# Add EMOVIS to path
sys.path.insert(0, '/home/administrator/code/EMOVIS')
from emovis.isp import process_emotion_isp_pipeline

import torch
import torch.nn.functional as F
import clip

DNG_DIR = "/data/data1/EmotionISP_dataset/dng_filtered"
RESULTS_DIR = "/data/data1/semantic_isp/results/exp06_emovis_cmaes"
DEVICE = "cuda"
TMP_DIR = "/tmp/emovis_opt"

WORDS = ['colorful', 'dark', 'bright', 'warm', 'cool',
         'monochrome', 'saturated', 'faded', 'vivid', 'muted']

# Parameter bounds matching EMOVIS
PARAM_NAMES = ['saturation', 'brightness', 'sharpness',
               'red_green_bias', 'yellow_blue_bias', 'contrast', 'hue_shift']
PARAM_BOUNDS = {
    'saturation':      (-0.5, 0.5),
    'brightness':      (-0.5, 0.5),
    'sharpness':       (-1.0, 1.0),
    'red_green_bias':  (-0.3, 0.3),
    'yellow_blue_bias':(-0.3, 0.3),
    'contrast':        (-0.5, 0.5),
    'hue_shift':       (-30, 30),
}


class CLIPEvaluator:
    """Evaluates CLIP similarity between a JPEG image and text."""

    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load('ViT-B/32', device=device)
        self.model.eval()
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                      device=device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                     device=device).view(1, 3, 1, 1)

    def encode_text(self, text):
        tok = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tok)
            return feat / feat.norm(dim=-1, keepdim=True)

    def evaluate_image(self, img_path, text_feat, neutral_feat):
        """Compute CLIP similarity for an image file."""
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        resized = F.interpolate(img_tensor, size=(224, 224), mode='bicubic', align_corners=False)
        normed = (resized - self.clip_mean) / self.clip_std

        with torch.no_grad():
            img_feat = self.model.encode_image(normed)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim_target = (img_feat @ text_feat.T).item()
            sim_neutral = (img_feat @ neutral_feat.T).item()

        return sim_target, sim_neutral


def params_vec_to_dict(vec):
    """Convert parameter vector to EMOVIS dict."""
    d = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[name]
        d[name] = float(np.clip(vec[i], lo, hi))
    return d


def objective(params_vec, dng_path, evaluator, text_feat, neutral_feat, tmp_path):
    """CMA-ES objective: run EMOVIS ISP, evaluate with CLIP. Returns negative similarity."""
    params = params_vec_to_dict(params_vec)

    try:
        process_emotion_isp_pipeline(
            dng_path=dng_path,
            adaptation_params=params,
            output_path=Path(tmp_path),
            jpeg_quality=90,
        )
        sim_target, sim_neutral = evaluator.evaluate_image(tmp_path, text_feat, neutral_feat)
        # Objective: maximize (sim_target - 0.5*sim_neutral) → minimize negative
        score = sim_target - 0.5 * sim_neutral
        return -score  # CMA-ES minimizes
    except Exception as e:
        print(f"    ISP error: {e}")
        return 0.0  # bad score


def optimize_word(word, dng_path, evaluator, max_evals=150):
    """Optimize ISP params for one word using CMA-ES."""
    text_feat = evaluator.encode_text(f"a {word} photo")
    neutral_feat = evaluator.encode_text("a photo")
    tmp_path = os.path.join(TMP_DIR, f"opt_{word}.jpg")

    # CMA-ES: start from neutral (all zeros), sigma=0.15
    x0 = [0.0] * 7
    sigma0 = 0.15

    # Bounds for CMA-ES
    lower = [PARAM_BOUNDS[n][0] for n in PARAM_NAMES]
    upper = [PARAM_BOUNDS[n][1] for n in PARAM_NAMES]

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'bounds': [lower, upper],
        'maxfevals': max_evals,
        'verbose': -9,  # quiet
        'popsize': 10,
    })

    sim_history = []
    best_sim = -1
    best_params = None
    eval_count = 0

    while not es.stop():
        solutions = es.ask()
        fitnesses = []
        for sol in solutions:
            f = objective(sol, dng_path, evaluator, text_feat, neutral_feat, tmp_path)
            fitnesses.append(f)
            eval_count += 1

            # Track best
            sim = -f + 0.5 * evaluator.evaluate_image(tmp_path, text_feat, neutral_feat)[1]
            # Actually just use -f as approximate sim for tracking
            current_sim = -f
            if current_sim > best_sim:
                best_sim = current_sim  # this is (sim_target - 0.5*sim_neutral)
                best_params = params_vec_to_dict(sol)

        es.tell(solutions, fitnesses)
        sim_history.append(best_sim)

    # Final evaluation with best params
    process_emotion_isp_pipeline(
        dng_path=dng_path,
        adaptation_params=best_params,
        output_path=Path(tmp_path),
        jpeg_quality=90,
    )
    final_sim_target, _ = evaluator.evaluate_image(tmp_path, text_feat, neutral_feat)

    return best_params, final_sim_target, sim_history, eval_count


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    print("=" * 60)
    print("Experiment 06: Real EMOVIS ISP + CMA-ES")
    print("Full 16-bit ISP, gradient-free optimization")
    print(f"7 params: {PARAM_NAMES}")
    print("=" * 60)

    # Pick one DNG
    all_dngs = sorted([f for f in os.listdir(DNG_DIR) if f.endswith('.dng')])
    dng_name = all_dngs[0]  # 1.dng
    dng_path = os.path.join(DNG_DIR, dng_name)
    print(f"\nRAW: {dng_name}")

    evaluator = CLIPEvaluator(device=DEVICE)

    # First, show the neutral image
    neutral_path = os.path.join(RESULTS_DIR, 'neutral.jpg')
    process_emotion_isp_pipeline(
        dng_path=dng_path,
        adaptation_params={n: 0.0 for n in PARAM_NAMES},
        output_path=Path(neutral_path),
        jpeg_quality=95,
    )
    print(f"Neutral saved: {neutral_path}")

    results = {}
    t0 = time.time()

    for i, word in enumerate(WORDS):
        print(f"\n[{i+1}/{len(WORDS)}] '{word}'...", flush=True)
        params, sim, hist, evals = optimize_word(
            word, dng_path, evaluator, max_evals=150
        )
        results[word] = {'params': params, 'clip_sim': sim, 'num_evals': evals}

        p = params
        print(f"  sim={sim:.4f} ({evals} evals)  "
              f"sat={p['saturation']:+.3f} br={p['brightness']:+.3f} "
              f"sh={p['sharpness']:+.3f} co={p['contrast']:+.3f} "
              f"rg={p['red_green_bias']:+.3f} yb={p['yellow_blue_bias']:+.3f} "
              f"hue={p['hue_shift']:+.1f}")

        # Save the best image for this word
        word_path = os.path.join(RESULTS_DIR, f'word_{word}.jpg')
        process_emotion_isp_pipeline(
            dng_path=dng_path,
            adaptation_params=params,
            output_path=Path(word_path),
            jpeg_quality=95,
        )

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/len(WORDS):.0f}s/word)")

    # Build grid from saved JPEGs
    from PIL import ImageDraw, ImageFont
    sorted_words = sorted(WORDS, key=lambda w: -results[w]['clip_sim'])

    n_cols = 6
    n_rows = 2
    thumb = (256, 170)
    label_h = 24
    cell_w, cell_h = thumb

    canvas = Image.new('RGB', (n_cols * cell_w, n_rows * (cell_h + label_h)), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 11)
    except:
        font = ImageFont.load_default()

    # Original
    orig = Image.open(neutral_path).convert('RGB')
    canvas.paste(orig.resize(thumb, Image.LANCZOS), (0, label_h))
    draw.text((cell_w // 2, 4), "Original", fill=(255, 255, 255), font=font, anchor='mt')

    for idx, word in enumerate(sorted_words):
        pos = idx + 1
        col = pos % n_cols
        row = pos // n_cols
        x = col * cell_w
        y = row * (cell_h + label_h)

        word_img = Image.open(os.path.join(RESULTS_DIR, f'word_{word}.jpg')).convert('RGB')
        canvas.paste(word_img.resize(thumb, Image.LANCZOS), (x, y + label_h))
        sim = results[word]['clip_sim']
        draw.text((x + cell_w // 2, y + 4), f"{word} ({sim:.3f})",
                  fill=(100, 255, 100), font=font, anchor='mt')

    grid_path = os.path.join(RESULTS_DIR, 'grid.png')
    canvas.save(grid_path, quality=95)
    print(f"Grid: {grid_path}")

    # Save results
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump({
            'description': 'Real EMOVIS ISP + CMA-ES gradient-free optimization',
            'dng': dng_name,
            'isp': 'EMOVIS full 16-bit pipeline (11 stages)',
            'optimizer': 'CMA-ES (max 150 evaluations per word)',
            'params': PARAM_NAMES,
            'words': results,
            'total_time_sec': elapsed,
        }, f, indent=2)

    # Summary
    print(f"\n{'Word':<12} {'Sim':>6} {'Sat':>6} {'Br':>6} {'Sh':>6} {'Co':>6} {'R/G':>6} {'Y/B':>6} {'Hue':>6}")
    print("-" * 75)
    for word in sorted_words:
        p = results[word]['params']
        s = results[word]['clip_sim']
        print(f"{word:<12} {s:.4f} {p['saturation']:+.3f} {p['brightness']:+.3f} "
              f"{p['sharpness']:+.3f} {p['contrast']:+.3f} "
              f"{p['red_green_bias']:+.3f} {p['yellow_blue_bias']:+.3f} "
              f"{p['hue_shift']:+5.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
