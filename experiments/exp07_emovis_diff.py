"""
Experiment 07: Differentiable EMOVIS ISP from RAW  (main script)

Uses load_and_preprocess_raw (non-diff: demosaic + WB + denoise)
then DifferentiableEMOVIS (diff: WB → saturation → tone mapping → contrast → sharpening).
Each stage uses luminance-preserving chroma compression to avoid channel clipping.

All 92 corpus words on one image, gradient-based optimization.
VLM: SigLIP-B/16. Loss: 1.0*semantic (direct) + 1.0*realism + 1.0*beauty + reg.
Realism and beauty are two-directional (positive vs negative prompt, relative to the
neutral image): improvement is rewarded and degradation is penalized.
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
import open_clip

sys.path.insert(0, os.path.dirname(__file__))
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw

DNG_DIR = "/data/data1/EmotionISP_dataset/dng_filtered"
RESULTS_DIR = os.environ.get('RESULTS_DIR', "/data/data1/semantic_isp/results/exp07_emovis_diff")
DEVICE = "cuda"

# ---- Decided configuration (user decisions, Nov 1-2 2026) ----
VLM_NAME = 'ViT-B-16-SigLIP'      # SigLIP-B/16 (webli), replaces CLIP ViT-B/32
SEMANTIC_WEIGHT = 1.0             # direct semantic loss (not directional) — unchanged
REALISM_WEIGHT = 1.0              # two-directional realism loss (Nov 2: equal weights)
BEAUTY_WEIGHT = 1.0               # two-directional beauty loss (Nov 2: re-added, equal weight)
# (positive, negative) prompt pairs. Chosen by probe on 10 images (Nov 2): with SigLIP these
# pairs score degraded ISP outputs (over/under saturation, exposure, blur, WB, flat contrast)
# lower than the neutral output. "beautiful/ugly" and "Good/Bad photo." pairs were rejected:
# SigLIP scored grayscale/dark images HIGHER with them.
REALISM_PROMPTS = ("a natural color photo", "an artificial, over-processed photo")
BEAUTY_PROMPTS = ("a photo with good exposure and pleasing colors",
                  "a photo with bad exposure and unpleasant colors")
# Image normalization fed to the VLM. 'siglip' = (0.5, 0.5) as SigLIP was trained.
# 'clip' = CLIP mean/std, which exp09-exp12 used by mistake (kept for reproduction).
VLM_NORM = os.environ.get('VLM_NORM', 'siglip')


class SigLIP:
    """SigLIP-B/16 with the same encode_text/encode_image interface as OpenAI clip."""

    def __init__(self, device):
        self.model, _, _ = open_clip.create_model_and_transforms(VLM_NAME, pretrained='webli', device=device)
        self.model.eval()
        self._tok = open_clip.get_tokenizer(VLM_NAME)
        self.device = device

    def tokenize(self, texts):
        return self._tok(texts).to(self.device)

    def encode_text(self, tokens):
        return self.model.encode_text(tokens)

    def encode_image(self, x):
        return self.model.encode_image(x)


def vlm_norm_stats(device):
    if VLM_NORM == 'clip':
        mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    return (torch.tensor(mean, device=device).view(1, 3, 1, 1),
            torch.tensor(std, device=device).view(1, 3, 1, 1))

STYLE_CORPUS = "/data/data1/semantic_isp/datasets/style_words/style_corpus.json"

def load_all_words():
    """Load all single words from the style corpus."""
    import json as _json
    with open(STYLE_CORPUS) as f:
        corpus = _json.load(f)
    all_words = corpus['anchors'] + corpus['seen_words'] + corpus['unseen_test_words']
    seen = set()
    return [w.lower() for w in all_words if not (w.lower() in seen or seen.add(w.lower()))]

WORDS = load_all_words()


class PromptPairScore:
    """Two-directional prompt score: q(img) = cos(img, positive) − cos(img, negative).

    CLIP-IQA (Wang et al., AAAI 2023) antonym-pair idea, used as a differentiable loss.
    The difference of cosines is the log-odds of the pair softmax (up to the VLM
    temperature), so it is linear and does not saturate.
    """

    def __init__(self, vlm, prompts):
        self.prompts = prompts
        with torch.no_grad():
            t = vlm.encode_text(vlm.tokenize(list(prompts)))
            t = t / t.norm(dim=-1, keepdim=True)
        self.pos, self.neg = t[0:1], t[1:2]

    def __call__(self, img_feat):
        return (img_feat @ self.pos.T - img_feat @ self.neg.T).mean()


def build_guards(vlm, orig_feat):
    """Realism and beauty scorers plus their scores on the neutral (reference) image."""
    guards = {}
    for name, prompts in (('realism', REALISM_PROMPTS), ('beauty', BEAUTY_PROMPTS)):
        scorer = PromptPairScore(vlm, prompts)
        with torch.no_grad():
            guards[name] = (scorer, scorer(orig_feat))
    return guards


def optimize_for_word(word, img_tensor, guards,
                      clip_model, clip_mean, clip_std,
                      num_iters=500, lr=0.03, reg_weight=0.003,
                      semantic_weight=SEMANTIC_WEIGHT, realism_weight=REALISM_WEIGHT,
                      beauty_weight=BEAUTY_WEIGHT):
    """Optimize ISP params using semantic VLM loss + two-directional quality losses.

    Loss = semantic_weight * -cos(VLM(processed), VLM("a {word} photo"))
         + realism_weight  * (q_realism(orig) − q_realism(processed))
         + beauty_weight   * (q_beauty(orig)  − q_beauty(processed))
         + reg_weight * ||params − defaults||²     (in raw/tanh space)

    q(·) = cos(·, positive prompt) − cos(·, negative prompt), orig = neutral ISP output.
    The quality terms are negative when the image improves (rewarded) and positive
    when it degrades (penalized). word=None → autotune (no semantic term).
    Returns the params of the iterate with the lowest total loss.
    """
    text_feat = None
    if word is not None:
        with torch.no_grad():
            text_feat = clip_model.encode_text(clip_model.tokenize([f"a {word} photo"]))
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    isp = DifferentiableEMOVIS(device=DEVICE)
    optimizer = torch.optim.Adam(isp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters,
                                                            eta_min=lr * 0.1)
    (realism, orig_realism), (beauty, orig_beauty) = guards['realism'], guards['beauty']

    loss_hist, sim_hist = [], []
    best_loss, best_sim = float('inf'), float('nan')
    best_params = isp.get_params()

    for it in range(num_iters):
        optimizer.zero_grad()
        params_now = isp.get_params()
        processed = isp(img_tensor)

        resized = F.interpolate(processed, size=(224, 224), mode='bicubic', align_corners=False)
        normed = (resized - clip_mean) / clip_std
        img_feat = clip_model.encode_image(normed)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # Semantic loss (direct, unchanged)
        if text_feat is not None:
            sim_target = (img_feat @ text_feat.T).mean()
            L_semantic = -sim_target
        else:
            sim_target = torch.tensor(float('nan'))
            L_semantic = torch.zeros((), device=img_feat.device)

        # Two-directional quality losses (relative to the neutral image)
        L_realism = orig_realism - realism(img_feat)
        L_beauty = orig_beauty - beauty(img_feat)

        L_reg = reg_weight * ((isp.raw_params - isp.raw_default) ** 2).sum()

        total_loss = (semantic_weight * L_semantic
                      + realism_weight * L_realism
                      + beauty_weight * L_beauty
                      + L_reg)

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_sim = sim_target.item()
            best_params = params_now

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_hist.append(total_loss.item())
        sim_hist.append(sim_target.item())

    return best_params, best_sim, loss_hist, sim_hist


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Exp 07: Differentiable EMOVIS ISP from RAW")
    print("Non-diff: demosaic + WB + denoise")
    print("Diff: WB → sat → gamma → contrast → sharp")
    print("Gamut protection: luma-preserving chroma compression after every stage")
    print("5 learnable params, gradient-based optimization")
    print(f"VLM: {VLM_NAME} (norm={VLM_NORM})")
    print(f"Loss: {SEMANTIC_WEIGHT}*semantic + {REALISM_WEIGHT}*realism + "
          f"{BEAUTY_WEIGHT}*beauty (two-directional) + reg")
    print(f"Realism prompts: {REALISM_PROMPTS}")
    print(f"Beauty prompts:  {BEAUTY_PROMPTS}")
    print("=" * 60)

    # Load RAW
    all_dngs = sorted([f for f in os.listdir(DNG_DIR) if f.endswith('.dng')])
    if os.environ.get('DNG_PATH'):
        dng_path = os.environ['DNG_PATH']
        dng_name = os.path.basename(dng_path)
    else:
        dng_idx = int(os.environ.get('DNG_INDEX', '0'))
        dng_name = all_dngs[dng_idx]
        dng_path = os.path.join(DNG_DIR, dng_name)
    print(f"\nRAW: {dng_name}")

    img_tensor = load_and_preprocess_raw(dng_path, target_size=512).to(DEVICE)
    print(f"Loaded: {img_tensor.shape}, range [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")

    clip_model = SigLIP(DEVICE)
    clip_mean, clip_std = vlm_norm_stats(DEVICE)

    # Neutral (reference) image features and its realism/beauty scores
    isp_neutral = DifferentiableEMOVIS(device=DEVICE)
    with torch.no_grad():
        orig_processed = isp_neutral(img_tensor)
        resized = F.interpolate(orig_processed, size=(224, 224), mode='bicubic', align_corners=False)
        normed = (resized - clip_mean) / clip_std
        orig_feat = clip_model.encode_image(normed)
        orig_feat = orig_feat / orig_feat.norm(dim=-1, keepdim=True)

    guards = build_guards(clip_model, orig_feat)
    print(f"Original realism score: {guards['realism'][1].item():+.4f}")
    print(f"Original beauty score:  {guards['beauty'][1].item():+.4f}")

    # Optimize
    results = {}
    all_loss, all_sim = {}, {}
    t0 = time.time()

    for i, word in enumerate(WORDS):
        print(f"\n[{i+1}/{len(WORDS)}] '{word}'...", flush=True)
        params, sim, loss_h, sim_h = optimize_for_word(
            word, img_tensor, guards,
            clip_model, clip_mean, clip_std,
            num_iters=500, lr=0.03, reg_weight=0.003,
        )
        results[word] = {'params': params, 'clip_sim': sim}
        all_loss[word] = loss_h
        all_sim[word] = sim_h
        p = params
        print(f"  sim={sim:.4f}  wb={p['wb_warmth']:+.3f} sat={p['saturation']:+.3f} "
              f"br={p['brightness']:+.3f} sh={p['sharpness']:+.3f} ct={p['contrast']:+.3f}")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/len(WORDS):.1f}s/word)")

    # Convergence plot skipped for large word sets (>10 words)

    # Autotune: realism + beauty only, no semantic
    print("\nRunning autotune (realism + beauty, no semantic)...")
    autotune_params, _, _, _ = optimize_for_word(
        None, img_tensor, guards,
        clip_model, clip_mean, clip_std,
        num_iters=500, lr=0.03, reg_weight=0.003,
    )
    p = autotune_params
    print(f"  Autotune: wb={p['wb_warmth']:+.3f} sat={p['saturation']:+.3f} "
          f"br={p['brightness']:+.3f} sh={p['sharpness']:+.3f} ct={p['contrast']:+.3f}")

    # Visual grid
    sorted_words = sorted(WORDS, key=lambda w: -results[w]['clip_sim'])
    isp = DifferentiableEMOVIS(device=DEVICE)

    n_total = len(sorted_words) + 2  # +1 original, +1 autotune
    n_cols = 10
    n_rows = (n_total + n_cols - 1) // n_cols
    thumb = (180, 180)
    label_h = 20
    cell_w, cell_h = thumb

    canvas = Image.new('RGB', (n_cols*cell_w, n_rows*(cell_h+label_h)), (25,25,25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 9)
    except:
        font = ImageFont.load_default()

    # Original (neutral ISP)
    with torch.no_grad():
        orig = isp(img_tensor)
        orig_np = (orig[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    canvas.paste(Image.fromarray(orig_np).resize(thumb, Image.LANCZOS), (0, label_h))
    draw.text((cell_w//2, 4), "Original", fill=(255,255,255), font=font, anchor='mt')

    # Autotune (position 1)
    with torch.no_grad():
        for pi, name in enumerate(DifferentiableEMOVIS.PARAM_NAMES):
            lo, hi = DifferentiableEMOVIS.PARAM_RANGES[name]
            mid, half = (hi+lo)/2.0, (hi-lo)/2.0
            norm = (autotune_params[name]-mid)/half
            norm = max(-0.999, min(0.999, norm))
            isp.raw_params.data[pi] = np.arctanh(norm)
        auto_out = isp(img_tensor)
        auto_np = (auto_out[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    canvas.paste(Image.fromarray(auto_np).resize(thumb, Image.LANCZOS), (cell_w, label_h))
    draw.text((cell_w + cell_w//2, 4), "Autotune", fill=(255,200,100), font=font, anchor='mt')

    for idx, word in enumerate(sorted_words):
        pos = idx + 2  # +2 for original and autotune
        col = pos % n_cols
        row = pos // n_cols
        x, y = col * cell_w, row * (cell_h + label_h)

        p = results[word]['params']
        with torch.no_grad():
            # Set params
            vals = [p[n] for n in DifferentiableEMOVIS.PARAM_NAMES]
            # Inverse of tanh mapping to get raw_params
            for pi, name in enumerate(DifferentiableEMOVIS.PARAM_NAMES):
                lo, hi = DifferentiableEMOVIS.PARAM_RANGES[name]
                mapped_val = vals[pi]
                # Inverse: raw = atanh((val - mid) / half_range)
                mid = (hi + lo) / 2.0
                half = (hi - lo) / 2.0
                normalized = (mapped_val - mid) / half
                normalized = max(-0.999, min(0.999, normalized))
                raw_val = np.arctanh(normalized)
                isp.raw_params.data[pi] = raw_val

            processed = isp(img_tensor)
            proc_np = (processed[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)

        canvas.paste(Image.fromarray(proc_np).resize(thumb, Image.LANCZOS), (x, y+label_h))
        sim = results[word]['clip_sim']
        draw.text((x+cell_w//2, y+4), f"{word} ({sim:.3f})",
                  fill=(100,255,100), font=font, anchor='mt')

    grid_path = os.path.join(RESULTS_DIR, 'grid.png')
    canvas.save(grid_path, quality=95)
    print(f"Grid: {grid_path}")

    # Save
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump({
            'description': 'Differentiable EMOVIS ISP from RAW DNG',
            'pipeline': {
                'non_diff': 'rawpy demosaic (AHD) + camera WB + bilateral denoise',
                'diff': 'wb → saturation → tone_mapping (brightness) → contrast → sharpening',
            },
            'dng': dng_name,
            'config': {'vlm': VLM_NAME, 'vlm_norm': VLM_NORM, 'semantic_weight': SEMANTIC_WEIGHT,
                       'realism_weight': REALISM_WEIGHT, 'beauty_weight': BEAUTY_WEIGHT,
                       'realism_prompts': REALISM_PROMPTS, 'beauty_prompts': BEAUTY_PROMPTS,
                       'quality_loss': 'two-directional: q(orig) - q(proc), q = cos(pos) - cos(neg)',
                       'gamut_protection': 'luma-preserving chroma compression after every stage',
                       'defaults': DifferentiableEMOVIS.DEFAULT_PARAMS,
                       'reg_weight': 0.003, 'iters': 500, 'lr': 0.03},
            'autotune': autotune_params,
            'words': results,
            'time_sec': elapsed,
        }, f, indent=2)

    # Summary
    print(f"\n{'Word':<12} {'Sim':>6} {'WB':>6} {'Sat':>6} {'Br':>6} {'Sh':>6} {'Ct':>6}")
    print("-" * 50)
    for word in sorted_words:
        p = results[word]['params']
        s = results[word]['clip_sim']
        print(f"{word:<12} {s:.4f} {p['wb_warmth']:+.3f} {p['saturation']:+.3f} "
              f"{p['brightness']:+.3f} {p['sharpness']:+.3f} {p['contrast']:+.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
