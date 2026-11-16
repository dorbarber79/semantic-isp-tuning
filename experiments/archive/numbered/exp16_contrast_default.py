"""
Experiment 16: Default contrast ("no semantics") + new two-directional quality loss.

Question: with no text, which contrast should the ISP apply by default?
"No semantics" = the loss without the semantic term:
    realism_weight * (q_r(orig) − q_r(proc)) + beauty_weight * (q_b(orig) − q_b(proc))
so the best default contrast maximizes Q(c) = q_realism(c) + q_beauty(c) (equal weights).

Part A (SWEEP): all DNGs in dng_filtered, contrast c ∈ [−1, 1] (41 values), other params at
          default. Per image ΔQ(c) = Q(c) − Q(0). Default = argmax of the mean curve
          (bootstrap CI over images).
Part B (AUTOTUNE): gradient optimization of all 5 params with the same loss (word=None) on
          the 10 exp14 images, starting from the defaults. Cross-check of A.

Env: PART=A|B, SHARD=i/n (split images across processes), RESULTS_DIR. MERGE=1 → plots.
"""

import os, sys, json, warnings
import numpy as np
import torch
import torch.nn.functional as F
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw
import exp07_emovis_diff as E

DNG_DIR = "/data/data1/EmotionISP_dataset/dng_filtered"
RESULTS_DIR = os.environ.get('RESULTS_DIR', "/data/data1/semantic_isp/results/exp16_contrast_default")
DEVICE = "cuda"
CONTRASTS = np.round(np.linspace(-1.0, 1.0, 41), 3)
EXP14_IMAGES = [('1', 'desert'), ('28', 'dog'), ('2', 'child'), ('236', 'man'), ('194', 'swan'),
                ('277', 'snow'), ('263', 'dusk'), ('384', 'backlit'), ('33', 'indoor'), ('103', 'city')]


def embed(vlm, mean, std, y):
    f = vlm.encode_image((F.interpolate(y, size=(224, 224), mode='bicubic', align_corners=False) - mean) / std)
    return f / f.norm(dim=-1, keepdim=True)


def shard(items):
    i, n = map(int, os.environ.get('SHARD', '0/1').split('/'))
    return items[i::n]


def part_a():
    vlm = E.SigLIP(DEVICE)
    mean, std = E.vlm_norm_stats(DEVICE)
    realism = E.PromptPairScore(vlm, E.REALISM_PROMPTS)
    beauty = E.PromptPairScore(vlm, E.BEAUTY_PROMPTS)
    isp = DifferentiableEMOVIS(DEVICE)
    names = shard(sorted(f for f in os.listdir(DNG_DIR) if f.endswith('.dng')))
    out = {}
    for name in names:
        x = load_and_preprocess_raw(os.path.join(DNG_DIR, name), 512).to(DEVICE)
        r, b = [], []
        for c in CONTRASTS:
            isp.set_params({'contrast': float(c)})
            with torch.no_grad():
                f = embed(vlm, mean, std, isp(x))
                r.append(realism(f).item())
                b.append(beauty(f).item())
        out[name] = {'realism': r, 'beauty': b}
        q = np.array(r) + np.array(b)
        print(f"{name:<10} argmax c={CONTRASTS[q.argmax()]:+.2f}", flush=True)
    tag = os.environ.get('SHARD', '0/1').replace('/', 'of')
    json.dump({'contrasts': CONTRASTS.tolist(), 'images': out},
              open(os.path.join(RESULTS_DIR, f'sweep_{tag}.json'), 'w'))


def part_b():
    vlm = E.SigLIP(DEVICE)
    mean, std = E.vlm_norm_stats(DEVICE)
    out = {}
    for num, lab in shard(EXP14_IMAGES):
        x = load_and_preprocess_raw(os.path.join(DNG_DIR, f'{num}.dng'), 512).to(DEVICE)
        with torch.no_grad():
            f0 = embed(vlm, mean, std, DifferentiableEMOVIS(DEVICE)(x))
        guards = E.build_guards(vlm, f0)
        p, _, loss_h, _ = E.optimize_for_word(None, x, guards, vlm, mean, std,
                                              num_iters=500, lr=0.03, reg_weight=0.003)
        out[num] = {'label': lab, 'params': p, 'loss_first': loss_h[0], 'loss_best': min(loss_h)}
        print(f"{num:>4} {lab:<8} " + ' '.join(f"{n[:3]}={v:+.3f}" for n, v in p.items())
              + f"  loss {loss_h[0]:+.4f} → {min(loss_h):+.4f}", flush=True)
    tag = os.environ.get('SHARD', '0/1').replace('/', 'of')
    json.dump(out, open(os.path.join(RESULTS_DIR, f'autotune_{tag}.json'), 'w'), indent=2)


def merge():
    import glob
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    imgs = {}
    for fn in glob.glob(os.path.join(RESULTS_DIR, 'sweep_*.json')):
        imgs.update(json.load(open(fn))['images'])
    names = sorted(imgs)
    R = np.array([imgs[n]['realism'] for n in names])
    B = np.array([imgs[n]['beauty'] for n in names])
    i0 = int(np.argmin(np.abs(CONTRASTS)))
    dR, dB = R - R[:, i0:i0 + 1], B - B[:, i0:i0 + 1]
    dQ = dR + dB
    mean_q = dQ.mean(0)
    best = CONTRASTS[mean_q.argmax()]
    per_img = CONTRASTS[dQ.argmax(1)]
    rng = np.random.default_rng(0)
    boot = [CONTRASTS[dQ[rng.integers(0, len(names), len(names))].mean(0).argmax()] for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    lines = [f"images: {len(names)}",
             f"default contrast (argmax of mean ΔQ, Q = realism + beauty): {best:+.2f}",
             f"bootstrap 95% CI: [{lo:+.2f}, {hi:+.2f}]",
             f"realism-only argmax: {CONTRASTS[dR.mean(0).argmax()]:+.2f}   "
             f"beauty-only argmax: {CONTRASTS[dB.mean(0).argmax()]:+.2f}",
             f"per-image argmax: median {np.median(per_img):+.2f}, mean {per_img.mean():+.2f}, "
             f"std {per_img.std():.2f}",
             f"per-image argmax > 0: {np.mean(per_img > 0):.0%}, < 0: {np.mean(per_img < 0):.0%}, "
             f"= 0: {np.mean(per_img == 0):.0%}",
             f"mean ΔQ at best: {mean_q.max()*1e3:+.2f}e-3 ; images improved at best: "
             f"{np.mean(dQ[:, mean_q.argmax()] > 0):.0%}",
             "", "c      meanΔR(e-3)  meanΔB(e-3)  meanΔQ(e-3)  frac ΔQ>0"]
    for j, c in enumerate(CONTRASTS):
        if j % 2 == 0:
            lines.append(f"{c:+.2f}   {dR[:, j].mean()*1e3:+8.2f}    {dB[:, j].mean()*1e3:+8.2f}    "
                         f"{mean_q[j]*1e3:+8.2f}     {np.mean(dQ[:, j] > 0):.0%}")
    auto = {}
    for fn in glob.glob(os.path.join(RESULTS_DIR, 'autotune_*.json')):
        auto.update(json.load(open(fn)))
    if auto:
        lines += ["", "Part B autotune (gradient, all 5 params, realism+beauty):"]
        for num, a in auto.items():
            lines.append(f"  {num:>4} {a['label']:<8} " + ' '.join(f"{n[:3]}={v:+.3f}" for n, v in a['params'].items()))
        ct = np.array([a['params']['contrast'] for a in auto.values()])
        lines.append(f"  autotune contrast: mean {ct.mean():+.3f}, median {np.median(ct):+.3f}, std {ct.std():.3f}")
    txt = '\n'.join(lines)
    open(os.path.join(RESULTS_DIR, 'summary.txt'), 'w').write(txt)
    print(txt)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for q in dQ:
        ax[0].plot(CONTRASTS, q * 1e3, color='gray', alpha=0.15, lw=0.8)
    ax[0].plot(CONTRASTS, dR.mean(0) * 1e3, label='realism', lw=2)
    ax[0].plot(CONTRASTS, dB.mean(0) * 1e3, label='beauty', lw=2)
    ax[0].plot(CONTRASTS, mean_q * 1e3, 'k', label='realism + beauty', lw=2.5)
    ax[0].axvline(best, color='r', ls='--', label=f'default = {best:+.2f}')
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].set_xlabel('contrast'); ax[0].set_ylabel('Δscore vs contrast 0 (×1e-3)')
    ax[0].set_title(f'Mean over {len(names)} images (gray: per image ΔQ)'); ax[0].legend()
    ax[1].hist(per_img, bins=np.linspace(-1.025, 1.025, 42), color='steelblue')
    ax[1].axvline(best, color='r', ls='--')
    ax[1].set_xlabel('per-image best contrast'); ax[1].set_ylabel('# images')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'contrast_default.png'), dpi=110)


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.environ.get('MERGE'):
        merge()
    elif os.environ.get('PART', 'A') == 'A':
        part_a()
    else:
        part_b()
