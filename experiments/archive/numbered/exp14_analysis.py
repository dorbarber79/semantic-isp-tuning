"""
Experiment 14 analysis: 10 images x full word corpus (exp07 pipeline).
Env RESULTS_DIR selects the run folder (img_<n>/results.json inside).

Outputs (in RESULTS_DIR):
  summary_sheet.png   rows = images, cols = Original, Autotune, 14 key words
  param_by_word.png   per-word param values across images (consistency)
  stats.txt           objective output stats + per-word cross-image spread
"""

import os, re, sys, json, warnings
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw

RESULTS_DIR = os.environ.get('RESULTS_DIR', "/data/data1/semantic_isp/results/exp14_ten_images")
DNG_DIR = "/data/data1/EmotionISP_dataset/dng_filtered"
DEVICE = "cuda"
IMAGES = [('1', 'desert'), ('28', 'dog'), ('2', 'child'), ('236', 'man'), ('194', 'swan'),
          ('277', 'snow'), ('263', 'dusk'), ('384', 'backlit'), ('33', 'indoor'), ('103', 'city')]
IMAGES = [im for im in IMAGES if os.path.exists(os.path.join(RESULTS_DIR, f'img_{im[0]}', 'results.json'))]
KEY_WORDS = ['vivid', 'sunset', 'icy', 'antique', 'day', 'dreamy', 'dramatic', 'monochrome',
             'bright', 'dark', 'night', 'faded', 'noir', 'angelic']
NAMES = DifferentiableEMOVIS.PARAM_NAMES


def set_params(isp, p):
    isp.set_params(p)


def parse_autotune(log_path):
    txt = open(log_path).read()
    m = re.search(r"Autotune: wb=([-+.\d]+) sat=([-+.\d]+) br=([-+.\d]+) sh=([-+.\d]+)(?: ct=([-+.\d]+))?", txt)
    if m and m.group(5) is None:
        return dict(zip(NAMES[:4], map(float, m.groups()[:4])))
    return dict(zip(NAMES, map(float, m.groups()))) if m else None


def gamma_of(img_lin, isp):
    """Recompute the tone-mapping gamma the ISP used (after WB+sat are applied)."""
    p = isp._map_params()
    x = img_lin
    x = torch.clamp(torch.cat([x[:, :1] * (1 + p[0]), x[:, 1:2], x[:, 2:] * (1 - p[0])], 1), 0, 1)
    lum = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    t = torch.clamp(0.5 + p[2] * 0.5, 0.05, 0.95) ** 2.2
    return float(torch.clamp(torch.log(t) / torch.log(lum.median()), 0.3, 3.0))


def out_stats(y):
    L = (0.299 * y[0, 0] + 0.587 * y[0, 1] + 0.114 * y[0, 2]).flatten()
    return dict(med=float(L.median()),
                crushed=float((L < 0.02).float().mean() * 100),
                clipped=float((y[0] >= 0.995).any(0).float().mean() * 100))


def main():
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    except Exception:
        font = ImageFont.load_default()

    isp = DifferentiableEMOVIS(device=DEVICE)
    T, LH, RW = 128, 16, 70
    cols = ['Original', 'Autotune'] + KEY_WORDS
    sheet = Image.new('RGB', (RW + len(cols) * T, LH + len(IMAGES) * T), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        d.text((RW + c * T + T // 2, 2), name, fill=(255, 255, 120), font=font, anchor='mt')

    all_res, lines = {}, []
    stat_rows = []  # (image, word, br, gamma, med, crushed, clipped)
    for r, (num, label) in enumerate(IMAGES):
        rdir = os.path.join(RESULTS_DIR, f'img_{num}')
        rj = json.load(open(os.path.join(rdir, 'results.json')))
        res = rj['words']
        all_res[label] = res
        auto = rj.get('autotune') or parse_autotune(os.path.join(rdir, 'log.txt'))
        x = load_and_preprocess_raw(os.path.join(DNG_DIR, f'{num}.dng'), 512).to(DEVICE)
        d.text((4, LH + r * T + T // 2), label, fill=(255, 255, 255), font=font, anchor='lm')

        cfgs = {'Original': dict(DifferentiableEMOVIS.DEFAULT_PARAMS), 'Autotune': auto}
        cfgs.update({w: res[w]['params'] for w in res})
        with torch.no_grad():
            for w, p in cfgs.items():
                set_params(isp, p)
                y = isp(x)
                s = out_stats(y)
                stat_rows.append((label, w, p['brightness'], gamma_of(x, isp), s['med'], s['crushed'], s['clipped']))
                if w in cols:
                    im = Image.fromarray((y[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    sheet.paste(im.resize((T, T), Image.LANCZOS), (RW + cols.index(w) * T, LH + r * T))
        print(f'rendered {label}', flush=True)
    sheet.save(os.path.join(RESULTS_DIR, 'summary_sheet.png'))

    # ---- Objective stats ----
    words = list(all_res[IMAGES[0][1]].keys())
    sr = [s for s in stat_rows if s[1] not in ('Original', 'Autotune')]
    br = np.array([s[2] for s in sr]); g = np.array([s[3] for s in sr])
    crushed = np.array([s[5] for s in sr]); clipped = np.array([s[6] for s in sr])
    lines.append(f'Total word-runs: {len(sr)} ({len(IMAGES)} images x {len(words)} words)')
    lines.append(f'brightness at bound (|br|>0.45): {np.mean(np.abs(br) > 0.45) * 100:.1f}%')
    lines.append(f'gamma clamped (<=0.31 or >=2.99): {np.mean((g <= 0.31) | (g >= 2.99)) * 100:.1f}%')
    lines.append(f'runs with >5% crushed blacks (L<0.02): {np.mean(crushed > 5) * 100:.1f}%')
    lines.append(f'runs with >2% clipped pixels: {np.mean(clipped > 2) * 100:.1f}%')
    lines.append('\nPer image: Original(gamma, med) | Autotune(br, gamma) | words: mean br, %crushed>5, %clipped>2')
    for _, label in IMAGES:
        o = [s for s in stat_rows if s[0] == label and s[1] == 'Original'][0]
        a = [s for s in stat_rows if s[0] == label and s[1] == 'Autotune'][0]
        ws = [s for s in sr if s[0] == label]
        lines.append(f'  {label:<8} orig g={o[3]:.2f} med={o[4]:.2f} | auto br={a[2]:+.2f} g={a[3]:.2f} | '
                     f'words br={np.mean([s[2] for s in ws]):+.2f} '
                     f'crushed={np.mean([s[5] > 5 for s in ws]) * 100:4.0f}% clipped={np.mean([s[6] > 2 for s in ws]) * 100:4.0f}%')

    # ---- Cross-image consistency per word ----
    P = np.array([[[all_res[l][w]['params'][n] for n in NAMES] for _, l in IMAGES]
                  for w in words])  # (W, I, P)
    D = np.array([DifferentiableEMOVIS.DEFAULT_PARAMS[n] for n in NAMES])
    K = len(NAMES)
    rng = np.array([DifferentiableEMOVIS.PARAM_RANGES[n][1] - DifferentiableEMOVIS.PARAM_RANGES[n][0] for n in NAMES])
    std_norm = P.std(1) / rng  # (W, P), std as fraction of full range
    # sign agreement of the change vs the default: fraction of images agreeing with majority sign
    PD = P - D
    sign_agree = np.array([[max(np.mean(PD[i, :, k] > 0.02), np.mean(PD[i, :, k] < -0.02)) for k in range(K)]
                           for i in range(len(words))])
    lines.append('\nCross-image spread (std / range), mean over words:  ' +
                 '  '.join(f'{n}={std_norm[:, k].mean():.2f}' for k, n in enumerate(NAMES)))
    lines.append('Sign agreement (change vs default), mean over words: ' +
                 '  '.join(f'{n}={sign_agree[:, k].mean():.2f}' for k, n in enumerate(NAMES)))
    lines.append(f'\nDefaults: {DifferentiableEMOVIS.DEFAULT_PARAMS}  (sign % is of value − default)')
    lines.append(f'\n{"word":<14}' + ''.join(f'{n:>22}' for n in NAMES))
    for i, w in enumerate(words):
        lines.append(f'{w:<14}' + ''.join(f'{P[i, :, k].mean():+.2f}±{P[i, :, k].std():.2f} ({sign_agree[i, k]:.0%})'.rjust(22)
                                          for k in range(K)))

    open(os.path.join(RESULTS_DIR, 'stats.txt'), 'w').write('\n'.join(lines))
    print('\n'.join(lines[:20]))

    # ---- Plot: per-word param across images ----
    order = np.argsort(P[:, :, 2].mean(1))
    fig, axs = plt.subplots(K, 1, figsize=(22, 3.5 * K), sharex=True)
    for k, n in enumerate(NAMES):
        for j, (_, lab) in enumerate(IMAGES):
            axs[k].scatter(range(len(words)), P[order, j, k], s=10, label=lab)
        axs[k].plot(range(len(words)), P[order, :, k].mean(1), 'k-', lw=1)
        axs[k].set_ylabel(n); axs[k].axhline(D[k], color='gray', lw=0.5)
    axs[-1].set_xticks(range(len(words))); axs[-1].set_xticklabels([words[i] for i in order], rotation=90, fontsize=7)
    axs[0].legend(ncol=10, fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, 'param_by_word.png'), dpi=110)
    print('saved')


if __name__ == '__main__':
    main()
