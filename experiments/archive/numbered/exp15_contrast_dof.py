"""
Experiment 15: Contrast DoF check (5-param ISP).

Decided config as exp07 (SigLIP-B/16; loss at the time: semantic 1.0 + realism 3.5,
no beauty — rerun now uses the current exp07 loss), with the new contrast stage.
Contrast-relevant words on the exp14 images.

Env: IMAGES="1 28 ..." (subset for this process), RESULTS_DIR.
Outputs per image: results_<n>.json. Run with MERGE=1 to build grid.png + table.
"""

import os, sys, json, warnings
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw
import exp07_emovis_diff as E

DNG_DIR = "/data/data1/EmotionISP_dataset/dng_filtered"
RESULTS_DIR = os.environ.get('RESULTS_DIR', "/data/data1/semantic_isp/results/exp15_contrast")
DEVICE = "cuda"
ALL_IMAGES = [('1', 'desert'), ('28', 'dog'), ('2', 'child'), ('236', 'man'), ('194', 'swan'),
              ('277', 'snow'), ('263', 'dusk'), ('384', 'backlit'), ('33', 'indoor'), ('103', 'city')]
WORDS = ['dramatic', 'noir', 'gritty', 'bold', 'crisp', 'cinematic',
         'faded', 'hazy', 'foggy', 'dreamy', 'soft', 'washed']
NAMES = DifferentiableEMOVIS.PARAM_NAMES


def set_params(isp, p):
    isp.set_params(p)


def run(images):
    vlm = E.SigLIP(DEVICE)
    mean, std = E.vlm_norm_stats(DEVICE)

    for num in images:
        x = load_and_preprocess_raw(os.path.join(DNG_DIR, f'{num}.dng'), 512).to(DEVICE)
        with torch.no_grad():
            o = DifferentiableEMOVIS(DEVICE)(x)
            f = vlm.encode_image((F.interpolate(o, size=(224, 224), mode='bicubic', align_corners=False) - mean) / std)
            f = f / f.norm(dim=-1, keepdim=True)
        guards = E.build_guards(vlm, f)
        res = {}
        for w in WORDS:
            p, s, _, _ = E.optimize_for_word(
                w, x, guards, vlm, mean, std,
                num_iters=500, lr=0.03, reg_weight=0.003)
            res[w] = {'params': p, 'sim': s}
            print(f"{num:>4} {w:<10} " + ' '.join(f"{n[:3]}={p[n]:+.2f}" for n in NAMES), flush=True)
        json.dump(res, open(os.path.join(RESULTS_DIR, f'results_{num}.json'), 'w'), indent=2)


def merge():
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    except Exception:
        font = ImageFont.load_default()
    T, LH, RW = 128, 16, 70
    cols = ['Original'] + WORDS
    sheet = Image.new('RGB', (RW + len(cols) * T, LH + len(ALL_IMAGES) * T), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        d.text((RW + c * T + T // 2, 2), name, fill=(255, 255, 120), font=font, anchor='mt')
    isp = DifferentiableEMOVIS(DEVICE)
    C = np.zeros((len(WORDS), len(ALL_IMAGES)))
    for r, (num, lab) in enumerate(ALL_IMAGES):
        res = json.load(open(os.path.join(RESULTS_DIR, f'results_{num}.json')))
        x = load_and_preprocess_raw(os.path.join(DNG_DIR, f'{num}.dng'), 512).to(DEVICE)
        d.text((4, LH + r * T + T // 2), lab, fill=(255, 255, 255), font=font, anchor='lm')
        for c, w in enumerate(cols):
            set_params(isp, {} if w == 'Original' else res[w]['params'])
            with torch.no_grad():
                y = isp(x)
            im = Image.fromarray((y[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            sheet.paste(im.resize((T, T), Image.LANCZOS), (RW + c * T, LH + r * T))
            if w != 'Original':
                C[c - 1, r] = res[w]['params']['contrast']
    sheet.save(os.path.join(RESULTS_DIR, 'grid.png'))
    lines = [f"{'word':<10} contrast mean±std  (sign agreement)   per image: " + ' '.join(l[:6] for _, l in ALL_IMAGES)]
    for i, w in enumerate(WORDS):
        agree = max(np.mean(C[i] > 0.02), np.mean(C[i] < -0.02))
        lines.append(f"{w:<10} {C[i].mean():+.2f}±{C[i].std():.2f} ({agree:.0%})            " +
                     ' '.join(f"{v:+.2f}" for v in C[i]))
    open(os.path.join(RESULTS_DIR, 'contrast_table.txt'), 'w').write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.environ.get('MERGE'):
        merge()
    else:
        run(os.environ.get('IMAGES', ' '.join(n for n, _ in ALL_IMAGES)).split())
