"""
Experiment 17: Do our loss indicators agree with EMOVIS emotions? (EMOVIS paper, Fig. 1)

Score only the 8 images of EMOVIS Fig. 1 (girl / dog × Neutral, Sad, Angry, Happy;
cropped from reference_papers/2605.03131v1.pdf) with the exact loss terms of exp07:
    semantic  s_e(img) = cos(SigLIP(img), SigLIP("a {e} photo"))
    realism   L_r = q_r(neutral) − q_r(img)      (q = cos(pos) − cos(neg))
    beauty    L_b = q_b(neutral) − q_b(img)
No RAW loading, ISP processing, parameter optimization or parameter regularization.
Only the model's input resize and normalization are applied to the existing crops.
Changes are relative to the paper's Neutral crop of the same scene, not our ISP default.
The primary paired check uses the SAME emotion prompt for Neutral and emotional crops:
L_semantic(emotional, emotion) < L_semantic(Neutral, emotion).
For each emotion prompt, check whether the matching paper rendering beats all other
renderings of that scene, including Neutral.
"""

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from exp07_emovis_diff import (SigLIP, vlm_norm_stats, build_guards, DEVICE,
                               VLM_NAME, VLM_NORM, SEMANTIC_WEIGHT, REALISM_WEIGHT,
                               BEAUTY_WEIGHT, REALISM_PROMPTS, BEAUTY_PROMPTS)

OUT = os.environ.get('OUT', "/data/data1/semantic_isp/results/exp17_emovis_fig1")
CROPS = os.path.join(OUT, 'crops')
SCENES = ['girl', 'dog']
FIG_EMOS = ['sad', 'angry', 'happy']          # columns of EMOVIS Fig. 1 besides neutral
PROMPT_EMOS = ['sad', 'angry', 'happy', 'calm']
# Paper super-class members (Sec. 3): used as a synonym robustness check
SYNONYMS = {'sad': ['sad', 'lonely', 'subdued'], 'angry': ['angry', 'hostile', 'tense'],
            'happy': ['happy', 'joyful', 'amused'], 'calm': ['calm', 'relaxed', 'serene']}


def to_tensor(pil):
    return torch.from_numpy(np.asarray(pil.convert('RGB'), dtype=np.float32) / 255.).permute(2, 0, 1)[None].to(DEVICE)


class Scorer:
    def __init__(self):
        self.vlm = SigLIP(DEVICE)
        self.mean, self.std = vlm_norm_stats(DEVICE)
        words = sorted({w for v in SYNONYMS.values() for w in v})
        with torch.no_grad():
            t = self.vlm.encode_text(self.vlm.tokenize([f"a {w} photo" for w in words]))
        t = t / t.norm(dim=-1, keepdim=True)
        self.text = {word: t[i:i + 1] for i, word in enumerate(words)}

    @torch.no_grad()
    def feat(self, img):  # img: 1x3xHxW in [0,1], same path as exp07 loss
        x = F.interpolate(img, size=(224, 224), mode='bicubic', align_corners=False)
        f = self.vlm.encode_image((x - self.mean) / self.std)
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def indicators(self, feat, neutral_feat):
        guards = build_guards(self.vlm, neutral_feat)
        r = {}
        for name in ('realism', 'beauty'):
            scorer, q0 = guards[name]
            r[f'L_{name}'] = (q0 - scorer(feat)).item()
        for e in PROMPT_EMOS:
            r[f'sem_{e}'] = (feat @ self.text[e].T).item()
            r[f'syn_{e}'] = float(np.mean([(feat @ self.text[w].T).item() for w in SYNONYMS[e]]))
        return r


def analyse(tag, rows, lines):
    """rows: {img_emotion: indicators}, includes 'neutral'. Writes matrices to lines."""
    n = rows['neutral']
    variants = ['neutral'] + FIG_EMOS
    lines.append(f"\n=== {tag} ===")
    for key, title in (('sem', 'Δ semantic  cos("a {e} photo") − neutral   [×1e3]'),
                       ('syn', 'Δ semantic, mean of 3 paper synonyms − neutral [×1e3]')):
        lines.append(f"{title}\n  image\\prompt " + ''.join(f"{e:>9s}" for e in PROMPT_EMOS))
        D = np.array([[rows[i][f'{key}_{e}'] - n[f'{key}_{e}'] for e in PROMPT_EMOS] for i in variants])
        for i, img_e in enumerate(variants):
            cells = ''.join(('  *' if PROMPT_EMOS[j] == img_e else '   ') + f"{1e3*D[i, j]:+6.1f}"
                            for j in range(len(PROMPT_EMOS)))
            lines.append(f"  {img_e:>12s} {cells}")
        col_hits = [int(D[j + 1, j] > np.delete(D[:, j], j + 1).max())
                    for j in range(len(FIG_EMOS))]
        row_hits = [int(D[i + 1, i] > np.delete(D[i + 1, :len(FIG_EMOS)], i).max())
                    for i in range(len(FIG_EMOS))]
        diag_pos = [int(D[i + 1, i] > 0) for i in range(len(FIG_EMOS))]
        lines.append(f"  matching image strictly beats all variants: {sum(col_hits)}/3  {col_hits}")
        lines.append(f"  target prompt has largest Δ (sad/angry/happy): {sum(row_hits)}/3  {row_hits}")
        lines.append(f"  target Δ > 0 vs neutral:                       {sum(diag_pos)}/3  {diag_pos}")
    lines.append("Calm and synonyms are auxiliary checks; Fig. 1 has no Calm rendering.")
    lines.append("Quality losses vs neutral [×1e3]  (+ = degradation, − = improvement)")
    lines.append("  image        L_realism  L_beauty  ΔL_total(target), weights from exp07")
    metrics = {}
    for e in FIG_EMOS:
        r = rows[e]
        delta_sem = r[f'sem_{e}'] - n[f'sem_{e}']
        totals = {
            variant: (-SEMANTIC_WEIGHT * (row[f'sem_{e}'] - n[f'sem_{e}'])
                      + REALISM_WEIGHT * row['L_realism']
                      + BEAUTY_WEIGHT * row['L_beauty'])
            for variant, row in rows.items()
        }
        tot = totals[e]
        lines.append(f"  {e:>8s}    {1e3*r['L_realism']:+8.2f}  {1e3*r['L_beauty']:+8.2f}  {1e3*tot:+8.2f}")
        metrics[e] = {
            'target_semantic_delta': delta_sem,
            'target_total_loss_delta': tot,
            'pair': {
                'prompt': f"a {e} photo",
                'neutral_semantic_loss': -n[f'sem_{e}'],
                'emotion_semantic_loss': -r[f'sem_{e}'],
                'semantic_loss_delta': -delta_sem,
                'semantic_matches': delta_sem > 0,
                'neutral_total_loss': -SEMANTIC_WEIGHT * n[f'sem_{e}'],
                'emotion_total_loss': -SEMANTIC_WEIGHT * n[f'sem_{e}'] + tot,
                'total_loss_delta': tot,
                'total_matches': tot < 0,
            },
            'semantic_best_variants': [v for v in variants
                                       if rows[v][f'sem_{e}'] == max(rows[i][f'sem_{e}'] for i in variants)],
            'total_loss_best_variants': [v for v in variants if totals[v] == min(totals.values())],
            'total_loss_deltas': totals,
        }
    lines.append("Best rendering for each fixed target (including Neutral):")
    for e, metric in metrics.items():
        lines.append(f"  {e:>8s}: semantic={metric['semantic_best_variants']}, "
                     f"total loss={metric['total_loss_best_variants']}")
    lines.append("Paired test: SAME emotion prompt on Neutral and emotion crop. Lower loss is better.")
    lines.append("  emotion      L_sem(Neutral)  L_sem(emotion)  delta_L_sem   matches  delta_L_total")
    for e, metric in metrics.items():
        pair = metric['pair']
        lines.append(f"  {e:>8s}    {pair['neutral_semantic_loss']:+.6f}       "
                     f"{pair['emotion_semantic_loss']:+.6f}       "
                     f"{pair['semantic_loss_delta']:+.6f}    "
                     f"{str(pair['semantic_matches']):>5s}    {pair['total_loss_delta']:+.6f}")
    lines.append(f"  Semantic matches: {sum(m['pair']['semantic_matches'] for m in metrics.values())}/3; "
                 f"total-loss matches: {sum(m['pair']['total_matches'] for m in metrics.values())}/3.")
    return metrics


def label(img, text, w=300):
    img = img.resize((w, w), Image.LANCZOS)
    c = Image.new('RGB', (w, w + 22), (25, 25, 25))
    c.paste(img, (0, 22))
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except OSError:
        font = None
    ImageDraw.Draw(c).text((4, 3), text, fill=(230, 230, 230), font=font)
    return c


def main():
    sc = Scorer()
    lines = [
        "EMOVIS Figure 1: evaluation of the paper crops ONLY.",
        "No ISP, RAW loading, image optimization, or parameter regularization.",
        "The paper's Neutral crop is the reference for each scene.",
        "Model input only: RGB / 255, bicubic resize to 224x224, VLM normalization.",
        f"VLM: {VLM_NAME}, normalization: {VLM_NORM}",
        f"Weights: semantic={SEMANTIC_WEIGHT}, realism={REALISM_WEIGHT}, beauty={BEAUTY_WEIGHT}",
        "All eight crops are scored against every prompt; this is a small sanity check,",
        "not a statistical validation or a calibrated emotion classifier.",
    ]
    out_json = {
        'config': {
            'uses_isp': False,
            'optimization': False,
            'reference': 'paper Neutral crop of the same scene',
            'crops_dir': CROPS,
            'vlm': VLM_NAME,
            'normalization': VLM_NORM,
            'semantic_prompts': [f"a {e} photo" for e in PROMPT_EMOS],
            'realism_prompts': REALISM_PROMPTS,
            'beauty_prompts': BEAUTY_PROMPTS,
            'weights': {'semantic': SEMANTIC_WEIGHT, 'realism': REALISM_WEIGHT,
                        'beauty': BEAUTY_WEIGHT},
            'parameter_regularization': 'not applicable; no parameters optimized',
        },
        'metrics': {},
    }

    for scene in SCENES:
        feats = {e: sc.feat(to_tensor(Image.open(f"{CROPS}/{scene}_{e}.png")))
                 for e in ['neutral'] + FIG_EMOS}
        rows = {e: sc.indicators(f, feats['neutral']) for e, f in feats.items()}
        out_json[f'emovis_{scene}'] = rows
        out_json['metrics'][scene] = analyse(f"EMOVIS Fig.1 — {scene}", rows, lines)

    cols = ['neutral'] + FIG_EMOS
    w = 300
    canvas = Image.new('RGB', (len(cols) * w, len(SCENES) * (w + 22)), (25, 25, 25))
    for s, scene in enumerate(SCENES):
        for c, e in enumerate(cols):
            canvas.paste(label(Image.open(f"{CROPS}/{scene}_{e}.png"), f"{scene}: EMOVIS {e}"),
                         (c * w, s * (w + 22)))
    canvas.save(os.path.join(OUT, 'comparison.png'))

    with open(os.path.join(OUT, 'report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(out_json, f, indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
