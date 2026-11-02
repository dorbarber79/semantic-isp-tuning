"""Probe: do positive/negative prompt pairs rank neutral ISP above degraded ISP?"""
import os, sys, warnings
import numpy as np, torch, torch.nn.functional as F
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/administrator/code/semantic-isp-tuning/experiments')
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw
import exp07_emovis_diff as E

DNG = "/data/data1/EmotionISP_dataset/dng_filtered"
IMGS = ['1', '28', '2', '236', '194', '277', '263', '384', '33', '103']
PAIRS = {
    'R5': ("a natural photo", "an over-processed photo"),
    'R6': ("a natural color photo", "an artificial, over-processed photo"),
    'R7': ("a real photo taken with a camera", "a heavily edited, artificial image"),
    'R8': ("a natural, realistic photo", "an over-edited photo with unnatural colors and tones"),
    'B4': ("a high quality photo", "a low quality photo"),
    'B5': ("a professional, well-exposed photo", "an amateur, poorly exposed photo"),
    'B6': ("a beautiful, well-exposed color photo", "an ugly, badly exposed photo"),
    'B7': ("a photo with good exposure and pleasing colors", "a photo with bad exposure and unpleasant colors"),
    'R1': ("a realistic photo", "an unrealistic photo"),
    'R2': ("a realistic photo", "a fake, artificial-looking image"),
    'R3': ("a natural photo", "an artificial, over-processed photo"),
    'R4': ("a natural-looking photo", "an unnatural-looking photo"),
    'B1': ("a beautiful high quality photo", "an ugly low quality photo"),
    'B2': ("Good photo.", "Bad photo."),
    'B3': ("a beautiful photo", "an ugly photo"),
}
DEGR = {
    'sat+1': {'saturation': 0.99}, 'sat-1': {'saturation': -0.99},
    'br+.5': {'brightness': 0.49}, 'br-.5': {'brightness': -0.49},
    'sh+1': {'sharpness': 0.99}, 'sh-1': {'sharpness': -0.99},
    'wb+.3': {'wb_warmth': 0.29}, 'wb-.3': {'wb_warmth': -0.29},
    'ct+1': {'contrast': 0.99}, 'ct-1': {'contrast': -0.99},
    'sat+h': {'saturation': 0.5}, 'sat-h': {'saturation': -0.5}, 'br+h': {'brightness': 0.25}, 'br-h': {'brightness': -0.25},
    'ct+h': {'contrast': 0.5}, 'ct-h': {'contrast': -0.5},
}

vlm = E.SigLIP('cuda'); mean, std = E.vlm_norm_stats('cuda')
def tf(s):
    with torch.no_grad():
        t = vlm.encode_text(vlm.tokenize([s])); return t / t.norm(dim=-1, keepdim=True)
feats = {k: (tf(p), tf(n)) for k, (p, n) in PAIRS.items()}

def set_params(isp, p):
    isp.raw_params.data.zero_()
    for i, n in enumerate(isp.PARAM_NAMES):
        if n in p:
            lo, hi = isp.PARAM_RANGES[n]
            isp.raw_params.data[i] = float(np.arctanh((p[n] - (hi + lo) / 2) / ((hi - lo) / 2)))

isp = DifferentiableEMOVIS('cuda')
# score[pair][degr] = list over images of (d_degr - d_neutral)
out = {k: {d: [] for d in DEGR} for k in PAIRS}
for num in IMGS:
    x = load_and_preprocess_raw(os.path.join(DNG, f'{num}.dng'), 512).cuda()
    def emb(p):
        set_params(isp, p)
        with torch.no_grad():
            y = isp(x)
            f = vlm.encode_image((F.interpolate(y, size=(224, 224), mode='bicubic', align_corners=False) - mean) / std)
        return f / f.norm(dim=-1, keepdim=True)
    f0 = emb({})
    for dn, dp in DEGR.items():
        f1 = emb(dp)
        for k, (tp, tn) in feats.items():
            d0 = (f0 @ tp.T - f0 @ tn.T).item(); d1 = (f1 @ tp.T - f1 @ tn.T).item()
            out[k][dn].append(d1 - d0)

print(f"{'pair':<4} " + ' '.join(f"{d:>7}" for d in DEGR) + "   frac<0  (Δscore degraded−neutral, mean over 10 imgs ×1e3)")
for k in PAIRS:
    m = [np.mean(out[k][d]) * 1e3 for d in DEGR]
    frac = np.mean([v < 0 for d in DEGR for v in out[k][d]])
    print(f"{k:<4} " + ' '.join(f"{v:+7.1f}" for v in m) + f"   {frac:.0%}")
