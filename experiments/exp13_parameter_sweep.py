"""
Experiment 13: ISP Parameter Sweep

Sweep each of the 5 learnable parameters independently:
  - wb_warmth: [-0.3, 0.3]
  - saturation: [-1.0, 1.0]
  - brightness: [-0.5, 0.5]
  - sharpness: [-1.0, 1.0]
  - contrast: [-1.0, 1.0]

All other parameters held at 0 (neutral).
Visual grid: 5 rows (params) × 7 cols (sweep values).
"""

import os, sys, warnings
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
warnings.filterwarnings('ignore', message="Can't initialize NVML")

sys.path.insert(0, os.path.dirname(__file__))
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw

RESULTS_DIR = os.environ.get('RESULTS_DIR', "/data/data1/semantic_isp/results/exp13_param_sweep")
DEVICE = "cuda"


def render(img_tensor, params):
    isp = DifferentiableEMOVIS(device=DEVICE)
    with torch.no_grad():
        for pi, name in enumerate(DifferentiableEMOVIS.PARAM_NAMES):
            lo, hi = DifferentiableEMOVIS.PARAM_RANGES[name]
            mid, half = (hi+lo)/2.0, (hi-lo)/2.0
            norm = (params[name]-mid)/half
            norm = max(-0.999, min(0.999, norm))
            isp.raw_params.data[pi] = np.arctanh(norm)
        out = isp(img_tensor)
        return (out[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Exp 13: ISP Parameter Sweep")
    print("Sweep each parameter independently across its full range")
    print("=" * 60)

    dng_path = os.environ.get('DNG_PATH', '/data/data1/EmotionISP_dataset/dng/happy/8.dng')
    img_tensor = load_and_preprocess_raw(dng_path, target_size=512).to(DEVICE)

    # Define sweeps: (param_name, display_name, values, color)
    sweeps = [
        ('wb_warmth', 'White Balance', np.linspace(-0.3, 0.3, 7), (255, 180, 100)),
        ('saturation', 'Saturation', np.linspace(-1.0, 1.0, 7), (255, 100, 180)),
        ('brightness', 'Brightness', np.linspace(-0.5, 0.5, 7), (255, 255, 100)),
        ('sharpness', 'Sharpness', np.linspace(-1.0, 1.0, 7), (100, 200, 255)),
        ('contrast', 'Contrast', np.linspace(-1.0, 1.0, 7), (180, 255, 140)),
    ]

    thumb = (160, 160)
    label_h = 35
    row_label_w = 100
    cell_w, cell_h = thumb
    n_cols = 7
    n_rows = len(sweeps)

    canvas = Image.new('RGB', (row_label_w + n_cols*cell_w, n_rows*(cell_h+label_h)), (25,25,25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 9)
        font_big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 11)
    except:
        font = font_big = ImageFont.load_default()

    for row, (param_name, display_name, values, color) in enumerate(sweeps):
        y = row * (cell_h + label_h)

        # Row label
        draw.text((5, y + cell_h//2 + label_h), display_name,
                  fill=color, font=font_big, anchor='lm')

        print(f"\n{display_name} ({param_name})")

        for col, val in enumerate(values):
            x = row_label_w + col * cell_w

            # Set params: all 0 except the swept parameter
            params = {n: 0.0 for n in DifferentiableEMOVIS.PARAM_NAMES}
            params[param_name] = val

            img_np = render(img_tensor, params)
            canvas.paste(Image.fromarray(img_np).resize(thumb, Image.LANCZOS), (x, y + label_h))

            # Label: param value
            val_text = f"{val:+.2f}"
            draw.text((x + cell_w//2, y + 3), val_text,
                      fill=(200,200,200), font=font, anchor='mt')

            # Add "neutral" marker for value closest to 0
            if abs(val) < 0.05:
                draw.rectangle([x+2, y+label_h+2, x+cell_w-2, y+label_h+cell_h-2],
                              outline=color, width=2)
                draw.text((x + cell_w//2, y + 16), "neutral",
                          fill=color, font=font, anchor='mt')

            print(f"  {val:+.3f}", end="")
        print()

    out_path = os.path.join(RESULTS_DIR, 'sweep.png')
    canvas.save(out_path, quality=95)
    print(f"\nGrid: {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
