"""Render every DoF preset level (-3..+3) on several images.

One sheet per image: rows = DoFs, columns = word-table levels. In each row only that DoF
takes its preset value; all others stay at the ISP defaults (level 0).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw

DATA = Path("/data/data1/EmotionISP_dataset/dng_filtered")
RESULTS = Path("/data/data1/semantic_isp/results")
LEVELS = list(range(-3, 4))
LABELS = {"wb_warmth": "White balance", "saturation": "Saturation", "brightness": "Brightness",
          "sharpness": "Sharpness", "contrast": "Contrast"}


def load_presets(path):
    presets = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if "mean" in row:
                value = float(row["mean"]) if row["mean"] else None
                presets.setdefault(row["dof"], {})[int(row["level"])] = (value, "data" if row["mean"] else "none")
            else:
                presets.setdefault(row["dof"], {})[int(row["level"])] = (float(row["preset"]), row["source"])
    return presets


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


@torch.no_grad()
def render_sheet(isp, image, presets, title, cell):
    names = DifferentiableEMOVIS.PARAM_NAMES
    left, top, label_h = 150, 60, 34
    sheet = Image.new("RGB", (left + cell * len(LEVELS), top + len(names) * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 12), title, fill="black", font=font(22))
    for column, level in enumerate(LEVELS):
        draw.text((left + column * cell + cell // 2 - 12, top - 26), f"{level:+d}" if level else "0",
                  fill="black", font=font(20))
    for row, dof in enumerate(names):
        y = top + row * (cell + label_h)
        draw.text((10, y + cell // 2 - 10), LABELS[dof], fill="black", font=font(18))
        for column, level in enumerate(LEVELS):
            value, source = presets[dof][level]
            x = left + column * cell
            if value is None:
                draw.text((x + 10, y + cell // 2), "no words", fill="gray", font=font(16))
                continue
            isp.set_params({**DifferentiableEMOVIS.DEFAULT_PARAMS, dof: value})
            array = isp(image)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            tile = Image.fromarray(np.rint(array * 255).astype(np.uint8)).resize((cell, cell), Image.LANCZOS)
            sheet.paste(tile, (x, y))
            mark = "" if source in ("data", "default") else "*"
            draw.text((x + 6, y + cell + 6), f"{value:+.2f}{mark}", fill="black" if not mark else "gray",
                      font=font(15))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", type=Path, default=RESULTS / "exp21_word_presets_all_images" / "presets.csv")
    parser.add_argument("--images", nargs="+", default=["1", "28", "2", "236", "277", "263", "33", "103"])
    parser.add_argument("--out", type=Path, default=RESULTS / "exp21_word_presets_all_images" / "level_renders")
    parser.add_argument("--cell", type=int, default=256)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    presets = load_presets(args.presets)
    isp = DifferentiableEMOVIS("cuda")
    for image_id in args.images:
        image = load_and_preprocess_raw(str(DATA / f"{image_id}.dng"), 512).cuda()
        sheet = render_sheet(isp, image, presets, f"Image {image_id}: each row varies one DoF over its "
                             f"preset levels (others at default; * = interpolated)", args.cell)
        sheet.save(args.out / f"img_{image_id}_levels.jpg", quality=92)
        print("saved", args.out / f"img_{image_id}_levels.jpg", flush=True)


if __name__ == "__main__":
    main()
