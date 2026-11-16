"""Render every word with fixed table presets and compare side by side with the Exp 19 loss runs.

For each word, its word-table levels (-3..+3 per DoF) are mapped to the Exp 23 histogram presets.
The resulting image is scored exactly like Exp 19 (512 crop, SigLIP, guards from the Exp 19
neutral image) and saved at native resolution next to the loss-optimized image.
"""

import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import exp07_emovis_diff as experiment
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw
from exp19_full_evaluation import encode_image, save_native, score_image, write_csv, write_json

DATA = Path("/data/data1/EmotionISP_dataset/dng_filtered")
RESULTS = Path("/data/data1/semantic_isp/results")
SOURCE = RESULTS / "exp19_gamut_fullres"
TABLE = RESULTS / "word_dof_expectations.csv"
PRESETS = RESULTS / "exp23_histogram_levels" / "presets_for_render.csv"
OUT = RESULTS / "exp24_table_vs_loss"
NAMES = DifferentiableEMOVIS.PARAM_NAMES
SHORT = {"wb_warmth": "wb", "saturation": "sat", "brightness": "bright", "sharpness": "sharp",
         "contrast": "contr"}
SCORES = ("semantic_similarity", "realism", "beauty")


def load_table():
    with TABLE.open() as stream:
        return {row["word"]: {dof: int(row[f"{dof}_score"]) for dof in NAMES}
                for row in csv.DictReader(stream)}


def load_presets():
    presets = {}
    with PRESETS.open() as stream:
        for row in csv.DictReader(stream):
            presets.setdefault(row["dof"], {})[int(row["level"])] = float(row["preset"])
    return presets


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def read_png(path):
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise OSError(f"Could not read {path}")
    return Image.fromarray((cv2.cvtColor(array, cv2.COLOR_BGR2RGB) >> 8).astype(np.uint8))


def describe(title, params, levels, scores):
    values = "  ".join(f"{SHORT[d]} {params[d]:+.2f}" + (f" [{levels[d]:+d}]" if levels else "")
                       for d in NAMES)
    metrics = (f"sem {scores['semantic_similarity']:.4f}   real {-scores['realism']:+.4f}   "
               f"beauty {-scores['beauty']:+.4f}")
    return [title, values, metrics]


HEADER = 112


def draw_header(sheet, word, left_text, right_text, width):
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, HEADER - 1), fill="white")
    left_text = [f"{word}: {left_text[0]}", *left_text[1:]]
    for x, lines in ((0, left_text), (width + 10, right_text)):
        for index, line in enumerate(lines):
            draw.text((x + 10, 6 + index * 34), line, fill="darkred" if index == 0 and x == 0 else "black",
                      font=font(28 if index == 0 else 22))


def make_sbs(path, word, left, right, left_text, right_text, width):
    height = round(left.height * width / left.width)
    sheet = Image.new("RGB", (2 * width + 10, height + HEADER), "white")
    for x, image in ((0, left), (width + 10, right)):
        sheet.paste(image.resize((width, height), Image.LANCZOS), (x, HEADER))
    draw_header(sheet, word, left_text, right_text, width)
    sheet.save(path, quality=90)


@torch.no_grad()
def run_image(image_id, table, presets, width):
    source = json.loads((SOURCE / f"img_{image_id}" / "results.json").read_text())
    out = OUT / f"img_{image_id}"
    for directory in ("full_resolution", "sbs"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    dng = DATA / source["dng"]
    image = load_and_preprocess_raw(str(dng), 512).cuda()
    native = load_and_preprocess_raw(str(dng), None).cuda()
    model = experiment.SigLIP("cuda")
    mean, std = experiment.vlm_norm_stats("cuda")
    isp = DifferentiableEMOVIS("cuda")
    # Exp 19 guards came from the neutral image under its own defaults; reuse them for comparability.
    isp.set_params(source["config"]["defaults"])
    guards = experiment.build_guards(model, encode_image(model, isp(image), mean, std))
    rows = []
    for word in table:
        loss_entry = source["words"][word]
        levels = table[word]
        params = {dof: presets[dof][levels[dof]] for dof in NAMES}
        isp.set_params(params)
        table_scores = score_image(model, isp(image), word, guards, isp, mean, std)
        native_image = isp(native)
        save_native(out / "full_resolution" / f"{word}.png", native_image)
        loss_scores = {**loss_entry["losses"], "semantic_similarity": loss_entry["clip_sim"]}
        make_sbs(out / "sbs" / f"{word}.jpg", word,
                 read_png(SOURCE / f"img_{image_id}" / loss_entry["full_resolution_image"]),
                 read_png(out / "full_resolution" / f"{word}.png"),
                 describe("Loss-optimized (Exp 19)", loss_entry["params"], None, loss_scores),
                 describe("Table preset (Exp 23 levels)", params, levels, table_scores), width)
        rows.append({
            "image": image_id, "word": word,
            **{f"level_{dof}": levels[dof] for dof in NAMES},
            **{f"table_{dof}": params[dof] for dof in NAMES},
            **{f"loss_{dof}": loss_entry["params"][dof] for dof in NAMES},
            **{f"table_{name}": table_scores[name] for name in SCORES},
            **{f"loss_{name}": loss_scores[name] for name in SCORES},
            "sbs": f"img_{image_id}/sbs/{word}.jpg",
            "table_full_resolution": f"img_{image_id}/full_resolution/{word}.png",
            "loss_full_resolution": str(SOURCE / f"img_{image_id}" / loss_entry["full_resolution_image"]),
        })
        print(image_id, word, flush=True)
    write_csv(out / "comparison.csv", rows)
    return rows


def write_report(rows):
    words = sorted({row["word"] for row in rows})
    images = sorted({row["image"] for row in rows})

    def wins(name, higher_is_better):
        sign = 1 if higher_is_better else -1
        return sum(sign * (row[f"table_{name}"] - row[f"loss_{name}"]) > 0 for row in rows)

    summary = {
        "runs": len(rows),
        "table_better": {"semantic": wins("semantic_similarity", True),
                         "realism": wins("realism", False), "beauty": wins("beauty", False)},
        "mean": {f"{side}_{name}": float(np.mean([row[f"{side}_{name}"] for row in rows]))
                 for side in ("table", "loss") for name in SCORES},
    }
    write_json(OUT / "summary.json", summary)
    write_csv(OUT / "comparison.csv", rows)
    parts = ["<!doctype html><meta charset='utf-8'><title>Exp 24 table vs loss</title><style>"
             "body{font:14px sans-serif;margin:20px}img{width:100%;max-width:1600px}"
             "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}</style>",
             "<h1>Exp 24: table presets vs loss optimization (side by side)</h1>",
             "<p>Left = Exp 19 loss-optimized; right = word-table levels mapped to Exp 23 histogram "
             "presets. Scores: SigLIP similarity to 'a {word} photo' (higher = better); realism/beauty "
             "gain vs neutral (higher = better). Both scored identically.</p>",
             f"<p>Table preset better than loss in {summary['runs']} runs: semantic "
             f"{summary['table_better']['semantic']}, realism {summary['table_better']['realism']}, "
             f"beauty {summary['table_better']['beauty']}.</p><table><tr><th></th>"
             "<th>semantic sim</th><th>realism gain</th><th>beauty gain</th></tr>"]
    for side in ("loss", "table"):
        mean = summary["mean"]
        parts.append(f"<tr><td>{side}</td><td>{mean[f'{side}_semantic_similarity']:.4f}</td>"
                     f"<td>{-mean[f'{side}_realism']:+.4f}</td><td>{-mean[f'{side}_beauty']:+.4f}</td></tr>")
    parts.append("</table><p>Jump to word: " + " ".join(
        f"<a href='#{html.escape(word)}'>{html.escape(word)}</a>" for word in words) + "</p>")
    for word in words:
        parts.append(f"<h2 id='{html.escape(word)}'>{html.escape(word)}</h2>")
        for image_id in images:
            parts.append(f"<a href='img_{image_id}/sbs/{word}.jpg'><img loading='lazy' "
                         f"src='img_{image_id}/sbs/{word}.jpg'></a>")
    (OUT / "index.html").write_text("\n".join(parts))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=int)
    parser.add_argument("--width", type=int, default=1520, help="width of each half of the SBS image")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    images = args.images or sorted(int(path.name[4:]) for path in SOURCE.glob("img_*"))
    if not args.report_only:
        table, presets = load_table(), load_presets()
        for image_id in images:
            run_image(image_id, table, presets, args.width)
    rows = []
    for path in sorted(OUT.glob("img_*/comparison.csv")):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                rows.append({key: (value if key in ("word", "sbs") or key.endswith("resolution")
                                   else float(value)) for key, value in row.items()})
    write_report(rows)


if __name__ == "__main__":
    main()
