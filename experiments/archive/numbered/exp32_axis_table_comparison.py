"""Compare the existing DoF word table with a five-axis word table."""

import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

import exp07_emovis_diff as experiment
from axis_isp import MoranAxisISP, linear_to_srgb8
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw
from exp19_full_evaluation import encode_image

DATA = Path("/data/data1/EmotionISP_dataset/dng_filtered")
RESULTS = Path("/data/data1/semantic_isp/results")
OLD_TABLE = RESULTS / "word_dof_expectations.csv"
OLD_PRESETS = RESULTS / "exp23_histogram_levels" / "presets_for_render.csv"
OLD_RENDER = RESULTS / "exp24_table_vs_loss"
SOURCE = RESULTS / "exp19_gamut_fullres"
OUT = RESULTS / "exp32_axis_table_comparison"
IMAGES = [1, 28, 2, 236, 194, 277, 263, 384, 33, 103]
WORDS = [
    "vivid", "sunset", "icy", "antique", "dreamy", "dramatic", "monochrome",
    "warm", "cool", "bright", "dark", "soft", "sharp", "pastel", "retro",
    "vintage", "cinematic", "noir", "futuristic", "angelic", "natural", "washed",
    "glowing", "fiery", "maritime",
]
HUE_SHIFT_SCORES = {
    "retro": -3, "vintage": -2, "old": -2, "nostalgic": -2,
    "futuristic": -2, "psychedelic": -2, "filmic": -1,
    "cinematic": 2, "dramatic": 1, "fiery": 2, "sunset": 1,
    "tropical": 1, "maritime": 1, "opulent": 1,
}
SHORT_OLD = {"wb_warmth": "WB", "saturation": "Sat", "brightness": "Bright",
             "sharpness": "Sharp", "contrast": "Contr"}
SHORT_AXIS = {"thermal": "Thermal", "fidelity": "Fidelity", "moral": "Moral",
              "vitality": "Vitality", "hue_shift": "Hue"}
SCORE_NAMES = ("semantic_similarity", "realism_gain", "beauty_gain")
PANEL_W, HEADER_H = 1000, 130


def clamp_score(value):
    return max(-3, min(3, int(round(value))))


def make_axis_scores(old_scores, interpretations):
    axis_rows = {}
    for word, scores in old_scores.items():
        warmth = scores["wb_warmth"]
        saturation = scores["saturation"]
        brightness = scores["brightness"]
        sharpness = scores["sharpness"]
        contrast = scores["contrast"]
        scores = {
            "thermal": warmth,
            "fidelity": clamp_score((sharpness + contrast) / 2),
            "moral": clamp_score(brightness - 0.5 * contrast),
            "vitality": saturation,
            "hue_shift": HUE_SHIFT_SCORES.get(word.lower(), 0),
        }
        axis_rows[word] = {
            **scores,
            "basis": "derived from existing word ratings; hue is a semantic seed rating",
            "interpretation": interpretations.get(word, ""),
        }
    return axis_rows


def load_old_presets():
    source_rows = list(csv.DictReader(OLD_TABLE.open()))
    old_scores = {row["word"]: {
        dof: int(row[f"{dof}_score"]) for dof in DifferentiableEMOVIS.PARAM_NAMES
    } for row in source_rows}
    interpretations = {row["word"]: row.get("interpretation", "") for row in source_rows}
    presets = {}
    for row in csv.DictReader(OLD_PRESETS.open()):
        presets.setdefault(row["dof"], {})[int(row["level"])] = float(row["preset"])
    old_params = {
        word: {dof: presets[dof][score] for dof, score in levels.items()}
        for word, levels in old_scores.items()
    }
    return old_scores, old_params, interpretations


def save_axis_score_table(axis_rows, path):
    fields = ["word", *MoranAxisISP.AXES, "basis", "interpretation"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for word, row in axis_rows.items():
            writer.writerow({"word": word, **row})


def linear_to_srgb(image):
    x = image.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x,
                       1.055 * x.clamp_min(0.0031308).pow(1 / 2.4) - 0.055)


def save_png(path, image_linear):
    pixels = linear_to_srgb8(image_linear)[0].detach().cpu().permute(1, 2, 0).numpy()
    if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_PNG_COMPRESSION, 2]):
        raise OSError(f"Failed to write {path}")


def read_old_png(path):
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise OSError(f"Could not read previous table render: {path}")
    if array.dtype == np.uint16:
        array = (array >> 8).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def score_render(model, image_srgb, word, guards, mean, std):
    feature = encode_image(model, image_srgb, mean, std)
    text = model.encode_text(model.tokenize([f"a {word} photo"]))
    text = text / text.norm(dim=-1, keepdim=True)
    semantic = (feature @ text.T).mean().item()
    quality = {name: scorer(feature).item() for name, (scorer, _) in guards.items()}
    return {
        "semantic_similarity": semantic,
        "realism_gain": quality["realism"] - guards["realism"][1].item(),
        "beauty_gain": quality["beauty"] - guards["beauty"][1].item(),
    }


def old_description(levels, params):
    settings = "  ".join(f"{SHORT_OLD[key]} {levels[key]:+d} -> {params[key]:+.2f}"
                        for key in DifferentiableEMOVIS.PARAM_NAMES)
    return settings


def axis_description(scores, params):
    axes = "  ".join(f"{SHORT_AXIS[key]} {scores[key]:+d}"
                     for key in MoranAxisISP.AXES)
    controls = "  ".join(f"{key} {value:+.2f}" for key, value in params.items())
    return axes, controls


def make_sbs(path, word, old_image, new_image, old_metrics, new_metrics,
             old_levels, old_params, axis_scores, new_params):
    height = round(old_image.height * PANEL_W / old_image.width)
    sheet = Image.new("RGB", (2 * PANEL_W + 12, height + HEADER_H), "white")
    sheet.paste(old_image.resize((PANEL_W, height), Image.Resampling.LANCZOS), (0, HEADER_H))
    sheet.paste(new_image.resize((PANEL_W, height), Image.Resampling.LANCZOS), (PANEL_W + 12, HEADER_H))
    draw = ImageDraw.Draw(sheet)
    old_text = [f"Old DoF table: {word}", old_description(old_levels, old_params),
                f"sem {old_metrics['semantic_similarity']:.4f}  real {old_metrics['realism_gain']:+.4f}  "
                f"beauty {old_metrics['beauty_gain']:+.4f}"]
    axes, controls = axis_description(axis_scores, new_params)
    new_text = ["New axis table", axes, controls,
                f"sem {new_metrics['semantic_similarity']:.4f}  real {new_metrics['realism_gain']:+.4f}  "
                f"beauty {new_metrics['beauty_gain']:+.4f}"]
    for x, lines, color in ((8, old_text, "black"), (PANEL_W + 20, new_text, "black")):
        for row, line in enumerate(lines):
            draw.text((x, 5 + row * 30), line, fill="darkred" if row == 0 else color,
                      font=font(24 if row == 0 else 17))
    sheet.save(path, quality=91)


def make_guards(model, old_isp, crop, source_config, mean, std):
    old_isp.set_params(source_config["defaults"])
    neutral = old_isp(crop)
    neutral_feature = encode_image(model, neutral, mean, std)
    return experiment.build_guards(model, neutral_feature)


@torch.no_grad()
def run_image(image_id, words, old_scores, old_params, axis_rows, model, mean, std, output):
    source = json.loads((SOURCE / f"img_{image_id}" / "results.json").read_text())
    crop = load_and_preprocess_raw(str(DATA / source["dng"]), 512).cuda()
    native = load_and_preprocess_raw(str(DATA / source["dng"]), None).cuda()
    old_isp = DifferentiableEMOVIS("cuda")
    new_isp = MoranAxisISP("cuda", defaults=source["config"]["defaults"])
    guards = make_guards(model, old_isp, crop, source["config"], mean, std)
    image_output = output / f"img_{image_id}"
    new_dir = image_output / "new_axis_table"
    sbs_dir = image_output / "sbs"
    new_dir.mkdir(parents=True, exist_ok=True)
    sbs_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for word in words:
        old_levels = old_scores[word]
        old_control_params = old_params[word]
        old_isp.set_params(old_control_params)
        old_crop = old_isp(crop)
        old_metric = score_render(model, old_crop, word, guards, mean, std)
        old_full = read_old_png(OLD_RENDER / f"img_{image_id}" / "full_resolution" / f"{word}.png")

        word_axes = {axis: axis_rows[word][axis] * (0.4 / 3.0) for axis in MoranAxisISP.AXES}
        new_isp.set_axes(word_axes)
        new_crop_linear = new_isp(crop)
        new_crop_srgb = linear_to_srgb(new_crop_linear)
        new_metric = score_render(model, new_crop_srgb, word, guards, mean, std)
        new_full = new_isp(native)
        new_path = new_dir / f"{word}.png"
        save_png(new_path, new_full)
        with Image.open(new_path) as saved:
            new_full_preview = saved.convert("RGB")
        sheet_path = sbs_dir / f"{word}.jpg"
        make_sbs(sheet_path, word, old_full, new_full_preview, old_metric, new_metric,
             old_levels, old_control_params, axis_rows[word], new_isp.get_params())
        rows.append({
            "image": image_id, "word": word,
            **{f"old_score_{key}": value for key, value in old_levels.items()},
            **{f"old_param_{key}": value for key, value in old_control_params.items()},
            **{f"axis_score_{key}": value for key, value in axis_rows[word].items()
               if key in MoranAxisISP.AXES},
            **{f"new_param_{key}": value for key, value in new_isp.get_params().items()},
            **{f"old_{key}": value for key, value in old_metric.items()},
            **{f"new_{key}": value for key, value in new_metric.items()},
            "sbs": str(sheet_path.relative_to(output)),
            "new_full_resolution": str(new_path.relative_to(output)),
        })
        print(image_id, word, flush=True)

    with (image_output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(rows, output):
    means = {
        f"{side}_{key}": float(np.mean([row[f"{side}_{key}"] for row in rows]))
        for side in ("old", "new") for key in SCORE_NAMES
    }
    wins = {
        key: sum(row[f"new_{key}"] > row[f"old_{key}"] for row in rows)
        for key in SCORE_NAMES
    }
    summary = {"comparisons": len(rows), "images": len({row["image"] for row in rows}),
               "words": len({row["word"] for row in rows}), "mean_scores": means,
               "new_axis_table_wins": wins,
               "interpretation": "VLM scores are proxy metrics; subjective axis ratings are not human-validated."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    words = sorted({row["word"] for row in rows})
    images = sorted({int(row["image"]) for row in rows})
    parts = ["<!doctype html><meta charset='utf-8'><title>Old DoF table vs new axis table</title>",
             "<style>body{font:14px sans-serif;margin:20px}img{width:100%;max-width:1800px}</style>",
             "<h1>Old DoF table vs new axis table</h1>",
             f"<p>{summary['comparisons']} pairs across {summary['images']} images and {summary['words']} words. "
             "Left: previous DoF table rendered by the old ISP. Right: axis ratings mapped through the new ISP. "
             "Axis ratings are a transparent conversion of the existing subjective DoF ratings; hue is a semantic seed. "
             "VLM scores are proxies, not human preference.</p>",
             "<table><tr><th>Table</th><th>Semantic similarity</th><th>Realism gain</th><th>Beauty gain</th></tr>"]
    for side in ("old", "new"):
        parts.append(f"<tr><td>{side}</td>"
                     + "".join(f"<td>{means[f'{side}_{key}']:+.5f}</td>" for key in SCORE_NAMES)
                     + "</tr>")
    parts.append(
        "</table><p>New axis table wins (higher is better): "
        + ", ".join(f"{key}: {value}/{len(rows)}" for key, value in wins.items())
        + "</p>"
    )
    for word in words:
        parts.append(f"<h2 id='{html.escape(word)}'>{html.escape(word)}</h2>")
        for image_id in images:
            parts.append(f"<a href='img_{image_id}/sbs/{word}.jpg'>"
                         f"<img loading='lazy' src='img_{image_id}/sbs/{word}.jpg'></a>")
    (output / "index.html").write_text("\n".join(parts))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=int, default=IMAGES)
    parser.add_argument("--words", nargs="+", default=WORDS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only and args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    old_scores, old_params, interpretations = load_old_presets()
    axis_rows = make_axis_scores(old_scores, interpretations)
    if not set(args.words) <= set(old_scores):
        parser.error(f"Unknown words: {sorted(set(args.words) - set(old_scores))}")
    save_axis_score_table(axis_rows, args.out / "axis_word_scores.csv")
    config = {
        "axis_scores_for_all_words": len(axis_rows),
        "comparison_images": args.images,
        "comparison_words": args.words,
        "axis_coordinate_range": [-0.4, 0.4],
        "score_scale": [-3, 3],
        "score_to_coordinate": "score * (0.4 / 3)",
        "score_rubric": {
            "thermal": "copy old WB score",
            "fidelity": "round((old sharpness score + old contrast score) / 2)",
            "moral": "round(old brightness score - 0.5 * old contrast score)",
            "vitality": "copy old saturation score",
            "hue_shift": "explicit semantic seed dictionary in HUE_SHIFT_SCORES; unspecified words score zero",
            "status": "derived subjective seed ratings; not human-validated",
        },
        "old_render_source": str(OLD_RENDER),
        "new_isp": "MoranAxisISP",
        "baseline_defaults": json.loads((SOURCE / f"img_{args.images[0]}" / "results.json").read_text())
        ["config"]["defaults"] if args.images else None,
    }
    (args.out / "run_config.json").write_text(json.dumps(config, indent=2))
    if not args.report_only:
        torch.set_num_threads(2)
        model = experiment.SigLIP("cuda")
        model.model.requires_grad_(False)
        mean, std = experiment.vlm_norm_stats("cuda")
        rows = []
        for image_id in args.images:
            rows.extend(run_image(image_id, args.words, old_scores, old_params, axis_rows,
                                  model, mean, std, args.out))
    else:
        rows = []
        for path in sorted(args.out.glob("img_*/comparison.csv")):
            rows.extend(csv.DictReader(path.open()))
            rows[-len(list(csv.DictReader(path.open()))):] if False else None
        if not rows:
            raise ValueError(f"No comparison rows found in {args.out}")
        for row in rows:
            for key in SCORE_NAMES:
                for side in ("old", "new"):
                    row[f"{side}_{key}"] = float(row[f"{side}_{key}"])
    summarize(rows, args.out)


if __name__ == "__main__":
    main()
