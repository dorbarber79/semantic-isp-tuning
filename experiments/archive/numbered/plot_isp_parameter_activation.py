"""Plot actual ISP-control changes from neutral; never connect signed values as polygons."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from moran_semantic_isp import CONTROL_BOUNDS, CONTROL_NAMES, CONTROL_NEUTRAL


FIELDS = ("warmth", "saturation", "tone_gain", "contrast", "sharpness", "palette_coordinate")
LABELS = ("White-balance warmth", "Saturation adjustment", "Tone gain",
          "Contrast slope", "Sharpness / blur amount", "Palette coordinate")
DIRECTIONS = (("Cooler / bluer", "Warmer / redder"),
              ("Less saturated", "More saturated"), ("Darker", "Brighter"),
              ("Flatter", "Higher contrast"), ("More blur", "More sharpening"),
              ("Retro / Magenta-Green", "Blockbuster / Teal-Orange"))


def control_deltas(rows, bounds=CONTROL_BOUNDS):
    settings = np.array([[float(row[field]) for field in FIELDS] for row in rows])
    if not np.isfinite(settings).all():
        raise ValueError("Control values must be finite")
    neutral = np.asarray(CONTROL_NEUTRAL)
    bounds = np.asarray(bounds)
    if ((settings < bounds[:, 0] - 1e-6) | (settings > bounds[:, 1] + 1e-6)).any():
        raise ValueError("Invalid physical control value")
    return settings, settings - neutral


def number(value):
    if abs(value) < 1e-7:
        return "0"
    if abs(value) < .001:
        return f"{value:+.2e}"
    return f"{value:+.4f}"


def bar_figure(path, words, deltas, old_deltas=None, *, title=None, bounds=CONTROL_BOUNDS,
               comparison_labels=("Previous ungated mapping", "New gated + constrained mapping")):
    figure, axes = plt.subplots(3, 2, figsize=(18, 18))
    neutral = np.asarray(CONTROL_NEUTRAL)
    bounds = np.asarray(bounds)
    positions = np.arange(len(words))
    for index, ax in enumerate(axes.flat):
        values = deltas[:, index]
        colors = np.where(values < 0, "#286f9e", "#b94b32")
        if old_deltas is None:
            ax.barh(positions, values, height=.60, color=colors)
        else:
            ax.barh(positions - .18, old_deltas[:, index], height=.32,
                    color="#657d67", label=comparison_labels[0])
            ax.barh(positions + .18, values, height=.32, color=colors,
                    label=comparison_labels[1])
        ax.axvline(0, color="#222222", linewidth=1)
        ax.set_yticks(positions, words, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(bounds[index, 0] - neutral[index], bounds[index, 1] - neutral[index])
        ax.set_ylim(len(words) - .4, -.7)
        ax.set_title(f"{LABELS[index]} (neutral setting = {neutral[index]:g})", fontsize=12)
        ax.set_xlabel(f"{DIRECTIONS[index][0]}  <---  change from neutral  --->  {DIRECTIONS[index][1]}",
                      fontsize=9)
        ax.grid(axis="x", color="#dedede", linewidth=.7)
        ax.set_axisbelow(True)
        if old_deltas is None:
            for position, value in zip(positions, values):
                ax.annotate(number(value), (value, position),
                            xytext=(5 if value >= 0 else -5, 0), textcoords="offset points",
                            ha="left" if value >= 0 else "right", va="center", fontsize=8)
        else:
            for offset, series in ((-.18, old_deltas[:, index]), (.18, values)):
                for position, value in zip(positions, series):
                    ax.annotate(number(value), (value, position + offset),
                                xytext=(5 if value >= 0 else -5, 0), textcoords="offset points",
                                ha="left" if value >= 0 else "right", va="center", fontsize=7)
            if index == 0:
                ax.legend(loc="lower right", fontsize=8)
    if title is None:
        title = "New mapping: exact ISP-control changes from neutral" if old_deltas is None else (
            "Previous vs new mapping: physical-control changes on the same full-range scales")
    figure.suptitle(title, fontsize=17)
    figure.text(.5, .012,
                "Each panel uses that control's full allowed range. Numeric labels are physical deltas, not semantic-axis coordinates. "
                "Near-zero bars mean little or no activation; bar lengths are not comparable across different controls.",
                ha="center", fontsize=10)
    figure.tight_layout(rect=(0, .035, 1, .97), h_pad=3)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def physical_control_row(parameters):
    return {"warmth": float(parameters["wb"][0] - 1),
            **{field: float(parameters[field]) for field in FIELDS[1:]}}


def compare_calibrations(root, output, words, device):
    import open_clip
    import torch
    from moran_semantic_isp import SemanticProjection, compose_parameters

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    versions = ("cinematic_ungated", "cinematic_strict", "cinematic_soft",
                "direct_ungated", "direct_soft")
    configs = {name: json.loads((root / f"{name}_calibration.json").read_text())
               for name in versions}
    reference = configs[versions[0]]
    for config in configs.values():
        for field in ("model", "pretrained", "force_quick_gelu"):
            if config.get(field) != reference.get(field):
                raise ValueError(f"Compared calibrations must share {field}")
    model, _, _ = open_clip.create_model_and_transforms(
        reference["model"], pretrained=reference["pretrained"], device=device,
        force_quick_gelu=reference.get("force_quick_gelu", True))
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(reference["model"])

    @torch.no_grad()
    def encode(texts):
        return model.encode_text(tokenizer(texts).to(device))

    prompts = [f"a {word} photo" for word in words]
    features = encode(prompts)
    bounds_by_version = {name: config.get("control_bounds",
        config["learned_calibration"].get("control_bounds", CONTROL_BOUNDS))
        for name, config in configs.items()}
    plot_bounds = np.asarray(bounds_by_version[versions[0]], dtype=float)
    for version in versions[1:]:
        next_bounds = np.asarray(bounds_by_version[version], dtype=float)
        plot_bounds[:, 0] = np.minimum(plot_bounds[:, 0], next_bounds[:, 0])
        plot_bounds[:, 1] = np.maximum(plot_bounds[:, 1], next_bounds[:, 1])
    tables, deltas = [], {}
    with torch.no_grad():
        for version, config in configs.items():
            projection = SemanticProjection(encode, config)
            _, coordinates = projection.project_features(features)
            _, weights = projection.relevance_weights(features)
            rows = [physical_control_row(compose_parameters(coordinate, config))
                    for coordinate in coordinates]
            _, deltas[version] = control_deltas(rows, bounds_by_version[version])
            for index, row in enumerate(rows):
                tables.append({"version": version, "word": words[index], "prompt": prompts[index],
                    **{f"setting_{field}": row[field] for field in FIELDS},
                    **{f"delta_{field}": float(deltas[version][index, column])
                       for column, field in enumerate(FIELDS)},
                    "coordinates_json": json.dumps(dict(zip(projection.axes, coordinates[index].cpu().tolist()))),
                    "gates_json": json.dumps(dict(zip(projection.axes, weights[index].cpu().tolist())))})
    output.mkdir(parents=True)
    charts = [("Direct ISP vs cinematic: soft gates", "parameter_changes.png")]
    bar_figure(output / charts[0][1], words, deltas["direct_soft"], deltas["cinematic_soft"],
               title="Direct ISP vs cinematic: soft-gated physical-control changes",
               bounds=plot_bounds,
               comparison_labels=("Cinematic soft", "Direct ISP soft"))
    charts.append(("Direct ISP vs cinematic: no gates", "ungated_comparison.png"))
    bar_figure(output / charts[-1][1], words, deltas["direct_ungated"], deltas["cinematic_ungated"],
               title="Direct ISP vs cinematic: ungated physical-control changes",
               bounds=plot_bounds,
               comparison_labels=("Cinematic ungated", "Direct ISP ungated"))
    for version in versions:
        title = version.replace("_", " ").title()
        filename = f"{version}_parameter_changes.png"
        charts.append((title, filename))
        bar_figure(output / filename, words, deltas[version],
                   title=f"{title}: exact ISP-control changes from neutral", bounds=plot_bounds)
    with (output / "parameter_activation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)
    manifest = {"calibration_root": str(root), "words": words, "prompts": prompts,
        "versions": list(versions), "neutral_settings": list(CONTROL_NEUTRAL),
        "control_bounds": plot_bounds.tolist(), "per_version_control_bounds": bounds_by_version,
        "control_order": list(CONTROL_NAMES),
        "config_sha256": {name: hashlib.sha256((root / f"{name}_calibration.json").read_bytes()).hexdigest()
                          for name in versions},
        "plot_units": "Physical-control deltas from neutral, full legal ranges; values below 1e-7 displayed as zero",
        "note": "Text-only mappings are image-independent. Same prompts for every version; no new optimization."}
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    page = ["<!doctype html><meta charset='utf-8'><title>Direct ISP vs cinematic parameters</title>",
            "<h1>Direct ISP vs cinematic: physical parameter activation</h1>",
            "<p>Same words and photo templates across all calibrations. Parameters are image-independent. "
            "Tone and contrast neutral settings are one; other controls are zero. "
            "Each panel shows its full legal range. Bar lengths across controls are not perceptual comparisons. "
            "Values below 1e-7 display as zero; the CSV retains full precision.</p>",
            "<p><a href='parameter_activation.csv'>Exact settings, deltas, coordinates and gates</a></p>"]
    for title, filename in charts:
        page.append(f"<h2>{html.escape(title)}</h2><a href='{filename}'><img style='max-width:100%' src='{filename}'></a>")
    (output / "index.html").write_text("\n".join(page))
    print(output / "index.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--comparison", type=Path, help="Exp49 directory containing the five calibrated mappings")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--words", nargs="+", default=["warm", "cold", "dreamy", "documentary",
        "chiaroscuro", "high key", "vivid", "bleached", "retro", "blockbuster"])
    args = parser.parse_args()
    if args.comparison:
        if args.results or args.previous:
            parser.error("--comparison cannot be combined with --results or --previous")
        compare_calibrations(args.comparison, args.out or args.comparison / "figures", args.words, args.device)
        return
    if not args.results or not args.previous:
        parser.error("Provide --comparison, or both --results and --previous")
    rows = list(csv.DictReader((args.results / "per_word_values.csv").open()))
    old_rows = list(csv.DictReader((args.previous / "per_word_values.csv").open()))
    old_by_word = {row["word"]: row for row in old_rows}
    old_rows = [old_by_word[row["word"]] for row in rows]
    settings, deltas = control_deltas(rows)
    _, old_deltas = control_deltas(old_rows)
    words = [row["word"] for row in rows]
    output = args.results / "figures"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir()
    bar_figure(output / "parameter_changes.png", words, deltas)
    bar_figure(output / "previous_vs_new.png", words, deltas, old_deltas)

    figure, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(-.5, 5.5)
    ax.set_ylim(len(words) - .5, -.5)
    ax.set_xticks(range(6), [f"{name}\nneutral {value:g}" for name, value in zip(LABELS, CONTROL_NEUTRAL)], fontsize=9)
    ax.set_yticks(range(len(words)), words, fontsize=11)
    ax.xaxis.tick_top()
    for row_index in range(len(words)):
        for column in range(6):
            color = "#286f9e" if deltas[row_index, column] < -1e-7 else (
                "#b94b32" if deltas[row_index, column] > 1e-7 else "#666666")
            ax.text(column, row_index,
                    f"setting {settings[row_index,column]:.6f}\nchange {number(deltas[row_index,column])}",
                    ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(np.arange(-.5, 6), minor=True)
    ax.set_yticks(np.arange(-.5, len(words)), minor=True)
    ax.grid(which="minor", color="#dddddd")
    ax.tick_params(which="both", length=0)
    figure.suptitle("Actual settings and signed changes (blue = negative, red = positive)", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .93))
    figure.savefig(output / "exact_settings.png", dpi=180)
    plt.close(figure)

    page = ["<!doctype html><meta charset='utf-8'><title>Image 2 ISP parameter activation</title>",
            "<h1>Image 2: new mapping, actual ISP parameter activation</h1>",
            "<p>New relevance-gated, sign-constrained calibration. Negative/positive directions and neutral settings "
            "are shown separately for each physical control. Controls have different units, so bar lengths across "
            "different panels are not perceptual effect comparisons.</p>",
            "<p>Palette is a physical palette-projection coordinate, not the semantic hue-axis score. "
            "Saturation is an adjustment: zero means unchanged, not grayscale. "
            "Tone and contrast are unchanged at one, not zero. Values below 1e-7 are displayed as zero; full values are in CSV.</p>"]
    for title, filename in (("Signed changes: full control ranges", "parameter_changes.png"),
                            ("Exact settings and changes", "exact_settings.png"),
                            ("Previous vs new mapping", "previous_vs_new.png")):
        page.append(f"<h2>{title}</h2><a href='{filename}'><img style='max-width:100%' src='{filename}'></a>")
    neutral = args.results / "neutral.png"
    with Image.open(neutral) as image:
        neutral_pixels = np.array(image)
    table = []
    for row_index, row in enumerate(rows):
        word = row["word"]
        filename = word.replace(" ", "_") + ".png"
        paths = [neutral, args.previous / filename, args.results / filename]
        sheet = Image.new("RGB", (1200, 330), "white")
        draw = ImageDraw.Draw(sheet)
        titles = ["Neutral", "Previous mapping", "New mapping"]
        for index, (path, title) in enumerate(zip(paths, titles)):
            with Image.open(path) as image:
                image.thumbnail((400, 290), Image.Resampling.LANCZOS)
                sheet.paste(image, (400 * index + (400 - image.width) // 2,
                                   30 + (290 - image.height) // 2))
            draw.text((400 * index + 10, 6), f"{title}: {word}", fill="black")
        sheet_path = f"{word.replace(' ', '_')}_comparison.jpg"
        sheet.save(output / sheet_path, quality=94)
        with Image.open(args.results / filename) as image:
            difference = np.abs(np.asarray(image).astype(float) - neutral_pixels.astype(float))
        table.append({"word": word,
                      **{f"setting_{field}": settings[row_index, index] for index, field in enumerate(FIELDS)},
                      **{f"delta_{field}": deltas[row_index, index] for index, field in enumerate(FIELDS)},
                      "mean_pixel_change_0_255": float(difference.mean()),
                      "max_pixel_change_0_255": float(difference.max())})
        page.append(f"<h2>{html.escape(word)}</h2><img style='max-width:100%' src='{sheet_path}'>"
                    + "<p>" + "; ".join(f"{html.escape(name)}: {number(deltas[row_index,index])}"
                                         for index, name in enumerate(LABELS))
                    + f". Mean absolute pixel change from neutral: {difference.mean():.3f}/255.</p>")
    with (output / "parameter_activation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (output / "index.html").write_text("\n".join(page))
    (output / "figure_manifest.json").write_text(json.dumps({
        "new_results": str(args.results), "previous_results": str(args.previous),
        "words": words, "control_order": list(CONTROL_NAMES),
        "neutral_settings": list(CONTROL_NEUTRAL), "control_bounds": list(CONTROL_BOUNDS),
        "plot_units": "Physical parameter delta from neutral; full control ranges; no radial polygons",
    }, indent=2))
    print(output / "index.html")


if __name__ == "__main__":
    main()