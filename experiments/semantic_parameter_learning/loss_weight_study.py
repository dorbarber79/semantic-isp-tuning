"""Compare semantic and quality-loss weights on one FiveK image and sentence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from autotune import BEAUTY_PROMPTS, REALISM_PROMPTS, CONTROL_NAMES, optimize
from fivek_preview import DEFAULT_CONFIG, DEFAULT_DATA, DEFAULT_SENTENCES, load_linear_png, resize_short_side, write_render
from fivek_portraits import CRITERIA
from moran_semantic_isp import linear_to_srgb, parameters_from_controls, render_linear
from pole_normalization import normalized_to_physical, pole_endpoints


DEFAULT_OUT = Path("/data/data1/semantic_isp/results/semantic_parameter_learning/fivek_brightness_loss_study")
DEFAULT_IMAGE = "a0002"
DEFAULT_SENTENCE = "D1_008"
CONDITIONS = (("semantic_only", 0.), ("current", 1.), ("stronger_quality", 3.))
ESTIMATE_KEYS = ("warmth", "saturation", "brightness", "contrast", "sharpness", "palette")
LOSS_NAMES = ("total", "semantic", "realism", "beauty", "regularization")


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def plot_convergence(path, record, quality_weight):
    restarts = record["convergence"]["restarts"]
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    colors = ("#3d687d", "#d57846", "#609b72", "#a84f75")
    for restart_index, restart in enumerate(restarts):
        rows = restart["trace"]
        times = [row["iteration"] for row in rows]
        label = f"{restart.get('start_id', 'palette ' + format(restart['palette_start'], '+.2f'))}; converged={restart['converged']}"
        for axis, name in zip(axes, LOSS_NAMES):
            axis.plot(times, [row[name] for row in rows], color=colors[restart_index % len(colors)],
                      linewidth=1.1, label=label)
            axis.set_ylabel(name)
            axis.grid(alpha=.22)
    axes[0].legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("Update")
    fig.suptitle(f"Loss components across all starts | realism={quality_weight:g}, beauty={quality_weight:g}\n"
                 "Components are unweighted except regularization; total includes the specified weights")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_comparison(path, entries, sentence, calibration_columns):
    columns, tile_width, image_height, title_height, label_height = len(entries), 400, 570, 82, 105
    table_top = title_height + label_height + image_height
    table_height = 48 + len(CONTROL_NAMES) * 28
    sheet = Image.new("RGB", (columns * tile_width, table_top + table_height), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    title_font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    draw.text((12, 12), sentence, fill="#17212b", font=title_font)
    draw.text((12, 46), "Neutral, agent estimate, and autotuned results", fill="#34444c", font=font)
    for index, (label, image_path, details) in enumerate(entries):
        left = index * tile_width
        draw.multiline_text((left + 10, title_height + 5), f"{label}\n{details}", fill="#17212b", font=font, spacing=4)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width - 12, image_height - 10), Image.Resampling.LANCZOS)
            sheet.paste(image, (left + (tile_width - image.width) // 2,
                                title_height + label_height + (image_height - image.height) // 2))
    draw.rectangle((0, table_top, columns * tile_width, table_top + table_height), fill="#f2f5f6")
    draw.text((12, table_top + 9), "Control", fill="#17212b", font=font)
    for label, controls, image_column in calibration_columns:
        draw.text((image_column * tile_width + 10, table_top + 9), label, fill="#17212b", font=font)
    for index, name in enumerate(CONTROL_NAMES):
        y = table_top + 37 + index * 28
        label = "brightness / tone_gain" if name == "tone_gain" else name
        draw.text((12, y), label, fill="#17212b", font=font)
        for _, controls, image_column in calibration_columns:
            draw.text((image_column * tile_width + 10, y), f"{controls[name]:+.3f}",
                      fill="#17212b", font=font)
    sheet.save(path)


def render_existing(args):
    summary_path = args.out / "summary.json"
    summary = json.loads(summary_path.read_text())
    manifest = json.loads((args.out / "manifest.json").read_text())
    config = json.loads(Path(manifest["config"]).read_text())
    endpoints = pole_endpoints(config)
    sentence = summary["sentence"]
    agent_normalized = dict(zip(CONTROL_NAMES,
        (sentence["ratings"][name] for name in ESTIMATE_KEYS)))
    agent_physical_tensor = normalized_to_physical(
        torch.tensor(list(agent_normalized.values()), dtype=torch.float64), endpoints)
    agent_physical = dict(zip(CONTROL_NAMES, agent_physical_tensor.tolist()))
    summary["agent_normalized_controls"] = agent_normalized
    summary["agent_physical_controls"] = agent_physical
    entries = [("Neutral", args.out / "images/neutral.png", "Physical neutral settings")]
    calibration_columns = [("Agent P90", agent_normalized, 1)]
    entries.append(("Agent estimate", args.out / "images/agent.png", "Applied directly; not optimized"))
    for name, weight in CONDITIONS:
        condition = summary["conditions"].get(name)
        if condition is None:
            continue
        entries.append((f"{name} | quality weight {weight:g}", args.out / f"images/{name}.png",
                        f"Brightness P90: {condition['normalized_controls']['tone_gain']:+.3f}"))
        image_column = len(entries) - 1
        calibration_columns.append((f"Quality {weight:g} P90", condition["normalized_controls"], image_column))
    for _, image_path, _ in entries:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    save_json(summary_path, summary)
    plot_comparison(args.out / "brightness_weight_comparison.png", entries, sentence["text"], calibration_columns)
    print(json.dumps({"output": str(args.out), "image": str(args.out / "brightness_weight_comparison.png"),
                      "image_id": manifest["image_id"], "sentence_id": sentence["id"]}, indent=2))


def component_stability(record, window):
    selected = next(restart for restart in record["convergence"]["restarts"]
                    if (restart.get("start_id") == record["start_id"] if "start_id" in record
                        else restart["palette_start"] == record["palette_start"]))
    recent = selected["trace"][-window:]
    return {name: {"first": float(recent[0][name]), "last": float(recent[-1][name]),
                   "range": float(max(row[name] for row in recent) - min(row[name] for row in recent))}
            for name in LOSS_NAMES}


def run(args):
    if args.render_existing:
        render_existing(args)
        return
    if args.max_iters < CRITERIA["min_iterations"] or args.size < 1 or args.lr <= 0:
        raise ValueError("Invalid optimization budget, image size, or learning rate")
    conditions = tuple((name, weight) for name, weight in CONDITIONS if name in args.conditions)
    config = json.loads(args.config.read_text())
    config["image_mean"] = [.5, .5, .5]
    config["image_std"] = [.5, .5, .5]
    sentences = json.loads(args.sentences.read_text())["sentences"]
    sentence = next((row for row in sentences if row["id"] == args.sentence_id), None)
    if sentence is None:
        raise ValueError(f"Unknown sentence ID: {args.sentence_id}")
    matches = sorted(args.data.glob(args.image_id + "-*.png"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one FiveK image for {args.image_id}, found {len(matches)}")
    source_paths = [Path(__file__), Path(__file__).with_name("autotune.py"),
                    Path(__file__).with_name("pole_normalization.py"),
                    Path(__file__).with_name("fivek_preview.py"), Path(__file__).with_name("fivek_portraits.py")]
    manifest = {"config": str(args.config.resolve()), "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "sentences": str(args.sentences.resolve()), "sentences_sha256": hashlib.sha256(args.sentences.read_bytes()).hexdigest(),
                "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
                "image_path": str(matches[0].resolve()), "image_id": args.image_id, "sentence": sentence,
                "semantic_model": "ViT-B-16-SigLIP (webli)", "image_mean": [.5, .5, .5], "image_std": [.5, .5, .5],
                "quality_prompts": {"realism": list(REALISM_PROMPTS), "beauty": list(BEAUTY_PROMPTS)},
                "semantic_weight": 1., "quality_weights": dict(conditions), "regularization_weight": .003,
                "parameterization": "P90 normalized; tanh raw coordinates mapped piecewise linearly to physical values",
                "palette_starts": [0., -.02, .02], "learning_rate": args.lr, "max_iterations_per_start": args.max_iters,
                "convergence_criteria": CRITERIA, "optimization_short_side": args.size,
                "caveat": "Proxy scores and convergence do not establish human-perceived realism or semantic correctness."}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing manifest differs; use a new output directory")
    elif any(args.out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.out}")
    else:
        save_json(manifest_path, manifest)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli", device=args.device)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    with torch.no_grad():
        text = F.normalize(model.encode_text(tokenizer([sentence["text"]]).to(args.device)).float(), dim=-1)
        quality_text = F.normalize(model.encode_text(tokenizer(list(REALISM_PROMPTS + BEAUTY_PROMPTS)).to(args.device)).float(), dim=-1)
    native = load_linear_png(matches[0]).to(args.device)
    small = resize_short_side(native, args.size)
    endpoints = pole_endpoints(config)
    neutral = torch.tensor([0., 0., 1., 1., 0., 0.], dtype=native.dtype, device=native.device)
    agent_normalized = torch.tensor([sentence["ratings"][name] for name in ESTIMATE_KEYS], dtype=native.dtype, device=native.device)
    if not torch.isfinite(agent_normalized).all() or bool((agent_normalized.abs() > 1).any()):
        raise ValueError("Agent estimates must be finite values within [-1, 1]")
    agent_controls = normalized_to_physical(agent_normalized, endpoints)
    images = args.out / "images"
    images.mkdir(exist_ok=True)
    entries = []
    for name, controls, label in (("neutral", neutral, "Neutral"), ("agent", agent_controls, "Agent estimate")):
        with torch.no_grad():
            rendered = render_linear(native, parameters_from_controls(controls, config))
            write_render(images / f"{name}.png", rendered)
            brightness = float(linear_to_srgb(rendered.clamp(0, 1)).mean())
        entries.append((label, images / f"{name}.png", f"Mean display RGB: {brightness:.3f}"))
    records = {}
    calibration_columns = [("Agent P90", dict(zip(CONTROL_NAMES, agent_normalized.detach().cpu().tolist())), 1)]
    for name, quality_weight in conditions:
        path = args.out / f"{name}.json"
        if path.exists():
            record = json.loads(path.read_text())
        else:
            print(f"Starting {name}: realism={quality_weight:g}, beauty={quality_weight:g}", flush=True)
            record = optimize(small, model, model, text, quality_text, config, iterations=args.max_iters,
                learning_rate=args.lr, realism_weight=quality_weight, beauty_weight=quality_weight,
                parameterization="p90", convergence=CRITERIA)
            save_json(path, record)
        records[name] = record
        controls = torch.tensor([record["controls"][control] for control in CONTROL_NAMES],
                                dtype=native.dtype, device=native.device)
        with torch.no_grad():
            rendered = render_linear(native, parameters_from_controls(controls, config))
            write_render(images / f"{name}.png", rendered)
            brightness = float(linear_to_srgb(rendered.clamp(0, 1)).mean())
        record_summary = {"quality_weight": quality_weight,
                          "normalized_controls": record["normalized_controls"], "physical_controls": record["controls"],
                          "losses": record["losses"], "semantic_similarity": record["semantic_similarity"],
                          "quality_scores": record["quality_scores"], "neutral_quality_scores": record["neutral_quality_scores"],
                          "mean_display_rgb": brightness,
                          "selected_restart_converged": record["convergence"]["selected_restart_converged"],
                          "all_restarts_converged": record["convergence"]["all_restarts_converged"],
                          "selected_restart_best_iteration": record["iteration"],
                          "selected_restart_palette_start": record["palette_start"],
                          "selected_restart_last_window_component_stability": component_stability(record, CRITERIA["window"])}
        save_json(args.out / f"{name}_summary.json", record_summary)
        plot_convergence(args.out / f"{name}_convergence.png", record, quality_weight)
        with (args.out / f"{name}_trace.csv").open("w", newline="") as stream:
            fieldnames = ["palette_start", "iteration", *LOSS_NAMES, "semantic_similarity", "quality_realism", "quality_beauty",
                          "gradient_max_abs", "learning_rate", *CONTROL_NAMES]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for restart in record["convergence"]["restarts"]:
                for row in restart["trace"]:
                    writer.writerow({"palette_start": restart["palette_start"], "iteration": row["iteration"],
                        **{key: row[key] for key in LOSS_NAMES}, "semantic_similarity": row["semantic_similarity"],
                        "quality_realism": row["quality_scores"][0], "quality_beauty": row["quality_scores"][1],
                        "gradient_max_abs": row["gradient_max_abs"], "learning_rate": row["learning_rate"],
                        **dict(zip(CONTROL_NAMES, row["controls"]))})
        entries.append((f"{name} | quality weight {quality_weight:g}", images / f"{name}.png",
                        f"Brightness P90: {record['normalized_controls']['tone_gain']:+.3f}\n"
                        f"Mean display RGB: {brightness:.3f}\n"
                        f"Selected converged: {record['convergence']['selected_restart_converged']}"))
        calibration_columns.append((f"Quality {quality_weight:g} P90", record["normalized_controls"], len(entries) - 1))
        print(f"Completed {name}: brightness={record['normalized_controls']['tone_gain']:+.3f}, "
              f"selected_converged={record['convergence']['selected_restart_converged']}, "
              f"all_converged={record['convergence']['all_restarts_converged']}", flush=True)
    plot_comparison(args.out / "brightness_weight_comparison.png", entries, sentence["text"], calibration_columns)
    summary = {"sentence": sentence, "image": str(matches[0]),
        "agent_normalized_controls": dict(zip(CONTROL_NAMES, agent_normalized.detach().cpu().tolist())),
        "agent_physical_controls": dict(zip(CONTROL_NAMES, agent_controls.detach().cpu().tolist())), "conditions": {
        name: json.loads((args.out / f"{name}_summary.json").read_text()) for name, _ in conditions},
        "warning": "Weighted totals across different objectives are not directly comparable; quality scores are model proxies."}
    save_json(args.out / "summary.json", summary)
    print(json.dumps({"output": str(args.out), "conditions": list(records),
                      "image": str(args.out / "brightness_weight_comparison.png")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--image-id", default=DEFAULT_IMAGE)
    parser.add_argument("--sentence-id", default=DEFAULT_SENTENCE)
    parser.add_argument("--conditions", nargs="+", choices=[name for name, _ in CONDITIONS],
                        default=[name for name, _ in CONDITIONS])
    parser.add_argument("--render-existing", action="store_true",
                        help="Regenerate comparison and calibration table from saved results without rerunning optimization")
    parser.add_argument("--max-iters", type=int, default=1200)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=.03)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()