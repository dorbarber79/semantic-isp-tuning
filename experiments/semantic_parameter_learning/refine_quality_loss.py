"""Refine a stronger-quality loss from previously found ISP solutions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

from autotune import BEAUTY_PROMPTS, REALISM_PROMPTS, CONTROL_NAMES, optimize
from fivek_portraits import CRITERIA
from fivek_preview import load_linear_png, resize_short_side, write_render
from loss_weight_study import ESTIMATE_KEYS, LOSS_NAMES, component_stability, plot_comparison, plot_convergence, save_json
from semantic_isp import linear_to_srgb, parameters_from_controls, render_linear


DEFAULT_BASE = Path("/data/data1/semantic_isp/results/semantic_parameter_learning/fivek_brightness_loss_study")
DEFAULT_OUT = Path("/data/data1/semantic_isp/results/semantic_parameter_learning/fivek_brightness_weight3_refined")


def run(base, output, device, max_iters):
    if max_iters < CRITERIA["min_iterations"]:
        raise ValueError("Max iterations must meet the convergence minimum")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = json.loads((base / "manifest.json").read_text())
    config_path = Path(manifest["config"])
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("Configuration no longer matches the original study")
    config = json.loads(config_path.read_text())
    config["image_mean"], config["image_std"] = [.5, .5, .5], [.5, .5, .5]
    previous_current = json.loads((base / "current.json").read_text())
    previous_strong = json.loads((base / "stronger_quality.json").read_text())
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli", device=device)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    with torch.no_grad():
        text = F.normalize(model.encode_text(tokenizer([manifest["sentence"]["text"]]).to(device)).float(), dim=-1)
        quality_text = F.normalize(model.encode_text(tokenizer(list(REALISM_PROMPTS + BEAUTY_PROMPTS)).to(device)).float(), dim=-1)
    image = load_linear_png(Path(manifest["image_path"])).to(device)
    small = resize_short_side(image, manifest["optimization_short_side"])
    result = optimize(small, model, model, text, quality_text, config, iterations=max_iters,
        learning_rate=manifest["learning_rate"], realism_weight=3., beauty_weight=3.,
        parameterization="p90", convergence=CRITERIA, palette_starts=(0.,),
        additional_raw_starts=[{"name": "from_weight_1", "raw_controls": previous_current["raw_controls"]},
                               {"name": "from_weight_3", "raw_controls": previous_strong["raw_controls"]}])
    for restart in result["convergence"]["restarts"]:
        if restart["start_id"] in ("from_weight_1", "from_weight_3"):
            if result["losses"]["total"] > restart["trace"][0]["total"] + 2e-6:
                raise AssertionError("Refinement returned a worse objective than a supplied warm start")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "refined.json", result)
    controls = torch.tensor([result["controls"][name] for name in CONTROL_NAMES], device=device, dtype=image.dtype)
    with torch.no_grad():
        rendered = render_linear(image, parameters_from_controls(controls, config))
        write_render(output / "refined.png", rendered)
        brightness = float(linear_to_srgb(rendered.clamp(0, 1)).mean())
    plot_convergence(output / "refined_convergence.png", result, 3.)
    with (output / "refined_trace.csv").open("w", newline="") as stream:
        fieldnames = ["start_id", "iteration", *LOSS_NAMES, "semantic_similarity", "quality_realism", "quality_beauty",
                      "gradient_max_abs", "learning_rate", *CONTROL_NAMES]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for restart in result["convergence"]["restarts"]:
            for row in restart["trace"]:
                writer.writerow({"start_id": restart["start_id"], "iteration": row["iteration"],
                    **{key: row[key] for key in LOSS_NAMES}, "semantic_similarity": row["semantic_similarity"],
                    "quality_realism": row["quality_scores"][0], "quality_beauty": row["quality_scores"][1],
                    "gradient_max_abs": row["gradient_max_abs"], "learning_rate": row["learning_rate"],
                    **dict(zip(CONTROL_NAMES, row["controls"]))})
    entries = [("Neutral", base / "images/neutral.png", ""),
               ("Agent estimate", base / "images/agent.png", ""),
               ("Weight 1 result", base / "images/current.png", f"P90 brightness: {previous_current['normalized_controls']['tone_gain']:+.3f}"),
               ("Weight 3 original", base / "images/stronger_quality.png", f"P90 brightness: {previous_strong['normalized_controls']['tone_gain']:+.3f}"),
               ("Weight 3 refined", output / "refined.png", f"P90 brightness: {result['normalized_controls']['tone_gain']:+.3f}\n"
                f"Selected converged: {result['convergence']['selected_restart_converged']}")]
    agent_normalized = dict(zip(CONTROL_NAMES,
        (manifest["sentence"]["ratings"][name] for name in ESTIMATE_KEYS)))
    calibration_columns = [("Agent P90", agent_normalized, 1)]
    for label, controls, image_column in (
            ("Weight 1 P90", previous_current["normalized_controls"], 2),
            ("Weight 3 original P90", previous_strong["normalized_controls"], 3),
            ("Weight 3 refined P90", result["normalized_controls"], 4)):
        calibration_columns.append((label, controls, image_column))
    plot_comparison(output / "refined_comparison.png", entries, manifest["sentence"]["text"], calibration_columns)
    selected = next(restart for restart in result["convergence"]["restarts"] if restart["start_id"] == result["start_id"])
    summary = {"base_study": str(base.resolve()), "image": manifest["image_path"], "sentence": manifest["sentence"],
               "semantic_model": manifest["semantic_model"], "quality_weight": 3.,
               "max_iterations_per_start": max_iters, "criteria": CRITERIA,
               "restart_initial_objectives": {restart["start_id"]: restart["trace"][0]["total"]
                                              for restart in result["convergence"]["restarts"]},
               "selected_start": result["start_id"], "selected_best_iteration": result["iteration"],
               "selected_stop_iteration": selected["updates"], "selected_final_gradient_max": selected["trace"][-1]["gradient_max_abs"],
               "selected_converged": result["convergence"]["selected_restart_converged"],
               "all_restarts_converged": result["convergence"]["all_restarts_converged"],
               "agent_normalized_controls": agent_normalized,
               "losses": result["losses"], "normalized_controls": result["normalized_controls"],
               "physical_controls": result["controls"], "semantic_similarity": result["semantic_similarity"],
               "quality_scores": result["quality_scores"], "neutral_quality_scores": result["neutral_quality_scores"],
               "mean_display_rgb": brightness,
               "selected_last_window_component_stability": component_stability(result, CRITERIA["window"]),
               "caveat": "A locally converged proxy objective is not proof of global optimality or perceptual quality."}
    save_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), "selected_start": result["start_id"],
        "best_total": result["losses"]["total"], "p90_brightness": result["normalized_controls"]["tone_gain"],
        "selected_converged": result["convergence"]["selected_restart_converged"],
        "all_restarts_converged": result["convergence"]["all_restarts_converged"],
        "restarts": [(restart["start_id"], restart["updates"], restart["converged"])
                     for restart in result["convergence"]["restarts"]]}, indent=2))


def render_existing(base, output):
    manifest = json.loads((base / "manifest.json").read_text())
    previous_current = json.loads((base / "current.json").read_text())
    previous_strong = json.loads((base / "stronger_quality.json").read_text())
    result = json.loads((output / "refined.json").read_text())
    agent_normalized = dict(zip(CONTROL_NAMES,
        (manifest["sentence"]["ratings"][name] for name in ESTIMATE_KEYS)))
    entries = [("Neutral", base / "images/neutral.png", ""),
               ("Agent estimate", base / "images/agent.png", ""),
               ("Weight 1 result", base / "images/current.png",
                f"P90 brightness: {previous_current['normalized_controls']['tone_gain']:+.3f}"),
               ("Weight 3 original", base / "images/stronger_quality.png",
                f"P90 brightness: {previous_strong['normalized_controls']['tone_gain']:+.3f}"),
               ("Weight 3 refined", output / "refined.png",
                f"P90 brightness: {result['normalized_controls']['tone_gain']:+.3f}\n"
                f"Selected converged: {result['convergence']['selected_restart_converged']}")]
    calibration_columns = [("Agent P90", agent_normalized, 1),
                           ("Weight 1 P90", previous_current["normalized_controls"], 2),
                           ("Weight 3 original P90", previous_strong["normalized_controls"], 3),
                           ("Weight 3 refined P90", result["normalized_controls"], 4)]
    for _, image_path, _ in entries:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    plot_comparison(output / "refined_comparison.png", entries, manifest["sentence"]["text"], calibration_columns)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["agent_normalized_controls"] = agent_normalized
    save_json(summary_path, summary)
    print(json.dumps({"output": str(output), "image": str(output / "refined_comparison.png"),
                      "agent_normalized_controls": agent_normalized}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-iters", type=int, default=1200)
    parser.add_argument("--render-existing", action="store_true",
                        help="Regenerate comparison and agent calibration table from saved results")
    args = parser.parse_args()
    if args.render_existing:
        render_existing(args.base, args.out)
    else:
        run(args.base, args.out, args.device, args.max_iters)


if __name__ == "__main__":
    main()