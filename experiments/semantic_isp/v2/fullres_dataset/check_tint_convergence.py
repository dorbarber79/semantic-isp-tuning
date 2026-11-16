"""Compare full-budget Tint fits at increasing step counts on a frozen hard set."""

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

SCRIPT = Path(__file__).resolve()
V2_ROOT = SCRIPT.parents[1]
FULLRES_ROOT = SCRIPT.parent
for path in (V2_ROOT, FULLRES_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from moran_semantic_isp import CONTROL_NAMES, parameters_from_controls, render_linear
from process_fullres import (CONFIG_PATH, calibration_endpoints,
                             display_quality_metrics, fit_controls)
from fit_tint_ablation import load_linear_rgb16
from rank_hard_fits import read_latest_records, safe_relative_path

DEFAULT_ROOT = Path("/data/data1/FiveK/C_data/datasets/FiveK-FullRes-v1")
DEFAULT_HARD_SET = Path("analysis_results/hard_fit_crop_only_v1/hard_fit_test_set.json")
BUDGETS = (2000, 4000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--hard-set", type=Path, default=None)
    parser.add_argument("--manifest-name", default="process_crop_only_manifest_v1.jsonl")
    parser.add_argument("--output-name", default="tint_convergence_pilot_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=.03)
    args = parser.parse_args()

    root = args.root.expanduser().resolve(strict=True)
    hard_path = (args.hard_set or root / DEFAULT_HARD_SET).expanduser().resolve(strict=True)
    hard_set = json.loads(hard_path.read_text(encoding="utf-8"))
    records = read_latest_records(root / "metadata" / args.manifest_name)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    torch.manual_seed(0)
    torch.set_num_threads(2)
    endpoints, _ = calibration_endpoints(config, device, torch.float32)

    output = root / "analysis_results" / args.output_name
    output.mkdir(parents=True, exist_ok=False)
    detailed, rows = [], []
    final_comparison = f"{BUDGETS[-1]}_vs_{BUDGETS[-2]}"
    for pair_index, pair in enumerate(hard_set["pairs"], start=1):
        scene_id, expert = pair["scene_id"], pair["expert"]
        record = records.get(scene_id)
        if not record or record.get("status") != "complete":
            raise ValueError(f"Hard-set scene is not complete: {scene_id}")
        fit_record = record["fits"][expert]
        if fit_record["target_alignment"].get("status") != "aligned_crop_only":
            raise ValueError(f"Hard-set target is not crop aligned: {scene_id}/{expert}")
        files = fit_record["files"]
        baseline_path = safe_relative_path(root, files["aligned_baseline_linear_16bit"]["path"])
        target_path = safe_relative_path(root, files["target_linear_16bit"]["path"])
        baseline = load_linear_rgb16(baseline_path, device)
        target = load_linear_rgb16(target_path, device)
        pair_result = {"scene_id": scene_id, "expert": expert, "budgets": {}}
        metric_row = {"scene_id": scene_id, "expert": expert}

        for budget in BUDGETS:
            fit = fit_controls(baseline, target, config, endpoints, budget, args.learning_rate)
            controls = torch.tensor([fit["physical_controls"][name] for name in CONTROL_NAMES],
                                    device=device, dtype=baseline.dtype)
            parameters = parameters_from_controls(controls, config)
            parameters["tint"] = baseline.new_tensor(fit["tint"])
            with torch.no_grad():
                candidate = render_linear(baseline, parameters)
            metrics = display_quality_metrics(target, candidate)
            fit["quality_metrics"] = metrics
            pair_result["budgets"][str(budget)] = fit
            metric_row[f"steps_{budget}"] = budget
            metric_row[f"loss_{budget}"] = fit["loss"]
            metric_row[f"psnr_db_{budget}"] = metrics["psnr_db"]
            metric_row[f"ssim_{budget}"] = metrics["ssim"]
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for lower_budget, upper_budget in zip(BUDGETS, BUDGETS[1:]):
            suffix = f"{upper_budget}_vs_{lower_budget}"
            metric_row[f"delta_psnr_db_{suffix}"] = (
                metric_row[f"psnr_db_{upper_budget}"] - metric_row[f"psnr_db_{lower_budget}"])
            metric_row[f"delta_ssim_{suffix}"] = (
                metric_row[f"ssim_{upper_budget}"] - metric_row[f"ssim_{lower_budget}"])
            metric_row[f"delta_loss_{suffix}"] = (
                metric_row[f"loss_{upper_budget}"] - metric_row[f"loss_{lower_budget}"])
        rows.append(metric_row)
        detailed.append(pair_result)
        print(json.dumps({"completed_pairs": pair_index, "total_pairs": len(hard_set["pairs"]),
                          "scene_id": scene_id, "expert": expert,
                          f"delta_psnr_db_{final_comparison}": metric_row[f"delta_psnr_db_{final_comparison}"],
                          f"delta_ssim_{final_comparison}": metric_row[f"delta_ssim_{final_comparison}"]}), flush=True)
        del baseline, target

    fields = ["scene_id", "expert"]
    for budget in BUDGETS:
        fields.extend((f"steps_{budget}", f"loss_{budget}", f"psnr_db_{budget}", f"ssim_{budget}"))
    for lower_budget, upper_budget in zip(BUDGETS, BUDGETS[1:]):
        suffix = f"{upper_budget}_vs_{lower_budget}"
        fields.extend((f"delta_loss_{suffix}", f"delta_psnr_db_{suffix}", f"delta_ssim_{suffix}"))
    with (output / "convergence_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    convergence_by_budget = {}
    for budget in BUDGETS:
        restart_rows = [restart for pair in detailed
                        for restart in pair["budgets"][str(budget)]["convergence"]["restart_diagnostics"]]
        convergence_by_budget[str(budget)] = {
            "converged_restarts": sum(row["converged_by_0_5_percent_tail_rule"] for row in restart_rows),
            "total_restarts": len(restart_rows),
            "pairs_converged_all_restarts": sum(
                all(restart["converged_by_0_5_percent_tail_rule"]
                    for restart in pair["budgets"][str(budget)]["convergence"]["restart_diagnostics"])
                for pair in detailed),
            "median_best_iteration": sorted(row["best_iteration"] for row in restart_rows)[len(restart_rows) // 2],
            "max_final_to_best_relative_gap": max(row["final_to_best_relative_gap"] for row in restart_rows),
        }
    budget_comparisons = {}
    for lower_budget, upper_budget in zip(BUDGETS, BUDGETS[1:]):
        suffix = f"{upper_budget}_vs_{lower_budget}"
        psnr_deltas = [row[f"delta_psnr_db_{suffix}"] for row in rows]
        ssim_deltas = [row[f"delta_ssim_{suffix}"] for row in rows]
        budget_comparisons[suffix] = {
            "mean_delta_psnr_db": sum(psnr_deltas) / len(psnr_deltas),
            "median_delta_psnr_db": sorted(psnr_deltas)[len(psnr_deltas) // 2],
            "pairs_improved_psnr": sum(value > 0 for value in psnr_deltas),
            "mean_delta_ssim": sum(ssim_deltas) / len(ssim_deltas),
            "median_delta_ssim": sorted(ssim_deltas)[len(ssim_deltas) // 2],
            "pairs_improved_ssim": sum(value > 0 for value in ssim_deltas),
        }
    summary = {
        "hard_set": hard_path.relative_to(root).as_posix(),
        "eligible_pairs": len(rows),
        "budgets_are_optimizer_steps_per_restart": list(BUDGETS),
        "restarts_per_pair": 3,
        "early_stopping": False,
        "tail_rule": "Every restart: best loss at least 5 steps before budget; <0.5% relative improvement over last 50 steps; <0.5% final-to-best gap.",
        "convergence_by_budget": convergence_by_budget,
        "budget_comparisons": budget_comparisons,
        "metrics_csv": "convergence_metrics.csv",
        "full_diagnostics": "convergence_details.json",
    }
    (output / "convergence_details.json").write_text(
        json.dumps({"summary": summary, "pairs": detailed}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    print(json.dumps({"output_directory": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()