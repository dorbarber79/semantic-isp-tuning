"""Warm-start added color DoFs on a frozen hard-fit pair set."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from axis_isp import linear_to_srgb8
from semantic_isp import CONTROL_NAMES, parameters_from_controls, render_linear
from process_fullres import (CONFIG_PATH, calibration_endpoints, objective,
                             physical_controls)
from rank_hard_fits import read_latest_records, safe_relative_path, score_pair

DEFAULT_ROOT = Path("/data/data1/FiveK/C_data/datasets/FiveK-FullRes-v1")
DEFAULT_HARD_SET = Path("analysis_results/hard_fit_crop_only_v1/hard_fit_test_set.json")
DOF_SPECS = {
    "tint": {"limit": 0.3,
             "definition": "R/B gains 1+t; G gain 1-((0.299+0.114)/0.587)*t"},
    "vibrance": {"limit": 0.5,
                 "definition": "Luma-preserving chroma boost weighted by 1 - clipped RGB channel range"},
}
EXTRA_DOF_REGULARIZATION = 0.002


def load_linear_rgb16(path, device):
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype != np.uint16 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected a uint16 RGB PNG: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32) / 65535
    return tensor.permute(2, 0, 1).unsqueeze(0)


def fit_extra_dofs(baseline, target, saved_fit, config, endpoints, dofs,
                   iterations, learning_rate):
    height, width = baseline.shape[-2:]
    scale = 256 / min(height, width)
    fit_height = max(1, round(height * scale))
    fit_width = max(1, round(width * scale))
    fit_size = (fit_height, fit_width)
    fit_image = F.interpolate(baseline, size=fit_size, mode="area")
    fit_target = F.interpolate(target, size=fit_size, mode="area")

    initial = torch.tensor([saved_fit["normalized_controls"][name] for name in CONTROL_NAMES],
                           device=baseline.device, dtype=baseline.dtype).clamp(-.999999, .999999)
    initial_raw = torch.atanh(initial)
    raw = torch.nn.Parameter(torch.cat((initial_raw, initial.new_zeros(len(dofs)))))
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    best = None

    for iteration in range(iterations + 1):
        normalized = torch.tanh(raw[:len(CONTROL_NAMES)])
        extra_normalized = torch.tanh(raw[len(CONTROL_NAMES):])
        controls = physical_controls(normalized, endpoints)
        parameters = parameters_from_controls(controls, config)
        for dof, amount in zip(dofs, extra_normalized):
            parameters[dof] = DOF_SPECS[dof]["limit"] * amount
        rendered = render_linear(fit_image, parameters)
        loss = objective(rendered, fit_target, normalized)
        loss = loss + EXTRA_DOF_REGULARIZATION * extra_normalized.square().sum()
        value = float(loss.detach())
        if best is None or value < best["loss"]:
            best = {"loss": value, "iteration": iteration,
                    "normalized_controls": normalized.detach().cpu().tolist(),
                    "extra_dofs_normalized": extra_normalized.detach().cpu().tolist()}
        if iteration < iterations:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    final_normalized = baseline.new_tensor(best["normalized_controls"])
    final_extra_normalized = baseline.new_tensor(best["extra_dofs_normalized"])
    final_controls = physical_controls(final_normalized, endpoints)
    final_parameters = parameters_from_controls(final_controls, config)
    for dof, amount in zip(dofs, final_extra_normalized):
        final_parameters[dof] = DOF_SPECS[dof]["limit"] * amount
    with torch.no_grad():
        full_render = render_linear(baseline, final_parameters)
    best["physical_controls"] = dict(zip(CONTROL_NAMES, final_controls.detach().cpu().tolist()))
    best["extra_dofs"] = {
        dof: float((DOF_SPECS[dof]["limit"] * amount).detach())
        for dof, amount in zip(dofs, final_extra_normalized)
    }
    return full_render, best


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--hard-set", type=Path, default=None)
    parser.add_argument("--manifest-name", default="process_crop_only_manifest_v1.jsonl")
    parser.add_argument("--dof", choices=tuple(DOF_SPECS), action="append", default=None,
                        help="Add a candidate DoF; repeat to fit multiple DoFs together")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=.03)
    args = parser.parse_args()
    if args.iterations < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Iterations and learning rate must be positive")
    dofs = args.dof or ["tint"]
    if len(set(dofs)) != len(dofs):
        raise ValueError("Each candidate DoF may be specified only once")

    root = args.root.expanduser().resolve(strict=True)
    hard_path = args.hard_set or root / DEFAULT_HARD_SET
    hard_path = hard_path.expanduser().resolve(strict=True)
    hard_set = json.loads(hard_path.read_text(encoding="utf-8"))
    manifest_path = root / "metadata" / args.manifest_name
    records = read_latest_records(manifest_path)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    endpoints, _ = calibration_endpoints(config, device, torch.float32)

    dof_label = "_".join(dofs)
    output_name = args.output_name or f"{dof_label}_hard_fit_v1"
    output = root / "analysis_results" / output_name
    output.mkdir(parents=True, exist_ok=False)
    render_dir = output / "candidate_renders"
    render_dir.mkdir()
    results = []
    for index, pair in enumerate(hard_set["pairs"], start=1):
        scene_id, expert = pair["scene_id"], pair["expert"]
        record = records.get(scene_id)
        if not record or record.get("status") != "complete":
            raise ValueError(f"Hard-set scene is not complete in crop manifest: {scene_id}")
        expert_record = record["fits"][expert]
        alignment = expert_record["target_alignment"]
        if alignment.get("status") != "aligned_crop_only" or alignment.get("resized_or_warped"):
            raise ValueError(f"Hard-set pair is not crop-only aligned: {scene_id}/{expert}")
        files = expert_record["files"]
        baseline_path = safe_relative_path(root, files["aligned_baseline_linear_16bit"]["path"])
        target_path = safe_relative_path(root, files["target_linear_16bit"]["path"])
        target_display_path = safe_relative_path(root, files["target_display_8bit"]["path"])
        baseline_display_path = safe_relative_path(root, files["display_8bit"]["path"])
        baseline = load_linear_rgb16(baseline_path, device)
        target = load_linear_rgb16(target_path, device)
        if baseline.shape != target.shape:
            raise ValueError(f"Aligned linear dimensions differ: {scene_id}/{expert}")

        candidate, fit = fit_extra_dofs(baseline, target, expert_record["fit"], config,
                        endpoints, dofs, args.iterations, args.learning_rate)
        if "/" in scene_id or "\\" in scene_id:
            raise ValueError(f"Unexpected path separator in scene ID: {scene_id}")
        candidate_path = render_dir / f"{scene_id}_{expert}.png"
        pixels = linear_to_srgb8(candidate)[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(pixels, "RGB").save(candidate_path)
        baseline_metrics = score_pair(target_display_path, baseline_display_path)
        candidate_metrics = score_pair(target_display_path, candidate_path)
        results.append({
            "scene_id": scene_id, "expert": expert,
            "baseline_psnr_db": baseline_metrics["psnr_db"],
            "candidate_psnr_db": candidate_metrics["psnr_db"],
            "delta_psnr_db": candidate_metrics["psnr_db"] - baseline_metrics["psnr_db"],
            "baseline_ssim": baseline_metrics["ssim"],
            "candidate_ssim": candidate_metrics["ssim"],
            "delta_ssim": candidate_metrics["ssim"] - baseline_metrics["ssim"],
            "baseline_loss": expert_record["fit"].get("loss"),
            "candidate_loss": fit["loss"], "best_iteration": fit["iteration"],
            "dof_values": json.dumps(fit["extra_dofs"], sort_keys=True),
            "normalized_controls": fit["normalized_controls"],
            "physical_controls": fit["physical_controls"],
            "candidate_path": candidate_path.relative_to(root).as_posix(),
        })
        print(json.dumps({"completed": index, "total": len(hard_set["pairs"]),
                          "scene_id": scene_id, "expert": expert,
                          "delta_psnr_db": results[-1]["delta_psnr_db"],
                          "delta_ssim": results[-1]["delta_ssim"],
                          "dof_values": fit["extra_dofs"]}), flush=True)
        del baseline, target, candidate
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fields = ["scene_id", "expert", "baseline_psnr_db", "candidate_psnr_db", "delta_psnr_db",
              "baseline_ssim", "candidate_ssim", "delta_ssim", "baseline_loss", "candidate_loss",
              "best_iteration", "dof_values", "candidate_path"]
    results_filename = f"{dof_label}_ablation_metrics.csv"
    write_csv(output / results_filename, results, fields)
    summary = {
        "hard_set": hard_path.relative_to(root).as_posix(),
        "hard_set_sha256": hashlib.sha256(hard_path.read_bytes()).hexdigest(),
        "manifest": f"metadata/{args.manifest_name}",
        "candidate": f"six learned controls plus {', '.join(dofs)}",
        "dofs": {dof: {**DOF_SPECS[dof]} for dof in dofs},
        "optimizer": {"iterations": args.iterations, "learning_rate": args.learning_rate,
                  "initialization": "saved six-control hard-fit optimum; added DoFs initialized to zero"},
        "pairs": len(results),
        "mean_baseline_psnr_db": float(np.mean([row["baseline_psnr_db"] for row in results])),
        "mean_candidate_psnr_db": float(np.mean([row["candidate_psnr_db"] for row in results])),
        "mean_delta_psnr_db": float(np.mean([row["delta_psnr_db"] for row in results])),
        "median_delta_psnr_db": float(np.median([row["delta_psnr_db"] for row in results])),
        "pairs_improved_psnr": sum(row["delta_psnr_db"] > 0 for row in results),
        "mean_baseline_ssim": float(np.mean([row["baseline_ssim"] for row in results])),
        "mean_candidate_ssim": float(np.mean([row["candidate_ssim"] for row in results])),
        "mean_delta_ssim": float(np.mean([row["delta_ssim"] for row in results])),
        "median_delta_ssim": float(np.median([row["delta_ssim"] for row in results])),
        "pairs_improved_ssim": sum(row["delta_ssim"] > 0 for row in results),
        "results_csv": results_filename,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    print(json.dumps({"output_directory": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()