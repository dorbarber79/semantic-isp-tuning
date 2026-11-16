"""Rank crop-only photographer fits by native-size display PSNR and SSIM."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

DEFAULT_ROOT = Path("/data/data1/FiveK/C_data/datasets/FiveK-FullRes-v1")


def read_latest_records(path):
    latest = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            scene_id = record.get("scene_id")
            if scene_id:
                latest[scene_id] = record
    return latest


def safe_relative_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a safe relative output path: {relative}")
    return root / path


def score_pair(target_path, fit_path):
    target = np.asarray(Image.open(target_path).convert("RGB"), dtype=np.uint8)
    fit = np.asarray(Image.open(fit_path).convert("RGB"), dtype=np.uint8)
    if target.shape != fit.shape:
        raise ValueError(f"Target/fit dimensions differ: {target_path} / {fit_path}")
    delta = target.astype(np.float64) - fit.astype(np.float64)
    mse = float(np.mean(delta ** 2))
    psnr = float("inf") if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))
    ssim = float(structural_similarity(target, fit, channel_axis=-1, data_range=255))
    return {
        "psnr_db": psnr,
        "ssim": ssim,
        "mae_8bit": float(np.mean(np.abs(delta))),
        "dimensions_wh": [int(target.shape[1]), int(target.shape[0])],
    }


def add_low_score_percentiles(rows, key, destination):
    order = sorted(range(len(rows)), key=lambda index: (rows[index][key], index))
    if len(order) == 1:
        rows[order[0]][destination] = 0.0
        return
    for rank, index in enumerate(order):
        rows[index][destination] = rank / (len(order) - 1)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest-name", default="process_crop_only_manifest_v1.jsonl")
    parser.add_argument("--out-name", default="hard_fit_crop_only_v1")
    parser.add_argument("--hard-pairs", type=int, default=20)
    args = parser.parse_args()
    if args.hard_pairs < 1:
        raise ValueError("--hard-pairs must be positive")

    root = args.root.expanduser().resolve(strict=True)
    manifest_path = root / "metadata" / args.manifest_name
    records = read_latest_records(manifest_path)
    rows, rejected_scenes = [], []
    for scene_id, record in sorted(records.items()):
        if record.get("status") != "complete" or not isinstance(record.get("fits"), dict):
            rejected_scenes.append({"scene_id": scene_id, "status": record.get("status"),
                                    "error": record.get("error")})
            continue
        for expert, expert_record in sorted(record["fits"].items()):
            alignment = expert_record.get("target_alignment", {})
            if alignment.get("status") != "aligned_crop_only" or alignment.get("resized_or_warped"):
                raise ValueError(f"Non-crop-only fit found in {scene_id}/{expert}")
            files = expert_record["files"]
            target_path = safe_relative_path(root, files["target_display_8bit"]["path"])
            fit_path = safe_relative_path(root, files["display_8bit"]["path"])
            metrics = score_pair(target_path, fit_path)
            rows.append({"scene_id": scene_id, "expert": expert, **metrics,
                         "optimization_loss": expert_record["fit"].get("loss"),
                         "target_path": target_path.relative_to(root).as_posix(),
                         "fit_path": fit_path.relative_to(root).as_posix()})

    if not rows:
        raise RuntimeError(f"No completed crop-only expert fits in {manifest_path}")
    add_low_score_percentiles(rows, "psnr_db", "psnr_low_percentile")
    add_low_score_percentiles(rows, "ssim", "ssim_low_percentile")
    for row in rows:
        row["combined_low_percentile"] = (row["psnr_low_percentile"]
                                           + row["ssim_low_percentile"]) / 2
    ranked = sorted(rows, key=lambda row: (row["combined_low_percentile"],
                                           row["psnr_db"], row["ssim"]))
    hard_rows = ranked[:min(args.hard_pairs, len(ranked))]

    output = root / "analysis_results" / args.out_name
    output.mkdir(parents=True, exist_ok=False)
    fields = ["scene_id", "expert", "psnr_db", "ssim", "mae_8bit", "dimensions_wh",
              "psnr_low_percentile", "ssim_low_percentile", "combined_low_percentile",
              "optimization_loss", "target_path", "fit_path"]
    write_csv(output / "all_fit_metrics.csv", rows, fields)
    write_csv(output / "hard_fit_test_set.csv", hard_rows, fields)
    test_set = {
        "selection_rule": "Lowest mean of ascending PSNR and SSIM percentile ranks across completed crop-only expert pairs.",
        "metric_scope": "Native-dimension 8-bit sRGB target and fitted render; higher PSNR/SSIM is better.",
        "manifest": f"metadata/{args.manifest_name}",
        "eligible_pairs": len(rows),
        "requested_hard_pairs": args.hard_pairs,
        "selected_hard_pairs": len(hard_rows),
        "selected_scenes": sorted({row["scene_id"] for row in hard_rows}),
        "pairs": [{key: row[key] for key in fields} for row in hard_rows],
        "rejected_scenes": rejected_scenes,
    }
    (output / "hard_fit_test_set.json").write_text(
        json.dumps(test_set, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"analysis_directory": str(output),
                      "eligible_pairs": len(rows),
                      "selected_hard_pairs": len(hard_rows),
                      "selected_scenes": test_set["selected_scenes"],
                      "rejected_scenes": rejected_scenes}, indent=2))


if __name__ == "__main__":
    main()