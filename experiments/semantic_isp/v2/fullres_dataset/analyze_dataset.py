"""Create a standalone analysis snapshot for the full-resolution FiveK dataset."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rawpy
import torch

SCRIPT = Path(__file__).resolve()
V2_ROOT = SCRIPT.parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from axis_isp import linear_to_srgb8
from semantic_isp import CONTROL_NAMES, parameters_from_controls, render_linear

DEFAULT_ROOT = Path("/data/data1/FiveK/C_data/datasets/FiveK-FullRes-v1")


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def latest_by_scene(rows):
    latest = {}
    for row in rows:
        scene_id = row.get("scene_id", row.get("image_id"))
        if scene_id:
            latest[scene_id] = row
    return latest


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def image_psnr(target_path, fit_path):
    target = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
    fit = cv2.imread(str(fit_path), cv2.IMREAD_UNCHANGED)
    if target is None or fit is None:
        raise FileNotFoundError(f"Missing PSNR image: {target_path} / {fit_path}")
    if target.dtype != np.uint8 or fit.dtype != np.uint8 or target.shape != fit.shape:
        raise ValueError(f"Expected equal-shape uint8 display PNGs: {target_path}, {fit_path}")
    delta = target.astype(np.float32) - fit.astype(np.float32)
    mse = float(np.mean(delta * delta))
    mae = float(np.mean(np.abs(delta)))
    psnr = float("inf") if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))
    return {"psnr_db": psnr, "mae_8bit": mae, "mse_8bit": mse,
            "dimensions_wh": [int(target.shape[1]), int(target.shape[0])]}


def source_dimension_rows(root, source_records):
    rows = []
    for record in source_records:
        sources = record.get("sources", [])
        raw_record = next((item for item in sources if item.get("role") == "dng"), None)
        if not raw_record or not raw_record.get("path"):
            continue
        raw_path = root / raw_record["path"]
        with rawpy.imread(str(raw_path)) as raw:
            sizes = raw.sizes
            flip = int(sizes.flip)
            visible_width, visible_height = int(sizes.iwidth), int(sizes.iheight)
            crop_width, crop_height = int(sizes.crop_width), int(sizes.crop_height)
        swaps_axes = flip in (5, 6)
        visible_wh = ((visible_height, visible_width) if swaps_axes
                      else (visible_width, visible_height))
        crop_wh = ((crop_height, crop_width) if swaps_axes else (crop_width, crop_height))
        for expert in "ABCDE":
            tiff = next((item for item in sources if item.get("role") == "tiff16"
                         and item.get("photographer") == expert), None)
            if not tiff:
                continue
            image = tiff.get("image", {})
            width, height = image.get("width"), image.get("height")
            if not width or not height:
                continue
            rows.append({
                "image_id": record.get("image_id"), "expert": expert,
                "rawpy_flip": flip,
                "dng_visible_width": visible_wh[0], "dng_visible_height": visible_wh[1],
                "dng_crop_width": crop_wh[0], "dng_crop_height": crop_wh[1],
                "tiff_width": width, "tiff_height": height,
                "tiff_area_fraction_of_visible": (width * height) / (visible_wh[0] * visible_wh[1]),
                "tiff_area_fraction_of_default_crop": (width * height) / (crop_wh[0] * crop_wh[1]),
                "visible_aspect_error": abs(width / height - visible_wh[0] / visible_wh[1]),
                "default_crop_aspect_error": abs(width / height - crop_wh[0] / crop_wh[1]),
                "matches_default_crop_dimensions": width == crop_wh[0] and height == crop_wh[1],
                "matches_default_crop_dimensions_rotated": width == crop_wh[1] and height == crop_wh[0],
            })
    return rows


def collect_fit_rows(root, process_records):
    rows, complete_scenes = [], set()
    for record in process_records.values():
        if record.get("status") != "complete" or not isinstance(record.get("fits"), dict):
            continue
        scene_id = record["scene_id"]
        for expert in "ABCDE":
            expert_record = record["fits"].get(expert)
            if not expert_record:
                break
            files = expert_record.get("files", {})
            target_rel = Path(files.get("target_display_8bit", {}).get("path", ""))
            fit_rel = Path(files.get("display_8bit", {}).get("path", ""))
            if target_rel.is_absolute() or fit_rel.is_absolute() or ".." in target_rel.parts or ".." in fit_rel.parts:
                raise ValueError(f"Non-relative processed file path in {scene_id}/{expert}")
            metrics = image_psnr(root / target_rel, root / fit_rel)
            fit = expert_record["fit"]
            rows.append({"image_id": scene_id, "expert": expert, **metrics,
                         "optimization_loss": fit.get("loss"), "target_path": target_rel.as_posix(),
                         "fit_path": fit_rel.as_posix(),
                         **{f"{name}_physical": fit["physical_controls"][name] for name in CONTROL_NAMES},
                         **{f"{name}_normalized": fit["normalized_controls"][name] for name in CONTROL_NAMES}})
        else:
            complete_scenes.add(scene_id)
    return rows, complete_scenes


def save_metadata_plot(metadata, output):
    fields = (("location", "Location"), ("subject", "Subject"),
              ("lighting", "Lighting"), ("time_of_day", "Time of day"))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    summary = {}
    for axis, (key, title) in zip(axes.flat, fields):
        counts = {}
        for row in metadata:
            value = row.get(key) or "unknown"
            counts[value] = counts.get(value, 0) + 1
        summary[key] = counts
        labels, values = zip(*sorted(counts.items(), key=lambda item: (-item[1], item[0])))
        axis.barh(labels, values, color="#287c8e")
        axis.invert_yaxis()
        axis.set(title=f"{title} (n={len(metadata)})", xlabel="Scenes")
        for y, value in enumerate(values):
            axis.text(value, y, f" {value}", va="center", fontsize=9)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return summary


def save_dimension_plot(rows, output):
    area = np.asarray([row["tiff_area_fraction_of_default_crop"] for row in rows], dtype=np.float64)
    aspect_error = np.asarray([row["default_crop_aspect_error"] for row in rows], dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].hist(area, bins=40, color="#dd8452", edgecolor="white")
    axes[0].axvline(1, color="#333333", linestyle="--", linewidth=1)
    axes[0].set(title="Expert TIFF area / rawpy DNG default-crop area",
                xlabel="Area ratio", ylabel="Expert TIFFs")
    zoom_limit = .02
    axes[1].hist(aspect_error, bins=np.linspace(0, zoom_limit, 41), color="#4c9f70", edgecolor="white")
    outliers = int(np.sum(aspect_error > zoom_limit))
    axes[1].set(title=f"Crop aspect error ≤0.02 (n={len(aspect_error)-outliers}; {outliers} larger outliers)",
                xlabel="|TIFF ratio - rawpy crop ratio|", ylabel="Expert TIFFs")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_quality_plot(rows, output):
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for expert in "ABCDE":
        values = [row["psnr_db"] for row in rows if row["expert"] == expert
                  and math.isfinite(row["psnr_db"])]
        if values:
            axis.hist(values, bins=30, histtype="step", linewidth=1.7, label=f"Expert {expert}")
    axis.set(title=f"Native-dimension display fit PSNR (n={len(rows)} expert fits)",
             xlabel="PSNR (dB), 8-bit sRGB target vs fit", ylabel="Fits")
    axis.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def quality_summary(rows):
    result = {}
    for expert in "ABCDE":
        values = np.asarray([row["psnr_db"] for row in rows if row["expert"] == expert
                             and math.isfinite(row["psnr_db"])], dtype=np.float64)
        if values.size:
            result[expert] = {"n": int(values.size), "mean_db": float(values.mean()),
                "median_db": float(np.median(values)), "p10_db": float(np.quantile(values, .10)),
                "p90_db": float(np.quantile(values, .90)), "min_db": float(values.min()),
                "max_db": float(values.max())}
    return result


def dimension_summary(rows):
    total = len(rows)
    exact_crop = sum(row["matches_default_crop_dimensions"] for row in rows)
    rotated_crop = sum(row["matches_default_crop_dimensions_rotated"] for row in rows)
    area_close = sum(abs(row["tiff_area_fraction_of_default_crop"] - 1) <= .01 for row in rows)
    aspect_close = sum(row["default_crop_aspect_error"] <= .01 for row in rows)
    mismatch_scenes = sorted({row["image_id"] for row in rows if row["default_crop_aspect_error"] > .01})
    return {"expert_pairs": total, "exact_default_crop_dimensions": exact_crop,
        "exact_default_crop_dimensions_after_90deg": rotated_crop,
        "area_ratio_within_1_percent_of_default_crop": area_close,
        "aspect_ratio_within_1_percent_of_default_crop": aspect_close,
        "scenes_with_any_expert_crop_aspect_mismatch_over_1_percent": mismatch_scenes}


def save_control_plot(rows, bounds, output):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    summaries = []
    for axis, control, bound in zip(axes.flat, CONTROL_NAMES, bounds):
        edges = np.linspace(float(bound[0]), float(bound[1]), 31)
        for expert in "ABCDE":
            values = [row[f"{control}_physical"] for row in rows if row["expert"] == expert]
            if values:
                axis.hist(values, bins=edges, histtype="step", linewidth=1.5, label=expert)
        axis.set(title=f"{control} [{bound[0]}, {bound[1]}]", xlabel="Physical control", ylabel="Fits")
        if axis is axes.flat[0]:
            axis.legend(title="Expert")
        for expert in "ABCDE":
            values = np.asarray([row[f"{control}_physical"] for row in rows if row["expert"] == expert])
            if values.size:
                summaries.append({"expert": expert, "control": control, "n": int(values.size),
                    "median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
                    "q75": float(np.quantile(values, .75)), "mean": float(values.mean()),
                    "std": float(values.std())})
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return summaries


def save_default_vs_median(rows, root, config, output, preview_short_side=720):
    pooled_median = {control: float(np.median([row[f"{control}_physical"] for row in rows]))
                     for control in CONTROL_NAMES}
    scales = {control: max(float(np.quantile([r[f"{control}_physical"] for r in rows], .75)
                                 - np.quantile([r[f"{control}_physical"] for r in rows], .25)), 1e-6)
              for control in CONTROL_NAMES}
    by_scene = {}
    for row in rows:
        by_scene.setdefault(row["image_id"], []).append(row)
    distances = {scene: math.sqrt(sum(((float(np.median([fit[f"{c}_physical"] for fit in fits]))
                                       - pooled_median[c]) / scales[c]) ** 2 for c in CONTROL_NAMES))
                 for scene, fits in by_scene.items()}
    scene_id = min(distances, key=distances.get)
    baseline_path = root / "processed" / scene_id / "baseline_linear_16bit.png"
    bgr = cv2.imread(str(baseline_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype != np.uint16 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected independent uint16 linear baseline: {baseline_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535
    height, width = rgb.shape[:2]
    scale = preview_short_side / min(height, width)
    preview_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgb = cv2.resize(rgb, preview_size, interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    controls = torch.tensor([pooled_median[name] for name in CONTROL_NAMES], dtype=image.dtype)
    parameters = parameters_from_controls(controls, config)
    with torch.no_grad():
        median_render = render_linear(image, parameters)
        default_pixels = linear_to_srgb8(image)[0].permute(1, 2, 0).numpy()
        median_pixels = linear_to_srgb8(median_render)[0].permute(1, 2, 0).numpy()
    gap, header = 12, 54
    canvas = Image.new("RGB", (default_pixels.shape[1] * 2 + gap, default_pixels.shape[0] + header), "white")
    canvas.paste(Image.fromarray(default_pixels), (0, header))
    canvas.paste(Image.fromarray(median_pixels), (default_pixels.shape[1] + gap, header))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), f"V2 native-RAW baseline: {scene_id}", fill="#18303a", font=font)
    draw.text((default_pixels.shape[1] + gap + 8, 8), "Pooled median of all completed A-E fit controls",
              fill="#18303a", font=font)
    return pooled_median, {"scene_id": scene_id, "normalized_distance_to_pooled_median": distances[scene_id],
        "preview_dimensions_wh": [canvas.width, canvas.height - header],
        "source_baseline": baseline_path.relative_to(root).as_posix()}, canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-name", default="snapshot_current")
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    out = root / "analysis_results" / args.out_name
    out.mkdir(parents=True, exist_ok=False)
    page_metadata = read_jsonl(root / "metadata/fivek_page_metadata.jsonl")
    source_records = read_jsonl(root / "metadata/source_manifest.jsonl")
    process_records = latest_by_scene(read_jsonl(root / "metadata/process_manifest.jsonl"))
    complete = {scene: record for scene, record in process_records.items()
                if record.get("status") == "complete" and isinstance(record.get("fits"), dict)}
    fit_rows, complete_scenes = collect_fit_rows(root, complete)
    if not fit_rows:
        raise RuntimeError("No complete expert fits with valid target/render PNGs are available")
    dimensions = source_dimension_rows(root, source_records)
    metadata_summary = save_metadata_plot(page_metadata, out / "scene_metadata_categories.png")
    save_dimension_plot(dimensions, out / "dng_tiff_dimensions.png")
    save_quality_plot(fit_rows, out / "fit_psnr_by_expert.png")
    config = json.loads((V2_ROOT / "calibration/exp81/direct_soft_calibration.json").read_text())
    control_summary = save_control_plot(fit_rows, config["control_bounds"], out / "controls_by_expert_dof.png")
    write_csv(out / "fit_metrics.csv", fit_rows,
        ["image_id", "expert", "dimensions_wh", "psnr_db", "mae_8bit", "mse_8bit", "optimization_loss",
         *[f"{name}_physical" for name in CONTROL_NAMES], *[f"{name}_normalized" for name in CONTROL_NAMES],
         "target_path", "fit_path"])
    finite_rows = [row for row in fit_rows if math.isfinite(row["psnr_db"])]
    low = sorted(finite_rows, key=lambda row: row["psnr_db"])[:25]
    write_csv(out / "lowest_psnr_examples.csv", low,
        ["image_id", "expert", "psnr_db", "mae_8bit", "optimization_loss", "dimensions_wh",
         *[f"{name}_physical" for name in CONTROL_NAMES], "target_path", "fit_path"])
    write_csv(out / "source_dimensions.csv", dimensions, list(dimensions[0]) if dimensions else [])
    write_csv(out / "control_summary_by_expert.csv", control_summary,
        ["expert", "control", "n", "median", "q25", "q75", "mean", "std"])
    median_controls, preview_info, preview = save_default_vs_median(
        fit_rows, root, config, out / "default_vs_median_controls.png")
    preview.save(out / "default_vs_median_controls.png")
    psnr_all = np.asarray([row["psnr_db"] for row in fit_rows if math.isfinite(row["psnr_db"])])
    summary = {
        "analysis_snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "acquired_scenes": len(source_records), "webpage_metadata_scenes": len(page_metadata),
        "processed_complete_scenes_in_snapshot": len(complete_scenes), "fit_pairs_in_snapshot": len(fit_rows),
        "processing_errors_in_latest_ledger": sum(row.get("status") == "error" for row in process_records.values()),
        "fit_metric": "PSNR/MAE of saved native-dimension 8-bit sRGB expert target vs V2 fitted render.",
        "resolution_note": "Fit controls optimized at 256px short side; target and fit exports are native oriented dimensions.",
        "license_note": "MIT-Adobe research-only subset; rights review required before redistribution or commercial use.",
        "page_metadata_counts": metadata_summary,
        "dimension_summary": dimension_summary(dimensions),
        "psnr_summary_by_expert": quality_summary(fit_rows),
        "pooled_psnr_summary": {"n": int(psnr_all.size), "mean_db": float(psnr_all.mean()),
            "median_db": float(np.median(psnr_all)), "p10_db": float(np.quantile(psnr_all, .1)),
            "min_db": float(psnr_all.min()), "max_db": float(psnr_all.max())},
        "lowest_10_psnr_pairs": [{"image_id": row["image_id"], "expert": row["expert"],
            "psnr_db": row["psnr_db"], "mae_8bit": row["mae_8bit"]} for row in low[:10]],
        "median_physical_controls_pooled_over_completed_expert_fits": median_controls,
        "median_definition": "Independent median per physical DOF over completed A-E fits; diagnostic render, not refit.",
        "default_vs_median_preview": preview_info,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "README.md").write_text(
        "# Full-Resolution FiveK Analysis Snapshot\n\n"
        f"This snapshot covers {len(source_records)} downloaded scenes and {len(complete_scenes)} completed scenes "
        f"({len(fit_rows)} expert fits). PSNR/MAE compare native-dimension saved 8-bit sRGB expert targets and fits. "
        "The fit optimizer uses a 256px short-side image. The DNG/TIFF dimension audit compares visible raster and "
        "rawpy default crop after decoder orientation. The median-control view is diagnostic, not a refit. "
        "Source images remain in the sibling source tree; no links/copies are placed in this analysis folder.\n")
    links = ["summary.json", "README.md", "scene_metadata_categories.png", "dng_tiff_dimensions.png",
             "fit_psnr_by_expert.png", "controls_by_expert_dof.png", "default_vs_median_controls.png",
             "fit_metrics.csv", "lowest_psnr_examples.csv", "source_dimensions.csv", "control_summary_by_expert.csv"]
    page = ["<!doctype html><meta charset='utf-8'><title>FiveK FullRes Analysis</title>",
        f"<h1>FiveK FullRes analysis: {len(complete_scenes)} completed scenes, {len(fit_rows)} fits</h1>",
        "<p>Snapshot only. PSNR compares saved native-size display PNGs; fitting uses a 256px short-side objective. "
        "Median controls are independently pooled per DOF, not refit.</p>",
        "<ul>" + "".join(f"<li><a href='{name}'>{name}</a></li>" for name in links) + "</ul>",
        "<h2>Webpage scene metadata</h2><img style='max-width:100%' src='scene_metadata_categories.png'>",
        "<h2>DNG visible raster vs default crop and expert TIFF sizes</h2><img style='max-width:100%' src='dng_tiff_dimensions.png'>",
        "<h2>PSNR by expert</h2><img style='max-width:100%' src='fit_psnr_by_expert.png'>",
        "<h2>Per-DOF control histograms by expert</h2><img style='max-width:100%' src='controls_by_expert_dof.png'>",
        "<h2>Native V2 baseline vs pooled median controls</h2><img style='max-width:100%' src='default_vs_median_controls.png'>"]
    (out / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(json.dumps({"analysis_directory": str(out), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()