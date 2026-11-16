"""Full analysis of 7-DOF tint fits: comparisons, statistics, PSNR trimming, median ISP."""

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
import torch
import torch.nn.functional as F

SCRIPT = Path(__file__).resolve()
V2_ROOT = SCRIPT.parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from axis_isp import linear_to_srgb8
from semantic_isp import (CONTROL_NAMES, apply_green_magenta_tint,
                                 parameters_from_controls, render_linear)

DEFAULT_ROOT = Path("/data/data1/FiveK/C_data/datasets/FiveK-FullRes-v1")
ALL_DOF_NAMES = (*CONTROL_NAMES, "tint")
DOF_BOUNDS = {
    "warmth": (-0.3, 0.3), "saturation": (-1.0, 1.0), "tone_gain": (0.05, 5.0),
    "contrast": (0.2, 3.0), "sharpness": (-0.9, 0.9), "palette": (-0.4, 0.4),
    "tint": (-0.3, 0.3),
}
DOF_NEUTRAL = {
    "warmth": 0.0, "saturation": 0.0, "tone_gain": 1.0,
    "contrast": 1.0, "sharpness": 0.0, "palette": 0.0, "tint": 0.0,
}
PSNR_TRIM_PERCENTILE = 10


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


def image_metrics(target_path, fit_path):
    target = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
    fit = cv2.imread(str(fit_path), cv2.IMREAD_UNCHANGED)
    if target is None or fit is None:
        raise FileNotFoundError(f"Missing image: {target_path} / {fit_path}")
    if target.dtype != np.uint8 or fit.dtype != np.uint8 or target.shape != fit.shape:
        raise ValueError(f"Shape/dtype mismatch: {target_path}, {fit_path}")
    delta = target.astype(np.float64) - fit.astype(np.float64)
    mse = float(np.mean(delta * delta))
    mae = float(np.mean(np.abs(delta)))
    psnr = float("inf") if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))
    from skimage.metrics import structural_similarity
    ssim = float(structural_similarity(target, fit, channel_axis=-1, data_range=255))
    return {"psnr_db": psnr, "ssim": ssim, "mae_8bit": mae, "mse_8bit": mse,
            "dimensions_wh": [int(target.shape[1]), int(target.shape[0])]}


def collect_fit_rows(root, process_records):
    rows, complete_scenes = [], set()
    for record in process_records.values():
        if record.get("status") != "complete" or not isinstance(record.get("fits"), dict):
            continue
        scene_id = record["scene_id"]
        native_wh = record.get("native_dimensions_wh", [None, None])
        scene_rows = []
        for expert in "ABCDE":
            expert_record = record["fits"].get(expert)
            if not expert_record:
                break
            files = expert_record.get("files", {})
            target_rel = Path(files.get("target_display_8bit", {}).get("path", ""))
            fit_rel = Path(files.get("display_8bit", {}).get("path", ""))
            if target_rel.is_absolute() or fit_rel.is_absolute():
                raise ValueError(f"Non-relative path in {scene_id}/{expert}")
            metrics = image_metrics(root / target_rel, root / fit_rel)
            fit = expert_record["fit"]
            alignment = expert_record.get("target_alignment", {})
            row = {
                "image_id": scene_id, "expert": expert,
                "native_width": native_wh[0], "native_height": native_wh[1],
                "crop_width": metrics["dimensions_wh"][0],
                "crop_height": metrics["dimensions_wh"][1],
                **metrics,
                "optimization_loss": fit.get("loss"),
                "best_iteration": fit.get("iteration"),
                "palette_seed": fit.get("palette_seed"),
                "target_path": target_rel.as_posix(),
                "fit_path": fit_rel.as_posix(),
                "alignment_status": alignment.get("status", "unknown"),
            }
            for name in CONTROL_NAMES:
                row[f"{name}_physical"] = fit["physical_controls"][name]
                row[f"{name}_normalized"] = fit["normalized_controls"][name]
            row["tint_physical"] = fit.get("tint", 0.0)
            row["tint_normalized"] = fit.get("normalized_tint", 0.0)
            scene_rows.append(row)
        if len(scene_rows) == 5:
            rows.extend(scene_rows)
            complete_scenes.add(scene_id)
    return rows, complete_scenes


def psnr_trim(rows, percentile=PSNR_TRIM_PERCENTILE):
    finite = [r for r in rows if math.isfinite(r["psnr_db"])]
    if not finite:
        return rows, float("nan"), 0
    threshold = float(np.percentile([r["psnr_db"] for r in finite], percentile))
    kept = [r for r in finite if r["psnr_db"] >= threshold]
    removed = len(finite) - len(kept)
    return kept, threshold, removed


def save_quality_histograms(rows, output, title_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, metric, label, unit in zip(
        axes, ("psnr_db", "ssim", "mae_8bit"),
        ("PSNR", "SSIM", "MAE"), ("dB", "", "/255"),
    ):
        for expert in "ABCDE":
            values = [r[metric] for r in rows if r["expert"] == expert and math.isfinite(r[metric])]
            if values:
                ax.hist(values, bins=30, histtype="step", linewidth=1.5, label=f"Expert {expert}")
        ax.set(title=f"{label}{title_suffix} (n={len(rows)} fits)",
               xlabel=f"{label} ({unit})" if unit else label, ylabel="Fits")
        if ax is axes[0]:
            ax.legend(title="Expert", fontsize=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_dof_histograms(rows, output, title_suffix=""):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    for ax_idx, (ax, dof) in enumerate(zip(axes.flat, ALL_DOF_NAMES)):
        lo, hi = DOF_BOUNDS[dof]
        edges = np.linspace(lo, hi, 41)
        for expert in "ABCDE":
            values = [r[f"{dof}_physical"] for r in rows if r["expert"] == expert]
            if values:
                ax.hist(values, bins=edges, histtype="step", linewidth=1.3, label=expert)
        ax.axvline(DOF_NEUTRAL[dof], color="#333", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set(title=f"{dof} [{lo}, {hi}]{title_suffix}", xlabel="Physical value", ylabel="Fits")
        if ax_idx == 0:
            ax.legend(title="Expert", fontsize=7)
    if len(ALL_DOF_NAMES) < len(axes.flat):
        axes.flat[-1].set_visible(False)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def dof_summary_table(rows):
    summaries = []
    for dof in ALL_DOF_NAMES:
        for expert in [*"ABCDE", "all"]:
            if expert == "all":
                values = np.asarray([r[f"{dof}_physical"] for r in rows], dtype=np.float64)
            else:
                values = np.asarray([r[f"{dof}_physical"] for r in rows if r["expert"] == expert],
                                    dtype=np.float64)
            if values.size == 0:
                continue
            summaries.append({
                "dof": dof, "expert": expert, "n": int(values.size),
                "mean": float(values.mean()), "std": float(values.std()),
                "median": float(np.median(values)),
                "q10": float(np.quantile(values, 0.10)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "q90": float(np.quantile(values, 0.90)),
                "min": float(values.min()), "max": float(values.max()),
                "neutral": DOF_NEUTRAL[dof],
                "deviation_from_neutral": float(np.abs(values - DOF_NEUTRAL[dof]).mean()),
            })
    return summaries


def quality_summary(rows):
    result = {}
    for metric in ("psnr_db", "ssim", "mae_8bit"):
        for expert in [*"ABCDE", "all"]:
            if expert == "all":
                values = np.asarray([r[metric] for r in rows if math.isfinite(r[metric])],
                                    dtype=np.float64)
            else:
                values = np.asarray([r[metric] for r in rows if r["expert"] == expert
                                     and math.isfinite(r[metric])], dtype=np.float64)
            if values.size == 0:
                continue
            key = f"{metric}_{expert}"
            result[key] = {"n": int(values.size), "mean": float(values.mean()),
                           "median": float(np.median(values)),
                           "std": float(values.std()),
                           "p10": float(np.quantile(values, 0.1)),
                           "p90": float(np.quantile(values, 0.9)),
                           "min": float(values.min()), "max": float(values.max())}
    return result


def compute_median_default_isp(rows):
    medians = {}
    for dof in ALL_DOF_NAMES:
        values = [r[f"{dof}_physical"] for r in rows]
        medians[dof] = float(np.median(values))
    per_expert = {}
    for expert in "ABCDE":
        expert_medians = {}
        for dof in ALL_DOF_NAMES:
            values = [r[f"{dof}_physical"] for r in rows if r["expert"] == expert]
            expert_medians[dof] = float(np.median(values)) if values else DOF_NEUTRAL[dof]
        per_expert[expert] = expert_medians
    return medians, per_expert


def render_median_preview(median_controls, root, rows, config, output, short_side=720):
    by_scene = {}
    for row in rows:
        by_scene.setdefault(row["image_id"], []).append(row)
    iqr_scales = {}
    for dof in ALL_DOF_NAMES:
        vals = [r[f"{dof}_physical"] for r in rows]
        iqr_scales[dof] = max(float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25)), 1e-6)
    distances = {}
    for scene, fits in by_scene.items():
        scene_medians = {dof: float(np.median([f[f"{dof}_physical"] for f in fits])) for dof in ALL_DOF_NAMES}
        distances[scene] = math.sqrt(sum(((scene_medians[d] - median_controls[d]) / iqr_scales[d]) ** 2
                                          for d in ALL_DOF_NAMES))
    scene_id = min(distances, key=distances.get)
    baseline_path = root / f"processed_tint_v1/{scene_id}/baseline_linear_16bit.png"
    if not baseline_path.is_file():
        for alt_dir in ("processed_crop_only_v1", "processed"):
            alt = root / f"{alt_dir}/{scene_id}/baseline_linear_16bit.png"
            if alt.is_file():
                baseline_path = alt
                break
    bgr = cv2.imread(str(baseline_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype != np.uint16:
        raise ValueError(f"Cannot load baseline: {baseline_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535
    h, w = rgb.shape[:2]
    scale = short_side / min(h, w)
    preview_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    rgb = cv2.resize(rgb, preview_size, interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    controls_6 = torch.tensor([median_controls[name] for name in CONTROL_NAMES], dtype=image.dtype)
    parameters = parameters_from_controls(controls_6, config)
    parameters["tint"] = image.new_tensor(median_controls["tint"])
    with torch.no_grad():
        median_render = render_linear(image, parameters)
        default_pixels = linear_to_srgb8(image)[0].permute(1, 2, 0).numpy()
        median_pixels = linear_to_srgb8(median_render)[0].permute(1, 2, 0).numpy()

    gap, header = 12, 60
    canvas = Image.new("RGB", (default_pixels.shape[1] * 2 + gap, default_pixels.shape[0] + header), "white")
    canvas.paste(Image.fromarray(default_pixels), (0, header))
    canvas.paste(Image.fromarray(median_pixels), (default_pixels.shape[1] + gap, header))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), f"Default ISP baseline: {scene_id}", fill="#18303a", font=font)
    ctrl_text = " | ".join(f"{d}={median_controls[d]:.3f}" for d in ALL_DOF_NAMES)
    draw.text((default_pixels.shape[1] + gap + 8, 6),
              "Median of all expert fits (7 DOF)", fill="#18303a", font=font)
    draw.text((default_pixels.shape[1] + gap + 8, 28), ctrl_text, fill="#555555", font=font)
    canvas.save(output)
    return median_controls, {"scene_id": scene_id, "distance": distances[scene_id]}


def save_comparison_sheet(rows, root, output_dir, max_scenes=50):
    sheet_dir = output_dir / "comparison_sheets"
    sheet_dir.mkdir(exist_ok=True)
    by_scene = {}
    for row in rows:
        by_scene.setdefault(row["image_id"], []).append(row)
    scenes = sorted(by_scene.keys())[:max_scenes]
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_title = font
    generated = []
    for scene_id in scenes:
        scene_rows = sorted(by_scene[scene_id], key=lambda r: r["expert"])
        thumb_w, thumb_h = 480, 320
        panel_h = thumb_h + 80
        canvas_w = thumb_w * 2 + 20
        canvas_h = 40 + len(scene_rows) * panel_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f"Scene: {scene_id}", fill="#18303a", font=font_title)
        for idx, row in enumerate(scene_rows):
            y_base = 40 + idx * panel_h
            target_path = root / row["target_path"]
            fit_path = root / row["fit_path"]
            for x_off, img_path, label in [
                (0, target_path, f"Expert {row['expert']} target"),
                (thumb_w + 20, fit_path, f"7-DOF fit"),
            ]:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
                canvas.paste(img, (x_off, y_base))
                draw.text((x_off + 4, y_base + thumb_h + 2), label, fill="#333", font=font)
            info_parts = [
                f"PSNR={row['psnr_db']:.1f}dB  SSIM={row['ssim']:.3f}  MAE={row['mae_8bit']:.1f}",
            ]
            ctrl_parts = []
            for dof in ALL_DOF_NAMES:
                val = row[f"{dof}_physical"]
                neutral = DOF_NEUTRAL[dof]
                ctrl_parts.append(f"{dof}={val:.3f}")
            info_parts.append("  ".join(ctrl_parts))
            for line_idx, text in enumerate(info_parts):
                draw.text((thumb_w + 24, y_base + thumb_h + 16 + line_idx * 16), text,
                          fill="#555", font=font)
        sheet_path = sheet_dir / f"{scene_id}.jpg"
        canvas.save(sheet_path, quality=90)
        generated.append(scene_id)
    index_path = sheet_dir / "index.html"
    html = ["<!doctype html><meta charset='utf-8'><title>Expert vs Fit Comparisons</title>",
            f"<h1>{len(generated)} scene comparison sheets</h1><ul>"]
    for sid in generated:
        html.append(f"<li><a href='{sid}.jpg'>{sid}</a></li>")
    html.append("</ul>")
    index_path.write_text("\n".join(html), encoding="utf-8")
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest-name", default="process_tint_v1*manifest.jsonl")
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--psnr-trim-percentile", type=float, default=PSNR_TRIM_PERCENTILE)
    parser.add_argument("--comparison-scenes", type=int, default=50)
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    manifest_paths = sorted(root.glob(f"metadata/{args.manifest_name}"))
    if not manifest_paths:
        manifest_paths = sorted(root.glob("metadata/process_tint_v1*manifest.jsonl"))
    if not manifest_paths:
        raise FileNotFoundError(f"No manifests matching {args.manifest_name}")
    all_rows = []
    for mp in manifest_paths:
        all_rows.extend(read_jsonl(mp))
        print(f"  Loaded {mp.name}")
    records = latest_by_scene(all_rows)
    complete = {s: r for s, r in records.items()
                if r.get("status") == "complete" and isinstance(r.get("fits"), dict)}
    errors = {s: r for s, r in records.items() if r.get("status") == "error"}
    print(f"Manifest: {len(records)} records, {len(complete)} complete, {len(errors)} errors")
    if not complete:
        raise RuntimeError("No complete records in manifest")
    fit_rows, complete_scenes = collect_fit_rows(root, complete)
    print(f"Collected {len(fit_rows)} fit rows from {len(complete_scenes)} complete scenes")
    if not fit_rows:
        raise RuntimeError("No fit rows collected")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_name = args.out_name or f"tint_analysis_{timestamp}"
    out = root / "analysis_results" / out_name
    out.mkdir(parents=True, exist_ok=False)
    print(f"Output directory: {out}")

    # 1. Full quality metrics CSV
    csv_fields = ["image_id", "expert", "native_width", "native_height",
                  "crop_width", "crop_height", "psnr_db", "ssim", "mae_8bit", "mse_8bit",
                  "optimization_loss", "best_iteration", "palette_seed", "alignment_status",
                  *[f"{d}_physical" for d in ALL_DOF_NAMES],
                  *[f"{d}_normalized" for d in ALL_DOF_NAMES],
                  "target_path", "fit_path"]
    write_csv(out / "all_fit_metrics.csv", fit_rows, csv_fields)
    print(f"  Wrote all_fit_metrics.csv ({len(fit_rows)} rows)")

    # 2. Quality histograms (all data)
    save_quality_histograms(fit_rows, out / "quality_histograms_all.png", " (all)")
    print(f"  Wrote quality_histograms_all.png")

    # 3. DOF histograms (all data)
    save_dof_histograms(fit_rows, out / "dof_histograms_all.png", " (all)")
    print(f"  Wrote dof_histograms_all.png")

    # 4. PSNR trimming
    trimmed_rows, threshold, removed = psnr_trim(fit_rows, args.psnr_trim_percentile)
    print(f"  PSNR trim: threshold={threshold:.2f} dB, removed {removed} fits, kept {len(trimmed_rows)}")
    write_csv(out / "trimmed_fit_metrics.csv", trimmed_rows, csv_fields)

    # 5. Quality histograms (trimmed)
    save_quality_histograms(trimmed_rows, out / "quality_histograms_trimmed.png",
                            f" (PSNR>={threshold:.1f}dB)")
    print(f"  Wrote quality_histograms_trimmed.png")

    # 6. DOF histograms (trimmed)
    save_dof_histograms(trimmed_rows, out / "dof_histograms_trimmed.png",
                        f" (PSNR>={threshold:.1f}dB)")
    print(f"  Wrote dof_histograms_trimmed.png")

    # 7. Lowest PSNR examples
    finite_rows = sorted([r for r in fit_rows if math.isfinite(r["psnr_db"])],
                         key=lambda r: r["psnr_db"])
    write_csv(out / "lowest_psnr_25.csv", finite_rows[:25],
              ["image_id", "expert", "psnr_db", "ssim", "mae_8bit", "optimization_loss",
               "alignment_status", *[f"{d}_physical" for d in ALL_DOF_NAMES]])
    print(f"  Wrote lowest_psnr_25.csv")

    # 8. DOF summary statistics
    all_dof_summary = dof_summary_table(fit_rows)
    write_csv(out / "dof_summary_all.csv", all_dof_summary,
              ["dof", "expert", "n", "mean", "std", "median", "q10", "q25", "q75", "q90",
               "min", "max", "neutral", "deviation_from_neutral"])
    trimmed_dof_summary = dof_summary_table(trimmed_rows)
    write_csv(out / "dof_summary_trimmed.csv", trimmed_dof_summary,
              ["dof", "expert", "n", "mean", "std", "median", "q10", "q25", "q75", "q90",
               "min", "max", "neutral", "deviation_from_neutral"])
    print(f"  Wrote dof_summary_all.csv and dof_summary_trimmed.csv")

    # 9. Quality summary
    quality_all = quality_summary(fit_rows)
    quality_trimmed = quality_summary(trimmed_rows)

    # 10. Median default ISP (all data and trimmed)
    median_all, median_per_expert = compute_median_default_isp(fit_rows)
    median_trimmed, median_trimmed_per_expert = compute_median_default_isp(trimmed_rows)
    config = json.loads((V2_ROOT / "calibration/exp81/direct_soft_calibration.json").read_text())
    try:
        _, preview_info = render_median_preview(
            median_trimmed, root, trimmed_rows, config,
            out / "median_default_isp_preview.png")
        print(f"  Wrote median_default_isp_preview.png (scene: {preview_info['scene_id']})")
    except Exception as exc:
        preview_info = {"error": str(exc)}
        print(f"  WARNING: Could not render median preview: {exc}")

    # 11. Comparison sheets
    try:
        generated = save_comparison_sheet(fit_rows, root, out, args.comparison_scenes)
        print(f"  Wrote {len(generated)} comparison sheets in comparison_sheets/")
    except Exception as exc:
        generated = []
        print(f"  WARNING: Could not create comparison sheets: {exc}")

    # 12. Summary JSON
    summary = {
        "analysis_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest_name,
        "total_manifest_records": len(records),
        "complete_scenes": len(complete_scenes),
        "error_scenes": len(errors),
        "total_fit_pairs": len(fit_rows),
        "fitted_dofs": list(ALL_DOF_NAMES),
        "dof_bounds": DOF_BOUNDS,
        "dof_neutral": DOF_NEUTRAL,
        "quality_summary_all": quality_all,
        "quality_summary_trimmed": quality_trimmed,
        "psnr_trim": {
            "percentile": args.psnr_trim_percentile,
            "threshold_db": threshold,
            "removed_pairs": removed,
            "kept_pairs": len(trimmed_rows),
        },
        "median_default_isp_all": median_all,
        "median_default_isp_trimmed": median_trimmed,
        "median_default_isp_per_expert": median_per_expert,
        "median_default_isp_per_expert_trimmed": median_trimmed_per_expert,
        "median_preview": preview_info,
        "comparison_sheets_generated": len(generated),
        "error_scenes_list": [{"scene_id": s, "error": r.get("error")}
                              for s, r in errors.items()],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"  Wrote summary.json")

    # 13. HTML index
    page = [
        "<!doctype html><meta charset='utf-8'><title>7-DOF Tint Fit Analysis</title>",
        "<style>body{font-family:sans-serif;max-width:1400px;margin:auto;padding:20px}"
        "img{max-width:100%;border:1px solid #ddd;margin:10px 0}"
        "table{border-collapse:collapse;margin:10px 0}td,th{padding:4px 8px;border:1px solid #ccc}"
        ".stat{color:#555}</style>",
        f"<h1>7-DOF Tint Fit Analysis: {len(complete_scenes)} scenes, {len(fit_rows)} fits</h1>",
        f"<p>Manifest: <code>{args.manifest_name}</code> | "
        f"Errors: {len(errors)} | Trim: bottom {args.psnr_trim_percentile}% PSNR "
        f"(&lt;{threshold:.1f} dB, {removed} removed)</p>",
        "<h2>Quality metrics (all)</h2><img src='quality_histograms_all.png'>",
        "<h2>Quality metrics (PSNR-trimmed)</h2><img src='quality_histograms_trimmed.png'>",
        "<h2>Per-DOF parameter histograms (all)</h2><img src='dof_histograms_all.png'>",
        "<h2>Per-DOF parameter histograms (PSNR-trimmed)</h2><img src='dof_histograms_trimmed.png'>",
        "<h2>Median default ISP (trimmed)</h2>",
        "<table><tr><th>DOF</th><th>Neutral</th><th>Median (all experts)</th></tr>",
    ]
    for dof in ALL_DOF_NAMES:
        page.append(f"<tr><td>{dof}</td><td>{DOF_NEUTRAL[dof]:.3f}</td>"
                    f"<td>{median_trimmed[dof]:.4f}</td></tr>")
    page.append("</table>")
    page.append("<img src='median_default_isp_preview.png'>")
    page.append("<h2>Per-expert median controls (trimmed)</h2><table>")
    page.append("<tr><th>Expert</th>" + "".join(f"<th>{d}</th>" for d in ALL_DOF_NAMES) + "</tr>")
    for expert in "ABCDE":
        vals = median_trimmed_per_expert.get(expert, {})
        page.append("<tr><td>" + expert + "</td>" +
                    "".join(f"<td>{vals.get(d, 0):.4f}</td>" for d in ALL_DOF_NAMES) + "</tr>")
    page.append("</table>")
    if generated:
        page.append(f"<h2>Comparison sheets ({len(generated)} scenes)</h2>")
        page.append(f"<p><a href='comparison_sheets/index.html'>Browse all comparisons</a></p>")
    page.append("<h2>Data files</h2><ul>")
    for name in ["summary.json", "all_fit_metrics.csv", "trimmed_fit_metrics.csv",
                 "lowest_psnr_25.csv", "dof_summary_all.csv", "dof_summary_trimmed.csv"]:
        page.append(f"<li><a href='{name}'>{name}</a></li>")
    page.append("</ul>")
    (out / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(f"  Wrote index.html")
    print(f"\nDone. Analysis directory: {out}")
    print(f"Open: file://{out}/index.html")


if __name__ == "__main__":
    main()
