"""Finalize CLIP/SigLIP poles with gray/skin protection scoped to palette only."""

import argparse
import csv
import html
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp38_learn_poles import save_json
from exp55_large_kernel_detail_retune import run as retune_axis


MODEL_KEYS = ("clip", "siglip")
STAGES = (
    ("01_warmth_unprotected", "warmth", "both", "none", None),
    ("02_saturation_positive_unprotected", "saturation", "positive", "none", None),
    ("03_palette_protected", "palette", "both", "gray_white_and_skin", ["palette"]),
)


def run(args):
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    models = {}
    endpoint_rows = []
    for model_key in MODEL_KEYS:
        base = args.comparison_root / model_key / "axis_06_palette"
        model_root = args.out / model_key
        model_root.mkdir()
        for stage_name, axis, side, protection_mode, protection_operations in STAGES:
            output = model_root / stage_name
            options = SimpleNamespace(
                base=base, out=output, iters=args.iters, sigma=8., side=side, axis=axis,
                contrast_range=None, sharpen_sigma=None,
                palette_strength_weight=args.palette_strength_weight,
                palette_strength_target=args.palette_strength_target,
                palette_positive_mix=.85, palette_negative_mix=.75,
                palette_softness_ratio=.18, sharpen_gain=None, sharpen_limit=None,
                blur_color_mode="preserve", device=args.device, smoke=args.smoke,
                color_protection_mode=protection_mode,
                color_protection_operations=protection_operations,
                skip_gallery=stage_name != "03_palette", quiet=True,
            )
            with (model_root / f"{stage_name}.log").open("w") as log:
                with redirect_stdout(log):
                    retune_axis(options)
            base = output
            print(f"completed {model_key}/{stage_name}", flush=True)
        stats = list(csv.DictReader((base / "pole_quantile_statistics.csv").open()))
        for row in stats:
            if row["axis"] in {"warmth", "saturation", "palette"}:
                endpoint_rows.append({"model": model_key, **row})
        final_config = base / "direct_soft_calibration.json"
        config = json.loads(final_config.read_text())
        assert config["color_protection"]["operations"] == ["palette"]
        config["status"] = f"{model_key} calibration; combined gray/skin protection applies only to palette"
        config["color_protection_policy"] = {
            "protected_operations": ["palette"],
            "unprotected_color_operations": ["warmth", "saturation"],
            "palette_modes": ["gray_white_and_skin"],
            "warmth_and_positive_saturation": "freshly autotuned without chromaticity protection",
        }
        final_config_path = base / "direct_soft_calibration.json"
        save_json(final_config_path, config)
        save_json(base / "calibrated_poles.json", config)
        models[model_key] = {
            "config": str(final_config_path),
            "gallery": str(base / "pole_results/index.html"),
            "root": str(base),
            "fresh_optimization_runs": 80 if args.smoke else 560,
        }

    fields = ["model", "axis", "sign", "control", "sample_count", "median", "q10", "q90", "selected_setting"]
    with (args.out / "color_endpoint_statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in endpoint_rows)
    summary = {
        "base_comparison_root": str(args.comparison_root),
        "models": models,
        "stages": [{"name": name, "axis": axis, "side": side,
                    "protection_mode": mode, "protected_operations": operations}
                   for name, axis, side, mode, operations in STAGES],
        "protection_policy": "Gray/white and skin protection applies only to palette; warmth and positive saturation are retuned without protection.",
        "fresh_runs_per_model": 80 if args.smoke else 560,
        "total_fresh_runs": (80 if args.smoke else 560) * len(MODEL_KEYS),
        "smoke": args.smoke,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    document = ["<!doctype html><meta charset='utf-8'><title>Palette-only protection calibration</title>",
        "<h1>Final palette-only gray/skin protection configs</h1>",
        "<p>Warmth and positive saturation are freshly retuned unprotected. Gray/white and skin "
        "chromaticity masks apply only to both palette poles.</p>",
        "<p><a href='color_endpoint_statistics.csv'>Selected q90 endpoints</a> | "
        "<a href='summary.json'>Run summary</a></p>"]
    for model, result in models.items():
        document.append(f"<h2>{html.escape(model.upper())}</h2><p>"
                        f"<a href='{html.escape(str(Path(result['gallery']).relative_to(args.out)))}'>"
                        "Open final all-pole gallery</a></p>")
    (args.out / "index.html").write_text("\n".join(document))
    print(json.dumps({"output": str(args.out), "fresh_runs": summary["total_fresh_runs"],
                      "endpoint_rows": len(endpoint_rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path,
        default=Path("/data/data1/semantic_isp/results/exp70_clip_vs_siglip_all_axes"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--palette-strength-weight", type=float, default=.6)
    parser.add_argument("--palette-strength-target", type=float, default=.17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
