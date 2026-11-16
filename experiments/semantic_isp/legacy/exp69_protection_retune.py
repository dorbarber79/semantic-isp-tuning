"""Freshly autotune color-control poles for each gray/skin protection mode."""

import argparse
import csv
import html
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp55_large_kernel_detail_retune import run as retune


PROTECTION_MODES = ("none", "gray_white", "skin", "gray_white_and_skin")
RETUNING_STAGES = (
    ("01_warmth", "warmth", "both"),
    ("02_saturation_positive", "saturation", "positive"),
    ("03_palette", "palette", "both"),
)


def run(args):
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    results = {}
    endpoint_rows = []
    for mode in PROTECTION_MODES:
        mode_root = args.out / mode
        mode_root.mkdir()
        base = args.base
        final_output = None
        for stage_name, axis, side in RETUNING_STAGES:
            output = mode_root / stage_name
            options = SimpleNamespace(
                base=base, out=output, iters=args.iters, sigma=8., side=side, axis=axis,
                contrast_range=None, sharpen_sigma=None,
                palette_strength_weight=args.palette_strength_weight,
                palette_strength_target=args.palette_strength_target,
                palette_positive_mix=.85, palette_negative_mix=.75,
                palette_softness_ratio=.18, sharpen_gain=None, sharpen_limit=None,
                blur_color_mode="preserve", device=args.device, smoke=args.smoke,
                color_protection_mode=mode, skip_gallery=stage_name != "03_palette",
                quiet=True,
            )
            mode_root.mkdir(exist_ok=True)
            log_path = mode_root / f"{stage_name}.log"
            with log_path.open("w") as log:
                with redirect_stdout(log):
                    retune(options)
            base = output
            final_output = output
            print(f"completed {mode}/{stage_name}", flush=True)

        stats_path = final_output / "pole_quantile_statistics.csv"
        with stats_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if (row["axis"], row["sign"]) in {
                    ("warmth", "negative"), ("warmth", "positive"),
                    ("saturation", "positive"), ("palette", "negative"),
                    ("palette", "positive"),
                }:
                    endpoint_rows.append({"protection_mode": mode, **row})
        results[mode] = {
            "final_calibration": str(final_output / "direct_soft_calibration.json"),
            "gallery": str(final_output / "pole_results/index.html"),
            "fresh_optimization_runs": 80 if args.smoke else 560,
            "retuned_stages": [axis for _, axis, _ in RETUNING_STAGES],
        }

    fields = ["protection_mode", "axis", "sign", "control", "sample_count",
              "median", "q10", "q90", "selected_setting"]
    with (args.out / "color_endpoint_statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in endpoint_rows)
    summary = {
        "base_calibration": str(args.base),
        "protection_modes": list(PROTECTION_MODES),
        "stages": [{"name": name, "axis": axis, "side": side}
                   for name, axis, side in RETUNING_STAGES],
        "palette_strength_weight": args.palette_strength_weight,
        "palette_strength_target": args.palette_strength_target,
        "smoke": args.smoke,
        "fresh_runs_per_mode": 80 if args.smoke else 560,
        "total_fresh_runs": (80 if args.smoke else 560) * len(PROTECTION_MODES),
        "inherited": "Unselected physical poles are inherited at each stage; negative saturation is unchanged because protection is positive-only.",
        "results": results,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    document = ["<!doctype html><meta charset='utf-8'><title>Protection-aware autotuning</title>",
                "<h1>Protection-aware color-pole autotuning</h1>",
                "<p>Each mode has fresh warmth, positive-saturation, and two-direction palette "
                "autotuning, followed by its isolated pole gallery.</p>",
                "<p><a href='color_endpoint_statistics.csv'>Selected q90 endpoints</a> | "
                "<a href='summary.json'>Run summary</a></p>"]
    for mode, value in results.items():
        document.append(f"<h2>{html.escape(mode)}</h2><p>"
                        f"<a href='{html.escape(value['gallery'])}'>Open calibrated pole gallery</a></p>")
    (args.out / "index.html").write_text("\n".join(document))
    print(json.dumps({"output": str(args.out), "fresh_runs": summary["total_fresh_runs"],
                      "endpoint_rows": len(endpoint_rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
        default=Path("/data/data1/semantic_isp/results/exp67_moderate_palette_q90_pm020"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--palette-strength-weight", type=float, default=.6)
    parser.add_argument("--palette-strength-target", type=float, default=.17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
