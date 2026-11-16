"""Freshly retune stronger CLIP warmth, brightness and positive-saturation poles."""

import argparse
import csv
import html
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from exp38_learn_poles import save_json
from exp55_large_kernel_detail_retune import run as retune_axis


STAGES = (
    ("01_warmth", "warmth", "both"),
    ("02_brightness", "brightness", "both"),
    ("03_positive_saturation", "saturation", "positive"),
)
TARGET_POLES = {
    ("warmth", "negative"), ("warmth", "positive"),
    ("brightness", "negative"), ("brightness", "positive"),
    ("saturation", "positive"),
}


def run(args):
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    base_config = json.loads((args.base / "direct_soft_calibration.json").read_text())
    if base_config["model"] != "ViT-B-32" or base_config["pretrained"] != "openai":
        raise ValueError("The active strength retuner is CLIP-only; expected OpenAI ViT-B/32 config")
    if base_config.get("color_protection", {}).get("operations") != ["palette"]:
        raise ValueError("Base config must protect palette only")
    args.out.mkdir(parents=True)
    current_base = args.base
    for stage_name, axis, side in STAGES:
        output = args.out / stage_name
        options = SimpleNamespace(
            base=current_base, out=output, iters=args.iters, sigma=8., side=side, axis=axis,
            contrast_range=None, sharpen_sigma=None,
            palette_strength_weight=args.palette_strength_weight,
            palette_strength_target=args.palette_strength_target,
            palette_positive_mix=.85, palette_negative_mix=.75,
            palette_softness_ratio=.18, sharpen_gain=None, sharpen_limit=None,
            blur_color_mode="preserve", device=args.device, smoke=args.smoke,
            color_protection_mode=None, color_protection_operations=None,
            skip_gallery=stage_name != STAGES[-1][0], quiet=True,
        )
        with (args.out / f"{stage_name}.log").open("w") as log:
            with redirect_stdout(log):
                retune_axis(options)
        current_base = output
        print(f"completed {stage_name}", flush=True)

    final_config_path = current_base / "direct_soft_calibration.json"
    config = json.loads(final_config_path.read_text())
    if config["color_protection"].get("operations") != ["palette"]:
        raise ValueError("Palette-only protection was not preserved")
    config["descriptor_table"]["descriptor_revision"] = "moran_style_v8_extreme_warmth_brightness_vivid_4x4"
    config["descriptor_table"].setdefault("revisions_by_axis", {})
    config["descriptor_table"]["revisions_by_axis"].update({
        "warmth": "moran_style_v8_extreme_warmth_brightness_vivid_4x4",
        "brightness": "moran_style_v8_extreme_warmth_brightness_vivid_4x4",
        "saturation": "moran_style_v8_extreme_warmth_brightness_vivid_4x4",
    })
    config["status"] = "CLIP-only calibration; v8 stronger warmth/brightness/positive-saturation descriptors; palette-only protection"
    config["strength_retuning"] = {
        "model": "OpenAI CLIP ViT-B/32",
        "descriptor_revision": "moran_style_v8_extreme_warmth_brightness_vivid_4x4",
        "freshly_retuned_poles": [list(key) for key in sorted(TARGET_POLES)],
        "unprotected_color_operations": ["warmth", "saturation"],
        "protected_operations": ["palette"],
        "endpoint_rule": "direction-aligned q90; positive numerical q90, negative numerical q10",
    }
    save_json(final_config_path, config)
    save_json(current_base / "calibrated_poles.json", config)

    statistics = list(csv.DictReader((current_base / "pole_quantile_statistics.csv").open()))
    endpoint_rows = [row for row in statistics if (row["axis"], row["sign"]) in TARGET_POLES]
    with (args.out / "strength_endpoint_statistics.csv").open("w", newline="") as stream:
        fields = ["axis", "sign", "control", "sample_count", "median", "q10", "q90", "selected_setting"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in endpoint_rows)

    summary = {
        "base_calibration": str(args.base),
        "final_calibration": str(final_config_path),
        "final_gallery": str(current_base / "pole_results/index.html"),
        "descriptor_revision": config["descriptor_table"]["descriptor_revision"],
        "protection_policy": config["color_protection"],
        "freshly_retuned_poles": [list(key) for key in sorted(TARGET_POLES)],
        "fresh_runs": 80 if args.smoke else 560,
        "smoke": args.smoke,
        "palette_endpoint_inherited_from": str(args.base),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    document = ["<!doctype html><meta charset='utf-8'><title>CLIP stronger direct poles</title>",
        "<h1>CLIP v8 stronger warmth, brightness, and vivid poles</h1>",
        "<p>Freshly retuned: warmth both directions, brightness both directions, and positive saturation. "
        "Palette remains at the adopted palette-only-protected Exp71 endpoint.</p>",
        "<p><a href='strength_endpoint_statistics.csv'>Selected q90 endpoints</a> | "
        "<a href='summary.json'>Run summary</a> | "
        f"<a href='{html.escape(str(Path(summary['final_gallery']).relative_to(args.out)))}'>Final pole gallery</a></p>"]
    for image_id in (2, 33, 384):
        sheet = current_base / "pole_results" / f"img_{image_id}" / "pole_centroids.jpg"
        document.append(f"<h2>Scene {image_id}</h2><a href='{html.escape(str(sheet.relative_to(args.out)))}'>"
                        f"<img style='max-width:100%' src='{html.escape(str(sheet.relative_to(args.out)))}'></a>")
    (args.out / "index.html").write_text("\n".join(document))
    print(json.dumps({"output": str(args.out), "fresh_runs": summary["fresh_runs"],
                      "retuned_poles": len(endpoint_rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
        default=Path("/data/data1/semantic_isp/results/exp71_palette_only_protection/clip/03_palette_protected"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--palette-strength-weight", type=float, default=.6)
    parser.add_argument("--palette-strength-target", type=float, default=.17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
