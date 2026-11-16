"""Moderate palette endpoints by averaging their saved one-DOF tuning samples."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from exp38_learn_poles import save_csv, save_json
from exp49_direct_isp_comparison import directional_mean_endpoint
from moran_semantic_isp import CONTROL_BOUNDS, CONTROL_NAMES, CONTROL_NEUTRAL
from render_pole_centroids import run as render_poles


def run(args):
    base_config_path = args.base / "direct_soft_calibration.json"
    config = json.loads(base_config_path.read_text())
    axes = list(config["axes"])
    if "palette" not in axes:
        raise ValueError("Base calibration has no palette axis")
    rows = np.asarray(config["learned_calibration"]["pole_raw_deltas"], dtype=float)
    original_rows = rows.copy()
    raw_neutral = np.asarray(config["learned_calibration"]["raw_neutral"], dtype=float)
    bounds = np.asarray(config.get("control_bounds",
        config["learned_calibration"].get("control_bounds", CONTROL_BOUNDS)), dtype=float)
    palette_index = axes.index("palette")
    control_index = CONTROL_NAMES.index("palette")
    neutral = float(CONTROL_NEUTRAL[control_index])
    low, high = bounds[control_index]
    half_range = (high - low) / 2
    midpoint = (high + low) / 2
    pole_rows = []
    source_files = list(args.base.glob("train_img_*/*.json"))
    source_files = [path for path in source_files if path.name.startswith("palette_")]
    if len(source_files) != 224:
        raise ValueError(f"Expected 224 Exp63 palette tuning records, found {len(source_files)}")

    for side, direction in (("negative", -1), ("positive", 1)):
        samples = []
        for record_path in source_files:
            record = json.loads(record_path.read_text())
            if record.get("axis") == "palette" and record.get("sign") == side:
                values = np.asarray(record["controls"], dtype=float)
                if values.shape != (len(CONTROL_NAMES),):
                    raise ValueError(f"Invalid palette sample shape: {record_path}")
                if any(values[i] != CONTROL_NEUTRAL[i] for i in range(len(values)) if i != control_index):
                    raise ValueError(f"Non-palette controls changed during palette tuning: {record_path}")
                samples.append(float(values[control_index]))
        if len(samples) != 112:
            raise ValueError(f"Expected 112 {side} palette samples, found {len(samples)}")
        endpoint = directional_mean_endpoint(samples, neutral, direction)
        normalized = np.clip((endpoint - midpoint) / half_range, -1 + 1e-7, 1 - 1e-7)
        raw_endpoint = float(np.arctanh(normalized))
        row_index = palette_index + (len(axes) if side == "negative" else 0)
        rows[row_index, control_index] = raw_endpoint - raw_neutral[control_index]
        pole_rows.append({"axis": "palette", "sign": side, "control": "palette",
            "sample_count": len(samples), "q90_endpoint": float(np.quantile(samples, .9)) if direction > 0 else None,
            "q10_endpoint": float(np.quantile(samples, .1)) if direction < 0 else None,
            "sample_mean": float(np.mean(samples)), "sample_median": float(np.median(samples)),
            "moderated_endpoint": endpoint,
            "endpoint_rule": "direction-aligned arithmetic mean of 112 saved one-DOF tuning settings"})

    if not 0 < args.scale <= 1:
        raise ValueError("Palette moderation scale must be in (0, 1]")
    for row_index, side in ((palette_index, "positive"), (palette_index + len(axes), "negative")):
        direction = 1 if side == "positive" else -1
        row = next(item for item in pole_rows if item["sign"] == side)
        endpoint = neutral + args.scale * (row["moderated_endpoint"] - neutral)
        normalized = np.clip((endpoint - midpoint) / half_range, -1 + 1e-7, 1 - 1e-7)
        rows[row_index, control_index] = float(np.arctanh(normalized)) - raw_neutral[control_index]
        row["moderation_scale"] = args.scale
        row["selected_endpoint"] = endpoint

    for index in range(len(rows)):
        if index not in (palette_index, palette_index + len(axes)):
            np.testing.assert_array_equal(rows[index], original_rows[index])
    config["learned_calibration"]["pole_raw_deltas"] = rows.tolist()
    config["learned_calibration"]["pole_statistic"] = {
        "method": "directional_mean_palette_moderation",
        "source": "Saved Exp63 one-DOF phrase-image optima; no new image optimization",
        "sample_count_per_pole": 112, "moderation_scale": args.scale,
        "prior": config.get("palette_tuning_objective"),
    }
    config["palette_moderation"] = {"base": str(args.base), "source_record_roots": sorted({str(p.parent.parent) for p in source_files}),
        "rule": "direction-aligned mean of saved physical palette settings, then moderation toward neutral",
        "scale": args.scale, "changed_coefficients": [palette_index, palette_index + len(axes)],
        "frozen_coefficients": [i for i in range(len(rows)) if i not in (palette_index, palette_index + len(axes))]}
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / "direct_soft_calibration.json", config)
    save_json(args.out / "calibrated_poles.json", config)
    save_json(args.out / "manifest.json", {"base_config_sha256": hashlib.sha256(base_config_path.read_bytes()).hexdigest(),
        "sample_count_per_pole": 112, "moderation_scale": args.scale,
        "source_records": [str(path) for path in source_files],
        "note": "Endpoints averaged from prior auto-tuning results; no new CLIP optimization was run."})
    save_csv(args.out / "palette_average_settings.csv", pole_rows)
    render_poles(args.out / "calibrated_poles.json", args.out / "pole_results", [2, 33, 384], args.device, isolated=True)
    print(json.dumps({"output": str(args.out), "settings": pole_rows,
        "gallery": str(args.out / "pole_results/index.html")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/data/data1/semantic_isp/results/exp63_nearest_swatch_palette"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.,
        help="Optional additional moderation of the directional mean toward neutral")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args)