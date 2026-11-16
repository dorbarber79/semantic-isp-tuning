"""Rebuild direct poles and centroids from descriptor-level autotune tails."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from exp42_phrase_pole_calibration import TRAIN_IMAGES, descriptor_rows
from exp49_direct_isp_comparison import direct_quantile_calibration, recalibrate_semantic_mapping
from render_pole_centroids import run as render_poles

DEFAULT_BASE = Path("/data/data1/semantic_isp/results/exp80_v14_additive_contrast/contrast")
DEFAULT_OUT = Path("/data/data1/semantic_isp/results/exp81_joint_descriptor_tail_final")
DEFAULT_RENDER_IMAGES = (2, 33, 384)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    import csv
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_record_roots(base, config):
    roots = [base.resolve()]
    seen = set(roots)
    ancestor = config
    while ancestor.get("detail_retuning", {}).get("base"):
        root = Path(ancestor["detail_retuning"]["base"]).resolve()
        if root in seen:
            raise ValueError("Circular tuning-record provenance")
        seen.add(root)
        roots.append(root)
        ancestor_path = root / "direct_soft_calibration.json"
        if not ancestor_path.is_file():
            raise FileNotFoundError(ancestor_path)
        ancestor = json.loads(ancestor_path.read_text())
    return roots


def load_tuning_records(config, roots, images):
    rows = descriptor_rows(config)
    records = []
    counts = {}
    for image_id in images:
        for row in rows:
            relative = Path(
                f"train_img_{image_id}/{row['axis']}_{row['sign']}_{row['id']}.json")
            record_path = next((root / relative for root in roots
                                if (root / relative).is_file()), None)
            if record_path is None:
                raise FileNotFoundError(f"Missing tuning record: {relative}")
            record = json.loads(record_path.read_text())
            if record.get("text") != row["text"]:
                raise ValueError(f"Tuning record text mismatch: {record_path}")
            if record.get("image") != image_id:
                raise ValueError(f"Tuning record image mismatch: {record_path}")
            key = (row["axis"], row["sign"], row["id"])
            counts[key] = counts.get(key, 0) + 1
            records.append(record)
    expected = len(images)
    invalid = {key: count for key, count in counts.items() if count != expected}
    if len(counts) != len(rows) or invalid:
        raise ValueError(f"Incomplete per-descriptor image samples: {invalid}")
    return rows, records


def rebuild(base, output, render_images, device):
    import open_clip

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    base_config_path = base / "direct_soft_calibration.json"
    config = json.loads(base_config_path.read_text())
    roots = load_record_roots(base, config)
    rows, records = load_tuning_records(config, roots, TRAIN_IMAGES)
    calibrated, statistics = direct_quantile_calibration(config, records, quantile=.9)

    torch.manual_seed(0)
    torch.set_num_threads(2)
    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=device,
        force_quick_gelu=config.get("force_quick_gelu", config["pretrained"] == "openai"))
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])

    @torch.no_grad()
    def encode_text(texts):
        return model.encode_text(tokenizer(texts).to(device)).float()

    recalibrated, gate_split = recalibrate_semantic_mapping(calibrated, rows, encode_text)
    recalibrated["status"] = (
        "Direct poles and centroids jointly selected from per-descriptor median autotune tails; "
        "unit-normalized CLIP text embeddings")
    recalibrated["centroid_calibration"] = {
        "base_calibration": str(base_config_path),
        "record_roots": [str(root) for root in roots],
        "training_images": list(TRAIN_IMAGES),
        "tuning_records_reused": len(records),
        "score_calibration": "Recomputed using selected unit-normalized descriptor-tail centroids",
        "relevance_gate": "Recomputed using selected unit-normalized descriptor-tail centroids",
        "gate_diagnostics": gate_split["diagnostics"],
    }

    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "direct_soft_calibration.json", recalibrated)
    save_json(output / "calibrated_poles.json", recalibrated)
    save_csv(output / "pole_quantile_statistics.csv", statistics)
    manifest = {
        "base": str(base),
        "base_config_sha256": hashlib.sha256(base_config_path.read_bytes()).hexdigest(),
        "training_images": list(TRAIN_IMAGES),
        "records_reused": len(records),
        "descriptor_count": len(rows),
        "quantile": .9,
        "centroid_rule": "Median tune setting per phrase across images; select the top ceil(10% of phrases) per pole; average unit-normalized CLIP embeddings of those phrases and normalize the result",
        "query_rule": "Normalize every CLIP query embedding before projection",
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("exp49_direct_isp_comparison.py"),
                         Path(__file__).with_name("semantic_isp.py"))},
    }
    save_json(output / "experiment.json", manifest)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    render_poles(output / "direct_soft_calibration.json", output / "pole_results",
                 list(render_images), device, isolated=True)
    print(json.dumps({
        "output": str(output), "tuning_records_reused": len(records),
        "descriptor_count": len(rows), "selected_centroids": {
            axis: {side: ids for side, ids in sides.items()}
            for axis, sides in recalibrated["learned_calibration"][
                "pole_centroid_descriptor_ids"].items()},
        "render_images": list(render_images),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--images", type=int, nargs="+", default=list(DEFAULT_RENDER_IMAGES))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.images or len(set(args.images)) != len(args.images):
        parser.error("Provide one or more unique render image IDs")
    rebuild(args.base, args.out, args.images, args.device)


if __name__ == "__main__":
    main()