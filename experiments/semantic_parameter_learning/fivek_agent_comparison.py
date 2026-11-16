"""Render agent estimates alongside saved FiveK P90 autotuning results."""

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from autotune import CONTROL_NAMES
from fivek_preview import load_linear_png, write_render
from moran_semantic_isp import parameters_from_controls, render_linear
from pole_normalization import normalized_to_physical, pole_endpoints, physical_to_normalized


RESULT_ROOT = Path("/data/data1/semantic_isp/results/semantic_parameter_learning")
DEFAULT_BASE = RESULT_ROOT / "fivek_portraits_siglip_p90"
DEFAULT_EXTENDED = RESULT_ROOT / "fivek_siglip_p90_a0008_extended"
DEFAULT_OUT = RESULT_ROOT / "fivek_p90_agent_comparisons"
ESTIMATE_KEYS = ("warmth", "saturation", "brightness", "contrast", "sharpness", "palette")


def read_result(root):
    manifest = json.loads((root / "manifest.json").read_text())
    records = json.loads((root / "results.json").read_text())["records"]
    return manifest, records


def compatible_manifests(first, second):
    keys = ("config_sha256", "sentence_table_sha256", "source_sha256", "semantic_backbone", "semantic_model",
            "semantic_image_mean", "semantic_image_std", "quality_model", "quality_prompts", "objective_weights",
            "parameterization", "pole_endpoints", "optimization_short_side", "render_source", "learning_rate")
    return all(first.get(key) == second.get(key) for key in keys)


def assert_matches_saved(rendered, original_path):
    with Image.open(rendered) as current, Image.open(original_path) as original:
        current_pixels = np.asarray(current.convert("RGB"), dtype=np.int16)
        original_pixels = np.asarray(original.convert("RGB"), dtype=np.int16)
    if current_pixels.shape != original_pixels.shape:
        raise ValueError(f"Render dimensions differ from saved output: {original_path}")
    difference = np.abs(current_pixels - original_pixels)
    if float(difference.mean()) > .05 or int(difference.max()) > 3:
        raise ValueError(f"Current renderer does not reproduce the saved output: {original_path}; "
                         f"mean difference {difference.mean():.4f}, max {difference.max()}")


def make_sheet(path, images, record, agent_physical):
    width, header_height, image_height, table_height = 1800, 120, 690, 540
    sheet = Image.new("RGB", (width, header_height + image_height + table_height), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    normal = ImageFont.truetype(str(font_path), 21) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 18) if font_path.exists() else normal
    title = ImageFont.truetype(str(font_path), 26) if font_path.exists() else normal
    sentence = record["sentence_metadata"]
    draw.text((18, 12), f"FiveK {record['image_id']} | {sentence['id']} | {sentence['group']} | "
              f"ambiguity: {sentence['ambiguity']}", fill="#15212a", font=title)
    draw.text((18, 52), sentence["text"], fill="#15212a", font=normal)
    for column, (label, image_path) in enumerate(images):
        left = column * (width // 3)
        draw.text((left + 18, 88), label, fill="#15212a", font=small)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((width // 3 - 20, image_height - 10), Image.Resampling.LANCZOS)
            sheet.paste(image, (left + (width // 3 - image.width) // 2,
                                header_height + (image_height - image.height) // 2))
    top = header_height + image_height
    draw.rectangle((0, top, width, top + table_height), fill="#f2f5f6")
    columns = (18, 360, 700, 1060, 1430)
    headings = ("Control", "Agent P90", "Agent physical", "Autotune P90", "Autotune physical")
    for position, heading in zip(columns, headings):
        draw.text((position, top + 16), heading, fill="#15212a", font=normal)
    for index, (estimate_key, physical_key) in enumerate(zip(ESTIMATE_KEYS, CONTROL_NAMES)):
        position = top + 57 + index * 36
        label = "brightness / tone_gain" if estimate_key == "brightness" else estimate_key
        values = (label, f"{sentence['ratings'][estimate_key]:+.3f}",
                  f"{agent_physical[physical_key]:+.4f}",
                  f"{record['normalized_controls'][physical_key]:+.3f}",
                  f"{record['controls'][physical_key]:+.4f}")
        for column, value in zip(columns, values):
            draw.text((column, position), value, fill="#15212a", font=normal)
    status = record["convergence"]
    draw.text((18, top + 290),
        f"Selected restart converged: {status['selected_restart_converged']} | "
        f"All restarts converged: {status['all_restarts_converged']} | "
        f"Best update: {record['iteration']} | Start: {record['palette_start']:+.2f}",
        fill="#15212a", font=small)
    draw.text((18, top + 324), "P90: -1 = negative pole, 0 = neutral, +1 = positive pole. Agent estimates are subjective.",
              fill="#34444c", font=small)
    draw.text((18, top + 358),
        f"Loss: semantic {record['losses']['semantic']:+.5f} | realism {record['losses']['realism']:+.5f} | "
        f"beauty {record['losses']['beauty']:+.5f} | regularization {record['losses']['regularization']:+.5f}",
        fill="#34444c", font=small)
    draw.text((18, top + 392),
        f"Quality (neutral -> tuned): realism {record['neutral_quality_scores'][0]:+.4f} -> "
        f"{record['quality_scores'][0]:+.4f} | beauty {record['neutral_quality_scores'][1]:+.4f} -> "
        f"{record['quality_scores'][1]:+.4f}", fill="#34444c", font=small)
    draw.text((18, top + 426), "Autotune: SigLIP semantics and quality proxies; 512px short-side tuning, 720p render.",
              fill="#34444c", font=small)
    draw.text((18, top + 460), "Agent image applies the listed estimated controls directly; it is not autotuned.",
              fill="#34444c", font=small)
    draw.text((18, top + 494), "Convergence concerns the proxy objective; it does not guarantee semantic correctness or visual quality.",
              fill="#34444c", font=small)
    sheet.save(path)


def run(base, extended, output, device):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest, records = read_result(base)
    extended_manifest, extended_records = read_result(extended)
    if not compatible_manifests(manifest, extended_manifest):
        raise ValueError("Extended run is not compatible with the original P90 run")
    config_path = Path(manifest["config_path"])
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("Calibration no longer matches the saved result")
    config = json.loads(config_path.read_text())
    endpoints = pole_endpoints(config)
    for index, name in enumerate(CONTROL_NAMES):
        for endpoint_index, label in enumerate(("negative", "neutral", "positive")):
            if abs(float(endpoints[endpoint_index][index]) - manifest["pole_endpoints"][name][label]) > 1e-8:
                raise ValueError(f"Saved P90 endpoint mismatch: {name}/{label}")
    extended_by_key = {(row["image_id"], row["sentence_metadata"]["id"]): row for row in extended_records}
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    detail_dir = output / "details"
    image_dir.mkdir()
    detail_dir.mkdir()
    summaries = []
    torch.set_num_threads(2)
    for original in records:
        key = (original["image_id"], original["sentence_metadata"]["id"])
        candidate = extended_by_key.get(key)
        source_root = base
        record = original
        if candidate is not None:
            if (candidate["image_path"] != original["image_path"] or
                    candidate["sentence_metadata"] != original["sentence_metadata"]):
                raise ValueError(f"Extended result identity mismatch: {key}")
            if (candidate["convergence"]["selected_restart_converged"] and
                    candidate["losses"]["total"] < original["losses"]["total"]):
                record, source_root = candidate, extended
        image_id, sentence_id = key
        ratings = record["sentence_metadata"]["ratings"]
        normalized = torch.tensor([ratings[name] for name in ESTIMATE_KEYS], dtype=torch.float32, device=device)
        if not torch.isfinite(normalized).all() or bool((normalized.abs() > 1).any()):
            raise ValueError(f"Agent estimates are not finite P90 coordinates: {key}")
        agent_controls = normalized_to_physical(normalized, endpoints)
        tuned_controls = torch.tensor([record["controls"][name] for name in CONTROL_NAMES],
                                      dtype=torch.float32, device=device)
        reported = torch.tensor([record["normalized_controls"][name] for name in CONTROL_NAMES],
                                dtype=torch.float32, device=device)
        torch.testing.assert_close(physical_to_normalized(tuned_controls, endpoints), reported, atol=2e-6, rtol=0)
        if bool((reported.abs() > 1.000001).any()):
            raise ValueError(f"Autotuned controls exceed P90 bounds: {key}")
        neutral = torch.tensor([0., 0., 1., 1., 0., 0.], dtype=torch.float32, device=device)
        image = load_linear_png(Path(record["image_path"])).to(device)
        files = {}
        with torch.no_grad():
            for label, controls in (("neutral", neutral), ("agent", agent_controls), ("autotuned", tuned_controls)):
                path = image_dir / f"{image_id}_{sentence_id}_{label}.png"
                write_render(path, render_linear(image, parameters_from_controls(controls, config)))
                files[label] = path
        assert_matches_saved(files["neutral"], source_root / record["neutral_file"])
        assert_matches_saved(files["autotuned"], source_root / record["tuned_file"])
        agent_physical = dict(zip(CONTROL_NAMES, agent_controls.detach().cpu().tolist()))
        detail_path = detail_dir / f"{image_id}_{sentence_id}.png"
        make_sheet(detail_path, [("Neutral", files["neutral"]), ("Agent estimate", files["agent"]),
                                 ("Autotuned", files["autotuned"])], record, agent_physical)
        summaries.append({"image_id": image_id, "sentence_metadata": record["sentence_metadata"],
                          "agent_physical_controls": agent_physical,
                          "autotuned_normalized_controls": record["normalized_controls"],
                          "autotuned_physical_controls": record["controls"],
                          "autotune_source": str(source_root), "autotune_record": f"records/{image_id}_{sentence_id}.json",
                          "selected_restart_converged": record["convergence"]["selected_restart_converged"],
                          "all_restarts_converged": record["convergence"]["all_restarts_converged"],
                          "detail_file": str(detail_path.relative_to(output)),
                          "agent_image_file": str(files["agent"].relative_to(output)),
                          "neutral_image_file": str(files["neutral"].relative_to(output)),
                          "autotuned_image_file": str(files["autotuned"].relative_to(output))})
        print(f"Rendered {image_id} {sentence_id}: source={source_root.name}", flush=True)
        del image
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    (output / "results.json").write_text(json.dumps({"calibration": str(config_path),
        "normalization": "Piecewise linear between P90 endpoints, neutral at zero",
        "agent_note": "Agent estimates are applied directly and are not autotuned or human-validated.",
        "records": summaries}, indent=2, allow_nan=False))
    links = [f"<li>{html.escape(row['image_id'])} / {html.escape(row['sentence_metadata']['id'])}: "
             f"<a href='{html.escape(row['detail_file'])}'>neutral, agent, autotuned and settings</a></li>"
             for row in summaries]
    (output / "index.html").write_text("\n".join([
        "<!doctype html><meta charset='utf-8'><title>FiveK agent and autotune</title>",
        "<h1>FiveK agent and autotune comparisons</h1><ul>", *links,
        "</ul><p><a href='results.json'>Settings and result provenance</a></p>"]))
    print(json.dumps({"output": str(output), "comparisons": len(summaries),
                      "extended_results_used": sum(row["autotune_source"] == str(extended) for row in summaries)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--extended", type=Path, default=DEFAULT_EXTENDED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.base, args.extended, args.out, args.device)


if __name__ == "__main__":
    main()