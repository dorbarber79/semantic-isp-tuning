"""Create screen-friendly FiveK comparisons and natural-language style prompts."""

import argparse
import csv
import ctypes
import ctypes.util
import hashlib
import html
import json
import os
from pathlib import Path
import select

from PIL import Image, ImageDraw

from fivek_semantic_descriptions import (
    GROUPS, atomic_json, get_font, paste_image, sha256,
)
DEFAULT_SOURCE = Path("/data/data1/FiveK/C_data/results/isp_match_105_v3")
DEFAULT_OUTPUT = Path("/data/data1/FiveK/C_data/results/isp_match_105_screen_v1")
SCREEN_SIZE = (1800, 1100)
PANEL_WIDTH, PANEL_HEIGHT = 590, 508
VISUAL_FILTER_NAME_FILES = (Path(__file__).with_name("fivek_visual_filter_names.json"),
                            Path(__file__).with_name("fivek_visual_filter_names_additional.json"))
VISUAL_ENRICHMENTS = Path(__file__).with_name("fivek_visual_enrichments.json")
FIT_VISUAL_DESCRIPTIONS = Path(__file__).with_name("fivek_fit_visual_descriptions.json")


def draw_fitted_line(draw, position, text, max_width, size=15, color="#34464b"):
    font = get_font(size)
    while size > 9 and draw.textlength(text, font=font) > max_width:
        size -= 1
        font = get_font(size)
    if draw.textlength(text, font=font) > max_width:
        raise ValueError(f"Text cannot fit within its panel: {text}")
    draw.text(position, text, fill=color, font=font)


def draw_wrapped_block(draw, position, text, max_width, size=12, max_lines=2, color="#34464b"):
    left, top = position
    while size >= 10:
        font = get_font(size)
        lines, current = [], ""
        for word in text.split():
            candidate = f"{current} {word}" if current else word
            if draw.textlength(candidate, font=font) > max_width:
                if current:
                    lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        if len(lines) <= max_lines and all(draw.textlength(line, font=font) <= max_width for line in lines):
            for index, line in enumerate(lines):
                draw.text((left, top + index * (size + 4)), line, fill=color, font=font)
            return top + len(lines) * (size + 4)
        size -= 1
    raise ValueError(f"Text cannot fit within its panel: {text}")


def load_fit_references():
    filter_names = {}
    for names_path in VISUAL_FILTER_NAME_FILES:
        names_document = json.loads(names_path.read_text())
        duplicates = set(names_document["names"]) & set(filter_names)
        if duplicates:
            raise ValueError(f"Duplicate fit look references: {sorted(duplicates)}")
        filter_names.update(names_document["names"])
    enrichment_document = json.loads(VISUAL_ENRICHMENTS.read_text())
    comparisons = enrichment_document["comparisons"]
    if set(comparisons) != set(filter_names):
        raise ValueError("Fit look names and preset references cover different images")
    for image_id, names in filter_names.items():
        if set(names) != set(comparisons[image_id]):
            raise ValueError(f"Preset references do not cover all fit names: {image_id}")
        for photographer, note in comparisons[image_id].items():
            if len(note["filter_references"]) < 2:
                raise ValueError(f"Incomplete fit look references: {image_id}/{photographer}")
    return filter_names, comparisons, enrichment_document["reference_note"]


def load_fit_visual_descriptions(path=None):
    document = json.loads((path or FIT_VISUAL_DESCRIPTIONS).read_text())
    descriptions = document["images"]
    for image_id, fits in descriptions.items():
        if set(fits) != set("ABCDE"):
            raise ValueError(f"Fit descriptions must cover all five fits: {image_id}")
        for fit_name, groups in fits.items():
            if set(groups) != set(GROUPS) or any(not isinstance(groups[group], str) or not groups[group].strip()
                                                 for group in GROUPS):
                raise ValueError(f"Incomplete fit description: {image_id}/{fit_name}")
        for group in GROUPS:
            phrases = [fits[fit_name][group].strip().casefold() for fit_name in fits]
            if len(phrases) != len(set(phrases)):
                raise ValueError(f"Repeated {group} fit description: {image_id}")
    return document


def screen_fit_record(fit):
    return {key: value for key, value in fit.items() if key not in {"descriptors", "agent_prompt"}}


def content_sha256(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fit_descriptor_record(image_id, fit_name, description_document, filter_references, reference_note):
    fit_notes = description_document["images"].get(image_id, {}).get(fit_name)
    if fit_notes is None:
        return {"groups": {group: {"variants": []} for group in GROUPS},
                "semantic_candidates": [], "fit_based": False,
                "status": "Visual fit description pending"}
    return {"groups": {group: {"variants": [fit_notes[group]]} for group in GROUPS},
            "semantic_candidates": [], "fit_based": True,
            "source": description_document["method"],
            "caveat": description_document["limitations"],
            "filter_references": list(filter_references),
            "filter_reference_note": reference_note}


def draw_comparison(path, image_id, rows, heading=None, subheading=None):
    sheet = Image.new("RGB", SCREEN_SIZE, "#f6f7f6")
    draw = ImageDraw.Draw(sheet)
    title_font = get_font(25)
    label_font = get_font(18)
    small_font = get_font(15)
    tiny_font = get_font(13)
    draw.text((20, 12), heading or f"FiveK {image_id}", fill="#15272d", font=title_font)
    draw.text((20, 46), subheading or "Default ISP once  |  D1-D4 describe ISP fits  |  Look references are approximate",
              fill="#41545a", font=small_font)
    panels = [None, *rows]
    for index, row in enumerate(panels):
        left = 10 + (index % 3) * 595
        top = 75 + (index // 3) * PANEL_HEIGHT
        draw.rectangle((left, top, left + PANEL_WIDTH - 8, top + PANEL_HEIGHT - 10),
                       fill="white", outline="#d9e0df", width=1)
        if row is None:
            draw.text((left + 14, top + 12), "Default ISP", fill="#15272d", font=label_font)
            paste_image(sheet, rows[0]["fit"]["neutral_display_path"],
                        (left + 12, top + 47, PANEL_WIDTH - 32, 395))
            continue
        fit = row["fit"]
        title = row.get("title", f"Fit {row['photographer']}")
        if row.get("filter_name"):
            title += f" | {row['filter_name']} look reference"
        draw_fitted_line(draw, (left + 14, top + 10), title, PANEL_WIDTH - 28, size=17, color="#15272d")
        if "prediction_display_path" in row:
            for offset, label, image_path in (
                    (12, "Expert TIFF", fit["display_path"]),
                    (201, "Target ISP fit", fit["fit_display_path"]),
                    (390, "Prediction", row["prediction_display_path"])):
                draw.text((left + offset, top + 39), label, fill="#52646b", font=tiny_font)
                paste_image(sheet, image_path, (left + offset, top + 61, 177, 225))
        else:
            draw.text((left + 15, top + 39), row.get("target_label", "Expert TIFF"), fill="#52646b", font=tiny_font)
            draw.text((left + 303, top + 39), row.get("render_label", "ISP fit"), fill="#52646b", font=tiny_font)
            paste_image(sheet, fit["display_path"], (left + 12, top + 61, 276, 225))
            paste_image(sheet, fit["fit_display_path"], (left + 300, top + 61, 276, 225))
        if "input_text" in row:
            draw_wrapped_block(draw, (left + 13, top + 295),
                "Input: " + row["input_text"], PANEL_WIDTH - 34, size=11, max_lines=8)
        elif not row["descriptors"]["fit_based"]:
            draw_fitted_line(draw, (left + 13, top + 344),
                "Fit-specific visual description pending review.",
                PANEL_WIDTH - 34, size=13, color="#52646b")
        else:
            groups = row["descriptors"]["groups"]
            y = top + 295
            for group in GROUPS:
                y = draw_wrapped_block(draw, (left + 13, y),
                    f"{group}  {groups[group]['variants'][0]}", PANEL_WIDTH - 34,
                    size=11, max_lines=2)
                y += 2
        references = "; ".join(f"{item['app']} {item['name']}"
                                for item in row["descriptors"].get("filter_references", []))
        if references and row["descriptors"]["fit_based"] and "input_text" not in row:
            draw_wrapped_block(draw, (left + 13, y), "Approximate look references: " + references,
                PANEL_WIDTH - 34, size=10, max_lines=2)
        controls = fit["normalized_controls"]
        if "scores" in row:
            scores = row["scores"]
            draw_fitted_line(draw, (left + 13, top + 351),
                f"Parameter MAE (P90): {scores['parameter_mae']:.4f}  |  Lower is better",
                PANEL_WIDTH - 34, size=12)
            psnr = "identical" if scores["image_psnr_db"] is None else f"{scores['image_psnr_db']:.2f} dB"
            draw_fitted_line(draw, (left + 13, top + 373),
                f"Image vs fit: MAE {scores['image_mae_255']:.2f}/255  |  PSNR {psnr}",
                PANEL_WIDTH - 34, size=12)
            if "parameter_absolute_errors" in scores:
                errors = scores["parameter_absolute_errors"]
                draw_fitted_line(draw, (left + 13, top + 394),
                    f"Abs error P90  Warm {errors['warmth']:.3f}  Color {errors['saturation']:.3f}  "
                    f"Light {errors['tone_gain']:.3f}", PANEL_WIDTH - 34, size=10)
                draw_fitted_line(draw, (left + 13, top + 408),
                    f"Abs error P90  Contrast {errors['contrast']:.3f}  Detail {errors['sharpness']:.3f}  "
                    f"Palette {errors['palette']:.3f}", PANEL_WIDTH - 34, size=10)
        if "target_controls" in row:
            for control_index, (label, values) in enumerate((
                    ("GT P90", row["target_controls"]), ("Pred P90", controls))):
                y = top + 422 + control_index * 36
                draw_fitted_line(draw, (left + 13, y),
                    f"{label}  Warm {values['warmth']:+.2f}  Color {values['saturation']:+.2f}  "
                    f"Light {values['tone_gain']:+.2f}", PANEL_WIDTH - 34, size=11, color="#52646b")
                draw_fitted_line(draw, (left + 13, y + 16),
                    f"{label}  Contrast {values['contrast']:+.2f}  Detail {values['sharpness']:+.2f}  "
                    f"Palette {values['palette']:+.2f}", PANEL_WIDTH - 34, size=11, color="#52646b")
            continue
        draw_fitted_line(draw, (left + 13, top + 444),
                         f"P90  Warm {controls['warmth']:+.2f}  Color {controls['saturation']:+.2f}  "
                         f"Light {controls['tone_gain']:+.2f}", PANEL_WIDTH - 34, size=13, color="#52646b")
        draw_fitted_line(draw, (left + 13, top + 463),
                         f"Contrast {controls['contrast']:+.2f}  Detail {controls['sharpness']:+.2f}  "
                         f"Palette {controls['palette']:+.2f}", PANEL_WIDTH - 34, size=13, color="#52646b")
    sheet.save(path)


def process(source, output, description_path=None):
    index = json.loads((source / "index.json").read_text())
    filter_names, comparisons, reference_note = load_fit_references()
    description_path = description_path or FIT_VISUAL_DESCRIPTIONS
    description_document = load_fit_visual_descriptions(description_path)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(source.resolve()), "generator_sha256": sha256(Path(__file__)),
                "fit_visual_descriptions": str(description_path.resolve()),
                "fit_visual_descriptions_sha256": sha256(description_path),
                "visual_reference_inputs": [{"path": str(path.resolve()), "sha256": sha256(path)}
                                             for path in (*VISUAL_FILTER_NAME_FILES, VISUAL_ENRICHMENTS)],
                "descriptor_source": "Manually written fit-specific D1-D4 descriptions from Default ISP and fitted ISP renders; unreviewed fit pairs have no generated descriptors.",
                "filter_reference_note": reference_note,
                "limitations": description_document["limitations"] + " Preset references are approximate look associations, not exact simulations."}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text())
        if previous_manifest.get("source") != manifest["source"]:
            raise ValueError("Refusing to mix different fit sources in one report directory")
    entries, csv_rows = [], []
    for entry in index["images"]:
        image_id = entry["image_id"]
        source_path = Path(entry["results"])
        source_hash = sha256(source_path)
        image_inputs = {
            "fit_descriptions": description_document["images"].get(image_id, {}),
            "filter_names": filter_names.get(image_id, {}),
            "filter_references": comparisons.get(image_id, {}),
            "reference_note": reference_note,
        }
        description_hash = content_sha256(image_inputs)
        image_dir = output / "images" / image_id
        image_dir.mkdir(parents=True, exist_ok=True)
        result_path = image_dir / "descriptions.json"
        figure_path = image_dir / "comparison.png"
        rebuild = not result_path.exists() or not figure_path.exists()
        if not rebuild:
            result = json.loads(result_path.read_text())
            if result["source_sha256"] != source_hash:
                raise ValueError(f"Original fit changed: {image_id}")
            rebuild = (result.get("description_sha256") != description_hash
                       or result.get("renderer_sha256") != manifest["generator_sha256"])
        if rebuild:
            fit_result = json.loads(source_path.read_text())
            rows = []
            per_image_fits = fit_result["per_image"][image_id]
            for photographer, fit in per_image_fits.items():
                reference_data = comparisons.get(image_id, {}).get(photographer, {})
                descriptors = fit_descriptor_record(image_id, photographer,
                    description_document, reference_data.get("filter_references", []), reference_note)
                filter_name = filter_names.get(image_id, {}).get(photographer, "")
                rows.append({"photographer": photographer, "filter_name": filter_name,
                             "fit": screen_fit_record(fit), "descriptors": descriptors})
            draw_comparison(figure_path, image_id, rows)
            result = {"image_id": image_id, "source_result": str(source_path), "source_sha256": source_hash,
                      "description_sha256": description_hash,
                      "renderer_sha256": manifest["generator_sha256"],
                      "figure": str(figure_path), "rows": rows}
            atomic_json(result_path, result)
            print(f"Created screen comparison: {image_id}", flush=True)
        entries.append({"image_id": image_id, "figure": str(figure_path), "descriptions": str(result_path)})
        for row in result["rows"]:
            values = {"image_id": image_id, "photographer": row["photographer"],
                      "proposed_filter_name": row.get("filter_name", ""),
                      "filter_references_json": json.dumps(row["descriptors"].get("filter_references", []), sort_keys=True),
                      "p90_controls_json": json.dumps(row["fit"]["normalized_controls"], sort_keys=True),
                      "semantic_candidates_json": json.dumps(row["descriptors"]["semantic_candidates"], sort_keys=True)}
            for group in GROUPS:
                variants = row["descriptors"]["groups"][group]["variants"]
                for position in range(1, 5):
                    values[f"{group}_{position}"] = variants[position - 1] if len(variants) >= position else ""
            csv_rows.append(values)
    if csv_rows:
        temporary = output / "human_descriptors.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        temporary.replace(output / "human_descriptors.csv")
    links = [f"<li>{html.escape(entry['image_id'])}: <a href='images/{html.escape(entry['image_id'])}/comparison.png'>comparison</a> | "
             f"<a href='images/{html.escape(entry['image_id'])}/descriptions.json'>descriptions and controls</a></li>" for entry in entries]
    page = ("<!doctype html><meta charset='utf-8'><title>FiveK screen comparisons</title>"
            "<h1>FiveK screen comparisons</h1><p>Fit descriptions are image-specific visual notes, written from the Default ISP and fitted ISP renders. Unreviewed fits have no generated descriptor text. Preset names are approximate look references.</p><ul>"
            + "\n".join(links) + "</ul><p><a href='human_descriptors.csv'>All descriptors</a></p>")
    temporary_page = output / "index.html.tmp"
    temporary_page.write_text(page)
    temporary_page.replace(output / "index.html")
    atomic_json(output / "index.json", {"total_images": index["total_images"], "completed_images": len(entries),
                "completed_pairs": len(csv_rows), "images": entries, "manifest": str(manifest_path)})
    atomic_json(manifest_path, manifest)
    return len(entries)


def watch(source, output, description_path=None):
    library = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    descriptor = library.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1 failed")
    try:
        if library.inotify_add_watch(descriptor, os.fsencode(source), 0x00000080 | 0x00000008 | 0x00000100) < 0:
            raise OSError(ctypes.get_errno(), f"Cannot watch {source}")
        while True:
            print(f"Screen comparisons: {process(source, output, description_path)} images", flush=True)
            progress = source / "progress.json"
            if progress.exists() and json.loads(progress.read_text()).get("state") in ("completed", "completed_with_errors", "failed"):
                return
            select.select([descriptor], [], [])
            try:
                while os.read(descriptor, 65536):
                    pass
            except BlockingIOError:
                pass
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--descriptions", type=Path)
    args = parser.parse_args()
    if args.watch:
        watch(args.source, args.out, args.descriptions)
    else:
        print(f"Created {process(args.source, args.out, args.descriptions)} screen comparisons")


if __name__ == "__main__":
    main()