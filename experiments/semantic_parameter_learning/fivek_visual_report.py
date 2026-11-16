"""Join visually authored FiveK annotations to previously computed ISP fits."""

import argparse
import csv
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from fivek_screen_comparisons import draw_comparison
from fivek_semantic_descriptions import GROUPS, atomic_json, get_font, paste_image, sha256


ANNOTATIONS = Path(__file__).with_name("fivek_visual_annotations.json")
FILTER_NAMES = Path(__file__).with_name("fivek_visual_filter_names.json")
SOURCE = Path("/data/data1/FiveK/C_data/results/isp_match_105_v3")
OUTPUT = Path("/data/data1/FiveK/C_data/results/isp_match_visual_filters_v1")


def draw_overview(path, image_id, rows):
    draw_comparison(path, image_id, rows)
    with Image.open(path) as original:
        image = original.copy()
    draw = ImageDraw.Draw(image)
    for index, row in enumerate(rows, 1):
        left = 10 + (index % 3) * 595
        top = 75 + (index // 3) * 508
        draw.rectangle((left + 10, top + 8, left + 570, top + 36), fill="white")
        draw.text((left + 14, top + 10), f"Expert {row['photographer']} | {row['filter_name']}",
                  fill="#15272d", font=get_font(18))
    image.save(path)


def wrap_line(draw, text, font, max_width):
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if draw.textlength(candidate, font=font) > max_width:
            if not current:
                raise ValueError(f"Word does not fit: {word}")
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_descriptor_detail(path, image_id, row):
    width, height = 1920, 1150
    sheet = Image.new("RGB", (width, height), "#f6f7f6")
    draw = ImageDraw.Draw(sheet)
    fit = row["fit"]
    draw.text((24, 12), f"{image_id} | Expert {row['photographer']} | {row['filter_name']}",
              fill="#15272d", font=get_font(27))
    draw.text((24, 49), "Proposed filter name; visual descriptions are subjective.",
              fill="#52646b", font=get_font(17))
    for index, (label, key) in enumerate((("Default ISP", "neutral_display_path"),
                                           ("Expert", "display_path"), ("ISP fit", "fit_display_path"))):
        left = 18 + index * 635
        draw.text((left + 6, 79), label, fill="#15272d", font=get_font(20))
        paste_image(sheet, fit[key], (left + 6, 108, 615, 300))
    headings = {"D1": "D1 | Visible Edit", "D2": "D2 | Perceptual Feeling",
                "D3": "D3 | Combined Look", "D4": "D4 | Semantic Association"}
    for index, group in enumerate(GROUPS):
        left = 24 + (index % 2) * 950
        top = 425 + (index // 2) * 290
        draw.text((left, top), headings[group], fill="#15272d", font=get_font(24))
        y = top + 40
        text_font = get_font(20)
        for phrase_index, phrase in enumerate(row["descriptors"]["groups"][group]["variants"], 1):
            for line_index, line in enumerate(wrap_line(draw, phrase, text_font, 885)):
                draw.text((left + 28, y), line, fill="#34464b", font=text_font)
                if line_index == 0:
                    draw.text((left, y), f"{phrase_index}.", fill="#52646b", font=text_font)
                y += 24
            y += 8
        if y > top + 280:
            raise ValueError(f"Descriptor text overflows: {image_id}/{row['photographer']}/{group}")
    controls = fit["normalized_controls"]
    draw.text((24, 1026), "P90 controls relative to Default ISP", fill="#15272d", font=get_font(19))
    draw.text((24, 1055), "  |  ".join(f"{name} {controls[name]:+.3f}" for name in
              ("warmth", "saturation", "tone_gain", "contrast", "sharpness", "palette")),
              fill="#34464b", font=get_font(18))
    sheet.save(path)


def build_report(annotation_path, source, output, filter_names_path=FILTER_NAMES):
    annotations = json.loads(annotation_path.read_text())
    filter_data = json.loads(filter_names_path.read_text())
    filter_names = filter_data["names"]
    if set(filter_names) != set(annotations["annotations"]):
        raise ValueError("Filter names must cover exactly the annotated images")
    source_index = json.loads((source / "index.json").read_text())
    source_entries = {entry["image_id"]: entry for entry in source_index["images"]}
    missing = set(annotations["annotations"]) - set(source_entries)
    if missing:
        raise ValueError(f"Annotated images do not have completed fits: {sorted(missing)}")
    manifest = {"annotation_path": str(annotation_path.resolve()), "annotation_sha256": sha256(annotation_path),
                "source": str(source.resolve()), "generator_sha256": sha256(Path(__file__)),
                "method": annotations["method"], "limitations": annotations["limitations"],
                "filter_names_path": str(filter_names_path.resolve()), "filter_names_sha256": sha256(filter_names_path),
                "filter_names_method": filter_data["method"],
                "descriptor_generation": "Manually authored by the assistant from displayed expert/default previews; no control-to-word template was used."}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Report manifest differs; use a new output directory")
    if not manifest_path.exists():
        atomic_json(manifest_path, manifest)
    entries, csv_rows = [], []
    for image_id, experts in annotations["annotations"].items():
        if set(experts) != set("ABCDE"):
            raise ValueError(f"Expected five experts for {image_id}")
        if set(filter_names[image_id]) != set(experts):
            raise ValueError(f"Filter names do not cover the experts: {image_id}")
        source_path = Path(source_entries[image_id]["results"])
        source_result = json.loads(source_path.read_text())
        source_hash = sha256(source_path)
        image_dir = output / "images" / image_id
        image_dir.mkdir(parents=True, exist_ok=True)
        result_path, figure_path = image_dir / "descriptions.json", image_dir / "comparison.png"
        rows = []
        for photographer, annotation in experts.items():
            filter_name = filter_names[image_id][photographer].strip()
            if not filter_name:
                raise ValueError(f"Missing proposed filter name: {image_id}/{photographer}")
            for group in GROUPS:
                if len(annotation[group]) != 4 or len(set(annotation[group])) != 4:
                    raise ValueError(f"Expected four distinct {group} phrases: {image_id}/{photographer}")
            descriptor = {"groups": {group: {"variants": annotation[group]} for group in GROUPS},
                          "semantic_candidates": [{"word": word} for word in annotation["semantic_words"]],
                          "observation": annotation["observation"], "source": annotations["method"],
                          "caveat": annotations["limitations"]}
            fit = source_result["per_image"][image_id][photographer]
            row = {"photographer": photographer, "filter_name": filter_name, "fit": fit, "descriptors": descriptor}
            rows.append(row)
            detail_path = image_dir / f"expert_{photographer}_descriptors.png"
            draw_descriptor_detail(detail_path, image_id, row)
            record = {"image_id": image_id, "photographer": photographer, "observation": annotation["observation"],
                      "proposed_filter_name": filter_name, "descriptor_figure": str(detail_path),
                      "semantic_words": ", ".join(annotation["semantic_words"]),
                      "p90_controls_json": json.dumps(fit["normalized_controls"], sort_keys=True)}
            for group in GROUPS:
                for position, phrase in enumerate(annotation[group], 1):
                    record[f"{group}_{position}"] = phrase
            csv_rows.append(record)
        draw_overview(figure_path, image_id, rows)
        atomic_json(result_path, {"image_id": image_id, "source_result": str(source_path),
                                 "source_sha256": source_hash, "rows": rows, "figure": str(figure_path)})
        entries.append({"image_id": image_id, "figure": str(figure_path), "descriptions": str(result_path),
                "expert_figures": {photographer: str(image_dir / f"expert_{photographer}_descriptors.png")
                           for photographer in experts}})
        print(f"Created visual report: {image_id}", flush=True)
    if csv_rows:
        temporary = output / "visual_descriptors.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        temporary.replace(output / "visual_descriptors.csv")
    links = [f"<li>{html.escape(entry['image_id'])}: <a href='images/{html.escape(entry['image_id'])}/comparison.png'>comparison</a> | "
             f"<a href='images/{html.escape(entry['image_id'])}/descriptions.json'>descriptions and controls</a> | "
             + " ".join(f"<a href='images/{html.escape(entry['image_id'])}/expert_{photographer}_descriptors.png'>"
                        f"Expert {photographer}</a>" for photographer in "ABCDE") + "</li>" for entry in entries]
    page = ("<!doctype html><meta charset='utf-8'><title>FiveK visual annotations</title>"
            "<h1>FiveK visual annotations</h1><p>Assistant observations of expert versus Default ISP; associative descriptions and proposed filter names are subjective. These are not Instagram presets.</p><ul>"
            + "\n".join(links) + "</ul><p><a href='visual_descriptors.csv'>All visual descriptors</a></p>")
    temporary_page = output / "index.html.tmp"
    temporary_page.write_text(page)
    temporary_page.replace(output / "index.html")
    atomic_json(output / "index.json", {"completed_images": len(entries), "completed_pairs": len(csv_rows),
                                       "images": entries, "manifest": str(manifest_path)})
    return len(entries), len(csv_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--filter-names", type=Path, default=FILTER_NAMES)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    images, pairs = build_report(args.annotations, args.source, args.out, args.filter_names)
    print(f"Created visual reports for {images} images and {pairs} expert edits")


if __name__ == "__main__":
    main()