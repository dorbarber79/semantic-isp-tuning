"""Create human-language FiveK descriptors and compact expert/fit comparisons."""

import argparse
import csv
import ctypes
import ctypes.util
import hashlib
import html
import json
import os
from pathlib import Path
import re
import select
import textwrap
from xml.etree import ElementTree
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/data/data1/FiveK/C_data/results/isp_match_105_v3")
DEFAULT_OUTPUT = Path("/data/data1/FiveK/C_data/results/isp_match_105_semantic_v1")
DEFAULT_WORDS = ROOT / "docs/dof scale per word.xlsx"
GROUPS = ("D1", "D2", "D3", "D4")
AXES = ("warmth", "saturation", "tone_gain", "sharpness", "contrast")
STYLES = frozenset(("angelic", "ancient", "ethereal", "noir", "tropical", "gothic",
    "melancholic", "futuristic", "pastoral", "mystical", "whimsical", "bohemian",
    "dreamy", "dramatic", "vintage", "nostalgic", "cinematic", "filmic", "romantic",
    "cheerful", "somber", "serene", "tranquil", "elegant", "fresh", "pastel",
    "golden", "gloomy", "radiant", "sunny", "crisp", "vivid", "subtle", "natural",
    "classic", "faded", "moody", "washed", "hazy", "rustic", "clean", "rich"))
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_word_bank(path):
    with ZipFile(path) as workbook:
        strings_xml = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
        strings = ["".join(node.text or "" for node in entry.iter(f"{{{NS['s']}}}t"))
                   for entry in strings_xml]
        sheet = ElementTree.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    words = []
    for row in sheet.findall(".//s:sheetData/s:row", NS):
        cells = {}
        for cell in row.findall("s:c", NS):
            value = cell.find("s:v", NS)
            if value is None:
                continue
            column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
            text = value.text or ""
            cells[column] = strings[int(text)] if cell.attrib.get("t") == "s" else text
        word = cells.get("A", "").strip().lower()
        if word not in STYLES or not all(column in cells for column in "BCDEF"):
            continue
        try:
            ratings = {axis: float(cells[column].replace("−", "-")) / 3.0
                       for axis, column in zip(AXES, "BCDEF")}
        except ValueError:
            continue
        words.append({"word": word, "ratings": ratings, "appearance": cells.get("G", "")})
    if len(words) < 20:
        raise ValueError(f"Expected a populated five-axis word bank: {path}")
    return words


def style_candidates(values, words):
    active = {axis: float(values[axis]) for axis in AXES if abs(float(values[axis])) >= 0.13}
    candidates, fallback = [], []
    for entry in words:
        ratings = entry["ratings"]
        direction_conflict = any(value * ratings[axis] < 0 and abs(value) >= 0.18
                                  and abs(ratings[axis]) >= 0.3 for axis, value in active.items())
        error = sum((float(values[axis]) - ratings[axis]) ** 2 for axis in AXES) / len(AXES)
        candidate = {"word": entry["word"], "score": error, "ratings": ratings,
                     "appearance": entry["appearance"], "direction_conflict": direction_conflict}
        (fallback if direction_conflict else candidates).append(candidate)
    candidates.sort(key=lambda entry: (entry["score"], entry["word"]))
    fallback.sort(key=lambda entry: (entry["score"], entry["word"]))
    if len(candidates) < 4:
        candidates.extend(fallback[:4 - len(candidates)])
    if len(candidates) < 4:
        raise ValueError("Not enough compatible semantic candidates for the fitted controls")
    return candidates[:4]


def descriptions(values, words):
    choices = (
        ("warmth", 0.12, ("warmer", "golden", "cozier", "golden warmth"), ("cooler", "cool-toned", "crisper and cooler", "cooler tones")),
        ("saturation", 0.16, ("more colorful", "vivid", "richer in color", "livelier colors"), ("less colorful", "muted", "softer in color", "quieter colors")),
        ("tone_gain", 0.12, ("brighter", "luminous", "lighter", "light"), ("darker", "moody", "deeper", "shadow")),
        ("contrast", 0.16, ("more striking", "punchy", "richer in contrast", "definition"), ("gentler", "soft-toned", "flatter", "gentler tones")),
        ("sharpness", 0.16, ("clearer", "crisp", "more defined", "crisper detail"), ("softer", "soft-focus", "gentler in detail", "softer detail")),
        ("palette", 0.2, ("more cinematic in color", "teal-and-orange", "cinematic in color", "teal-and-orange color"), ("more stylized in color", "magenta-and-green", "stylized in color", "magenta-and-green color")),
    )
    active = [(abs(float(values[axis])), positive if float(values[axis]) > 0 else negative)
              for axis, threshold, positive, negative in choices if abs(float(values[axis])) >= threshold]
    active.sort(key=lambda item: -item[0])
    if active:
        primary = active[0][1]
        secondary = active[1][1] if len(active) > 1 else None
        d1 = [f"Make the photo {primary[0]}.", f"Give this image a {primary[1]} look.",
              f"Let the picture feel {primary[2]}.", f"Bring out {primary[3]} in the image."]
        d2 = [f"Give it a {primary[1]} feeling.", f"Make the scene feel {primary[2]}.",
              f"I would like a {primary[1]} atmosphere.", f"Give the image a touch of {primary[3]}."]
        if secondary:
            d3 = [f"Make it {primary[0]} with {secondary[3]}.",
                  f"Give it a {primary[1]} look and {secondary[3]}.",
                  f"Let the image feel {primary[2]} and {secondary[2]}.",
                  f"Bring out {primary[3]} while keeping it {secondary[2]}."]
        else:
            d3 = [f"Give the whole photo a {primary[1]} appearance.",
                  f"Make the overall image feel {primary[2]}.",
                  f"Bring out {primary[3]} in the scene.",
                  f"I would like the image to look {primary[0]} overall."]
    else:
        d1 = ["Keep the image natural.", "Give the photo a balanced look.",
              "Keep the colors and light subtle.", "Let the image look true to life."]
        d2 = ["Give it a calm, natural feeling.", "Let the scene feel clean and gentle.",
              "I would like a quiet, balanced atmosphere.", "Keep the mood understated."]
        d3 = ["Keep the scene natural and easy on the eyes.", "Give the photograph a simple, balanced appearance.",
              "Let the image feel clean without looking dramatic.", "Keep the overall look gentle and natural."]
    matches = style_candidates(values, words)
    d4 = [f"Give the photo {'an' if candidate['word'][0] in 'aeiou' else 'a'} {candidate['word']} feel."
          for candidate in matches]
    groups = {group: {"variants": texts} for group, texts in zip(GROUPS, (d1, d2, d3, d4))}
    return {"groups": groups, "semantic_candidates": matches,
            "source": "Rule-based suggestions from fitted controls and the earlier human-authored word-rating spreadsheet.",
            "caveat": "Subjective candidate interpretations, not photographer instructions or independent image observations."}


def get_font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def paste_image(sheet, path, box):
    left, top, width, height = box
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        sheet.paste(image, (left + (width - image.width) // 2, top + (height - image.height) // 2))


def draw_comparison(path, image_id, rows):
    width, header, height, image_height = 1200, 360, 325, 245
    sheet = Image.new("RGB", (width, header + len(rows) * height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), f"FiveK {image_id}", fill="#19242a", font=get_font(23))
    draw.text((20, 48), "Default ISP", fill="#19242a", font=get_font(18))
    paste_image(sheet, rows[0]["fit"]["neutral_display_path"], (20, 76, 550, 265))
    draw.text((610, 94), "Expert edit and fitted ISP", fill="#19242a", font=get_font(20))
    draw.multiline_text((610, 132), "Human-language suggestions are draft interpretations.\n"
                        "Controls are P90 coordinates relative to the default.\n"
                        "Expert and fit previews have matching dimensions.",
                        fill="#34444c", font=get_font(15), spacing=8)
    for index, row in enumerate(rows):
        top = header + index * height
        fit = row["fit"]
        draw.line((16, top, width - 16, top), fill="#b7c3c6", width=1)
        draw.text((20, top + 8), f"Expert {row['photographer']}", fill="#19242a", font=get_font(18))
        draw.text((20, top + 34), "Expert", fill="#34444c", font=get_font(14))
        draw.text((310, top + 34), "ISP fit", fill="#34444c", font=get_font(14))
        paste_image(sheet, fit["display_path"], (20, top + 58, 280, image_height))
        paste_image(sheet, fit["fit_display_path"], (310, top + 58, 280, image_height))
        texts = row["descriptors"]["groups"]
        labels = ("D1 Visual", "D2 Feeling", "D3 Combined look", "D4 Style ideas")
        detail = [f"{label}: {texts[group]['variants'][0]}" for group, label in zip(GROUPS, labels)]
        detail[3] = "D4 Style ideas: " + ", ".join(candidate["word"] for candidate in row["descriptors"]["semantic_candidates"])
        controls = fit["normalized_controls"]
        detail.extend([f"P90: warmth {controls['warmth']:+.2f}, color {controls['saturation']:+.2f}, "
                       f"brightness {controls['tone_gain']:+.2f}",
                       f"contrast {controls['contrast']:+.2f}, detail {controls['sharpness']:+.2f}, "
                       f"palette {controls['palette']:+.2f}"])
        y = top + 33
        for line in detail:
            for wrapped in textwrap.wrap(line, width=64):
                draw.text((615, y), wrapped, fill="#34444c", font=get_font(13))
                y += 20
            y += 8
    sheet.save(path)


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def process(source, output, bank_path):
    index = json.loads((source / "index.json").read_text())
    bank = read_word_bank(bank_path)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = {"source": str(source.resolve()), "word_bank": str(bank_path.resolve()),
                "word_bank_sha256": sha256(bank_path), "generator_sha256": sha256(Path(__file__)),
                "descriptor_source": "Fitted ISP coordinates and human-authored word-to-control ratings; no image-understanding model.",
                "limitations": "D4 associations are subjective candidates, not equivalent to or verified against the expert edit."}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Semantic output manifest differs; use a new output directory")
    if not manifest_path.exists():
        atomic_json(manifest_path, manifest)
    entries, csv_rows = [], []
    for entry in index["images"]:
        image_id = entry["image_id"]
        source_path = Path(entry["results"])
        source_hash = sha256(source_path)
        image_dir = output / "images" / image_id
        image_dir.mkdir(parents=True, exist_ok=True)
        target = image_dir / "descriptions.json"
        figure = image_dir / "comparison.png"
        if target.is_file() and figure.is_file():
            cached = json.loads(target.read_text())
            if cached.get("source_sha256") != source_hash:
                raise ValueError(f"Source result changed after enrichment: {image_id}")
            enriched = cached
        else:
            original = json.loads(source_path.read_text())
            rows = []
            for photographer, fit in original["per_image"][image_id].items():
                rows.append({"photographer": photographer, "fit": fit,
                             "descriptors": descriptions(fit["normalized_controls"], bank)})
            draw_comparison(figure, image_id, rows)
            enriched = {"image_id": image_id, "source_result": str(source_path),
                        "source_sha256": source_hash, "rows": rows, "figure": str(figure)}
            atomic_json(target, enriched)
            print(f"Enriched {image_id}: {len(rows)} experts", flush=True)
        entries.append({"image_id": image_id, "descriptions": str(target), "figure": str(figure)})
        for row in enriched["rows"]:
            flattened = {"image_id": image_id, "photographer": row["photographer"],
                         "p90_controls_json": json.dumps(row["fit"]["normalized_controls"], sort_keys=True),
                         "semantic_candidates_json": json.dumps(row["descriptors"]["semantic_candidates"], sort_keys=True)}
            for group in GROUPS:
                for position, text in enumerate(row["descriptors"]["groups"][group]["variants"], 1):
                    flattened[f"{group}_{position}"] = text
            csv_rows.append(flattened)
    if csv_rows:
        csv_path = output / "human_descriptors.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        temporary.replace(csv_path)
    links = [f"<li>{html.escape(entry['image_id'])}: <a href='images/{html.escape(entry['image_id'])}/comparison.png'>comparison</a> | "
             f"<a href='images/{html.escape(entry['image_id'])}/descriptions.json'>descriptions and controls</a></li>" for entry in entries]
    page = "<!doctype html><meta charset='utf-8'><title>FiveK semantic comparisons</title>\n" + \
           "<h1>FiveK semantic comparisons</h1><p>Subjective control-derived draft descriptors, not expert instructions.</p><ul>\n" + \
           "\n".join(links) + "\n</ul><p><a href='human_descriptors.csv'>All descriptors</a></p>\n"
    temporary = output / "index.html.tmp"
    temporary.write_text(page)
    temporary.replace(output / "index.html")
    atomic_json(output / "index.json", {"total_images": index["total_images"], "completed_images": len(entries),
                "completed_pairs": len(csv_rows), "images": entries, "manifest": str(manifest_path)})
    return len(entries)


def watch(source, output, bank_path):
    library = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    descriptor = library.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1 failed")
    try:
        if library.inotify_add_watch(descriptor, os.fsencode(source), 0x00000080 | 0x00000008 | 0x00000100) < 0:
            raise OSError(ctypes.get_errno(), f"Cannot watch {source}")
        while True:
            print(f"Enriched {process(source, output, bank_path)} images", flush=True)
            progress_path = source / "progress.json"
            if progress_path.exists():
                state = json.loads(progress_path.read_text()).get("state")
                if state in ("completed", "completed_with_errors", "failed"):
                    print(f"Source batch finished with state: {state}", flush=True)
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
    parser.add_argument("--word-bank", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--watch", action="store_true", help="Enrich newly completed source checkpoints until the batch finishes")
    args = parser.parse_args()
    if args.watch:
        watch(args.source, args.out, args.word_bank)
    else:
        print(f"Enriched {process(args.source, args.out, args.word_bank)} images", flush=True)


if __name__ == "__main__":
    main()