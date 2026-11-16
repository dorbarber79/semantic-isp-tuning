"""Probe current semantic poles on a small word and RAW-image sample."""

import argparse
import csv
import html
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from differentiable_isp import load_and_preprocess_raw
from exp42_phrase_pole_calibration import TRAIN_IMAGES
from semantic_isp import (
    CONTROL_NAMES, SemanticProjection, compose_parameters, linear_to_srgb8, render_linear,
)

DEFAULT_CALIBRATION = Path(
    "/data/data1/semantic_isp/results/exp81_joint_descriptor_tail_final/direct_soft_calibration.json")
DEFAULT_CORPUS = Path("/data/data1/semantic_isp/datasets/style_words/style_corpus.json")
DEFAULT_DATA = Path("/data/data1/EmotionISP_dataset/dng_filtered")
DEFAULT_OUT = Path("/data/data1/semantic_isp/results/exp81_joint_descriptor_tail_final/word_probe_small")
DEFAULT_IMAGES = (2, 33, 384)
UNSEEN_IMAGES = (2, 33, 384)
WORD_PROBES = (
    {"id": "anchor_vivid", "category": "anchor", "field": "anchors", "text": "vivid"},
    {"id": "seen_bright", "category": "seen_word", "field": "seen_words", "text": "bright"},
    {"id": "unseen_noir", "category": "unseen_word", "field": "unseen_test_words", "text": "noir"},
    {"id": "compound_soft_dreamy", "category": "compound_phrase",
     "field": "compound_phrases", "text": "soft dreamy"},
    {"id": "seen_retro", "category": "seen_word", "field": "seen_words", "text": "retro"},
)
IMAGE_MEAN = (.48145466, .4578275, .40821073)
IMAGE_STD = (.26862954, .26130258, .27577711)


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_probes(corpus_path, unseen_count=0, selected_words=None):
    corpus = json.loads(corpus_path.read_text())
    probes = []
    if unseen_count:
        words = corpus["unseen_test_words"]
        if unseen_count > len(words):
            raise ValueError(f"Requested {unseen_count} unseen words; corpus has {len(words)}")
        for text in words[:unseen_count]:
            slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
            probes.append({"id": f"unseen_{slug}", "source": "style_corpus",
                           "category": "unseen_word", "text": text,
                           "prompt": f"a {text} photo"})
        return probes
    if selected_words:
        fields = ("anchors", "seen_words", "unseen_test_words", "compound_phrases")
        for text in selected_words:
            field = next((name for name in fields if text in corpus[name]), None)
            if field is None:
                raise ValueError(f"Word is missing from the style corpus: {text}")
            slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
            category = "unseen_word" if field == "unseen_test_words" else field
            probes.append({"id": f"selected_{slug}", "source": "style_corpus",
                           "category": category, "text": text,
                           "prompt": f"a {text} photo"})
        return probes
    for probe in WORD_PROBES:
        if probe["text"] not in corpus[probe["field"]]:
            raise ValueError(f"Probe is missing from {probe['field']}: {probe['text']}")
        probes.append({**probe, "source": "style_corpus",
                   "prompt": f"a {probe['text']} photo"})
    return probes


def parameter_values(parameters):
    return {
        "warmth": float(parameters["wb"][0].detach()) - 1.,
        "saturation": float(parameters["saturation"].detach()),
        "tone_gain": float(parameters["tone_gain"].detach()),
        "contrast": float(parameters["contrast"].detach()),
        "sharpness": float(parameters["sharpness"].detach()),
        "palette": float(parameters["palette_coordinate"].detach()),
    }


def save_png(path, image):
    pixels = linear_to_srgb8(image)[0].permute(1, 2, 0).detach().cpu().numpy()
    Image.fromarray(pixels).save(path)


def make_word_sheet(image_dir, probes):
    columns, tile_width, tile_height, header_height = 4, 320, 220, 32
    entries = [("neutral", "neutral.png")]
    entries.extend((probe["text"], f"{re.sub(r'[^a-z0-9]+', '-', probe['id'].lower()).strip('-')}.png")
                   for probe in probes)
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + header_height)), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.load_default()
    for index, (label, filename) in enumerate(entries):
        x, y = (index % columns) * tile_width, (index // columns) * (tile_height + header_height)
        draw.text((x + 8, y + 8), label, fill="black", font=label_font)
        with Image.open(image_dir / filename) as image:
            image = image.convert("RGB")
            image.thumbnail((tile_width - 12, tile_height), Image.Resampling.LANCZOS)
            sheet.paste(image, (x + (tile_width - image.width) // 2,
                                y + header_height + (tile_height - image.height) // 2))
    sheet.save(image_dir / "word_comparison_sheet.jpg", quality=92)


def encode_image(model, image, device):
    srgb = image.clamp(0, 1)
    srgb = torch.where(srgb <= .0031308, 12.92 * srgb,
                       1.055 * srgb.clamp_min(.0031308).pow(1 / 2.4) - .055)
    resized = F.interpolate(srgb, (224, 224), mode="bicubic", align_corners=False)
    mean = resized.new_tensor(IMAGE_MEAN).view(1, 3, 1, 1)
    std = resized.new_tensor(IMAGE_STD).view(1, 3, 1, 1)
    return F.normalize(model.encode_image((resized - mean) / std), dim=-1)


def mapping_rows(projection, features, prompts, probes, config):
    raw_scores, coordinates = projection.project_features(features)
    rows = []
    settings = []
    for index, probe in enumerate(probes):
        parameters = compose_parameters(coordinates[index], config)
        setting = parameter_values(parameters)
        settings.append(parameters)
        row = {"probe_id": probe["id"], "source": probe["source"],
             "category": probe["category"],
               "text": probe["text"], "prompt": prompts[index]}
        row.update({f"raw_{axis}": float(raw_scores[index, axis_index])
                    for axis_index, axis in enumerate(projection.axes)})
        row.update({f"coord_{axis}": float(coordinates[index, axis_index])
                    for axis_index, axis in enumerate(projection.axes)})
        row.update({f"control_{name}": setting[name] for name in CONTROL_NAMES})
        rows.append(row)
    return raw_scores, coordinates, settings, rows


def centroid_rows(projection, config):
    axes = projection.axes
    features = projection.centroids.reshape(len(axes) * 2, -1)
    raw_scores, coordinates = projection.project_features(features)
    rows = []
    for index, (axis, sign) in enumerate(
            (pair for axis in axes for pair in ((axis, "negative"), (axis, "positive")))):
        parameters = compose_parameters(coordinates[index], config)
        row = {"source_axis": axis, "source_sign": sign}
        row.update({f"raw_{name}": float(raw_scores[index, axis_index])
                    for axis_index, name in enumerate(axes)})
        row.update({f"coord_{name}": float(coordinates[index, axis_index])
                    for axis_index, name in enumerate(axes)})
        row.update({f"control_{name}": value
                    for name, value in parameter_values(parameters).items()})
        rows.append(row)
    return rows


def write_comparison_page(path, probe, image_id, neutral_rel, mapped_rel,
                          mapping, image_score):
    coordinates = "".join(
        f"<tr><th>{html.escape(axis)}</th><td>{mapping[f'coord_{axis}']:+.3f}</td></tr>"
        for axis in mapping["axes"])
    document = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(probe['text'])} on image {image_id}</title>",
        "<style>body{font:15px sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%;height:auto}"
        "table{border-collapse:collapse}th,td{padding:4px 12px;border-bottom:1px solid #ccc}</style>",
        f"<h1>{html.escape(probe['text'])}: image {image_id}</h1>",
        f"<p>{html.escape(probe['category'])}; prompt: {html.escape(probe['prompt'])}</p>",
        "<div class='pair'>",
        f"<figure><figcaption>Neutral</figcaption><img src='../{html.escape(neutral_rel)}'></figure>",
        f"<figure><figcaption>Word mapped through poles</figcaption><img src='../{html.escape(mapped_rel)}'></figure>",
        "</div>",
        f"<p>CLIP cosine: neutral {image_score['neutral_cosine']:.4f}, "
        f"mapped {image_score['mapped_cosine']:.4f}, gain {image_score['cosine_gain']:+.4f}. "
        "This is self-evaluation by the calibration model, not an independent quality score.</p>",
        f"<table>{coordinates}</table>",
        "<p><a href='../index.html'>Back to word probe</a></p>",
    ]
    path.write_text("\n".join(document))


def run(args):
    import open_clip

    if len(args.images) > 5 or len(set(args.images)) != len(args.images):
        raise ValueError("Select one to five unique image IDs")
    if args.unseen_count < 0:
        raise ValueError("Unseen-word count must be nonnegative")
    if args.unseen_count and args.words:
        raise ValueError("Choose either --unseen-count or --words")
    if args.unseen_count and not set(args.images).issubset(UNSEEN_IMAGES):
        raise ValueError(f"Unseen-word evaluation requires held-out images from {UNSEEN_IMAGES}")
    if args.words and set(args.images).intersection(TRAIN_IMAGES):
        raise ValueError("Custom corpus-word evaluation images must not overlap autotune images")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    config = json.loads(args.calibration.read_text())
    probes = validate_probes(args.corpus, args.unseen_count, args.words)
    if not args.unseen_count and not args.words:
        palette_phrase = next(
            entry["text"] for entry in config["descriptor_table"]["axes"]["palette"]["negative"]
            if entry["id"] == "D4_2")
        probes.append({"id": "palette_negative_d4_2", "source": "active_pole_descriptor",
                       "category": "pole_phrase_control", "text": palette_phrase,
                       "prompt": f"a photo with {palette_phrase.lower()}"})
    axes = tuple(config.get("axes", ()))
    if len(axes) != 6:
        raise ValueError(f"Expected the active six-axis calibration, got {axes}")

    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=args.device,
        force_quick_gelu=config.get("force_quick_gelu", config["pretrained"] == "openai"))
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])

    @torch.no_grad()
    def encode_text(texts):
        return model.encode_text(tokenizer(texts).to(args.device)).float()

    projection = SemanticProjection(encode_text, config)
    prompts = [probe["prompt"] for probe in probes]
    with torch.no_grad():
        text_features = F.normalize(encode_text(prompts), dim=-1)
        _, coordinates, settings, word_rows = mapping_rows(
            projection, text_features, prompts, probes, config)
    centroid_data = centroid_rows(projection, config)

    args.out.mkdir(parents=True, exist_ok=True)
    save_csv(args.out / "word_mapping.csv", word_rows)
    save_csv(args.out / "centroid_mapping.csv", centroid_data)
    (args.out / "run_config.json").write_text(json.dumps({
        "calibration": str(args.calibration), "corpus": str(args.corpus),
        "model": config["model"], "images": args.images,
        "evaluation_split": "unseen corpus words x held-out images" if args.unseen_count else (
            "selected corpus words x images not used for autotuning" if args.words else "small mixed probe"),
        "probes": [{"id": probe["id"], "category": probe["category"],
                    "source": probe["source"], "text": probe["text"]} for probe in probes],
        "evaluation_note": "CLIP cosine is self-evaluation by the calibration model.",
    }, indent=2))

    score_rows = []
    page_rows = []
    torch.set_num_threads(2)
    with torch.no_grad():
        for image_id in args.images:
            image_path = args.data / f"{image_id}.dng"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            native = load_and_preprocess_raw(str(image_path), None).to(args.device)
            neutral_parameters = compose_parameters(
                torch.zeros(len(axes), device=args.device), config)
            neutral = render_linear(native, neutral_parameters)
            neutral_feature = encode_image(model, neutral, args.device)
            image_dir = args.out / "images" / f"img_{image_id}"
            image_dir.mkdir(parents=True, exist_ok=True)
            neutral_path = image_dir / "neutral.png"
            save_png(neutral_path, neutral)
            del neutral
            for index, probe in enumerate(probes):
                mapped = render_linear(native, settings[index])
                mapped_feature = encode_image(model, mapped, args.device)
                neutral_cosine = float((neutral_feature @ text_features[index:index + 1].T).item())
                mapped_cosine = float((mapped_feature @ text_features[index:index + 1].T).item())
                slug = re.sub(r"[^a-z0-9]+", "-", probe["id"].lower()).strip("-")
                mapped_path = image_dir / f"{slug}.png"
                save_png(mapped_path, mapped)
                score = {"image": image_id, "probe_id": probe["id"],
                         "category": probe["category"], "text": probe["text"],
                         "neutral_cosine": neutral_cosine,
                         "mapped_cosine": mapped_cosine,
                         "cosine_gain": mapped_cosine - neutral_cosine,
                         "neutral_file": f"images/img_{image_id}/neutral.png",
                         "mapped_file": f"images/img_{image_id}/{slug}.png"}
                score_rows.append(score)
                page_dir = args.out / "comparisons"
                page_dir.mkdir(exist_ok=True)
                page_path = page_dir / f"{slug}_img_{image_id}.html"
                mapping = {"axes": axes,
                           **{f"coord_{axis}": float(coordinates[index, axis_index])
                              for axis_index, axis in enumerate(axes)}}
                write_comparison_page(page_path, probe, image_id,
                                      score["neutral_file"], score["mapped_file"], mapping, score)
                page_rows.append({**score, "page": f"comparisons/{page_path.name}"})
                del mapped
            if not args.skip_sheets:
                make_word_sheet(image_dir, probes)
            print(f"rendered {len(probes)} words on image {image_id}", flush=True)
            del native
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    save_csv(args.out / "image_scores.csv", score_rows)
    links = "".join(
        f"<tr><td>{html.escape(row['text'])}</td><td>{row['image']}</td>"
        f"<td>{row['cosine_gain']:+.4f}</td><td><a href='{html.escape(row['page'])}'>Compare</a></td></tr>"
        for row in page_rows)
    pole_gallery = args.calibration.parent / "pole_results" / "index.html"
    gallery_link = ""
    if pole_gallery.exists():
        import os
        relative_gallery = Path(os.path.relpath(pole_gallery, args.out)).as_posix()
        gallery_link = f"<p><a href='{html.escape(relative_gallery)}'>Current isolated pole gallery</a></p>"
    image_sheet_links = "" if args.skip_sheets else "".join(
        f"<p><a href='images/img_{image_id}/word_comparison_sheet.jpg'>Image {image_id}: neutral and all mapped words</a></p>"
        for image_id in args.images)
    document = [
        "<!doctype html><meta charset='utf-8'><title>Word-to-pole test</title>",
        "<style>body{font:15px sans-serif;max-width:950px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse;width:100%}th,td{padding:6px;border-bottom:1px solid #ccc}</style>",
        "<h1>Unseen-word test</h1>" if args.unseen_count else (
            "<h1>Selected corpus-word test</h1>" if args.words else "<h1>Small word-to-pole probe</h1>"),
        f"<p>{len(probes)} prompts, {len(args.images)} images, {len(page_rows)} comparisons.</p>",
        "<p><a href='word_mapping.csv'>Word coordinates and ISP controls</a> | "
        "<a href='centroid_mapping.csv'>Pole centroid mappings</a> | "
        "<a href='image_scores.csv'>Per-image CLIP scores</a></p>",
        image_sheet_links,
        gallery_link,
        "<table><thead><tr><th>Word</th><th>Image</th><th>CLIP gain</th><th>View</th></tr></thead>",
        f"<tbody>{links}</tbody></table>",
        "<p>CLIP scores use the calibration model and are self-evaluation, not independent quality judgments.</p>",
    ]
    (args.out / "index.html").write_text("\n".join(document))
    print(json.dumps({"output": str(args.out), "words": len(probes),
                      "images": len(args.images), "comparisons": len(page_rows),
                      "centroids": len(centroid_data)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--images", type=int, nargs="+", default=list(DEFAULT_IMAGES))
    parser.add_argument("--unseen-count", type=int, default=0,
                        help="Probe this many unseen_test_words instead of the mixed probe set")
    parser.add_argument("--words", nargs="+",
                        help="Probe these exact words from anchors, seen, unseen, or compound corpus groups")
    parser.add_argument("--skip-sheets", action="store_true",
                        help="Keep comparisons in separate two-image pages without contact sheets")
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()