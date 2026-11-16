"""Evaluate learned Moran poles on the complete semantic word and RAW image corpus."""

import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from differentiable_isp import load_and_preprocess_raw
from moran_semantic_isp import (
    SemanticProjection, compose_parameters, linear_to_srgb8, render_linear,
)

CORPUS = Path("/data/data1/semantic_isp/datasets/style_words/style_corpus.json")
CALIBRATION = Path("/data/data1/semantic_isp/results/exp42_phrase_pole_clip_100/calibrated_poles.json")
DATA = Path("/data/data1/EmotionISP_dataset/dng_filtered")
IMAGES = [1, 28, 2, 236, 194, 277, 263, 384, 33, 103]
IMAGE_NAMES = {1: "desert", 28: "dog", 2: "child", 236: "man", 194: "swan",
               277: "snow", 263: "dusk", 384: "backlit", 33: "indoor", 103: "city"}
AXIS_NAMES = ("thermal", "fidelity", "moral", "vitality", "hue_shift")


def load_requests(corpus_path=CORPUS):
    corpus = json.loads(Path(corpus_path).read_text())
    requests = []
    seen = set()

    def append(text, category, template=None):
        text = str(text).strip()
        prompt = template.format(text=text) if template else f"a {text} photo"
        key = (category, text.lower())
        if key not in seen:
            requests.append({"request_id": f"req_{len(requests):04d}",
                             "text": text, "category": category,
                             "clip_prompt": prompt})
            seen.add(key)

    for category, field in (("anchor", "anchors"), ("seen_word", "seen_words"),
                            ("unseen_word", "unseen_test_words"),
                            ("compound_phrase", "compound_phrases")):
        for word in corpus[field]:
            append(word, category)
    for category, values in corpus["robustness_test"].items():
        for value in values:
            append(value, f"robustness_{category}")
    for filter_name in corpus["instagram_filters"]["filters"]:
        append(filter_name, "named_filter", "a photo processed with the {text} filter")
    return requests


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def encode_linear(model, image_linear, mean, std):
    image_srgb = image_linear.clamp(0, 1)
    image_srgb = torch.where(image_srgb <= 0.0031308, 12.92 * image_srgb,
                             1.055 * image_srgb.clamp_min(0.0031308).pow(1 / 2.4) - 0.055)
    resized = F.interpolate(image_srgb, (224, 224), mode="bicubic", align_corners=False)
    return F.normalize(model.encode_image((resized - mean) / std), dim=-1)


def save_image(path, image_linear):
    pixels = linear_to_srgb8(image_linear)[0].permute(1, 2, 0).detach().cpu().numpy()
    if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write {path}")


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_contact_sheet(image_dir, requests):
    columns, tile_w, tile_h, header = 6, 240, 190, 48
    rows = (len(requests) + 1 + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + header)), "white")
    draw = ImageDraw.Draw(sheet)
    entries = [{"request_id": "neutral", "text": "neutral", "file": "neutral.png"}]
    entries += [{**request, "file": f"{request['request_id']}.png"} for request in requests]
    for index, entry in enumerate(entries):
        x, y = (index % columns) * tile_w, (index // columns) * (tile_h + header)
        label = f"{entry['category']}: {entry['text']}" if entry.get("category") else "neutral"
        draw.text((x + 4, y + 3), label[:38], fill="black", font=font(12))
        with Image.open(image_dir / entry["file"]) as image:
            image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
            sheet.paste(image, (x + (tile_w - image.width) // 2,
                                y + header + (tile_h - image.height) // 2))
    sheet.save(image_dir / "contact_sheet.jpg", quality=90)


def summarize_requests(requests, coordinates, rows):
    aggregate = []
    for request_index, request in enumerate(requests):
        selected = [row for row in rows if row["request_id"] == request["request_id"]]
        aggregate.append({**request,
                          "mean_cosine_gain": float(np.mean([float(r["cosine_gain"]) for r in selected])),
                          "positive_gain_images": sum(float(r["cosine_gain"]) > 0 for r in selected),
                                  **{axis: float(coordinates[request_index][axis_index])
                             for axis_index, axis in enumerate(AXIS_NAMES)}})
    return aggregate


def run(args):
    import open_clip

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    config = json.loads(args.calibration.read_text())
    requests = load_requests(args.corpus)
    if args.max_requests:
        requests = requests[:args.max_requests]
    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=args.device,
        force_quick_gelu=config.get("force_quick_gelu", config["pretrained"] == "openai"))
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])
    prompts = [request["clip_prompt"] for request in requests]
    with torch.no_grad():
        text_features = F.normalize(model.encode_text(tokenizer(prompts).to(args.device)), dim=-1)
        projection = SemanticProjection(
            lambda texts: model.encode_text(tokenizer(texts).to(args.device)).float(), config)
        raw_scores, coordinates = projection.project(prompts)
    parameters = [compose_parameters(coordinates[index], config) for index in range(len(requests))]

    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / "run_config.json", {
        "model": config["model"], "pretrained": config["pretrained"],
        "quick_gelu": config.get("force_quick_gelu", False),
        "calibration": str(args.calibration), "images": args.images,
        "image_names": {key: value for key, value in IMAGE_NAMES.items() if key in args.images},
        "request_count": len(requests),
        "request_categories": {category: sum(r["category"] == category for r in requests)
                               for category in sorted({r["category"] for r in requests})},
        "scoring": "Full native frame converted to sRGB, resized to 224x224, CLIP cosine with request text",
        "metric_caveat": "Same CLIP family as calibration; self-evaluation, not human or independent evaluation",
        "isp": "linear RGB float32 through sharpening; final sRGB OETF and 8-bit RGB PNG export",
    })
    save_csv(args.out / "requests_and_axis_scores.csv", [
        {**request,
         **{f"raw_{axis}": float(raw_scores[index, axis_index])
            for axis_index, axis in enumerate(AXIS_NAMES)},
         **{axis: float(coordinates[index, axis_index])
            for axis_index, axis in enumerate(AXIS_NAMES)},
         **{f"control_{name}": float(value.detach())
            for name, value in parameters[index].items()
            if name in ("tone_gain", "contrast", "sharpness", "palette_coordinate")}}
        for index, request in enumerate(requests)
    ])

    rows = []
    mean = torch.tensor([.48145466, .4578275, .40821073], device=args.device).view(1, 3, 1, 1)
    std = torch.tensor([.26862954, .26130258, .27577711], device=args.device).view(1, 3, 1, 1)
    neutral_parameters = compose_parameters(torch.zeros(5, device=args.device), config)
    torch.set_num_threads(2)
    with torch.no_grad():
        for image_id in args.images:
            image_path = DATA / f"{image_id}.dng"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            native = load_and_preprocess_raw(str(image_path), None).to(args.device)
            image_dir = args.out / f"img_{image_id}"
            image_dir.mkdir()
            neutral = render_linear(native, neutral_parameters)
            save_image(image_dir / "neutral.png", neutral)
            neutral_feature = encode_linear(model, neutral, mean, std)
            for index, request in enumerate(requests):
                output = render_linear(native, parameters[index])
                feature = encode_linear(model, output, mean, std)
                similarity = float((feature @ text_features[index:index + 1].T).mean())
                neutral_similarity = float((neutral_feature @ text_features[index:index + 1].T).mean())
                filename = f"{request['request_id']}.png"
                save_image(image_dir / filename, output)
                rows.append({
                    "image": image_id, "scene": IMAGE_NAMES[image_id],
                    **request,
                    **{f"axis_{axis}": float(coordinates[index, axis_index])
                       for axis_index, axis in enumerate(AXIS_NAMES)},
                    "neutral_cosine": neutral_similarity,
                    "output_cosine": similarity,
                    "cosine_gain": similarity - neutral_similarity,
                    "output_file": f"img_{image_id}/{filename}",
                })
                del output
            make_contact_sheet(image_dir, requests)
            print(f"evaluated {image_id}: {IMAGE_NAMES[image_id]} ({len(requests)} requests)", flush=True)
            del native
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    save_csv(args.out / "image_request_scores.csv", rows)
    aggregate = summarize_requests(requests, coordinates.cpu().tolist(), rows)
    save_csv(args.out / "request_summary.csv", aggregate)
    overall = {
        "requests": len(requests), "images": len(args.images), "image_request_pairs": len(rows),
        "mean_cosine_gain": float(np.mean([float(row["cosine_gain"]) for row in rows])),
        "positive_gain_pairs": sum(float(row["cosine_gain"]) > 0 for row in rows),
        "warning": "Scores are self-evaluation by the CLIP family used for calibration; they are not independent or human judgments.",
    }
    save_json(args.out / "summary.json", overall)
    page = ["<!doctype html><meta charset='utf-8'><title>Semantic ISP full-corpus evaluation</title>",
            "<style>body{font:14px sans-serif;margin:20px}img{width:100%;max-width:1800px}</style>",
            "<h1>Learned Moran poles: full word and image corpus</h1>",
            f"<p>{len(requests)} requests, {len(IMAGES)} RAW images, {len(rows)} request-image renders. "
            "CLIP cosine gains are self-evaluation, not independent perceptual quality scores.</p>",
            "<p>Request groups: " + ", ".join(f"{key}={value}" for key,value in
                json.loads((args.out / "run_config.json").read_text())["request_categories"].items()) + "</p>"]
    for image_id in args.images:
        page.append(f"<h2>Image {image_id}: {IMAGE_NAMES[image_id]}</h2>"
                    f"<a href='img_{image_id}/contact_sheet.jpg'><img loading='lazy' src='img_{image_id}/contact_sheet.jpg'></a>")
    page.append("<h2>Request score summary</h2><table border='1'><tr>"
                "<th>Request</th><th>Group</th><th>Mean cosine gain</th><th>Images improved</th>"
                + "".join(f"<th>{axis}</th>" for axis in AXIS_NAMES) + "</tr>")
    for row in aggregate:
        page.append("<tr><td>" + html.escape(row["text"]) + "</td><td>"
                    + html.escape(row["category"]) + f"</td><td>{row['mean_cosine_gain']:+.4f}</td>"
                    + f"<td>{row['positive_gain_images']}/{len(args.images)}</td>"
                    + "".join(f"<td>{row[axis]:+.3f}</td>" for axis in AXIS_NAMES) + "</tr>")
    page.append("</table>")
    (args.out / "index.html").write_text("\n".join(page))
    print(json.dumps(overall, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=Path("/data/data1/semantic_isp/results/exp43_full_corpus_learned_poles"))
    parser.add_argument("--images", type=int, nargs="+", default=IMAGES)
    parser.add_argument("--max-requests", type=int, default=0,
                        help="Smoke-test limit; 0 evaluates the full prompt corpus")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    invalid_images = set(args.images) - set(IMAGES)
    if invalid_images:
        parser.error(f"Unknown image IDs: {sorted(invalid_images)}")
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
