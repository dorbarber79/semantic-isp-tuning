"""Convergence-monitored FiveK portrait autotunes and annotated comparisons."""

import argparse
import hashlib
import html
import json
from pathlib import Path

import open_clip
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from autotune import BEAUTY_PROMPTS, REALISM_PROMPTS, CONTROL_NAMES, optimize
from fivek_preview import DEFAULT_CONFIG, DEFAULT_DATA, DEFAULT_SENTENCES, load_linear_png, resize_short_side, write_render, write_sheet
from semantic_isp import parameters_from_controls, render_linear
from pole_normalization import pole_endpoints


DEFAULT_OUT = Path("/data/data1/semantic_isp/results/semantic_parameter_learning/fivek_portraits_siglip_p90")
DEFAULT_IMAGES = ("a0002", "a0008", "a0015")
DEFAULT_SENTENCES_IDS = ("D1_002", "D1_003", "D1_008", "D2_011")
CRITERIA = {"min_iterations": 100, "window": 30, "loss_tolerance": 2e-5,
            "control_tolerance": .01, "gradient_tolerance": .003}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def make_detail_sheet(path, neutral_path, tuned_path, record, optimization_size, semantic_label):
    width, header_height, image_height, table_height = 1280, 115, 690, 515
    sheet = Image.new("RGB", (width, header_height + image_height + table_height), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    regular = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 17) if font_path.exists() else regular
    title = ImageFont.truetype(str(font_path), 24) if font_path.exists() else regular
    sentence = record["sentence_metadata"]
    draw.text((18, 12), f"FiveK {record['image_id']} | {sentence['id']} | {sentence['group']} | "
              f"ambiguity: {sentence['ambiguity']}", fill="#15212a", font=title)
    draw.text((18, 52), sentence["text"], fill="#15212a", font=regular)
    for column, (label, image_path) in enumerate((("Neutral", neutral_path), ("Autotuned", tuned_path))):
        left = column * (width // 2)
        draw.text((left + 18, 82), label, fill="#15212a", font=small)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((width // 2 - 20, image_height - 10), Image.Resampling.LANCZOS)
            sheet.paste(image, (left + (width // 2 - image.width) // 2,
                                header_height + (image_height - image.height) // 2))
    top = header_height + image_height
    draw.rectangle((0, top, width, top + table_height), fill="#f2f5f6")
    draw.text((18, top + 12), "Control", fill="#15212a", font=regular)
    draw.text((330, top + 12), "Agent (-1 to 1)", fill="#15212a", font=regular)
    draw.text((630, top + 12), "Tuned P90 (-1 to 1)", fill="#15212a", font=regular)
    draw.text((975, top + 12), "Physical ISP", fill="#15212a", font=regular)
    table = (("warmth", "warmth"), ("saturation", "saturation"), ("brightness", "tone_gain"),
             ("contrast", "contrast"), ("sharpness", "sharpness"), ("palette", "palette"))
    for index, (estimate_key, physical_key) in enumerate(table):
        y = top + 52 + index * 36
        label = "brightness / tone_gain" if estimate_key == "brightness" else estimate_key
        draw.text((18, y), label, fill="#15212a", font=regular)
        draw.text((330, y), f"{sentence['ratings'][estimate_key]:+.2f}", fill="#15212a", font=regular)
        draw.text((630, y), f"{record['normalized_controls'][physical_key]:+.4f}", fill="#15212a", font=regular)
        draw.text((975, y), f"{record['controls'][physical_key]:+.4f}", fill="#15212a", font=regular)
    status = record["convergence"]
    status_text = (f"Selected restart converged: {status['selected_restart_converged']} | "
                   f"All restarts converged: {status['all_restarts_converged']} | "
                   f"Best update: {record['iteration']} | Start: {record['palette_start']:+.2f}")
    draw.text((18, top + 285), status_text, fill="#15212a", font=small)
    draw.text((18, top + 315), "P90 normalized: -1 = negative pole, 0 = neutral, +1 = positive pole. Estimates are subjective.",
              fill="#34444c", font=small)
    draw.text((18, top + 345),
              f"Loss: semantic {record['losses']['semantic']:+.5f} | realism {record['losses']['realism']:+.5f} | "
              f"beauty {record['losses']['beauty']:+.5f} | regularization {record['losses']['regularization']:+.5f}",
              fill="#34444c", font=small)
    draw.text((18, top + 375),
              f"Quality scores (neutral -> tuned): realism {record['neutral_quality_scores'][0]:+.4f} -> "
              f"{record['quality_scores'][0]:+.4f} | beauty {record['neutral_quality_scores'][1]:+.4f} -> "
              f"{record['quality_scores'][1]:+.4f}", fill="#34444c", font=small)
    draw.text((18, top + 405), f"Semantic {semantic_label} + SigLIP quality proxies; {optimization_size}px short-side tuning, 720p render.",
              fill="#34444c", font=small)
    draw.text((18, top + 435), "Quality scores are proxies, not independent human assessments.",
              fill="#34444c", font=small)
    sheet.save(path)


def run(args):
    if args.max_iters < CRITERIA["min_iterations"] or args.size < 1 or args.lr <= 0:
        raise ValueError("Max iterations must meet the convergence minimum; size and learning rate must be positive")
    if not args.images or len(set(args.images)) != len(args.images):
        raise ValueError("Provide unique FiveK image IDs")
    if not args.sentence_ids or len(set(args.sentence_ids)) != len(args.sentence_ids):
        raise ValueError("Provide unique sentence IDs")
    if args.shared_sentence not in args.sentence_ids:
        raise ValueError("Shared sentence must be included among sentence IDs")
    config = json.loads(args.config.read_text())
    endpoints = pole_endpoints(config)
    if args.semantic_backbone == "siglip":
        semantic_model_name, semantic_pretrained = "ViT-B-16-SigLIP", "webli"
        config["image_mean"] = [.5, .5, .5]
        config["image_std"] = [.5, .5, .5]
    else:
        semantic_model_name, semantic_pretrained = config["model"], config["pretrained"]
    all_sentences = json.loads(args.sentences.read_text())
    sentence_map = {sentence["id"]: sentence for sentence in all_sentences["sentences"]}
    if any(sentence_id not in sentence_map for sentence_id in args.sentence_ids):
        raise ValueError("Requested sentence is missing from the agent-estimate table")
    image_paths = {}
    for prefix in args.images:
        matches = sorted(args.data.glob(prefix + "-*.png"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one FiveK image for {prefix}, found {len(matches)}")
        image_paths[prefix] = matches[0]
    source_paths = [Path(__file__), Path(__file__).with_name("autotune.py"), Path(__file__).with_name("fivek_preview.py"),
                    Path(__file__).with_name("pole_normalization.py")]
    manifest = {"config_path": str(args.config.resolve()),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "sentence_table_path": str(args.sentences.resolve()),
                "sentence_table_sha256": hashlib.sha256(args.sentences.read_bytes()).hexdigest(),
                "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
                "dataset": "FiveK preprocessed 16-bit linear RGB PNG, not original camera RAW",
                "images": {key: str(path.resolve()) for key, path in image_paths.items()},
                "sentence_ids": args.sentence_ids, "shared_sentence": args.shared_sentence,
                "semantic_backbone": args.semantic_backbone,
                "semantic_model": f"{semantic_model_name} ({semantic_pretrained})",
                "semantic_image_mean": config.get("image_mean", [.48145466, .4578275, .40821073]),
                "semantic_image_std": config.get("image_std", [.26862954, .26130258, .27577711]),
                "quality_model": "ViT-B-16-SigLIP (webli)",
                "quality_prompts": {"realism": list(REALISM_PROMPTS), "beauty": list(BEAUTY_PROMPTS)},
                "objective_weights": {"semantic": 1., "realism": 1., "beauty": 1., "regularization": .003},
                "parameterization": "P90-calibrated pole-relative controls in [-1,1], mapped piecewise linearly to physical values; optimizer uses tanh raw coordinates",
                "pole_endpoints": {name: {"negative": float(endpoints[0][index]),
                    "neutral": float(endpoints[1][index]), "positive": float(endpoints[2][index])}
                    for index, name in enumerate(CONTROL_NAMES)},
                "palette_starts": [0., -.02, .02], "max_iterations_per_start": args.max_iters,
                "convergence_criteria": CRITERIA, "optimization_short_side": args.size,
                "render_source": "Full 720p linear PNG", "learning_rate": args.lr,
                "caveat": "Convergence is measured on the proxy objective and does not guarantee semantic correctness or human-perceived quality."}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing manifest differs; use a new output directory")
    elif any(args.out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.out}")
    else:
        save_json(manifest_path, manifest)
    images_out = args.out / "images"
    details_out = args.out / "details"
    records_out = args.out / "records"
    for folder in (images_out, details_out, records_out):
        folder.mkdir(exist_ok=True)

    torch.manual_seed(0)
    torch.set_num_threads(2)
    semantic_model, _, _ = open_clip.create_model_and_transforms(
        semantic_model_name, pretrained=semantic_pretrained, device=args.device,
        force_quick_gelu=config.get("force_quick_gelu", semantic_pretrained == "openai")
        if args.semantic_backbone == "clip" else False)
    if args.semantic_backbone == "siglip":
        quality_model = semantic_model
    else:
        quality_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli", device=args.device)
    semantic_model.eval().requires_grad_(False)
    quality_model.eval().requires_grad_(False)
    semantic_tokenizer = open_clip.get_tokenizer(semantic_model_name)
    quality_tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    with torch.no_grad():
        text_embeddings = F.normalize(semantic_model.encode_text(semantic_tokenizer(
            [sentence_map[sentence_id]["text"] for sentence_id in args.sentence_ids]).to(args.device)).float(), dim=-1)
        quality_embeddings = F.normalize(quality_model.encode_text(quality_tokenizer(
            list(REALISM_PROMPTS + BEAUTY_PROMPTS)).to(args.device)).float(), dim=-1)

    records = []
    for image_id in args.images:
        native = load_linear_png(image_paths[image_id]).to(args.device)
        small = resize_short_side(native, args.size)
        neutral = torch.tensor([0., 0., 1., 1., 0., 0.], dtype=native.dtype, device=native.device)
        neutral_path = images_out / f"{image_id}_neutral.png"
        with torch.no_grad():
            write_render(neutral_path, render_linear(native, parameters_from_controls(neutral, config)))
        sentence_ids = args.sentence_ids if image_id == args.images[0] else [args.shared_sentence]
        for sentence_id in sentence_ids:
            sentence_index = args.sentence_ids.index(sentence_id)
            sentence = sentence_map[sentence_id]
            record_path = records_out / f"{image_id}_{sentence_id}.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record["sentence_metadata"] != sentence or record["image_id"] != image_id:
                    raise ValueError(f"Record does not match the manifest: {record_path}")
            else:
                print(f"Starting {image_id} {sentence_id}: {sentence['text']}", flush=True)
                result = optimize(small, semantic_model, quality_model,
                    text_embeddings[sentence_index:sentence_index + 1], quality_embeddings, config,
                    iterations=args.max_iters, learning_rate=args.lr, convergence=CRITERIA,
                    parameterization="p90")
                record = {"image_id": image_id, "image_path": str(image_paths[image_id].resolve()),
                          "sentence_metadata": sentence, **result}
                save_json(record_path, record)
            controls = torch.tensor([record["controls"][name] for name in CONTROL_NAMES],
                                    device=args.device, dtype=native.dtype)
            tuned_path = images_out / f"{image_id}_{sentence_id}.png"
            with torch.no_grad():
                write_render(tuned_path, render_linear(native, parameters_from_controls(controls, config)))
            detail_path = details_out / f"{image_id}_{sentence_id}.png"
            make_detail_sheet(detail_path, neutral_path, tuned_path, record, args.size, args.semantic_backbone.upper())
            records.append({**record, "neutral_file": str(neutral_path.relative_to(args.out)),
                            "tuned_file": str(tuned_path.relative_to(args.out)),
                            "detail_file": str(detail_path.relative_to(args.out))})
            status = record["convergence"]
            print(f"Completed {image_id} {sentence_id}: selected_converged={status['selected_restart_converged']} "
                  f"all_converged={status['all_restarts_converged']} "
                  f"restart_updates={[r['updates'] for r in status['restarts']]}", flush=True)
        del native, small
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    first_image = args.images[0]
    same_image = [(f"{first_image} | Neutral", images_out / f"{first_image}_neutral.png")]
    same_image.extend((f"{sentence_id} | {sentence_map[sentence_id]['text']}",
                       images_out / f"{first_image}_{sentence_id}.png") for sentence_id in args.sentence_ids)
    write_sheet(args.out / "same_image_different_sentences.png", same_image,
                f"FiveK {first_image}: neutral and different sentences")
    shared = []
    for image_id in args.images:
        shared.extend([(f"{image_id} | Neutral", images_out / f"{image_id}_neutral.png"),
                       (f"{image_id} | {sentence_map[args.shared_sentence]['text']}",
                        images_out / f"{image_id}_{args.shared_sentence}.png")])
    write_sheet(args.out / "different_images_same_sentence.png", shared,
                "FiveK portraits: one sentence, multiple images")
    save_json(args.out / "results.json", {"manifest": "manifest.json", "records": records})
    links = [f"<li>{html.escape(row['image_id'])} / {html.escape(row['sentence_metadata']['id'])}: "
             f"<a href='{html.escape(row['detail_file'])}'>image, controls and metadata</a></li>" for row in records]
    page = ["<!doctype html><meta charset='utf-8'><title>FiveK portrait autotuning</title>",
            "<h1>FiveK portrait autotuning</h1>",
            "<p><a href='same_image_different_sentences.png'>Same image, different sentences</a></p>",
            "<p><a href='different_images_same_sentence.png'>Different images, same sentence</a></p>",
            "<ul>", *links, "</ul>", "<p><a href='results.json'>Full records and convergence traces</a></p>"]
    (args.out / "index.html").write_text("\n".join(page))
    print(json.dumps({"output": str(args.out), "pairs": len(records),
                      "selected_converged": sum(r["convergence"]["selected_restart_converged"] for r in records),
                      "all_restarts_converged": sum(r["convergence"]["all_restarts_converged"] for r in records)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--images", nargs="+", default=list(DEFAULT_IMAGES))
    parser.add_argument("--sentence-ids", nargs="+", default=list(DEFAULT_SENTENCES_IDS))
    parser.add_argument("--shared-sentence", default="D1_002")
    parser.add_argument("--max-iters", type=int, default=600)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=.03)
    parser.add_argument("--semantic-backbone", choices=("siglip", "clip"), default="siglip")
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()