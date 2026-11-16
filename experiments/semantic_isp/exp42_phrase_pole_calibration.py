"""Optimize each Moran pole phrase, then fit poles and distance scales from its statistics."""

import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from differentiable_isp import load_and_preprocess_raw
from exp38_learn_poles import (encode_image, fit_calibration, make_pair_sheet,
                               optimize_controls, save_csv, save_json)
from moran_semantic_isp import (
    AXES, CONTROL_NAMES, SemanticProjection, bounded_controls, compose_parameters,
    neutral_raw_controls, parameters_from_controls, render_linear, seed_config,
    linear_to_srgb8,
)

TRAIN_IMAGES = [1, 28, 236, 194, 277, 263, 103]
TEST_IMAGES = [2, 33, 384]
CORPUS = Path("/data/data1/semantic_isp/datasets/style_words/style_corpus.json")
DATA = Path("/data/data1/EmotionISP_dataset/dng_filtered")


def descriptor_rows(config):
    rows = []
    for axis in config.get("axes", AXES):
        axis_row = config["descriptor_table"]["axes"][axis]
        for sign in ("negative", "positive"):
            for descriptor in axis_row[sign]:
                rows.append({"axis": axis, "sign": sign,
                             "pole": axis_row[f"{sign}_pole"], **descriptor})
    phrases = [row["text"] for row in rows]
    if len(phrases) != len(set(phrases)):
        raise ValueError("Every pole descriptor phrase must be unique for trace naming")
    return rows


def write_render(path, image):
    pixels = linear_to_srgb8(image)[0].permute(1, 2, 0).detach().cpu().numpy()
    if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed export: {path}")


def render_comparison(path, neutral_path, original_path, learned_path, phrase):
    sheet = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(sheet)
    for col, (title, source) in enumerate((("Neutral", neutral_path),
                                           ("Seed poles", original_path),
                                           ("Phrase-calibrated", learned_path))):
        with Image.open(source) as image:
            image.thumbnail((300, 220), Image.Resampling.LANCZOS)
            sheet.paste(image, (col * 300 + (300 - image.width) // 2,
                                32 + (220 - image.height) // 2))
        draw.text((col * 300 + 6, 5), f"{title}: {phrase}", fill="black")
    sheet.save(path, quality=92)


def save_descriptor_stats(records, rows, out):
    medians = []
    descriptor_stats = []
    for descriptor in rows:
        group = [record for record in records if record["id"] == descriptor["id"]
                 and record["axis"] == descriptor["axis"] and record["sign"] == descriptor["sign"]]
        raw = np.asarray([record["raw_controls"] for record in group])
        controls = np.asarray([record["controls"] for record in group])
        medians.append(np.median(raw, axis=0))
        for index, name in enumerate(CONTROL_NAMES):
            descriptor_stats.append({
                "axis": descriptor["axis"], "sign": descriptor["sign"],
                "pole": descriptor["pole"], "descriptor_id": descriptor["id"],
                "phrase": descriptor["text"], "control": name, "n_images": len(group),
                "mean": float(controls[:, index].mean()),
                "median": float(np.median(controls[:, index])),
                "std": float(controls[:, index].std()),
                "q10": float(np.quantile(controls[:, index], .1)),
                "q90": float(np.quantile(controls[:, index], .9)),
            })
    save_csv(out / "phrase_parameter_statistics.csv", descriptor_stats)

    pole_stats = []
    for axis in dict.fromkeys(row["axis"] for row in rows):
        for sign in ("negative", "positive"):
            selected = [index for index, row in enumerate(rows)
                        if row["axis"] == axis and row["sign"] == sign]
            values = np.asarray([medians[index] for index in selected])
            for index, name in enumerate(CONTROL_NAMES):
                pole_stats.append({
                    "axis": axis, "sign": sign, "pole": rows[selected[0]]["pole"],
                    "control": name, "n_descriptors": len(selected),
                    "median_of_phrase_medians": float(np.median(values[:, index])),
                    "q10": float(np.quantile(values[:, index], .1)),
                    "q90": float(np.quantile(values[:, index], .9)),
                })
    save_csv(out / "pole_parameter_statistics.csv", pole_stats)
    return medians


def run(args):
    import open_clip

    torch.manual_seed(0)
    torch.set_num_threads(2)
    config = seed_config()
    descriptors = descriptor_rows(config)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": config["model"], "pretrained": config["pretrained"],
        "force_quick_gelu": config["force_quick_gelu"],
        "train_images": TRAIN_IMAGES, "heldout_images": TEST_IMAGES,
        "pole_count": 10, "descriptors_per_pole": 4, "descriptor_count": len(descriptors),
        "train_runs": len(TRAIN_IMAGES) * len(descriptors),
        "iterations": args.iters, "learning_rate": 0.03,
        "loss": "negative cosine between processed image embedding and the exact pole phrase embedding",
        "loss_terms": ["negative CLIP cosine only"],
        "training_phrases": [row["text"] for row in descriptors],
        "seed_defaults": config,
        "split_policy": "train on pole descriptor phrases and seven images; evaluate unseen corpus words on three held-out images",
    }
    manifest_path = args.out / "experiment.json"
    if manifest_path.exists():
        normalized_manifest = json.loads(json.dumps(manifest))
        if json.loads(manifest_path.read_text()) != normalized_manifest:
            raise ValueError(f"Existing run configuration differs: {manifest_path}")
    elif any(path.name != "__pycache__" for path in args.out.iterdir()):
        raise FileExistsError(f"Refusing to resume output without a matching manifest: {args.out}")
    else:
        save_json(manifest_path, manifest)
    save_csv(args.out / "pole_descriptor_table.csv", descriptors)

    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=args.device,
        force_quick_gelu=config["force_quick_gelu"])
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])

    @torch.no_grad()
    def encode_text(texts):
        return model.encode_text(tokenizer(texts).to(args.device)).float()

    phrases = [row["text"] for row in descriptors]
    phrase_features = F.normalize(encode_text(phrases), dim=-1)
    records = []
    for image_id in TRAIN_IMAGES:
        image = load_and_preprocess_raw(str(DATA / f"{image_id}.dng"), 512).to(args.device)
        folder = args.out / f"train_img_{image_id}"
        folder.mkdir(exist_ok=True)
        for index, descriptor in enumerate(descriptors):
            stem = f"{descriptor['axis']}_{descriptor['sign']}_{descriptor['id']}"
            result_path = folder / f"{stem}.json"
            if result_path.exists():
                record = json.loads(result_path.read_text())
            else:
                result, trace = optimize_controls(image, phrase_features[index:index + 1], model,
                                                  config, args.iters, 0.03)
                record = {**result, "image": image_id, **descriptor}
                save_csv(folder / f"{stem}_trace.csv", trace)
                save_json(result_path, record)
                print(image_id, descriptor["axis"], descriptor["sign"], descriptor["id"],
                      f"gain={result['best_similarity'] - result['initial_similarity']:+.5f}", flush=True)
            records.append(record)
        del image

    median_raw_controls = save_descriptor_stats(records, descriptors, args.out)
    projection = SemanticProjection(encode_text, config)
    descriptor_scores, _ = projection.project(phrases)
    neutral_score, _ = projection.project(["a photo"])
    calibrated = fit_calibration(descriptor_scores.cpu().numpy(), neutral_score[0].cpu().numpy(),
                                 median_raw_controls, config)
    calibrated["status"] = "Phrase-pole CLIP-only calibration; not human-validated"
    calibrated["pole_descriptor_table"] = descriptors
    calibrated["training_split"] = {"images": TRAIN_IMAGES, "phrases": phrases}
    save_json(args.out / "calibrated_poles.json", calibrated)

    learned_projection = SemanticProjection(encode_text, calibrated)
    corpus = json.loads(CORPUS.read_text())
    heldout_words = list(dict.fromkeys(word.lower() for word in corpus["unseen_test_words"]))
    if len(heldout_words) != 30:
        raise ValueError(f"Expected 30 held-out words, found {len(heldout_words)}")
    _, seed_coordinates = projection.project(heldout_words)
    _, learned_coordinates = learned_projection.project(heldout_words)
    seed_parameters = [compose_parameters(value, config) for value in seed_coordinates]
    learned_parameters = [compose_parameters(value, calibrated) for value in learned_coordinates]
    word_features = F.normalize(encode_text([f"a {word} photo" for word in heldout_words]), dim=-1)
    heldout_rows = []
    page = ["<!doctype html><meta charset='utf-8'><title>Phrase-calibrated pole ISP</title>",
            "<h1>Phrase-calibrated poles on held-out words and images</h1>",
            "<p>Pole phrases are the calibration prompts. Results below use disjoint unseen words and images. "
            "Similarity is CLIP self-evaluation, not independent perceptual evidence.</p>"]
    with torch.no_grad():
        for image_id in TEST_IMAGES:
            crop = load_and_preprocess_raw(str(DATA / f"{image_id}.dng"), 512).to(args.device)
            native = load_and_preprocess_raw(str(DATA / f"{image_id}.dng"), None).to(args.device)
            folder = args.out / f"heldout_img_{image_id}"
            folder.mkdir(exist_ok=True)
            neutral = compose_parameters(torch.zeros(5, device=args.device), config)
            neutral_image = render_linear(native, neutral)
            write_render(folder / "neutral.png", neutral_image)
            neutral_crop = encode_image(model, render_linear(crop, neutral))
            for index, word in enumerate(heldout_words):
                row = {"image": image_id, "word": word,
                       "neutral_similarity": float((neutral_crop @ word_features[index:index + 1].T).mean())}
                for name, parameters in (("seed", seed_parameters[index]),
                                         ("phrase_calibrated", learned_parameters[index])):
                    crop_output = render_linear(crop, parameters)
                    row[f"{name}_similarity"] = float((encode_image(model, crop_output)
                                                        @ word_features[index:index + 1].T).mean())
                    write_render(folder / f"{word}_{name}.png", render_linear(native, parameters))
                make_pair_sheet(folder / f"{word}_comparison.jpg", folder / "neutral.png",
                                folder / f"{word}_seed.png", folder / f"{word}_phrase_calibrated.png", word)
                heldout_rows.append(row)
                page.append(f"<h2>{html.escape(word)} — image {image_id}</h2>"
                            f"<img style='max-width:100%' src='heldout_img_{image_id}/{word}_comparison.jpg'>")
            print(f"evaluated held-out image {image_id}", flush=True)
    save_csv(args.out / "heldout_scores.csv", heldout_rows)
    summary = {
        "optimization_runs": len(records), "phrases": len(descriptors), "images": len(TRAIN_IMAGES),
        "iterations_per_run": args.iters,
        "mean_training_cosine_gain": float(np.mean([
            record["best_similarity"] - record["initial_similarity"] for record in records])),
        "heldout_pairs": len(heldout_rows),
        "seed_mean_similarity": float(np.mean([row["seed_similarity"] for row in heldout_rows])),
        "phrase_calibrated_mean_similarity": float(np.mean([
            row["phrase_calibrated_similarity"] for row in heldout_rows])),
        "phrase_calibrated_wins": sum(row["phrase_calibrated_similarity"] > row["seed_similarity"]
                                       for row in heldout_rows),
        "warning": "All tuning uses cosine-only CLIP loss; no human ratings or independent evaluator.",
    }
    save_json(args.out / "summary.json", summary)
    page.insert(3, f"<p>{summary['optimization_runs']} phrase-image optimizations; "
                f"mean training cosine gain {summary['mean_training_cosine_gain']:.5f}; "
                f"held-out mean cosine {summary['seed_mean_similarity']:.5f} seed, "
                f"{summary['phrase_calibrated_mean_similarity']:.5f} phrase-calibrated "
                f"({summary['phrase_calibrated_wins']}/{summary['heldout_pairs']} pair wins).</p>")
    (args.out / "index.html").write_text("\n".join(page))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.iters < 1:
        parser.error("--iters must be positive")
    run(args)


if __name__ == "__main__":
    main()
