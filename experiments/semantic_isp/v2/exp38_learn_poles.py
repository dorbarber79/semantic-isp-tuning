"""Learn poles and semantic distance transforms from CLIP cosine-only ISP optimization."""

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
from moran_semantic_isp import (
    AXES, CONTROL_NAMES, CONTROL_BOUNDS, CONTROL_NEUTRAL, SemanticProjection, bounded_controls,
    compose_parameters, neutral_raw_controls, parameters_from_controls,
    render_linear, seed_config, linear_to_srgb8,
)


TRAIN_IMAGES = [1, 28, 236, 194, 277, 263, 103]
TEST_IMAGES = [2, 33, 384]
CORPUS = Path("/data/data1/semantic_isp/datasets/style_words/style_corpus.json")
DATA = Path("/data/data1/EmotionISP_dataset/dng_filtered")


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def srgb(image):
    image = image.clamp(0, 1)
    return torch.where(image <= 0.0031308, 12.92 * image,
                       1.055 * image.clamp_min(0.0031308).pow(1 / 2.4) - 0.055)


def encode_image(model, image, config=None):
    image = F.interpolate(srgb(image), (224, 224), mode="bicubic", align_corners=False)
    config = config or {}
    mean = image.new_tensor(config.get("image_mean", [.48145466, .4578275, .40821073]))[None, :, None, None]
    std = image.new_tensor(config.get("image_std", [.26862954, .26130258, .27577711]))[None, :, None, None]
    return F.normalize(model.encode_image((image - mean) / std), dim=-1)


def save_render(path, image):
    pixels = linear_to_srgb8(image)[0].permute(1, 2, 0).detach().cpu().numpy()
    if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed export: {path}")


def optimize_controls(image, text_feature, model, config, iterations, learning_rate, active_control=None,
                     strength_target_sign=None, strength_weight=0.):
    bounds = config.get("control_bounds", config.get("learned_calibration", {}).get("control_bounds", CONTROL_BOUNDS))
    neutral = neutral_raw_controls(image.device, bounds=bounds)
    if active_control is not None and active_control not in CONTROL_NAMES:
        raise ValueError(f"Unknown active ISP control: {active_control}")
    index = CONTROL_NAMES.index(active_control) if active_control is not None else None
    if strength_target_sign is not None and (index is None or strength_target_sign not in (-1, 1)):
        raise ValueError("A directional strength target requires one active control and sign +/-1")
    if not np.isfinite(strength_weight) or strength_weight < 0:
        raise ValueError("strength_weight must be finite and nonnegative")
    if strength_weight and strength_target_sign is None:
        raise ValueError("A positive strength weight requires a directional target sign")
    indices = torch.tensor([index], device=image.device) if index is not None else None
    inactive_indices = (torch.tensor([column for column in range(len(CONTROL_NAMES)) if column != index],
                                    device=image.device) if index is not None else None)
    neutral_physical = image.new_tensor(CONTROL_NEUTRAL)
    raw = torch.nn.Parameter(neutral.clone() if index is None else neutral[index:index + 1].clone())
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iterations, eta_min=learning_rate * .1)
    best_loss = float("inf")
    best_raw = neutral.detach().clone()
    best_similarity = float("-inf")
    history = []
    strength_target_coordinate = config.get("palette_tuning_objective", {}).get(
        "directional_strength_target_coordinate")
    for iteration in range(iterations + 1):
        full_raw = raw if indices is None else neutral.index_copy(0, indices, raw)
        controls = bounded_controls(full_raw, bounds)
        if inactive_indices is not None:
            controls = controls.index_copy(0, inactive_indices, neutral_physical.index_select(0, inactive_indices))
        rendered = render_linear(image, parameters_from_controls(controls, config))
        similarity = (encode_image(model, rendered, config) @ text_feature.T).mean()
        cosine_loss = -similarity
        strength_penalty = cosine_loss.new_zeros(())
        if strength_weight:
            span = (bounds[index][1] - bounds[index][0]) / 2
            progress = (controls[index] - neutral_physical[index]) / span
            target_progress = 1.
            if strength_target_coordinate is not None:
                if (active_control != "palette" or not np.isfinite(strength_target_coordinate)
                        or not 0 <= strength_target_coordinate <= span):
                    raise ValueError("Palette strength target must be within the palette control range")
                target_progress = strength_target_coordinate / span
            strength_penalty = strength_weight * (
                target_progress - strength_target_sign * progress).square()
        objective = cosine_loss + strength_penalty
        if not torch.isfinite(objective):
            raise ValueError("Nonfinite autotune objective")
        values = controls.detach().cpu().tolist()
        raw_values = full_raw.detach().cpu().tolist()
        history.append({"iteration": iteration, "cosine_loss": cosine_loss.item(),
                "strength_penalty": strength_penalty.item(), "objective_loss": objective.item(),
                "strength_target_sign": strength_target_sign or 0,
                "strength_weight": strength_weight,
                "strength_target_coordinate": strength_target_coordinate,
                "active_control": active_control or "all",
                        **dict(zip(CONTROL_NAMES, values)),
                        **dict(zip((f"raw_{name}" for name in CONTROL_NAMES), raw_values))})
        if objective.item() < best_loss:
            best_loss = objective.item()
            best_raw = full_raw.detach().clone()
            best_similarity = similarity.item()
        if iteration < iterations:
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            scheduler.step()
    best_index = min(range(len(history)), key=lambda index: history[index]["objective_loss"])
    best_controls = bounded_controls(best_raw, bounds)
    if inactive_indices is not None:
        best_controls = best_controls.index_copy(
            0, inactive_indices, neutral_physical.index_select(0, inactive_indices))
    return {"raw_controls": best_raw.cpu().tolist(),
            "controls": best_controls.cpu().tolist(),
            "active_controls": [active_control] if active_control is not None else list(CONTROL_NAMES),
            "strength_target_sign": strength_target_sign,
            "strength_weight": strength_weight,
            "initial_similarity": -history[0]["cosine_loss"],
            "best_similarity": best_similarity, "best_objective": best_loss,
            "best_iteration": best_index}, history


def fit_score_calibration(raw_scores, origin, config):
    axes = tuple(config.get("axes", AXES))
    centered = np.asarray(raw_scores) - np.asarray(origin)
    positive_scale, negative_scale = [], []
    for index in range(len(axes)):
        positive = centered[:, index][centered[:, index] > 0]
        negative = -centered[:, index][centered[:, index] < 0]
        positive_scale.append(float(max(np.quantile(positive, .9) if len(positive) else .05, .01)))
        negative_scale.append(float(max(np.quantile(negative, .9) if len(negative) else .05, .01)))
    calibrated = json.loads(json.dumps(config))
    calibrated["learned_calibration"] = {
        "score_origin": np.asarray(origin).tolist(), "positive_scale": positive_scale,
        "negative_scale": negative_scale,
        "raw_neutral": neutral_raw_controls(bounds=config.get("control_bounds", CONTROL_BOUNDS)).numpy().astype(np.float64).tolist(),
        "pole_raw_deltas": np.zeros((2 * len(axes), len(CONTROL_NAMES))).tolist(),
        "feature_order": [f"{axis}_positive" for axis in axes] + [f"{axis}_negative" for axis in axes],
        "control_order": list(CONTROL_NAMES), "control_bounds": config.get("control_bounds", CONTROL_BOUNDS),
        "ridge": None, "target": "Text-score calibration only; pole strengths not yet assigned",
        "distance_transform": "Signed 90th-percentile scales relative to generic photo text",
    }
    return calibrated


def fit_calibration(raw_scores, origin, word_raw_medians, config):
    calibrated = fit_score_calibration(raw_scores, origin, config)
    axes = tuple(config.get("axes", AXES))
    centered = np.asarray(raw_scores) - np.asarray(origin)
    positive_scale = calibrated["learned_calibration"]["positive_scale"]
    negative_scale = calibrated["learned_calibration"]["negative_scale"]
    normalized = np.clip(centered / np.where(centered >= 0, positive_scale, negative_scale), -1, 1)
    features = np.concatenate((np.maximum(normalized, 0), np.maximum(-normalized, 0)), axis=1)
    neutral = neutral_raw_controls(bounds=config.get("control_bounds", CONTROL_BOUNDS)).numpy().astype(np.float64)
    target = np.asarray(word_raw_medians) - neutral
    ridge_candidates = [.01, .1, 1., 10.]
    errors = {}
    for ridge in ridge_candidates:
        fold_errors = []
        for fold in range(5):
            validation = np.arange(len(features)) % 5 == fold
            training = ~validation
            coefficients = np.linalg.solve(
                features[training].T @ features[training] + ridge * np.eye(2 * len(axes)),
                features[training].T @ target[training])
            fold_errors.append(np.mean((features[validation] @ coefficients - target[validation]) ** 2))
        errors[str(ridge)] = float(np.mean(fold_errors))
    ridge = min(ridge_candidates, key=lambda value: errors[str(value)])
    coefficients = np.linalg.solve(features.T @ features + ridge * np.eye(2 * len(axes)), features.T @ target)
    calibrated["status"] = "Learned from cosine-only CLIP optimization; not human-validated"
    calibrated["learned_calibration"] = {
        "score_origin": np.asarray(origin).tolist(), "positive_scale": positive_scale,
        "negative_scale": negative_scale, "raw_neutral": neutral.tolist(),
        "pole_raw_deltas": coefficients.tolist(), "feature_order": [f"{axis}_positive" for axis in axes]
        + [f"{axis}_negative" for axis in axes],
        "control_order": list(CONTROL_NAMES), "control_bounds": config.get("control_bounds", CONTROL_BOUNDS),
        "ridge": ridge, "word_fold_mse": errors,
        "target": "Per-word median raw controls across calibration images",
        "distance_transform": "Signed 90th-percentile scales relative to generic photo text",
    }
    return calibrated


def make_pair_sheet(path, neutral_path, before_path, after_path, prompt):
    sheet = Image.new("RGB", (900, 250), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (label, source) in enumerate((
        ("Neutral", neutral_path), ("Original text mapping", before_path),
        ("Learned poles and mapping", after_path))):
        with Image.open(source) as image:
            image.thumbnail((300, 210), Image.Resampling.LANCZOS)
            sheet.paste(image, (column * 300 + (300 - image.width) // 2,
                               35 + (210 - image.height) // 2))
        draw.text((column * 300 + 5, 5), f"{label}: {prompt}", fill="black")
    sheet.save(path, quality=92)


def run(args):
    import open_clip
    torch.manual_seed(0)
    torch.set_num_threads(2)
    config = seed_config()
    corpus = json.loads(CORPUS.read_text())
    train_words = list(dict.fromkeys(word.lower() for word in corpus["anchors"] + corpus["seen_words"]))
    test_words = list(dict.fromkeys(word.lower() for word in corpus["unseen_test_words"]))
    if set(train_words) & set(test_words):
        raise ValueError("Training and held-out word sets overlap")
    manifest = {"model": config["model"], "pretrained": config["pretrained"],
                "quick_gelu": True, "train_images": TRAIN_IMAGES, "test_images": TEST_IMAGES,
                "train_words": train_words, "test_words": test_words,
                "iterations": args.iters, "learning_rate": .03, "loss": "negative CLIP cosine similarity only",
                "optimizer": "Adam with cosine LR decay; best iterate includes neutral initialization",
                "crop_size": 512, "seed": 0, "device": args.device,
                "bounds": dict(zip(CONTROL_NAMES, CONTROL_BOUNDS)),
                "fit_protocol": "Word-median raw controls; signed text distance scales; additive ridge pole coefficients"}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "experiment.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("Existing output configuration differs; use a fresh directory")
    save_json(manifest_path, manifest)
    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=args.device, force_quick_gelu=True)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])

    @torch.no_grad()
    def encode_text(texts):
        return model.encode_text(tokenizer(texts).to(args.device)).float()

    text_features = F.normalize(encode_text([f"a {word} photo" for word in train_words]), dim=-1)
    records = []
    for image_id in TRAIN_IMAGES:
        crop = load_and_preprocess_raw(str(DATA / f"{image_id}.dng"), 512).to(args.device)
        directory = args.out / f"train_img_{image_id}"
        directory.mkdir(exist_ok=True)
        for index, word in enumerate(train_words):
            path = directory / f"{word}.json"
            if path.exists():
                record = json.loads(path.read_text())
            else:
                record, history = optimize_controls(crop, text_features[index:index + 1], model,
                                                    config, args.iters, .03)
                record.update(image=image_id, word=word)
                save_csv(directory / f"{word}_trace.csv", history)
                save_json(path, record)
                print(image_id, word, f"gain={record['best_similarity'] - record['initial_similarity']:+.5f}", flush=True)
            records.append(record)
        del crop

    medians, statistics_rows = [], []
    for word in train_words:
        entries = [record for record in records if record["word"] == word]
        raw = np.array([entry["raw_controls"] for entry in entries])
        controls = np.array([entry["controls"] for entry in entries])
        medians.append(np.median(raw, axis=0))
        for index, name in enumerate(CONTROL_NAMES):
            statistics_rows.append({"word": word, "control": name, "n_images": len(entries),
                                    "mean": float(controls[:, index].mean()),
                                    "median": float(np.median(controls[:, index])),
                                    "std": float(controls[:, index].std()),
                                    "q10": float(np.quantile(controls[:, index], .1)),
                                    "q90": float(np.quantile(controls[:, index], .9))})
    save_csv(args.out / "parameter_statistics.csv", statistics_rows)
    original_projection = SemanticProjection(encode_text, config)
    train_scores, _ = original_projection.project(train_words)
    neutral_scores, _ = original_projection.project(["a photo"])
    calibrated = fit_calibration(train_scores.cpu().numpy(), neutral_scores[0].cpu().numpy(),
                                 medians, config)
    save_json(args.out / "calibrated_poles.json", calibrated)
    learned_projection = SemanticProjection(encode_text, calibrated)
    _, old_coordinates = original_projection.project(test_words)
    _, learned_coordinates = learned_projection.project(test_words)
    old_parameters = [compose_parameters(value, config) for value in old_coordinates]
    learned_parameters = [compose_parameters(value, calibrated) for value in learned_coordinates]
    held_text = F.normalize(encode_text([f"a {word} photo" for word in test_words]), dim=-1)
    rows = []
    document = ["<!doctype html><meta charset='utf-8'><title>Learned CLIP poles</title>",
                "<h1>Held-out words and images: original vs learned mapping</h1>",
                "<p>Cosine-only optimization and statistical calibration; scores use the same CLIP model "
                "and are not independent perceptual validation.</p>"]
    with torch.no_grad():
        for image_id in TEST_IMAGES:
            crop = load_and_preprocess_raw(str(DATA / f"{image_id}.dng"), 512).to(args.device)
            native = load_and_preprocess_raw(str(DATA / f"{image_id}.dng"), None).to(args.device)
            directory = args.out / f"test_img_{image_id}"
            directory.mkdir(exist_ok=True)
            neutral_parameters = parameters_from_controls(bounded_controls(neutral_raw_controls(args.device)), config)
            save_render(directory / "neutral.png", render_linear(native, neutral_parameters))
            neutral_feature = encode_image(model, render_linear(crop, neutral_parameters))
            for index, word in enumerate(test_words):
                row = {"image": image_id, "word": word,
                       "neutral_similarity": float((neutral_feature @ held_text[index:index + 1].T).mean())}
                for side, parameters in (("original", old_parameters[index]),
                                          ("learned", learned_parameters[index])):
                    crop_output = render_linear(crop, parameters)
                    row[f"{side}_similarity"] = float((encode_image(model, crop_output)
                                                       @ held_text[index:index + 1].T).mean())
                    save_render(directory / f"{word}_{side}.png", render_linear(native, parameters))
                make_pair_sheet(directory / f"{word}_comparison.jpg", directory / "neutral.png",
                                directory / f"{word}_original.png", directory / f"{word}_learned.png", word)
                rows.append(row)
                document.append(f"<h2>{html.escape(word)}: image {image_id}</h2>"
                                f"<img style='max-width:100%' src='test_img_{image_id}/{word}_comparison.jpg'>")
            print(f"evaluated held-out image {image_id}", flush=True)
    save_csv(args.out / "heldout_scores.csv", rows)
    summary = {"training_runs": len(records), "iterations_per_run": args.iters,
               "training_mean_cosine_gain": float(np.mean([r["best_similarity"] - r["initial_similarity"]
                                                           for r in records])),
               "heldout_pairs": len(rows),
               "heldout_original_mean_similarity": float(np.mean([row["original_similarity"] for row in rows])),
               "heldout_learned_mean_similarity": float(np.mean([row["learned_similarity"] for row in rows])),
               "heldout_learned_wins": sum(row["learned_similarity"] > row["original_similarity"] for row in rows),
               "warning": "Calibration corpus is a development set; no human preference or independent VLM evaluation"}
    save_json(args.out / "summary.json", summary)
    (args.out / "index.html").write_text("\n".join(document))
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