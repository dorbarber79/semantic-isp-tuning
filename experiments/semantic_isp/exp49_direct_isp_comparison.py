"""Learn explicit ISP-control poles and compare them with cinematic axes under softer gating."""

import argparse
import copy
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from scipy.optimize import lsq_linear
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from differentiable_isp import load_and_preprocess_raw
from exp38_learn_poles import encode_image, fit_calibration, fit_score_calibration, optimize_controls, save_csv, save_json, save_render
from exp42_phrase_pole_calibration import descriptor_rows, save_descriptor_stats, TRAIN_IMAGES, TEST_IMAGES
from exp47_relevance_calibration import (
    TRAIN_UNRELATED, VALIDATION_UNRELATED, TEST_UNRELATED, FEATURE_NAMES, constraint_signs,
)
from moran_semantic_isp import (
    AXES, CONTROL_NAMES, CONTROL_BOUNDS, CONTROL_NEUTRAL, SemanticProjection, bounded_controls,
    neutral_raw_controls, compose_parameters, relevance_features, render_linear, seed_config,
)


TABLE = Path(__file__).with_name("isp_control_pole_descriptors.json")
RESULTS = Path("/data/data1/semantic_isp/results")
DATA = Path("/data/data1/EmotionISP_dataset/dng_filtered")
VALIDATION_POSITIVE = {
    "warmth": ["a photograph with amber illumination", "a photograph with a bluish color cast"],
    "saturation": ["an image with colorful rich pigments", "an image with nearly gray colors"],
    "brightness": ["a photograph with light midtones", "a dimly exposed photograph"],
    "contrast": ["an image with bold tonal separation", "an image with gentle tonal separation"],
    "sharpness": ["an image with clearly resolved fine details", "an out of focus image"],
    "palette": ["a photo dominated by teal and orange", "a photo dominated by green and magenta"],
}
DEVELOPMENT = [
    ("warm", "warmth", "thermal"), ("cool", "warmth", "thermal"),
    ("bright", "brightness", "moral"), ("dark", "brightness", "moral"),
    ("saturated", "saturation", "vitality"), ("desaturated", "saturation", "vitality"),
    ("sharp", "sharpness", "fidelity"), ("blurry", "sharpness", "fidelity"),
    ("high contrast", "contrast", "fidelity"), ("low contrast", "contrast", "fidelity"),
    ("magenta green palette", "palette", "hue_shift"), ("teal orange palette", "palette", "hue_shift"),
]


def direct_config(table_path=TABLE):
    config = seed_config()
    table = json.loads(Path(table_path).read_text())
    axes = list(table["axes"])
    if axes != ["warmth", "saturation", "brightness", "contrast", "sharpness", "palette"]:
        raise ValueError("Direct descriptor axes must match physical control order")
    config["axes"] = axes
    config["descriptor_table"] = table
    prototype = copy.deepcopy(config["poles"]["thermal"]["positive"])
    config["poles"] = {}
    for axis in axes:
        config["poles"][axis] = {}
        for side in ("negative", "positive"):
            pole = copy.deepcopy(prototype)
            pole.update(wb=[1., 1., 1.], ccm_delta=np.zeros((3, 3)).tolist(),
                        tone_gain=1., contrast=1., sharpness=0.)
            entries = table["axes"][axis][side]
            per_group = table.get("sentences_per_descriptor_type", 1)
            groups = [row.get("group", row["id"]) for row in entries]
            if len(entries) != 4 * per_group or any(
                    groups.count(group) != per_group for group in ("D1", "D2", "D3", "D4")) or (
                    len({row["id"] for row in entries}) != len(entries)):
                raise ValueError(f"Invalid descriptors for {axis}/{side}")
            pole["texts"] = [row["text"] for row in entries]
            config["poles"][axis][side] = pole
    config["descriptor_provenance"] = str(table_path)
    config["color_protection"] = {
        "mode": "gray_white_and_skin",
        "strength": 1.,
        "gray_sigma": .07,
        "skin_sigma": .10,
        "skin_reference_srgb": [.76, .57, .47],
        "operations": ["palette"],
        "saturation_positive_only": True,
    }
    config["status"] = "Direct physical ISP semantic axes; uncalibrated numeric poles"
    return config


def select_soft_band(probabilities, labels):
    choices = []
    for lower in (0., .01, .025, .05, .1, .15, .2, .3, .4):
        for upper in (.15, .2, .3, .4, .5, .65, .8, 1.):
            if upper <= lower:
                continue
            progress = np.clip((probabilities - lower) / (upper - lower), 0, 1)
            weights = progress ** 2 * (3 - 2 * progress)
            positive = float(weights[labels == 1].mean())
            negative = float(weights[labels == 0].mean())
            objective = (2 * positive + (1 - negative)) / 3
            choices.append((objective, positive, -negative, lower, upper))
    _, positive, minus_negative, lower, upper = max(choices)
    return lower, upper, positive, -minus_negative


def fit_soft_gate(config, encode):
    axes = list(config.get("axes", AXES))
    descriptors = descriptor_rows(config)
    training = [row for row in descriptors if row.get("group", row["id"]) != "D4"]
    validation = [row for row in descriptors if row.get("group", row["id"]) == "D4"]
    auxiliary = VALIDATION_POSITIVE
    if "axes" not in config:
        auxiliary = {"thermal": auxiliary["warmth"],
                     "fidelity": auxiliary["sharpness"] + auxiliary["contrast"],
                     "moral": auxiliary["brightness"], "vitality": auxiliary["saturation"],
                     "hue_shift": auxiliary["palette"]}
    projection = SemanticProjection(encode, config)
    training_texts = [row["text"] for row in training] + [f"a {word} photo" for word in TRAIN_UNRELATED]
    validation_texts = [row["text"] for row in validation]
    validation_axes = [row["axis"] for row in validation]
    for axis in axes:
        validation_texts.extend(auxiliary[axis])
        validation_axes.extend([axis] * len(auxiliary[axis]))
    validation_texts.extend(f"a {word} photo" for word in VALIDATION_UNRELATED)
    validation_axes.extend(["unrelated"] * len(VALIDATION_UNRELATED))
    with torch.no_grad():
        training_features = relevance_features(encode(training_texts), projection.centroids).cpu().numpy()
        validation_features = relevance_features(encode(validation_texts), projection.centroids).cpu().numpy()
    gate = {"feature_names": FEATURE_NAMES, "feature_mean": [], "feature_scale": [],
            "coefficients": [], "intercepts": [], "thresholds": [], "full_weight_at": [],
            "weight_rule": "smoothstep((score-lower)/(upper-lower)); calibrated upper allows full activation",
            "selection": "Validation objective (2*mean_positive_weight + 1-mean_negative_weight)/3",
            "caveat": "Weak axis-membership labels; D4 contributes to centroids; development calibration"}
    diagnostics = []
    for index, axis in enumerate(axes):
        labels = np.array([int(row["axis"] == axis) for row in training] + [0] * len(TRAIN_UNRELATED))
        valid_labels = np.array([int(value == axis) for value in validation_axes])
        scaler = StandardScaler().fit(training_features[:, index])
        classifier = LogisticRegression(C=1., class_weight="balanced", random_state=0, max_iter=1000)
        classifier.fit(scaler.transform(training_features[:, index]), labels)
        probabilities = classifier.predict_proba(scaler.transform(validation_features[:, index]))[:, 1]
        lower, upper, positive, negative = select_soft_band(probabilities, valid_labels)
        gate["feature_mean"].append(scaler.mean_.tolist())
        gate["feature_scale"].append(scaler.scale_.tolist())
        gate["coefficients"].append(classifier.coef_[0].tolist())
        gate["intercepts"].append(float(classifier.intercept_[0]))
        gate["thresholds"].append(lower)
        gate["full_weight_at"].append(upper)
        diagnostics.append({"axis": axis, "lower": lower, "full_weight_at": upper,
                            "validation_mean_relevant_weight": positive,
                            "validation_mean_irrelevant_weight": negative})
    split = {"classifier_training": training_texts, "validation_texts": validation_texts,
             "validation_axes": validation_axes, "diagnostics": diagnostics}
    return gate, split


def fit_signed(features, targets, signs, ridge):
    coefficients = np.zeros_like(signs, dtype=float)
    for control in range(6):
        selected = np.flatnonzero(signs[:, control])
        design = features[:, selected] * signs[selected, control]
        matrix = np.vstack((design, np.sqrt(ridge) * np.eye(len(selected))))
        response = np.concatenate((targets[:, control], np.zeros(len(selected))))
        result = lsq_linear(matrix, response, bounds=(0, np.inf), tol=1e-10)
        if not result.success:
            raise RuntimeError(result.message)
        coefficients[selected, control] = result.x * signs[selected, control]
    return coefficients


def refit_controls(config, medians, encode, signs):
    descriptors = descriptor_rows(config)
    projection = SemanticProjection(encode, config)
    _, coordinates = projection.project([row["text"] for row in descriptors])
    normalized = coordinates.cpu().numpy() / config["coordinate_limit"]
    features = np.concatenate((np.maximum(normalized, 0), np.maximum(-normalized, 0)), axis=1)
    target = np.asarray(medians) - np.asarray(config["learned_calibration"]["raw_neutral"])
    result = copy.deepcopy(config)
    coefficients = fit_signed(features, target, signs, config["learned_calibration"]["ridge"])
    result["learned_calibration"]["pole_raw_deltas"] = coefficients.tolist()
    result["learned_calibration"]["constraints"] = {"sign_matrix": signs.tolist(),
        "warning": "Explicit engineering restrictions; coefficient magnitudes fit to cosine-only optimization medians; unsupported magnitudes may be zero"}
    return result


def direct_quantile_calibration(config, records, quantile=.9):
    if not .5 < quantile < 1:
        raise ValueError("An extreme quantile must lie strictly between 0.5 and 1")
    axes = list(config["axes"])
    if len(axes) != len(CONTROL_NAMES):
        raise ValueError("Direct quantile calibration requires one axis per physical control")
    bounds = np.asarray(config.get("control_bounds",
        config.get("learned_calibration", {}).get("control_bounds", CONTROL_BOUNDS)), dtype=float)
    neutral = np.asarray(CONTROL_NEUTRAL, dtype=float)
    raw_neutral = np.asarray(config["learned_calibration"]["raw_neutral"], dtype=float)
    coefficients = np.zeros((2 * len(axes), len(CONTROL_NAMES)))
    centroid_texts = {axis: {} for axis in axes}
    centroid_descriptor_ids = {axis: {} for axis in axes}
    rows = []
    for column, axis in enumerate(axes):
        for side, direction in (("negative", -1), ("positive", 1)):
            samples = [record for record in records if record["axis"] == axis and record["sign"] == side]
            if not samples:
                raise ValueError(f"No tuning samples for {axis}/{side}")
            control_matrix = np.asarray([record["controls"] for record in samples], dtype=float)
            inactive = np.arange(len(CONTROL_NAMES)) != column
            if control_matrix.shape != (len(samples), len(CONTROL_NAMES)) or not np.isfinite(control_matrix).all():
                raise ValueError(f"Invalid physical tuning records for {axis}/{side}")
            if not np.array_equal(control_matrix[:, inactive], np.broadcast_to(
                    neutral[inactive], control_matrix[:, inactive].shape)):
                raise ValueError(f"Inactive controls moved in {axis}/{side}; single-DOF tuning is required")
            if any(record.get("active_controls", [CONTROL_NAMES[column]]) != [CONTROL_NAMES[column]]
                   for record in samples):
                raise ValueError(f"Incorrect active-control metadata for {axis}/{side}")
            values = control_matrix[:, column]
            if not np.isfinite(values).all() or (values < bounds[column, 0] - 1e-6).any() or (
                    values > bounds[column, 1] + 1e-6).any():
                raise ValueError(f"Invalid physical tuning values for {axis}/{side}")
            descriptor_samples = {}
            descriptor_text_by_id = {}
            for sample_index, (record, value) in enumerate(zip(samples, values)):
                descriptor_id = record.get("id", f"sample_{sample_index}")
                descriptor_samples.setdefault(descriptor_id, []).append(float(value))
                if "text" in record:
                    previous_text = descriptor_text_by_id.setdefault(descriptor_id, record["text"])
                    if previous_text != record["text"]:
                        raise ValueError(f"Inconsistent descriptor text for {axis}/{side}/{descriptor_id}")
            descriptor_ids = list(descriptor_samples)
            descriptor_medians = np.asarray([
                np.median(descriptor_samples[descriptor_id]) for descriptor_id in descriptor_ids])
            directional_descriptor_values = direction * (descriptor_medians - neutral[column])
            extreme = float(np.quantile(directional_descriptor_values, quantile))
            selected_count = max(1, int(np.ceil((1 - quantile) * len(descriptor_ids))))
            selected_indices = np.argsort(-directional_descriptor_values, kind="stable")[:selected_count]
            selected_ids = [descriptor_ids[index] for index in selected_indices]
            if all(descriptor_id in descriptor_text_by_id for descriptor_id in selected_ids):
                selected_texts = [descriptor_text_by_id[descriptor_id] for descriptor_id in selected_ids]
            elif "descriptor_table" in config:
                entries = config["descriptor_table"]["axes"][axis][side]
                text_by_id = {entry["id"]: entry["text"] for entry in entries}
                selected_texts = [text_by_id[descriptor_id] for descriptor_id in selected_ids
                                  if descriptor_id in text_by_id]
            else:
                selected_texts = []
            if not selected_texts:
                selected_texts = list(config["poles"][axis][side]["texts"])
            centroid_texts[axis][side] = selected_texts
            centroid_descriptor_ids[axis][side] = selected_ids
            setting = neutral[column] + direction * max(0., extreme)
            midpoint = bounds[column].mean()
            half_range = (bounds[column, 1] - bounds[column, 0]) / 2
            normalized = np.clip((setting - midpoint) / half_range, -1 + 1e-7, 1 - 1e-7)
            selected_raw = float(np.arctanh(normalized)) if extreme > 0 else raw_neutral[column]
            row_index = column + (len(axes) if side == "negative" else 0)
            coefficients[row_index, column] = direction * max(0., direction * (selected_raw - raw_neutral[column]))
            rows.append({"axis": axis, "sign": side, "control": CONTROL_NAMES[column],
                "sample_count": len(samples), "neutral_setting": neutral[column],
                "physical_percentile": quantile if direction > 0 else 1 - quantile,
                "q10": float(np.quantile(values, .1)), "median": float(np.median(values)),
                "q90": float(np.quantile(values, .9)),
                "descriptor_count": len(descriptor_ids),
                "selected_descriptor_count": len(selected_ids),
                "selected_descriptor_ids": " | ".join(selected_ids),
                "descriptor_median_q10": float(np.quantile(descriptor_medians, .1)),
                "descriptor_median": float(np.median(descriptor_medians)),
                "descriptor_median_q90": float(np.quantile(descriptor_medians, .9)),
                "selected_setting": setting,
                "selected_delta": setting - neutral[column],
                "supported_direction": extreme > 0})
    result = copy.deepcopy(config)
    result["status"] = "Direct physical pole endpoints from pooled directional quantiles; not human-validated"
    result["learned_calibration"].update({
        "pole_raw_deltas": coefficients.tolist(),
        "target": f"Direction-aligned {100 * quantile:g}th percentile of per-descriptor median physical controls per pole",
        "pole_centroid_texts": centroid_texts,
        "pole_centroid_descriptor_ids": centroid_descriptor_ids,
        "pole_statistic": {"method": "descriptor_median_directional_quantile", "quantile": quantile,
            "pool": "One median physical tuning value per descriptor, computed across calibration images",
            "centroid_selection": "Top ceil((1 - quantile) * descriptor_count) descriptors by direction-aligned median ISP tuning value",
            "centroid_embedding": "L2-normalize each selected CLIP text embedding, average, then L2-normalize the centroid",
            "autotune_controls": "Only the matching control is trainable; all other controls fixed at neutral",
            "negative_poles": f"Numerical {100 * (1 - quantile):g}th percentile; opposite-direction extremes fall back to neutral",
            "strength_fitting": "Direct endpoint assignment from descriptor-level medians; no ridge shrinkage"},
        "constraints": {"sign_matrix": np.concatenate((np.eye(len(axes)), -np.eye(len(axes)))).tolist(),
            "warning": "Only the matching physical control is assigned for each semantic pole"}})
    return result, rows


def recalibrate_semantic_mapping(config, descriptors, encode):
    projection = SemanticProjection(encode, config)
    raw_scores, _ = projection.project([row["text"] for row in descriptors])
    origin, _ = projection.project(["a photo"])
    recalibrated = fit_score_calibration(
        raw_scores.cpu().numpy(), origin[0].cpu().numpy(), config)
    endpoint_state = config["learned_calibration"]
    for key in ("pole_raw_deltas", "raw_neutral", "control_bounds", "pole_centroid_texts",
                "pole_centroid_descriptor_ids", "pole_statistic", "constraints", "training_split"):
        if key in endpoint_state:
            recalibrated["learned_calibration"][key] = endpoint_state[key]
    gate, split = fit_soft_gate(recalibrated, encode)
    recalibrated["relevance_gate"] = gate
    recalibrated["mapping_recalibration"] = {
        "text_score_calibration": "Recomputed using selected normalized pole centroids",
        "relevance_gate": "Recomputed using selected normalized pole centroids",
        "gate_diagnostics": split["diagnostics"],
    }
    return recalibrated, split


def directional_mean_endpoint(values, neutral, direction):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Directional mean requires a nonempty finite vector")
    if direction not in (-1, 1) or not np.isfinite(neutral):
        raise ValueError("Require a finite neutral and direction sign +/-1")
    displacement = direction * (values - neutral)
    return float(neutral + direction * np.maximum(displacement, 0).mean())


def image_sheet(path, images, titles, word):
    sheet = Image.new("RGB", (320 * len(images), 280), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (source, title) in enumerate(zip(images, titles)):
        with Image.open(source) as image:
            image.thumbnail((320, 235), Image.Resampling.LANCZOS)
            sheet.paste(image, (320 * index + (320 - image.width) // 2, 40 + (235 - image.height) // 2))
        draw.text((320 * index + 5, 6), title, fill="black")
        draw.text((320 * index + 5, 22), word, fill="black")
    sheet.save(path, quality=93)


def run(args):
    import open_clip
    torch.manual_seed(0)
    torch.set_num_threads(2)
    config = direct_config(args.table)
    axis_controls = dict(zip(config["axes"], CONTROL_NAMES))
    config["autotune_policy"] = {"axis_controls": axis_controls,
        "trainable_parameter_count": 1, "inactive_controls": "Fixed at neutral for every iteration"}
    descriptors = descriptor_rows(config)
    training_images = TRAIN_IMAGES[:1] if args.smoke else TRAIN_IMAGES
    comparison_images = TEST_IMAGES[:1] if args.smoke else TEST_IMAGES
    manifest = {"config": config, "iterations": args.iters, "images": training_images,
                "smoke": args.smoke, "pole_statistic": args.pole_statistic,
                "loss": "Negative CLIP cosine only, exact descriptor phrase; only the axis-matching control optimized",
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in (Path(__file__).name, "moran_semantic_isp.py", "exp38_learn_poles.py")}}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "experiment.json"
    normalized_manifest = json.loads(json.dumps(manifest))
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != normalized_manifest:
            raise ValueError("Existing run manifest differs")
    elif any(args.out.iterdir()):
        raise FileExistsError("Output is not empty and has no matching manifest")
    else:
        save_json(manifest_path, manifest)
    save_json(args.out / "descriptor_table.json", config["descriptor_table"])
    save_csv(args.out / "descriptor_table.csv", descriptors)
    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=args.device, force_quick_gelu=True)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])

    @torch.no_grad()
    def encode(texts):
        return model.encode_text(tokenizer(texts).to(args.device)).float()

    text_features = F.normalize(encode([row["text"] for row in descriptors]), dim=-1)
    records = []
    for image_id in training_images:
        crop = load_and_preprocess_raw(str(DATA / f"{image_id}.dng"), 512).to(args.device)
        folder = args.out / f"train_img_{image_id}"
        folder.mkdir(exist_ok=True)
        for index, row in enumerate(descriptors):
            stem = f"{row['axis']}_{row['sign']}_{row['id']}"
            path = folder / f"{stem}.json"
            if path.exists():
                record = json.loads(path.read_text())
            else:
                record, trace = optimize_controls(crop, text_features[index:index + 1], model,
                                                  config, args.iters, .03, active_control=axis_controls[row["axis"]])
                record.update(image=image_id, **row)
                save_csv(folder / f"{stem}_trace.csv", trace)
                save_json(path, record)
                print(image_id, stem, f"gain={record['best_similarity'] - record['initial_similarity']:+.5f}", flush=True)
            records.append(record)
        del crop
    medians = save_descriptor_stats(records, descriptors, args.out)
    projection = SemanticProjection(encode, config)
    raw_scores, _ = projection.project([row["text"] for row in descriptors])
    origin, _ = projection.project(["a photo"])
    if args.pole_statistic == "directional-q90":
        learned, extreme_statistics = direct_quantile_calibration(config, records)
        learned, direct_split = recalibrate_semantic_mapping(learned, descriptors, encode)
        save_csv(args.out / "direct_pole_quantile_statistics.csv", extreme_statistics)
    else:
        learned = fit_calibration(raw_scores.cpu().numpy(), origin[0].cpu().numpy(), medians, config)
    learned["training_split"] = {"images": training_images, "phrases": [row["text"] for row in descriptors]}
    direct_signs = np.concatenate((np.eye(6), -np.eye(6)), axis=0)
    direct_ungated = (copy.deepcopy(learned) if args.pole_statistic == "directional-q90"
                      else refit_controls(learned, medians, encode, direct_signs))
    direct_ungated.pop("relevance_gate", None)
    gate = (learned["relevance_gate"] if args.pole_statistic == "directional-q90"
            else fit_soft_gate(learned, encode)[0])
    direct_soft = copy.deepcopy(learned)
    direct_soft["relevance_gate"] = gate
    if args.pole_statistic != "directional-q90":
        direct_soft = refit_controls(direct_soft, medians, encode, direct_signs)
    direct_soft["status"] = f"Direct ISP descriptors; cosine-only tuning, {args.pole_statistic} pole strengths, soft relevance gate"

    cinema_root = RESULTS / "exp42_phrase_pole_clip_100"
    cinema = json.loads((cinema_root / "calibrated_poles.json").read_text())
    cinema_medians = []
    for row in descriptor_rows(cinema):
        samples = [json.loads((cinema_root / f"train_img_{image}" /
                    f"{row['axis']}_{row['sign']}_{row['id']}.json").read_text())["raw_controls"] for image in TRAIN_IMAGES]
        cinema_medians.append(np.median(samples, axis=0))
    cinema_gate, cinema_split = fit_soft_gate(cinema, encode)
    cinema_soft = copy.deepcopy(cinema)
    cinema_soft["relevance_gate"] = cinema_gate
    cinema_soft = refit_controls(cinema_soft, cinema_medians, encode, constraint_signs())
    cinema_strict = json.loads((RESULTS / "exp47b_relevance_gated_consistent_poles_report/calibrated_poles.json").read_text())
    versions = {"cinematic_ungated": cinema, "cinematic_strict": cinema_strict,
                "cinematic_soft": cinema_soft, "direct_ungated": direct_ungated, "direct_soft": direct_soft}
    for name, value in versions.items():
        save_json(args.out / f"{name}_calibration.json", value)
        axes = list(value.get("axes", AXES))
        projection = SemanticProjection(encode, value)
        torch.save(projection.centroids.cpu(), args.out / f"{name}_centroids.pt")
        save_csv(args.out / f"{name}_pole_controls.csv", [
            {"axis": axis, "sign": sign,
             **dict(zip(CONTROL_NAMES, bounded_controls(torch.tensor(
                 value["learned_calibration"]["raw_neutral"]) + torch.tensor(
                     value["learned_calibration"]["pole_raw_deltas"][index + (0 if sign == "positive" else len(axes))]
                 ), value.get("control_bounds", value["learned_calibration"].get("control_bounds", CONTROL_BOUNDS))).tolist()))}
            for index, axis in enumerate(axes) for sign in ("positive", "negative")])
    save_json(args.out / "gate_data_splits.json", {"direct": direct_split, "cinematic": cinema_split,
        "unrelated_train": TRAIN_UNRELATED, "unrelated_validation": VALIDATION_UNRELATED,
        "unrelated_test": TEST_UNRELATED,
        "caveat": "D4 excluded from classifier fitting but included in frozen centroids; development weak relevance labels"})

    corpus = json.loads(Path('/data/data1/semantic_isp/datasets/style_words/style_corpus.json').read_text())
    requests = [{"text": word, "group": "unseen_word", "direct_axis": "", "cinematic_axis": ""}
                for word in corpus["unseen_test_words"]]
    requests += [{"text": word, "group": "control_probe", "direct_axis": direct, "cinematic_axis": cinematic}
                 for word, direct, cinematic in DEVELOPMENT]
    requests += [{"text": word, "group": "unrelated", "direct_axis": "", "cinematic_axis": ""}
                 for word in TEST_UNRELATED]
    requests += [{"text": text, "group": "compound", "direct_axis": "", "cinematic_axis": ""}
                 for text in ("warm and desaturated", "bright and sharp", "dark high contrast", "cool and blurry")]
    prompts = [f"a {row['text']} photo" for row in requests]
    encoded = F.normalize(encode(prompts), dim=-1)
    evaluated = {}
    mapping_rows = []
    for name, value in versions.items():
        axes = list(value.get("axes", AXES))
        projection = SemanticProjection(encode, value)
        _, coordinates = projection.project_features(encoded)
        probabilities, weights = projection.relevance_weights(encoded)
        parameters = [compose_parameters(coordinate, value) for coordinate in coordinates]
        evaluated[name] = parameters
        for index, request in enumerate(requests):
            axis = request["direct_axis"] if name.startswith("direct") else request["cinematic_axis"]
            row = {**request, "version": name,
                   "max_activation": float(coordinates[index].abs().max()),
                   "expected_axis_gate": float(weights[index, axes.index(axis)]) if axis else None,
                   "axes_json": json.dumps(dict(zip(axes, coordinates[index].cpu().tolist()))),
                   "gates_json": json.dumps(dict(zip(axes, weights[index].cpu().tolist())))}
            mapping_rows.append(row)
    save_csv(args.out / "mapping_comparison.csv", mapping_rows)
    mapping_summary = {}
    for name in versions:
        probes = [row for row in mapping_rows if row["version"] == name and row["group"] == "control_probe"]
        unrelated = [row for row in mapping_rows if row["version"] == name and row["group"] == "unrelated"]
        mapping_summary[name] = {
            "probe_mean_expected_gate": float(np.mean([row["expected_axis_gate"] for row in probes])),
            "probe_weight_at_least_0_1": sum(row["expected_axis_gate"] >= .1 for row in probes),
            "probe_weight_at_least_0_5": sum(row["expected_axis_gate"] >= .5 for row in probes),
            "probe_count": len(probes),
            "unrelated_mean_max_activation": float(np.mean([row["max_activation"] for row in unrelated])),
            "unrelated_zero_all_axes": sum(row["max_activation"] == 0 for row in unrelated),
        }
    scores = []
    page = ["<!doctype html><meta charset='utf-8'><title>Direct ISP vs cinematic semantic mappings</title>",
            "<h1>Direct ISP axes vs cinematic axes</h1>",
            "<p>Same CLIP, renderer, physical-control bounds, image split and cosine-only objective. "
            f"Direct poles use {len(descriptors)} training phrases and {args.pole_statistic} strengths; cinematic uses 40 existing phrases. "
            "Direct phrase optimization trains only its matching physical control; all other controls stay neutral. "
            "Representation, phrase calibration and the direct strength statistic may differ. Development relevance probes are not an untouched benchmark.</p>",
            "<pre>" + html.escape(json.dumps(mapping_summary, indent=2)) + "</pre>"]
    with torch.no_grad():
        for image_id in comparison_images:
            native = load_and_preprocess_raw(str(DATA / f"{image_id}.dng"), None).to(args.device)
            crop = load_and_preprocess_raw(str(DATA / f"{image_id}.dng"), 512).to(args.device)
            folder = args.out / f"test_img_{image_id}"
            folder.mkdir(exist_ok=True)
            neutral = compose_parameters(torch.zeros(6, device=args.device), direct_soft)
            save_render(folder / "neutral.png", render_linear(native, neutral))
            neutral_feature = encode_image(model, render_linear(crop, neutral))
            for index, request in enumerate(requests):
                if args.smoke and index >= 3:
                    break
                row = {"image": image_id, **request,
                       "neutral_cosine": float((neutral_feature @ encoded[index:index+1].T).mean())}
                paths = [folder / "neutral.png"]
                for name, parameters in evaluated.items():
                    setting = parameters[index]
                    crop_output = render_linear(crop, setting)
                    row[name] = float((encode_image(model, crop_output) @ encoded[index:index+1].T).mean())
                    filename = folder / f"req_{index:03d}_{name}.png"
                    save_render(filename, render_linear(native, setting))
                    paths.append(filename)
                sheet = folder / f"req_{index:03d}_comparison.jpg"
                image_sheet(sheet, paths, ["Neutral", *versions], request["text"])
                scores.append(row)
                page.append(f"<h2>{html.escape(request['text'])}: image {image_id}</h2>"
                            f"<a href='test_img_{image_id}/{sheet.name}'><img style='max-width:100%' src='test_img_{image_id}/{sheet.name}'></a>")
            print(f"evaluated comparison image {image_id}: {len(requests)} requests, {len(versions)} versions", flush=True)
            del native, crop
    save_csv(args.out / "image_request_comparison.csv", scores)
    summary = {"optimization_runs": len(records), "descriptors": len(descriptors),
               "direct_pole_statistic": args.pole_statistic,
               "direct_autotune_trainable_parameters": 1,
               "iterations": args.iters, "comparison_requests": len(requests),
               "comparison_images": comparison_images, "mapping_metrics": mapping_summary,
               "rendered_comparison_pairs": len(scores), "smoke": args.smoke,
               "mean_training_cosine_gain": float(np.mean([row["best_similarity"] - row["initial_similarity"] for row in records])),
               "heldout_unseen_words": {}}
    unseen = [row for row in scores if row["group"] == "unseen_word"]
    for name in versions:
        summary["heldout_unseen_words"][name] = {
            "pairs": len(unseen), "mean_cosine": float(np.mean([row[name] for row in unseen])),
            "mean_gain_vs_neutral": float(np.mean([row[name] - row["neutral_cosine"] for row in unseen])),
            "positive_gain_pairs": sum(row[name] > row["neutral_cosine"] for row in unseen)}
    summary["direct_soft_vs_cinematic_soft_unseen_wins"] = sum(row["direct_soft"] > row["cinematic_soft"] for row in unseen)
    summary["caveat"] = "Same-CLIP self-evaluation; no human study. New descriptors, six axes, softer gates and constrained coefficients jointly change the method."
    save_json(args.out / "summary.json", summary)
    (args.out / "index.html").write_text("\n".join(page))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--pole-statistic", choices=("directional-q90", "median-fit"), default="directional-q90")
    parser.add_argument("--smoke", action="store_true", help="One training image and three rendered comparisons")
    args = parser.parse_args()
    if args.iters < 1:
        parser.error("--iters must be positive")
    run(args)


if __name__ == "__main__":
    main()