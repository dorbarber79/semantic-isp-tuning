"""Calibrate independent axis rejection gates and sign-constrained ISP pole mappings."""

import argparse
import copy
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import lsq_linear
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from differentiable_isp import load_and_preprocess_raw
from exp38_learn_poles import encode_image, make_pair_sheet, save_csv, save_json, save_render
from exp42_phrase_pole_calibration import descriptor_rows
from semantic_isp import (
    AXES, CONTROL_NAMES, SemanticProjection, bounded_controls, compose_parameters,
    neutral_raw_controls, relevance_features, render_linear,
)


TRAIN_UNRELATED = ["invoice", "algebra", "calendar", "keyboard", "contract", "parliament",
                   "bicycle", "bread", "telephone", "airport", "dictionary", "wallet",
                   "spoon", "mountain", "bottle", "notebook", "insurance", "software",
                   "pencil", "agriculture"]
VALIDATION_UNRELATED = ["taxation", "elevator", "microscope", "cucumber", "accounting",
                        "railway", "passport", "equation", "furniture", "chemistry"]
TEST_UNRELATED = ["calculator", "democracy", "sandwich", "spreadsheet", "constitution",
                  "toothbrush", "submarine", "arithmetic", "teaspoon", "bureaucracy"]
TEST_RELEVANT = {
    "thermal": ["warm", "cold", "warmer white balance", "cooler white balance"],
    "fidelity": ["sharp", "blurry", "crisp fine details", "soft focus"],
    "moral": ["bright", "dark", "high key", "chiaroscuro"],
    "vitality": ["vivid", "monochrome", "strong saturation", "desaturated colors"],
    "hue_shift": ["retro", "blockbuster", "magenta green palette", "teal orange palette"],
}
FEATURE_NAMES = ["mean_pole_cosine", "max_pole_cosine", "mean_vs_other_axis_max",
                 "perpendicular_line_distance", "absolute_pole_difference"]


def constraint_signs():
    signs = np.zeros((10, 6))
    for row, col, sign in ((0, 0, 1), (5, 0, -1),
                           (1, 3, 1), (1, 4, 1), (6, 3, -1), (6, 4, -1),
                           (2, 2, 1), (2, 3, -1), (7, 2, -1), (7, 3, 1),
                           (3, 1, 1), (8, 1, -1), (4, 5, 1), (9, 5, -1)):
        signs[row, col] = sign
    return signs


def constrained_coefficients(features, target, ridge):
    features, target = np.asarray(features), np.asarray(target)
    signs = constraint_signs()
    coefficients = np.zeros((10, 6))
    for control in range(6):
        selected = np.flatnonzero(signs[:, control])
        matrix = features[:, selected] * signs[selected, control]
        augmented = np.vstack((matrix, np.sqrt(ridge) * np.eye(len(selected))))
        response = np.concatenate((target[:, control], np.zeros(len(selected))))
        result = lsq_linear(augmented, response, bounds=(0, np.inf), tol=1e-10)
        if not result.success:
            raise RuntimeError(result.message)
        coefficients[selected, control] = result.x * signs[selected, control]
    return coefficients


def select_threshold(probabilities, labels):
    candidates = np.unique(np.concatenate(([.01], probabilities, [1 - 1e-6])))
    choices = []
    for threshold in candidates:
        active = probabilities > threshold
        false_positive = float(active[labels == 0].mean())
        recall = float(active[labels == 1].mean())
        if false_positive <= .05:
            choices.append((recall, -threshold, threshold, false_positive))
    if not choices:
        raise ValueError("No valid threshold")
    recall, _, threshold, false_positive = max(choices)
    return float(threshold), recall, false_positive


def run(args):
    import open_clip
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    root = args.source
    original = json.loads((root / "calibrated_poles.json").read_text())
    model, _, _ = open_clip.create_model_and_transforms(
        original["model"], pretrained=original["pretrained"], device=args.device,
        force_quick_gelu=True)
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(original["model"])

    @torch.no_grad()
    def encode(texts):
        return model.encode_text(tokenizer(texts).to(args.device)).float()

    base = SemanticProjection(encode, original)
    descriptors = descriptor_rows(original)
    phrases = [row["text"] for row in descriptors]
    train_descriptors = [row for row in descriptors if row["id"] != "D4"]
    validation_descriptors = [row for row in descriptors if row["id"] == "D4"]
    train_texts = [row["text"] for row in train_descriptors] + [f"a {word} photo" for word in TRAIN_UNRELATED]
    validation_texts = [row["text"] for row in validation_descriptors] + [f"a {word} photo" for word in VALIDATION_UNRELATED]
    with torch.no_grad():
        train_measurements = relevance_features(encode(train_texts), base.centroids).cpu().numpy()
        val_measurements = relevance_features(encode(validation_texts), base.centroids).cpu().numpy()
    gate = {"feature_names": FEATURE_NAMES, "feature_mean": [], "feature_scale": [],
            "coefficients": [], "intercepts": [], "thresholds": [],
            "weight_rule": "smoothstep((probability-threshold)/(1-threshold)); zero below threshold",
            "labels": "weak axis labels from descriptor membership; unrelated nouns labeled negative",
            "centroid_caveat": "D4 excluded from classifier fitting but already part of frozen pole centroids",
            "threshold_rule": "Maximum relevant recall subject to validation false-positive rate <= 5%"}
    validation_report = []
    for index, axis in enumerate(AXES):
        labels = np.array([int(row["axis"] == axis) for row in train_descriptors] + [0] * len(TRAIN_UNRELATED))
        val_labels = np.array([int(row["axis"] == axis) for row in validation_descriptors] + [0] * len(VALIDATION_UNRELATED))
        scaler = StandardScaler().fit(train_measurements[:, index])
        classifier = LogisticRegression(C=1., class_weight="balanced", random_state=0, max_iter=1000)
        classifier.fit(scaler.transform(train_measurements[:, index]), labels)
        probabilities = classifier.predict_proba(scaler.transform(val_measurements[:, index]))[:, 1]
        threshold, recall, false_positive = select_threshold(probabilities, val_labels)
        gate["feature_mean"].append(scaler.mean_.tolist())
        gate["feature_scale"].append(scaler.scale_.tolist())
        gate["coefficients"].append(classifier.coef_[0].tolist())
        gate["intercepts"].append(float(classifier.intercept_[0]))
        gate["thresholds"].append(threshold)
        validation_report.append({"axis": axis, "threshold": threshold,
                                  "validation_recall": recall, "validation_false_positive_rate": false_positive})

    gated_only = copy.deepcopy(original)
    gated_only["relevance_gate"] = gate
    gated_projection = SemanticProjection(encode, gated_only)
    _, phrase_coordinates = gated_projection.project(phrases)
    normalized = phrase_coordinates.cpu().numpy() / original["coordinate_limit"]
    design = np.concatenate((np.maximum(normalized, 0), np.maximum(-normalized, 0)), axis=1)
    medians = []
    for descriptor in descriptors:
        values = []
        for image in original["training_split"]["images"]:
            filename = f"{descriptor['axis']}_{descriptor['sign']}_{descriptor['id']}.json"
            record = json.loads((root / f"train_img_{image}" / filename).read_text())
            values.append(record["raw_controls"])
        medians.append(np.median(values, axis=0))
    target = np.array(medians) - np.array(original["learned_calibration"]["raw_neutral"])
    coefficients = constrained_coefficients(design, target, original["learned_calibration"]["ridge"])
    calibrated = copy.deepcopy(gated_only)
    calibrated["learned_calibration"]["pole_raw_deltas"] = coefficients.tolist()
    calibrated["learned_calibration"]["constraints"] = {
        "sign_matrix": constraint_signs().tolist(),
        "status": "Explicit engineering restrictions; magnitudes fitted to Exp42 medians with bounded least squares",
        "retained_score_scales": "Exp42 signed distance scales unchanged",
        "fit_warning": "No minimum effect imposed: unsupported poles can have zero coefficients"}
    calibrated["status"] = "Relevance-gated and sign-constrained development calibration"
    args.out.mkdir(parents=True)
    save_json(args.out / "calibrated_poles.json", calibrated)
    save_json(args.out / "gate_only.json", gated_only)
    save_csv(args.out / "gate_validation.csv", validation_report)
    save_csv(args.out / "constrained_pole_coefficients.csv", [
        {"pole": name, **dict(zip(CONTROL_NAMES, row.tolist()))}
        for name, row in zip(original["learned_calibration"]["feature_order"], coefficients)])
    all_tests = [(text, axis) for axis, words in TEST_RELEVANT.items() for text in words]
    all_tests += [(word, "unrelated") for word in TEST_UNRELATED]
    test_prompts = [f"a {text} photo" for text, _ in all_tests]
    test_features = F.normalize(encode(test_prompts), dim=-1)
    _, before = base.project_features(test_features)
    _, after = gated_projection.project_features(test_features)
    probabilities, weights = gated_projection.relevance_weights(test_features)
    rows = []
    for index, (text, expected) in enumerate(all_tests):
        row = {"request": text, "expected_axis": expected,
               "old_max_activation": float(before[index].abs().max()),
               "gated_max_activation": float(after[index].abs().max())}
        for axis_index, axis in enumerate(AXES):
            row.update({f"old_{axis}": float(before[index, axis_index]),
                        f"gated_{axis}": float(after[index, axis_index]),
                        f"probability_{axis}": float(probabilities[index, axis_index]),
                        f"gate_{axis}": float(weights[index, axis_index])})
        rows.append(row)
    save_csv(args.out / "heldout_axis_mapping.csv", rows)
    unrelated_rows = [row for row in rows if row["expected_axis"] == "unrelated"]
    relevant_rows = [row for row in rows if row["expected_axis"] != "unrelated"]
    summary = {
        "test_unrelated_requests": len(unrelated_rows),
        "unrelated_zero_all_axes": sum(row["gated_max_activation"] == 0 for row in unrelated_rows),
        "unrelated_mean_old_max_activation": float(np.mean([row["old_max_activation"] for row in unrelated_rows])),
        "unrelated_mean_gated_max_activation": float(np.mean([row["gated_max_activation"] for row in unrelated_rows])),
        "test_relevant_requests": len(relevant_rows),
        "relevant_expected_axis_retained": sum(row[f"gate_{row['expected_axis']}"] > 0 for row in relevant_rows),
        "relevant_expected_axis_weight_at_least_0_1": sum(
            row[f"gate_{row['expected_axis']}"] >= .1 for row in relevant_rows),
        "relevant_mean_expected_axis_weight": float(np.mean([
            row[f"gate_{row['expected_axis']}"] for row in relevant_rows])),
        "weak_label_caveat": "Other-axis descriptors are treated as negatives although multi-axis phrases can be relevant to several axes",
        "test_caveat": "Development prompts, no human relevance study; previous vocabulary overlaps, unrelated split is disjoint"}
    save_json(args.out / "summary.json", summary)
    save_json(args.out / "data_splits.json", {
        "training_descriptors": train_descriptors, "validation_descriptors": validation_descriptors,
        "training_unrelated": TRAIN_UNRELATED, "validation_unrelated": VALIDATION_UNRELATED,
        "test_requests": all_tests, "validation_report": validation_report})
    figure, axes = plt.subplots(1, 2, figsize=(15, 9))
    for ax, values, title in ((axes[0], before.cpu().numpy(), "Previous activation"),
                               (axes[1], after.cpu().numpy(), "Relevance-gated activation")):
        image = ax.imshow(values, vmin=-.4, vmax=.4, cmap="coolwarm", aspect="auto")
        ax.set_yticks(range(len(all_tests)), [text for text, _ in all_tests], fontsize=8)
        ax.set_xticks(range(5), AXES, rotation=35, ha="right")
        ax.set_title(title)
        figure.colorbar(image, ax=ax)
    figure.tight_layout()
    figure.savefig(args.out / "activation_comparison.png", dpi=180)
    plt.close(figure)

    examples = ["warm", "cold", "sharp", "blurry", "bright", "dark", "vivid",
                "monochrome", "retro", "blockbuster", "calculator", "sandwich"]
    indices = [next(index for index, (text, _) in enumerate(all_tests) if text == word) for word in examples]
    configurations = {"previous": original, "gate_only": gated_only, "full": calibrated}
    render_rows = []
    page = ["<!doctype html><meta charset='utf-8'><title>Axis relevance and control constraints</title>",
            "<h1>Relevance gating and control-consistent poles</h1>",
            "<p>New calibration only; previous mapping preserved. Weak semantic labels and development prompts, "
            "not independent relevance or human-preference validation.</p>",
            "<pre>" + html.escape(json.dumps(summary, indent=2)) + "</pre>",
            "<img style='max-width:100%' src='activation_comparison.png'>"]
    torch.set_num_threads(2)
    with torch.no_grad():
        for image_id in (2, 33):
            path = Path(f"/data/data1/EmotionISP_dataset/dng_filtered/{image_id}.dng")
            native = load_and_preprocess_raw(str(path), None).to(args.device)
            crop = load_and_preprocess_raw(str(path), 512).to(args.device)
            directory = args.out / f"img_{image_id}"
            directory.mkdir()
            neutral = compose_parameters(torch.zeros(5, device=args.device), original)
            save_render(directory / "neutral.png", render_linear(native, neutral))
            for index in indices:
                text, _ = all_tests[index]
                row = {"image": image_id, "request": text}
                for name, config in configurations.items():
                    coordinate = before[index] if name == "previous" else after[index]
                    parameters = compose_parameters(coordinate, config)
                    output = render_linear(native, parameters)
                    save_render(directory / f"{text}_{name}.png", output)
                    feature = encode_image(model, render_linear(crop, parameters))
                    row[f"{name}_cosine"] = float((feature @ test_features[index:index + 1].T).mean())
                make_pair_sheet(directory / f"{text}_comparison.jpg", directory / "neutral.png",
                                directory / f"{text}_previous.png", directory / f"{text}_full.png", text)
                render_rows.append(row)
                page.append(f"<h2>{html.escape(text)}: image {image_id}</h2>"
                            f"<img style='max-width:100%' src='img_{image_id}/{text}_comparison.jpg'>")
            print(f"rendered relevance comparison image {image_id}", flush=True)
    save_csv(args.out / "render_scores.csv", render_rows)
    (args.out / "index.html").write_text("\n".join(page))
    print(json.dumps(summary, indent=2))
    print("validation", validation_report)
    print("constrained coefficients", coefficients.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "/data/data1/semantic_isp/results/exp42_phrase_pole_clip_100"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()