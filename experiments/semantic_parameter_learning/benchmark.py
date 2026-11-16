"""Compare text-only ISP regressors on disjoint sentence and image splits."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from autotune import CONTROL_NAMES
from semantic_isp import CONTROL_BOUNDS, CONTROL_NEUTRAL

DEFAULT_DATASET = Path("/data/data1/semantic_isp/results/semantic_parameter_learning/pilot_multistart_8steps")


def load_dataset(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    records = json.loads((directory / "dataset.json").read_text())["records"]
    with np.load(directory / "text_embeddings.npz", allow_pickle=False) as data:
        embeddings = data["embeddings"].astype(np.float64)
        ids = data["ids"].tolist()
    if len(set(ids)) != len(ids) or len(embeddings) != len(ids):
        raise ValueError("Invalid text-embedding IDs")
    by_id = dict(zip(ids, embeddings))
    pairs = [(row["sentence_id"], row["image_id"]) for row in records]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate sentence-image pairs")
    sentence_splits = {}
    image_splits = {}
    for row in records:
        if row["sentence_id"] not in by_id:
            raise ValueError(f"Missing text embedding: {row['sentence_id']}")
        if sentence_splits.setdefault(row["sentence_id"], row["sentence_split"]) != row["sentence_split"]:
            raise ValueError("Sentence ID occurs in multiple splits")
        if image_splits.setdefault(row["image_id"], row["image_split"]) != row["image_split"]:
            raise ValueError("Image ID occurs in multiple splits")
    features = np.stack([by_id[row["sentence_id"]] for row in records])
    physical_targets = np.asarray([[row["controls"][name] for name in CONTROL_NAMES] for row in records], dtype=float)
    if manifest.get("parameterization") == "p90":
        endpoints = manifest.get("pole_endpoints")
        if not endpoints or any("normalized_controls" not in row for row in records):
            raise ValueError("P90 dataset must contain pole endpoints and normalized controls for every record")
        targets = np.asarray([[row["normalized_controls"][name] for name in CONTROL_NAMES] for row in records], dtype=float)
        negative = np.asarray([endpoints[name]["negative"] for name in CONTROL_NAMES], dtype=float)
        neutral = np.asarray([endpoints[name]["neutral"] for name in CONTROL_NAMES], dtype=float)
        positive = np.asarray([endpoints[name]["positive"] for name in CONTROL_NAMES], dtype=float)
        if not (np.all(negative < neutral) and np.all(neutral < positive)):
            raise ValueError("P90 endpoints must strictly straddle neutral")
        difference = physical_targets - neutral
        expected = np.where(difference >= 0, difference / (positive - neutral), difference / (neutral - negative))
        if not np.all(np.abs(targets) <= 1 + 1e-6) or not np.allclose(targets, expected, atol=2e-6, rtol=0):
            raise ValueError("Normalized P90 labels disagree with their physical controls or exceed pole bounds")
        bounds = np.tile(np.asarray([[-1., 1.]]), (len(CONTROL_NAMES), 1))
    else:
        targets = physical_targets
        config = json.loads(Path(manifest["config"]).read_text())
        bounds = np.asarray(config.get("control_bounds", CONTROL_BOUNDS), dtype=float)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Nonfinite features or targets")
    if bounds.shape != (len(CONTROL_NAMES), 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Invalid ISP control bounds")
    return manifest, records, features, targets, bounds


def score(predictions, targets, half_ranges):
    absolute = np.abs(predictions - targets)
    return {"pairs": int(len(targets)), "normalized_mae": float(np.mean(absolute / half_ranges)),
            "mae_by_control": {name: float(value) for name, value in zip(CONTROL_NAMES, absolute.mean(axis=0))}}


def run(dataset, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest, records, features, targets, bounds = load_dataset(dataset)
    groups = {
        "train": np.asarray([row["sentence_split"] == "train" and row["image_split"] == "train" for row in records]),
        "validation_unseen_sentence": np.asarray([row["sentence_split"] == "validation" and row["image_split"] == "train" for row in records]),
        "test_unseen_sentence": np.asarray([row["sentence_split"] == "test" and row["image_split"] == "train" for row in records]),
        "test_unseen_image": np.asarray([row["sentence_split"] == "train" and row["image_split"] == "test" for row in records]),
        "test_unseen_sentence_and_image": np.asarray([row["sentence_split"] == "test" and row["image_split"] == "test" for row in records]),
    }
    if any(not mask.any() for mask in groups.values()):
        raise ValueError(f"All evaluation groups must be nonempty: {[name for name, mask in groups.items() if not mask.any()]}")
    training = groups["train"]
    target_scaler = StandardScaler().fit(targets[training])
    targets_scaled = target_scaler.transform(targets[training])
    models = {"ridge": Ridge(alpha=1.),
              "svr_rbf": MultiOutputRegressor(SVR(kernel="rbf", C=1., epsilon=.1))}
    for model in models.values():
        model.fit(features[training], targets_scaled)
    half_ranges = (bounds[:, 1] - bounds[:, 0]) / 2
    neutral = np.zeros(len(CONTROL_NAMES)) if manifest.get("parameterization") == "p90" else np.asarray(CONTROL_NEUTRAL, dtype=float)
    predictions = {
        "neutral": np.broadcast_to(neutral, targets.shape).copy(),
        "train_mean": np.broadcast_to(targets[training].mean(axis=0), targets.shape).copy(),
    }
    for name, model in models.items():
        values = target_scaler.inverse_transform(model.predict(features))
        predictions[name] = np.clip(values, bounds[:, 0], bounds[:, 1])
    metrics = {name: {group: score(values[mask], targets[mask], half_ranges)
                      for group, mask in groups.items()} for name, values in predictions.items()}
    selected = min(models, key=lambda name: metrics[name]["validation_unseen_sentence"]["normalized_mae"])
    output.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": str(dataset.resolve()), "sentence_image_training_pairs": int(training.sum()),
               "features": f"L2-normalized {manifest['semantic_model']} text embeddings only; image content is not a model input",
               "target_space": "p90_normalized" if manifest.get("parameterization") == "p90" else "physical",
               "control_order": list(CONTROL_NAMES), "model_selection": "Lowest validation unseen-sentence MAE on training images; fixed hyperparameters",
               "selected_model": selected, "metrics": metrics,
               "caveat": "Pilot labels are short-run optimization outcomes; scores measure agreement with those labels, not perceptual quality or generalization to natural language."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    with (output / "predictions.csv").open("w", newline="") as stream:
        fieldnames = ["model", "sentence_id", "sentence_split", "image_id", "image_split", "control", "target", "prediction"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name, values in predictions.items():
            for index, row in enumerate(records):
                for control_index, control in enumerate(CONTROL_NAMES):
                    writer.writerow({"model": name, "sentence_id": row["sentence_id"], "sentence_split": row["sentence_split"],
                                     "image_id": row["image_id"], "image_split": row["image_split"], "control": control,
                                     "target": targets[index, control_index], "prediction": values[index, control_index]})
    joblib.dump({"model": models[selected], "target_scaler": target_scaler, "bounds": bounds,
                 "control_names": CONTROL_NAMES, "semantic_model": manifest["semantic_model"],
                 "semantic_pretrained": manifest["semantic_pretrained"],
                 "target_space": summary["target_space"], "pole_endpoints": manifest.get("pole_endpoints"),
                 "text_normalization": "L2"}, output / "selected_model.joblib")
    print(json.dumps({"output": str(output), "selected_model": selected,
                      "validation_normalized_mae": {name: value["validation_unseen_sentence"]["normalized_mae"] for name, value in metrics.items()},
                      "test_unseen_both_normalized_mae": {name: value["test_unseen_sentence_and_image"]["normalized_mae"] for name, value in metrics.items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.out)


if __name__ == "__main__":
    main()