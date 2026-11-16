"""Visualize a linear pole classifier in a boundary-aligned CLIP plane."""

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import LinearSVC


DEFAULT_CONFIG = Path("/data/data1/semantic_isp/results/exp81_joint_descriptor_tail_final/direct_soft_calibration.json")
DEFAULT_OUT = Path("/data/data1/semantic_isp/results/exp81_joint_descriptor_tail_final/warmth_svm")
COLORS = {"negative": "#167d9a", "positive": "#de6a32"}


def fit_classifier(features, labels, regularization):
    classifier = LinearSVC(C=regularization, dual="auto", random_state=0, max_iter=10000)
    classifier.fit(features, labels)
    return classifier


def run(config_path, output, axis, device, regularization):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("SVM C must be finite and positive")
    config = json.loads(config_path.read_text())
    if axis not in config["axes"]:
        raise ValueError(f"Unknown axis {axis!r}")
    descriptor_bank = config["descriptor_table"]["axes"][axis]
    descriptors = [(sign, entry) for sign in ("negative", "positive")
                   for entry in descriptor_bank[sign]]
    if any(sum(side == sign for side, _ in descriptors) < 2 for sign in ("negative", "positive")):
        raise ValueError("At least two descriptors are needed per pole")

    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=device,
        force_quick_gelu=config.get("force_quick_gelu", config["pretrained"] == "openai"))
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])
    with torch.no_grad():
        tokens = tokenizer([entry["text"] for _, entry in descriptors]).to(device)
        features = F.normalize(model.encode_text(tokens).float(), dim=-1).cpu().numpy().astype(np.float64)
    labels = np.asarray([1 if sign == "positive" else -1 for sign, _ in descriptors])
    classifier = fit_classifier(features, labels, regularization)
    normal = classifier.coef_[0]
    normal_length = float(np.linalg.norm(normal))
    if normal_length == 0:
        raise ValueError("SVM produced a zero boundary normal")
    direction = normal / normal_length
    distances = classifier.decision_function(features) / normal_length
    centered = features - features.mean(axis=0)
    perpendicular = centered - np.outer(centered @ direction, direction)
    _, singular_values, right_vectors = np.linalg.svd(perpendicular, full_matrices=False)
    if singular_values[0] <= 1e-12:
        raise ValueError("No variance orthogonal to the SVM boundary")
    second_direction = right_vectors[0]
    vertical = centered @ second_direction
    np.testing.assert_allclose(np.dot(direction, second_direction), 0., atol=1e-10)
    np.testing.assert_allclose(distances * normal_length, classifier.decision_function(features), atol=1e-10)

    loo_predictions = np.empty_like(labels)
    loo_distances = np.empty(len(labels), dtype=np.float64)
    for training_indices, test_indices in LeaveOneOut().split(features):
        folded = fit_classifier(features[training_indices], labels[training_indices], regularization)
        loo_predictions[test_indices] = folded.predict(features[test_indices])
        loo_distances[test_indices] = folded.decision_function(features[test_indices]) / np.linalg.norm(folded.coef_[0])
    fitted_predictions = classifier.predict(features)
    rows = []
    for index, (sign, descriptor) in enumerate(descriptors):
        rows.append({"axis": axis, "sign": sign, "descriptor_id": descriptor["id"],
                     "text": descriptor["text"], "svm_x_signed_distance_512d": float(distances[index]),
                     "orthogonal_pca_coordinate": float(vertical[index]),
                     "train_prediction": "positive" if fitted_predictions[index] == 1 else "negative",
                     "loo_prediction": "positive" if loo_predictions[index] == 1 else "negative",
                     "loo_signed_distance_512d": float(loo_distances[index]),
                     "loo_correct": bool(loo_predictions[index] == labels[index])})

    output.mkdir(parents=True, exist_ok=True)
    with (output / "svm_descriptors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train_correct = int(np.sum(fitted_predictions == labels))
    loo_correct = int(np.sum(loo_predictions == labels))
    metadata = {"axis": axis, "calibration": str(config_path), "model": config["model"],
                "model_type": "linear SVM", "C": regularization,
                "embedding_normalization": "L2-normalized CLIP text embeddings",
                "descriptor_count": len(descriptors), "training_correct": train_correct,
                "leave_one_out_correct": loo_correct,
                "leave_one_out_mismatches": [row for row in rows if not row["loo_correct"]],
                "boundary_view": "x=(w dot x+b)/||w||; y=first principal direction orthogonal to w; x=0 is the exact 512D SVM boundary",
                "caveat": "In-sample boundary and leave-one-out predictions concern authored descriptor phrases; no independent unseen-word or image evaluation."}
    (output / "svm_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(3, 2))
    ax = fig.add_subplot(grid[0, 0])
    info_ax = fig.add_subplot(grid[0, 1])
    info_ax.axis("off")
    for sign, label in (("negative", -1), ("positive", 1)):
        mask = labels == label
        ax.scatter(distances[mask], vertical[mask], s=40, color=COLORS[sign],
                   edgecolors="white", linewidths=.5, label=f"{sign.title()} descriptors")
        for index in np.flatnonzero(mask):
            marker = "-" if sign == "negative" else "+"
            annotation = ax.annotate(f"{marker}{descriptors[index][1]['id']}",
                (distances[index], vertical[index]), xytext=(3, 3), textcoords="offset points",
                fontsize=6.5, color="#17212b")
            annotation.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground="white")])
    incorrect = loo_predictions != labels
    if incorrect.any():
        ax.scatter(distances[incorrect], vertical[incorrect], s=120, facecolors="none",
                   edgecolors="#bb2638", linewidths=1.5, label="Leave-one-out error")
    ax.axvline(0, color="#303a42", linestyle="--", linewidth=1.4, label="SVM boundary")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Signed distance to SVM boundary in 512D (negative = cool)")
    ax.set_ylabel("Orthogonal principal direction (CLIP units)")
    ax.set_title(f"{axis.title()} descriptors: linear SVM on 512D CLIP embeddings\n"
                 f"Training {train_correct}/{len(rows)} | Leave-one-out {loo_correct}/{len(rows)} | C={regularization:g}")
    ax.grid(color="#dfe4e7", linewidth=.6)
    ax.legend(loc="best", fontsize=8)
    info_ax.text(.01, .98, "Descriptor | signed distance | LOO prediction", va="top",
                 fontsize=10, fontweight="bold", transform=info_ax.transAxes)
    for sign, ypos in (("negative", .91), ("positive", .47)):
        info_ax.text(.01, ypos, f"{sign.title()} descriptors", va="top", fontsize=9,
                     fontweight="bold", color=COLORS[sign], transform=info_ax.transAxes)
        lines = []
        for row in rows:
            if row["sign"] != sign:
                continue
            flag = "!" if not row["loo_correct"] else " "
            phrase = textwrap.shorten(row["text"], width=42, placeholder="...")
            lines.append(f"{flag}{row['descriptor_id']:4} {row['svm_x_signed_distance_512d']:+.3f} "
                         f"{row['loo_prediction'][:3]:3}  {phrase}")
        info_ax.text(.01, ypos - .032, "\n".join(lines), va="top", fontsize=6.5,
                     family="DejaVu Sans Mono", linespacing=1.5, color="#26323a",
                     transform=info_ax.transAxes)
    info_ax.text(.01, .02, "! = leave-one-out error. Signed distances use the full-data SVM;\n"
                 "LOO predictions use a separately fitted SVM for each held-out phrase.",
                 fontsize=8, transform=info_ax.transAxes, va="bottom")
    fig.savefig(output / f"{axis}_svm.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(json.dumps({"output": str(output), "figure": str(output / f"{axis}_svm.png"),
                      "training_correct": train_correct, "leave_one_out_correct": loo_correct,
                      "descriptors": len(rows)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--axis", default="warmth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()
    run(args.config, args.out, args.axis, args.device, args.C)


if __name__ == "__main__":
    main()