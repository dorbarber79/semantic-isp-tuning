"""Plot selected semantic pole centroids in a metric-MDS CLIP plane."""

import argparse
import csv
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from semantic_isp import SemanticProjection

DEFAULT_CONFIG = Path(
    "/data/data1/semantic_isp/results/exp81_joint_descriptor_tail_final/direct_soft_calibration.json")
DEFAULT_OUT = Path("/data/data1/semantic_isp/results/exp81_joint_descriptor_tail_final/latent_mds_final")
SIGN_COLORS = {"negative": "#167d9a", "positive": "#de6a32"}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path, output, axis, device, centroid_mode="selected"):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    config = json.loads(config_path.read_text())
    axes = tuple(config.get("axes", ()))
    if axis not in axes:
        raise ValueError(f"Unknown axis {axis!r}; expected one of {axes}")
    axis_index = axes.index(axis)
    if centroid_mode not in ("selected", "all"):
        raise ValueError(f"Unknown centroid mode: {centroid_mode}")
    if centroid_mode == "all":
        for sign in ("negative", "positive"):
            entries = config["descriptor_table"]["axes"][axis][sign]
            config["learned_calibration"]["pole_centroid_texts"][axis][sign] = [
                entry["text"] for entry in entries]
            config["learned_calibration"]["pole_centroid_descriptor_ids"][axis][sign] = [
                entry["id"] for entry in entries]
    selected_texts = config["learned_calibration"]["pole_centroid_texts"][axis]
    if any(not selected_texts[sign] for sign in ("negative", "positive")):
        raise ValueError("Expected descriptor texts for each pole")
    pole_labels = config["descriptor_table"]["axes"][axis]

    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=device,
        force_quick_gelu=config.get("force_quick_gelu", config["pretrained"] == "openai"))
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])

    @torch.no_grad()
    def encode_text(texts):
        return model.encode_text(tokenizer(texts).to(device)).float()

    projection = SemanticProjection(encode_text, config)
    descriptor_rows = [
        (axis, sign, descriptor)
        for sign in ("negative", "positive")
        for descriptor in config["descriptor_table"]["axes"][axis][sign]
    ]
    descriptor_texts = [descriptor["text"] for _, _, descriptor in descriptor_rows]
    with torch.no_grad():
        descriptor_features = F.normalize(encode_text(descriptor_texts), dim=-1)
    features_np = descriptor_features.cpu().numpy().astype(np.float64)
    centroid_features = projection.centroids[axis_index]
    centroid_np = centroid_features.cpu().numpy().astype(np.float64)
    all_features = np.concatenate((features_np, centroid_np), axis=0)
    input_distances = pairwise_distances(all_features, metric="euclidean")
    mds = MDS(n_components=2, metric_mds=True, n_init=8, init="random",
              max_iter=500, eps=1e-6, n_jobs=1, random_state=0,
              metric="euclidean", normalized_stress=False)
    all_coordinates = mds.fit_transform(all_features)
    descriptor_coordinates = all_coordinates[:len(features_np)]
    centroid_coordinates = all_coordinates[len(features_np):]
    upper = np.triu_indices(len(all_features), k=1)
    input_squared_distance = np.square(input_distances[upper]).sum()
    stress_1 = float(np.sqrt(mds.stress_ / input_squared_distance)) if input_squared_distance else 0.

    negative_index, positive_index = 0, 1
    negative_point = centroid_coordinates[negative_index]
    positive_point = centroid_coordinates[positive_index]
    original_distance = float(torch.linalg.vector_norm(
        projection.centroids[axis_index, 1] - projection.centroids[axis_index, 0]))
    projected_distance = float(np.linalg.norm(positive_point - negative_point))
    distance_retention = projected_distance / original_distance if original_distance else 0.
    selected_pairs = {
        (pole_axis, sign, text)
        for pole_axis in axes
        for sign in ("negative", "positive")
        for text in config["learned_calibration"]["pole_centroid_texts"][pole_axis][sign]
    }

    output.mkdir(parents=True, exist_ok=True)
    distance_to_negative = np.linalg.norm(features_np - centroid_np[negative_index], axis=1)
    distance_to_positive = np.linalg.norm(features_np - centroid_np[positive_index], axis=1)
    rows = []
    for index, (pole_axis, sign, descriptor) in enumerate(descriptor_rows):
        rows.append({"kind": "descriptor", "axis": pole_axis, "sign": sign,
            "descriptor_id": descriptor["id"], "selected_for_centroid":
                (pole_axis, sign, descriptor["text"]) in selected_pairs,
            "text": descriptor["text"], "mds1": descriptor_coordinates[index, 0],
            "mds2": descriptor_coordinates[index, 1],
            "distance_to_negative_pole_512d": float(distance_to_negative[index]),
            "distance_to_positive_pole_512d": float(distance_to_positive[index]),
            "nearest_pole_512d": ("negative" if distance_to_negative[index] < distance_to_positive[index]
                                  else "positive" if distance_to_positive[index] < distance_to_negative[index]
                                  else "tie")})
    for sign_index, sign in enumerate(("negative", "positive")):
            point = centroid_coordinates[sign_index]
            rows.append({"kind": "selected_centroid", "axis": axis, "sign": sign,
                "descriptor_id": " | ".join(config["learned_calibration"][
                    "pole_centroid_descriptor_ids"][axis][sign]),
                "selected_for_centroid": True,
                "text": " | ".join(config["learned_calibration"]["pole_centroid_texts"][axis][sign]),
                "mds1": point[0], "mds2": point[1],
                "distance_to_negative_pole_512d": 0. if sign == "negative" else original_distance,
                "distance_to_positive_pole_512d": original_distance if sign == "negative" else 0.,
                "nearest_pole_512d": sign})
    write_csv(output / "mds_coordinates.csv", rows)

    metadata = {
        "axis": axis,
        "calibration": str(config_path),
        "model": config["model"],
        "input_dimension": int(features_np.shape[1]),
        "descriptor_fit_points": int(features_np.shape[0]),
        "centroid_fit_points": int(centroid_np.shape[0]),
        "fit_points": int(all_features.shape[0]),
        "method": "metric multidimensional scaling",
        "dissimilarity": "Euclidean distance in normalized 512D CLIP embedding space",
        "initializations": 8,
        "max_iterations": 500,
        "random_state": 0,
        "input_vectors": "L2-normalized CLIP text embeddings",
        "centroid_rule": "L2-normalized selected descriptor embeddings averaged, then L2-normalized",
        "centroid_mode": centroid_mode,
        "centroid_descriptor_counts": {sign: len(selected_texts[sign]) for sign in ("negative", "positive")},
        "assignment_evaluation": "In-sample nearest centroid using original 512D Euclidean distances; each phrase contributes to its own centroid when centroid_mode is all",
        "assignment_counts": {sign: {
            "correct": sum(row["sign"] == sign and row["nearest_pole_512d"] == sign for row in rows if row["kind"] == "descriptor"),
            "incorrect": sum(row["sign"] == sign and row["nearest_pole_512d"] not in (sign, "tie") for row in rows if row["kind"] == "descriptor"),
            "ties": sum(row["sign"] == sign and row["nearest_pole_512d"] == "tie" for row in rows if row["kind"] == "descriptor")
        } for sign in ("negative", "positive")},
        "mapping_basis": f"MDS fit over only the {axis} descriptors and its two selected pole centroids",
        "aspect": "Equal x/y units; no axis-specific scaling",
        "raw_stress": float(mds.stress_),
        "stress_1": stress_1,
        "selected_centroid_descriptor_ids": config["learned_calibration"][
            "pole_centroid_descriptor_ids"][axis],
        "negative_pole_label": pole_labels["negative_pole"],
        "positive_pole_label": pole_labels["positive_pole"],
        "original_pole_distance": original_distance,
        "projected_pole_distance": projected_distance,
        "pole_distance_retention": distance_retention,
    }
    (output / "mds_metadata.json").write_text(json.dumps(metadata, indent=2))

    selected_axis_indices = {sign: [index for index, (_, sign_value, _) in enumerate(descriptor_rows)
                                    if sign_value == sign]
                             for sign in ("negative", "positive")}
    selected_phrase_mask = {
        sign: [index for index in selected_axis_indices[sign]
               if (axis, sign, descriptor_rows[index][2]["text"]) in selected_pairs]
        for sign in ("negative", "positive")
    }
    descriptor_distances = {
        (row["sign"], row["descriptor_id"]): row
        for row in rows if row["kind"] == "descriptor"
    }
    fig = plt.figure(figsize=(14, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(3.1, 1.9), wspace=.04)
    ax = fig.add_subplot(grid[0, 0])
    info_ax = fig.add_subplot(grid[0, 1])
    info_ax.axis("off")
    for sign in ("negative", "positive"):
        indices = selected_axis_indices[sign]
        color = SIGN_COLORS[sign]
        ax.scatter(descriptor_coordinates[indices, 0], descriptor_coordinates[indices, 1],
                   s=34, color=color, alpha=.68, edgecolors="white", linewidths=.35,
                   label=f"{axis.title()} {sign} descriptors")
        chosen = selected_phrase_mask[sign]
        ax.scatter(descriptor_coordinates[chosen, 0], descriptor_coordinates[chosen, 1],
                   s=86, marker="D", facecolors="none", edgecolors=color, linewidths=1.5,
                   label=f"Centroid members ({sign})")
        for index in indices:
            descriptor_id = descriptor_rows[index][2]["id"]
            label = ax.annotate(descriptor_id, descriptor_coordinates[index],
                xytext=(3, 2), textcoords="offset points", fontsize=6.5,
                color="#17212b", zorder=6)
            label.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground="white")])
    ax.plot([negative_point[0], positive_point[0]], [negative_point[1], positive_point[1]],
            color="#313c45", linewidth=1.3, linestyle="--", zorder=2)
    for sign, index, point in (("negative", negative_index, negative_point),
                               ("positive", positive_index, positive_point)):
        color = SIGN_COLORS[sign]
        ax.scatter(point[0], point[1], marker="*", s=280, color=color,
                   edgecolors="#17212b", linewidths=.8, zorder=5,
                   label=f"{axis.title()} {sign} centroid")
    info_ax.text(.02, .98, f"{axis.title()} descriptor distances", va="top",
                 fontsize=12, fontweight="bold", color="#17212b",
                 transform=info_ax.transAxes)
    for sign, y, group_name in (("negative", .91, "Cool descriptors"),
                                ("positive", .47, "Warm descriptors")):
        color = SIGN_COLORS[sign]
        info_ax.text(.02, y,
                     f"{group_name} | d(cool) / d(warm) (512D)", va="top",
                     fontsize=8.2, fontweight="bold", color=color,
                     transform=info_ax.transAxes)
        distance_lines = []
        for index in selected_axis_indices[sign]:
            _, _, descriptor = descriptor_rows[index]
            row = descriptor_distances[(sign, descriptor["id"])]
            selected_marker = "*" if row["selected_for_centroid"] else " "
            short_text = textwrap.shorten(descriptor["text"], width=35, placeholder="...")
            distance_lines.append(
                f"{selected_marker}{descriptor['id']:4} "
                f"{row['distance_to_negative_pole_512d']:.2f} / "
                f"{row['distance_to_positive_pole_512d']:.2f}  {short_text}")
        info_ax.text(.02, y - .035, "\n".join(distance_lines), va="top",
                     fontsize=6.3, family="DejaVu Sans Mono", color="#26323a",
                     linespacing=1.5, transform=info_ax.transAxes)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("MDS 1 (CLIP Euclidean distance units)")
    ax.set_ylabel("MDS 2 (CLIP Euclidean distance units)")
    ax.set_title(f"{axis.title()} poles in normalized CLIP descriptor space ({centroid_mode} descriptors)\n"
                 f"Metric MDS stress-1 {stress_1:.3f}; pole-distance retention {distance_retention * 100:.1f}%\n"
                 "Every descriptor: distance to cool pole / distance to warm pole (512D CLIP)")
    ax.grid(color="#dfe4e7", linewidth=.6, alpha=.8)
    ax.legend(loc="best", fontsize=8, frameon=True, framealpha=.95, ncol=2)
    fig.savefig(output / f"{axis}_poles_mds.png", dpi=200, facecolor="white")
    plt.close(fig)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    print(json.dumps({"output": str(output), "figure": str(output / f"{axis}_poles_mds.png"),
        "axis": axis, "stress_1": stress_1,
        "original_pole_distance": original_distance, "projected_pole_distance": projected_distance,
        "distance_retention": distance_retention}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--axis", default="warmth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--centroid-mode", choices=("selected", "all"), default="selected",
                        help="Use the calibrated selected descriptors or all descriptors per pole")
    args = parser.parse_args()
    run(args.config, args.out, args.axis, args.device, args.centroid_mode)


if __name__ == "__main__":
    main()
