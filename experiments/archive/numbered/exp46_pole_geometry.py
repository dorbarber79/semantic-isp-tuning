"""Diagnose pole preference versus axis relevance without changing the ISP mapping."""

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from moran_semantic_isp import AXES, SemanticProjection


WORDS = ["warm", "cold", "sharp", "blurry", "bright", "dark", "vivid",
         "monochrome", "retro", "blockbuster", "calculator", "democracy", "sandwich"]
CONTROLS = {"calculator", "democracy", "sandwich"}
POLES = {"thermal": ("Cyan", "Golden"), "fidelity": ("Ethereal", "Documentary"),
         "moral": ("Chiaroscuro", "High key"), "vitality": ("Bleached", "Vivid"),
         "hue_shift": ("Magenta-Green", "Teal-Orange")}


def axis_geometry(features, negative, positive):
    features = np.asarray(features, dtype=np.float64)
    negative, positive = np.asarray(negative, dtype=np.float64), np.asarray(positive, dtype=np.float64)
    difference = positive - negative
    length = np.linalg.norm(difference)
    if length <= 1e-10:
        raise ValueError("Coincident poles cannot define an axis")
    unit = difference / length
    midpoint = (negative + positive) / 2
    delta = features - midpoint
    along = delta @ unit
    perpendicular = np.linalg.norm(delta - along[:, None] * unit, axis=1)
    segment_along = np.clip(along, -length / 2, length / 2)
    segment_distance = np.sqrt(perpendicular ** 2 + (along - segment_along) ** 2)
    return {"along": along, "perpendicular": perpendicular,
            "segment_distance": segment_distance, "half_length": float(length / 2),
            "negative_distance": np.linalg.norm(features - negative, axis=1),
            "positive_distance": np.linalg.norm(features - positive, axis=1)}


def annotate(ax, horizontal, vertical, labels, colors, size=7):
    for index, (left, top, label, color) in enumerate(zip(horizontal, vertical, labels, colors)):
        ax.scatter(left, top, color=color, s=24, zorder=3)
        ax.annotate(str(index + 1), (left, top), xytext=(5, 6 if index % 2 else -11),
                    textcoords="offset points", fontsize=size, color=color)


def run(args):
    import open_clip
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    config = json.loads(args.config.read_text())
    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=args.device,
        force_quick_gelu=config.get("force_quick_gelu", True))
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(config["model"])

    @torch.no_grad()
    def encode(texts):
        return model.encode_text(tokenizer(texts).to(args.device)).float()

    projection = SemanticProjection(encode, config)
    labels = [*args.words, "generic photo"]
    prompts = [*[f"a {word} photo" for word in args.words], "a photo"]
    features = torch.nn.functional.normalize(encode(prompts), dim=-1)
    raw_scores, coordinates = projection.project_features(features)
    queries = features.cpu().numpy().astype(np.float64)
    poles = projection.centroids.cpu().numpy().astype(np.float64)
    pair_similarity = np.einsum("bd,ksd->bks", queries, poles)
    colors = ["#b24730" if word in CONTROLS else "#176979" for word in labels]
    colors[-1] = "#777777"
    args.out.mkdir(parents=True)
    rows = []
    geometries = []
    max_error = 0.0
    for index, axis in enumerate(AXES):
        geometry = axis_geometry(queries, poles[index, 0], poles[index, 1])
        geometries.append(geometry)
        for side, sign in (("negative", -1), ("positive", 1)):
            reconstructed = np.sqrt((geometry["along"] - sign * geometry["half_length"]) ** 2
                                    + geometry["perpendicular"] ** 2)
            max_error = max(max_error, float(np.max(np.abs(reconstructed - geometry[f"{side}_distance"]))))
        for query_index, label in enumerate(labels):
            negative, positive = pair_similarity[query_index, index]
            rows.append({"word": label, "prompt": prompts[query_index], "axis": axis,
                         "negative_pole": POLES[axis][0], "positive_pole": POLES[axis][1],
                         "cosine_negative": float(negative), "cosine_positive": float(positive),
                         "max_pole_cosine": float(max(negative, positive)),
                         "raw_difference": float(raw_scores[query_index, index]),
                         "origin": config["learned_calibration"]["score_origin"][index],
                         "mapped_coordinate": float(coordinates[query_index, index]),
                         **{key: float(geometry[key][query_index]) for key in (
                             "along", "perpendicular", "segment_distance", "negative_distance", "positive_distance")}})
    if max_error > 1e-10:
        raise AssertionError(f"Two-distance geometry mismatch: {max_error}")
    with (args.out / "word_axis_geometry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    vectors = np.concatenate((poles.reshape(10, -1), queries), axis=0)
    centered = vectors - vectors.mean(axis=0)
    _, singular, directions = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ directions[:2].T
    ratios = singular[:2] ** 2 / np.sum(singular ** 2)
    fig, ax = plt.subplots(figsize=(12, 8))
    palette = plt.get_cmap("tab10")
    for index, axis in enumerate(AXES):
        endpoints = projected[2 * index:2 * index + 2]
        ax.plot(endpoints[:, 0], endpoints[:, 1], color=palette(index), linewidth=2, alpha=.7)
        for side in range(2):
            ax.scatter(*endpoints[side], color=palette(index), marker="s", s=70)
            ax.annotate(f"{axis}: {POLES[axis][side]}", endpoints[side], xytext=(5, 7 + side * 4),
                        textcoords="offset points", fontsize=8)
    annotate(ax, projected[10:, 0], projected[10:, 1], labels, colors, 9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"PC1 ({ratios[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ratios[1]:.1%} variance)")
    ax.set_title(f"CLIP pole centroids and request embeddings: PCA ({sum(ratios):.1%} retained)")
    word_key = "Request key\n" + "\n".join(f"{index + 1}: {word}" for index, word in enumerate(labels))
    ax.text(1.02, .98, word_key, transform=ax.transAxes, fontsize=9, va="top")
    ax.grid(alpha=.2)
    fig.text(.5, .01, "PCA is exploratory: omitted dimensions can change apparent proximity. Red = unrelated control requests.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .03, .86, 1))
    fig.savefig(args.out / "pca.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    max_perpendicular = max(float(g["perpendicular"].max()) for g in geometries)
    x_limit = max(max(float(np.abs(g["along"]).max()), g["half_length"]) for g in geometries) + .17
    for index, axis in enumerate(AXES):
        ax = axes.flat[index]
        geometry = geometries[index]
        half = geometry["half_length"]
        ax.plot([-half, half], [0, 0], color="#222222", linewidth=2)
        ax.scatter([-half, half], [0, 0], marker="s", s=60, color="#222222")
        ax.text(-half, -.04, POLES[axis][0], ha="center", va="top", fontsize=8)
        ax.text(half, -.04, POLES[axis][1], ha="center", va="top", fontsize=8)
        annotate(ax, geometry["along"], geometry["perpendicular"], labels, colors)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_ylim(-.12, max_perpendicular + .14)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(axis)
        ax.set_xlabel("Position along pole line (unit-embedding Euclidean units)")
        ax.set_ylabel("Exact perpendicular distance to pole line")
        ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.03, .8,
        "Exact per-axis geometry\n\nHorizontal = coordinate along the line.\nVertical = norm of ALL perpendicular components.\n"
        "All five panels use the same scale.\n\nDistance to either pole is exactly preserved.\n"
        "Distances between request points are NOT preserved.\n\nCosine = 1 - (distance to pole)^2 / 2\nfor unit vectors. No similarity threshold is assumed.",
        transform=axes.flat[-1].transAxes, fontsize=9, va="top")
    axes.flat[-1].text(.03, .43, word_key, transform=axes.flat[-1].transAxes,
                        fontsize=9, va="top")
    fig.suptitle("Preference along an axis is different from proximity to that axis", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(args.out / "exact_axis_distances.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(19, 8))
    matrices = [pair_similarity.reshape(len(labels), 10), coordinates.cpu().numpy(),
                np.stack([g["segment_distance"] for g in geometries], axis=1)]
    headings = ["Absolute cosine to each pole (not a probability)",
                "Current mapped coordinate: no relevance gate", "Exact distance to pole segment (lower = closer)"]
    column_labels = [[f"{axis} {'-' if side == 0 else '+'}" for axis in AXES for side in range(2)],
                     list(AXES), list(AXES)]
    for index, (ax, matrix, heading, columns) in enumerate(zip(axes, matrices, headings, column_labels)):
        options = {"vmin": -.4, "vmax": .4, "cmap": "coolwarm"} if index == 1 else {"cmap": "viridis"}
        image = ax.imshow(matrix, aspect="auto", **options)
        ax.set_xticks(range(len(columns)), columns, rotation=65, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(heading, fontsize=10)
        for row in range(len(labels)):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f"{matrix[row,col]:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if index == 1 else "white")
        fig.colorbar(image, ax=ax, fraction=.04)
    fig.tight_layout()
    fig.savefig(args.out / "similarity_and_coordinates.png", dpi=180)
    plt.close(fig)

    manifest = {"model": config["model"], "config": str(args.config), "words": args.words,
                "text_templates": prompts, "pca_explained_variance_ratio": ratios.tolist(),
                "maximum_pole_distance_reconstruction_error": max_error,
                "production_mapping_modified": False,
                "caveat": "Line/segment proximity is a geometric diagnostic, not validated semantic relevance. No activation threshold fitted."}
    (args.out / "experiment.json").write_text(json.dumps(manifest, indent=2))
    page = ["<!doctype html><meta charset='utf-8'><title>Word-to-pole geometry</title>",
            "<h1>Word preference vs. axis relevance</h1>",
            "<p>The current mapper uses a cosine difference, generic-photo centering and signed scaling. "
            "It does not gate by absolute pole similarity or perpendicular distance. "
            "All prompts use the same photo template, including unrelated controls.</p>"]
    for filename in ("pca.png", "exact_axis_distances.png", "similarity_and_coordinates.png"):
        page.append(f"<a href='{filename}'><img style='max-width:100%' src='{filename}'></a>")
    page.append("<h2>Exact measurements</h2><table border='1'><tr>"
                + "".join(f"<th>{html.escape(key)}</th>" for key in rows[0]) + "</tr>")
    for row in rows:
        page.append("<tr>" + "".join(f"<td>{value:.4f}</td>" if isinstance(value, float)
                                    else f"<td>{html.escape(str(value))}</td>" for value in row.values()) + "</tr>")
    page.append("</table>")
    (args.out / "index.html").write_text("\n".join(page))
    print(json.dumps(manifest, indent=2))
    for word in labels:
        selected = [row for row in rows if row["word"] == word]
        strongest = max(selected, key=lambda row: abs(row["mapped_coordinate"]))
        print(word, strongest["axis"], f"coordinate={strongest['mapped_coordinate']:+.3f}",
              f"cos-={strongest['cosine_negative']:.3f} cos+={strongest['cosine_positive']:.3f}",
              f"axis distance={strongest['perpendicular']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "/data/data1/semantic_isp/results/exp42_phrase_pole_clip_100/calibrated_poles.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--words", nargs="+", default=WORDS)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()