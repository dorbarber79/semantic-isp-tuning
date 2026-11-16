"""Derive per-level DoF presets from the best Exp 19 runs and the word-to-DoF table.

A run is selected when it improves all three objectives relative to its neutral image
(semantic gain = neutral semantic loss - optimized semantic loss; realism and beauty
gains = -loss) and every gain is at or above the QUANTILE of all word-runs.
For each DoF and table level -3..+3, the preset is the median optimized DoF value
of selected runs whose word has that level. Level 0 is the ISP default by definition.
A weighted isotonic fit enforces monotone presets; levels with fewer than MIN_N
runs are interpolated (or extrapolated) and flagged.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from differentiable_isp import DifferentiableEMOVIS

RESULTS = Path("/data/data1/semantic_isp/results")
DOFS = DifferentiableEMOVIS.PARAM_NAMES
LEVELS = list(range(-3, 4))
TABLE_COLUMNS = {name: f"{name}_score" for name in DOFS}
LABELS = {"wb_warmth": "White balance (cool \u2194 warm)", "saturation": "Saturation",
          "brightness": "Brightness", "sharpness": "Sharpness (blur \u2194 sharpen)",
          "contrast": "Contrast"}


def load_runs(source):
    """Return (runs, defaults the runs were optimized with)."""
    runs, defaults = [], set()
    for path in sorted(source.glob("img_*/results.json")):
        result = json.loads(path.read_text())
        defaults.add(json.dumps(result["config"]["defaults"], sort_keys=True))
        for word, entry in result["words"].items():
            runs.append({
                "image": path.parent.name, "word": word, "params": entry["params"],
                "semantic_gain": entry["initial_losses"]["semantic"] - entry["losses"]["semantic"],
                "realism_gain": -entry["losses"]["realism"],
                "beauty_gain": -entry["losses"]["beauty"],
            })
    if len(defaults) != 1:
        raise ValueError(f"Source images were optimized with different defaults: {defaults}")
    return runs, json.loads(defaults.pop())


GAINS = ("semantic_gain", "realism_gain", "beauty_gain")


def select(runs, quantile, per_image=True, top_k=None):
    """Keep runs whose three gains are all > 0.
    top_k: per image, rank those runs by their weakest per-image percentile rank among the three
    gains and keep the best top_k (every image with any all-positive run contributes).
    Otherwise: also require each gain >= the quantile (within each image if per_image)."""
    groups = {}
    for run in runs:
        groups.setdefault(run["image"] if per_image or top_k else "all", []).append(run)
    chosen, thresholds = [], {}
    for name, members in groups.items():
        if top_k:
            ranks = {key: {id(run): rank for rank, run in
                           enumerate(sorted(members, key=lambda r: r[key]), 1)} for key in GAINS}
            positive = [run for run in members if all(run[key] > 0 for key in GAINS)]
            for run in positive:
                run["min_rank_pct"] = min(ranks[key][id(run)] for key in GAINS) / len(members)
            best = sorted(positive, key=lambda run: -run["min_rank_pct"])[:top_k]
            thresholds[name] = {"available": len(positive), "kept": len(best),
                                "lowest_min_rank_pct": best[-1]["min_rank_pct"] if best else None}
            chosen += best
            continue
        limits = {key: max(0.0, float(np.quantile([run[key] for run in members], quantile))) for key in GAINS}
        thresholds[name] = limits
        chosen += [run for run in members if all(run[key] > 0 and run[key] >= limits[key] for key in GAINS)]
    return chosen, thresholds


def isotonic(values, weights):
    """Weighted pool-adjacent-violators, non-decreasing."""
    blocks = [[v, w, 1] for v, w in zip(values, weights)]
    merged = []
    for block in blocks:
        merged.append(block)
        while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
            b = merged.pop()
            a = merged.pop()
            w = a[1] + b[1]
            merged.append([(a[0] * a[1] + b[0] * b[1]) / w, w, a[2] + b[2]])
    out = []
    for value, _, count in merged:
        out.extend([value] * count)
    return out


def calibrate_default(values, images, current, resamples=2000):
    """Level 0 must mean 'no change'. Cluster-bootstrap (by image) a 95% CI of the level-0
    median; if it excludes the current default, the median becomes the recommended default."""
    if len(values) < 2:
        return {"current": current, "recommended": current, "changed": False, "ci95": None}
    values, images = np.asarray(values), np.asarray(images)
    groups = [values[images == name] for name in np.unique(images)]
    rng = np.random.default_rng(0)
    medians = [np.median(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
               for _ in range(resamples)]
    ci = [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]
    changed = not ci[0] <= current <= ci[1]
    median = float(np.median(values))
    return {"current": current, "recommended": round(median, 2) if changed else current,
            "changed": changed, "median": median, "ci95": ci}


def derive(selected, table, min_n, source_defaults):
    presets, defaults = {}, {}
    for dof in DOFS:
        lo, hi = DifferentiableEMOVIS.PARAM_RANGES[dof]
        per_level = {}
        for level in LEVELS:
            members = [run for run in selected if int(table[run["word"]][TABLE_COLUMNS[dof]]) == level]
            values = [run["params"][dof] for run in members]
            words = sorted({run["word"] for run in members})
            per_level[level] = {
                "n": len(values), "words": words, "n_images": len({run["image"] for run in members}),
                "_images": [run["image"] for run in members],
                "median": float(np.median(values)) if values else None,
                "q25": float(np.percentile(values, 25)) if values else None,
                "q75": float(np.percentile(values, 75)) if values else None,
                "std": float(np.std(values)) if values else None,
                "values": values,
            }
        defaults[dof] = calibrate_default(per_level[0]["values"], per_level[0]["_images"],
                                          source_defaults[dof])
        default = defaults[dof]["recommended"]
        # Level 0 is the (recalibrated) default by definition; dominant weight in the monotone fit.
        known = [level for level in LEVELS if level == 0 or per_level[level]["n"] >= min_n]
        medians = [default if level == 0 else per_level[level]["median"] for level in known]
        weights = [1e6 if level == 0 else per_level[level]["n"] for level in known]
        fitted = dict(zip(known, isotonic(medians, weights)))
        for level in LEVELS:
            entry = per_level[level]
            if level in fitted:
                entry["preset"], entry["source"] = fitted[level], "default" if level == 0 else "data"
                continue
            below = [k for k in known if k < level]
            above = [k for k in known if k > level]
            if below and above:
                a, b = below[-1], above[0]
                value = fitted[a] + (fitted[b] - fitted[a]) * (level - a) / (b - a)
                entry["source"] = "interpolated"
            elif len(known) == 1:
                value, entry["source"] = default, "no data"
            else:
                side = below if below else above
                a, b = ((side[-2], side[-1]) if below else (side[0], side[1])) if len(side) >= 2 \
                    else (known[0], known[-1])
                slope = (fitted[b] - fitted[a]) / (b - a)
                anchor = side[-1] if below else side[0]
                value = fitted[anchor] + slope * (level - anchor)
                entry["source"] = "extrapolated"
            entry["preset"] = float(np.clip(value, lo, hi))
        presets[dof] = per_level
    return presets, defaults


def plot(presets, defaults, selected, out_path, title):
    images = sorted({run["image"] for run in selected}, key=lambda name: int(name[4:]))
    per_image_colors = len(images) <= 10
    colors = dict(zip(images, plt.cm.tab10(np.linspace(0, 1, 10)))) if per_image_colors \
        else {name: "#3182bd" for name in images}
    fig, axes = plt.subplots(1, len(DOFS), figsize=(5.2 * len(DOFS), 6.2))
    rng = np.random.default_rng(0)
    for axis, dof in zip(axes, DOFS):
        per_level = presets[dof]
        lo, hi = DifferentiableEMOVIS.PARAM_RANGES[dof]
        positions = [level for level in LEVELS if len(per_level[level]["values"]) >= 2]
        if positions:
            parts = axis.violinplot([per_level[p]["values"] for p in positions], positions,
                                    widths=0.8, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor("#9ecae1")
                body.set_alpha(0.45)
            axis.boxplot([per_level[p]["values"] for p in positions], positions=positions, widths=0.22,
                         showfliers=False, medianprops={"color": "black"})
        for level in LEVELS:
            members = [run for run in selected
                       if run["_levels"][dof] == level]
            xs = level + rng.uniform(-0.28, 0.28, len(members))
            axis.scatter(xs, [run["params"][dof] for run in members], s=7,
                         c=[colors[run["image"]] for run in members], alpha=0.65, linewidths=0)
        preset = [per_level[level]["preset"] for level in LEVELS]
        axis.plot(LEVELS, preset, color="crimson", lw=2, zorder=5, label="Preset")
        for level, value in zip(LEVELS, preset):
            source = per_level[level]["source"]
            filled = source in ("data", "default")
            axis.scatter([level], [value], s=70, zorder=6, edgecolors="crimson",
                         facecolors="crimson" if filled else "white", linewidths=2,
                         marker="D" if source == "default" else "o")
            axis.annotate(f"{value:+.2f}", (level, value), textcoords="offset points",
                          xytext=(6, 6), fontsize=8, color="crimson")
            axis.text(level, lo - 0.05 * (hi - lo), f"n={per_level[level]['n']}",
                      ha="center", va="top", fontsize=8)
        raw = [(level, per_level[level]["median"]) for level in LEVELS if per_level[level]["n"]]
        axis.scatter([l for l, _ in raw], [m for _, m in raw], marker="_", s=260, color="black",
                     linewidths=2.5, zorder=7)
        axis.axhline(defaults[dof]["current"], color="gray", ls="--", lw=0.8)
        if defaults[dof]["changed"]:
            axis.axhline(defaults[dof]["recommended"], color="darkgreen", ls="--", lw=1.2)
        if defaults[dof]["ci95"]:
            axis.plot([0, 0], defaults[dof]["ci95"], color="darkgreen", lw=5, alpha=0.5, zorder=4)
        axis.set_xticks(LEVELS, [f"{level:+d}" if level else "0" for level in LEVELS])
        axis.set_xlim(-3.6, 3.6)
        axis.set_ylim(lo - 0.12 * (hi - lo), hi + 0.04 * (hi - lo))
        note = (f"default {defaults[dof]['current']:+.2f} \u2192 {defaults[dof]['recommended']:+.2f}"
                if defaults[dof]["changed"] else f"default {defaults[dof]['current']:+.2f} kept")
        axis.set_title(f"{LABELS[dof]}\n{note}")
        axis.set_xlabel("Word-table level")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Optimized ISP parameter value")
    handles = [plt.Line2D([], [], color="crimson", lw=2, marker="o", label="Preset (filled: data)"),
               plt.Line2D([], [], color="crimson", lw=0, marker="o", markerfacecolor="white",
                          label="Preset (hollow: interpolated / extrapolated)"),
               plt.Line2D([], [], color="crimson", lw=0, marker="D", label="Level 0 = (recalibrated) default"),
               plt.Line2D([], [], color="black", lw=0, marker="_", markersize=14, markeredgewidth=2.5,
                          label="Raw median of selected runs"),
               plt.Line2D([], [], color="gray", ls="--", label="Current ISP default"),
               plt.Line2D([], [], color="darkgreen", ls="--", label="Recommended new default"),
               plt.Line2D([], [], color="darkgreen", lw=5, alpha=0.5,
                          label="95% CI of level-0 median (image bootstrap)")]
    if per_image_colors:
        handles += [plt.Line2D([], [], lw=0, marker="o", color=colors[name], label=name) for name in images]
    else:
        handles.append(plt.Line2D([], [], lw=0, marker="o", color="#3182bd",
                                  label=f"Selected run ({len(images)} images)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RESULTS / "exp19_gamut_fullres")
    parser.add_argument("--table", type=Path, default=RESULTS / "word_dof_expectations.csv")
    parser.add_argument("--out", type=Path, default=RESULTS / "exp20_word_presets")
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--min-n", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10,
                        help="Per-image top-k by weakest gain rank (0 = use quantile thresholds)")
    parser.add_argument("--global-threshold", action="store_true",
                        help="Rank runs across all images instead of within each image")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.table.open() as stream:
        table = {row["word"]: row for row in csv.DictReader(stream)}
    runs, source_defaults = load_runs(args.source)
    n_images = len({run["image"] for run in runs})
    if len(runs) != n_images * len(table) or set(run["word"] for run in runs) != set(table):
        raise ValueError(f"Expected {n_images} x {len(table)} word-runs covering exactly the table's words")
    for run in runs:
        run["_levels"] = {dof: int(table[run["word"]][TABLE_COLUMNS[dof]]) for dof in DOFS}

    summary = {"source": str(args.source), "images": n_images, "word_runs": len(runs),
               "selection": (f"per image: all three gains > 0, top {args.top_k} by weakest per-image percentile rank"
                             if args.top_k else "all three gains > 0 and >= the given quantile "
                             + ("over all word-runs" if args.global_threshold else "within each image")),
               "preset": "median optimized value per level; weighted isotonic; level 0 = default, recalibrated "
                         "to the level-0 median when its image-bootstrap 95% CI excludes the default",
               "min_n": args.min_n, "source_defaults": source_defaults,
               "ranges": DifferentiableEMOVIS.PARAM_RANGES, "sensitivity": {}}
    variants = sorted({5, args.top_k, 20}) if args.top_k else sorted({0.33, args.quantile, 0.67})
    for variant in variants:
        quantile = args.quantile if args.top_k else variant
        selected, thresholds = select(runs, quantile, not args.global_threshold, variant if args.top_k else None)
        presets, defaults = derive(selected, table, args.min_n, source_defaults)
        summary["sensitivity"][f"top{variant}" if args.top_k else f"{quantile:.2f}"] = {
            "selected_runs": len(selected), "thresholds": thresholds, "default_calibration": defaults,
            "presets": {dof: {str(level): presets[dof][level]["preset"] for level in LEVELS} for dof in DOFS},
        }
        if variant != (args.top_k or args.quantile):
            continue
        summary.update(selected_runs=len(selected), thresholds=thresholds, quantile=quantile,
                       selected_words=len({run["word"] for run in selected}),
                       selected_per_image={image: sum(run["image"] == image for run in selected)
                                           for image in sorted({run["image"] for run in runs})})
        summary["default_calibration"] = defaults
        summary["levels"] = {dof: {str(level): {key: value for key, value in entry.items()
                                                if key not in ("values", "_images")}
                                   for level, entry in presets[dof].items()} for dof in DOFS}
        with (args.out / "selected_runs.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "word", "semantic_gain", "realism_gain", "beauty_gain", "min_rank_pct", *DOFS,
                             *(f"{dof}_level" for dof in DOFS)])
            for run in sorted(selected, key=lambda r: (int(r["image"][4:]), r["word"])):
                writer.writerow([run["image"], run["word"], run["semantic_gain"], run["realism_gain"],
                                 run["beauty_gain"], run.get("min_rank_pct", ""),
                                 *(run["params"][dof] for dof in DOFS),
                                 *(run["_levels"][dof] for dof in DOFS)])
        with (args.out / "presets.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["dof", "level", "preset", "source", "n", "n_images", "median", "q25", "q75", "std",
                             "words"])
            for dof in DOFS:
                for level in LEVELS:
                    entry = presets[dof][level]
                    writer.writerow([dof, level, entry["preset"], entry["source"], entry["n"], entry["n_images"],
                                     entry["median"],
                                     entry["q25"], entry["q75"], entry["std"], " ".join(entry["words"])])
        plot(presets, defaults, selected, args.out / "dof_level_presets.png",
             f"DoF presets per word-table level from {len(selected)} best of {len(runs)} runs "
             f"({n_images} images; "
             + (f"top {args.top_k} per image improving semantic, realism and beauty, ranked by weakest gain)"
                if args.top_k else f"semantic, realism and beauty gains all \u2265 q{quantile:.2f} "
                f"{'overall' if args.global_threshold else 'per image'})"))
    (args.out / "presets.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ("images", "word_runs", "selected_runs", "selected_words",
                                                     "selected_per_image")}, indent=2))
    for dof in DOFS:
        print(dof.ljust(11), " ".join(f"{level:+d}:{summary['levels'][dof][str(level)]['preset']:+.3f}"
                                      f"({summary['levels'][dof][str(level)]['n']},"
                                      f"{summary['levels'][dof][str(level)]['source'][0]})"
                                      for level in LEVELS))
    for dof, item in summary["default_calibration"].items():
        print(f"default {dof}: current {item['current']:+.3f}, level-0 median "
              f"{item.get('median', float('nan')):+.3f}, CI {item['ci95']}, recommended {item['recommended']:+.3f}"
              f"{' (CHANGE)' if item['changed'] else ''}")


if __name__ == "__main__":
    main()
