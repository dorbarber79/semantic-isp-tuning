"""Histogram levels: for each DoF, pool the optimized value of every Exp 19 word-run
(all images x all words), split the distribution into 7 equal-count levels (-3..+3) by
quantiles, and use each bin's median as the preset. Level 0 (middle bin) = ISP default.
No word table is used: levels come from the data distribution alone.
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
from exp20_word_presets import DOFS, LABELS, LEVELS, RESULTS, load_runs

COLORS = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(LEVELS)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RESULTS / "exp19_gamut_fullres")
    parser.add_argument("--out", type=Path, default=RESULTS / "exp23_histogram_levels")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs, source_defaults = load_runs(args.source)

    rows, summary = [], {}
    fig, axes = plt.subplots(len(DOFS), 1, figsize=(13, 3.4 * len(DOFS)))
    for axis, dof in zip(axes, DOFS):
        lo, hi = DifferentiableEMOVIS.PARAM_RANGES[dof]
        values = np.array([run["params"][dof] for run in runs])
        edges = np.quantile(values, np.linspace(0, 1, len(LEVELS) + 1))
        level_of = np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, len(LEVELS) - 1)
        bins = np.linspace(lo, hi, 61)
        summary[dof] = {}
        for index, level in enumerate(LEVELS):
            members = values[level_of == index]
            words = [runs[i]["word"] for i in np.flatnonzero(level_of == index)]
            top_words = sorted(set(words), key=lambda w: -words.count(w))[:6]
            entry = {"dof": dof, "level": level, "low": float(edges[index]), "high": float(edges[index + 1]),
                     "n_runs": int(len(members)), "preset": float(np.median(members)),
                     "mean": float(members.mean()), "frequent_words": " ".join(top_words)}
            rows.append(entry)
            summary[dof][str(level)] = entry
            axis.hist(members, bins=bins, color=COLORS[index], edgecolor="white", linewidth=0.3,
                      label=f"{level:+d}" if level else "0")
            axis.axvline(entry["preset"], color="black", lw=1.2)
            axis.text(entry["preset"], 0.98 - 0.09 * (index % 2), f"{level:+d}: {entry['preset']:+.2f}" if level
                      else f"0: {entry['preset']:+.2f}", transform=axis.get_xaxis_transform(),
                      ha="center", va="top", fontsize=8,
                      bbox={"facecolor": "white", "alpha": 0.8, "pad": 1, "linewidth": 0})
        for edge in edges[1:-1]:
            axis.axvline(edge, color="gray", ls=":", lw=0.8)
        axis.axvline(source_defaults[dof], color="green", ls="--", lw=1.2)
        axis.set_xlim(lo, hi)
        axis.set_ylabel("Word-runs")
        axis.set_title(f"{LABELS[dof]}: {len(values)} runs, 7 equal-count levels "
                       f"(black = preset = bin median; dotted = level boundaries; green = default used in Exp 19)",
                       fontsize=10)
    axes[-1].set_xlabel("Optimized ISP parameter value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Level", loc="center right")
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    fig.savefig(args.out / "histogram_levels.png", dpi=120)
    plt.close(fig)

    with (args.out / "histogram_levels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "presets_for_render.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dof", "level", "preset", "source"])
        for row in rows:
            writer.writerow([row["dof"], row["level"], row["preset"], "data"])
    new_defaults = {dof: summary[dof]["0"]["preset"] for dof in DOFS}
    (args.out / "histogram_levels.json").write_text(json.dumps({
        "rule": "7 equal-count quantile bins of all word-run values per DoF; preset = bin median; level 0 = default",
        "source": str(args.source), "word_runs": len(runs), "source_defaults": source_defaults,
        "new_defaults": new_defaults, "levels": summary}, indent=2))

    print("dof         " + " ".join(f"{level:>+8d}" for level in LEVELS))
    for dof in DOFS:
        print(dof.ljust(11), " ".join(f"{summary[dof][str(level)]['preset']:+8.3f}" for level in LEVELS))
    print("boundaries:")
    for dof in DOFS:
        print(dof.ljust(11), " ".join(f"{summary[dof][str(level)]['low']:+.3f}" for level in LEVELS),
              f"{summary[dof]['3']['high']:+.3f}")
    print("new defaults:", {dof: round(value, 3) for dof, value in new_defaults.items()})


if __name__ == "__main__":
    main()
