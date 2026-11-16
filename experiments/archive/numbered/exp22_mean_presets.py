"""Mean presets: for each DoF and word-table level, the preset is the mean optimized DoF value
over the Exp 19 word-runs (every image) whose word has that level in the table. Level 0
means are the new ISP defaults.

Monotonicity: presets must strictly increase with level. While an adjacent pair violates
this, the single run that most reduces the violation is removed as an outlier: either the
largest value of the lower level or the smallest value of the upper level.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp20_word_presets import DOFS, LABELS, LEVELS, RESULTS, TABLE_COLUMNS, load_runs
from differentiable_isp import DifferentiableEMOVIS


def monotone_trim(groups, min_step, min_keep):
    """groups: {level: [(value, run), ...]} -> (kept groups, removed list). Greedy outlier removal."""
    kept = {level: sorted(items, key=lambda item: item[0]) for level, items in groups.items()}
    removed = []

    def mean(level):
        return float(np.mean([value for value, _ in kept[level]]))

    while True:
        levels = [level for level in LEVELS if kept.get(level)]
        gaps = [(mean(b) - mean(a), a, b) for a, b in zip(levels, levels[1:])]
        violations = [gap for gap in gaps if gap[0] < min_step]
        if not violations:
            return kept, removed
        best = None
        for _, a, b in violations:
            for level, index in ((a, -1), (b, 0)):
                if len(kept[level]) <= min_keep:
                    continue
                before = sum(min_step - g for g, *_ in gaps if g < min_step)
                trial = kept[level].pop(index)
                after_gaps = [mean(y) - mean(x) for x, y in zip(levels, levels[1:])]
                after = sum(min_step - g for g in after_gaps if g < min_step)
                kept[level].insert(0 if index == 0 else len(kept[level]), trial)
                if best is None or before - after > best[0]:
                    best = (before - after, level, index)
        if best is None:
            raise RuntimeError("Cannot reach monotonicity without going below min_keep runs per level")
        _, level, index = best
        value, run = kept[level].pop(index)
        removed.append({"level": level, "value": value, "image": run["image"], "word": run["word"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RESULTS / "exp19_gamut_fullres")
    parser.add_argument("--table", type=Path, default=RESULTS / "word_dof_expectations.csv")
    parser.add_argument("--out", type=Path, default=RESULTS / "exp22_mean_presets")
    parser.add_argument("--min-step-frac", type=float, default=0.01,
                        help="Minimum increase between adjacent levels, as a fraction of the DoF range")
    parser.add_argument("--min-keep", type=int, default=5, help="Never trim a level below this many runs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.table.open() as stream:
        table = {row["word"]: row for row in csv.DictReader(stream)}
    runs, source_defaults = load_runs(args.source)
    if set(run["word"] for run in runs) != set(table):
        raise ValueError("Word table and experiment words differ")

    rows, presets, removed_rows, raw_means, kept_values, removed_values = [], {}, [], {}, {}, {}
    for dof in DOFS:
        presets[dof] = {}
        lo, hi = DifferentiableEMOVIS.PARAM_RANGES[dof]
        groups = {level: [(run["params"][dof], run) for run in runs
                          if int(table[run["word"]][TABLE_COLUMNS[dof]]) == level] for level in LEVELS}
        raw_means[dof] = {level: float(np.mean([v for v, _ in items])) if items else None
                          for level, items in groups.items()}
        kept, removed = monotone_trim(groups, args.min_step_frac * (hi - lo), args.min_keep)
        removed_rows += [{"dof": dof, **item} for item in removed]
        kept_values[dof] = {level: [v for v, _ in kept[level]] for level in LEVELS}
        removed_values[dof] = removed
        for level in LEVELS:
            members = [run for _, run in kept[level]]
            values = np.array([v for v, _ in kept[level]])
            words = sorted({run["word"] for run in members})
            entry = {
                "dof": dof, "level": level, "n_words": len({run["word"] for _, run in groups[level]}),
                "n_runs": len(groups[level]), "n_removed": len(groups[level]) - len(values),
                "n_kept": len(values), "raw_mean": raw_means[dof][level],
                "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std()) if len(values) else None,
                "sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
                "median": float(np.median(values)) if len(values) else None,
                "words": " ".join(words),
            }
            presets[dof][level] = entry
            rows.append(entry)

    with (args.out / "mean_presets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "removed_outliers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dof", "level", "value", "image", "word"])
        writer.writeheader()
        writer.writerows(removed_rows)
    new_defaults = {dof: presets[dof][0]["mean"] for dof in DOFS}
    (args.out / "mean_presets.json").write_text(json.dumps({
        "rule": "preset = mean optimized value over word-runs whose word has that level, after greedy outlier "
                "removal until strictly increasing (min step = min_step_frac x range); level 0 = new default",
        "min_step_frac": args.min_step_frac, "min_keep": args.min_keep, "removed_runs": len(removed_rows),
        "source": str(args.source), "images": len({run["image"] for run in runs}), "word_runs": len(runs),
        "source_defaults": source_defaults, "new_defaults": new_defaults,
        "presets": {dof: {str(level): presets[dof][level]["mean"] for level in LEVELS} for dof in DOFS},
    }, indent=2))

    fig, axes = plt.subplots(1, len(DOFS), figsize=(5.2 * len(DOFS), 5.6))
    rng = np.random.default_rng(0)
    for axis, dof in zip(axes, DOFS):
        lo, hi = DifferentiableEMOVIS.PARAM_RANGES[dof]
        for level in LEVELS:
            values = kept_values[dof][level]
            dropped = [item["value"] for item in removed_values[dof] if item["level"] == level]
            axis.scatter(level + rng.uniform(-0.25, 0.25, len(dropped)), dropped, s=9, marker="x",
                         color="gray", alpha=0.6, linewidths=0.8)
            if len(values) >= 2:
                parts = axis.violinplot([values], [level], widths=0.8, showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor("#9ecae1")
                    body.set_alpha(0.5)
            axis.scatter(level + rng.uniform(-0.25, 0.25, len(values)), values, s=4, color="#3182bd",
                         alpha=0.35, linewidths=0)
            axis.text(level, lo - 0.05 * (hi - lo),
                      f"{presets[dof][level]['n_words']}w\n{len(values)}/{presets[dof][level]['n_runs']}r",
                      ha="center", va="top", fontsize=7)
        raw = [(level, raw_means[dof][level]) for level in LEVELS if raw_means[dof][level] is not None]
        axis.plot([l for l, _ in raw], [m for _, m in raw], color="black", ls=":", marker=".", lw=1,
                  zorder=4)
        means = [presets[dof][level]["mean"] for level in LEVELS]
        sds = [presets[dof][level]["std"] or 0 for level in LEVELS]
        axis.errorbar([l for l, m in zip(LEVELS, means) if m is not None],
                      [m for m in means if m is not None],
                      yerr=[s for m, s in zip(means, sds) if m is not None],
                      color="crimson", lw=2, marker="o", capsize=4, zorder=5)
        for level, mean in zip(LEVELS, means):
            if mean is not None:
                axis.annotate(f"{mean:+.2f}", (level, mean), textcoords="offset points", xytext=(6, 6),
                              fontsize=8, color="crimson")
        axis.axhline(source_defaults[dof], color="gray", ls="--", lw=0.8)
        axis.axhline(new_defaults[dof], color="darkgreen", ls="--", lw=1.2)
        axis.set_xticks(LEVELS, [f"{level:+d}" if level else "0" for level in LEVELS])
        axis.set_ylim(lo - 0.16 * (hi - lo), hi + 0.04 * (hi - lo))
        axis.set_title(f"{LABELS[dof]}\nold default {source_defaults[dof]:+.2f} \u2192 new {new_defaults[dof]:+.2f}")
        axis.set_xlabel("Word-table level")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Optimized ISP parameter value")
    handles = [plt.Line2D([], [], color="crimson", lw=2, marker="o", label="Preset = mean after outlier removal (\u00b1 1 SD)"),
               plt.Line2D([], [], color="black", ls=":", marker=".", label="Mean before removal"),
               plt.Line2D([], [], color="gray", lw=0, marker="x", label="Removed outlier"),
               plt.Line2D([], [], color="gray", ls="--", label="Old default"),
               plt.Line2D([], [], color="darkgreen", ls="--", label="New default (level-0 mean)"),
               plt.Line2D([], [], lw=0, marker="o", color="#3182bd", label="Every word-run (w = words, r = runs)")]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=9, frameon=False)
    fig.suptitle(f"Monotone mean preset per DoF level: {len(runs)} word-runs "
                 f"({len({run['image'] for run in runs})} images), {len(removed_rows)} outlier values removed "
                 f"(w = words, r = kept/total runs)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(args.out / "mean_presets.png", dpi=130)
    plt.close(fig)

    print("dof         " + " ".join(f"{level:>+12d}" for level in LEVELS))
    for dof in DOFS:
        print(dof.ljust(11), " ".join(
            f"{presets[dof][level]['mean']:+.3f}(-{presets[dof][level]['n_removed']:>3})"
            if presets[dof][level]["mean"] is not None else "      -     " for level in LEVELS))
    print("new defaults:", {dof: round(value, 3) for dof, value in new_defaults.items()})


if __name__ == "__main__":
    main()
