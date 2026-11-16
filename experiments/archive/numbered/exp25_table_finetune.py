"""Fine-tune table presets with signed realism loss only; compare three methods.

For every word and image the optimizer starts from the Exp 24 table preset (word-table levels mapped
to Exp 23 histogram presets) and minimizes signed realism loss only. Negative realism loss rewards
improvements and positive loss penalizes degradations. Semantic similarity and beauty are measured
but not optimized. Guards come from the Exp 19 neutral; Adam lr 0.03 cosine, 500 iterations,
best iterate. No regularization term is applied.
"""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

import exp07_emovis_diff as experiment
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw
from exp19_full_evaluation import (LOSSES, LR, encode_image, save_native, save_trace,
                                   score_image, write_csv, write_json)
from exp24_table_vs_loss import DATA, NAMES, OUT as TABLE_OUT, RESULTS, SOURCE, describe, font, read_png

OUT = RESULTS / "exp27_table_finetune_realism_only"
WEIGHTS = {"semantic_weight": 0.0, "realism_weight": 1.0, "beauty_weight": 0.0}
SCORES = ("semantic_similarity", "realism", "beauty")
HEADER = 112


def make_sbs(path, word, panels, width):
    height = round(panels[0][0].height * width / panels[0][0].width)
    sheet = Image.new("RGB", (len(panels) * (width + 10) - 10, height + HEADER), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, lines) in enumerate(panels):
        x = index * (width + 10)
        sheet.paste(image.resize((width, height), Image.LANCZOS), (x, HEADER))
        lines = [f"{word}: {lines[0]}" if index == 0 else lines[0], *lines[1:]]
        for row, line in enumerate(lines):
            draw.text((x + 8, 6 + row * 34), line, fill="darkred" if row == 0 else "black",
                      font=font(24 if row == 0 else 17))
    sheet.save(path, quality=90)


def run_image(image_id, iterations, width):
    out = OUT / f"img_{image_id}"
    if (out / "comparison.csv").exists():
        print("skip existing", out, flush=True)
        return
    for directory in ("full_resolution", "sbs", "convergence"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    source = json.loads((SOURCE / f"img_{image_id}" / "results.json").read_text())
    with (TABLE_OUT / f"img_{image_id}" / "comparison.csv").open() as stream:
        table_rows = {row["word"]: row for row in csv.DictReader(stream)}
    dng = DATA / source["dng"]
    image = load_and_preprocess_raw(str(dng), 512).cuda()
    native = load_and_preprocess_raw(str(dng), None).cuda()
    model = experiment.SigLIP("cuda")
    model.model.requires_grad_(False)
    mean, std = experiment.vlm_norm_stats("cuda")
    isp = DifferentiableEMOVIS("cuda")
    with torch.no_grad():
        isp.set_params(source["config"]["defaults"])
        guards = experiment.build_guards(model, encode_image(model, isp(image), mean, std))
    rows, results = [], {}
    for word, table_row in table_rows.items():
        table_params = {dof: float(table_row[f"table_{dof}"]) for dof in NAMES}
        levels = {dof: int(float(table_row[f"level_{dof}"])) for dof in NAMES}
        history = []
        params, similarity, _, _ = experiment.optimize_for_word(
            word, image, guards, model, mean, std, num_iters=iterations, lr=LR, reg_weight=0.0,
            history=history, init_params=table_params, **WEIGHTS)
        best = min(history, key=lambda entry: entry["total"])
        if params != best["params"] or abs(history[0]["semantic_similarity"]
                                            - float(table_row["table_semantic_similarity"])) > 1e-4:
            raise AssertionError(f"Start/best mismatch for {image_id} {word}")
        with torch.no_grad():
            isp.raw_params.copy_(torch.tensor(best["raw_params"], device="cuda"))
            check = score_image(model, isp(image), word, guards, isp, mean, std)
            if abs(check["semantic"] - best["semantic"]) > 1e-6:
                raise AssertionError(f"Rendered/selected mismatch for {image_id} {word}")
            save_native(out / "full_resolution" / f"{word}.png", isp(native))
        save_trace(out, word, history)
        tuned_scores = {"semantic_similarity": similarity, "realism": best["realism"], "beauty": best["beauty"]}
        loss_entry = source["words"][word]
        loss_scores = {**loss_entry["losses"], "semantic_similarity": loss_entry["clip_sim"]}
        table_scores = {name: float(table_row[f"table_{name}"]) for name in SCORES}
        make_sbs(out / "sbs" / f"{word}.jpg", word, [
            (read_png(SOURCE / f"img_{image_id}" / loss_entry["full_resolution_image"]),
             describe("Loss (Exp 19)", loss_entry["params"], None, loss_scores)),
            (read_png(TABLE_OUT / f"img_{image_id}" / "full_resolution" / f"{word}.png"),
             describe("Table preset (Exp 24)", table_params, levels, table_scores)),
            (read_png(out / "full_resolution" / f"{word}.png"),
             describe(f"Table + realism-only fine-tune (best it {best['iteration']})", params, None, tuned_scores)),
        ], width)
        results[word] = {"start_params": table_params, "levels": levels, "params": params,
                         "best_iteration": best["iteration"],
                         "losses": {key: best[key] for key in LOSSES},
                         "initial_losses": {key: history[0][key] for key in LOSSES},
                         "semantic_similarity": similarity}
        rows.append({
            "image": image_id, "word": word,
            **{f"{side}_{dof}": value for side, values in
               (("loss", loss_entry["params"]), ("table", table_params), ("tuned", params))
               for dof, value in values.items()},
            **{f"{side}_{name}": scores[name] for side, scores in
               (("loss", loss_scores), ("table", table_scores), ("tuned", tuned_scores)) for name in SCORES},
            "best_iteration": best["iteration"],
        })
        print(image_id, word, best["iteration"], flush=True)
    write_json(out / "results.json", {
        "description": __doc__, "image": image_id, "dng": source["dng"],
        "config": {"iters": iterations, "lr": LR, "reg_weight": 0.0,
               "reg_anchor": None, "loss_weights": WEIGHTS, "guards": "Exp 19 neutral (its defaults)",
                   "exp19_defaults": source["config"]["defaults"]},
        "words": results})
    write_csv(out / "comparison.csv", rows)


def write_report():
    rows = []
    for path in sorted(OUT.glob("img_*/comparison.csv")):
        with path.open() as stream:
            rows += [{key: value if key == "word" else float(value) for key, value in row.items()}
                     for row in csv.DictReader(stream)]
    neutral = {}
    for path in SOURCE.glob("img_*/results.json"):
        data = json.loads(path.read_text())
        image_id = float(path.parent.name[4:])
        for word, entry in data["words"].items():
            neutral[(image_id, word)] = -entry["initial_losses"]["semantic"]
    sides = ("loss", "table", "tuned")
    summary = {"runs": len(rows), "images": len({row["image"] for row in rows})}
    for side in sides:
        gains = [row[f"{side}_semantic_similarity"] - neutral[(row["image"], row["word"])] for row in rows]
        summary[side] = {
            "semantic_gain_vs_neutral": float(np.mean(gains)),
            "realism_gain_vs_neutral": float(-np.mean([row[f"{side}_realism"] for row in rows])),
            "beauty_gain_vs_neutral": float(-np.mean([row[f"{side}_beauty"] for row in rows])),
            "semantic_improved": int(sum(gain > 0 for gain in gains)),
            "realism_not_worse": int(sum(row[f"{side}_realism"] <= 0 for row in rows)),
            "beauty_not_worse": int(sum(row[f"{side}_beauty"] <= 0 for row in rows)),
            "all_three_ok": int(sum(gain > 0 and row[f"{side}_realism"] <= 0 and row[f"{side}_beauty"] <= 0
                                    for gain, row in zip(gains, rows))),
        }
    summary["tuned_vs_loss_semantic_better"] = int(sum(
        row["tuned_semantic_similarity"] > row["loss_semantic_similarity"] for row in rows))
    summary["mean_abs_param_distance"] = {
        f"{a}_to_{b}": {dof: float(np.mean([abs(row[f"{a}_{dof}"] - row[f"{b}_{dof}"]) for row in rows]))
                        for dof in NAMES}
        for a, b in (("tuned", "table"), ("tuned", "loss"), ("table", "loss"))}
    write_json(OUT / "summary.json", summary)
    write_csv(OUT / "comparison.csv", rows)
    words = sorted({row["word"] for row in rows})
    images = sorted({int(row["image"]) for row in rows})
    metrics = ("semantic_gain_vs_neutral", "realism_gain_vs_neutral", "beauty_gain_vs_neutral",
               "semantic_improved", "realism_not_worse", "beauty_not_worse", "all_three_ok")
    parts = ["<!doctype html><meta charset='utf-8'><title>Exp 27 realism-only fine-tune</title><style>"
             "body{font:14px sans-serif;margin:20px}img{width:100%;max-width:1800px}"
             "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}</style>",
             "<h1>Exp 27: loss vs table preset vs table + realism-only fine-tune</h1>",
             f"<p>{summary['runs']} runs. Left = Exp 19 loss from defaults; middle = table preset; right = "
             "signed realism only, started from the table preset with no regularization. Semantic similarity "
             "and beauty are measured but not optimized. Gains are vs the neutral image; "
             "higher = better. Counts are out of all runs.</p><table><tr><th></th>"
             + "".join(f"<th>{metric}</th>" for metric in metrics) + "</tr>"]
    for side in sides:
        parts.append(f"<tr><td>{side}</td>" + "".join(
            f"<td>{summary[side][metric]:.4f}</td>" if isinstance(summary[side][metric], float)
            else f"<td>{summary[side][metric]}</td>" for metric in metrics) + "</tr>")
    parts.append("</table><p>Jump to word: " + " ".join(
        f"<a href='#{html.escape(word)}'>{html.escape(word)}</a>" for word in words) + "</p>")
    for word in words:
        parts.append(f"<h2 id='{html.escape(word)}'>{html.escape(word)}</h2>")
        parts += [f"<a href='img_{image_id}/sbs/{word}.jpg'><img loading='lazy' src='img_{image_id}/sbs/"
                  f"{word}.jpg'></a>" for image_id in images if (OUT / f"img_{image_id}/sbs/{word}.jpg").exists()]
    (OUT / "index.html").write_text("\n".join(parts))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=int, default=[])
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--width", type=int, default=1000, help="width of each SBS panel")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        torch.set_num_threads(2)
        for image_id in args.images:
            run_image(image_id, args.iters, args.width)
    write_report()


if __name__ == "__main__":
    main()
