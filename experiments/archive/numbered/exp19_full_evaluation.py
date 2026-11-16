"""Full-corpus evaluation with loss traces, exact best iterates and native RGB exports."""

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import struct
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import rawpy

import exp07_emovis_diff as experiment
from differentiable_isp import DifferentiableEMOVIS, load_and_preprocess_raw


LOSSES = ("semantic", "realism", "beauty", "regularization", "total")
IMAGES = {1: "desert", 28: "dog", 2: "child", 236: "man", 194: "swan",
          277: "snow", 263: "dusk", 384: "backlit", 33: "indoor", 103: "city"}
REG_WEIGHT = 0.003
LR = 0.03


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def encode_image(model, image, mean, std):
    resized = F.interpolate(image, (224, 224), mode="bicubic", align_corners=False)
    feature = model.encode_image((resized - mean) / std)
    return feature / feature.norm(dim=-1, keepdim=True)


@torch.no_grad()
def score_image(model, image, word, guards, isp, mean, std):
    feature = encode_image(model, image, mean, std)
    similarity = None
    if word is not None:
        text = model.encode_text(model.tokenize([f"a {word} photo"]))
        text = text / text.norm(dim=-1, keepdim=True)
        similarity = (feature @ text.T).mean().item()
    quality = {name: scorer(feature).item() for name, (scorer, _) in guards.items()}
    losses = {
        "semantic": -similarity if similarity is not None else 0.0,
        "realism": guards["realism"][1].item() - quality["realism"],
        "beauty": guards["beauty"][1].item() - quality["beauty"],
        "regularization": (REG_WEIGHT * (isp.raw_params - isp.raw_default).square().sum()).item(),
    }
    losses["total"] = (experiment.SEMANTIC_WEIGHT * losses["semantic"]
                       + experiment.REALISM_WEIGHT * losses["realism"]
                       + experiment.BEAUTY_WEIGHT * losses["beauty"]
                       + losses["regularization"])
    return {**losses, "semantic_similarity": similarity,
            "realism_score": quality["realism"], "beauty_score": quality["beauty"]}


def save_preview(path, image):
    array = image[0].detach().cpu().permute(1, 2, 0).numpy()
    Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(path, quality=95)


@torch.no_grad()
def save_native(path, image):
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError(f"Invalid native RGB output: {path}")
    # Score the same 16-bit-quantized pixels that are written, not the JPEG preview.
    image.mul_(65535).round_().div_(65535)
    array = np.rint(image[0].cpu().permute(1, 2, 0).numpy() * 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_PNG_COMPRESSION, 2]):
        raise OSError(f"Could not save 16-bit RGB PNG: {path}")


def save_trace(out, name, history):
    rows = []
    for entry in history:
        rows.append({
            "iteration": entry["iteration"],
            **{loss: entry[loss] for loss in LOSSES},
            "semantic_similarity": entry["semantic_similarity"],
            **entry["params"],
            **{f"raw_{key}": value for key, value in
               zip(DifferentiableEMOVIS.PARAM_NAMES, entry["raw_params"])},
        })
    write_csv(out / "convergence" / f"{name}.csv", rows)
    best = min(history, key=lambda entry: entry["total"])
    iterations = [entry["iteration"] for entry in history]
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(iterations, [entry["total"] for entry in history], label="Total")
    axes[1].plot(iterations, [entry["semantic"] for entry in history], label="Semantic")
    for loss in ("realism", "beauty", "regularization"):
        axes[2].plot(iterations, [entry[loss] for entry in history], label=loss)
    for key in DifferentiableEMOVIS.PARAM_NAMES:
        axes[3].plot(iterations, [entry["params"][key] for entry in history], label=key)
    for axis, label in zip(axes, ("Total loss", "Semantic loss", "Other loss terms", "Mapped DoFs")):
        axis.set_ylabel(label)
        axis.axvline(best["iteration"], color="gray", linestyle="--", linewidth=0.8)
        axis.legend(fontsize=8, loc="best")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Iteration (pre-optimizer-step evaluation)")
    fig.suptitle(f"{name}: all iterations; selected best = {best['iteration']}")
    fig.tight_layout()
    fig.savefig(out / "convergence" / f"{name}.png", dpi=110)
    plt.close(fig)


def make_grid(out, names):
    cell, label_height, columns = 160, 38, 10
    canvas = Image.new("RGB", (columns * cell, ((len(names) + columns - 1) // columns)
                               * (cell + label_height)), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(names):
        x, y = (index % columns) * cell, (index // columns) * (cell + label_height)
        with Image.open(out / "previews" / f"{name}.jpg") as image:
            canvas.paste(image.resize((cell, cell)), (x, y + label_height))
        draw.text((x + 5, y + 8), name, fill="white")
    canvas.save(out / "grid.jpg", quality=94)


def run_image(dng, out, words, iterations):
    if (out / "results.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for directory in ("full_resolution", "previews", "convergence"):
        (out / directory).mkdir(exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    start = time.time()
    image = load_and_preprocess_raw(str(dng), 512).cuda()
    native_input = load_and_preprocess_raw(str(dng), None).cuda()
    model = experiment.SigLIP("cuda")
    model.model.requires_grad_(False)
    mean, std = experiment.vlm_norm_stats("cuda")
    if experiment.VLM_NORM != "siglip":
        raise ValueError("This evaluation requires the locked SigLIP normalization.")
    isp = DifferentiableEMOVIS("cuda")
    with torch.no_grad():
        neutral = isp(image)
        guards = experiment.build_guards(model, encode_image(model, neutral, mean, std))
        native_neutral = isp(native_input)
        save_native(out / "full_resolution" / "neutral.png", native_neutral)
        native_feature = encode_image(model, native_neutral, mean, std)
        native_guards = {name: (scorer, scorer(native_feature)) for name, (scorer, _) in guards.items()}
    save_preview(out / "previews" / "neutral.jpg", neutral)
    files = ("differentiable_isp.py", "exp07_emovis_diff.py", Path(__file__).name)
    result = {
        "description": "Post-gamut-fix full-corpus evaluation",
        "dng": dng.name,
        "dng_path": str(dng),
        "config": {
            "vlm": experiment.VLM_NAME, "pretrained": "webli", "vlm_norm": experiment.VLM_NORM,
            "semantic_weight": experiment.SEMANTIC_WEIGHT,
            "realism_weight": experiment.REALISM_WEIGHT, "beauty_weight": experiment.BEAUTY_WEIGHT,
            "realism_prompts": experiment.REALISM_PROMPTS, "beauty_prompts": experiment.BEAUTY_PROMPTS,
            "reg_weight": REG_WEIGHT, "iters": iterations, "lr": LR,
            "defaults": DifferentiableEMOVIS.DEFAULT_PARAMS,
            "gamut_protection": "reference-bounded luminance-preserving chroma compression at all five stages",
            "optimization_size_wh": [512, 512],
            "native_size_wh": [native_input.shape[3], native_input.shape[2]],
            "export": "native uncropped RGB, lossless 16-bit PNG",
            "scoring": {
                "losses": "optimized 512x512 center crop; unquantized float RGB",
                "full_resolution_scores": "entire native frame resized to 224x224; saved 16-bit pixels",
                "quality_reference": "neutral ISP at the respective resolution/framing",
                "native_adaptation": "same DoFs; brightness/contrast medians recomputed on the full frame",
            },
            "seed": 0, "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
            "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in files},
        },
        "requested_words": words,
        "neutral": {
            "params": isp.get_params(),
            "losses": score_image(model, neutral, None, guards, isp, mean, std),
            "full_resolution_scores": score_image(model, native_neutral, None, native_guards, isp, mean, std),
            "full_resolution_image": "full_resolution/neutral.png",
        },
        "words": {}, "complete": False,
    }
    del native_neutral, neutral
    write_json(out / "results.json", result)
    print(f"RAW={dng.name}; native={result['config']['native_size_wh']}; words={len(words)}; "
          f"iterations={iterations}; SigLIP; weights=1:1:1; defaults={DifferentiableEMOVIS.DEFAULT_PARAMS}", flush=True)
    for index, word in enumerate([*words, None]):
        name = word if word is not None else "autotune"
        word_start = time.time()
        history = []
        params, similarity, losses, _ = experiment.optimize_for_word(
            word, image, guards, model, mean, std, num_iters=iterations,
            lr=LR, reg_weight=REG_WEIGHT, history=history,
        )
        if len(history) != iterations or not np.isfinite(losses).all():
            raise RuntimeError(f"Incomplete or nonfinite optimization for {name}")
        best = min(history, key=lambda entry: entry["total"])
        if params != best["params"]:
            raise AssertionError(f"Best parameter snapshot mismatch: {name}")
        with torch.no_grad():
            isp.raw_params.copy_(torch.tensor(best["raw_params"], device="cuda"))
            optimized = isp(image)
            check = score_image(model, optimized, word, guards, isp, mean, std)
            for loss in LOSSES:
                if abs(check[loss] - best[loss]) > 1e-6:
                    raise AssertionError(f"Selected/rendered {loss} mismatch for {name}")
            save_preview(out / "previews" / f"{name}.jpg", optimized)
            native = isp(native_input)
            save_native(out / "full_resolution" / f"{name}.png", native)
            native_scores = score_image(model, native, word, native_guards, isp, mean, std)
        del optimized, native
        save_trace(out, name, history)
        entry = {
            "params": params, "raw_params": best["raw_params"],
            "clip_sim": similarity if word is not None else None,
            "best_iteration": best["iteration"],
            "losses": {key: best[key] for key in LOSSES},
            "initial_losses": {key: history[0][key] for key in LOSSES},
            "final_losses": {key: history[-1][key] for key in LOSSES},
            "full_resolution_scores": native_scores,
            "history_file": f"convergence/{name}.csv",
            "convergence_plot": f"convergence/{name}.png",
            "full_resolution_image": f"full_resolution/{name}.png",
            "preview": f"previews/{name}.jpg",
            "time_sec": time.time() - word_start,
        }
        if word is None:
            result["autotune"] = params
            result["autotune_result"] = entry
        else:
            result["words"][word] = entry
        result["time_sec"] = time.time() - start
        write_json(out / "results.json", result)
        print(f"[{index + 1}/{len(words) + 1}] {name}: "
              + " ".join(f"{key}={best[key]:+.6f}" for key in LOSSES)
              + f" best_iter={best['iteration']} params={params} "
              + f"{entry['time_sec']:.1f}s", flush=True)
    make_grid(out, ["neutral", "autotune", *words])
    result["complete"] = True
    result["time_sec"] = time.time() - start
    result["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    write_json(out / "results.json", result)
    print(f"COMPLETE {dng.name}: {len(words)} words + autotune; {result['time_sec']:.1f}s; "
          f"peak CUDA {result['peak_cuda_memory_mb']:.0f} MB", flush=True)


def summarize(root, image_ids=None):
    image_ids = list(IMAGES) if image_ids is None else image_ids
    rows, table, histories, image_results = [], [], [], []
    for path in sorted(root.glob("img_*/results.json"), key=lambda p: int(p.parent.name[4:])):
        result = json.loads(path.read_text())
        image_id = int(path.parent.name[4:])
        image_results.append((path.parent, result))
        entries = list(result["words"].items())
        if "autotune_result" in result:
            entries.append(("autotune", result["autotune_result"]))
        for word, entry in entries:
            relative = path.parent.relative_to(root)
            row = {
                "image": image_id, "scene": IMAGES.get(image_id, str(image_id)), "word": word,
                **entry["losses"], **entry["params"], "best_iteration": entry["best_iteration"],
                "initial_total": entry["initial_losses"]["total"],
                **{f"native_{loss}": entry["full_resolution_scores"][loss] for loss in LOSSES},
                "native_width": result["config"]["native_size_wh"][0],
                "native_height": result["config"]["native_size_wh"][1],
                "full_resolution_image": str(relative / entry["full_resolution_image"]),
                "convergence_plot": str(relative / entry["convergence_plot"]),
                "history_file": str(relative / entry["history_file"]),
            }
            rows.append(row)
            cells = [f"{image_id}: {row['scene']}", word]
            cells.extend(f"{row[key]:+.6f}" for key in (*LOSSES, *DifferentiableEMOVIS.PARAM_NAMES))
            cells.append(str(row["best_iteration"]))
            links = [(row["full_resolution_image"], "16-bit full image"),
                     (row["convergence_plot"], "Convergence"), (row["history_file"], "Trace CSV")]
            table.append("<tr><td>" + "</td><td>".join(html.escape(cell) for cell in cells)
                         + "</td><td>" + " | ".join(f'<a href="{html.escape(link)}">{label}</a>'
                                                   for link, label in links)
                         + "</td></tr>")
            if word != "autotune":
                with (root / row["history_file"]).open() as stream:
                    history = list(csv.DictReader(stream))
                histories.append(np.array([[float(item[key]) for key in LOSSES] for item in history]))
    if not rows:
        raise ValueError(f"No finished word results found under {root}")
    write_csv(root / "scores.csv", rows)
    word_rows = [row for row in rows if row["word"] != "autotune"]
    report = {
        "images_started": len(image_results),
        "images_complete": sum(result["complete"] for _, result in image_results),
        "word_runs": len(word_rows),
        "expected_word_runs": len(image_ids) * len(experiment.WORDS),
        "autotune_runs": len(rows) - len(word_rows),
        "mean_optimization_losses": {key: float(np.mean([row[key] for row in word_rows])) for key in LOSSES},
        "mean_total_improvement": float(np.mean([row["initial_total"] - row["total"] for row in word_rows])),
    }
    write_json(root / "report.json", report)
    if len({history.shape for history in histories}) == 1:
        mean = np.stack(histories).mean(axis=0)
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        x = np.arange(1, len(mean) + 1)
        axes[0].plot(x, mean[:, 4], label="Mean total")
        axes[1].plot(x, mean[:, 0], label="Mean semantic")
        for index, key in enumerate(LOSSES[1:4], 1):
            axes[2].plot(x, mean[:, index], label=key)
        for axis in axes:
            axis.legend()
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel("Iteration")
        fig.suptitle(f"Mean convergence over {len(histories)} word-runs (not best-so-far)")
        fig.tight_layout()
        fig.savefig(root / "mean_convergence.png", dpi=130)
        plt.close(fig)
    keys = ["neutral", "autotune", "vivid", "night", "soft", "nostalgic", "psychedelic"]
    cell, header = 150, 25
    sheet = Image.new("RGB", (cell * (len(keys) + 1), header + len(image_results) * cell), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)
    for column, key in enumerate(keys, 1):
        draw.text((column * cell + 4, 5), key, fill="white")
    for index, (directory, _) in enumerate(image_results):
        y = header + index * cell
        draw.text((5, y + 10), directory.name, fill="white")
        for column, key in enumerate(keys, 1):
            preview = directory / "previews" / f"{key}.jpg"
            if preview.exists():
                with Image.open(preview) as image:
                    sheet.paste(image.resize((cell, cell)), (column * cell, y))
    sheet.save(root / "comparison_sheet.jpg", quality=95)
    headers = ["Image", "Word", *LOSSES, *DifferentiableEMOVIS.PARAM_NAMES, "Best iteration", "Files"]
    document = """<!doctype html><meta charset="utf-8"><title>Semantic ISP evaluation</title>
<style>body{font:14px sans-serif;margin:24px}table{border-collapse:collapse;font-size:12px}
td,th{border:1px solid #ccc;padding:6px;white-space:nowrap}th{position:sticky;top:0;background:#eee}
tr:nth-child(even){background:#f5f5f5}img{max-width:100%}input{padding:10px;width:360px}</style>
<h1>SigLIP semantic ISP: full-corpus evaluation</h1>
<p>Weights: semantic = realism = beauty = 1.0; regularization coefficient = 0.003.
Defaults: saturation +0.26, contrast +0.40, others 0. Negative realism/beauty loss means improvement; positive means degradation.
The best iterate minimizes the total loss, not semantic loss alone.</p>
<p>Table: scores on the optimized 512x512 center crop. All native-frame scores are separately
labeled in <a href="scores.csv">scores.csv</a> and each image's results.json.
Native exports are uncropped, lossless 16-bit RGB PNGs. Brightness/contrast medians are
recomputed on the entire native frame. JPEG previews are only for browsing.</p>
<p><a href="scores.csv">Download all scores and DoFs (CSV)</a> |
<a href="report.json">Run status and summary</a></p>
"""
    document += f"<p>{report['word_runs']}/{report['expected_word_runs']} word-runs; "
    document += f"{report['images_complete']}/{len(image_ids)} images complete.</p>"
    document += '<img src="comparison_sheet.jpg"><br><img src="mean_convergence.png">'
    document += '<p><input id="filter" placeholder="Filter by image or word"></p><table><thead><tr>'
    document += "".join(f"<th>{html.escape(key)}</th>" for key in headers)
    document += "</tr></thead><tbody>" + "\n".join(table) + "</tbody></table>"
    document += """<script>document.getElementById('filter').addEventListener('input',e=>{
const q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>{
r.hidden=!r.textContent.toLowerCase().includes(q);});});</script>"""
    (root / "index.html").write_text(document)
    print(json.dumps(report, indent=2), flush=True)


def verify(root, image_ids=None):
    image_ids = list(IMAGES) if image_ids is None else image_ids
    expected_directories = {f"img_{number}" for number in image_ids}
    actual_directories = {path.name for path in root.glob("img_*") if path.is_dir()}
    if actual_directories != expected_directories:
        raise ValueError(f"Image directory mismatch: {actual_directories ^ expected_directories}")
    checked_words, checked_traces, checked_native = 0, 0, 0
    expected_config = {
        "vlm": experiment.VLM_NAME, "pretrained": "webli", "vlm_norm": "siglip",
        "semantic_weight": 1.0, "realism_weight": 1.0, "beauty_weight": 1.0,
        "reg_weight": REG_WEIGHT, "iters": 500, "lr": LR,
        "defaults": DifferentiableEMOVIS.DEFAULT_PARAMS,
    }
    raw_default = np.array([DifferentiableEMOVIS._to_raw(key, DifferentiableEMOVIS.DEFAULT_PARAMS[key])
                            for key in DifferentiableEMOVIS.PARAM_NAMES])
    for image_id in image_ids:
        directory = root / f"img_{image_id}"
        result = json.loads((directory / "results.json").read_text())
        if not result["complete"] or result["requested_words"] != experiment.WORDS:
            raise ValueError(f"Incomplete or wrong word list: {directory}")
        if set(result["words"]) != set(experiment.WORDS) or "autotune_result" not in result:
            raise ValueError(f"Missing word/autotune results: {directory}")
        for key, value in expected_config.items():
            if result["config"][key] != value:
                raise ValueError(f"Unexpected {key}: {directory}")
        with rawpy.imread(result["dng_path"]) as raw:
            width, height = raw.sizes.width, raw.sizes.height
            if raw.sizes.flip in (5, 6):
                width, height = height, width
        if result["config"]["native_size_wh"] != [width, height]:
            raise ValueError(f"Native dimensions do not match RAW metadata: {directory}")
        entries = [*result["words"].items(), ("autotune", result["autotune_result"])]
        for name, entry in entries:
            with (directory / entry["history_file"]).open() as stream:
                history = list(csv.DictReader(stream))
            if [int(row["iteration"]) for row in history] != list(range(1, 501)):
                raise ValueError(f"Missing/duplicate iterations: {directory}/{name}")
            numbers = np.array([[float(row[key]) for key in
                                 (*LOSSES, *DifferentiableEMOVIS.PARAM_NAMES,
                                  *(f"raw_{key}" for key in DifferentiableEMOVIS.PARAM_NAMES))]
                                for row in history])
            if not np.isfinite(numbers).all():
                raise ValueError(f"Nonfinite loss/parameters: {directory}/{name}")
            np.testing.assert_allclose(numbers[:, :4].sum(axis=1), numbers[:, 4], rtol=0, atol=1e-6)
            raw_parameters = numbers[:, 10:]
            regularization = REG_WEIGHT * np.square(raw_parameters - raw_default).sum(axis=1)
            np.testing.assert_allclose(regularization, numbers[:, 3], rtol=0, atol=1e-6)
            for index, key in enumerate(DifferentiableEMOVIS.PARAM_NAMES):
                lo, hi = DifferentiableEMOVIS.PARAM_RANGES[key]
                if np.any((numbers[:, 5 + index] < lo) | (numbers[:, 5 + index] > hi)):
                    raise ValueError(f"Out-of-range {key}: {directory}/{name}")
            best_index = int(numbers[:, 4].argmin())
            if entry["best_iteration"] != best_index + 1:
                raise ValueError(f"Incorrect best iterate: {directory}/{name}")
            for index, key in enumerate(LOSSES):
                np.testing.assert_allclose(entry["losses"][key], numbers[best_index, index], rtol=0, atol=1e-9)
                np.testing.assert_allclose(entry["initial_losses"][key], numbers[0, index], rtol=0, atol=1e-9)
                np.testing.assert_allclose(entry["final_losses"][key], numbers[-1, index], rtol=0, atol=1e-9)
            for index, key in enumerate(DifferentiableEMOVIS.PARAM_NAMES):
                if entry["params"][key] != float(history[best_index][key]):
                    raise ValueError(f"Incorrect selected DoF: {directory}/{name}/{key}")
            native = entry["full_resolution_scores"]
            if not all(np.isfinite(native[key]) for key in LOSSES):
                raise ValueError(f"Nonfinite native score: {directory}/{name}")
            np.testing.assert_allclose(sum(native[key] for key in LOSSES[:-1]), native["total"], rtol=0, atol=1e-6)
            np.testing.assert_allclose(native["regularization"], entry["losses"]["regularization"], rtol=0, atol=1e-6)
            for key in ("convergence_plot", "history_file", "full_resolution_image", "preview"):
                if not (directory / entry[key]).is_file():
                    raise FileNotFoundError(directory / entry[key])
            checked_traces += 1
            checked_words += name != "autotune"
        for name in [*experiment.WORDS, "autotune", "neutral"]:
            path = directory / "full_resolution" / f"{name}.png"
            with path.open("rb") as stream:
                header = stream.read(33)
            if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                raise ValueError(f"Invalid native PNG: {path}")
            w, h, depth, color, _, _, _ = struct.unpack(">IIBBBBB", header[16:29])
            if (w, h, depth, color) != (width, height, 16, 2):
                raise ValueError(f"Not native 16-bit RGB: {path}: {(w, h, depth, color)}")
            checked_native += 1
    validation = {
        "verified": True, "images": len(image_ids), "words_per_image": len(experiment.WORDS),
        "word_runs": checked_words, "autotune_runs": checked_traces - checked_words,
        "iterations_per_trace": 500, "total_iterations": checked_traces * 500,
        "native_16bit_rgb_images": checked_native,
        "checks": ["locked configuration", "complete word/image sets", "all loss decompositions",
                   "regularization vs exact raw parameters", "minimum-total-loss best iterates",
                   "all five selected DoFs", "native loss decompositions", "trace/graph/image files",
                   "PNG bit depth, RGB encoding and RAW-metadata dimensions"],
    }
    write_json(root / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--dng", type=Path, default=os.environ.get("DNG_PATH"))
    parser.add_argument("--out", type=Path, default=os.environ.get("RESULTS_DIR"))
    parser.add_argument("--words", nargs="+", choices=experiment.WORDS)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--images", nargs="+", type=int,
                        help="Image IDs expected by --summarize/--verify (default: the ten Exp 19 images)")
    args = parser.parse_args()
    if args.verify is not None:
        verify(args.verify, args.images)
    elif args.summarize is not None:
        summarize(args.summarize, args.images)
    elif args.dng is None or args.out is None or args.iterations < 1:
        parser.error("Provide --dng/--out (or DNG_PATH/RESULTS_DIR) and a positive iteration count")
    else:
        run_image(args.dng, args.out, args.words or experiment.WORDS, args.iterations)


if __name__ == "__main__":
    main()
