"""Quick visual pilot: FiveK image/sentence pairs and neutral comparisons."""

import argparse
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
import open_clip
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from autotune import BEAUTY_PROMPTS, REALISM_PROMPTS, optimize
from semantic_isp import CONTROL_NAMES, linear_to_srgb8, parameters_from_controls, render_linear


DEFAULT_DATA = Path("/data/data1/FiveK/default_isp_linear_720p")
DEFAULT_CONFIG = Path("/data/data1/semantic_isp/results/exp81_joint_descriptor_tail_final/direct_soft_calibration.json")
DEFAULT_SENTENCES = Path(__file__).with_name("sentences_agent_estimates.json")
DEFAULT_OUT = Path("/data/data1/semantic_isp/results/semantic_parameter_learning/fivek_quick_preview")
IMAGE_PREFIXES = ("a0001", "a0004", "a0005")
SENTENCE_IDS = ("D1_002", "D1_003", "D1_008", "D1_004", "D2_011")
SHARED_SENTENCE = "D1_002"


def load_linear_png(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint16 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a 16-bit three-channel linear RGB PNG: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def resize_short_side(image, size):
    height, width = image.shape[-2:]
    scale = size / min(height, width)
    dimensions = (max(1, round(height * scale)), max(1, round(width * scale)))
    return F.interpolate(image, dimensions, mode="area")


def write_render(path, image):
    if not torch.isfinite(image).all():
        raise ValueError(f"Nonfinite rendered image: {path}")
    pixels = linear_to_srgb8(image)[0].permute(1, 2, 0).detach().cpu().numpy()
    Image.fromarray(pixels).save(path)


def write_sheet(path, entries, title):
    columns, tile_width, tile_height, title_height, label_height = 3, 480, 300, 58, 64
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, title_height + rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    title_font = ImageFont.truetype(str(font_path), 21) if font_path.exists() else font
    draw.text((12, 14), title, fill="#17212b", font=title_font)
    for index, (label, image_path) in enumerate(entries):
        left = (index % columns) * tile_width
        top = title_height + (index // columns) * (tile_height + label_height)
        label_text = "\n".join(textwrap.wrap(label, width=48))
        draw.multiline_text((left + 8, top + 5), label_text, fill="#17212b", font=font, spacing=3)
        with Image.open(image_path) as source:
            tile = source.convert("RGB")
            tile.thumbnail((tile_width - 12, tile_height), Image.Resampling.LANCZOS)
            sheet.paste(tile, (left + (tile_width - tile.width) // 2,
                               top + label_height + (tile_height - tile.height) // 2))
    sheet.save(path)


def run(args):
    if args.iters < 1 or args.size < 1 or args.lr <= 0:
        raise ValueError("Iterations, image size, and learning rate must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    config = json.loads(args.config.read_text())
    sentence_data = json.loads(args.sentences.read_text())
    sentence_map = {entry["id"]: entry for entry in sentence_data["sentences"]}
    if any(sentence_id not in sentence_map for sentence_id in SENTENCE_IDS):
        raise ValueError("A requested preview sentence is missing from the agent-estimate table")
    image_paths = {}
    for prefix in IMAGE_PREFIXES:
        matches = sorted(args.data.glob(prefix + "-*.png"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one FiveK image for {prefix}, found {len(matches)}")
        image_paths[prefix] = matches[0]
    torch.manual_seed(0)
    torch.set_num_threads(2)
    semantic_model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=config["pretrained"], device=args.device,
        force_quick_gelu=config.get("force_quick_gelu", config["pretrained"] == "openai"))
    quality_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli", device=args.device)
    semantic_model.eval().requires_grad_(False)
    quality_model.eval().requires_grad_(False)
    semantic_tokenizer = open_clip.get_tokenizer(config["model"])
    quality_tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    with torch.no_grad():
        text_embeddings = F.normalize(semantic_model.encode_text(semantic_tokenizer(
            [sentence_map[sentence_id]["text"] for sentence_id in SENTENCE_IDS]).to(args.device)).float(), dim=-1)
        quality_embeddings = F.normalize(quality_model.encode_text(quality_tokenizer(
            list(REALISM_PROMPTS + BEAUTY_PROMPTS)).to(args.device)).float(), dim=-1)

    args.out.mkdir(parents=True, exist_ok=True)
    images_out = args.out / "images"
    images_out.mkdir()
    records = []
    for image_id in IMAGE_PREFIXES:
        native = load_linear_png(image_paths[image_id]).to(args.device)
        optimization_image = resize_short_side(native, args.size)
        neutral = torch.tensor([0., 0., 1., 1., 0., 0.], dtype=native.dtype, device=native.device)
        with torch.no_grad():
            write_render(images_out / f"{image_id}_neutral.png",
                         render_linear(native, parameters_from_controls(neutral, config)))
        selected_ids = SENTENCE_IDS if image_id == IMAGE_PREFIXES[0] else (SHARED_SENTENCE,)
        for sentence_id in selected_ids:
            sentence_index = SENTENCE_IDS.index(sentence_id)
            sentence = sentence_map[sentence_id]
            result = optimize(optimization_image, semantic_model, quality_model,
                text_embeddings[sentence_index:sentence_index + 1], quality_embeddings, config,
                iterations=args.iters, learning_rate=args.lr)
            controls = torch.tensor([result["controls"][name] for name in CONTROL_NAMES],
                                    device=args.device, dtype=native.dtype)
            render_path = images_out / f"{image_id}_{sentence_id}.png"
            with torch.no_grad():
                write_render(render_path, render_linear(native, parameters_from_controls(controls, config)))
            record = {"image_id": image_id, "image_path": str(image_paths[image_id]),
                      "sentence_id": sentence_id, "sentence": sentence["text"],
                      "agent_estimated_directions": sentence["ratings"],
                      "render_path": str(render_path.relative_to(args.out)), **result}
            records.append(record)
            print(f"{image_id} {sentence_id}: iteration={result['iteration']} "
                  f"palette_start={result['palette_start']:+.2f} loss={result['losses']['total']:.5f}", flush=True)
        del native, optimization_image
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    first_image = IMAGE_PREFIXES[0]
    same_image_entries = [(f"{first_image} | Neutral", images_out / f"{first_image}_neutral.png")]
    same_image_entries.extend((f"{sentence_id} | {sentence_map[sentence_id]['text']}",
                               images_out / f"{first_image}_{sentence_id}.png") for sentence_id in SENTENCE_IDS)
    write_sheet(args.out / "same_image_different_sentences.png", same_image_entries,
                f"FiveK {first_image}: neutral and five different sentences")
    shared_entries = []
    for image_id in IMAGE_PREFIXES:
        shared_entries.extend([(f"{image_id} | Neutral", images_out / f"{image_id}_neutral.png"),
                               (f"{image_id} | {sentence_map[SHARED_SENTENCE]['text']}",
                                images_out / f"{image_id}_{SHARED_SENTENCE}.png")])
    write_sheet(args.out / "different_images_same_sentence.png", shared_entries,
                "FiveK: same sentence, three images (neutral and tuned)")
    report = {"dataset": "FiveK preprocessed 16-bit linear RGB PNG; not original camera RAW",
              "semantic_model": f"{config['model']} ({config['pretrained']})",
              "quality_model": "ViT-B-16-SigLIP (webli)",
              "quality_caveat": "Realism and beauty are SigLIP prompt-pair proxies, not independent human assessments.",
              "config": str(args.config), "sentences_source": str(args.sentences),
              "optimization_short_side": args.size, "render_source": "Full 720p linear PNG",
              "iterations_per_start": args.iters, "learning_rate": args.lr,
              "palette_starts": [0., -.02, .02], "records": records}
    (args.out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.out), "pairs": len(records),
                      "same_image_sheet": str(args.out / "same_image_different_sentences.png"),
                      "same_sentence_sheet": str(args.out / "different_images_same_sentence.png")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=.03)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()