"""Generate text-image-ISP training pairs in a standalone results directory."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from autotune import BEAUTY_PROMPTS, REALISM_PROMPTS, optimize
from differentiable_isp import load_and_preprocess_raw
from pole_normalization import CONTROL_NAMES, pole_endpoints

DEFAULT_CONFIG = Path("/data/data1/semantic_isp/results/exp81_joint_descriptor_tail_final/direct_soft_calibration.json")
DEFAULT_DATA = Path("/data/data1/EmotionISP_dataset/dng_filtered")
DEFAULT_SENTENCES = Path(__file__).with_name("sentences.json")
DEFAULT_OUT = Path("/data/data1/semantic_isp/results/semantic_parameter_learning/p90_dataset")
QUALITY_MODEL = "ViT-B-16-SigLIP"
QUALITY_PRETRAINED = "webli"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def validate_sentences(path):
    rows = json.loads(path.read_text())["sentences"]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Sentence IDs must be unique and nonempty")
    if len({row["text"].strip().casefold() for row in rows}) != len(rows):
        raise ValueError("Sentence text must be unique across splits")
    for row in rows:
        if not row["id"].replace("_", "").isalnum() or not row["text"].strip():
            raise ValueError(f"Invalid sentence record: {row}")
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError(f"Unknown sentence split: {row['split']}")
    if {row["split"] for row in rows} != {"train", "validation", "test"}:
        raise ValueError("Sentence table must contain train, validation, and test splits")
    return rows


def run(args):
    if args.iters < 1 or args.size < 1 or args.lr <= 0:
        raise ValueError("Iterations, image size and learning rate must be positive")
    if not args.train_images or not args.test_images or set(args.train_images) & set(args.test_images):
        raise ValueError("Training and held-out image IDs must be nonempty and disjoint")
    image_ids = args.train_images + args.test_images
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Image IDs must be unique")
    image_paths = {image_id: args.data / f"{image_id}.dng" for image_id in image_ids}
    for path in image_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    sentences = validate_sentences(args.sentences)
    config = json.loads(args.config.read_text())
    if args.semantic_backbone == "siglip":
        semantic_model_name, semantic_pretrained = QUALITY_MODEL, QUALITY_PRETRAINED
        config["image_mean"] = [.5, .5, .5]
        config["image_std"] = [.5, .5, .5]
    else:
        semantic_model_name, semantic_pretrained = config["model"], config["pretrained"]
    endpoints = pole_endpoints(config) if args.parameterization == "p90" else None
    source_paths = (Path(__file__), Path(__file__).with_name("autotune.py"),
                    Path(__file__).with_name("pole_normalization.py"))
    manifest = {"config": str(args.config.resolve()), "config_sha256": sha256(args.config),
                "sentences": str(args.sentences.resolve()), "sentences_sha256": sha256(args.sentences),
                "source_sha256": {path.name: sha256(path) for path in source_paths},
                "image_paths": {str(key): str(value.resolve()) for key, value in image_paths.items()},
                "image_splits": {"train": args.train_images, "test": args.test_images},
                "semantic_backbone": args.semantic_backbone,
                "semantic_model": semantic_model_name, "semantic_pretrained": semantic_pretrained,
                "semantic_image_mean": config.get("image_mean", [.48145466, .4578275, .40821073]),
                "semantic_image_std": config.get("image_std", [.26862954, .26130258, .27577711]),
                "quality_model": QUALITY_MODEL, "quality_pretrained": QUALITY_PRETRAINED,
                "realism_prompts": REALISM_PROMPTS, "beauty_prompts": BEAUTY_PROMPTS,
                "quality_warning": "SigLIP prompt-pair scores are proxies, not verified measures of realism or beauty.",
                "iterations": args.iters, "image_short_side": args.size, "learning_rate": args.lr,
                "palette_starts": [0., -.02, .02],
                "semantic_weight": 1., "realism_weight": 1., "beauty_weight": 1., "regularization_weight": .003,
                "parameterization": args.parameterization,
                "parameterization_description": (
                    "Six pole-relative controls in [-1,1], mapped piecewise linearly to physical P90 endpoints; raw optimizer coordinates use tanh."
                    if endpoints is not None else
                    "Six physical controls jointly tuned through the existing differentiable ISP; tanh-bounded by the supplied configuration."),
                "pole_endpoints": ({name: {"negative": float(endpoints[0][index]),
                    "neutral": float(endpoints[1][index]), "positive": float(endpoints[2][index])}
                    for index, name in enumerate(CONTROL_NAMES)} if endpoints is not None else None)}
    manifest = json.loads(json.dumps(manifest))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing manifest differs; use a new output directory")
    elif any(args.out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.out}")
    else:
        save_json(manifest_path, manifest)
    record_dir = args.out / "records"
    record_dir.mkdir(exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    semantic_model, _, _ = open_clip.create_model_and_transforms(
        semantic_model_name, pretrained=semantic_pretrained, device=args.device,
        force_quick_gelu=(config.get("force_quick_gelu", semantic_pretrained == "openai")
                          if args.semantic_backbone == "clip" else False))
    if args.semantic_backbone == "siglip":
        quality_model = semantic_model
    else:
        quality_model, _, _ = open_clip.create_model_and_transforms(
            QUALITY_MODEL, pretrained=QUALITY_PRETRAINED, device=args.device)
    semantic_model.eval().requires_grad_(False)
    quality_model.eval().requires_grad_(False)
    semantic_tokenizer = open_clip.get_tokenizer(semantic_model_name)
    quality_tokenizer = open_clip.get_tokenizer(QUALITY_MODEL)
    with torch.no_grad():
        text_embeddings = F.normalize(semantic_model.encode_text(
            semantic_tokenizer([row["text"] for row in sentences]).to(args.device)).float(), dim=-1)
        quality_text_embeddings = F.normalize(quality_model.encode_text(
            quality_tokenizer(list(REALISM_PROMPTS + BEAUTY_PROMPTS)).to(args.device)).float(), dim=-1)
    embeddings_path = args.out / "text_embeddings.npz"
    if embeddings_path.exists():
        with np.load(embeddings_path, allow_pickle=False) as previous:
            if list(previous["ids"]) != [row["id"] for row in sentences] or not np.allclose(
                    previous["embeddings"], text_embeddings.cpu().numpy(), atol=1e-5, rtol=1e-5):
                raise ValueError("Existing text embeddings do not match this run")
    else:
        np.savez_compressed(embeddings_path, ids=np.asarray([row["id"] for row in sentences]),
                            embeddings=text_embeddings.cpu().numpy())
    records = []
    for image_id in image_ids:
        image_split = "train" if image_id in args.train_images else "test"
        image = load_and_preprocess_raw(str(image_paths[image_id]), args.size).to(args.device)
        for index, sentence in enumerate(sentences):
            record_path = record_dir / f"{sentence['id']}_img_{image_id}.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record["sentence"] != sentence["text"] or record["image_id"] != image_id:
                    raise ValueError(f"Existing record identity mismatch: {record_path}")
            else:
                result = optimize(image, semantic_model, quality_model, text_embeddings[index:index + 1],
                                  quality_text_embeddings, config, iterations=args.iters, learning_rate=args.lr,
                                  parameterization=args.parameterization)
                record = {"sentence_id": sentence["id"], "sentence": sentence["text"],
                          "sentence_split": sentence["split"], "image_id": image_id,
                          "image_split": image_split, "image_path": str(image_paths[image_id].resolve()),
                          **result}
                save_json(record_path, record)
                print(f"image {image_id} {sentence['id']}: loss={result['losses']['total']:.5f}", flush=True)
            records.append(record)
        del image
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    save_json(args.out / "dataset.json", {"manifest": "manifest.json", "records": records})
    print(json.dumps({"output": str(args.out), "sentences": len(sentences), "images": len(image_ids),
                      "pairs": len(records)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-images", type=int, nargs="+", default=[1, 28])
    parser.add_argument("--test-images", type=int, nargs="+", default=[2])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=.03)
    parser.add_argument("--parameterization", choices=("p90", "physical"), default="p90")
    parser.add_argument("--semantic-backbone", choices=("siglip", "clip"), default="siglip")
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()