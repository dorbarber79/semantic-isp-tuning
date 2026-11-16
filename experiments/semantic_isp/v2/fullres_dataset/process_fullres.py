"""Consume completed FiveK-FullRes-v1 acquisitions into native-size V2 fits."""

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, ImageCms
import rawpy
import tifffile
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

SCRIPT_PATH = Path(__file__).resolve()
V2_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE = SCRIPT_PATH.parents[4]
DEFAULT_DATASET_ROOT = Path("/data/data1/FiveK/C_data/datasets/FiveK-FullRes-v1")
CONFIG_PATH = V2_ROOT / "calibration/exp81/direct_soft_calibration.json"
TINT_LIMIT = 0.5
for path in (V2_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from differentiable_isp import load_and_preprocess_raw
from semantic_isp import (CONTROL_NAMES, EXTENDED_CONTROL_NAMES, EXTENDED_BOUNDS,
                          parameters_from_controls, parameters_from_extended_controls,
                          render_linear)
from axis_isp import linear_to_srgb8
from crop_alignment import align_expert_crop, oriented_default_crop_rect

PROPHOTO_TO_XYZ_D50 = np.array([
    [0.7976749, 0.1351917, 0.0313534],
    [0.2880402, 0.7118741, 0.0000857],
    [0.0, 0.0, 0.8252100],
], dtype=np.float64)
D50_TO_D65 = np.array([
    [0.9555766, -0.0230393, 0.0631636],
    [-0.0282895, 1.0099416, 0.0210077],
    [0.0122982, -0.0204830, 1.3299098],
], dtype=np.float64)
XYZ_D65_TO_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
], dtype=np.float64)
PROPHOTO_TO_LINEAR_SRGB = XYZ_D65_TO_SRGB @ D50_TO_D65 @ PROPHOTO_TO_XYZ_D50


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_paths(root, scene_id):
    paths = {"raw": root / "source" / "dng" / f"{scene_id}.dng"}
    paths.update({f"expert_{expert}": root / "source" / f"tiff16_{expert.lower()}" / f"{scene_id}.tif"
                  for expert in "ABCDE"})
    return paths


def postprocess_dimensions_wh(sizes):
    width, height = int(sizes.iwidth), int(sizes.iheight)
    if int(sizes.flip) in (5, 6):
        width, height = height, width
    return [width, height]


def align_target_to_native(target, native_shape, tolerance=.002):
    source_shape = tuple(int(value) for value in target.shape[-2:])
    native_shape = tuple(int(value) for value in native_shape)
    source_aspect = source_shape[1] / source_shape[0]
    native_aspect = native_shape[1] / native_shape[0]
    if abs(source_aspect - native_aspect) > tolerance:
        raise ValueError(f"Expert TIFF aspect {source_shape} does not match oriented RAW {native_shape}")
    if source_shape != native_shape:
        target = F.interpolate(target, size=native_shape, mode="area").clamp(0, 1)
    return target, {
        "source_dimensions_wh": [source_shape[1], source_shape[0]],
        "aligned_dimensions_wh": [native_shape[1], native_shape[0]],
        "method": "none" if source_shape == native_shape else "area resampling after aspect-ratio validation",
    }


def manifest_source_hashes(record):
    expected = {}
    for source in record.get("sources", []):
        if not isinstance(source, dict):
            continue
        role = str(source.get("role", "")).lower()
        photographer = str(source.get("photographer", source.get("expert", ""))).upper()
        if role in {"dng", "raw"}:
            key = "raw"
        elif role in {"tiff", "tiff16", "expert"} and photographer in "ABCDE":
            key = f"expert_{photographer}"
        else:
            continue
        digest = source.get("sha256")
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            expected[key] = digest.lower()
    return expected


def parse_pair(value):
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        return None
    try:
        result = [int(item) for item in parts]
    except (TypeError, ValueError):
        return None
    return result if len(result) == 2 else None


def dng_default_crop(acquisition, raw_sizes):
    dng = next((item for item in acquisition.get("sources", []) if item.get("role") in {"dng", "raw"}), {})
    tags = dng.get("exiftool", {})
    origin = parse_pair(tags.get("SubIFD:DefaultCropOrigin"))
    size = parse_pair(tags.get("SubIFD:DefaultCropSize"))
    source = "DNG SubIFD DefaultCropOrigin/DefaultCropSize"
    if origin is None or size is None:
        origin = [int(raw_sizes.crop_left_margin - raw_sizes.left_margin),
                  int(raw_sizes.crop_top_margin - raw_sizes.top_margin)]
        size = [int(raw_sizes.crop_width), int(raw_sizes.crop_height)]
        source = "rawpy camera crop margins fallback"
    return {"origin_xy": origin, "size_wh": size, "source": source}


def is_complete_record(record):
    status = str(record.get("status", record.get("state", ""))).strip().lower()
    if record.get("complete") is True or status in {"complete", "completed"}:
        return True
    sources = record.get("sources")
    if not isinstance(sources, list) or len(sources) != 6:
        return False
    raw_present = False
    experts = set()
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            return False
        digest = source.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            return False
        role = str(source.get("role", "")).lower()
        photographer = str(source.get("photographer", source.get("expert", ""))).upper()
        if role in {"dng", "raw"} and not raw_present:
            raw_present = True
        elif role in {"tiff", "tiff16", "expert"} and photographer in "ABCDE":
            if photographer in experts:
                return False
            experts.add(photographer)
        else:
            return False
    return raw_present and experts == set("ABCDE")


def scene_id_for(record):
    scene_id = record.get("scene_id", record.get("image_id", record.get("id")))
    if not isinstance(scene_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", scene_id):
        return None
    return scene_id


def load_acquisition_records(path):
    latest = {}
    malformed = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                malformed.append({"line": line_number, "error": str(error)})
                continue
            scene_id = scene_id_for(record) if isinstance(record, dict) else None
            if scene_id is None:
                malformed.append({"line": line_number, "error": "Missing or invalid scene id"})
                continue
            latest[scene_id] = (record, line_number, canonical_hash(record))
    return latest, malformed


def local_file_signature(root, paths):
    signature = {}
    for name, path in paths.items():
        try:
            info = path.lstat()
            signature[name] = {"size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
                               "regular_file": stat.S_ISREG(info.st_mode)}
        except FileNotFoundError:
            signature[name] = None
    return signature


def can_reuse_completed_output(root, prior, source_signature):
    if not prior or prior.get("status") != "complete" or prior.get("source_signature") != source_signature:
        return False
    relative = Path(str(prior.get("output_directory", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return False
    return (Path(root) / relative).is_dir()


def validate_regular_in_root(root, paths):
    root_path = Path(root).absolute()
    if root_path.is_symlink():
        raise ValueError("Dataset root must not be a symlink")
    root_resolved = root_path.resolve(strict=True)
    for name, path in paths.items():
        path = Path(path).absolute()
        relative = path.relative_to(root_path)
        current = root_path
        info = None
        for index, part in enumerate(relative.parts):
            current = current / part
            try:
                info = current.lstat()
            except FileNotFoundError as error:
                raise ValueError(f"Missing source file {name}: {relative}") from error
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"Symlink in source path: {relative}")
            if index < len(relative.parts) - 1 and not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"Non-directory in source path: {relative}")
        if info is None or not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Source is not a regular file: {relative}")
        if info.st_nlink != 1:
            raise ValueError(f"Source has unexpected hard links: {relative}")
        if not current.resolve(strict=True).is_relative_to(root_resolved):
            raise ValueError(f"Source resolves outside dataset root: {name}")
        if info.st_size <= 0:
            raise ValueError(f"Empty source file: {relative}")


def ensure_directory_inside_root(root, path):
    root_path = Path(root).absolute()
    if root_path.is_symlink():
        raise ValueError("Dataset root must not be a symlink")
    root_resolved = root_path.resolve(strict=True)
    path = Path(path).absolute()
    relative = path.relative_to(root_path)
    current = root_path
    for part in relative.parts:
        current = current / part
        if not current.exists() and not current.is_symlink():
            current.mkdir()
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Dataset directory must be a real directory: {relative}")
        if not current.resolve(strict=True).is_relative_to(root_resolved):
            raise ValueError(f"Dataset directory resolves outside root: {relative}")
    return path


def validate_manifest_file(root, path):
    root_path = Path(root).absolute()
    if root_path.is_symlink():
        raise ValueError("Dataset root must not be a symlink")
    root_resolved = root_path.resolve(strict=True)
    path = Path(path).absolute()
    relative = path.relative_to(root_path)
    current = root_path
    info = None
    for index, part in enumerate(relative.parts):
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Symlink in manifest path: {relative}")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Non-directory in manifest path: {relative}")
    if info is None or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"Manifest must be an independent regular file: {relative}")
    if not current.resolve(strict=True).is_relative_to(root_resolved):
        raise ValueError(f"Manifest resolves outside dataset root: {path.name}")


def prophoto_tiff_to_linear_srgb(path, verify_icc=True):
    with tifffile.TiffFile(path) as source:
        page = source.pages[0]
        profile_tag = page.tags.get("InterColorProfile")
        if profile_tag is None:
            raise ValueError(f"Missing embedded ICC profile: {path.name}")
        profile_bytes = profile_tag.value
        encoded = page.asarray()
    if encoded.dtype != np.uint16 or encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError(f"Expected a 16-bit RGB TIFF: {path.name}")
    profile = ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes))
    description = ImageCms.getProfileDescription(profile).strip()
    if "ProPhoto RGB" not in description:
        raise ValueError(f"Expected ProPhoto RGB ICC, got {description!r}: {path.name}")
    encoded_float = encoded.astype(np.float64) / 65535.0
    prophoto_linear = np.where(encoded_float <= 16 / 512,
                               encoded_float / 16.0,
                               np.maximum(encoded_float, 0.0) ** 1.8)
    srgb_unclipped = np.einsum("ij,hwj->hwi", PROPHOTO_TO_LINEAR_SRGB, prophoto_linear)
    out_of_gamut_fraction = float(np.mean((srgb_unclipped < 0) | (srgb_unclipped > 1)))
    srgb_linear = np.clip(srgb_unclipped, 0.0, 1.0).astype(np.float32)
    validation = None
    if verify_icc:
        sample = np.rint(encoded_float[::16, ::16] * 255).astype(np.uint8)
        source_8bit = Image.fromarray(sample, "RGB")
        srgb_profile = ImageCms.createProfile("sRGB")
        reference = np.asarray(ImageCms.profileToProfile(
            source_8bit, profile, srgb_profile, outputMode="RGB", renderingIntent=1), dtype=np.int16)
        linear_sample = srgb_linear[::16, ::16]
        manual_encoded = np.where(linear_sample <= .0031308, 12.92 * linear_sample,
                                  1.055 * np.maximum(linear_sample, .0031308) ** (1 / 2.4) - .055)
        manual_encoded = np.rint(np.clip(manual_encoded, 0, 1) * 255).astype(np.int16)
        difference = np.abs(reference - manual_encoded)
        validation = {"mean_abs_8bit": float(difference.mean()),
                      "p95_abs_8bit": float(np.percentile(difference, 95)),
                      "max_abs_8bit": int(difference.max())}
        if validation["mean_abs_8bit"] > 2.0 or validation["p95_abs_8bit"] > 3.0:
            raise ValueError(f"ProPhoto transform does not match ICC conversion: {validation}")
    del encoded, encoded_float, prophoto_linear, srgb_unclipped
    return srgb_linear, {"profile_description": description,
                         "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
                         "shape_hwc": list(srgb_linear.shape), "source_dtype": "uint16",
                         "out_of_gamut_fraction_before_clip": out_of_gamut_fraction,
                         "icc_check": validation}


def validate_sources(root, paths, acquisition, neutral_shape=None):
    validate_regular_in_root(root, paths)
    with rawpy.imread(str(paths["raw"])) as raw:
        sizes = raw.sizes
        flip = int(sizes.flip)
        visible_height, visible_width = int(sizes.iheight), int(sizes.iwidth)
        oriented_width, oriented_height = postprocess_dimensions_wh(sizes)
        raw_shape = (oriented_height, oriented_width)
        raw_metadata = {
            "raw_dimensions_wh": [int(sizes.width), int(sizes.height)],
            "visible_dimensions_wh": [visible_width, visible_height],
            "orientation_flip": flip,
            "postprocess_dimensions_wh": [oriented_width, oriented_height],
            "white_level": int(raw.white_level),
            "black_level_per_channel": [int(value) for value in raw.black_level_per_channel],
            "camera_whitebalance": [float(value) for value in raw.camera_whitebalance],
            "raw_pattern": raw.raw_pattern.tolist(),
            "color_description": raw.color_desc.decode("ascii", errors="replace"),
            "dng_default_crop": dng_default_crop(acquisition, sizes),
        }
    if neutral_shape is not None and raw_shape != neutral_shape:
        raise ValueError(f"RAW native dimensions changed: {raw_shape} != {neutral_shape}")
    target_shapes = {}
    for expert in "ABCDE":
        path = paths[f"expert_{expert}"]
        with tifffile.TiffFile(path) as source:
            page = source.pages[0]
            profile_tag = page.tags.get("InterColorProfile")
            if profile_tag is None:
                raise ValueError(f"Missing embedded ICC profile: {path.name}")
            profile = ImageCms.ImageCmsProfile(io.BytesIO(profile_tag.value))
            description = ImageCms.getProfileDescription(profile).strip()
            if "ProPhoto RGB" not in description:
                raise ValueError(f"Expected ProPhoto RGB ICC, got {description!r}: {path.name}")
            if page.dtype != np.dtype("uint16") or len(page.shape) != 3 or page.shape[2] != 3:
                raise ValueError(f"Expected a 16-bit RGB TIFF: {path.name}")
            shape = tuple(int(value) for value in page.shape[:2])
            # Decode every TIFF before accepting the six-file scene as valid.
            decoded = page.asarray()
            if decoded.dtype != np.uint16 or decoded.shape[:2] != shape:
                raise ValueError(f"Could not decode a valid 16-bit TIFF: {path.name}")
            target_shapes[expert] = shape
            del decoded
    return raw_metadata, target_shapes


def calibration_endpoints(config, device, dtype):
    learned = config.get("learned_calibration")
    if not learned:
        raise ValueError("The copied calibration has no learned control mapping")
    order = tuple(learned.get("control_order", ()))
    if order != CONTROL_NAMES:
        raise ValueError(f"Unexpected calibration control order: {order}")
    bounds = torch.tensor(config.get("control_bounds", learned["control_bounds"]),
                          device=device, dtype=dtype)
    if bounds.shape != (len(CONTROL_NAMES), 2) or not torch.isfinite(bounds).all():
        raise ValueError("Invalid calibrated control bounds")
    if (bounds[:, 0] >= bounds[:, 1]).any():
        raise ValueError("Calibration control bounds must be increasing")

    def bounded(raw):
        return bounds.mean(dim=1) + torch.tanh(raw) * (bounds[:, 1] - bounds[:, 0]) / 2

    neutral_raw = torch.tensor(learned["raw_neutral"], device=device, dtype=dtype)
    deltas = torch.tensor(learned["pole_raw_deltas"], device=device, dtype=dtype)
    if neutral_raw.shape != (len(CONTROL_NAMES),) or deltas.shape != (12, len(CONTROL_NAMES)):
        raise ValueError("Unexpected learned calibration dimensions")
    neutral = bounded(neutral_raw)
    positive = bounded(neutral_raw + deltas[:6].diagonal())
    negative = bounded(neutral_raw + deltas[6:].diagonal())
    return {"negative": negative, "neutral": neutral, "positive": positive}, bounds


def physical_controls(normalized, endpoints):
    positive_amount = normalized.clamp_min(0)
    negative_amount = (-normalized).clamp_min(0)
    return (endpoints["neutral"] + positive_amount * (endpoints["positive"] - endpoints["neutral"])
            + negative_amount * (endpoints["negative"] - endpoints["neutral"]))


def linear_to_srgb_float(image):
    image = image.clamp(0, 1)
    return torch.where(image <= .0031308, image * 12.92,
                       1.055 * image.clamp_min(.0031308).pow(1 / 2.4) - .055)


def objective(rendered, target, normalized):
    rendered_display = linear_to_srgb_float(rendered)
    target_display = linear_to_srgb_float(target)
    pixel = F.smooth_l1_loss(rendered_display, target_display, beta=.025)
    low_rendered = F.avg_pool2d(rendered_display, 5, 5)
    low_target = F.avg_pool2d(target_display, 5, 5)
    regularization = normalized.square().mean()
    total = pixel + .25 * F.smooth_l1_loss(low_rendered, low_target, beta=.025) + .002 * regularization
    return total


def fit_controls(neutral, target, config, endpoints, iterations, learning_rate,
                 bounds=None):
    height, width = neutral.shape[-2:]
    scale = 256 / min(height, width)
    fit_height = max(1, round(height * scale))
    fit_width = max(1, round(width * scale))
    fit_size = (fit_height, fit_width)
    fit_image = F.interpolate(neutral, size=fit_size, mode="area")
    fit_target = F.interpolate(target, size=fit_size, mode="area")
    palette_index = CONTROL_NAMES.index("palette")
    use_full_bounds = bounds is not None
    best = None
    restarts = []
    for restart_index, palette_seed in enumerate((0.0, -0.12, 0.12)):
        raw = torch.nn.Parameter(fit_image.new_zeros(len(CONTROL_NAMES) + 1))
        with torch.no_grad():
            raw[palette_index] = palette_seed
        optimizer = torch.optim.Adam([raw], lr=learning_rate)
        local_best = float("inf")
        local_result = None
        losses = []
        trace = []
        for iteration in range(iterations + 1):
            normalized = torch.tanh(raw[:len(CONTROL_NAMES)])
            normalized_tint = torch.tanh(raw[len(CONTROL_NAMES)])
            if use_full_bounds:
                controls = bounds.mean(dim=1) + normalized * (bounds[:, 1] - bounds[:, 0]) / 2
            else:
                controls = physical_controls(normalized, endpoints)
            parameters = parameters_from_controls(controls, config)
            parameters["tint"] = TINT_LIMIT * normalized_tint
            all_normalized = torch.cat((normalized, normalized_tint.reshape(1)))
            rendered = render_linear(fit_image, parameters)
            loss = objective(rendered, fit_target, all_normalized)
            value = float(loss.detach())
            losses.append(value)
            if iteration % 10 == 0 or iteration == iterations:
                trace.append({"iteration": iteration, "loss": value})
            if value < local_best:
                local_best = value
                local_result = {
                    "loss": value,
                    "iteration": iteration,
                    "normalized_controls": dict(zip(CONTROL_NAMES, normalized.detach().cpu().tolist())),
                    "physical_controls": dict(zip(CONTROL_NAMES, controls.detach().cpu().tolist())),
                    "normalized_tint": float(normalized_tint.detach()),
                    "tint": float((TINT_LIMIT * normalized_tint).detach()),
                }
            if best is None or value < best["loss"]:
                best = {**local_result, "palette_seed": palette_seed,
                        "restart_index": restart_index}
            if iteration < iterations:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        best_iteration = int(np.argmin(losses))
        tail_start = max(0, iterations - 50)
        tail_base = losses[tail_start]
        tail_min = min(losses[tail_start:])
        final_loss = losses[-1]
        relative_tail_improvement = (tail_base - tail_min) / max(abs(tail_base), 1e-12)
        final_to_best_gap = max(0.0, (final_loss - local_best) / max(abs(local_best), 1e-12))
        restarts.append({
            "restart_index": restart_index,
            "palette_seed_raw": palette_seed,
            "optimizer_steps": iterations,
            "evaluations_including_initial": len(losses),
            "best_loss": local_best,
            "best_iteration": best_iteration,
            "final_loss": final_loss,
            "relative_improvement_last_50_steps": relative_tail_improvement,
            "final_to_best_relative_gap": final_to_best_gap,
            "converged_by_0_5_percent_tail_rule": (
                relative_tail_improvement < .005 and final_to_best_gap < .005
                and best_iteration < iterations - 5),
            "loss_trace_every_10_steps": trace,
        })
    best["restarts"] = restarts
    best["convergence"] = {
        "iteration_budget_per_restart": iterations,
        "optimizer_steps_per_restart": iterations,
        "total_optimizer_steps": iterations * len(restarts),
        "early_stopping": False,
        "diagnostic_tail_steps": min(50, iterations),
        "convergence_rule": "Tail loss improvement and final-to-best gap both <0.5%; diagnostic only, never stops optimization.",
        "restart_diagnostics": restarts,
    }
    best["parameterization"] = ("tanh-normalized [-1,1] mapped to full physical bounds"
                                if use_full_bounds else
                                "tanh-normalized [-1,1] mapped linearly from neutral to learned pole endpoints")
    best["optimization_dimensions_wh"] = [fit_width, fit_height]
    return best


def fit_extended_controls(neutral, target, config, iterations, learning_rate):
    """13-DOF extended ISP fitting."""
    ext_bounds = torch.tensor([list(b) for b in EXTENDED_BOUNDS],
                               device=neutral.device, dtype=neutral.dtype)
    height, width = neutral.shape[-2:]
    scale = 256 / min(height, width)
    fit_size = (max(1, round(height * scale)), max(1, round(width * scale)))
    fit_image = F.interpolate(neutral, size=fit_size, mode="area")
    fit_target = F.interpolate(target, size=fit_size, mode="area")
    palette_index = list(EXTENDED_CONTROL_NAMES).index("palette")
    best = None
    restarts = []
    for restart_index, palette_seed in enumerate((0.0, -0.12, 0.12)):
        raw = torch.nn.Parameter(fit_image.new_zeros(len(EXTENDED_CONTROL_NAMES)))
        with torch.no_grad():
            raw[palette_index] = palette_seed
        optimizer = torch.optim.Adam([raw], lr=learning_rate)
        local_best = float("inf")
        local_result = None
        losses = []
        trace = []
        for iteration in range(iterations + 1):
            normalized = torch.tanh(raw)
            controls = ext_bounds.mean(dim=1) + normalized * (ext_bounds[:, 1] - ext_bounds[:, 0]) / 2
            parameters = parameters_from_extended_controls(controls, config)
            rendered = render_linear(fit_image, parameters)
            loss = objective(rendered, fit_target, normalized)
            value = float(loss.detach())
            losses.append(value)
            if iteration % 10 == 0 or iteration == iterations:
                trace.append({"iteration": iteration, "loss": value})
            if value < local_best:
                local_best = value
                local_result = {
                    "loss": value, "iteration": iteration,
                    "physical_controls": dict(zip(EXTENDED_CONTROL_NAMES,
                                                   controls.detach().cpu().tolist())),
                    "normalized_controls": dict(zip(EXTENDED_CONTROL_NAMES,
                                                     normalized.detach().cpu().tolist())),
                }
            if best is None or value < best["loss"]:
                best = {**local_result, "palette_seed": palette_seed,
                        "restart_index": restart_index}
            if iteration < iterations:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        best_iteration = int(np.argmin(losses))
        tail_start = max(0, iterations - 50)
        tail_base = losses[tail_start]
        tail_min = min(losses[tail_start:])
        final_loss = losses[-1]
        relative_tail_improvement = (tail_base - tail_min) / max(abs(tail_base), 1e-12)
        final_to_best_gap = max(0.0, (final_loss - local_best) / max(abs(local_best), 1e-12))
        restarts.append({
            "restart_index": restart_index, "palette_seed_raw": palette_seed,
            "optimizer_steps": iterations,
            "evaluations_including_initial": len(losses),
            "best_loss": local_best, "best_iteration": best_iteration,
            "final_loss": final_loss,
            "relative_improvement_last_50_steps": relative_tail_improvement,
            "final_to_best_relative_gap": final_to_best_gap,
            "converged_by_0_5_percent_tail_rule": (
                relative_tail_improvement < .005 and final_to_best_gap < .005
                and best_iteration < iterations - 5),
            "loss_trace_every_10_steps": trace,
        })
    best["restarts"] = restarts
    best["convergence"] = {
        "iteration_budget_per_restart": iterations,
        "optimizer_steps_per_restart": iterations,
        "total_optimizer_steps": iterations * len(restarts),
        "early_stopping": False,
        "diagnostic_tail_steps": min(50, iterations),
        "convergence_rule": "Tail loss improvement and final-to-best gap both <0.5%",
        "restart_diagnostics": restarts,
    }
    best["parameterization"] = "tanh-normalized [-1,1] mapped to full physical bounds"
    best["control_names"] = list(EXTENDED_CONTROL_NAMES)
    best["n_dofs"] = len(EXTENDED_CONTROL_NAMES)
    best["optimization_dimensions_wh"] = [fit_size[1], fit_size[0]]
    return best


def image_statistics(image):
    display = linear_to_srgb_float(image)
    return {"mean_display_rgb": display.mean(dim=(0, 2, 3)).detach().cpu().tolist(),
            "dimensions_wh": [int(image.shape[-1]), int(image.shape[-2])]}


def display_quality_metrics(target, fitted):
    target_display = linear_to_srgb8(target)[0].permute(1, 2, 0).detach().cpu().numpy()
    fitted_display = linear_to_srgb8(fitted)[0].permute(1, 2, 0).detach().cpu().numpy()
    difference = target_display.astype(np.float64) - fitted_display.astype(np.float64)
    mse = float(np.mean(difference ** 2))
    psnr = float("inf") if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))
    ssim = float(structural_similarity(target_display, fitted_display,
                                       channel_axis=-1, data_range=255))
    return {"psnr_db": psnr, "ssim": ssim,
            "mae_8bit": float(np.mean(np.abs(difference))),
            "dimensions_wh": [int(target.shape[-1]), int(target.shape[-2])],
            "color_space": "sRGB uint8 display"}


def write_png(path, image, bit_depth):
    rgb = image[0].detach().cpu().permute(1, 2, 0).numpy()
    if bit_depth == 16:
        encoded = np.rint(np.clip(rgb, 0, 1) * 65535).astype(np.uint16)
    elif bit_depth == 8:
        encoded = linear_to_srgb8(image)[0].permute(1, 2, 0).detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported PNG depth: {bit_depth}")
    if not cv2.imwrite(str(path), cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write PNG: {path.name}")
    check = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    expected_dtype = np.uint16 if bit_depth == 16 else np.uint8
    expected_shape = (int(image.shape[-2]), int(image.shape[-1]))
    if (check is None or check.dtype != expected_dtype or check.ndim != 3 or check.shape[2] != 3
            or check.shape[:2] != expected_shape):
        raise OSError(f"PNG verification failed: {path.name}")
    output_stat = path.lstat()
    if not stat.S_ISREG(output_stat.st_mode) or output_stat.st_nlink != 1:
        raise OSError(f"PNG is not an independent regular file: {path.name}")
    return {"path": path.name, "format": "PNG", "bit_depth_per_channel": bit_depth,
            "dimensions_wh": [int(check.shape[1]), int(check.shape[0])],
            "channels": "RGB", "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def software_hashes(config_path):
    return {
        "processor_sha256": sha256_file(SCRIPT_PATH),
        "v2_renderer_sha256": sha256_file(V2_ROOT / "semantic_isp.py"),
        "v2_low_level_isp_sha256": sha256_file(V2_ROOT / "axis_isp.py"),
        "v2_raw_preprocessor_sha256": sha256_file(V2_ROOT / "differentiable_isp.py"),
        "v2_config_sha256": sha256_file(config_path),
    }


def append_record(manifest_path, record):
    with manifest_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_process_records(path):
    latest = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and isinstance(record.get("scene_id"), str):
                latest[record["scene_id"]] = record
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return latest


def process_scene(root, scene_id, acquisition, acquisition_line, acquisition_hash,
                  args, config, hashes, process_manifest, latest_processed):
    paths = source_paths(root, scene_id)
    stats = local_file_signature(root, paths)
    source_signature = canonical_hash({"acquisition_record_sha256": acquisition_hash, "files": stats})
    prior = latest_processed.get(scene_id)
    if prior and prior.get("source_signature") == source_signature:
        if can_reuse_completed_output(root, prior, source_signature):
            return "skipped"
        if prior.get("status") == "error" and prior.get("software_hashes") == hashes:
            return "skipped_error"
    base = {"scene_id": scene_id, "acquisition_manifest_line": acquisition_line,
            "acquisition_record_sha256": acquisition_hash, "source_signature": source_signature,
            "software_hashes": hashes, "processor_version": "fullres_v3_semantic_isp_13dof_v1",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not is_complete_record(acquisition):
        return "not_complete"

    try:
        validate_regular_in_root(root, paths)
        raw_metadata, target_shapes = validate_sources(root, paths, acquisition)
        source_hashes = {name: {"path": path.relative_to(root).as_posix(),
                                "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                         for name, path in paths.items()}
        for name, expected_hash in manifest_source_hashes(acquisition).items():
            if source_hashes[name]["sha256"] != expected_hash:
                raise ValueError(f"Source SHA-256 differs from completed acquisition record: {name}")
        for name, expected_hash in manifest_source_hashes(acquisition).items():
            if source_hashes[name]["sha256"] != expected_hash:
                raise ValueError(f"Source SHA-256 differs from completed acquisition record: {name}")
        # A second stat check catches files that changed while checksums and decoders read them.
        if local_file_signature(root, paths) != stats:
            raise ValueError("One or more source files changed during validation")

        device = torch.device(args.device)
        torch.manual_seed(0)
        torch.set_num_threads(2)
        neutral = load_and_preprocess_raw(str(paths["raw"]), target_size=None).to(
            device=device, dtype=torch.float32)
        dimensions = [int(neutral.shape[-1]), int(neutral.shape[-2])]
        if dimensions != raw_metadata["postprocess_dimensions_wh"]:
            raise ValueError("Native V2 RAW output dimensions do not match rawpy orientation metadata")
        crop = raw_metadata["dng_default_crop"]
        crop = raw_metadata["dng_default_crop"]
        default_crop_rect = oriented_default_crop_rect(
            raw_metadata["visible_dimensions_wh"][0], raw_metadata["visible_dimensions_wh"][1],
            raw_metadata["orientation_flip"],
            crop["origin_xy"][0], crop["origin_xy"][1], crop["size_wh"][0], crop["size_wh"][1])
        baseline_linear = neutral[0].detach().cpu().permute(1, 2, 0).numpy()
        endpoints, bounds = calibration_endpoints(config, device, neutral.dtype)
        files = {}
        output_dir = Path(args.output_dir)
        if output_dir.is_absolute() or ".." in output_dir.parts or not output_dir.parts:
            raise ValueError("Output directory must be a relative path inside the dataset")
        stage_parent = ensure_directory_inside_root(root, root / output_dir)
        final_dir = stage_parent / scene_id
        if final_dir.exists():
            raise FileExistsError(f"Output already exists without a matching completion record: {scene_id}")
        stage_dir = Path(tempfile.mkdtemp(prefix=f".{scene_id}.", dir=stage_parent))
        try:
            files["baseline"] = {
                "linear_16bit": write_png(stage_dir / "baseline_linear_16bit.png", neutral, 16),
                "display_8bit": write_png(stage_dir / "baseline_display_8bit.png", neutral, 8),
                "statistics": image_statistics(neutral),
                "controls": dict(zip(CONTROL_NAMES, endpoints["neutral"].detach().cpu().tolist())),
            }
            fit_records = {}
            for expert in "ABCDE":
                target_array, profile_info = prophoto_tiff_to_linear_srgb(paths[f"expert_{expert}"])
                baseline_crop_array, target_aligned_array, target_alignment = align_expert_crop(
                    baseline_linear, target_array, default_crop_rect,
                    raw_metadata["orientation_flip"])
                baseline_crop = torch.from_numpy(np.ascontiguousarray(baseline_crop_array)).permute(2, 0, 1).unsqueeze(0).to(
                    device=device, dtype=neutral.dtype)
                target = torch.from_numpy(np.ascontiguousarray(target_aligned_array)).permute(2, 0, 1).unsqueeze(0).to(
                    device=device, dtype=neutral.dtype)
                if baseline_crop.shape != target.shape:
                    raise ValueError(f"Crop registration produced unequal tensor sizes for expert {expert}")
                fit = fit_extended_controls(baseline_crop, target, config, args.iterations, args.learning_rate)
                physical = torch.tensor([fit["physical_controls"][name] for name in EXTENDED_CONTROL_NAMES],
                                        device=device, dtype=neutral.dtype)
                fit_parameters = parameters_from_extended_controls(physical, config)
                with torch.no_grad():
                    aligned_fit = render_linear(baseline_crop, fit_parameters)
                fit["quality_metrics"] = display_quality_metrics(target, aligned_fit)
                fit_records[expert] = {
                    "expert_source": source_hashes[f"expert_{expert}"],
                    "icc_transform": profile_info,
                    "target_alignment": target_alignment,
                    "fit": fit,
                    "aligned_baseline_statistics": image_statistics(baseline_crop),
                    "target_statistics": {"dimensions_wh": [int(target.shape[-1]), int(target.shape[-2])]},
                    "fit_statistics": image_statistics(aligned_fit),
                    "files": {
                        "aligned_baseline_linear_16bit": write_png(stage_dir / f"expert_{expert}_baseline_crop_linear_16bit.png", baseline_crop, 16),
                        "aligned_baseline_display_8bit": write_png(stage_dir / f"expert_{expert}_baseline_crop_display_8bit.png", baseline_crop, 8),
                        "target_linear_16bit": write_png(stage_dir / f"expert_{expert}_target_linear_16bit.png", target, 16),
                        "target_display_8bit": write_png(stage_dir / f"expert_{expert}_target_display_8bit.png", target, 8),
                        "linear_16bit": write_png(stage_dir / f"expert_{expert}_fit_linear_16bit.png", aligned_fit, 16),
                        "display_8bit": write_png(stage_dir / f"expert_{expert}_fit_display_8bit.png", aligned_fit, 8),
                    },
                }
                del target_array, baseline_crop, target, aligned_fit
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if local_file_signature(root, paths) != stats:
                raise ValueError("One or more source files changed during processing")
            record = {
                **base,
                "status": "complete",
                "source_files": source_hashes,
                "rawpy_metadata": raw_metadata,
                "native_dimensions_wh": dimensions,
                "raw_preprocessing": {
                    "implementation": "copied V2 differentiable_isp.load_and_preprocess_raw(target_size=None)",
                    "camera_white_balance": True, "demosaic": "AHD", "gamma": [1, 1],
                    "no_auto_bright": True, "output_color": "linear sRGB primaries",
                    "rawpy_output_bits_per_channel": 16, "denoise": "OpenCV bilateralFilter d=5, sigmaColor=.05, sigmaSpace=35",
                    "normalization": "none (removed p99 gain; rawpy handles black/white levels)",
                    "native_size_processing": True,
                },
                "fit_protocol": {
                    "objective": "smooth-L1 in sRGB display space + 0.25 * pooled smooth-L1 + 0.002 * normalized-control L2 mean",
                    "fitted_dofs": list(EXTENDED_CONTROL_NAMES),
                    "n_dofs": len(EXTENDED_CONTROL_NAMES),
                    "isp": "semantic_isp v3 — 13-DOF with shadow/highlight TM, vibrance, local TM, hue-selective shift, adaptive sat ceiling, dark-pixel sharp protection",
                    "control_parameterization": "tanh-normalized mapped to full physical bounds",
                    "control_bounds": [list(b) for b in EXTENDED_BOUNDS],
                    "optimization_short_side_px": 256,
                    "alignment": "Per-expert DNG default-crop prior; image-feature translation/orientation verification; integer crop only; scale/warp forbidden",
                    "optimization_resize": "area downsample of already crop-aligned GT/input pair to 256px short side; same geometry on both",
                    "palette_restarts_raw": [0.0, -0.12, 0.12],
                    "iterations": args.iterations, "learning_rate": args.learning_rate,
                    "optimizer": "Adam", "seed": 0, "device": str(device),
                    "early_stopping": False,
                    "convergence_diagnostics": "All iterations run for each of three restarts; record best/final losses and every-10-step checkpoints.",
                    "psnr_ssim": "Computed for every expert pair from native-dimension aligned sRGB uint8 target and fit.",
                    "full_native_baseline_export": True,
                    "fitted_export_geometry": "Each fit is rendered on its expert-specific integer-cropped and orientation-aligned RAW window, exactly matching the aligned GT dimensions; no resize or warp is applied after registration.",
                    "resolution_distinction": "Full baseline export retains native rawpy dimensions. Per-expert aligned baseline, target, and fit exports use the crop-only registration window. Controls are optimized after equal area downsampling of that aligned pair to a 256px short side.",
                },
                "baseline": files["baseline"],
                "fits": fit_records,
                "output_directory": final_dir.relative_to(root).as_posix(),
            }
            # Keep the scene result self-contained inside the dataset; the JSONL is the index.
            for file_record in (record["baseline"]["linear_16bit"], record["baseline"]["display_8bit"]):
                file_record["path"] = f"{final_dir.relative_to(root).as_posix()}/{file_record['path']}"
            for fit_record in record["fits"].values():
                for file_record in fit_record["files"].values():
                    file_record["path"] = f"{final_dir.relative_to(root).as_posix()}/{file_record['path']}"
            os.replace(stage_dir, final_dir)
        except Exception:
            if stage_dir.exists():
                for item in stage_dir.iterdir():
                    item.unlink()
                stage_dir.rmdir()
            raise

        record["source_signature"] = canonical_hash({"acquisition_record_sha256": acquisition_hash,
                                                      "files": local_file_signature(root, paths)})
        append_record(process_manifest, record)
        latest_processed[scene_id] = record
        return "complete"
    except Exception as error:
        error_record = {**base, "status": "error", "error": f"{type(error).__name__}: {error}",
                        "source_file_stats": stats}
        append_record(process_manifest, error_record)
        latest_processed[scene_id] = error_record
        return "error"


def run_scan(args, config, hashes):
    root = args.root
    source_manifest = root / "metadata" / "source_manifest.jsonl"
    manifest_name = Path(args.manifest_name)
    if manifest_name.is_absolute() or ".." in manifest_name.parts or len(manifest_name.parts) != 1:
        raise ValueError("Process manifest name must be a basename inside metadata/")
    process_manifest = root / "metadata" / manifest_name
    if not root.is_dir() or not source_manifest.is_file():
        return {"status": "waiting", "root_exists": root.is_dir(),
                "acquisition_manifest_exists": source_manifest.is_file(),
                "complete_records": 0, "processed": 0, "errors": 0}

    # Hold the process log lock throughout a scan so concurrent consumers cannot fit the same scene.
    ensure_directory_inside_root(root, process_manifest.parent)
    validate_manifest_file(root, source_manifest)
    if process_manifest.exists() or process_manifest.is_symlink():
        validate_manifest_file(root, process_manifest)
    with process_manifest.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        lock_stream.seek(0)
        latest_processed = {}
        for line in lock_stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("scene_id"), str):
                latest_processed[row["scene_id"]] = row
        acquisitions, malformed = load_acquisition_records(source_manifest)
        complete_rows = [(scene_id, row, line_no, row_hash)
                         for scene_id, (row, line_no, row_hash) in acquisitions.items()
                         if is_complete_record(row)]
        if args.scene_ids is not None:
            requested = set(args.scene_ids)
            unknown = requested - set(acquisitions)
            if unknown:
                raise ValueError(f"Requested scene IDs are absent from acquisition manifest: {sorted(unknown)}")
            complete_rows = [item for item in complete_rows if item[0] in requested]
        counts = {"complete": 0, "skipped": 0, "skipped_error": 0, "error": 0}
        for scene_id, acquisition, line_no, row_hash in complete_rows:
            outcome = process_scene(root, scene_id, acquisition, line_no, row_hash,
                                    args, config, hashes, process_manifest, latest_processed)
            counts[outcome] = counts.get(outcome, 0) + 1
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    latest_processed = read_process_records(process_manifest)
    return {
        "status": "scanned", "complete_records": len(complete_rows),
        "processed": counts["complete"],
        "skipped_complete": counts["skipped"], "skipped_prior_errors": counts["skipped_error"],
        "errors": counts["error"],
        "latest_scene_statuses": {key: value.get("status") for key, value in latest_processed.items()},
        "malformed_acquisition_lines": malformed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=.03)
    parser.add_argument("--output-dir", default="processed_crop_only_v1")
    parser.add_argument("--manifest-name", default="process_crop_only_manifest_v1.jsonl")
    parser.add_argument("--watch", action="store_true", help="Poll acquisition manifest until interrupted")
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--scene-ids", nargs="*", help="Optional scene allowlist for a focused run")
    args = parser.parse_args()
    if args.iterations < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Iterations and learning rate must be positive")
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("Poll interval must be positive")
    root = args.root.expanduser().absolute()
    output_dir = Path(args.output_dir)
    if output_dir.is_absolute() or ".." in output_dir.parts or not output_dir.parts:
        parser.error("--output-dir must be a relative path inside the dataset root")
    config_path = args.config.resolve(strict=True)
    if config_path != CONFIG_PATH.resolve(strict=True):
        parser.error("Only the copied V2 Exp81 calibration is allowed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hashes = software_hashes(config_path)
    while True:
        status = run_scan(argparse.Namespace(**{**vars(args), "root": root}), config, hashes)
        print(json.dumps(status, sort_keys=True), flush=True)
        if not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()