"""Per-expert TIFF-to-RAW orientation and crop registration with no scaling."""

from __future__ import annotations

import math

import cv2
import numpy as np


class CropAlignmentError(ValueError):
    pass


def oriented_default_crop_rect(frame_width, frame_height, rawpy_flip,
                               origin_x, origin_y, crop_width, crop_height):
    frame_width, frame_height, rawpy_flip = int(frame_width), int(frame_height), int(rawpy_flip)
    x, y, width, height = map(int, (origin_x, origin_y, crop_width, crop_height))
    if min(x, y, width, height) < 0 or width == 0 or height == 0:
        raise ValueError("Invalid DNG default-crop rectangle")
    if rawpy_flip == 0:
        rect = (x, y, width, height)
    elif rawpy_flip == 3:
        rect = (frame_width - x - width, frame_height - y - height, width, height)
    elif rawpy_flip == 5:
        rect = (y, frame_width - x - width, height, width)
    elif rawpy_flip == 6:
        rect = (frame_height - y - height, x, height, width)
    else:
        raise CropAlignmentError(f"Unsupported rawpy orientation flip: {rawpy_flip}")
    rx, ry, rw, rh = rect
    output_width, output_height = ((frame_height, frame_width)
                                   if rawpy_flip in (5, 6)
                                   else (frame_width, frame_height))
    if rx < 0 or ry < 0 or rx + rw > output_width or ry + rh > output_height:
        raise CropAlignmentError(f"DNG default crop {rect} is outside oriented frame {(output_width, output_height)}")
    return rect


def oriented_crop_rectangle(image_width, image_height, rawpy_flip,
                            origin_x, origin_y, crop_width, crop_height):
    """Map DNG default-crop coordinates into rawpy's postprocess orientation."""
    x, y, width, height = map(int, (origin_x, origin_y, crop_width, crop_height))
    image_width, image_height, rawpy_flip = int(image_width), int(image_height), int(rawpy_flip)
    if min(x, y, width, height) < 0 or width == 0 or height == 0:
        raise ValueError("Invalid DNG default crop rectangle")
    if rawpy_flip == 0:
        return x, y, width, height
    if rawpy_flip == 3:
        return image_width - x - width, image_height - y - height, width, height
    if rawpy_flip == 5:
        return y, image_width - x - width, height, width
    if rawpy_flip == 6:
        return image_height - y - height, x, height, width
    raise CropAlignmentError(f"Unsupported rawpy orientation flip: {rawpy_flip}")


def _display_gray(linear_rgb, max_dimension):
    image = np.asarray(linear_rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected HWC linear RGB")
    display = np.where(image <= .0031308, image * 12.92,
                       1.055 * np.maximum(image, .0031308) ** (1 / 2.4) - .055)
    display = np.rint(np.clip(display, 0, 1) * 255).astype(np.uint8)
    height, width = display.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1:
        display = cv2.resize(display, (round(width * scale), round(height * scale)),
                             interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(display, cv2.COLOR_RGB2GRAY), scale


def _match_crop(target_linear, baseline_crop_linear, max_dimension=1400):
    target_gray, target_scale = _display_gray(target_linear, max_dimension)
    baseline_gray, baseline_scale = _display_gray(baseline_crop_linear, max_dimension)
    if target_gray.shape != baseline_gray.shape or abs(target_scale - baseline_scale) > 1e-8:
        return None
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=.015, edgeThreshold=12)
    target_points, target_desc = sift.detectAndCompute(target_gray, None)
    base_points, base_desc = sift.detectAndCompute(baseline_gray, None)
    if target_desc is None or base_desc is None or len(target_desc) < 20 or len(base_desc) < 20:
        return None
    matches = cv2.BFMatcher().knnMatch(target_desc, base_desc, k=2)
    good = [first for pair in matches if len(pair) == 2
            for first, second in [pair] if first.distance < .72 * second.distance]
    if len(good) < 30:
        return None
    source = np.float32([target_points[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    destination = np.float32([base_points[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.estimateAffinePartial2D(
        source, destination, method=cv2.RANSAC, ransacReprojThreshold=2,
        maxIters=10000, confidence=.999, refineIters=20)
    if matrix is None or mask is None:
        return None
    inliers = mask.reshape(-1).astype(bool)
    linear_scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    angle_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    projected = (source[inliers, 0] @ matrix[:, :2].T) + matrix[:, 2]
    residual = np.linalg.norm(projected - destination[inliers, 0], axis=1)
    target_inliers = source[inliers, 0]
    span_x = float(np.ptp(target_inliers[:, 0]) / max(target_gray.shape[1], 1))
    span_y = float(np.ptp(target_inliers[:, 1]) / max(target_gray.shape[0], 1))
    return {
        "scale": linear_scale,
        "residual_rotation_deg": angle_deg,
        "translation_xy_px": [float(matrix[0, 2] / target_scale),
                              float(matrix[1, 2] / target_scale)],
        "matches": len(good),
        "inliers": int(inliers.sum()),
        "inlier_fraction": float(inliers.mean()),
        "inlier_span_fraction_xy": [span_x, span_y],
        "median_reprojection_error_px": float(np.median(residual) / target_scale),
        "p95_reprojection_error_px": float(np.quantile(residual, .95) / target_scale),
    }


def _initial_origin(crop_rect, target_width, target_height, frame_width, frame_height):
    crop_x, crop_y, crop_width, crop_height = crop_rect
    x = crop_x + (crop_width - target_width) // 2
    y = crop_y + (crop_height - target_height) // 2
    x = max(0, min(x, frame_width - target_width))
    y = max(0, min(y, frame_height - target_height))
    if target_width > frame_width or target_height > frame_height:
        raise CropAlignmentError("Expert TIFF dimensions exceed the rawpy frame; crop-only registration cannot proceed")
    return x, y


def align_expert_crop(baseline_linear, target_linear, default_crop_rect, rawpy_flip,
                      min_inliers=80, min_inlier_fraction=.20,
                      max_scale_error=.001, max_rotation_deg=.15,
                      max_reprojection_p95_px=3.0, max_refine_shift_px=4096):
    """Find a per-expert integer crop/orientation; never rescale either image.

    SIFT is used only to estimate geometry on temporary display-domain thumbnails.
    Returned baseline and target pixels are cropped/rotated from their original
    linear arrays without resize, affine warp, or interpolation.
    """
    baseline = np.asarray(baseline_linear, dtype=np.float32)
    target = np.asarray(target_linear, dtype=np.float32)
    if baseline.ndim != 3 or baseline.shape[2] != 3 or target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("Expected HWC RGB arrays")
    frame_height, frame_width = baseline.shape[:2]
    candidates = []
    for rotation_ccw in range(4):
        oriented_target = np.rot90(target, rotation_ccw).copy()
        target_height, target_width = oriented_target.shape[:2]
        if target_width > frame_width or target_height > frame_height:
            continue
        start_x, start_y = _initial_origin(default_crop_rect, target_width, target_height,
                                           frame_width, frame_height)
        initial_crop = baseline[start_y:start_y + target_height,
                                start_x:start_x + target_width]
        registration = _match_crop(oriented_target, initial_crop)
        if registration is None:
            continue
        scale_error = abs(registration["scale"] - 1.)
        rotation_error = abs(registration["residual_rotation_deg"])
        if scale_error > max_scale_error or rotation_error > max_rotation_deg:
            continue
        if (registration["inliers"] < min_inliers
                or registration["inlier_fraction"] < min_inlier_fraction
                or min(registration["inlier_span_fraction_xy"]) < .15
                or registration["p95_reprojection_error_px"] > max_reprojection_p95_px):
            continue
        shift_x, shift_y = registration["translation_xy_px"]
        if max(abs(shift_x), abs(shift_y)) > max_refine_shift_px:
            continue
        final_x = start_x + int(round(shift_x))
        final_y = start_y + int(round(shift_y))
        if (final_x < 0 or final_y < 0 or final_x + target_width > frame_width
                or final_y + target_height > frame_height):
            continue
        aligned_crop = baseline[final_y:final_y + target_height,
                                final_x:final_x + target_width]
        verification = _match_crop(oriented_target, aligned_crop)
        if verification is None:
            continue
        if (abs(verification["scale"] - 1.) > max_scale_error
                or abs(verification["residual_rotation_deg"]) > max_rotation_deg
                or max(map(abs, verification["translation_xy_px"])) > 1.0
                or verification["inliers"] < min_inliers
                or verification["inlier_fraction"] < min_inlier_fraction
                or verification["p95_reprojection_error_px"] > max_reprojection_p95_px):
            continue
        score = (verification["inlier_fraction"], verification["inliers"],
                 -verification["p95_reprojection_error_px"], -rotation_ccw)
        candidates.append((score, oriented_target, aligned_crop, {
            "status": "aligned_crop_only",
            "method": "DNG default crop prior + SIFT/RANSAC translation refinement; no scaling",
            "target_rotation_ccw": rotation_ccw,
            "rawpy_flip": int(rawpy_flip),
            "initial_crop_xywh": [start_x, start_y, target_width, target_height],
            "refined_crop_xywh": [final_x, final_y, target_width, target_height],
            "initial_registration": registration,
            "verification_registration": verification,
            "resized_or_warped": False,
        }))
    if not candidates:
        raise CropAlignmentError(
            "No reliable crop-only transform: unit-scale registration, orientation, inlier coverage, "
            "or residual checks failed. Scene/expert is withheld for manual review; no scaling was applied.")
    _, aligned_target, aligned_baseline, record = max(candidates, key=lambda item: item[0])
    return aligned_baseline, aligned_target, record