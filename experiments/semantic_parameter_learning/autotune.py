"""Joint ISP optimization with semantic and paired quality objectives."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SEMANTIC_ISP_DIR = Path(__file__).resolve().parents[1] / "semantic_isp"
if str(SEMANTIC_ISP_DIR) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_ISP_DIR))

from exp38_learn_poles import encode_image
from moran_semantic_isp import (
    CONTROL_BOUNDS, CONTROL_NAMES, bounded_controls, neutral_raw_controls,
    parameters_from_controls, render_linear,
)
from pole_normalization import normalized_to_physical, physical_to_normalized, pole_endpoints

REALISM_PROMPTS = ("a natural color photo", "an artificial, over-processed photo")
BEAUTY_PROMPTS = ("a photo with good exposure and pleasing colors",
                  "a photo with bad exposure and unpleasant colors")


def quality_image_features(model, image):
    image = image.clamp(0, 1)
    image = torch.where(image <= .0031308, 12.92 * image,
                        1.055 * image.clamp_min(.0031308).pow(1 / 2.4) - .055)
    resized = F.interpolate(image, (224, 224), mode="bicubic", align_corners=False)
    return F.normalize(model.encode_image((resized - .5) / .5), dim=-1)


def paired_quality_scores(image_features, quality_text_features):
    if quality_text_features.shape[0] != 4:
        raise ValueError("Expected realism positive/negative and beauty positive/negative embeddings")
    similarities = image_features @ quality_text_features.T
    return torch.stack((similarities[:, 0] - similarities[:, 1],
                        similarities[:, 2] - similarities[:, 3]), dim=-1)


def objective_terms(semantic_similarity, quality_scores, neutral_scores, raw, neutral_raw,
                    semantic_weight=1., realism_weight=1., beauty_weight=1., regularization_weight=.003):
    semantic = -semantic_similarity.mean()
    quality_loss = (neutral_scores - quality_scores).mean(dim=0)
    regularization = regularization_weight * (raw - neutral_raw).square().sum()
    total = (semantic_weight * semantic + realism_weight * quality_loss[0]
             + beauty_weight * quality_loss[1] + regularization)
    return {"semantic": semantic, "realism": quality_loss[0], "beauty": quality_loss[1],
            "regularization": regularization, "total": total}


def optimize(image, semantic_model, quality_model, text_feature, quality_text_features, config,
             iterations=100, learning_rate=.03, semantic_weight=1., realism_weight=1.,
             beauty_weight=1., regularization_weight=.003, palette_starts=(0., -.02, .02),
             convergence=None, parameterization="physical", additional_raw_starts=None):
    if iterations < 1 or learning_rate <= 0:
        raise ValueError("Iterations and learning rate must be positive")
    if parameterization not in ("physical", "p90"):
        raise ValueError(f"Unknown parameterization: {parameterization}")
    bounds = config.get("control_bounds", config.get("learned_calibration", {}).get("control_bounds", CONTROL_BOUNDS))
    endpoints = pole_endpoints(config) if parameterization == "p90" else None
    neutral_raw = (torch.zeros(len(CONTROL_NAMES), device=image.device, dtype=image.dtype)
                   if endpoints is not None else neutral_raw_controls(device=image.device, dtype=image.dtype, bounds=bounds))

    def controls_from_raw(raw):
        return (normalized_to_physical(torch.tanh(raw), endpoints) if endpoints is not None
                else bounded_controls(raw, bounds))

    with torch.no_grad():
        neutral_image = render_linear(image, parameters_from_controls(controls_from_raw(neutral_raw), config))
        neutral_quality = paired_quality_scores(quality_image_features(quality_model, neutral_image), quality_text_features)
    best = None
    palette_index = CONTROL_NAMES.index("palette")
    palette_low, palette_high = ((float(endpoints[0][palette_index]), float(endpoints[2][palette_index]))
                                 if endpoints is not None else bounds[palette_index])
    if not palette_starts or 0. not in palette_starts or any(
            not palette_low < start < palette_high for start in palette_starts):
        raise ValueError("Palette starts must include neutral and lie strictly inside the configured bounds")
    starts = []
    for palette_start in palette_starts:
        initial = neutral_raw.detach().clone()
        if endpoints is not None:
            physical_start = endpoints[1].to(device=image.device, dtype=image.dtype).clone()
            physical_start[palette_index] = palette_start
            palette_normalized = physical_to_normalized(physical_start, endpoints)[palette_index]
            initial[palette_index] = torch.atanh(palette_normalized)
        else:
            midpoint = (palette_low + palette_high) / 2
            half_range = (palette_high - palette_low) / 2
            initial[palette_index] = torch.atanh(initial.new_tensor((palette_start - midpoint) / half_range))
        starts.append((f"palette_{palette_start:+.6f}", palette_start, initial))
    for extra in additional_raw_starts or ():
        initial = torch.as_tensor(extra["raw_controls"], device=image.device, dtype=image.dtype).detach().clone()
        if initial.shape != neutral_raw.shape or not torch.isfinite(initial).all():
            raise ValueError("Additional starts must provide six finite raw controls")
        if not extra.get("name") or any(start[0] == extra["name"] for start in starts):
            raise ValueError("Additional start names must be nonempty and unique")
        palette_start = float(controls_from_raw(initial)[palette_index])
        starts.append((extra["name"], palette_start, initial))
    restart_reports = []
    for start_id, palette_start, initial in starts:
        raw = torch.nn.Parameter(initial)
        optimizer = torch.optim.Adam([raw], lr=learning_rate)
        if convergence is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=iterations, eta_min=learning_rate * .1)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=.5, patience=20, threshold=1e-5,
                threshold_mode="abs", min_lr=1e-5)
        trace = []
        local_best = float("inf")
        local_best_iteration = 0
        converged = False
        for iteration in range(iterations + 1):
            controls = controls_from_raw(raw)
            rendered = render_linear(image, parameters_from_controls(controls, config))
            semantic_features = encode_image(semantic_model, rendered, config)
            similarity = (semantic_features * text_feature).sum(dim=-1)
            quality = paired_quality_scores(quality_image_features(quality_model, rendered), quality_text_features)
            losses = objective_terms(similarity, quality, neutral_quality, raw, neutral_raw,
                                     semantic_weight, realism_weight, beauty_weight, regularization_weight)
            if not torch.isfinite(losses["total"]):
                raise ValueError("Nonfinite optimization objective")
            loss_value = float(losses["total"].detach())
            if loss_value < local_best:
                local_best = loss_value
                local_best_iteration = iteration
            if best is None or loss_value < best["losses"]["total"]:
                best = {"iteration": iteration, "start_id": start_id, "palette_start": palette_start,
                    "parameterization": parameterization,
                        "controls": dict(zip(CONTROL_NAMES, controls.detach().cpu().tolist())),
                        "raw_controls": raw.detach().cpu().tolist(),
                        "semantic_similarity": float(similarity.detach().mean()),
                        "quality_scores": quality.detach().mean(dim=0).cpu().tolist(),
                        "neutral_quality_scores": neutral_quality.detach().mean(dim=0).cpu().tolist(),
                        "losses": {key: float(value.detach()) for key, value in losses.items()}}
                if endpoints is not None:
                    best["normalized_controls"] = dict(zip(CONTROL_NAMES,
                        physical_to_normalized(controls.detach(), endpoints).cpu().tolist()))
            if convergence is not None:
                optimizer.zero_grad()
                losses["total"].backward()
                gradient_norm = float(raw.grad.detach().abs().max())
                trace.append({"iteration": iteration, "total": loss_value,
                              "semantic": float(losses["semantic"].detach()),
                              "realism": float(losses["realism"].detach()),
                              "beauty": float(losses["beauty"].detach()),
                              "regularization": float(losses["regularization"].detach()),
                              "semantic_similarity": float(similarity.detach().mean()),
                              "quality_scores": quality.detach().mean(dim=0).cpu().tolist(),
                              "gradient_max_abs": gradient_norm,
                              "learning_rate": optimizer.param_groups[0]["lr"],
                              "controls": controls.detach().cpu().tolist()})
                window = convergence["window"]
                if iteration >= convergence["min_iterations"] and len(trace) > 2 * window:
                    prior = min(row["total"] for row in trace[-2 * window:-window])
                    recent = min(row["total"] for row in trace[-window:])
                    first = controls.new_tensor(trace[-window]["controls"])
                    span = ((endpoints[2] - endpoints[0]).to(device=controls.device, dtype=controls.dtype) / 2
                            if endpoints is not None else controls.new_tensor([(high - low) / 2 for low, high in bounds]))
                    control_change = float(((controls.detach() - first).abs() / span).max())
                    converged = (abs(prior - recent) <= convergence["loss_tolerance"]
                                 and control_change <= convergence["control_tolerance"]
                                 and gradient_norm <= convergence["gradient_tolerance"])
                    if converged:
                        break
                if iteration < iterations:
                    optimizer.step()
                    scheduler.step(loss_value)
            elif iteration < iterations:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()
                scheduler.step()
        if convergence is not None:
            restart_reports.append({"start_id": start_id, "palette_start": palette_start, "converged": converged,
                                    "updates": iteration, "best_iteration": local_best_iteration,
                                    "best_total": local_best, "trace": trace})
    if convergence is not None:
        chosen = next(report for report in restart_reports if report["start_id"] == best["start_id"])
        best["convergence"] = {"selected_restart_converged": chosen["converged"],
                               "all_restarts_converged": all(report["converged"] for report in restart_reports),
                               "criteria": dict(convergence), "restarts": restart_reports}
    return best