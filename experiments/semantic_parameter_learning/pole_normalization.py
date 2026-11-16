"""Convert ISP controls to and from calibrated directional pole units."""

import sys
from pathlib import Path

import torch

SEMANTIC_ISP_DIR = Path(__file__).resolve().parents[1] / "semantic_isp"
if str(SEMANTIC_ISP_DIR) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_ISP_DIR))

from moran_semantic_isp import CONTROL_BOUNDS, CONTROL_NAMES, CONTROL_NEUTRAL, bounded_controls


def pole_endpoints(config):
    learned = config["learned_calibration"]
    axes = tuple(config["axes"])
    if axes != ("warmth", "saturation", "brightness", "contrast", "sharpness", "palette"):
        raise ValueError(f"Expected six direct ISP axes in physical control order, got {axes}")
    bounds = config.get("control_bounds", learned.get("control_bounds", CONTROL_BOUNDS))
    raw_neutral = torch.as_tensor(learned["raw_neutral"], dtype=torch.float64)
    deltas = torch.as_tensor(learned["pole_raw_deltas"], dtype=torch.float64)
    if raw_neutral.shape != (6,) or deltas.shape != (12, 6):
        raise ValueError("Invalid learned pole calibration dimensions")
    neutral = torch.as_tensor(CONTROL_NEUTRAL, dtype=torch.float64)
    negative, positive = [], []
    for index in range(len(CONTROL_NAMES)):
        negative.append(bounded_controls(raw_neutral + deltas[index + len(CONTROL_NAMES)], bounds)[index])
        positive.append(bounded_controls(raw_neutral + deltas[index], bounds)[index])
    negative = torch.stack(negative)
    positive = torch.stack(positive)
    if not torch.isfinite(negative).all() or not torch.isfinite(positive).all():
        raise ValueError("Pole settings must be finite")
    if not torch.all(negative < neutral) or not torch.all(positive > neutral):
        raise ValueError("Every pole must have a nonzero, correctly signed physical span from neutral")
    return negative, neutral, positive


def normalized_to_physical(normalized, endpoints):
    negative, neutral, positive = (value.to(device=normalized.device, dtype=normalized.dtype)
                                   for value in endpoints)
    return neutral + torch.where(normalized >= 0, normalized * (positive - neutral),
                                 normalized * (neutral - negative))


def physical_to_normalized(physical, endpoints):
    negative, neutral, positive = (value.to(device=physical.device, dtype=physical.dtype)
                                   for value in endpoints)
    difference = physical - neutral
    return torch.where(difference >= 0, difference / (positive - neutral),
                       difference / (neutral - negative))