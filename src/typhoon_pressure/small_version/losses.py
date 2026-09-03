from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import DualLossConfig


def soft_distribution_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross entropy for Q_t(x), conditional on a distribution target being available."""
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    per_lead = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    denominator = mask.sum()
    if denominator.item() == 0:
        return logits.sum() * 0.0
    return (per_lead * mask).sum() / denominator


def survival_binary_cross_entropy(survival_probability, target, mask):
    """Masked BCE on the monotone S_t curve; unknown/censored labels are excluded."""
    probability = survival_probability.clamp(1e-6, 1.0 - 1e-6)
    per_lead = F.binary_cross_entropy(probability, target, reduction="none")
    denominator = mask.sum()
    if denominator.item() == 0:
        return survival_probability.sum() * 0.0
    return (per_lead * mask).sum() / denominator


def east_asia_track_error(prediction, target, mask, scale_km):
    km_per_degree = 111.32
    lat_error_km = (prediction[..., 0] - target[..., 0]) * km_per_degree
    lon_delta = torch.remainder(prediction[..., 1] - target[..., 1] + 180.0, 360.0) - 180.0
    lon_error_km = lon_delta * km_per_degree * torch.cos(torch.deg2rad(target[..., 0]))
    squared_km = lat_error_km.square() + lon_error_km.square()
    denominator = mask.sum()
    if denominator.item() == 0:
        zero = prediction.sum() * 0.0
        return zero, zero.detach()
    mean_squared_km = (squared_km * mask).sum() / denominator
    return mean_squared_km / (scale_km ** 2), torch.sqrt(mean_squared_km)


class DualObjectiveLoss(nn.Module):
    """Backward-compatible name for distribution + track + survival objective."""

    def __init__(self, config: DualLossConfig = DualLossConfig()):
        super().__init__()
        self.config = config

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        distribution_loss = soft_distribution_cross_entropy(
            outputs["distribution_logits"], batch["distribution_target"], batch["distribution_mask"]
        )
        track_loss, track_rmse_km = east_asia_track_error(
            outputs["track_latlon"], batch["track_target"], batch["track_mask"], self.config.track_scale_km
        )
        if "survival_probability" in outputs and "survival_target" in batch:
            survival_loss = survival_binary_cross_entropy(
                outputs["survival_probability"], batch["survival_target"], batch["survival_mask"]
            )
        else:
            # Keep old checkpoints/tests usable while migration completes.
            survival_loss = distribution_loss * 0.0
        total = (self.config.distribution_weight * distribution_loss
                 + self.config.local_track_weight * track_loss
                 + self.config.survival_weight * survival_loss)
        return {"loss": total, "distribution_ce": distribution_loss,
                "survival_bce": survival_loss,
                "local_track_mse_normalized": track_loss,
                "local_track_rmse_km": track_rmse_km}
