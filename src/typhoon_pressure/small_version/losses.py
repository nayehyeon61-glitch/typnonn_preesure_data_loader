from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import DualLossConfig


def storm_trajectory_gaussian_nll(
    mean_latlon: torch.Tensor,
    covariance: torch.Tensor,
    target_latlon: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    lat_error = target_latlon[..., 0] - mean_latlon[..., 0]
    lon_error = torch.remainder(target_latlon[..., 1] - mean_latlon[..., 1] + 180.0, 360.0) - 180.0
    residual = torch.stack((lat_error, lon_error), dim=-1)
    eye = torch.eye(2, dtype=covariance.dtype, device=covariance.device)
    stable_covariance = covariance + 1e-4 * eye
    solved = torch.linalg.solve(stable_covariance, residual.unsqueeze(-1)).squeeze(-1)
    mahalanobis = (residual * solved).sum(dim=-1)
    sign, logdet = torch.linalg.slogdet(stable_covariance)
    if not torch.all(sign > 0):
        raise ValueError("Trajectory covariance must be positive definite")
    per_lead = 0.5 * (mahalanobis + logdet + 2.0 * torch.log(
        torch.tensor(2.0 * torch.pi, dtype=covariance.dtype, device=covariance.device)
    ))
    denominator = mask.sum()
    if denominator.item() == 0:
        return mean_latlon.sum() * 0.0
    return (per_lead * mask).sum() / denominator


def absorbing_survival_loss(probability, target, mask):
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    per_lead = F.binary_cross_entropy(probability, target, reduction="none")
    denominator = mask.sum()
    if denominator.item() == 0:
        return probability.sum() * 0.0
    return (per_lead * mask).sum() / denominator


def soft_distribution_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy against an empirical IBTrACS probability distribution."""
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    per_lead = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    denominator = mask.sum()
    if denominator.item() == 0:
        return logits.sum() * 0.0
    return (per_lead * mask).sum() / denominator


def sampled_distribution_cross_entropy(
    log_probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Soft-label CE on the log KDE mixture of sampled trajectories.

    The log probabilities are produced by reparameterized recursive samples and
    a log-sum-exp mixture, so the loss remains differentiable and numerically
    stable even for distant global-grid cells.
    """
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    per_lead = -(target * log_probabilities).sum(dim=-1)
    denominator = mask.sum()
    if denominator.item() == 0:
        return log_probabilities.sum() * 0.0
    return (per_lead * mask).sum() / denominator


def east_asia_track_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale_km: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local tangent-plane MSE and unscaled RMSE, both dateline safe."""
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
    """Track loss + long-range distribution loss.

    WeatherNextFusionTransformer supplies ``distribution_log_probabilities``
    generated from recursive stochastic trajectories. Older/smaller models can
    still provide direct ``distribution_logits`` and use the legacy CE path.
    """

    def __init__(self, config: DualLossConfig = DualLossConfig()):
        super().__init__()
        self.config = config

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        if (
            "distribution_mean_latlon" in outputs
            and "distribution_marginal_covariance" in outputs
            and "future_track_target" in batch
        ):
            distribution_loss = storm_trajectory_gaussian_nll(
                outputs["distribution_mean_latlon"],
                outputs["distribution_marginal_covariance"],
                batch["future_track_target"],
                batch["future_track_mask"],
            )
        elif "distribution_log_probabilities" in outputs:
            distribution_loss = sampled_distribution_cross_entropy(
                outputs["distribution_log_probabilities"],
                batch["distribution_target"],
                batch["distribution_mask"],
            )
        else:
            distribution_loss = soft_distribution_cross_entropy(
                outputs["distribution_logits"],
                batch["distribution_target"],
                batch["distribution_mask"],
            )
        track_loss, track_rmse_km = east_asia_track_error(
            outputs["track_latlon"], batch["track_target"], batch["track_mask"], self.config.track_scale_km
        )
        if "survival_probability" in outputs and "future_alive_target" in batch:
            survival_loss = absorbing_survival_loss(
                outputs["survival_probability"], batch["future_alive_target"], batch["future_alive_mask"]
            )
        else:
            survival_loss = track_loss * 0.0
        total = (
            self.config.distribution_weight * distribution_loss
            + self.config.local_track_weight * track_loss
            + self.config.survival_weight * survival_loss
        )
        return {
            "loss": total,
            "distribution_ce": distribution_loss,
            "distribution_nll": distribution_loss,
            "survival_bce": survival_loss,
            "local_track_mse_normalized": track_loss,
            "local_track_rmse_km": track_rmse_km,
        }
