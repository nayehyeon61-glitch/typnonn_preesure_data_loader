from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def hazard_logits_to_survival(hazard_logits: torch.Tensor) -> torch.Tensor:
    """Convert per-lead hazards to a monotone survival curve S_t.

    The first logit is the hazard between initialization and the first configured
    lead (normally day 15); later logits are incremental hazards between leads.
    """
    if hazard_logits.ndim != 2:
        raise ValueError("hazard_logits must have shape [batch, leads]")
    hazards = torch.sigmoid(hazard_logits)
    return torch.cumprod((1.0 - hazards).clamp_min(1e-6), dim=-1)


def conditional_distribution(distribution_logits: torch.Tensor) -> torch.Tensor:
    """Q_t(x): location distribution conditional on the storm being alive."""
    if distribution_logits.ndim != 3:
        raise ValueError("distribution_logits must have shape [batch, leads, cells]")
    return F.softmax(distribution_logits, dim=-1)


def joint_distribution(
    survival_probability: torch.Tensor,
    conditional_probability: torch.Tensor,
) -> torch.Tensor:
    """P_t(x)=S_t Q_t(x); remaining mass 1-S_t corresponds to storm absence."""
    if survival_probability.shape != conditional_probability.shape[:2]:
        raise ValueError("survival and conditional distribution lead shapes do not match")
    return survival_probability.unsqueeze(-1) * conditional_probability


@dataclass(frozen=True)
class ProbabilisticSamples:
    alive: torch.Tensor
    cell_index: torch.Tensor


@torch.no_grad()
def sample_survival_locations(
    survival_probability: torch.Tensor,
    conditional_probability: torch.Tensor,
    *,
    num_samples: int = 1,
    generator: torch.Generator | None = None,
) -> ProbabilisticSamples:
    """Sample storm existence and, conditional on existence, a spatial cell.

    Sampling is deliberately detached from training autograd. Dead samples receive
    cell_index=-1 so downstream track/distribution code cannot mistake absence for
    a valid grid cell.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if survival_probability.shape != conditional_probability.shape[:2]:
        raise ValueError("survival and conditional distribution lead shapes do not match")
    batch, leads, cells = conditional_probability.shape
    survival = survival_probability.clamp(0.0, 1.0)
    alive = torch.rand(
        (num_samples, batch, leads),
        device=survival.device,
        generator=generator,
    ) < survival.unsqueeze(0)
    flat = conditional_probability.reshape(batch * leads, cells)
    sampled = torch.multinomial(
        flat,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    ).transpose(0, 1).reshape(num_samples, batch, leads)
    sampled = torch.where(alive, sampled, torch.full_like(sampled, -1))
    return ProbabilisticSamples(alive=alive, cell_index=sampled)
