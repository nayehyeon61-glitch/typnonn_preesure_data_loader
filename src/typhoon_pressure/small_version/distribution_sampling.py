"""Differentiable time-correlated trajectory sampling for long-range forecasts."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import DistributionSamplingConfig, SmallModelConfig


class AdaptiveDistributionSampler(nn.Module):
    """Kalman-inspired adaptive process-noise sampler for day 15--30.

    This is a process model rather than a full Kalman measurement update. The
    routed WeatherNext/fusion representation acts as the dynamical prior and a
    learned positive-definite Q_t controls uncertainty growth. When GPT state
    is present it explicitly conditions Q_t, so semantic confidence and track
    uncertainty can change the sampled spread.

    One sample index is propagated recursively through every lead day. The
    output therefore represents K coherent stochastic trajectories instead of
    independent per-day draws.
    """

    def __init__(
        self,
        model_config: SmallModelConfig,
        model_dim: int,
        *,
        gpt_state_dim: int = 0,
        sampling_config: DistributionSamplingConfig = DistributionSamplingConfig(),
    ):
        super().__init__()
        self.model_config = model_config
        self.model_dim = model_dim
        self.gpt_state_dim = gpt_state_dim
        self.config = sampling_config

        self.initial_state_head = nn.Linear(model_dim, 2)
        self.drift_head = nn.Linear(model_dim, 2)
        self.process_noise_head = nn.Linear(model_dim, 3)
        self.gpt_noise_context = None
        if gpt_state_dim > 0:
            self.gpt_noise_context = nn.Sequential(
                nn.Linear(gpt_state_dim * 2, model_dim),
                nn.LayerNorm(model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
            )

        # Start with modest uncorrelated Q_t. The diagonal raw parameters are
        # negative (small std), while rho starts exactly at zero.
        nn.init.zeros_(self.process_noise_head.weight)
        with torch.no_grad():
            self.process_noise_head.bias.copy_(torch.tensor([-2.0, 0.0, -2.0]))

        lat_centres = -90.0 + (
            torch.arange(model_config.n_lat, dtype=torch.float32) + 0.5
        ) * model_config.lat_bin_deg
        lon_centres = (
            torch.arange(model_config.n_lon, dtype=torch.float32) + 0.5
        ) * model_config.lon_bin_deg
        lat_grid, lon_grid = torch.meshgrid(lat_centres, lon_centres, indexing="ij")
        self.register_buffer("grid_lat", lat_grid.reshape(-1), persistent=False)
        self.register_buffer("grid_lon", lon_grid.reshape(-1), persistent=False)

    def _gpt_context(
        self,
        future_states: torch.Tensor,
        gpt_state_values: torch.Tensor | None,
        gpt_state_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.gpt_noise_context is None:
            return torch.zeros_like(future_states)
        if gpt_state_values is None or gpt_state_mask is None:
            raise ValueError("GPT state tensors are required for GPT-conditioned process noise")
        masked = torch.where(
            gpt_state_mask.bool(), gpt_state_values, torch.zeros_like(gpt_state_values)
        )
        available = gpt_state_mask.bool().any(dim=-1, keepdim=True).to(future_states.dtype)
        context = self.gpt_noise_context(torch.cat((masked, gpt_state_mask), dim=-1))
        return context.unsqueeze(1) * available.unsqueeze(1)

    def _cholesky(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.process_noise_head(condition)
        span = self.config.max_process_std_deg - self.config.min_process_std_deg
        std_lat = self.config.min_process_std_deg + span * torch.sigmoid(raw[..., 0])
        std_lon = self.config.min_process_std_deg + span * torch.sigmoid(raw[..., 2])
        rho = 0.95 * torch.tanh(raw[..., 1])
        residual = torch.sqrt((1.0 - rho.square()).clamp_min(1e-5))

        chol = torch.zeros(*raw.shape[:-1], 2, 2, dtype=raw.dtype, device=raw.device)
        chol[..., 0, 0] = std_lat
        chol[..., 1, 0] = rho * std_lon
        chol[..., 1, 1] = std_lon * residual
        covariance = chol @ chol.transpose(-1, -2)
        return chol, covariance

    @staticmethod
    def _wrap_state(state: torch.Tensor) -> torch.Tensor:
        lat = state[..., 0].clamp(-89.75, 89.75)
        lon = torch.remainder(state[..., 1], 360.0)
        return torch.stack((lat, lon), dim=-1)

    def _sample_grid_distribution(
        self, samples: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # samples: [B, K, T, 2]. Soft rasterization is differentiable, unlike a
        # hard histogram. Work in log space so distant grid cells keep usable
        # gradients even when their float32 probability would underflow to 0.
        sample_lat = samples[..., 0].unsqueeze(-1)
        sample_lon = samples[..., 1].unsqueeze(-1)
        dlat = sample_lat - self.grid_lat
        dlon = torch.remainder(sample_lon - self.grid_lon + 180.0, 360.0) - 180.0
        lon_scale = torch.cos(torch.deg2rad(sample_lat)).abs().clamp_min(0.1)
        distance2 = dlat.square() + (dlon * lon_scale).square()
        sigma2 = self.config.grid_kernel_std_deg ** 2
        per_sample_log_probability = torch.log_softmax(-0.5 * distance2 / sigma2, dim=-1)
        log_probability = torch.logsumexp(per_sample_log_probability, dim=1) - math.log(
            self.config.num_samples
        )
        return log_probability.exp(), log_probability

    def forward(
        self,
        future_states: torch.Tensor,
        gpt_state_values: torch.Tensor | None = None,
        gpt_state_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if future_states.ndim != 3:
            raise ValueError("future_states must have shape [batch, lead, model_dim]")
        if future_states.shape[1] != len(self.model_config.lead_days):
            raise ValueError("future_states lead dimension does not match model lead_days")

        condition = future_states + self._gpt_context(
            future_states, gpt_state_values, gpt_state_mask
        )
        chol, covariance = self._cholesky(condition)

        initial_unit = torch.sigmoid(self.initial_state_head(future_states[:, 0]))
        initial_mean = torch.stack((
            -90.0 + 180.0 * initial_unit[..., 0],
            360.0 * initial_unit[..., 1],
        ), dim=-1)
        daily_scale = self.config.max_daily_displacement_deg * self.model_config.distribution_step_days
        drift = daily_scale * torch.tanh(self.drift_head(future_states))
        drift = drift.clone()
        drift[:, 0] = 0.0

        means = [self._wrap_state(initial_mean)]
        for lead in range(1, future_states.shape[1]):
            means.append(self._wrap_state(means[-1] + drift[:, lead]))
        mean_latlon = torch.stack(means, dim=1)

        batch, leads, _ = mean_latlon.shape
        samples = []
        previous = None
        for lead in range(leads):
            epsilon = torch.randn(
                batch, self.config.num_samples, 2,
                dtype=future_states.dtype,
                device=future_states.device,
            )
            noise = torch.einsum("bij,bkj->bki", chol[:, lead], epsilon)
            if lead == 0:
                current = mean_latlon[:, 0].unsqueeze(1) + noise
            else:
                current = previous + drift[:, lead].unsqueeze(1) + noise
            current = self._wrap_state(current)
            samples.append(current)
            previous = current
        sample_trajectories = torch.stack(samples, dim=2)
        probabilities, log_probabilities = self._sample_grid_distribution(sample_trajectories)

        return {
            "distribution_mean_latlon": mean_latlon,
            "distribution_process_cholesky": chol,
            "distribution_process_covariance": covariance,
            "distribution_samples": sample_trajectories,
            "distribution_probabilities": probabilities,
            "distribution_log_probabilities": log_probabilities,
            "distribution_process_std_mean": torch.sqrt(
                covariance.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
            ).mean().detach(),
            "distribution_sample_spread_deg": sample_trajectories.std(dim=1).mean().detach(),
        }
