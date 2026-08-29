from __future__ import annotations

import torch
from torch import nn

from .config import EastAsiaBounds, SmallModelConfig


class SmallDualScaleModel(nn.Module):
    """One history encoder with global-distribution and local-track heads."""

    def __init__(self, config: SmallModelConfig, bounds: EastAsiaBounds = EastAsiaBounds()):
        super().__init__()
        self.config = config
        self.bounds = bounds
        self.input_projection = nn.Sequential(
            nn.Linear(config.input_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.encoder = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.distribution_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU(),
            nn.Linear(config.hidden_dim, len(config.lead_days) * config.n_cells),
        )
        self.track_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU(),
            nn.Linear(config.hidden_dim, config.local_track_steps * 2),
        )

    def forward(self, history: torch.Tensor, history_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded_input = self.input_projection(torch.cat((history, history_mask), dim=-1))
        _, hidden = self.encoder(encoded_input)
        state = hidden[-1]
        distribution_logits = self.distribution_head(state).view(
            -1, len(self.config.lead_days), self.config.n_cells
        )
        raw_track = self.track_head(state).view(-1, self.config.local_track_steps, 2)
        unit_track = torch.sigmoid(raw_track)
        lat = self.bounds.lat_min + (self.bounds.lat_max - self.bounds.lat_min) * unit_track[..., 0]
        lon = self.bounds.lon_min + (self.bounds.lon_max - self.bounds.lon_min) * unit_track[..., 1]
        return {"distribution_logits": distribution_logits, "track_latlon": torch.stack((lat, lon), dim=-1)}

