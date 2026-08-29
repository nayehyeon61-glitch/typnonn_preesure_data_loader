from __future__ import annotations

import torch
from torch import nn

from .config import EastAsiaBounds, SmallModelConfig, TransformerConfig


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


class GPTForecastRouter(nn.Module):
    """Use a structured GPT state to actively route WeatherNext forecast tokens.

    The router learns two multiplicative gates:

    * token gate: which spatial/lead-time WeatherNext tokens matter now;
    * channel gate: which latent forecast channels should be emphasized.

    Both gates are initialized to the identity (1.0), and a missing GPT state is
    guaranteed to be an exact identity transformation.  Token representations
    already contain forecast values, masks and spatial/lead-time positions, so
    the router can learn rules such as emphasizing northeast/long-lead tokens
    when the GPT semantic state indicates recurvature.
    """

    def __init__(self, gpt_state_dim: int, model_dim: int):
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(gpt_state_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.token_gate = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.channel_gate = nn.Linear(model_dim, model_dim)

        # 2 * sigmoid(0) == 1, so training starts from the old model exactly.
        nn.init.zeros_(self.token_gate[-1].weight)
        nn.init.zeros_(self.token_gate[-1].bias)
        nn.init.zeros_(self.channel_gate.weight)
        nn.init.zeros_(self.channel_gate.bias)

    def forward(
        self,
        forecast_tokens: torch.Tensor,
        gpt_state_values: torch.Tensor,
        gpt_state_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masked_gpt_state = torch.where(
            gpt_state_mask.bool(), gpt_state_values, torch.zeros_like(gpt_state_values)
        )
        available = gpt_state_mask.bool().any(dim=-1, keepdim=True).to(forecast_tokens.dtype)
        context = self.context(torch.cat((masked_gpt_state, gpt_state_mask), dim=-1))

        # Forecast tokens already encode value/mask/lat/lon/lead-time information.
        token_logits = self.token_gate(forecast_tokens + context.unsqueeze(1))
        channel_logits = self.channel_gate(context).unsqueeze(1)
        raw_token_gate = 2.0 * torch.sigmoid(token_logits)
        raw_channel_gate = 2.0 * torch.sigmoid(channel_logits)

        # If GPT is unavailable the router becomes an exact identity, not an
        # inferred pseudo-state based on zeros.
        available_3d = available.unsqueeze(-1)
        token_gate = 1.0 + available_3d * (raw_token_gate - 1.0)
        channel_gate = 1.0 + available_3d * (raw_channel_gate - 1.0)
        routed = forecast_tokens * token_gate * channel_gate
        return routed, token_gate, channel_gate, available


class WeatherNextFusionTransformer(nn.Module):
    """Fuse masked 0–15 day WeatherNext tokens with observed history."""

    def __init__(
        self,
        model_config: SmallModelConfig,
        transformer_config: TransformerConfig,
        bounds: EastAsiaBounds = EastAsiaBounds(),
    ):
        super().__init__()
        self.model_config = model_config
        self.transformer_config = transformer_config
        self.bounds = bounds
        hidden = model_config.hidden_dim

        self.history_projection = nn.Sequential(
            nn.Linear(model_config.input_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.history_encoder = nn.GRU(hidden, hidden, batch_first=True)

        forecast_projection_dim = transformer_config.forecast_input_dim * 2 + 6
        self.forecast_projection = nn.Sequential(
            nn.Linear(forecast_projection_dim, transformer_config.model_dim),
            nn.LayerNorm(transformer_config.model_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_config.model_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_config.model_dim,
            nhead=transformer_config.num_heads,
            dim_feedforward=transformer_config.feedforward_dim,
            dropout=transformer_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.forecast_encoder = nn.TransformerEncoder(layer, transformer_config.num_layers)

        self.gpt_history_conditioner = None
        self.gpt_forecast_router = None
        if transformer_config.gpt_state_dim > 0:
            self.gpt_history_conditioner = nn.Sequential(
                nn.Linear(transformer_config.gpt_state_dim * 2, hidden * 2),
                nn.LayerNorm(hidden * 2),
                nn.GELU(),
                nn.Linear(hidden * 2, hidden * 2),
            )
            self.gpt_forecast_router = GPTForecastRouter(
                transformer_config.gpt_state_dim,
                transformer_config.model_dim,
            )

        self.fusion = nn.Sequential(
            nn.Linear(hidden + transformer_config.model_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.history_to_forecast_dim = nn.Linear(hidden, transformer_config.model_dim)
        self.future_queries = nn.Parameter(torch.zeros(
            1, len(model_config.lead_days), transformer_config.model_dim
        ))
        nn.init.normal_(self.future_queries, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_config.model_dim,
            nhead=transformer_config.num_heads,
            dim_feedforward=transformer_config.feedforward_dim,
            dropout=transformer_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.future_decoder = nn.TransformerDecoder(
            decoder_layer, transformer_config.decoder_layers
        )
        self.distribution_projection = nn.Linear(
            transformer_config.model_dim, model_config.n_cells
        )
        self.track_head = nn.Linear(hidden, model_config.local_track_steps * 2)

    def _apply_input_mask(
        self,
        values: torch.Tensor,
        feature_mask: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine missing-data, token-validity and random training masks."""
        effective_feature_mask = feature_mask.bool()
        probability = self.transformer_config.input_mask_probability
        if self.training and probability > 0:
            random_keep = torch.rand_like(values) >= probability
            effective_feature_mask = effective_feature_mask & random_keep
        effective_token_mask = token_mask.bool() & effective_feature_mask.any(dim=-1)
        masked_values = torch.where(effective_feature_mask, values, torch.zeros_like(values))
        return masked_values, effective_feature_mask.to(values.dtype), effective_token_mask

    def forward(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        forecast_values: torch.Tensor,
        forecast_feature_mask: torch.Tensor,
        forecast_token_mask: torch.Tensor,
        forecast_positions: torch.Tensor,
        gpt_state_values: torch.Tensor | None = None,
        gpt_state_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        masked_history = torch.where(history_mask.bool(), history, torch.zeros_like(history))
        history_input = self.history_projection(torch.cat((masked_history, history_mask), dim=-1))
        gpt_conditioning_fraction = None
        if self.gpt_history_conditioner is not None:
            if gpt_state_values is None or gpt_state_mask is None:
                raise ValueError("GPT state tensors are required when gpt_state_dim > 0")
            masked_gpt_state = torch.where(
                gpt_state_mask.bool(), gpt_state_values, torch.zeros_like(gpt_state_values)
            )
            gamma, beta = self.gpt_history_conditioner(
                torch.cat((masked_gpt_state, gpt_state_mask), dim=-1)
            ).chunk(2, dim=-1)
            state_available = gpt_state_mask.bool().any(dim=-1, keepdim=True).to(history.dtype)
            gamma = 0.5 * torch.tanh(gamma) * state_available
            beta = torch.tanh(beta) * state_available
            # GPT changes history dynamics before the GRU; a missing state is exact identity.
            history_input = history_input * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            gpt_conditioning_fraction = state_available.mean().detach()
        _, history_hidden = self.history_encoder(history_input)

        masked_values, effective_feature_mask, effective_token_mask = self._apply_input_mask(
            forecast_values, forecast_feature_mask, forecast_token_mask
        )
        forecast_input = self.forecast_projection(torch.cat((
            masked_values, effective_feature_mask, forecast_positions,
        ), dim=-1))

        router_token_gate = None
        router_channel_gate = None
        router_active_fraction = None
        if self.gpt_forecast_router is not None:
            if gpt_state_values is None or gpt_state_mask is None:
                raise ValueError("GPT state tensors are required when gpt_state_dim > 0")
            forecast_input, router_token_gate, router_channel_gate, router_available = (
                self.gpt_forecast_router(
                    forecast_input,
                    gpt_state_values,
                    gpt_state_mask,
                )
            )
            router_active_fraction = router_available.mean().detach()

        cls = self.cls_token.expand(forecast_input.shape[0], -1, -1)
        forecast_input = torch.cat((cls, forecast_input), dim=1)
        # CLS is always valid, preventing all-masked sequences from producing NaNs.
        padding_mask = torch.cat((
            torch.zeros((effective_token_mask.shape[0], 1), dtype=torch.bool, device=effective_token_mask.device),
            ~effective_token_mask,
        ), dim=1)
        forecast_memory = self.forecast_encoder(
            forecast_input, src_key_padding_mask=padding_mask
        )
        forecast_state = forecast_memory[:, 0]
        state = self.fusion(torch.cat((history_hidden[-1], forecast_state), dim=-1))

        # One learned query per future day cross-attends to the routed 0–15 day memory.
        queries = self.future_queries.expand(history.shape[0], -1, -1)
        queries = queries + self.history_to_forecast_dim(state).unsqueeze(1)
        future_states = self.future_decoder(
            tgt=queries,
            memory=forecast_memory,
            memory_key_padding_mask=padding_mask,
        )
        distribution_logits = self.distribution_projection(future_states)
        raw_track = self.track_head(state).view(-1, self.model_config.local_track_steps, 2)
        unit_track = torch.sigmoid(raw_track)
        lat = self.bounds.lat_min + (self.bounds.lat_max - self.bounds.lat_min) * unit_track[..., 0]
        lon = self.bounds.lon_min + (self.bounds.lon_max - self.bounds.lon_min) * unit_track[..., 1]
        result = {
            "distribution_logits": distribution_logits,
            "track_latlon": torch.stack((lat, lon), dim=-1),
            "effective_forecast_token_fraction": effective_token_mask.float().mean().detach(),
            "effective_forecast_feature_fraction": effective_feature_mask.mean().detach(),
        }
        if gpt_conditioning_fraction is not None:
            result["gpt_history_conditioning_fraction"] = gpt_conditioning_fraction
        if router_token_gate is not None and router_channel_gate is not None:
            valid = effective_token_mask.to(router_token_gate.dtype)
            token_gate_values = router_token_gate.squeeze(-1)
            token_gate_mean = (token_gate_values * valid).sum() / valid.sum().clamp_min(1.0)
            result.update(
                {
                    "gpt_forecast_router_active_fraction": router_active_fraction,
                    "gpt_forecast_token_gate_mean": token_gate_mean.detach(),
                    "gpt_forecast_channel_gate_mean": router_channel_gate.mean().detach(),
                    # Detached maps make routing decisions inspectable without
                    # retaining the training graph.
                    "gpt_forecast_token_gate": token_gate_values.detach(),
                    "gpt_forecast_channel_gate": router_channel_gate.squeeze(1).detach(),
                }
            )
        return result
