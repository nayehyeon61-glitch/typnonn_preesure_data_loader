from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EastAsiaBounds:
    """Local evaluation domain; longitudes use the [0, 360) convention here."""

    lat_min: float = 0.0
    lat_max: float = 60.0
    lon_min: float = 100.0
    lon_max: float = 180.0


@dataclass(frozen=True)
class SmallModelConfig:
    input_dim: int
    history_steps: int = 8
    hidden_dim: int = 128
    distribution_start_day: int = 15
    distribution_end_day: int = 30
    distribution_step_days: int = 1
    local_track_steps: int = 20
    lat_bin_deg: float = 5.0
    lon_bin_deg: float = 5.0

    @property
    def lead_days(self) -> tuple[int, ...]:
        return tuple(range(
            self.distribution_start_day,
            self.distribution_end_day + 1,
            self.distribution_step_days,
        ))

    @property
    def n_lat(self) -> int:
        return round(180.0 / self.lat_bin_deg)

    @property
    def n_lon(self) -> int:
        return round(360.0 / self.lon_bin_deg)

    @property
    def n_cells(self) -> int:
        return self.n_lat * self.n_lon


@dataclass(frozen=True)
class DualLossConfig:
    distribution_weight: float = 1.0
    local_track_weight: float = 1.0
    survival_weight: float = 1.0
    track_scale_km: float = 500.0


@dataclass(frozen=True)
class DistributionSamplingConfig:
    """Kalman-inspired adaptive process-noise sampling for day 15-30 states.

    The sampler uses a learned random-walk transition with a positive-definite
    process covariance Q_t. Samples are propagated recursively in time, so one
    sample index corresponds to one coherent stochastic trajectory rather than
    independent per-day draws.
    """

    num_samples: int = 32
    min_process_std_deg: float = 0.25
    max_process_std_deg: float = 12.0
    min_initial_std_deg: float = 0.25
    max_initial_std_deg: float = 8.0
    max_initial_correction_deg: float = 5.0
    max_daily_displacement_deg: float = 15.0
    grid_kernel_std_deg: float = 5.0

    def __post_init__(self):
        if self.num_samples < 2:
            raise ValueError("num_samples must be at least 2")
        if self.min_process_std_deg <= 0:
            raise ValueError("min_process_std_deg must be positive")
        if self.max_process_std_deg <= self.min_process_std_deg:
            raise ValueError("max_process_std_deg must exceed min_process_std_deg")
        if self.min_initial_std_deg <= 0:
            raise ValueError("min_initial_std_deg must be positive")
        if self.max_initial_std_deg <= self.min_initial_std_deg:
            raise ValueError("max_initial_std_deg must exceed min_initial_std_deg")
        if self.max_initial_correction_deg < 0:
            raise ValueError("max_initial_correction_deg must be non-negative")
        if self.max_daily_displacement_deg <= 0:
            raise ValueError("max_daily_displacement_deg must be positive")
        if self.grid_kernel_std_deg <= 0:
            raise ValueError("grid_kernel_std_deg must be positive")


@dataclass(frozen=True)
class WeatherNextTokenConfig:
    """Compression settings for a small Transformer over WeatherNext fields."""

    variables: tuple[str, ...] = (
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
    )
    max_lead_hours: int = 360
    max_time_steps: int = 10
    target_lat_tokens: int = 6
    target_lon_tokens: int = 12
    min_valid_fraction: float = 0.5

    @property
    def max_tokens(self) -> int:
        return self.max_time_steps * self.target_lat_tokens * self.target_lon_tokens


@dataclass(frozen=True)
class TransformerConfig:
    forecast_input_dim: int
    gpt_state_dim: int = 0
    model_dim: int = 128
    num_heads: int = 8
    num_layers: int = 4
    decoder_layers: int = 2
    feedforward_dim: int = 384
    dropout: float = 0.1
    input_mask_probability: float = 0.15

    def __post_init__(self):
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= self.input_mask_probability < 1.0:
            raise ValueError("input_mask_probability must be in [0, 1)")


@dataclass(frozen=True)
class GPTStateConfig:
    model: str = "gpt-5.6"
