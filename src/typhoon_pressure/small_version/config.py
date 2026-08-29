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
    track_scale_km: float = 500.0

