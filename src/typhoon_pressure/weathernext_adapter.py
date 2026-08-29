from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import xarray as xr

from .initial_condition import WeatherInitialCondition


class WeatherNextRunner(Protocol):
    """Small boundary around the version-pinned Google WN2 runner."""

    def rollout(self, initial_state: xr.Dataset, horizon_hours: int) -> xr.Dataset: ...


@dataclass
class WeatherNextRequest:
    initial_state: xr.Dataset
    tracker_seed: dict
    horizon_hours: int
    initialization_metadata: dict


def make_weathernext_request(
    condition: WeatherInitialCondition,
    horizon_hours: int = 360,
) -> WeatherNextRequest:
    if horizon_hours <= 0 or horizon_hours > 360 or horizon_hours % 6:
        raise ValueError("WeatherNext horizon must be a positive 6-hour multiple up to 360")
    storm = condition.storm
    return WeatherNextRequest(
        initial_state=condition.atmospheric_state,
        tracker_seed={
            "storm_id": storm.storm_id,
            "time": storm.time,
            "lat": storm.lat,
            "lon": storm.lon,
            "pressure_hpa": storm.pressure_hpa,
            "wind_kt": storm.wind_kt,
        },
        horizon_hours=horizon_hours,
        initialization_metadata=condition.metadata(),
    )


def run_weathernext(runner: WeatherNextRunner, request: WeatherNextRequest) -> xr.Dataset:
    """Call a concrete, release-pinned WN2 runner without coupling this package to JAX/TPU."""
    return runner.rollout(request.initial_state, request.horizon_hours)

