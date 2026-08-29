from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import xarray as xr

if TYPE_CHECKING:
    from .initial_condition import WeatherInitialCondition


class WeatherNextRunner(Protocol):
    """Common boundary implemented by trainable, pretrained and API backends."""

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


def _validate_forecast(forecast: xr.Dataset) -> None:
    if not isinstance(forecast, xr.Dataset):
        raise TypeError("WeatherNext backend must return an xarray.Dataset")
    if not forecast.data_vars:
        raise ValueError("WeatherNext backend returned an empty dataset")
    time_names = {"time", "valid_time", "datetime"}
    if not time_names.intersection(forecast.coords) and not time_names.intersection(forecast.dims):
        raise ValueError("WeatherNext output requires a time or valid_time coordinate")


def run_weathernext(runner: WeatherNextRunner, request: WeatherNextRequest) -> xr.Dataset:
    """Run the selected backend and attach reproducible backend provenance."""
    forecast = runner.rollout(request.initial_state, request.horizon_hours)
    _validate_forecast(forecast)
    provenance = getattr(runner, "provenance", None)
    if callable(provenance):
        attrs = dict(forecast.attrs)
        attrs.update({key: value for key, value in provenance().items() if value is not None})
        forecast = forecast.assign_attrs(attrs)
    return forecast
