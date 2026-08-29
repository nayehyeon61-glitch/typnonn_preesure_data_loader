"""Normalize HRES/ERA5 data into the official WeatherNext input contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
ATMOSPHERIC_VARIABLES = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
SURFACE_VARIABLES = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "sea_surface_temperature",
)
STATIC_VARIABLES = ("geopotential_at_surface", "land_sea_mask")
WN2_ONLY_VARIABLES = ("100m_u_component_of_wind", "100m_v_component_of_wind")

ALIASES = {
    "latitude": "lat",
    "longitude": "lon",
    "valid_time": "time",
    "pressure_level": "level",
    "isobaricInhPa": "level",
    "t": "temperature",
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
    "t2m": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "mslp": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "sst": "sea_surface_temperature",
    "u100": "100m_u_component_of_wind",
    "v100": "100m_v_component_of_wind",
    "z_surf": "geopotential_at_surface",
    "orography": "geopotential_at_surface",
    "lsm": "land_sea_mask",
}


@dataclass(frozen=True)
class WeatherNextInputConfig:
    model_variant: str = "WeatherNext2"
    time_tolerance: str = "3h"
    regrid: bool = False

    @property
    def resolution_degrees(self) -> float:
        return 1.0 if "mini" in self.model_variant.lower() else 0.25

    @property
    def required_variables(self) -> tuple[str, ...]:
        values = ATMOSPHERIC_VARIABLES + SURFACE_VARIABLES + STATIC_VARIABLES
        if "cyclone" not in self.model_variant.lower():
            values += WN2_ONLY_VARIABLES
        return values


def _rename_aliases(dataset: xr.Dataset) -> xr.Dataset:
    rename = {
        source: target
        for source, target in ALIASES.items()
        if (source in dataset.variables or source in dataset.dims)
        and target not in dataset.variables
        and target not in dataset.dims
    }
    return dataset.rename(rename)


def _absolute_time(dataset: xr.Dataset) -> xr.Dataset:
    if "time" not in dataset.dims:
        return dataset
    if np.issubdtype(dataset.time.dtype, np.datetime64):
        return dataset.assign_coords(time=dataset.time.values.astype("datetime64[ns]"))
    if "datetime" not in dataset.coords:
        raise ValueError("Relative time input requires an absolute 'datetime' coordinate")
    values = dataset.datetime
    if "batch" in values.dims:
        if dataset.sizes.get("batch") != 1:
            raise ValueError("WeatherNext input preparation supports batch size 1")
        values = values.isel(batch=0)
    if values.dims != ("time",):
        raise ValueError("datetime coordinate must have time or batch,time dimensions")
    output = dataset.drop_vars("datetime")
    if "batch" in output.dims:
        output = output.isel(batch=0, drop=True)
    return output.assign_coords(time=np.asarray(values.values).astype("datetime64[ns]"))


def normalize_weathernext_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    """Normalize aliases, absolute time, longitude and coordinate ordering."""
    state = _absolute_time(_rename_aliases(dataset))
    if "lon" in state.coords:
        state = state.assign_coords(lon=np.mod(state.lon.astype(float), 360.0)).sortby("lon")
        if np.unique(np.round(state.lon.values, 6)).size != state.sizes["lon"]:
            raise ValueError("Longitude contains duplicates after wrapping to [0, 360)")
    if "lat" in state.coords:
        state = state.sortby("lat")
    if "level" in state.coords:
        state = state.assign_coords(level=state.level.astype(int)).sortby("level")
    if "time" in state.coords:
        state = state.sortby("time")
    return state


def merge_weathernext_sources(primary: xr.Dataset, *supplements: xr.Dataset) -> xr.Dataset:
    """Merge primary HRES/ERA5 with SST/static/100 m supplemental sources."""
    merged = normalize_weathernext_coordinates(primary)
    for supplement in supplements:
        extra = normalize_weathernext_coordinates(supplement)
        collisions = sorted(set(merged.data_vars).intersection(extra.data_vars))
        if collisions:
            extra = extra.drop_vars(collisions)
        merged = xr.merge((merged, extra), compat="no_conflicts", join="outer")
    return merged


def _select_two_times(state: xr.Dataset, init_time: pd.Timestamp, tolerance: str) -> xr.Dataset:
    if "time" not in state.dims:
        raise ValueError("WeatherNext input requires a time dimension")
    targets = pd.DatetimeIndex([init_time - pd.Timedelta("6h"), init_time])
    selected = state.reindex(
        time=targets.values.astype("datetime64[ns]"),
        method="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    if selected.time.size != 2 or bool(selected.time.isnull().any()):
        raise ValueError(f"Could not select WeatherNext initial fields at {targets.tolist()}")
    # reindex uses target coordinates. Verify source availability separately so a
    # tolerance miss cannot become an all-NaN field unnoticed.
    for target in targets:
        distance = np.abs(state.time.values.astype("datetime64[ns]") - np.datetime64(target))
        if distance.size == 0 or distance.min() > np.timedelta64(pd.Timedelta(tolerance).value, "ns"):
            raise ValueError(f"No atmospheric field within {tolerance} of {target}")
    return selected


def _target_grid(resolution: float) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(-90.0, 90.0, round(180.0 / resolution) + 1)
    lon = np.arange(0.0, 360.0, resolution)
    return lat, lon


def _ensure_grid(state: xr.Dataset, config: WeatherNextInputConfig) -> xr.Dataset:
    for name in ("lat", "lon"):
        if name not in state.coords:
            raise ValueError(f"WeatherNext input requires {name!r} coordinate")
    target_lat, target_lon = _target_grid(config.resolution_degrees)
    lat_values = np.asarray(state.lat.values, dtype=float)
    lon_values = np.asarray(state.lon.values, dtype=float)
    if lat_values.min() > -89.999 or lat_values.max() < 89.999:
        raise ValueError("WeatherNext input/regridding requires full global latitude coverage")
    if lon_values.size < 2:
        raise ValueError("WeatherNext input requires a global longitude axis")
    cyclic_gaps = np.diff(np.r_[lon_values, lon_values[0] + 360.0])
    typical_gap = float(np.median(cyclic_gaps))
    if typical_gap <= 0 or float(cyclic_gaps.max()) > typical_gap * 1.5:
        raise ValueError("WeatherNext input/regridding requires full global longitude coverage")
    exact = (
        state.sizes["lat"] == target_lat.size
        and state.sizes["lon"] == target_lon.size
        and np.allclose(state.lat.values, target_lat, atol=1e-5)
        and np.allclose(state.lon.values, target_lon, atol=1e-5)
    )
    if exact:
        return state
    if not config.regrid:
        raise ValueError(
            f"{config.model_variant} requires a global {config.resolution_degrees} degree grid "
            f"({target_lat.size}x{target_lon.size}); received "
            f"{state.sizes['lat']}x{state.sizes['lon']}. Enable regrid explicitly."
        )
    return state.interp(lat=target_lat, lon=target_lon, method="linear")


def prepare_weathernext_input(
    primary: xr.Dataset,
    init_time,
    *,
    supplements: tuple[xr.Dataset, ...] = (),
    config: WeatherNextInputConfig = WeatherNextInputConfig(),
) -> xr.Dataset:
    """Build and strictly validate one two-step WeatherNext initial state.

    Missing physical variables are never fabricated. Supply them through
    ``supplements`` (for example ERA5 SST/static fields and HRES 100 m winds).
    """
    state = merge_weathernext_sources(primary, *supplements)
    state = _select_two_times(state, pd.Timestamp(init_time), config.time_tolerance)
    state = _ensure_grid(state, config)
    if "level" not in state.coords:
        raise ValueError("WeatherNext input requires a pressure-level coordinate")
    received_levels = set(np.asarray(state.level.values, dtype=int).tolist())
    missing_levels = sorted(set(PRESSURE_LEVELS).difference(received_levels))
    if missing_levels:
        raise ValueError(f"WeatherNext input is missing pressure levels: {missing_levels}")
    state = state.sel(level=list(PRESSURE_LEVELS))
    missing_variables = sorted(set(config.required_variables).difference(state.data_vars))
    if missing_variables:
        raise ValueError(
            "WeatherNext input is missing variables; add supplemental source(s): "
            + ", ".join(missing_variables)
        )
    return state[list(config.required_variables)]
