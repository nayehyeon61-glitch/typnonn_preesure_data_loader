from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter, minimum_filter

from .geometry import distance_bearing_km


InitializationMode = Literal["tracker_seed", "vortex_correction", "auto"]


@dataclass(frozen=True)
class StormObservation:
    storm_id: str
    time: pd.Timestamp
    lat: float
    lon: float
    pressure_hpa: float | None = None
    wind_kt: float | None = None

    @classmethod
    def from_series(cls, row: pd.Series) -> "StormObservation":
        def optional_float(value):
            return None if pd.isna(value) else float(value)

        return cls(
            storm_id=str(row["storm_id"]),
            time=pd.Timestamp(row["time"]),
            lat=float(row["typhoon_lat"]),
            lon=float(row["typhoon_lon"]),
            pressure_hpa=optional_float(row.get("typhoon_pressure_hpa")),
            wind_kt=optional_float(row.get("typhoon_wind_kt")),
        )


@dataclass(frozen=True)
class ModelStormCenter:
    lat: float
    lon: float
    pressure_hpa: float


@dataclass(frozen=True)
class CorrectionConfig:
    position_threshold_km: float = 100.0
    pressure_threshold_hpa: float = 5.0
    search_radius_km: float = 500.0
    correction_radius_km: float = 400.0
    local_window: int = 7
    max_pressure_correction_hpa: float = 25.0
    background_sigma_gridpoints: float = 8.0


@dataclass
class WeatherInitialCondition:
    atmospheric_state: xr.Dataset
    storm: StormObservation
    requested_mode: InitializationMode
    applied_mode: Literal["tracker_seed", "vortex_correction"]
    model_center_before: ModelStormCenter | None
    position_error_km: float | None
    pressure_error_hpa: float | None
    correction_applied: bool

    def metadata(self) -> dict:
        center = self.model_center_before
        return {
            "storm_id": self.storm.storm_id,
            "init_time": self.storm.time,
            "requested_mode": self.requested_mode,
            "applied_mode": self.applied_mode,
            "correction_applied": self.correction_applied,
            "position_error_km": self.position_error_km,
            "pressure_error_hpa": self.pressure_error_hpa,
            "model_center_lat": None if center is None else center.lat,
            "model_center_lon": None if center is None else center.lon,
            "model_center_pressure_hpa": None if center is None else center.pressure_hpa,
            "input_history_steps": self.atmospheric_state.sizes.get("time", 1),
        }


def _time_history_state(
    state: xr.Dataset,
    time: pd.Timestamp,
    tolerance: str,
    history_steps: int,
) -> xr.Dataset:
    rename = {
        old: new
        for old, new in {
            "valid_time": "time", "latitude": "lat", "longitude": "lon",
            "msl": "mslp", "mean_sea_level_pressure": "mslp",
        }.items()
        if old in state.variables or old in state.dims
    }
    state = state.rename(rename)
    if "time" in state.dims and not np.issubdtype(state.time.dtype, np.datetime64):
        if "datetime" not in state.coords:
            raise ValueError("Relative time input requires an absolute 'datetime' coordinate")
        datetimes = state.datetime
        if "batch" in datetimes.dims:
            if state.sizes.get("batch") != 1:
                raise ValueError("InitialConditionBuilder supports batch size 1")
            datetimes = datetimes.isel(batch=0)
            state = state.isel(batch=0, drop=True)
        if datetimes.dims != ("time",):
            raise ValueError("datetime coordinate must have time or batch,time dimensions")
        state = state.drop_vars("datetime").assign_coords(
            time=np.asarray(datetimes.values).astype("datetime64[ns]")
        )
    if "time" not in state.dims:
        if history_steps != 1:
            raise ValueError("Multiple WeatherNext history steps require a time dimension")
        return state.copy(deep=True)
    nearest = state.sel(
        time=np.datetime64(time), method="nearest", tolerance=pd.Timedelta(tolerance)
    ).time.values
    eligible = state.sel(time=slice(None, nearest))
    if eligible.sizes["time"] < history_steps:
        raise ValueError(
            f"Initial atmospheric state needs {history_steps} history steps at or before {time}"
        )
    selected = eligible.isel(time=slice(-history_steps, None))
    if history_steps == 1:
        selected = selected.isel(time=0, drop=True)
    return selected.copy(deep=True)


def detect_model_storm_center(
    state: xr.Dataset,
    storm: StormObservation,
    config: CorrectionConfig = CorrectionConfig(),
) -> ModelStormCenter | None:
    if "mslp" not in state:
        raise KeyError("Normalized initial state has no 'mslp' variable")
    mslp = state["mslp"]
    if "time" in mslp.dims:
        if mslp.sizes["time"] != 1:
            raise ValueError("Initial atmospheric state must contain exactly one time")
        mslp = mslp.isel(time=0)
    field = np.asarray(mslp.values, dtype=float)
    if np.nanmedian(field) > 2_000:
        field = field / 100.0
    finite = np.isfinite(field)
    if not finite.any():
        return None
    clean = np.where(finite, field, np.nanmedian(field))
    local_min = clean == minimum_filter(clean, size=config.local_window, mode=("nearest", "wrap"))
    lon_grid, lat_grid = np.meshgrid(mslp.lon.values, mslp.lat.values)
    distance, _ = distance_bearing_km(storm.lat, storm.lon, lat_grid, lon_grid)
    candidates = np.argwhere(finite & local_min & (distance <= config.search_radius_km))
    if not len(candidates):
        return None
    i, j = min(candidates, key=lambda ij: clean[ij[0], ij[1]])
    return ModelStormCenter(
        lat=float(mslp.lat.values[i]),
        lon=float(mslp.lon.values[j]),
        pressure_hpa=float(clean[i, j]),
    )


def correct_mslp_vortex(
    state: xr.Dataset,
    storm: StormObservation,
    model_center: ModelStormCenter,
    config: CorrectionConfig = CorrectionConfig(),
) -> xr.Dataset:
    """Relocate the model pressure anomaly and adjust it to IBTrACS pressure.

    Only MSLP is changed. This is an experimental preprocessing correction;
    callers should compare it against tracker_seed and validate balance after rollout.
    """
    if storm.pressure_hpa is None:
        return state
    output = state.copy(deep=True)
    variable = "mslp"
    if variable not in output:
        raise KeyError("Normalized initial state has no 'mslp' variable")
    da = output[variable]
    field_pa = np.asarray(da.values, dtype=float)
    was_pa = np.nanmedian(field_pa) > 2_000
    field = field_pa / 100.0 if was_pa else field_pa.copy()
    background = gaussian_filter(field, sigma=config.background_sigma_gridpoints, mode=("nearest", "wrap"))
    anomaly = field - background
    lon_grid, lat_grid = np.meshgrid(da.lon.values, da.lat.values)
    source_distance, _ = distance_bearing_km(
        model_center.lat, model_center.lon, lat_grid, lon_grid
    )
    target_distance, _ = distance_bearing_km(storm.lat, storm.lon, lat_grid, lon_grid)
    source_weight = np.exp(-0.5 * (source_distance / config.correction_radius_km) ** 2)
    target_weight = np.exp(-0.5 * (target_distance / config.correction_radius_km) ** 2)

    # Remove the old pressure vortex, then interpolate its anomaly onto the
    # IBTrACS-centred coordinate offset without moving the environmental field.
    cleaned = field - source_weight * anomaly
    lat_shift = storm.lat - model_center.lat
    lon_shift = ((storm.lon - model_center.lon + 180.0) % 360.0) - 180.0
    source_lat = xr.DataArray(da.lat.values - lat_shift, dims="lat", coords={"lat": da.lat.values})
    source_lon = xr.DataArray(da.lon.values - lon_shift, dims="lon", coords={"lon": da.lon.values})
    anomaly_da = xr.DataArray(anomaly, dims=("lat", "lon"), coords={"lat": da.lat, "lon": da.lon})
    shifted = anomaly_da.interp(lat=source_lat, lon=source_lon, kwargs={"fill_value": 0.0}).values
    relocated = cleaned + target_weight * shifted

    desired_delta = float(storm.pressure_hpa - relocated[np.unravel_index(np.argmin(target_distance), target_distance.shape)])
    desired_delta = float(np.clip(
        desired_delta,
        -config.max_pressure_correction_hpa,
        config.max_pressure_correction_hpa,
    ))
    corrected = relocated + target_weight * desired_delta
    output[variable] = xr.DataArray(
        corrected * 100.0 if was_pa else corrected,
        dims=da.dims,
        coords=da.coords,
        attrs=da.attrs,
    )
    output[variable].attrs["ibtracs_vortex_correction"] = "experimental_mslp_only"
    return output


class InitialConditionBuilder:
    def __init__(
        self,
        mode: InitializationMode = "auto",
        config: CorrectionConfig = CorrectionConfig(),
        time_tolerance: str = "3h",
        history_steps: int = 1,
    ):
        if mode not in {"tracker_seed", "vortex_correction", "auto"}:
            raise ValueError(f"Unknown initialization mode: {mode}")
        self.mode = mode
        self.config = config
        self.time_tolerance = time_tolerance
        if history_steps < 1:
            raise ValueError("history_steps must be at least 1")
        self.history_steps = history_steps

    def build(self, atmospheric_state: xr.Dataset, storm: StormObservation) -> WeatherInitialCondition:
        state = _time_history_state(
            atmospheric_state,
            storm.time,
            self.time_tolerance,
            self.history_steps,
        )
        current_state = state.isel(time=-1, drop=True) if "time" in state.dims else state
        center = detect_model_storm_center(current_state, storm, self.config)
        position_error = None
        pressure_error = None
        if center is not None:
            position_error = float(distance_bearing_km(
                center.lat, center.lon, storm.lat, storm.lon
            )[0])
            if storm.pressure_hpa is not None:
                pressure_error = center.pressure_hpa - storm.pressure_hpa

        correct = self.mode == "vortex_correction"
        if self.mode == "auto" and center is not None:
            correct = position_error > self.config.position_threshold_km
            if pressure_error is not None:
                correct |= abs(pressure_error) > self.config.pressure_threshold_hpa
        correct &= center is not None and storm.pressure_hpa is not None

        corrected = state
        if correct:
            corrected_current = correct_mslp_vortex(
                current_state, storm, center, self.config
            )
            if "time" in state.dims:
                corrected = state.copy(deep=True)
                corrected["mslp"].loc[{"time": state.time.values[-1]}] = (
                    corrected_current["mslp"]
                )
            else:
                corrected = corrected_current
        return WeatherInitialCondition(
            atmospheric_state=corrected,
            storm=storm,
            requested_mode=self.mode,
            applied_mode="vortex_correction" if correct else "tracker_seed",
            model_center_before=center,
            position_error_km=position_error,
            pressure_error_hpa=pressure_error,
            correction_applied=bool(correct),
        )
