from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import maximum_filter, uniform_filter

from .geometry import distance_bearing_km, relative_xy_km


@dataclass(frozen=True)
class HighPressureConfig:
    radius_km: float = 2500.0
    inner_exclusion_km: float = 250.0
    local_window: int = 9
    background_window: int = 21
    min_anomaly_hpa: float = 1.5
    min_separation_km: float = 400.0
    max_highs: int = 3
    time_tolerance: str = "3h"


def normalize_era5_mslp(ds: xr.Dataset | xr.DataArray) -> xr.DataArray:
    if isinstance(ds, xr.DataArray):
        da = ds
    else:
        rename_vars = {
            "valid_time": "time", "latitude": "lat", "longitude": "lon",
            "msl": "mslp", "mean_sea_level_pressure": "mslp",
        }
        rename = {old: new for old, new in rename_vars.items() if old in ds.variables or old in ds.dims}
        ds = ds.rename(rename)
        if "mslp" not in ds:
            raise KeyError(f"MSLP variable not found; available={list(ds.data_vars)}")
        da = ds["mslp"]
    rename = {old: new for old, new in {"valid_time": "time", "latitude": "lat", "longitude": "lon"}.items() if old in da.dims}
    da = da.rename(rename)
    return da.transpose("time", "lat", "lon")


def _separated(candidates: list[dict], minimum_km: float, limit: int) -> list[dict]:
    kept: list[dict] = []
    for row in candidates:
        if all(float(distance_bearing_km(row["high_lat"], row["high_lon"], old["high_lat"], old["high_lon"])[0]) >= minimum_km for old in kept):
            kept.append(row)
            if len(kept) >= limit:
                break
    return kept


def extract_surrounding_highs(
    track: pd.DataFrame,
    mslp: xr.DataArray,
    config: HighPressureConfig = HighPressureConfig(),
) -> pd.DataFrame:
    """Detect high-pressure centres around every IBTrACS point."""
    mslp = normalize_era5_mslp(mslp)
    era_times = pd.DatetimeIndex(pd.to_datetime(mslp.time.values))
    tolerance = pd.Timedelta(config.time_tolerance)
    lats = np.asarray(mslp.lat.values, dtype=float)
    lons = np.asarray(mslp.lon.values, dtype=float)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    output: list[dict] = []

    for point in track.itertuples(index=False):
        nearest = era_times.get_indexer([pd.Timestamp(point.time)], method="nearest", tolerance=tolerance)[0]
        if nearest < 0:
            continue
        field = np.asarray(mslp.isel(time=nearest).values, dtype=float)
        if np.nanmedian(field) > 2000:
            field = field / 100.0
        finite = np.isfinite(field)
        clean = np.where(finite, field, np.nanmedian(field))
        background = uniform_filter(clean, config.background_window, mode=("nearest", "wrap"))
        anomaly = clean - background
        local_max = clean == maximum_filter(clean, config.local_window, mode=("nearest", "wrap"))
        distance, bearing = distance_bearing_km(point.typhoon_lat, point.typhoon_lon, lat_grid, lon_grid)
        mask = finite & local_max & (anomaly >= config.min_anomaly_hpa)
        mask &= (distance <= config.radius_km) & (distance >= config.inner_exclusion_km)
        indices = np.argwhere(mask)
        candidates = []
        for i, j in indices:
            dx, dy = relative_xy_km(point.typhoon_lat, point.typhoon_lon, lats[i], lons[j])
            candidates.append({
                "storm_id": point.storm_id,
                "time": pd.Timestamp(point.time),
                "era5_time": era_times[nearest],
                "high_lat": float(lats[i]),
                "high_lon": float(lons[j]),
                "high_pressure_hpa": float(clean[i, j]),
                "high_anomaly_hpa": float(anomaly[i, j]),
                "high_distance_km": float(distance[i, j]),
                "high_bearing_deg": float(bearing[i, j]),
                "high_dx_km": float(dx),
                "high_dy_km": float(dy),
            })
        candidates.sort(key=lambda row: row["high_anomaly_hpa"], reverse=True)
        for rank, row in enumerate(_separated(candidates, config.min_separation_km, config.max_highs), start=1):
            row["high_rank"] = rank
            output.append(row)
    columns = [
        "storm_id", "time", "era5_time", "high_rank", "high_lat", "high_lon",
        "high_pressure_hpa", "high_anomaly_hpa", "high_distance_km",
        "high_bearing_deg", "high_dx_km", "high_dy_km",
    ]
    return pd.DataFrame(output, columns=columns)

