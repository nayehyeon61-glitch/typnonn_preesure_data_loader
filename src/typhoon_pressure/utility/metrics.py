from __future__ import annotations

from collections.abc import Mapping

import numpy as np


EARTH_RADIUS_KM = 6371.0088


def _float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _valid_mask(*values, mask=None) -> np.ndarray:
    arrays = [_float_array(value) for value in values]
    valid = np.ones(np.broadcast_shapes(*(array.shape for array in arrays)), dtype=bool)
    for array in arrays:
        valid &= np.isfinite(np.broadcast_to(array, valid.shape))
    if mask is not None:
        valid &= np.broadcast_to(np.asarray(mask, dtype=bool), valid.shape)
    return valid


def wrap_longitude_delta_deg(pred_lon, target_lon) -> np.ndarray:
    """Return pred-target longitude difference in the [-180, 180) interval."""
    delta = _float_array(pred_lon) - _float_array(target_lon)
    return (delta + 180.0) % 360.0 - 180.0


def great_circle_distance_km(
    pred_lat,
    pred_lon,
    target_lat,
    target_lon,
    *,
    mask=None,
) -> np.ndarray:
    """Vectorized haversine distance; invalid or masked entries become NaN."""
    pred_lat, pred_lon, target_lat, target_lon = np.broadcast_arrays(
        _float_array(pred_lat),
        _float_array(pred_lon),
        _float_array(target_lat),
        _float_array(target_lon),
    )
    valid = _valid_mask(pred_lat, pred_lon, target_lat, target_lon, mask=mask)
    valid &= np.abs(pred_lat) <= 90.0
    valid &= np.abs(target_lat) <= 90.0

    lat1 = np.radians(pred_lat)
    lat2 = np.radians(target_lat)
    dlat = lat1 - lat2
    dlon = np.radians(wrap_longitude_delta_deg(pred_lon, target_lon))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    distance = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.where(valid, distance, np.nan)


def _empty_statistics(names: tuple[str, ...]) -> dict[str, int | float | None]:
    return {"count": 0, **{name: None for name in names}}


def track_error_statistics(
    pred_lat,
    pred_lon,
    target_lat,
    target_lon,
    *,
    mask=None,
) -> dict[str, int | float | None]:
    """Aggregate great-circle track errors in kilometres."""
    error = great_circle_distance_km(
        pred_lat, pred_lon, target_lat, target_lon, mask=mask
    )
    valid = error[np.isfinite(error)]
    names = (
        "mean_track_error_km",
        "rmse_track_error_km",
        "median_track_error_km",
        "p90_track_error_km",
        "max_track_error_km",
    )
    if valid.size == 0:
        return _empty_statistics(names)
    return {
        "count": int(valid.size),
        "mean_track_error_km": float(np.mean(valid)),
        "rmse_track_error_km": float(np.sqrt(np.mean(valid**2))),
        "median_track_error_km": float(np.median(valid)),
        "p90_track_error_km": float(np.percentile(valid, 90)),
        "max_track_error_km": float(np.max(valid)),
    }


def scalar_error_statistics(
    prediction,
    target,
    *,
    mask=None,
    prefix: str,
) -> Mapping[str, int | float | None]:
    """Return count, bias, MAE and RMSE for a scalar prediction target."""
    prediction, target = np.broadcast_arrays(_float_array(prediction), _float_array(target))
    valid = _valid_mask(prediction, target, mask=mask)
    error = (prediction - target)[valid]
    names = (f"{prefix}_bias", f"{prefix}_mae", f"{prefix}_rmse")
    if error.size == 0:
        return _empty_statistics(names)
    return {
        "count": int(error.size),
        f"{prefix}_bias": float(np.mean(error)),
        f"{prefix}_mae": float(np.mean(np.abs(error))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(error**2))),
    }
