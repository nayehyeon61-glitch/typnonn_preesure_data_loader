import numpy as np
import pandas as pd
import pytest
import xarray as xr

from typhoon_pressure.weathernext_input import (
    ATMOSPHERIC_VARIABLES,
    PRESSURE_LEVELS,
    STATIC_VARIABLES,
    SURFACE_VARIABLES,
    WN2_ONLY_VARIABLES,
    WeatherNextInputConfig,
    prepare_weathernext_input,
)


def _source(*, include_100m=False, relative_time=False):
    times = pd.date_range("2025-01-01", periods=3, freq="6h")
    time_coord = np.asarray([-12, -6, 0], dtype="timedelta64[h]") if relative_time else times
    lat = np.asarray([-90.0, 0.0, 90.0])
    lon = np.asarray([0.0, 90.0, 180.0, 270.0])
    coords = {"time": time_coord, "level": np.asarray(PRESSURE_LEVELS), "latitude": lat, "longitude": lon}
    variables = {
        name: (("time", "level", "latitude", "longitude"), np.ones((3, 13, 3, 4), dtype=np.float32))
        for name in ATMOSPHERIC_VARIABLES
    }
    for name in SURFACE_VARIABLES:
        variables[name] = (("time", "latitude", "longitude"), np.ones((3, 3, 4), dtype=np.float32))
    for name in STATIC_VARIABLES:
        variables[name] = (("latitude", "longitude"), np.ones((3, 4), dtype=np.float32))
    if include_100m:
        for name in WN2_ONLY_VARIABLES:
            variables[name] = (("time", "latitude", "longitude"), np.ones((3, 3, 4), dtype=np.float32))
    state = xr.Dataset(variables, coords=coords)
    if relative_time:
        state = state.assign_coords(datetime=("time", times.values))
    return state


def test_input_preparer_merges_supplement_and_normalizes_relative_time(monkeypatch):
    monkeypatch.setattr(
        "typhoon_pressure.weathernext_input._target_grid",
        lambda resolution: (np.asarray([-90.0, 0.0, 90.0]), np.asarray([0.0, 90.0, 180.0, 270.0])),
    )
    primary = _source(relative_time=True)
    supplement = _source(include_100m=True)[list(WN2_ONLY_VARIABLES)]
    prepared = prepare_weathernext_input(
        primary,
        pd.Timestamp("2025-01-01 12:00"),
        supplements=(supplement,),
    )
    assert prepared.sizes["time"] == 2
    assert list(prepared.level.values) == list(PRESSURE_LEVELS)
    assert set(WN2_ONLY_VARIABLES).issubset(prepared.data_vars)
    assert np.issubdtype(prepared.time.dtype, np.datetime64)


def test_cyclone_profile_does_not_require_100m_wind(monkeypatch):
    monkeypatch.setattr(
        "typhoon_pressure.weathernext_input._target_grid",
        lambda resolution: (np.asarray([-90.0, 0.0, 90.0]), np.asarray([0.0, 90.0, 180.0, 270.0])),
    )
    prepared = prepare_weathernext_input(
        _source(),
        pd.Timestamp("2025-01-01 12:00"),
        config=WeatherNextInputConfig(model_variant="WeatherNextCyclones"),
    )
    assert not set(WN2_ONLY_VARIABLES).intersection(prepared.data_vars)


def test_weathernext2_missing_100m_wind_fails_explicitly(monkeypatch):
    monkeypatch.setattr(
        "typhoon_pressure.weathernext_input._target_grid",
        lambda resolution: (np.asarray([-90.0, 0.0, 90.0]), np.asarray([0.0, 90.0, 180.0, 270.0])),
    )
    with pytest.raises(ValueError, match="100m_u_component_of_wind"):
        prepare_weathernext_input(_source(), pd.Timestamp("2025-01-01 12:00"))
