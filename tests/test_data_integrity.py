import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from typhoon_pressure.dataset import TyphoonPressureDataset
from typhoon_pressure.small_version.config import WeatherNextTokenConfig
from typhoon_pressure.small_version.weathernext_bridge import (
    DirectoryForecastTokenStore,
    ForecastTokens,
    WeatherNextForecastTokenizer,
    save_forecast_tokens,
    tokenizer_fingerprint,
)
from typhoon_pressure.weathernext_input import (
    ATMOSPHERIC_VARIABLES,
    PRESSURE_LEVELS,
    STATIC_VARIABLES,
    SURFACE_VARIABLES,
    WN2_ONLY_VARIABLES,
    prepare_weathernext_input,
)


def _integrated(times):
    size = len(times)
    return pd.DataFrame({
        "storm_id": ["A"] * size,
        "time": times,
        "typhoon_lat": np.arange(size, dtype=float),
        "typhoon_lon": 130.0 + np.arange(size, dtype=float),
        "typhoon_pressure_hpa": np.linspace(1000, 980, size),
        "typhoon_wind_kt": np.linspace(20, 50, size),
        "high_rank": [np.nan] * size,
        "high_dx_km": [np.nan] * size,
        "high_dy_km": [np.nan] * size,
        "high_pressure_hpa": [np.nan] * size,
        "high_anomaly_hpa": [np.nan] * size,
    })


def test_dataset_rejects_windows_crossing_missing_six_hour_cycle():
    times = pd.DatetimeIndex([
        "2025-01-01 00:00", "2025-01-01 06:00", "2025-01-01 12:00",
        "2025-01-02 00:00", "2025-01-02 06:00",
    ])
    dataset = TyphoonPressureDataset(_integrated(times), history=2, horizon=2, max_highs=1)
    assert len(dataset) == 0


def _weather_source():
    times = pd.date_range("2025-01-01", periods=3, freq="6h")
    lat = np.asarray([-90.0, 0.0, 90.0])
    lon = np.asarray([0.0, 90.0, 180.0, 270.0])
    coords = {"time": times, "level": np.asarray(PRESSURE_LEVELS), "latitude": lat, "longitude": lon}
    variables = {
        name: (("time", "level", "latitude", "longitude"), np.ones((3, 13, 3, 4), dtype=np.float32))
        for name in ATMOSPHERIC_VARIABLES
    }
    for name in SURFACE_VARIABLES + WN2_ONLY_VARIABLES:
        variables[name] = (("time", "latitude", "longitude"), np.ones((3, 3, 4), dtype=np.float32))
    for name in STATIC_VARIABLES:
        variables[name] = (("latitude", "longitude"), np.ones((3, 4), dtype=np.float32))
    return xr.Dataset(variables, coords=coords)


def test_weather_input_rejects_low_finite_fraction(monkeypatch):
    monkeypatch.setattr(
        "typhoon_pressure.weathernext_input._target_grid",
        lambda resolution: (np.asarray([-90.0, 0.0, 90.0]), np.asarray([0.0, 90.0, 180.0, 270.0])),
    )
    source = _weather_source()
    source["temperature"].loc[{"time": source.time.values[-1], "level": 500}] = np.nan
    with pytest.raises(ValueError, match="temperature"):
        prepare_weathernext_input(source, pd.Timestamp("2025-01-01 12:00"))


def _forecast(include_v10=True):
    times = pd.date_range("2025-01-01", periods=3, freq="6h")
    lat = np.linspace(-60, 60, 3)
    lon = np.linspace(0, 270, 4)
    shape = (3, 3, 4)
    fields = {
        "msl": (("time", "latitude", "longitude"), np.ones(shape)),
        "u10": (("time", "latitude", "longitude"), np.ones(shape)),
        "t2m": (("time", "latitude", "longitude"), np.ones(shape)),
    }
    if include_v10:
        fields["v10"] = (("time", "latitude", "longitude"), np.ones(shape))
    return xr.Dataset(fields, coords={"time": times, "latitude": lat, "longitude": lon})


def test_tokenizer_requires_complete_canonical_feature_schema():
    with pytest.raises(ValueError, match="10m_v_component_of_wind"):
        WeatherNextForecastTokenizer()(_forecast(include_v10=False), pd.Timestamp("2025-01-01"))


def _tokens():
    return WeatherNextForecastTokenizer(
        WeatherNextTokenConfig(max_time_steps=3, target_lat_tokens=3, target_lon_tokens=4)
    )(_forecast(), pd.Timestamp("2025-01-01"))


def _provenance(checkpoint="abc"):
    config = WeatherNextTokenConfig(max_time_steps=3, target_lat_tokens=3, target_lon_tokens=4)
    return {
        "model_id": "WeatherNext2_test",
        "model_variant": "WeatherNext2",
        "release": "v0.3.0",
        "weight_origin": "finetuned",
        "checkpoint_fingerprint": checkpoint,
        "tokenizer_fingerprint": tokenizer_fingerprint(config),
        "feature_schema": json.dumps(list(config.variables), separators=(",", ":")),
        "initialization_mode": "auto",
    }


def test_provenance_mismatch_does_not_overwrite_existing_npz(tmp_path):
    init = pd.Timestamp("2025-01-01")
    path = save_forecast_tokens(_tokens(), tmp_path, storm_id="A", init_time=init, provenance=_provenance("first"))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="provenance mismatch"):
        save_forecast_tokens(_tokens(), tmp_path, storm_id="A", init_time=init, provenance=_provenance("second"))
    assert path.read_bytes() == original


def test_store_rejects_missing_and_corrupt_token_files(tmp_path):
    init = pd.Timestamp("2025-01-01")
    path = save_forecast_tokens(_tokens(), tmp_path, storm_id="A", init_time=init, provenance=_provenance())
    path.unlink()
    with pytest.raises(FileNotFoundError):
        DirectoryForecastTokenStore(tmp_path)

    path = save_forecast_tokens(_tokens(), tmp_path, storm_id="A", init_time=init, provenance=_provenance())
    path.write_bytes(b"not-an-npz")
    with pytest.raises(ValueError, match="Invalid WeatherNext token file"):
        DirectoryForecastTokenStore(tmp_path)


def test_store_rejects_duplicate_manifest_keys(tmp_path):
    init = pd.Timestamp("2025-01-01")
    save_forecast_tokens(_tokens(), tmp_path, storm_id="A", init_time=init, provenance=_provenance())
    manifest = pd.read_csv(tmp_path / "manifest.csv", keep_default_na=False)
    pd.concat([manifest, manifest], ignore_index=True).to_csv(tmp_path / "manifest.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        DirectoryForecastTokenStore(tmp_path, validate_files=False)
