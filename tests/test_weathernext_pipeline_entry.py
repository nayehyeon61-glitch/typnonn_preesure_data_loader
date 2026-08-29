import numpy as np
import pandas as pd
import xarray as xr

from typhoon_pressure.initial_condition import StormObservation
from typhoon_pressure.weathernext_pipeline import prepare_weathernext_sample
from typhoon_pressure.weathernext_resolver import (
    CheckpointOrigin,
    ResolvedWeatherNext,
    WeatherNextSelectionConfig,
)
from typhoon_pressure.small_version.weathernext_bridge import DirectoryForecastTokenStore


class FakeResolvedRunner:
    def rollout(self, initial_state, horizon_hours):
        init_time = pd.Timestamp(initial_state.time.values[-1])
        times = pd.date_range(init_time + pd.Timedelta("6h"), periods=4, freq="6h")
        lat = np.linspace(-60, 60, 6)
        lon = np.linspace(0, 330, 12)
        shape = (len(times), len(lat), len(lon))
        fields = {
            "msl": (("time", "latitude", "longitude"), np.ones(shape) * 100000.0),
            "u10": (("time", "latitude", "longitude"), np.ones(shape)),
            "v10": (("time", "latitude", "longitude"), np.ones(shape) * 2),
            "t2m": (("time", "latitude", "longitude"), np.ones(shape) * 290),
        }
        return xr.Dataset(fields, coords={"time": times, "latitude": lat, "longitude": lon})

    def provenance(self):
        return {"weathernext_backend": "pretrained"}


def test_resolver_rollout_forecast_and_token_cache_are_connected(tmp_path, monkeypatch):
    times = pd.date_range("2025-01-01", periods=2, freq="6h")
    lat = np.linspace(-90, 90, 7)
    lon = np.linspace(0, 300, 6)
    state = xr.Dataset(
        {"msl": (("time", "latitude", "longitude"), np.ones((2, 7, 6)) * 100000.0)},
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    storm = StormObservation(
        storm_id="TEST",
        time=times[-1],
        lat=20.0,
        lon=130.0,
        pressure_hpa=990.0,
        wind_kt=50.0,
    )
    resolved = ResolvedWeatherNext(
        runner=FakeResolvedRunner(),
        origin=CheckpointOrigin.FINETUNED,
        checkpoint="/fake/fine.npz",
    )
    monkeypatch.setattr(
        "typhoon_pressure.weathernext_pipeline.resolve_weathernext",
        lambda *args, **kwargs: resolved,
    )

    forecast_dir = tmp_path / "forecasts"
    token_dir = tmp_path / "tokens"
    result = prepare_weathernext_sample(
        state,
        storm,
        WeatherNextSelectionConfig(allow_download=False, allow_api_fallback=False),
        forecast_dir=forecast_dir,
        token_dir=token_dir,
        horizon_hours=24,
        initialization_mode="tracker_seed",
    )

    assert result.resolved.origin is CheckpointOrigin.FINETUNED
    assert result.forecast_path.is_file()
    assert result.token_path.is_file()
    stored = xr.open_dataset(result.forecast_path)
    assert stored.attrs["weathernext_weight_origin"] == "finetuned"
    token_store = DirectoryForecastTokenStore(token_dir)
    assert token_store.contains("TEST", int(times[-1].value))
