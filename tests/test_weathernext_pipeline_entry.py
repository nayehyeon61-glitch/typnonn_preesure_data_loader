import numpy as np
import pandas as pd
import xarray as xr

from typhoon_pressure.initial_condition import StormObservation
from typhoon_pressure.weathernext_pipeline import prepare_weathernext_batch, prepare_weathernext_sample
from typhoon_pressure.weathernext_resolver import (
    CheckpointOrigin,
    ResolvedWeatherNext,
    WeatherNextSelectionConfig,
)
from typhoon_pressure.small_version.weathernext_bridge import DirectoryForecastTokenStore
from typhoon_pressure.prepare_weathernext_pipeline import _storms_from_frame


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


def test_batch_resolves_once_and_generates_every_initialization(tmp_path, monkeypatch):
    times = pd.date_range("2025-01-01", periods=3, freq="6h")
    state = xr.Dataset(
        {"msl": (("time", "latitude", "longitude"), np.ones((3, 7, 6)) * 100000.0)},
        coords={"time": times, "latitude": np.linspace(-90, 90, 7), "longitude": np.linspace(0, 300, 6)},
    )
    storms = [
        StormObservation("A", times[1], 20.0, 130.0, 990.0, 50.0),
        StormObservation("A", times[2], 21.0, 131.0, 985.0, 55.0),
    ]
    resolved = ResolvedWeatherNext(FakeResolvedRunner(), CheckpointOrigin.OFFICIAL, "/fake/pre.npz")
    calls = []
    monkeypatch.setattr(
        "typhoon_pressure.weathernext_pipeline.resolve_weathernext",
        lambda *args, **kwargs: calls.append(1) or resolved,
    )
    result = prepare_weathernext_batch(
        state,
        storms,
        WeatherNextSelectionConfig(allow_download=False, allow_api_fallback=False),
        forecast_dir=tmp_path / "forecasts",
        token_dir=tmp_path / "tokens",
        horizon_hours=24,
        initialization_mode="tracker_seed",
    )
    assert len(calls) == 1
    assert len(result.completed) == 2
    store = DirectoryForecastTokenStore(tmp_path / "tokens")
    assert all(store.contains(storm.storm_id, int(storm.time.value)) for storm in storms)

    resumed = prepare_weathernext_batch(
        state,
        storms,
        WeatherNextSelectionConfig(allow_download=False, allow_api_fallback=False),
        forecast_dir=tmp_path / "forecasts",
        token_dir=tmp_path / "tokens",
        horizon_hours=24,
        initialization_mode="tracker_seed",
    )
    assert len(resumed.skipped) == 2


def test_integrated_jobs_match_training_window_initializations():
    times = pd.date_range("2025-01-01", periods=5, freq="6h")
    integrated = pd.DataFrame({
        "storm_id": ["A"] * 5,
        "time": times,
        "typhoon_lat": np.arange(5.0),
        "typhoon_lon": 130.0 + np.arange(5.0),
        "typhoon_pressure_hpa": [1000, 995, 990, 985, 980],
        "typhoon_wind_kt": [20, 25, 30, 35, 40],
    })
    storms = _storms_from_frame(integrated, history=2, horizon=2, max_highs=1)
    assert [storm.time for storm in storms] == [times[1], times[2]]
