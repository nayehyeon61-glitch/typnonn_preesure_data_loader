import numpy as np
import pandas as pd
import pytest
import xarray as xr

from typhoon_pressure.weathernext_adapter import WeatherNextRequest, run_weathernext
from typhoon_pressure.weathernext_backends import (
    WeatherNextBackendConfig,
    build_weathernext_runner,
)


def _state():
    return xr.Dataset(
        {"msl": (("time", "latitude", "longitude"), np.ones((1, 2, 2)))},
        coords={
            "time": [pd.Timestamp("2025-01-01")],
            "latitude": [0, 1],
            "longitude": [130, 131],
        },
    )


def _request():
    return WeatherNextRequest(_state(), {"time": pd.Timestamp("2025-01-01")}, 24, {})


class FakeModel:
    def __init__(self):
        self.fit_called = False

    def fit(self, training_data, **kwargs):
        self.fit_called = training_data == "train" and kwargs.get("epochs") == 2

    def rollout(self, initial_state, horizon_hours):
        return initial_state.copy()


class FakeAPI:
    def __init__(self):
        self.model_id = None

    def forecast(self, initial_state, horizon_hours, *, model_id):
        self.model_id = model_id
        return initial_state.copy()


def test_trainable_backend_requires_explicit_fit():
    model = FakeModel()
    runner = build_weathernext_runner(
        WeatherNextBackendConfig("trainable", training_kwargs={"epochs": 2}),
        trainable_model=model,
        training_data="train",
    )
    with pytest.raises(RuntimeError):
        run_weathernext(runner, _request())
    runner.fit()
    forecast = run_weathernext(runner, _request())
    assert model.fit_called
    assert forecast.attrs["weathernext_backend"] == "trainable"


def test_pretrained_backend_requires_checkpoint_and_records_release():
    with pytest.raises(ValueError):
        build_weathernext_runner(
            WeatherNextBackendConfig("pretrained"), pretrained_model=FakeModel()
        )
    runner = build_weathernext_runner(
        WeatherNextBackendConfig(
            "pretrained", release="v0.3.0", checkpoint="WeatherNext2_<2025_model1.npz"
        ),
        pretrained_model=FakeModel(),
    )
    forecast = run_weathernext(runner, _request())
    assert forecast.attrs["weathernext_release"] == "v0.3.0"
    assert forecast.attrs["weathernext_checkpoint"] == "WeatherNext2_<2025_model1.npz"


def test_pretrained_backend_can_build_read_only_official_runner(monkeypatch):
    created = {}

    class FakeOfficialModel(FakeModel):
        def __init__(self, **kwargs):
            super().__init__()
            created.update(kwargs)

    monkeypatch.setattr(
        "typhoon_pressure.weathernext_official.OfficialWeatherNextRunner",
        FakeOfficialModel,
    )
    runner = build_weathernext_runner(
        WeatherNextBackendConfig(
            "pretrained",
            model_id="regional-wn2",
            model_variant="WeatherNext2",
            checkpoint="/weights/weather-me-fine_tune_weight.npz",
        )
    )
    forecast = run_weathernext(runner, _request())
    assert created["model_name"] == "WeatherNext2"
    assert created["checkpoint_path"] == "/weights/weather-me-fine_tune_weight.npz"
    assert forecast.attrs["weathernext_backend"] == "pretrained"


def test_api_backend_calls_injected_client_and_records_provider():
    client = FakeAPI()
    runner = build_weathernext_runner(
        WeatherNextBackendConfig("api", api_provider="vertex-ai"), api_client=client
    )
    forecast = run_weathernext(runner, _request())
    assert client.model_id == "WeatherNext2_<2025"
    assert forecast.attrs["weathernext_backend"] == "api"
    assert forecast.attrs["weathernext_api_provider"] == "vertex-ai"
