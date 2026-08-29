from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from typhoon_pressure.weathernext_adapter import WeatherNextRequest, run_weathernext
from typhoon_pressure.weathernext_resolver import (
    CheckpointOrigin,
    WeatherNextSelectionConfig,
    resolve_weathernext,
)


class FakeRunner:
    def __init__(self, source=None):
        self.source = source

    def rollout(self, initial_state, horizon_hours):
        return initial_state.copy()

    def provenance(self):
        return {"weathernext_backend": "pretrained"}


class FakeAPI:
    def forecast(self, initial_state, horizon_hours, *, model_id):
        return initial_state.copy()


class FakeDownloader:
    def __init__(self, path: Path):
        self.path = path
        self.called = False

    def download(self, *, model_variant: str, release: str) -> Path:
        self.called = True
        return self.path


def _request():
    state = xr.Dataset(
        {"msl": (("time", "latitude", "longitude"), np.ones((1, 2, 2)))},
        coords={
            "time": [pd.Timestamp("2025-01-01")],
            "latitude": [0, 1],
            "longitude": [130, 131],
        },
    )
    return WeatherNextRequest(state, {}, 24, {})


def test_finetuned_checkpoint_has_highest_priority(tmp_path, monkeypatch):
    fine = tmp_path / "fine.npz"
    pre = tmp_path / "pre.npz"
    fine.write_bytes(b"fine")
    pre.write_bytes(b"pre")

    created = []
    monkeypatch.setattr(
        "typhoon_pressure.weathernext_resolver._checkpoint_runner",
        lambda config, path: created.append(path) or FakeRunner(path),
    )

    resolved = resolve_weathernext(
        WeatherNextSelectionConfig(
            finetuned_checkpoint=str(fine),
            pretrained_checkpoint=str(pre),
            allow_download=False,
            allow_api_fallback=False,
        )
    )

    assert resolved.origin is CheckpointOrigin.FINETUNED
    assert created == [fine.resolve()]


def test_pretrained_is_used_when_finetuned_is_missing(tmp_path, monkeypatch):
    pre = tmp_path / "pre.npz"
    pre.write_bytes(b"pre")
    monkeypatch.setattr(
        "typhoon_pressure.weathernext_resolver._checkpoint_runner",
        lambda config, path: FakeRunner(path),
    )

    resolved = resolve_weathernext(
        WeatherNextSelectionConfig(
            finetuned_checkpoint=str(tmp_path / "missing.npz"),
            pretrained_checkpoint=str(pre),
            allow_download=False,
            allow_api_fallback=False,
        )
    )

    assert resolved.origin is CheckpointOrigin.OFFICIAL
    assert resolved.checkpoint == str(pre.resolve())


def test_downloader_is_used_after_local_checkpoints(tmp_path, monkeypatch):
    downloaded = tmp_path / "downloaded.npz"
    downloaded.write_bytes(b"downloaded")
    downloader = FakeDownloader(downloaded)
    monkeypatch.setattr(
        "typhoon_pressure.weathernext_resolver._checkpoint_runner",
        lambda config, path: FakeRunner(path),
    )

    resolved = resolve_weathernext(
        WeatherNextSelectionConfig(
            allow_download=True,
            allow_api_fallback=False,
        ),
        downloader=downloader,
    )

    assert downloader.called
    assert resolved.origin is CheckpointOrigin.DOWNLOADED


def test_api_is_last_fallback(monkeypatch):
    class FakeAPIRunner(FakeRunner):
        def provenance(self):
            return {
                "weathernext_backend": "api",
                "weathernext_api_provider": "test-provider",
            }

    monkeypatch.setattr(
        "typhoon_pressure.weathernext_resolver.build_weathernext_runner",
        lambda config, api_client=None: FakeAPIRunner(),
    )

    resolved = resolve_weathernext(
        WeatherNextSelectionConfig(
            allow_download=False,
            allow_api_fallback=True,
            api_provider="test-provider",
        ),
        api_client=FakeAPI(),
    )

    assert resolved.origin is CheckpointOrigin.API
    forecast = run_weathernext(resolved, _request())
    assert forecast.attrs["weathernext_weight_origin"] == "api"
    assert forecast.attrs["weathernext_backend"] == "api"


def test_no_source_raises():
    with pytest.raises(RuntimeError):
        resolve_weathernext(
            WeatherNextSelectionConfig(
                allow_download=False,
                allow_api_fallback=False,
            )
        )
