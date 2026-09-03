import numpy as np
import pandas as pd
import xarray as xr

from typhoon_pressure.forecast_provenance import (
    backend_provenance,
    cache_forecast_provenance,
    run_and_save_forecast_tokens,
)
from typhoon_pressure.small_version.config import WeatherNextTokenConfig
from typhoon_pressure.small_version.weathernext_bridge import (
    DirectoryForecastTokenStore,
    WeatherNextForecastTokenizer,
)
from typhoon_pressure.weathernext_adapter import WeatherNextRequest


class FakeFlowRunner:
    def provenance(self):
        return {
            "forecast_backend": "flow_matching",
            "forecast_checkpoint": "/tmp/flow360.pt",
            "forecast_checkpoint_kind": "flow_matching",
            "forecast_checkpoint_sha256": "abc123",
            "forecast_checkpoint_format": "climate_diffusion.latent_flow.v3",
            "forecast_release": "climate-diffusion",
            "forecast_step_hours": 360,
            "forecast_schema_format": "climate_diffusion.fixed_step_state.v1",
            "parameters_frozen": True,
        }

    def rollout(self, initial_state, horizon_hours):
        assert horizon_hours == 360
        init = pd.Timestamp(initial_state.time.values[-1])
        times = [init + pd.Timedelta(hours=360)]
        lat = np.asarray([10.0, 20.0, 30.0])
        lon = np.asarray([120.0, 130.0, 140.0])
        shape = (1, 3, 3)
        pressure = np.full(shape, 1010.0, dtype=np.float32)
        pressure[0, 1, 1] = 970.0
        return xr.Dataset(
            {
                "msl": (("time", "latitude", "longitude"), pressure),
                "u10": (("time", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
                "v10": (("time", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
                "t2m": (("time", "latitude", "longitude"), np.ones(shape, dtype=np.float32) * 290),
            },
            coords={"time": times, "latitude": lat, "longitude": lon},
        )


class FakeWeatherNextRunner:
    def provenance(self):
        return {
            "weathernext_backend": "pretrained",
            "weathernext_model_id": "WeatherNext2_<2025_model1",
            "weathernext_model_variant": "WeatherNext2",
            "weathernext_release": "v0.3.0",
            "weathernext_checkpoint": "/not/a/local/file.npz",
        }


def test_flow_token_manifest_preserves_generic_provenance_and_360h_endpoint(tmp_path):
    init = pd.Timestamp("2025-01-01")
    initial = xr.Dataset(
        {"dummy": (("time",), [1.0])},
        coords={"time": [init]},
    )
    request = WeatherNextRequest(
        initial_state=initial,
        tracker_seed={
            "storm_id": "FLOW",
            "time": init,
            "lat": 20.0,
            "lon": 130.0,
            "pressure_hpa": 980.0,
            "wind_kt": 60.0,
        },
        horizon_hours=360,
        initialization_metadata={"requested_mode": "flow_fixed_step_history"},
    )
    tokenizer = WeatherNextForecastTokenizer(
        WeatherNextTokenConfig(
            max_lead_hours=360,
            max_time_steps=1,
            target_lat_tokens=3,
            target_lon_tokens=3,
        )
    )
    run_and_save_forecast_tokens(FakeFlowRunner(), request, tokenizer, tmp_path)

    store = DirectoryForecastTokenStore(tmp_path)
    tokens = store.load("FLOW", int(init.value))
    assert tokens.endpoint_mask
    assert tokens.endpoint_lead_hours == 360.0
    provenance = cache_forecast_provenance(store)
    assert provenance["forecast_backend"] == "flow_matching"
    assert provenance["forecast_checkpoint_sha256"] == "abc123"
    assert provenance["forecast_checkpoint_format"] == "climate_diffusion.latent_flow.v3"
    assert provenance["forecast_step_hours"] == "360"
    assert provenance["forecast_horizon_hours"] == "360"
    assert provenance["forecast_schema_format"] == "climate_diffusion.fixed_step_state.v1"


def test_weathernext_metadata_maps_into_same_generic_schema():
    forecast = xr.Dataset(
        {"msl": (("time", "lat", "lon"), np.ones((1, 1, 1), dtype=np.float32))},
        coords={"time": [pd.Timestamp("2025-01-16")], "lat": [20.0], "lon": [130.0]},
        attrs={
            "weathernext_checkpoint_kind": "official_pretrained",
            "weathernext_checkpoint_format": "weathernext.weathernext2.fgn.CheckPoint",
            "weathernext_checkpoint_sha256": "wn-sha",
        },
    )
    provenance = backend_provenance(FakeWeatherNextRunner(), forecast)
    assert provenance["forecast_backend"] == "pretrained"
    assert provenance["forecast_checkpoint"] == "/not/a/local/file.npz"
    assert provenance["forecast_checkpoint_kind"] == "official_pretrained"
    assert provenance["forecast_checkpoint_sha256"] == "wn-sha"
    assert provenance["forecast_checkpoint_format"] == "weathernext.weathernext2.fgn.CheckPoint"
    assert provenance["forecast_release"] == "v0.3.0"
    assert provenance["forecast_step_hours"] == "6"
