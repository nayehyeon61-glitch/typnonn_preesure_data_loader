import numpy as np
import pandas as pd
import xarray as xr

from typhoon_pressure.forecast_provenance import (
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
