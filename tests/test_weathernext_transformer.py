import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from typhoon_pressure.dataset import TyphoonPressureDataset
from typhoon_pressure.small_version import (
    DirectoryForecastTokenStore,
    DualObjectiveLoss,
    SmallModelConfig,
    SpatialDistributionLookup,
    TransformerConfig,
    WeatherNextDualTargetDataset,
    WeatherNextForecastTokenizer,
    WeatherNextFusionTransformer,
    WeatherNextTokenConfig,
    save_forecast_tokens,
)


def test_weathernext_output_reaches_masked_transformer(tmp_path):
    track_times = pd.date_range("2025-01-01", periods=8, freq="6h")
    integrated = pd.DataFrame({
        "storm_id": ["TEST"] * 8, "time": track_times,
        "typhoon_lat": np.linspace(20, 27, 8), "typhoon_lon": np.linspace(125, 132, 8),
        "typhoon_pressure_hpa": np.linspace(995, 980, 8),
        "typhoon_wind_kt": np.linspace(35, 60, 8),
        "high_rank": [1] * 8, "high_dx_km": [500] * 8, "high_dy_km": [700] * 8,
        "high_pressure_hpa": [1020] * 8, "high_anomaly_hpa": [5] * 8,
    })
    base = TyphoonPressureDataset(integrated, history=3, horizon=3, max_highs=1)
    init_time = track_times[2]
    forecast_times = pd.date_range(init_time, periods=4, freq="6h")
    lat, lon = np.linspace(-60, 60, 6), np.linspace(0, 330, 12)
    shape = (len(forecast_times), len(lat), len(lon))
    rng = np.random.default_rng(7)
    fields = {name: (("time", "latitude", "longitude"), rng.normal(size=shape)) for name in (
        "msl", "u10", "v10", "t2m"
    )}
    fields["msl"][1][0, 0, 0] = np.nan
    forecast = xr.Dataset(fields, coords={"time": forecast_times, "latitude": lat, "longitude": lon})
    token_config = WeatherNextTokenConfig(max_time_steps=4, target_lat_tokens=3, target_lon_tokens=4)
    tokens = WeatherNextForecastTokenizer(token_config)(forecast, init_time)
    assert tokens.values.shape == (4 * 3 * 4, 4)
    assert tokens.feature_mask.shape == tokens.values.shape
    save_forecast_tokens(
        tokens,
        tmp_path,
        storm_id="TEST",
        init_time=init_time,
        provenance={
            "weathernext_backend": "pretrained",
            "weathernext_model_variant": "WeatherNext2",
            "weathernext_checkpoint_kind": "official_pretrained",
        },
    )
    store = DirectoryForecastTokenStore(tmp_path)
    assert store.provenance()["weathernext_checkpoint_kind"] == "official_pretrained"
    store.require_checkpoint_kind("official_pretrained")
    with pytest.raises(ValueError, match="manifest records official_pretrained"):
        store.require_checkpoint_kind("fine_tuned")

    distribution = SpatialDistributionLookup.from_frame(pd.DataFrame({
        "calendar_month": [1, 1], "lat_bin": [1, 1], "lon_bin": [0, 1],
        "probability": [0.4, 0.6],
    }), lat_bin_deg=90, lon_bin_deg=180)
    model_config = SmallModelConfig(
        input_dim=len(base.feature_cols), history_steps=3, hidden_dim=16,
        distribution_start_day=15, distribution_end_day=16,
        local_track_steps=3, lat_bin_deg=90, lon_bin_deg=180,
    )
    dataset = WeatherNextDualTargetDataset(
        base, distribution, model_config, store,
        max_forecast_tokens=48, forecast_input_dim=4,
    )
    sample = dataset[0]
    batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
        for key, value in sample.items()
    }
    model = WeatherNextFusionTransformer(model_config, TransformerConfig(
        forecast_input_dim=4, gpt_state_dim=10, model_dim=16, num_heads=4, num_layers=1,
        feedforward_dim=32, input_mask_probability=0.5,
    ))
    gpt_values = torch.zeros((1, 10), dtype=torch.float32)
    gpt_mask = torch.zeros_like(gpt_values)
    outputs = model(
        batch["history"], batch["history_mask"], batch["forecast_values"],
        batch["forecast_feature_mask"], batch["forecast_token_mask"],
        batch["forecast_positions"], gpt_values, gpt_mask,
    )
    losses = DualObjectiveLoss()(outputs, batch)
    assert outputs["distribution_logits"].shape == (1, 2, 4)
    assert outputs["gpt_history_conditioning_fraction"].item() == 0.0
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()

    # Even a fully missing WeatherNext input remains numerically valid through CLS.
    model.eval()
    unconditioned_outputs = model(
        batch["history"], batch["history_mask"], batch["forecast_values"],
        batch["forecast_feature_mask"], batch["forecast_token_mask"],
        batch["forecast_positions"], gpt_values, gpt_mask,
    )
    masked_outputs = model(
        batch["history"], batch["history_mask"], batch["forecast_values"],
        torch.zeros_like(batch["forecast_feature_mask"]),
        torch.zeros_like(batch["forecast_token_mask"]),
        batch["forecast_positions"], gpt_values, gpt_mask,
    )
    assert torch.isfinite(masked_outputs["distribution_logits"]).all()
    conditioned_outputs = model(
        batch["history"], batch["history_mask"], batch["forecast_values"],
        batch["forecast_feature_mask"], batch["forecast_token_mask"],
        batch["forecast_positions"], torch.ones_like(gpt_values), torch.ones_like(gpt_mask),
    )
    assert conditioned_outputs["gpt_history_conditioning_fraction"].item() == 1.0
    assert not torch.allclose(
        conditioned_outputs["distribution_logits"], unconditioned_outputs["distribution_logits"]
    )
