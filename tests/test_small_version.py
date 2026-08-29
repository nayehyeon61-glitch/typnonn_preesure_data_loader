import numpy as np
import pandas as pd
import torch

from typhoon_pressure.dataset import TyphoonPressureDataset
from typhoon_pressure.small_version import (
    DualLossConfig,
    DualObjectiveLoss,
    DualTargetDataset,
    SmallDualScaleModel,
    SmallModelConfig,
    SpatialDistributionLookup,
)


def _integrated_frame():
    times = pd.date_range("2025-01-01", periods=10, freq="6h")
    return pd.DataFrame({
        "storm_id": ["TEST"] * len(times),
        "time": times,
        "typhoon_lat": np.linspace(20, 29, len(times)),
        "typhoon_lon": np.linspace(125, 134, len(times)),
        "typhoon_pressure_hpa": np.linspace(995, 975, len(times)),
        "typhoon_wind_kt": np.linspace(35, 65, len(times)),
        "high_rank": [1] * len(times),
        "high_dx_km": np.linspace(500, 700, len(times)),
        "high_dy_km": np.linspace(800, 600, len(times)),
        "high_pressure_hpa": [1020] * len(times),
        "high_anomaly_hpa": [5] * len(times),
    })


def test_dual_targets_model_and_loss():
    base = TyphoonPressureDataset(_integrated_frame(), history=3, horizon=4, max_highs=1)
    distribution_frame = pd.DataFrame({
        "calendar_month": [1, 1], "lat_bin": [1, 1], "lon_bin": [0, 1],
        "probability": [0.25, 0.75],
    })
    lookup = SpatialDistributionLookup.from_frame(
        distribution_frame, lat_bin_deg=90, lon_bin_deg=180
    )
    config = SmallModelConfig(
        input_dim=len(base.feature_cols), history_steps=3, hidden_dim=16,
        distribution_start_day=15, distribution_end_day=16,
        local_track_steps=4, lat_bin_deg=90, lon_bin_deg=180,
    )
    dataset = DualTargetDataset(base, lookup, config)
    sample = dataset[0]
    batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
        for key, value in sample.items()
    }
    model = SmallDualScaleModel(config)
    outputs = model(batch["history"], batch["history_mask"])
    losses = DualObjectiveLoss(DualLossConfig())(outputs, batch)
    assert outputs["distribution_logits"].shape == (1, 2, 4)
    assert outputs["track_latlon"].shape == (1, 4, 2)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

