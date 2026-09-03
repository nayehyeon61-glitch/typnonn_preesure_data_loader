import numpy as np
import pandas as pd
import torch

from typhoon_pressure.small_version import SmallModelConfig, WeatherNextDualTargetDataset
from typhoon_pressure.small_version.weathernext_bridge import ForecastTokens


class _Base:
    def __init__(self):
        self.time = pd.Timestamp("2025-08-01T00:00:00")
        self.groups = {
            "TEST": pd.DataFrame({
                "time": pd.date_range(self.time, periods=130, freq="6h"),
                "typhoon_lat": np.linspace(20.0, 30.0, 130),
                "typhoon_lon": np.linspace(130.0, 145.0, 130),
            })
        }

    def __len__(self):
        return 1

    def storm_id_at(self, index):
        return "TEST"

    def __getitem__(self, index):
        return {
            "history": torch.zeros(2, 4),
            "history_mask": torch.ones(2, 4),
            "target": torch.tensor([[21.0, 131.0], [22.0, 132.0]]),
            "target_mask": torch.ones(2, 2),
            "storm_id": "TEST",
            "init_time_ns": int(self.time.value),
        }


class _Store:
    def __init__(self, lead_hours):
        self.tokens = ForecastTokens(
            values=np.zeros((1, 4), dtype=np.float32),
            feature_mask=np.ones((1, 4), dtype=np.float32),
            token_mask=np.ones(1, dtype=np.float32),
            positions=np.zeros((1, 6), dtype=np.float32),
            feature_names=(
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ),
            endpoint_latlon=np.asarray([25.0, 140.0], dtype=np.float32),
            endpoint_mask=True,
            endpoint_lead_hours=float(lead_hours),
        )

    def contains(self, storm_id, init_time_ns):
        return True

    def load(self, storm_id, init_time_ns):
        return self.tokens


def _dataset(endpoint_lead_hours):
    config = SmallModelConfig(
        input_dim=4,
        history_steps=2,
        local_track_steps=2,
        distribution_start_day=15,
        distribution_end_day=16,
    )
    return WeatherNextDualTargetDataset(
        _Base(),
        None,
        config,
        _Store(endpoint_lead_hours),
        max_forecast_tokens=2,
        forecast_input_dim=4,
        require_endpoint_lead_hours=360.0,
    )


def test_day15_endpoint_accepts_exact_360h():
    dataset = _dataset(360.0)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["weathernext_endpoint_mask"].item() == 1.0
    assert sample["weathernext_endpoint_lead_hours"].item() == 360.0


def test_monthly_flow_720h_endpoint_is_not_used_as_day15_anchor():
    dataset = _dataset(720.0)
    assert len(dataset) == 0


def test_nearby_but_wrong_endpoint_is_rejected():
    dataset = _dataset(366.0)
    assert len(dataset) == 0
