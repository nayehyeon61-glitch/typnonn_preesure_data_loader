import numpy as np
import pandas as pd
import xarray as xr

from typhoon_pressure.prepare_weathernext_tokens import (
    build_token_cache,
    load_token_jobs,
    validate_token_cache_coverage,
)
from typhoon_pressure.small_version.config import WeatherNextTokenConfig
from typhoon_pressure.small_version.weathernext_bridge import WeatherNextForecastTokenizer


class FakeFlowRunner:
    def __init__(self, checkpoint_sha="test-sha"):
        self.calls = 0
        self.checkpoint_sha = checkpoint_sha

    def provenance(self):
        return {
            "forecast_backend": "flow_matching",
            "forecast_checkpoint_kind": "flow_matching",
            "forecast_checkpoint_sha256": self.checkpoint_sha,
        }

    def rollout(self, initial_state, horizon_hours):
        self.calls += 1
        init = pd.Timestamp(initial_state.time.values[-1])
        values = np.ones((1, 2, 3), dtype=np.float32) * self.calls
        return xr.Dataset(
            {"msl": (("time", "lat", "lon"), values)},
            coords={
                "time": [init + pd.Timedelta(hours=horizon_hours)],
                "lat": [-30.0, 30.0],
                "lon": [0.0, 120.0, 240.0],
            },
            attrs={
                "forecast_backend": "flow_matching",
                "forecast_checkpoint_kind": "flow_matching",
                "forecast_checkpoint_sha256": self.checkpoint_sha,
            },
        )


def test_integrated_jobs_are_unique_and_batch_cache_resumes(tmp_path):
    table = pd.DataFrame(
        {
            "storm_id": ["A", "A", "B"],
            "time": ["2020-03-01", "2020-03-01", "2020-04-01"],
            "typhoon_lat": [10.0, 10.0, 20.0],
            "typhoon_lon": [130.0, 130.0, 140.0],
            "typhoon_pressure_hpa": [990.0, 990.0, 980.0],
        }
    )
    path = tmp_path / "jobs.csv"
    table.to_csv(path, index=False)
    jobs = load_token_jobs(path)
    assert len(jobs) == 2

    times = pd.date_range("2019-01-01", periods=18, freq="MS")
    state = xr.Dataset(
        {"msl": (("time", "lat", "lon"), np.ones((18, 2, 3), dtype=np.float32))},
        coords={"time": times, "lat": [-30.0, 30.0], "lon": [0.0, 120.0, 240.0]},
    )
    tokenizer = WeatherNextForecastTokenizer(
        WeatherNextTokenConfig(
            variables=("mean_sea_level_pressure",),
            max_lead_hours=720,
            max_time_steps=1,
            target_lat_tokens=2,
            target_lon_tokens=3,
        )
    )
    runner = FakeFlowRunner()
    first = build_token_cache(
        runner,
        state,
        jobs,
        tokenizer,
        tmp_path / "tokens",
        backend="flow_matching",
        horizon_hours=720,
    )
    assert first == {
        "requested": 2, "available": 2, "missing": 0, "generated": 2, "skipped": 0
    }
    assert runner.calls == 2

    second = build_token_cache(
        runner,
        state,
        jobs,
        tokenizer,
        tmp_path / "tokens",
        backend="flow_matching",
        horizon_hours=720,
    )
    assert second["generated"] == 0
    assert second["skipped"] == 2
    assert runner.calls == 2
    assert validate_token_cache_coverage(jobs, tmp_path / "tokens")["missing"] == 0

    incompatible = FakeFlowRunner("different-sha")
    try:
        build_token_cache(
            incompatible,
            state,
            jobs,
            tokenizer,
            tmp_path / "tokens",
            backend="flow_matching",
            horizon_hours=720,
        )
    except ValueError as error:
        assert "different forecast_checkpoint_sha256" in str(error)
    else:
        raise AssertionError("resume must reject a mixed-checkpoint token cache")
