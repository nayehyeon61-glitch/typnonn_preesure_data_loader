from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import xarray as xr

from .initial_condition import InitialConditionBuilder, StormObservation
from .weathernext_adapter import make_weathernext_request, run_weathernext
from .weathernext_resolver import (
    CheckpointDownloader,
    ResolvedWeatherNext,
    WeatherNextSelectionConfig,
    resolve_weathernext,
)
from .small_version.config import WeatherNextTokenConfig
from .small_version.weathernext_bridge import (
    WeatherNextForecastTokenizer,
    save_forecast_tokens,
)


@dataclass(frozen=True)
class WeatherNextPreparationResult:
    resolved: ResolvedWeatherNext
    forecast_path: Path
    token_path: Path


def prepare_weathernext_sample(
    atmospheric_state: xr.Dataset,
    storm: StormObservation,
    selection: WeatherNextSelectionConfig,
    *,
    forecast_dir: str | Path = "data/weathernext_forecasts",
    token_dir: str | Path = "data/weathernext_tokens",
    horizon_hours: int = 360,
    initialization_mode: str = "auto",
    token_config: WeatherNextTokenConfig = WeatherNextTokenConfig(),
    downloader: CheckpointDownloader | None = None,
    api_client=None,
) -> WeatherNextPreparationResult:
    """Resolve WeatherNext, run frozen forecast, persist forecast, then tokenize.

    This is the pipeline boundary before Weather-GPT/Fusion training. The
    resulting token directory can be passed directly to
    ``train-weathernext-transformer --weathernext-token-dir``.
    """

    condition = InitialConditionBuilder(
        mode=initialization_mode,
        history_steps=2,
    ).build(atmospheric_state, storm)
    request = make_weathernext_request(condition, horizon_hours=horizon_hours)

    resolved = resolve_weathernext(
        selection,
        downloader=downloader,
        api_client=api_client,
    )
    forecast = run_weathernext(resolved, request)
    forecast.attrs.update(
        {
            "storm_id": storm.storm_id,
            "initialization_time": str(storm.time),
            "initialization_mode": condition.applied_mode,
        }
    )

    forecast_root = Path(forecast_dir)
    forecast_root.mkdir(parents=True, exist_ok=True)
    init_time_ns = int(storm.time.value)
    forecast_path = forecast_root / f"{storm.storm_id}__{init_time_ns}.nc"
    forecast.to_netcdf(forecast_path)

    tokenizer = WeatherNextForecastTokenizer(token_config)
    tokens = tokenizer(forecast, storm.time)
    token_path = save_forecast_tokens(
        tokens,
        token_dir,
        storm_id=storm.storm_id,
        init_time=storm.time,
    )

    return WeatherNextPreparationResult(
        resolved=resolved,
        forecast_path=forecast_path,
        token_path=token_path,
    )
