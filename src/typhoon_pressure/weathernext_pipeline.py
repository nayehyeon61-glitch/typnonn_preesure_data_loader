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
from .weathernext_input import WeatherNextInputConfig, prepare_weathernext_input
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


@dataclass(frozen=True)
class WeatherNextBatchResult:
    completed: tuple[WeatherNextPreparationResult, ...]
    skipped: tuple[tuple[str, int], ...]
    failed: tuple[tuple[str, int, str], ...]


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
    resolved: ResolvedWeatherNext | None = None,
    input_config: WeatherNextInputConfig | None = None,
    supplemental_states: tuple[xr.Dataset, ...] = (),
    trainable_model=None,
    training_data=None,
) -> WeatherNextPreparationResult:
    """Resolve WeatherNext, run frozen forecast, persist forecast, then tokenize.

    This is the pipeline boundary before Weather-GPT/Fusion training. The
    resulting token directory can be passed directly to
    ``train-weathernext-transformer --weathernext-token-dir``.
    """

    if input_config is not None:
        atmospheric_state = prepare_weathernext_input(
            atmospheric_state,
            storm.time,
            supplements=supplemental_states,
            config=input_config,
        )

    condition = InitialConditionBuilder(
        mode=initialization_mode,
        history_steps=2,
    ).build(atmospheric_state, storm)
    request = make_weathernext_request(condition, horizon_hours=horizon_hours)

    if resolved is None:
        resolved = resolve_weathernext(
            selection,
            downloader=downloader,
            api_client=api_client,
            trainable_model=trainable_model,
            training_data=training_data,
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


def prepare_weathernext_batch(
    atmospheric_state: xr.Dataset,
    storms: list[StormObservation] | tuple[StormObservation, ...],
    selection: WeatherNextSelectionConfig,
    *,
    forecast_dir: str | Path = "data/weathernext_forecasts",
    token_dir: str | Path = "data/weathernext_tokens",
    horizon_hours: int = 360,
    initialization_mode: str = "auto",
    token_config: WeatherNextTokenConfig = WeatherNextTokenConfig(),
    downloader: CheckpointDownloader | None = None,
    api_client=None,
    input_config: WeatherNextInputConfig | None = None,
    supplemental_states: tuple[xr.Dataset, ...] = (),
    trainable_model=None,
    training_data=None,
    resume: bool = True,
    on_error: str = "raise",
) -> WeatherNextBatchResult:
    """Generate tokens for every requested ``(storm_id, init_time)`` key.

    The WeatherNext source is resolved (and an opt-in trainable backend fitted)
    exactly once, then reused across all initialization times.
    """
    if on_error not in {"raise", "continue"}:
        raise ValueError("on_error must be 'raise' or 'continue'")
    resolved = resolve_weathernext(
        selection,
        downloader=downloader,
        api_client=api_client,
        trainable_model=trainable_model,
        training_data=training_data,
    )
    manifest_path = Path(token_dir) / "manifest.csv"
    existing: set[tuple[str, int]] = set()
    if resume and manifest_path.exists():
        import pandas as pd

        manifest = pd.read_csv(manifest_path)
        existing = set(
            zip(manifest.storm_id.astype(str), manifest.init_time_ns.astype("int64"))
        )

    completed: list[WeatherNextPreparationResult] = []
    skipped: list[tuple[str, int]] = []
    failed: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int]] = set()
    for storm in storms:
        key = (str(storm.storm_id), int(storm.time.value))
        if key in seen:
            raise ValueError(f"Duplicate WeatherNext batch key: {key}")
        seen.add(key)
        if key in existing:
            skipped.append(key)
            continue
        try:
            completed.append(
                prepare_weathernext_sample(
                    atmospheric_state,
                    storm,
                    selection,
                    forecast_dir=forecast_dir,
                    token_dir=token_dir,
                    horizon_hours=horizon_hours,
                    initialization_mode=initialization_mode,
                    token_config=token_config,
                    resolved=resolved,
                    input_config=input_config,
                    supplemental_states=supplemental_states,
                )
            )
        except Exception as exc:
            if on_error == "raise":
                raise
            failed.append((key[0], key[1], f"{type(exc).__name__}: {exc}"))
    return WeatherNextBatchResult(tuple(completed), tuple(skipped), tuple(failed))
