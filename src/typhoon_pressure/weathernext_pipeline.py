from __future__ import annotations

import json
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
    DirectoryForecastTokenStore,
    WeatherNextForecastTokenizer,
    file_fingerprint,
    save_forecast_tokens,
    tokenizer_fingerprint,
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


def _token_provenance(
    resolved: ResolvedWeatherNext,
    selection: WeatherNextSelectionConfig,
    token_config: WeatherNextTokenConfig,
    initialization_mode: str,
) -> dict[str, str]:
    return {
        "model_id": selection.model_id,
        "model_variant": selection.model_variant,
        "release": selection.release,
        "weight_origin": resolved.origin.value,
        "checkpoint_fingerprint": file_fingerprint(resolved.checkpoint),
        "tokenizer_fingerprint": tokenizer_fingerprint(token_config),
        "feature_schema": json.dumps(list(token_config.variables), separators=(",", ":")),
        "initialization_mode": initialization_mode,
    }


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
    """Resolve WeatherNext, run forecast, persist forecast, then tokenize."""
    if input_config is not None:
        atmospheric_state = prepare_weathernext_input(
            atmospheric_state,
            storm.time,
            supplements=supplemental_states,
            config=input_config,
        )

    condition = InitialConditionBuilder(mode=initialization_mode, history_steps=2).build(atmospheric_state, storm)
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
    forecast.attrs.update({
        "storm_id": storm.storm_id,
        "initialization_time": str(storm.time),
        "initialization_mode": condition.applied_mode,
    })

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
        provenance=_token_provenance(resolved, selection, token_config, initialization_mode),
    )

    return WeatherNextPreparationResult(resolved=resolved, forecast_path=forecast_path, token_path=token_path)


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
    """Generate provenance-consistent tokens for all requested initializations."""
    if on_error not in {"raise", "continue"}:
        raise ValueError("on_error must be 'raise' or 'continue'")
    resolved = resolve_weathernext(
        selection,
        downloader=downloader,
        api_client=api_client,
        trainable_model=trainable_model,
        training_data=training_data,
    )
    expected_provenance = _token_provenance(resolved, selection, token_config, initialization_mode)
    manifest_path = Path(token_dir) / "manifest.csv"
    existing_store = None
    if resume and manifest_path.exists():
        # Do not pre-validate every file here: a missing/corrupt file should be
        # regenerated rather than treated as a completed sample. Schema and
        # duplicate/manifest-column integrity are still checked immediately.
        existing_store = DirectoryForecastTokenStore(token_dir, validate_files=False)

    completed: list[WeatherNextPreparationResult] = []
    skipped: list[tuple[str, int]] = []
    failed: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int]] = set()
    for storm in storms:
        key = (str(storm.storm_id), int(storm.time.value))
        if key in seen:
            raise ValueError(f"Duplicate WeatherNext batch key: {key}")
        seen.add(key)
        if existing_store is not None and existing_store.contains(*key):
            if not existing_store.entry_matches_provenance(*key, expected_provenance):
                raise ValueError(
                    f"WeatherNext token provenance mismatch for existing sample {key}. "
                    "Use a new token directory or rebuild the cache."
                )
            try:
                existing_store.load(*key)
            except Exception:
                # Missing/corrupt NPZ: regenerate this sample below.
                pass
            else:
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
