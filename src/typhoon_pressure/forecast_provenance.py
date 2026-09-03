"""Backend-neutral provenance helpers for frozen forecast token caches."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .small_version.weathernext_bridge import (
    file_fingerprint,
    save_forecast_tokens,
    tokenizer_fingerprint,
)
from .weathernext_adapter import run_weathernext


FORECAST_PROVENANCE_COLUMNS = (
    "forecast_backend",
    "forecast_checkpoint",
    "forecast_checkpoint_kind",
    "forecast_checkpoint_sha256",
    "forecast_checkpoint_format",
    "forecast_release",
    "forecast_step_hours",
    "forecast_schema_format",
    "forecast_horizon_hours",
)


def _string(value) -> str:
    return "" if value is None else str(value)


def backend_provenance(runner, forecast) -> dict[str, str]:
    values: dict[str, object] = {}
    provider = getattr(runner, "provenance", None)
    if callable(provider):
        values.update(provider())
    values.update({key: value for key, value in forecast.attrs.items() if key.startswith("forecast_")})

    checkpoint = values.get("forecast_checkpoint")
    if not values.get("forecast_checkpoint_sha256") and checkpoint:
        values["forecast_checkpoint_sha256"] = file_fingerprint(str(checkpoint))
    return {name: _string(values.get(name)) for name in FORECAST_PROVENANCE_COLUMNS}


def _legacy_provenance(runner, tokenizer, request, generic: dict[str, str]) -> dict[str, str]:
    provider = getattr(runner, "provenance", None)
    raw = dict(provider()) if callable(provider) else {}
    return {
        "model_id": _string(raw.get("weathernext_model_id") or raw.get("forecast_backend")),
        "model_variant": _string(raw.get("weathernext_model_variant") or raw.get("forecast_checkpoint_kind")),
        "release": _string(raw.get("weathernext_release") or generic.get("forecast_release")),
        "weight_origin": _string(generic.get("forecast_backend") or raw.get("weathernext_backend")),
        "checkpoint_fingerprint": _string(generic.get("forecast_checkpoint_sha256")),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer.config),
        "feature_schema": "",  # save_forecast_tokens writes the canonical schema.
        "initialization_mode": _string(
            request.initialization_metadata.get("requested_mode")
            if request.initialization_metadata else ""
        ),
    }


def _augment_manifest(
    output_dir: str | Path,
    *,
    storm_id: str,
    init_time,
    provenance: dict[str, str],
) -> None:
    manifest_path = Path(output_dir) / "manifest.csv"
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    key = (
        (frame["storm_id"].astype(str) == str(storm_id))
        & (frame["init_time_ns"].astype("int64") == int(pd.Timestamp(init_time).value))
    )
    if key.sum() != 1:
        raise ValueError("Forecast manifest key is not unique while writing provenance")

    for name in FORECAST_PROVENANCE_COLUMNS:
        value = provenance[name]
        if name not in frame:
            frame[name] = ""
        other = frame.loc[~key, name].astype(str)
        existing = {item for item in other if item.strip()}
        if existing and existing != {value}:
            raise ValueError(
                f"Forecast token directory contains mixed {name}: {sorted(existing | {value})}"
            )
        frame.loc[key, name] = value

    temporary = manifest_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(manifest_path)


def run_and_save_forecast_tokens(runner, request, tokenizer, output_dir: str | Path) -> Path:
    """Run any frozen backend and persist tokens plus backend-neutral provenance."""
    forecast = run_weathernext(runner, request)
    tokens = tokenizer(
        forecast,
        request.tracker_seed["time"],
        tracker_seed_latlon=(
            float(request.tracker_seed["lat"]), float(request.tracker_seed["lon"])
        ),
    )
    generic = backend_provenance(runner, forecast)
    generic["forecast_horizon_hours"] = str(int(request.horizon_hours))
    legacy = _legacy_provenance(runner, tokenizer, request, generic)
    path = save_forecast_tokens(
        tokens,
        output_dir,
        storm_id=request.tracker_seed["storm_id"],
        init_time=request.tracker_seed["time"],
        provenance=legacy,
    )
    _augment_manifest(
        output_dir,
        storm_id=request.tracker_seed["storm_id"],
        init_time=request.tracker_seed["time"],
        provenance=generic,
    )
    return path


def cache_forecast_provenance(store) -> dict[str, str]:
    """Return one consistent generic forecast identity from a token store."""
    frame = store.manifest
    result: dict[str, str] = {}
    for name in FORECAST_PROVENANCE_COLUMNS:
        if name not in frame:
            continue
        values = {str(value) for value in frame[name].astype(str) if str(value).strip()}
        if len(values) > 1:
            raise ValueError(f"Forecast token cache mixes multiple {name} values: {sorted(values)}")
        if values:
            result[name] = values.pop()
    return result
