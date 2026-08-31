"""Convert WeatherNext xarray rollouts into masked Transformer tokens."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import WeatherNextTokenConfig

PROVENANCE_FIELDS = (
    "forecast_backend",
    "forecast_checkpoint",
    "forecast_checkpoint_kind",
    "forecast_checkpoint_sha256",
    "forecast_checkpoint_format",
    "forecast_release",
    "forecast_step_hours",
    "weathernext_backend",
    "weathernext_model_id",
    "weathernext_model_variant",
    "weathernext_release",
    "weathernext_checkpoint",
    "weathernext_checkpoint_kind",
)


@dataclass(frozen=True)
class ForecastTokens:
    values: np.ndarray
    feature_mask: np.ndarray
    token_mask: np.ndarray
    positions: np.ndarray
    feature_names: tuple[str, ...]

    def validate(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("values must have shape [tokens, features]")
        if self.feature_mask.shape != self.values.shape:
            raise ValueError("feature_mask must match values")
        if self.token_mask.shape != (self.values.shape[0],):
            raise ValueError("token_mask must have shape [tokens]")
        if self.positions.shape != (self.values.shape[0], 6):
            raise ValueError("positions must have shape [tokens, 6]")


def _coordinate_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.coords or name in dataset.dims:
            return name
    raise ValueError(f"None of the coordinates {candidates} exist in WeatherNext output")


def _normalise_variable_names(dataset: xr.Dataset, requested: tuple[str, ...]) -> list[str]:
    aliases = {
        "mean_sea_level_pressure": ("mean_sea_level_pressure", "msl", "mslp"),
        "10m_u_component_of_wind": ("10m_u_component_of_wind", "u10", "10m_u_wind"),
        "10m_v_component_of_wind": ("10m_v_component_of_wind", "v10", "10m_v_wind"),
        "2m_temperature": ("2m_temperature", "t2m", "2m_temp"),
    }
    resolved = []
    for name in requested:
        match = next((candidate for candidate in aliases.get(name, (name,)) if candidate in dataset), None)
        if match is not None:
            resolved.append(match)
    if not resolved:
        raise ValueError(f"No requested WeatherNext variables are present; available={list(dataset.data_vars)}")
    return resolved


class WeatherNextForecastTokenizer:
    """Patch-pool a configured forecast horizon and preserve validity masks."""

    def __init__(self, config: WeatherNextTokenConfig | None = None):
        self.config = config or WeatherNextTokenConfig()

    def __call__(self, forecast: xr.Dataset, init_time) -> ForecastTokens:
        time_name = _coordinate_name(forecast, ("time", "valid_time", "datetime"))
        lat_name = _coordinate_name(forecast, ("latitude", "lat"))
        lon_name = _coordinate_name(forecast, ("longitude", "lon"))
        variables = _normalise_variable_names(forecast, self.config.variables)
        selected = forecast[variables]
        init = pd.Timestamp(init_time)
        times = pd.to_datetime(selected[time_name].values)
        lead_hours = np.asarray([(pd.Timestamp(value) - init).total_seconds() / 3600.0 for value in times])
        valid_times = np.flatnonzero((lead_hours >= 0) & (lead_hours <= self.config.max_lead_hours))
        if valid_times.size == 0:
            raise ValueError(
                f"Forecast output contains no 0–{self.config.max_lead_hours} hour times"
            )
        if valid_times.size > self.config.max_time_steps:
            chosen = np.linspace(0, valid_times.size - 1, self.config.max_time_steps).round().astype(int)
            valid_times = valid_times[chosen]
        selected = selected.isel({time_name: valid_times})
        lead_hours = lead_hours[valid_times]

        lat_factor = max(1, int(np.ceil(selected.sizes[lat_name] / self.config.target_lat_tokens)))
        lon_factor = max(1, int(np.ceil(selected.sizes[lon_name] / self.config.target_lon_tokens)))
        coarsen = {lat_name: lat_factor, lon_name: lon_factor}
        pooled = selected.coarsen(coarsen, boundary="pad").mean(skipna=True)
        valid_fraction = selected.notnull().coarsen(coarsen, boundary="pad").mean()

        values_per_feature, masks_per_feature = [], []
        for name in variables:
            array = pooled[name]
            extra_dims = set(array.dims).difference({time_name, lat_name, lon_name})
            if extra_dims:
                raise ValueError(
                    f"Variable {name!r} has unsupported dimensions {sorted(extra_dims)}; "
                    "select a pressure level before tokenization"
                )
            values_per_feature.append(array.transpose(time_name, lat_name, lon_name).values)
            mask = valid_fraction[name].transpose(time_name, lat_name, lon_name).values
            masks_per_feature.append(mask >= self.config.min_valid_fraction)
        values = np.stack(values_per_feature, axis=-1).astype(np.float32)
        feature_mask = np.stack(masks_per_feature, axis=-1) & np.isfinite(values)

        # Per-variable robust scaling keeps pressure and wind on comparable ranges.
        for feature in range(values.shape[-1]):
            valid = feature_mask[..., feature]
            if valid.any():
                centre = np.nanmedian(values[..., feature][valid])
                scale = np.nanpercentile(np.abs(values[..., feature][valid] - centre), 75)
                values[..., feature] = (values[..., feature] - centre) / max(float(scale), 1e-6)
        values = np.where(feature_mask, values, 0.0).astype(np.float32)

        lat = pooled[lat_name].values.astype(float)
        lon = pooled[lon_name].values.astype(float)
        tt, yy, xx = np.meshgrid(lead_hours, lat, lon, indexing="ij")
        lat_rad, lon_rad = np.deg2rad(yy), np.deg2rad(xx)
        lead_fraction = np.clip(tt / self.config.max_lead_hours, 0.0, 1.0)
        positions = np.stack((
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
            lead_fraction,
            np.sin(np.pi * lead_fraction),
            np.cos(np.pi * lead_fraction),
        ), axis=-1).astype(np.float32)
        values = values.reshape(-1, values.shape[-1])
        feature_mask = feature_mask.reshape(-1, feature_mask.shape[-1])
        token_mask = feature_mask.any(axis=-1)
        result = ForecastTokens(
            values=values,
            feature_mask=feature_mask.astype(np.float32),
            token_mask=token_mask.astype(np.float32),
            positions=positions.reshape(-1, 6),
            feature_names=tuple(variables),
        )
        result.validate()
        return result


def _safe_storm_id(storm_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(storm_id))
    if not safe:
        raise ValueError("storm_id produced an empty filename")
    return safe


def save_forecast_tokens(
    tokens: ForecastTokens,
    output_dir: str | Path,
    *,
    storm_id: str,
    init_time,
    provenance: dict[str, object] | None = None,
) -> Path:
    """Persist one tokenized rollout and update the lookup manifest."""
    tokens.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    init_time_ns = int(pd.Timestamp(init_time).value)
    filename = f"{_safe_storm_id(storm_id)}__{init_time_ns}.npz"
    path = output / filename
    np.savez_compressed(
        path,
        values=tokens.values,
        feature_mask=tokens.feature_mask,
        token_mask=tokens.token_mask,
        positions=tokens.positions,
        feature_names=np.asarray(tokens.feature_names),
    )
    manifest = output / "manifest.csv"
    row_data = {
        "storm_id": str(storm_id),
        "init_time_ns": init_time_ns,
        "file": filename,
    }
    for field in PROVENANCE_FIELDS:
        value = None if provenance is None else provenance.get(field)
        if value is not None:
            row_data[field] = str(value)
    row = pd.DataFrame([row_data])
    if manifest.exists():
        current = pd.read_csv(manifest)
        keep = ~(
            (current["storm_id"].astype(str) == str(storm_id))
            & (current["init_time_ns"].astype("int64") == init_time_ns)
        )
        row = pd.concat((current.loc[keep], row), ignore_index=True)
    row.to_csv(manifest, index=False)
    return path


def run_and_save_weathernext_tokens(
    runner,
    request,
    tokenizer: WeatherNextForecastTokenizer,
    output_dir: str | Path,
) -> Path:
    """Execute the configured WeatherNext runner and persist Transformer inputs."""
    from typhoon_pressure.weathernext_adapter import run_weathernext

    forecast = run_weathernext(runner, request)
    tokens = tokenizer(forecast, request.tracker_seed["time"])
    provenance = {
        field: forecast.attrs[field]
        for field in PROVENANCE_FIELDS
        if forecast.attrs.get(field) is not None
    }
    return save_forecast_tokens(
        tokens,
        output_dir,
        storm_id=request.tracker_seed["storm_id"],
        init_time=request.tracker_seed["time"],
        provenance=provenance,
    )


class DirectoryForecastTokenStore:
    """Load pre-tokenized WeatherNext rollouts by (storm_id, initialization time)."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        manifest = pd.read_csv(self.directory / "manifest.csv")
        self.manifest = manifest
        if manifest.duplicated(["storm_id", "init_time_ns"]).any():
            raise ValueError("WeatherNext token manifest has duplicate sample keys")
        self.files = {
            (str(row.storm_id), int(row.init_time_ns)): self.directory / str(row.file)
            for row in manifest.itertuples(index=False)
        }

    def provenance(self) -> dict[str, str]:
        """Return one consistent checkpoint identity for the token cache."""
        result = {}
        for field in PROVENANCE_FIELDS:
            if field not in self.manifest:
                continue
            values = {
                str(value)
                for value in self.manifest[field].dropna().unique()
                if str(value).strip()
            }
            if len(values) > 1:
                raise ValueError(
                    f"WeatherNext token cache mixes multiple {field} values: {sorted(values)}"
                )
            if values:
                result[field] = values.pop()
        return result

    def require_checkpoint_kind(self, expected: str) -> None:
        provenance = self.provenance()
        actual = provenance.get("forecast_checkpoint_kind")
        if actual is None:
            actual = provenance.get("weathernext_checkpoint_kind")
        if actual is None:
            raise ValueError(
                "Forecast token manifest has no checkpoint provenance; regenerate "
                "tokens with prepare-weathernext-tokens"
            )
        if actual != expected:
            raise ValueError(
                f"Expected {expected} forecast tokens, but manifest records {actual}"
            )

    def contains(self, storm_id: str, init_time_ns: int) -> bool:
        return (str(storm_id), int(init_time_ns)) in self.files

    def load(self, storm_id: str, init_time_ns: int) -> ForecastTokens:
        key = (str(storm_id), int(init_time_ns))
        if key not in self.files:
            raise KeyError(f"No WeatherNext tokens for {key}")
        with np.load(self.files[key], allow_pickle=False) as data:
            result = ForecastTokens(
                values=data["values"].astype(np.float32),
                feature_mask=data["feature_mask"].astype(np.float32),
                token_mask=data["token_mask"].astype(np.float32),
                positions=data["positions"].astype(np.float32),
                feature_names=tuple(str(value) for value in data["feature_names"]),
            )
        result.validate()
        return result
