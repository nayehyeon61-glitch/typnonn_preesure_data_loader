"""Convert WeatherNext xarray rollouts into validated Transformer token caches."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import WeatherNextTokenConfig


TOKEN_PROVENANCE_COLUMNS = (
    "model_id", "model_variant", "release", "weight_origin",
    "checkpoint_fingerprint", "tokenizer_fingerprint",
    "feature_schema", "initialization_mode",
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
        if len(self.feature_names) != self.values.shape[1]:
            raise ValueError("feature_names must match token feature dimension")
        if not np.isfinite(self.values).all():
            raise ValueError("token values contain non-finite entries")
        if not np.isfinite(self.positions).all():
            raise ValueError("token positions contain non-finite entries")


def tokenizer_fingerprint(config: WeatherNextTokenConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path | None) -> str:
    if not path:
        return ""
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return ""
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coordinate_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.coords or name in dataset.dims:
            return name
    raise ValueError(f"None of the coordinates {candidates} exist in WeatherNext output")


def _resolve_required_variables(dataset: xr.Dataset, requested: tuple[str, ...]) -> list[str]:
    aliases = {
        "mean_sea_level_pressure": ("mean_sea_level_pressure", "msl", "mslp"),
        "10m_u_component_of_wind": ("10m_u_component_of_wind", "u10", "10m_u_wind"),
        "10m_v_component_of_wind": ("10m_v_component_of_wind", "v10", "10m_v_wind"),
        "2m_temperature": ("2m_temperature", "t2m", "2m_temp"),
    }
    resolved, missing = [], []
    for canonical in requested:
        match = next((candidate for candidate in aliases.get(canonical, (canonical,)) if candidate in dataset), None)
        if match is None:
            missing.append(canonical)
        else:
            resolved.append(match)
    if missing:
        raise ValueError(
            "WeatherNext tokenization requires the canonical feature schema; "
            f"missing={missing}, available={list(dataset.data_vars)}"
        )
    return resolved


class WeatherNextForecastTokenizer:
    """Patch-pool 0–15 day global fields using one canonical feature schema."""

    def __init__(self, config: WeatherNextTokenConfig = WeatherNextTokenConfig()):
        self.config = config

    def __call__(self, forecast: xr.Dataset, init_time) -> ForecastTokens:
        time_name = _coordinate_name(forecast, ("time", "valid_time", "datetime"))
        lat_name = _coordinate_name(forecast, ("latitude", "lat"))
        lon_name = _coordinate_name(forecast, ("longitude", "lon"))
        variables = _resolve_required_variables(forecast, self.config.variables)
        selected = forecast[variables]
        init = pd.Timestamp(init_time)
        times = pd.to_datetime(selected[time_name].values)
        lead_hours = np.asarray([(pd.Timestamp(value) - init).total_seconds() / 3600.0 for value in times])
        valid_times = np.flatnonzero((lead_hours >= 0) & (lead_hours <= self.config.max_lead_hours))
        if valid_times.size == 0:
            raise ValueError("WeatherNext output contains no 0–15 day forecast times")
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
            feature_names=tuple(self.config.variables),
        )
        result.validate()
        return result


def _safe_storm_id(storm_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(storm_id))
    if not safe:
        raise ValueError("storm_id produced an empty filename")
    return safe


def _normalise_provenance(provenance: dict | None) -> dict[str, str]:
    source = provenance or {}
    return {name: "" if source.get(name) is None else str(source.get(name)) for name in TOKEN_PROVENANCE_COLUMNS}


def _validate_manifest_frame(current: pd.DataFrame) -> None:
    required = {"storm_id", "init_time_ns", "file", *TOKEN_PROVENANCE_COLUMNS}
    missing_columns = sorted(required.difference(current.columns))
    if missing_columns:
        raise ValueError(
            "Existing WeatherNext token manifest is legacy/incomplete; use a new token directory or rebuild it. "
            f"Missing columns: {missing_columns}"
        )
    if current.duplicated(["storm_id", "init_time_ns"]).any():
        raise ValueError("WeatherNext token manifest has duplicate sample keys")


def save_forecast_tokens(
    tokens: ForecastTokens,
    output_dir: str | Path,
    *,
    storm_id: str,
    init_time,
    provenance: dict | None = None,
) -> Path:
    """Atomically persist a token file after manifest/provenance validation."""
    tokens.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    init_time_ns = int(pd.Timestamp(init_time).value)
    filename = f"{_safe_storm_id(storm_id)}__{init_time_ns}.npz"
    path = output / filename
    manifest = output / "manifest.csv"

    provenance_row = _normalise_provenance(provenance)
    provenance_row["feature_schema"] = json.dumps(list(tokens.feature_names), separators=(",", ":"))
    new_record = {"storm_id": str(storm_id), "init_time_ns": init_time_ns, "file": filename, **provenance_row}

    if manifest.exists():
        current = pd.read_csv(manifest, keep_default_na=False)
        _validate_manifest_frame(current)
        same = (current["storm_id"].astype(str) == str(storm_id)) & (current["init_time_ns"].astype("int64") == init_time_ns)
        if same.any():
            old = current.loc[same].iloc[0]
            mismatches = [name for name in TOKEN_PROVENANCE_COLUMNS if str(old[name]) != str(new_record[name])]
            if mismatches:
                raise ValueError(f"WeatherNext token provenance mismatch for {(storm_id, init_time_ns)}: {mismatches}")
            current = current.loc[~same]
        next_manifest = pd.concat((current, pd.DataFrame([new_record])), ignore_index=True)
    else:
        next_manifest = pd.DataFrame([new_record])

    # Write to temporary files first so a failed serialization never destroys a
    # previously valid cache entry or manifest.
    with tempfile.NamedTemporaryFile(dir=output, prefix=filename + ".", suffix=".tmp", delete=False) as handle:
        temp_npz = Path(handle.name)
    try:
        with temp_npz.open("wb") as handle:
            np.savez_compressed(
                handle,
                values=tokens.values,
                feature_mask=tokens.feature_mask,
                token_mask=tokens.token_mask,
                positions=tokens.positions,
                feature_names=np.asarray(tokens.feature_names),
            )
        # Re-open before replacement to verify archive integrity.
        with np.load(temp_npz, allow_pickle=False) as data:
            if set(("values", "feature_mask", "token_mask", "positions", "feature_names")).difference(data.files):
                raise ValueError("Temporary token NPZ failed integrity validation")
        temp_npz.replace(path)
    except Exception:
        temp_npz.unlink(missing_ok=True)
        raise

    temp_manifest = manifest.with_suffix(".csv.tmp")
    next_manifest.to_csv(temp_manifest, index=False)
    temp_manifest.replace(manifest)
    return path


def run_and_save_weathernext_tokens(runner, request, tokenizer: WeatherNextForecastTokenizer, output_dir: str | Path) -> Path:
    from typhoon_pressure.weathernext_adapter import run_weathernext
    forecast = run_weathernext(runner, request)
    tokens = tokenizer(forecast, request.tracker_seed["time"])
    return save_forecast_tokens(tokens, output_dir, storm_id=request.tracker_seed["storm_id"], init_time=request.tracker_seed["time"])


class DirectoryForecastTokenStore:
    """Load and validate pre-tokenized WeatherNext rollouts by sample key."""

    def __init__(self, directory: str | Path, *, validate_files: bool = True):
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"WeatherNext token manifest does not exist: {manifest_path}")
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        _validate_manifest_frame(manifest)
        self.manifest = manifest
        self.files = {
            (str(row.storm_id), int(row.init_time_ns)): self.directory / str(row.file)
            for row in manifest.itertuples(index=False)
        }
        self.provenance = {
            (str(row.storm_id), int(row.init_time_ns)): {name: str(getattr(row, name)) for name in TOKEN_PROVENANCE_COLUMNS}
            for row in manifest.itertuples(index=False)
        }
        if validate_files:
            schemas: set[tuple[str, ...]] = set()
            for key, path in self.files.items():
                if not path.is_file():
                    raise FileNotFoundError(f"WeatherNext token file is missing for {key}: {path}")
                try:
                    tokens = self.load(*key)
                except Exception as exc:
                    raise ValueError(f"Invalid WeatherNext token file for {key}: {path}: {exc}") from exc
                schemas.add(tokens.feature_names)
                try:
                    manifest_schema = tuple(json.loads(self.provenance[key]["feature_schema"]))
                except Exception as exc:
                    raise ValueError(f"Invalid manifest feature_schema for {key}") from exc
                if tokens.feature_names != manifest_schema:
                    raise ValueError(f"Token file/manifest feature schema mismatch for {key}")
            if len(schemas) > 1:
                raise ValueError(f"WeatherNext token directory contains mixed feature schemas: {sorted(schemas)}")

    def contains(self, storm_id: str, init_time_ns: int) -> bool:
        return (str(storm_id), int(init_time_ns)) in self.files

    def entry_matches_provenance(self, storm_id: str, init_time_ns: int, expected: dict) -> bool:
        key = (str(storm_id), int(init_time_ns))
        if key not in self.files or not self.files[key].is_file():
            return False
        actual = self.provenance[key]
        normalized = _normalise_provenance(expected)
        return all(str(actual[name]) == str(normalized[name]) for name in TOKEN_PROVENANCE_COLUMNS)

    def load(self, storm_id: str, init_time_ns: int) -> ForecastTokens:
        key = (str(storm_id), int(init_time_ns))
        if key not in self.files:
            raise KeyError(f"No WeatherNext tokens for {key}")
        path = self.files[key]
        if not path.is_file():
            raise FileNotFoundError(f"WeatherNext token file is missing for {key}: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"values", "feature_mask", "token_mask", "positions", "feature_names"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"Token NPZ is missing arrays: {missing}")
            result = ForecastTokens(
                values=data["values"].astype(np.float32),
                feature_mask=data["feature_mask"].astype(np.float32),
                token_mask=data["token_mask"].astype(np.float32),
                positions=data["positions"].astype(np.float32),
                feature_names=tuple(str(value) for value in data["feature_names"]),
            )
        result.validate()
        return result
