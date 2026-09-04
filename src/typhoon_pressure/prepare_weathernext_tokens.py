"""Run one official/fine-tuned WeatherNext checkpoint and cache its tokens."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from .initial_condition import InitialConditionBuilder, StormObservation
from .small_version.config import WeatherNextTokenConfig
from .small_version.weathernext_bridge import (
    DirectoryForecastTokenStore,
    WeatherNextForecastTokenizer,
    run_and_save_weathernext_tokens,
)
from .weathernext_adapter import WeatherNextRequest, make_weathernext_request
from .weathernext_backends import WeatherNextBackendConfig, build_weathernext_runner


def _open_dataset(path: str) -> xr.Dataset:
    return xr.open_zarr(path) if path.endswith(".zarr") else xr.open_dataset(path)


def load_token_jobs(path: str | Path) -> pd.DataFrame:
    """Load all unique cache identities from an integrated table."""
    value = str(path)
    frame = pd.read_parquet(value) if value.endswith(".parquet") else pd.read_csv(value)
    aliases = {
        "time": "init_time",
        "typhoon_lat": "storm_lat",
        "typhoon_lon": "storm_lon",
        "typhoon_pressure_hpa": "storm_pressure_hpa",
        "typhoon_wind_kt": "storm_wind_kt",
    }
    for old, new in aliases.items():
        if new not in frame and old in frame:
            frame[new] = frame[old]
    required = {"storm_id", "init_time", "storm_lat", "storm_lon"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Token jobs table is missing columns: {missing}")
    frame = frame.copy()
    frame["storm_id"] = frame["storm_id"].astype(str).str.strip()
    frame["init_time"] = pd.to_datetime(frame["init_time"], utc=True).dt.tz_localize(None)
    if frame[["storm_id", "init_time", "storm_lat", "storm_lon"]].isna().any().any():
        raise ValueError("Token jobs contain missing identity or storm-position values")
    if frame.empty:
        raise ValueError("Token jobs table contains no rows")
    identity = ["storm_id", "init_time"]
    value_columns = [
        name
        for name in ("storm_lat", "storm_lon", "storm_pressure_hpa", "storm_wind_kt")
        if name in frame
    ]
    conflicts = frame.groupby(identity, sort=False)[value_columns].nunique(dropna=False)
    if (conflicts > 1).any().any():
        raise ValueError("Duplicate token identity has conflicting storm initialization values")
    return (
        frame.drop_duplicates(identity)
        .sort_values(identity)
        .reset_index(drop=True)[identity + value_columns]
    )


def _token_keys(frame: pd.DataFrame) -> set[tuple[str, int]]:
    return {
        (str(row.storm_id), int(pd.Timestamp(row.init_time).value))
        for row in frame.itertuples(index=False)
    }


def validate_token_cache_coverage(
    jobs: pd.DataFrame, output_dir: str | Path
) -> dict[str, int]:
    """Require one readable cache entry for every requested identity."""
    expected = _token_keys(jobs)
    store = DirectoryForecastTokenStore(output_dir)
    missing = sorted(expected.difference(store.files))
    if missing:
        raise ValueError(
            f"Forecast token cache is missing {len(missing)} requested identities; "
            f"first={missing[0]}"
        )
    for key in expected:
        if not store.files[key].is_file():
            raise FileNotFoundError(f"Token manifest points to missing file for {key}")
        store.load(*key)
    store.provenance()
    return {"requested": len(expected), "available": len(expected), "missing": 0}


def _request_for_job(
    atmospheric_state: xr.Dataset,
    row,
    *,
    backend: str,
    horizon_hours: int,
    initialization_mode: str,
) -> WeatherNextRequest:
    storm = StormObservation(
        storm_id=str(row.storm_id),
        time=pd.Timestamp(row.init_time),
        lat=float(row.storm_lat),
        lon=float(row.storm_lon),
        pressure_hpa=(
            None
            if not hasattr(row, "storm_pressure_hpa") or pd.isna(row.storm_pressure_hpa)
            else float(row.storm_pressure_hpa)
        ),
        wind_kt=(
            None
            if not hasattr(row, "storm_wind_kt") or pd.isna(row.storm_wind_kt)
            else float(row.storm_wind_kt)
        ),
    )
    if backend != "flow_matching":
        condition = InitialConditionBuilder(
            mode=initialization_mode,
            history_steps=2,
        ).build(atmospheric_state, storm)
        return make_weathernext_request(condition, horizon_hours)

    time_name = next(
        (name for name in ("time", "valid_time", "datetime") if name in atmospheric_state.coords),
        None,
    )
    if time_name is None:
        raise ValueError("Flow initial state requires a time coordinate")
    causal_state = atmospheric_state.sel(
        {time_name: slice(None, storm.time.to_datetime64())}
    )
    return WeatherNextRequest(
        initial_state=causal_state,
        tracker_seed={
            "storm_id": storm.storm_id,
            "time": storm.time,
            "lat": storm.lat,
            "lon": storm.lon,
            "pressure_hpa": storm.pressure_hpa,
            "wind_kt": storm.wind_kt,
        },
        horizon_hours=horizon_hours,
        initialization_metadata={
            "requested_mode": "flow_monthly_history",
            "correction_applied": False,
        },
    )


def build_token_cache(
    runner,
    atmospheric_state: xr.Dataset,
    jobs: pd.DataFrame,
    tokenizer: WeatherNextForecastTokenizer,
    output_dir: str | Path,
    *,
    backend: str,
    horizon_hours: int,
    initialization_mode: str = "auto",
    resume: bool = True,
) -> dict[str, int]:
    """Build every token identity with one model load and resumable writes."""
    output = Path(output_dir)
    existing: set[tuple[str, int]] = set()
    if resume and (output / "manifest.csv").is_file():
        store = DirectoryForecastTokenStore(output)
        actual = store.provenance()
        get_provenance = getattr(runner, "provenance", None)
        expected = get_provenance() if callable(get_provenance) else {}
        for field, expected_value in expected.items():
            if field in actual and expected_value is not None and str(expected_value) != actual[field]:
                raise ValueError(
                    f"Cannot resume token cache with different {field}: "
                    f"existing={actual[field]!r}, requested={str(expected_value)!r}"
                )
        existing = set(store.files)
    generated = skipped = 0
    for row in jobs.itertuples(index=False):
        key = (str(row.storm_id), int(pd.Timestamp(row.init_time).value))
        if key in existing:
            skipped += 1
            continue
        request = _request_for_job(
            atmospheric_state,
            row,
            backend=backend,
            horizon_hours=horizon_hours,
            initialization_mode=initialization_mode,
        )
        run_and_save_weathernext_tokens(runner, request, tokenizer, output)
        generated += 1
    coverage = validate_token_cache_coverage(jobs, output)
    return {**coverage, "generated": generated, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a read-only WeatherNext checkpoint and save 0-15 day Transformer tokens"
        )
    )
    parser.add_argument("--initial-state", required=True, help="Global HRES/ERA5 NetCDF or Zarr")
    parser.add_argument(
        "--backend",
        choices=["pretrained", "flow_matching"],
        default="pretrained",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model-variant",
        choices=["WeatherNext2", "WeatherNextCyclones", "WeatherNextCyclones_Mini"],
    )
    parser.add_argument("--model-id")
    parser.add_argument("--release", default="v0.3.0")
    parser.add_argument("--jobs", help="Integrated CSV/Parquet; build every unique identity")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--storm-id")
    parser.add_argument("--init-time")
    parser.add_argument("--storm-lat", type=float)
    parser.add_argument("--storm-lon", type=float)
    parser.add_argument("--storm-pressure-hpa", type=float)
    parser.add_argument("--storm-wind-kt", type=float)
    parser.add_argument(
        "--initialization-mode",
        choices=["tracker_seed", "vortex_correction", "auto"],
        default="auto",
    )
    parser.add_argument("--horizon-hours", type=int)
    parser.add_argument("--max-lead-hours", type=int)
    parser.add_argument("--output-dir", default="data/weathernext_tokens")
    parser.add_argument("--variables", nargs="*")
    parser.add_argument("--max-time-steps", type=int, default=10)
    parser.add_argument("--lat-tokens", type=int, default=6)
    parser.add_argument("--lon-tokens", type=int, default=12)
    args = parser.parse_args(argv)

    if args.backend == "pretrained" and not args.model_variant:
        parser.error("--model-variant is required for the pretrained backend")
    horizon_hours = args.horizon_hours or (
        720 if args.backend == "flow_matching" else 360
    )

    if args.jobs:
        jobs = load_token_jobs(args.jobs)
    else:
        missing = [
            name
            for name, value in {
                "--storm-id": args.storm_id,
                "--init-time": args.init_time,
                "--storm-lat": args.storm_lat,
                "--storm-lon": args.storm_lon,
            }.items()
            if value is None
        ]
        if missing:
            parser.error(f"single-sample mode requires {', '.join(missing)}")
        jobs = pd.DataFrame(
            [{
                "storm_id": args.storm_id,
                "init_time": pd.Timestamp(args.init_time),
                "storm_lat": args.storm_lat,
                "storm_lon": args.storm_lon,
                "storm_pressure_hpa": args.storm_pressure_hpa,
                "storm_wind_kt": args.storm_wind_kt,
            }]
        )
    with _open_dataset(args.initial_state) as atmospheric_state:
        config = WeatherNextBackendConfig(
            backend=args.backend,
            model_id=args.model_id or args.model_variant or "monthly-flow-matching",
            model_variant=args.model_variant,
            release=args.release if args.backend == "pretrained" else "monthly-v1",
            checkpoint=args.checkpoint,
        )
        runner = build_weathernext_runner(config)
        if args.backend == "flow_matching" and (
            horizon_hours <= 0 or horizon_hours % 720
        ):
            parser.error("flow_matching horizon must be a positive 720-hour multiple")
        defaults = WeatherNextTokenConfig()
        tokenizer = WeatherNextForecastTokenizer(
            WeatherNextTokenConfig(
                variables=tuple(args.variables) if args.variables else defaults.variables,
                max_lead_hours=args.max_lead_hours or horizon_hours,
                max_time_steps=args.max_time_steps,
                target_lat_tokens=args.lat_tokens,
                target_lon_tokens=args.lon_tokens,
            )
        )
        summary = build_token_cache(
            runner,
            atmospheric_state,
            jobs,
            tokenizer,
            args.output_dir,
            backend=args.backend,
            horizon_hours=horizon_hours,
            initialization_mode=args.initialization_mode,
            resume=not args.no_resume,
        )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
