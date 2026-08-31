"""Run one official/fine-tuned WeatherNext checkpoint and cache its tokens."""

from __future__ import annotations

import argparse

import pandas as pd
import xarray as xr

from .initial_condition import InitialConditionBuilder, StormObservation
from .small_version.config import WeatherNextTokenConfig
from .small_version.weathernext_bridge import (
    WeatherNextForecastTokenizer,
    run_and_save_weathernext_tokens,
)
from .weathernext_adapter import WeatherNextRequest, make_weathernext_request
from .weathernext_backends import WeatherNextBackendConfig, build_weathernext_runner


def _open_dataset(path: str) -> xr.Dataset:
    return xr.open_zarr(path) if path.endswith(".zarr") else xr.open_dataset(path)


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
    parser.add_argument("--storm-id", required=True)
    parser.add_argument("--init-time", required=True)
    parser.add_argument("--storm-lat", required=True, type=float)
    parser.add_argument("--storm-lon", required=True, type=float)
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

    storm = StormObservation(
        storm_id=args.storm_id,
        time=pd.Timestamp(args.init_time),
        lat=args.storm_lat,
        lon=args.storm_lon,
        pressure_hpa=args.storm_pressure_hpa,
        wind_kt=args.storm_wind_kt,
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
        if args.backend == "flow_matching":
            if horizon_hours <= 0 or horizon_hours % 720:
                parser.error("flow_matching horizon must be a positive 720-hour multiple")
            time_name = next(
                (
                    name
                    for name in ("time", "valid_time", "datetime")
                    if name in atmospheric_state.coords
                ),
                None,
            )
            if time_name is None:
                raise ValueError("Flow initial state requires a time coordinate")
            causal_state = atmospheric_state.sel(
                {time_name: slice(None, storm.time.to_datetime64())}
            ).load()
            request = WeatherNextRequest(
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
        else:
            condition = InitialConditionBuilder(
                mode=args.initialization_mode,
                history_steps=2,
            ).build(atmospheric_state, storm)
            request = make_weathernext_request(condition, horizon_hours)
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
        path = run_and_save_weathernext_tokens(
            runner, request, tokenizer, args.output_dir
        )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
