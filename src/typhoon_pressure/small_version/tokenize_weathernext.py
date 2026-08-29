from __future__ import annotations

import argparse

import xarray as xr

from .config import WeatherNextTokenConfig
from .weathernext_bridge import WeatherNextForecastTokenizer, save_forecast_tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tokenize one WeatherNext 0–15 day rollout")
    parser.add_argument("--forecast", required=True, help="WeatherNext NetCDF/Zarr output")
    parser.add_argument("--storm-id", required=True)
    parser.add_argument("--init-time", required=True)
    parser.add_argument("--output-dir", default="data/weathernext_tokens")
    parser.add_argument("--variables", nargs="*")
    parser.add_argument("--max-time-steps", type=int, default=10)
    parser.add_argument("--lat-tokens", type=int, default=6)
    parser.add_argument("--lon-tokens", type=int, default=12)
    args = parser.parse_args(argv)

    dataset = xr.open_zarr(args.forecast) if args.forecast.endswith(".zarr") else xr.open_dataset(args.forecast)
    defaults = WeatherNextTokenConfig()
    config = WeatherNextTokenConfig(
        variables=tuple(args.variables) if args.variables else defaults.variables,
        max_time_steps=args.max_time_steps,
        target_lat_tokens=args.lat_tokens,
        target_lon_tokens=args.lon_tokens,
    )
    tokens = WeatherNextForecastTokenizer(config)(dataset, args.init_time)
    path = save_forecast_tokens(
        tokens, args.output_dir, storm_id=args.storm_id, init_time=args.init_time
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
