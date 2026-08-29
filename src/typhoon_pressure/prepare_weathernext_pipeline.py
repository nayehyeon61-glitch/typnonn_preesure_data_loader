from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from .initial_condition import StormObservation
from .weathernext_download import URLCheckpointDownloader
from .weathernext_pipeline import prepare_weathernext_sample
from .weathernext_resolver import WeatherNextSelectionConfig


def _load_storm(args) -> StormObservation:
    return StormObservation(
        storm_id=args.storm_id,
        time=pd.Timestamp(args.init_time),
        lat=args.lat,
        lon=args.lon,
        pressure_hpa=args.pressure_hpa,
        wind_kt=args.wind_kt,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve WeatherNext source, run frozen rollout, and build token cache"
    )
    parser.add_argument("--atmospheric-state", required=True, help="HRES/ERA5 NetCDF or Zarr with at least two 6-hour steps")
    parser.add_argument("--storm-id", required=True)
    parser.add_argument("--init-time", required=True)
    parser.add_argument("--lat", required=True, type=float)
    parser.add_argument("--lon", required=True, type=float)
    parser.add_argument("--pressure-hpa", type=float)
    parser.add_argument("--wind-kt", type=float)
    parser.add_argument("--finetuned-checkpoint")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--checkpoint-url", help="Optional public HTTPS checkpoint URL used only after local checkpoints are absent")
    parser.add_argument("--download-dir", default="download/weathernext")
    parser.add_argument("--model-variant", default="WeatherNext2")
    parser.add_argument("--model-id", default="WeatherNext2_<2025")
    parser.add_argument("--release", default="v0.3.0")
    parser.add_argument("--horizon-hours", type=int, default=360)
    parser.add_argument("--initialization-mode", choices=["tracker_seed", "vortex_correction", "auto"], default="auto")
    parser.add_argument("--forecast-dir", default="data/weathernext_forecasts")
    parser.add_argument("--token-dir", default="data/weathernext_tokens")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-api-fallback", action="store_true")
    args = parser.parse_args(argv)

    atmospheric_state = (
        xr.open_zarr(args.atmospheric_state)
        if args.atmospheric_state.endswith(".zarr")
        else xr.open_dataset(args.atmospheric_state)
    )
    storm = _load_storm(args)
    selection = WeatherNextSelectionConfig(
        model_variant=args.model_variant,
        model_id=args.model_id,
        release=args.release,
        finetuned_checkpoint=args.finetuned_checkpoint,
        pretrained_checkpoint=args.pretrained_checkpoint,
        allow_download=not args.no_download,
        allow_api_fallback=not args.no_api_fallback,
    )
    downloader = None
    if args.checkpoint_url and not args.no_download:
        downloader = URLCheckpointDownloader(
            args.checkpoint_url,
            cache_dir=args.download_dir,
        )

    # API clients require provider-specific authentication and are injected by
    # application code. This CLI intentionally handles local/downloaded frozen
    # checkpoint paths only.
    result = prepare_weathernext_sample(
        atmospheric_state,
        storm,
        selection,
        forecast_dir=args.forecast_dir,
        token_dir=args.token_dir,
        horizon_hours=args.horizon_hours,
        initialization_mode=args.initialization_mode,
        downloader=downloader,
    )
    print(
        {
            "weight_origin": result.resolved.origin.value,
            "checkpoint": result.resolved.checkpoint,
            "forecast": str(result.forecast_path),
            "tokens": str(result.token_path),
            "next": f"train-weathernext-transformer --weathernext-token-dir {Path(args.token_dir)} ...",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
