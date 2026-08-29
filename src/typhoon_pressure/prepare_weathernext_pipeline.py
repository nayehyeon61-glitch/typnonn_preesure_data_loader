from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pandas as pd
import xarray as xr

from .dataset import TyphoonPressureDataset
from .initial_condition import StormObservation
from .weathernext_download import URLCheckpointDownloader
from .weathernext_pipeline import prepare_weathernext_batch
from .weathernext_input import WeatherNextInputConfig
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


def _load_factory(spec: str):
    if ":" not in spec:
        raise ValueError("Factory must use 'module:callable' syntax")
    module_name, attribute = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory()


def _storms_from_frame(
    frame: pd.DataFrame,
    *,
    history: int = 8,
    horizon: int = 20,
    max_highs: int = 3,
) -> list[StormObservation]:
    integrated_columns = {
        "storm_id", "time", "typhoon_lat", "typhoon_lon",
        "typhoon_pressure_hpa", "typhoon_wind_kt",
    }
    if integrated_columns.issubset(frame.columns):
        frame = frame.copy()
        frame["time"] = pd.to_datetime(frame["time"])
        for column in (
            "high_rank", "high_dx_km", "high_dy_km",
            "high_pressure_hpa", "high_anomaly_hpa",
        ):
            if column not in frame:
                frame[column] = float("nan")
        dataset = TyphoonPressureDataset(
            frame, history=history, horizon=horizon, max_highs=max_highs
        )
        storms = []
        for storm_id, start in dataset.windows:
            row = dataset.groups[storm_id].iloc[start + history - 1]
            storms.append(StormObservation(
                storm_id=str(storm_id),
                time=pd.Timestamp(row.time),
                lat=float(row.typhoon_lat),
                lon=float(row.typhoon_lon),
                pressure_hpa=None if pd.isna(row.typhoon_pressure_hpa) else float(row.typhoon_pressure_hpa),
                wind_kt=None if pd.isna(row.typhoon_wind_kt) else float(row.typhoon_wind_kt),
            ))
        return storms
    aliases = {
        "time": "init_time",
        "typhoon_lat": "lat",
        "typhoon_lon": "lon",
        "typhoon_pressure_hpa": "pressure_hpa",
        "typhoon_wind_kt": "wind_kt",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if value not in frame})
    required = {"storm_id", "init_time", "lat", "lon"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Storm jobs file is missing columns: {missing}")
    storms = []
    for row in frame.itertuples(index=False):
        pressure = getattr(row, "pressure_hpa", None)
        wind = getattr(row, "wind_kt", None)
        storms.append(StormObservation(
            storm_id=str(row.storm_id),
            time=pd.Timestamp(row.init_time),
            lat=float(row.lat),
            lon=float(row.lon),
            pressure_hpa=None if pressure is None or pd.isna(pressure) else float(pressure),
            wind_kt=None if wind is None or pd.isna(wind) else float(wind),
        ))
    return storms


def _open_state(path: str) -> xr.Dataset:
    return xr.open_zarr(path) if path.rstrip("/").endswith(".zarr") else xr.open_dataset(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare inputs, select WeatherNext mode, and build token caches for one or many initializations"
    )
    parser.add_argument("--atmospheric-state", required=True, help="HRES/ERA5 NetCDF or Zarr with at least two 6-hour steps")
    parser.add_argument("--jobs", help="CSV/Parquet containing all storm_id, init_time, lat, lon jobs")
    parser.add_argument("--job-history", type=int, default=8, help="History steps when --jobs is an integrated dataset")
    parser.add_argument("--job-horizon", type=int, default=20, help="Future steps when --jobs is an integrated dataset")
    parser.add_argument("--job-max-highs", type=int, default=3)
    parser.add_argument("--storm-id")
    parser.add_argument("--init-time")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--pressure-hpa", type=float)
    parser.add_argument("--wind-kt", type=float)
    parser.add_argument("--finetuned-checkpoint")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--checkpoint-url", help="Optional public HTTPS checkpoint URL used only after local checkpoints are absent")
    parser.add_argument("--download-dir", default="download/weathernext")
    parser.add_argument("--model-variant", default="WeatherNext2")
    parser.add_argument("--model-id", default="WeatherNext2_<2025")
    parser.add_argument("--release", default="v0.3.0")
    parser.add_argument("--execution-mode", choices=["auto", "pretrained", "api", "trainable"], default="pretrained")
    parser.add_argument("--api-provider")
    parser.add_argument("--api-client-factory", help="module:callable returning a WeatherNext API client")
    parser.add_argument("--trainable-factory", help="module:callable returning (trainable_model, training_data)")
    parser.add_argument("--training-kwargs", default="{}", help="JSON passed to the trainable model fit method")
    parser.add_argument("--horizon-hours", type=int, default=360)
    parser.add_argument("--initialization-mode", choices=["tracker_seed", "vortex_correction", "auto"], default="auto")
    parser.add_argument("--forecast-dir", default="data/weathernext_forecasts")
    parser.add_argument("--token-dir", default="data/weathernext_tokens")
    parser.add_argument("--supplement-state", action="append", default=[], help="NetCDF/Zarr with missing SST/static/100 m variables; repeatable")
    parser.add_argument("--regrid", action="store_true", help="Explicitly interpolate a regular global grid to the selected model grid")
    parser.add_argument("--skip-input-preparation", action="store_true", help="Skip strict WeatherNext input merge/contract validation")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--on-error", choices=["raise", "continue"], default="raise")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-api-fallback", action="store_true")
    args = parser.parse_args(argv)

    atmospheric_state = _open_state(args.atmospheric_state)
    supplements = tuple(_open_state(path) for path in args.supplement_state)
    if args.jobs:
        jobs = pd.read_parquet(args.jobs) if args.jobs.endswith(".parquet") else pd.read_csv(args.jobs)
        storms = _storms_from_frame(
            jobs,
            history=args.job_history,
            horizon=args.job_horizon,
            max_highs=args.job_max_highs,
        )
    else:
        missing = [name for name in ("storm_id", "init_time", "lat", "lon") if getattr(args, name) is None]
        if missing:
            parser.error("single-sample mode requires --" + ", --".join(name.replace("_", "-") for name in missing))
        storms = [_load_storm(args)]
    if not storms:
        parser.error("No WeatherNext initialization jobs were supplied")
    selection = WeatherNextSelectionConfig(
        model_variant=args.model_variant,
        model_id=args.model_id,
        release=args.release,
        finetuned_checkpoint=args.finetuned_checkpoint,
        pretrained_checkpoint=args.pretrained_checkpoint,
        allow_download=not args.no_download,
        allow_api_fallback=not args.no_api_fallback,
        api_provider=args.api_provider,
        execution_mode=args.execution_mode,
        training_kwargs=json.loads(args.training_kwargs),
    )
    downloader = None
    if args.checkpoint_url and not args.no_download:
        downloader = URLCheckpointDownloader(
            args.checkpoint_url,
            cache_dir=args.download_dir,
        )

    api_client = _load_factory(args.api_client_factory) if args.api_client_factory else None
    trainable_model = training_data = None
    if args.trainable_factory:
        produced = _load_factory(args.trainable_factory)
        if not isinstance(produced, tuple) or len(produced) != 2:
            raise ValueError("trainable factory must return (model, training_data)")
        trainable_model, training_data = produced
    input_config = None if args.skip_input_preparation else WeatherNextInputConfig(
        model_variant=args.model_variant,
        regrid=args.regrid,
    )

    result = prepare_weathernext_batch(
        atmospheric_state,
        storms,
        selection,
        forecast_dir=args.forecast_dir,
        token_dir=args.token_dir,
        horizon_hours=args.horizon_hours,
        initialization_mode=args.initialization_mode,
        downloader=downloader,
        api_client=api_client,
        input_config=input_config,
        supplemental_states=supplements,
        trainable_model=trainable_model,
        training_data=training_data,
        resume=not args.no_resume,
        on_error=args.on_error,
    )
    print(
        {
            "requested": len(storms),
            "completed": len(result.completed),
            "skipped": len(result.skipped),
            "failed": len(result.failed),
            "failures": list(result.failed),
            "next": f"train-weathernext-transformer --weathernext-token-dir {Path(args.token_dir)} ...",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
