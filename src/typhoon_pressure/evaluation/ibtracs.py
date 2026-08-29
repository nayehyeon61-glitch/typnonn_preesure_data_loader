from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utility.metrics import (
    great_circle_distance_km,
    scalar_error_statistics,
    track_error_statistics,
)


@dataclass(frozen=True)
class IBTrACSEvaluationConfig:
    east_asia_only: bool = False
    min_lat: float = 0.0
    max_lat: float = 60.0
    min_lon: float = 100.0
    max_lon: float = 180.0


@dataclass(frozen=True)
class IBTrACSEvaluationResult:
    matched: pd.DataFrame
    overall: dict[str, int | float | None]
    by_lead: pd.DataFrame
    by_storm: pd.DataFrame


def _utc_naive(values) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None)


def _normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    required = {"storm_id", "pred_lat", "pred_lon"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    if "time" in frame:
        frame["time"] = _utc_naive(frame["time"])
    elif {"init_time", "lead_hours"}.issubset(frame.columns):
        init_time = _utc_naive(frame["init_time"])
        lead = pd.to_numeric(frame["lead_hours"], errors="coerce")
        frame["time"] = init_time + pd.to_timedelta(lead, unit="h")
    else:
        raise ValueError("Prediction table requires time or both init_time and lead_hours")

    frame["storm_id"] = frame["storm_id"].astype(str).str.strip()
    for column in ("pred_lat", "pred_lon", "pred_pressure_hpa", "pred_wind_kt", "lead_hours"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["storm_id", "time", "pred_lat", "pred_lon"])


def _normalize_observations(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"storm_id", "time", "typhoon_lat", "typhoon_lon"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"IBTrACS table is missing columns: {sorted(missing)}")
    frame = observations.copy()
    frame["storm_id"] = frame["storm_id"].astype(str).str.strip()
    frame["time"] = _utc_naive(frame["time"])
    for column in ("typhoon_lat", "typhoon_lon", "typhoon_pressure_hpa", "typhoon_wind_kt"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["storm_id", "time", "typhoon_lat", "typhoon_lon"]).drop_duplicates(
        ["storm_id", "time"]
    )


def match_predictions_to_ibtracs(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    config: IBTrACSEvaluationConfig = IBTrACSEvaluationConfig(),
) -> pd.DataFrame:
    """Join predictions to IBTrACS on storm_id and absolute valid time."""
    pred = _normalize_predictions(predictions)
    obs = _normalize_observations(observations)
    matched = pred.merge(obs, on=["storm_id", "time"], how="inner", validate="many_to_one")
    matched = matched.rename(columns={
        "typhoon_lat": "target_lat",
        "typhoon_lon": "target_lon",
        "typhoon_pressure_hpa": "target_pressure_hpa",
        "typhoon_wind_kt": "target_wind_kt",
    })
    if config.east_asia_only:
        lon = matched["target_lon"] % 360.0
        region = (
            matched["target_lat"].between(config.min_lat, config.max_lat)
            & lon.between(config.min_lon % 360.0, config.max_lon % 360.0)
        )
        matched = matched.loc[region].copy()
    matched["track_error_km"] = great_circle_distance_km(
        matched["pred_lat"], matched["pred_lon"], matched["target_lat"], matched["target_lon"]
    )
    if {"pred_pressure_hpa", "target_pressure_hpa"}.issubset(matched.columns):
        matched["pressure_error_hpa"] = matched["pred_pressure_hpa"] - matched["target_pressure_hpa"]
    if {"pred_wind_kt", "target_wind_kt"}.issubset(matched.columns):
        matched["wind_error_kt"] = matched["pred_wind_kt"] - matched["target_wind_kt"]
    return matched.sort_values(["storm_id", "time"]).reset_index(drop=True)


def _summary(frame: pd.DataFrame) -> dict[str, int | float | None]:
    summary: dict[str, int | float | None] = {
        "matched_count": int(len(frame)),
        "storm_count": int(frame["storm_id"].nunique()) if len(frame) else 0,
    }
    summary.update(track_error_statistics(
        frame.get("pred_lat", []), frame.get("pred_lon", []),
        frame.get("target_lat", []), frame.get("target_lon", []),
    ))
    summary["track_count"] = summary.pop("count")
    if {"pred_pressure_hpa", "target_pressure_hpa"}.issubset(frame.columns):
        pressure = dict(scalar_error_statistics(
            frame["pred_pressure_hpa"], frame["target_pressure_hpa"], prefix="pressure_hpa"
        ))
        summary["pressure_count"] = pressure.pop("count")
        summary.update(pressure)
    if {"pred_wind_kt", "target_wind_kt"}.issubset(frame.columns):
        wind = dict(scalar_error_statistics(
            frame["pred_wind_kt"], frame["target_wind_kt"], prefix="wind_kt"
        ))
        summary["wind_count"] = wind.pop("count")
        summary.update(wind)
    return summary


def _group_metrics(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in frame or frame.empty:
        return pd.DataFrame()
    rows = []
    for value, group in frame.groupby(column, dropna=False, sort=True):
        rows.append({column: value, **_summary(group)})
    return pd.DataFrame(rows)


def evaluate_ibtracs_predictions(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    config: IBTrACSEvaluationConfig = IBTrACSEvaluationConfig(),
) -> IBTrACSEvaluationResult:
    matched = match_predictions_to_ibtracs(predictions, observations, config)
    return IBTrACSEvaluationResult(
        matched=matched,
        overall=_summary(matched),
        by_lead=_group_metrics(matched, "lead_hours"),
        by_storm=_group_metrics(matched, "storm_id"),
    )


def _json_safe(mapping: dict) -> dict:
    return {
        key: None if isinstance(value, float) and not np.isfinite(value) else value
        for key, value in mapping.items()
    }


def write_ibtracs_evaluation(result: IBTrACSEvaluationResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.matched.to_csv(output / "matched_predictions.csv", index=False)
    result.by_storm.to_csv(output / "metrics_by_storm.csv", index=False)
    if not result.by_lead.empty:
        result.by_lead.to_csv(output / "metrics_by_lead.csv", index=False)
    with (output / "metrics_overall.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(result.overall), handle, ensure_ascii=False, indent=2, allow_nan=False)


def _read_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if Path(path).suffix.lower() == ".parquet" else pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    from ..ibtracs import IBTrACSConfig, load_ibtracs

    parser = argparse.ArgumentParser(description="Evaluate model predictions against IBTrACS only")
    parser.add_argument("--predictions", required=True, help="CSV or Parquet prediction table")
    parser.add_argument("--ibtracs", required=True, help="Raw IBTrACS CSV or NetCDF")
    parser.add_argument("--output-dir", default="evaluation/ibtracs")
    parser.add_argument("--basin", default="WP")
    parser.add_argument("--agency", default="TOKYO")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--east-asia-only", action="store_true")
    args = parser.parse_args(argv)

    predictions = _read_table(args.predictions)
    observations = load_ibtracs(args.ibtracs, IBTrACSConfig(
        basin=args.basin or None, agency=args.agency, start=args.start, end=args.end
    ))
    result = evaluate_ibtracs_predictions(
        predictions, observations, IBTrACSEvaluationConfig(east_asia_only=args.east_asia_only)
    )
    write_ibtracs_evaluation(result, args.output_dir)
    print(json.dumps(_json_safe(result.overall), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
