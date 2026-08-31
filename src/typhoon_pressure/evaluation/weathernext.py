"""Held-out storm evaluation for the WeatherNext fusion transformer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from typhoon_pressure.dataset import TyphoonPressureDataset
from typhoon_pressure.small_version.config import DistributionSamplingConfig, SmallModelConfig, TransformerConfig
from typhoon_pressure.small_version.dataset import WeatherNextDualTargetDataset
from typhoon_pressure.small_version.model import WeatherNextFusionTransformer
from typhoon_pressure.small_version.train import forward_batch
from typhoon_pressure.small_version.weathernext_bridge import DirectoryForecastTokenStore
from .storm_split import StormSplitSubset, load_storm_split

COVERAGE_LEVELS = (0.50, 0.80, 0.95)


def _wrapped_lon_delta(left, right):
    return torch.remainder(left - right + 180.0, 360.0) - 180.0


def great_circle_distance_km_torch(lat1, lon1, lat2, lon2):
    lat1_rad, lat2_rad = torch.deg2rad(lat1), torch.deg2rad(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = torch.deg2rad(_wrapped_lon_delta(lon2, lon1))
    haversine = (torch.sin(dlat / 2).square() + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2).square()).clamp(0.0, 1.0)
    return 6371.0088 * 2.0 * torch.arcsin(torch.sqrt(haversine))


def probabilistic_trajectory_metrics(mean_latlon, covariance, samples, target_latlon):
    residual = torch.stack((
        target_latlon[..., 0] - mean_latlon[..., 0],
        _wrapped_lon_delta(target_latlon[..., 1], mean_latlon[..., 1]),
    ), dim=-1)
    eye = torch.eye(2, dtype=covariance.dtype, device=covariance.device)
    stable = covariance + 1e-4 * eye
    solved = torch.linalg.solve(stable, residual.unsqueeze(-1)).squeeze(-1)
    mahalanobis2 = (residual * solved).sum(dim=-1)
    sign, logdet = torch.linalg.slogdet(stable)
    if not torch.all(sign > 0):
        raise ValueError("Trajectory covariance must be positive definite")
    gaussian_nll = 0.5 * (mahalanobis2 + logdet + 2.0 * math.log(2.0 * math.pi))
    position_error = great_circle_distance_km_torch(mean_latlon[..., 0], mean_latlon[..., 1], target_latlon[..., 0], target_latlon[..., 1])
    sample_target = great_circle_distance_km_torch(samples[..., 0], samples[..., 1], target_latlon[:, None, :, 0], target_latlon[:, None, :, 1])
    pairwise = great_circle_distance_km_torch(
        samples[:, :, None, :, 0], samples[:, :, None, :, 1],
        samples[:, None, :, :, 0], samples[:, None, :, :, 1],
    )
    result = {
        "position_error_km": position_error,
        "gaussian_nll": gaussian_nll,
        "energy_score_km": sample_target.mean(dim=1) - 0.5 * pairwise.mean(dim=(1, 2)),
        "mahalanobis2": mahalanobis2,
        "sharpness_std_deg": torch.sqrt(covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(0.0)),
    }
    for level in COVERAGE_LEVELS:
        result[f"coverage_{int(level * 100)}"] = (mahalanobis2 <= -2.0 * math.log(1.0 - level)).float()
    return result


def survival_metrics(probability, target):
    clipped = probability.clamp(1e-6, 1.0 - 1e-6)
    return {
        "survival_brier": (probability - target).square(),
        "survival_bce": -(target * torch.log(clipped) + (1.0 - target) * torch.log1p(-clipped)),
    }


def _mean(frame, column):
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def summarize_long_range(frame):
    position = frame.loc[frame["position_valid"].astype(bool)]
    survival = frame.loc[frame["survival_valid"].astype(bool)]
    result = {
        "initialization_count": int(frame[["storm_id", "init_time_ns"]].drop_duplicates().shape[0]),
        "storm_count": int(frame["storm_id"].nunique()),
        "position_count": int(len(position)), "survival_count": int(len(survival)),
        "position_mae_km": _mean(position, "position_error_km"), "position_rmse_km": None,
        "gaussian_nll": _mean(position, "gaussian_nll"), "energy_score_km": _mean(position, "energy_score_km"),
        "sharpness_std_deg": _mean(position, "sharpness_std_deg"),
        "survival_brier": _mean(survival, "survival_brier"), "survival_bce": _mean(survival, "survival_bce"),
        "alive_prevalence": _mean(survival, "alive_target"), "mean_survival_probability": _mean(survival, "survival_probability"),
    }
    if len(position):
        result["position_rmse_km"] = float(np.sqrt(np.mean(np.square(position["position_error_km"].to_numpy(float)))))
    for level in COVERAGE_LEVELS:
        result[f"coverage_{int(level * 100)}"] = _mean(position, f"coverage_{int(level * 100)}")
    return result


def summarize_short_track(frame):
    valid = frame.loc[frame["track_valid"].astype(bool)]
    if not len(valid):
        return {"short_track_count": 0, "short_track_mae_km": None, "short_track_rmse_km": None}
    error = valid["track_error_km"].to_numpy(float)
    return {"short_track_count": int(len(valid)), "short_track_mae_km": float(error.mean()), "short_track_rmse_km": float(np.sqrt(np.mean(error ** 2)))}


def _numpy(value):
    return value.detach().cpu().numpy()


def _prediction_records(outputs, batch, lead_days):
    trajectory = probabilistic_trajectory_metrics(outputs["distribution_mean_latlon"], outputs["distribution_marginal_covariance"], outputs["distribution_samples"], batch["future_track_target"])
    alive = survival_metrics(outputs["survival_probability"], batch["future_alive_target"])
    arrays = {
        "mean": _numpy(outputs["distribution_mean_latlon"]), "cov": _numpy(outputs["distribution_marginal_covariance"]),
        "target": _numpy(batch["future_track_target"]), "position_mask": _numpy(batch["future_track_mask"]),
        "alive_target": _numpy(batch["future_alive_target"]), "alive_mask": _numpy(batch["future_alive_mask"]),
        "survival": _numpy(outputs["survival_probability"]),
        **{key: _numpy(value) for key, value in trajectory.items()}, **{key: _numpy(value) for key, value in alive.items()},
    }
    storm_ids = [str(value) for value in batch["storm_id"]]
    init_times = _numpy(batch["init_time_ns"]).astype(np.int64)
    long_rows = []
    metric_keys = ("position_error_km", "gaussian_nll", "energy_score_km", "mahalanobis2", "sharpness_std_deg", "coverage_50", "coverage_80", "coverage_95", "survival_brier", "survival_bce")
    for b, storm_id in enumerate(storm_ids):
        for t, lead_day in enumerate(lead_days):
            cov = arrays["cov"][b, t]
            row = {
                "storm_id": storm_id, "init_time_ns": int(init_times[b]), "lead_day": int(lead_day),
                "pred_lat": float(arrays["mean"][b, t, 0]), "pred_lon": float(arrays["mean"][b, t, 1]),
                "target_lat": float(arrays["target"][b, t, 0]), "target_lon": float(arrays["target"][b, t, 1]),
                "position_valid": int(arrays["position_mask"][b, t] > 0),
                "alive_target": float(arrays["alive_target"][b, t]), "survival_valid": int(arrays["alive_mask"][b, t] > 0),
                "survival_probability": float(arrays["survival"][b, t]), "no_storm_probability": float(1.0 - arrays["survival"][b, t]),
                "cov_lat_lat": float(cov[0, 0]), "cov_lat_lon": float(cov[0, 1]), "cov_lon_lon": float(cov[1, 1]),
            }
            row.update({key: float(arrays[key][b, t]) for key in metric_keys})
            long_rows.append(row)
    track_error = great_circle_distance_km_torch(outputs["track_latlon"][..., 0], outputs["track_latlon"][..., 1], batch["track_target"][..., 0], batch["track_target"][..., 1])
    prediction, target, mask, error = map(_numpy, (outputs["track_latlon"], batch["track_target"], batch["track_mask"], track_error))
    short_rows = []
    for b, storm_id in enumerate(storm_ids):
        for step in range(prediction.shape[1]):
            short_rows.append({"storm_id": storm_id, "init_time_ns": int(init_times[b]), "lead_hours": (step + 1) * 6, "pred_lat": float(prediction[b, step, 0]), "pred_lon": float(prediction[b, step, 1]), "target_lat": float(target[b, step, 0]), "target_lon": float(target[b, step, 1]), "track_valid": int(mask[b, step] > 0), "track_error_km": float(error[b, step])})
    return long_rows, short_rows


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a WeatherNext fusion checkpoint on held-out storms")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--integrated", required=True)
    parser.add_argument("--weathernext-token-dir", required=True); parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--gpt-state-dir"); parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025); parser.add_argument("--output-dir", default="evaluation/weathernext_test")
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    for field in ("model", "model_config", "transformer_config", "sampling_config", "data_config"):
        if field not in checkpoint:
            raise ValueError(f"Checkpoint is missing required evaluation metadata: {field}")
    model_config = SmallModelConfig(**checkpoint["model_config"]); transformer_config = TransformerConfig(**checkpoint["transformer_config"]); sampling_config = DistributionSamplingConfig(**checkpoint["sampling_config"])
    data_config = checkpoint["data_config"]
    integrated = pd.read_parquet(args.integrated) if Path(args.integrated).suffix.lower() == ".parquet" else pd.read_csv(args.integrated)
    integrated["time"] = pd.to_datetime(integrated["time"])
    base = TyphoonPressureDataset(integrated, history=int(data_config["history"]), horizon=int(data_config["track_steps"]), max_highs=int(data_config["max_highs"]))
    store = DirectoryForecastTokenStore(args.weathernext_token_dir)
    kwargs = dict(base_dataset=base, distribution=None, model_config=model_config, forecast_store=store, max_forecast_tokens=int(data_config["max_forecast_tokens"]), forecast_input_dim=transformer_config.forecast_input_dim, require_endpoint_lead_hours=model_config.distribution_start_day * 24)
    if transformer_config.gpt_state_dim > 0:
        if not args.gpt_state_dir:
            raise ValueError("--gpt-state-dir is required by this checkpoint")
        from typhoon_pressure.small_version.gpt_state import DirectoryGPTStateStore, WeatherNextGPTDualTargetDataset
        gpt_store = DirectoryGPTStateStore(args.gpt_state_dir)
        dataset = WeatherNextGPTDualTargetDataset(**kwargs, gpt_state_store=gpt_store)
    else:
        dataset = WeatherNextDualTargetDataset(**kwargs)
    evaluation_dataset = StormSplitSubset(dataset, load_storm_split(args.split_manifest), args.split)
    trained = set(checkpoint.get("train_storm_ids", ())); validation = set(checkpoint.get("validation_storm_ids", ()))
    if not trained:
        raise ValueError("Checkpoint lacks train_storm_ids provenance")
    if evaluation_dataset.storm_ids & trained or (args.split == "test" and evaluation_dataset.storm_ids & validation):
        raise ValueError("Evaluation split leaks storms used by training or checkpoint selection")
    model = WeatherNextFusionTransformer(model_config, transformer_config, sampling_config=sampling_config).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    loader = DataLoader(evaluation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    long_rows, short_rows = [], []
    with torch.no_grad():
        for batch in loader:
            tensors = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            first, second = _prediction_records(forward_batch(model, tensors), tensors, model_config.lead_days)
            long_rows.extend(first); short_rows.extend(second)
    long_frame, short_frame = pd.DataFrame(long_rows), pd.DataFrame(short_rows)
    overall = {"split": args.split, "checkpoint_epoch": int(checkpoint.get("epoch", -1)), **summarize_long_range(long_frame), **summarize_short_track(short_frame)}
    per_lead = pd.DataFrame([{"lead_day": int(key), **summarize_long_range(group)} for key, group in long_frame.groupby("lead_day", sort=True)])
    per_storm = pd.DataFrame([{"storm_id": key, **summarize_long_range(group)} for key, group in long_frame.groupby("storm_id", sort=True)])
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    long_frame.to_csv(output / "long_range_predictions.csv", index=False); short_frame.to_csv(output / "short_track_predictions.csv", index=False)
    per_lead.to_csv(output / "metrics_by_lead.csv", index=False); per_storm.to_csv(output / "metrics_by_storm.csv", index=False)
    with (output / "metrics_overall.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(overall), handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(_json_safe(overall), ensure_ascii=False, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
