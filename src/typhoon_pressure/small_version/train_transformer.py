from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from typhoon_pressure.dataset import TyphoonPressureDataset
from typhoon_pressure.evaluation.storm_split import StormSplitSubset, load_storm_split

from .config import (
    DistributionSamplingConfig,
    DualLossConfig,
    SmallModelConfig,
    TransformerConfig,
)
from .dataset import SpatialDistributionLookup, WeatherNextDualTargetDataset
from .losses import DualObjectiveLoss
from .model import WeatherNextFusionTransformer
from .train import evaluate_epoch, train_epoch
from .weathernext_bridge import DirectoryForecastTokenStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train WeatherNext/GPT fusion with adaptive day 15-30 trajectory sampling"
    )
    parser.add_argument("--integrated", required=True)
    parser.add_argument("--distribution", help="Optional legacy grid metadata")
    parser.add_argument("--weathernext-token-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument(
        "--gpt-state-dir",
        help="Enable GPT Router/noise conditioning with cache produced by build-gpt-state-cache",
    )
    parser.add_argument(
        "--require-valid-gpt-states",
        action="store_true",
        help="Fail when any cached GPT record is masked due to an API failure",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--track-steps", type=int, default=20)
    parser.add_argument("--max-highs", type=int, default=3)
    parser.add_argument("--max-forecast-tokens", type=int, default=720)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--input-mask-probability", type=float, default=0.15)
    parser.add_argument("--distribution-weight", type=float, default=1.0)
    parser.add_argument("--track-weight", type=float, default=1.0)
    parser.add_argument("--survival-weight", type=float, default=1.0)
    parser.add_argument("--distribution-samples", type=int, default=32)
    parser.add_argument("--process-noise-min-std-deg", type=float, default=0.25)
    parser.add_argument("--process-noise-max-std-deg", type=float, default=12.0)
    parser.add_argument("--initial-std-min-deg", type=float, default=0.25)
    parser.add_argument("--initial-std-max-deg", type=float, default=8.0)
    parser.add_argument("--max-initial-correction-deg", type=float, default=5.0)
    parser.add_argument("--max-daily-displacement-deg", type=float, default=15.0)
    parser.add_argument("--distribution-kernel-std-deg", type=float, default=5.0)
    parser.add_argument("--output", default="checkpoints/weathernext_transformer.pt")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    integrated = pd.read_parquet(args.integrated) if args.integrated.endswith(".parquet") else pd.read_csv(args.integrated)
    integrated["time"] = pd.to_datetime(integrated["time"])
    base = TyphoonPressureDataset(
        integrated, history=args.history, horizon=args.track_steps, max_highs=args.max_highs
    )
    model_config = SmallModelConfig(
        input_dim=len(base.feature_cols), history_steps=args.history,
        local_track_steps=args.track_steps,
    )
    lookup = None
    if args.distribution:
        lookup = SpatialDistributionLookup.from_csv(
            args.distribution, lat_bin_deg=model_config.lat_bin_deg, lon_bin_deg=model_config.lon_bin_deg
        )
    store = DirectoryForecastTokenStore(args.weathernext_token_dir)
    if not store.files:
        raise ValueError("WeatherNext token manifest is empty")
    first_key = next(iter(store.files))
    forecast_input_dim = store.load(*first_key).values.shape[1]
    gpt_state_store = None
    gpt_state_dim = 0
    if args.gpt_state_dir:
        from .gpt_state import DirectoryGPTStateStore

        gpt_state_store = DirectoryGPTStateStore(args.gpt_state_dir)
        coverage = gpt_state_store.validate_coverage(
            store.files.keys(), require_valid=args.require_valid_gpt_states
        )
        print({"gpt_router": "enabled", "gpt_cache_coverage": coverage})
        gpt_state_dim = gpt_state_store.state_dim
    else:
        print({"gpt_router": "disabled", "reason": "--gpt-state-dir not supplied"})
    transformer_config = TransformerConfig(
        forecast_input_dim=forecast_input_dim,
        gpt_state_dim=gpt_state_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        decoder_layers=args.decoder_layers,
        input_mask_probability=args.input_mask_probability,
    )
    sampling_config = DistributionSamplingConfig(
        num_samples=args.distribution_samples,
        min_process_std_deg=args.process_noise_min_std_deg,
        max_process_std_deg=args.process_noise_max_std_deg,
        min_initial_std_deg=args.initial_std_min_deg,
        max_initial_std_deg=args.initial_std_max_deg,
        max_initial_correction_deg=args.max_initial_correction_deg,
        max_daily_displacement_deg=args.max_daily_displacement_deg,
        grid_kernel_std_deg=args.distribution_kernel_std_deg,
    )
    dataset_kwargs = dict(
        base_dataset=base,
        distribution=lookup,
        model_config=model_config,
        forecast_store=store,
        max_forecast_tokens=args.max_forecast_tokens,
        forecast_input_dim=forecast_input_dim,
        require_endpoint_lead_hours=model_config.distribution_start_day * 24,
    )
    if gpt_state_store is not None:
        from .gpt_state import WeatherNextGPTDualTargetDataset

        dataset = WeatherNextGPTDualTargetDataset(
            **dataset_kwargs, gpt_state_store=gpt_state_store
        )
    else:
        dataset = WeatherNextDualTargetDataset(**dataset_kwargs)
    if len(dataset) == 0:
        raise ValueError("No base samples match the WeatherNext token manifest")
    split_manifest = load_storm_split(args.split_manifest)
    train_dataset = StormSplitSubset(dataset, split_manifest, "train")
    validation_dataset = StormSplitSubset(dataset, split_manifest, "validation")
    if train_dataset.storm_ids & validation_dataset.storm_ids:
        raise RuntimeError("Storm leakage detected between train and validation datasets")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print({"train_windows": len(train_dataset), "validation_windows": len(validation_dataset), "train_storms": len(train_dataset.storm_ids), "validation_storms": len(validation_dataset.storm_ids)})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeatherNextFusionTransformer(
        model_config, transformer_config, sampling_config=sampling_config
    ).to(device)
    loss_config = DualLossConfig(
        distribution_weight=args.distribution_weight,
        local_track_weight=args.track_weight,
        survival_weight=args.survival_weight,
    )
    criterion = DualObjectiveLoss(loss_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        validation_metrics = evaluate_epoch(model, validation_loader, criterion, device)
        print(f"epoch={epoch} " + " ".join(f"train_{key}={value:.5f}" for key, value in train_metrics.items()) + " " + " ".join(f"validation_{key}={value:.5f}" for key, value in validation_metrics.items()))
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(), "model_config": model_config.__dict__,
                "transformer_config": transformer_config.__dict__, "sampling_config": sampling_config.__dict__,
                "loss_config": loss_config.__dict__,
                "data_config": {"history": args.history, "track_steps": args.track_steps, "max_highs": args.max_highs, "max_forecast_tokens": args.max_forecast_tokens},
                "split_manifest": str(Path(args.split_manifest).resolve()),
                "train_storm_ids": sorted(train_dataset.storm_ids),
                "validation_storm_ids": sorted(validation_dataset.storm_ids),
                "epoch": epoch, "validation_metrics": validation_metrics, "seed": args.seed,
            }, output)
    print({"best_checkpoint": str(output), "best_epoch": best_epoch, "best_validation_loss": best_validation_loss})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
