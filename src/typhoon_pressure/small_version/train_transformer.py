from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from typhoon_pressure.dataset import TyphoonPressureDataset

from .config import DualLossConfig, SmallModelConfig, TransformerConfig
from .dataset import SpatialDistributionLookup, WeatherNextDualTargetDataset
from .losses import DualObjectiveLoss
from .model import WeatherNextFusionTransformer
from .train import train_epoch
from .weathernext_bridge import DirectoryForecastTokenStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train 15–30 day predictions from masked WeatherNext Transformer inputs"
    )
    parser.add_argument("--integrated", required=True)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--weathernext-token-dir", required=True)
    parser.add_argument("--gpt-state-dir", help="Optional cache produced by build-gpt-state-cache")
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
    parser.add_argument(
        "--require-checkpoint-kind",
        choices=["official_pretrained", "fine_tuned", "pretrained_unknown"],
        help="Fail unless the WeatherNext token manifest records this checkpoint kind",
    )
    parser.add_argument("--output", default="checkpoints/weathernext_transformer.pt")
    args = parser.parse_args(argv)

    integrated = pd.read_parquet(args.integrated) if args.integrated.endswith(".parquet") else pd.read_csv(args.integrated)
    integrated["time"] = pd.to_datetime(integrated["time"])
    base = TyphoonPressureDataset(
        integrated, history=args.history, horizon=args.track_steps, max_highs=args.max_highs
    )
    model_config = SmallModelConfig(
        input_dim=len(base.feature_cols), history_steps=args.history,
        local_track_steps=args.track_steps,
    )
    lookup = SpatialDistributionLookup.from_csv(
        args.distribution,
        lat_bin_deg=model_config.lat_bin_deg,
        lon_bin_deg=model_config.lon_bin_deg,
    )
    store = DirectoryForecastTokenStore(args.weathernext_token_dir)
    if not store.files:
        raise ValueError("WeatherNext token manifest is empty")
    if args.require_checkpoint_kind:
        store.require_checkpoint_kind(args.require_checkpoint_kind)
    weathernext_provenance = store.provenance()
    first_key = next(iter(store.files))
    forecast_input_dim = store.load(*first_key).values.shape[1]
    gpt_state_store = None
    gpt_state_dim = 0
    if args.gpt_state_dir:
        from .gpt_state import DirectoryGPTStateStore

        gpt_state_store = DirectoryGPTStateStore(args.gpt_state_dir)
        gpt_state_dim = gpt_state_store.state_dim
    transformer_config = TransformerConfig(
        forecast_input_dim=forecast_input_dim,
        gpt_state_dim=gpt_state_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        decoder_layers=args.decoder_layers,
        input_mask_probability=args.input_mask_probability,
    )
    dataset_kwargs = {
        "base_dataset": base,
        "distribution": lookup,
        "model_config": model_config,
        "forecast_store": store,
        "max_forecast_tokens": args.max_forecast_tokens,
        "forecast_input_dim": forecast_input_dim,
    }
    if gpt_state_store is not None:
        from .gpt_state import WeatherNextGPTDualTargetDataset

        dataset = WeatherNextGPTDualTargetDataset(
            **dataset_kwargs, gpt_state_store=gpt_state_store
        )
    else:
        dataset = WeatherNextDualTargetDataset(**dataset_kwargs)
    if len(dataset) == 0:
        raise ValueError("No base samples match the WeatherNext token manifest")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeatherNextFusionTransformer(model_config, transformer_config).to(device)
    criterion = DualObjectiveLoss(DualLossConfig(
        distribution_weight=args.distribution_weight,
        local_track_weight=args.track_weight,
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, loader, optimizer, criterion, device)
        print(f"epoch={epoch} " + " ".join(f"{key}={value:.5f}" for key, value in metrics.items()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_config": model_config.__dict__,
        "transformer_config": transformer_config.__dict__,
        "weathernext_provenance": weathernext_provenance,
    }, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
