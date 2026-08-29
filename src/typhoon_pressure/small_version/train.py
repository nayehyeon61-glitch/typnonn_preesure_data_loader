from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from typhoon_pressure.dataset import TyphoonPressureDataset

from .config import DualLossConfig, SmallModelConfig
from .dataset import DualTargetDataset, SpatialDistributionLookup
from .losses import DualObjectiveLoss
from .model import SmallDualScaleModel


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0


def train_epoch(model, loader, optimizer, criterion, device: torch.device) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    examples = 0
    for batch in loader:
        tensors = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        if "forecast_values" in tensors:
            outputs = model(
                tensors["history"],
                tensors["history_mask"],
                tensors["forecast_values"],
                tensors["forecast_feature_mask"],
                tensors["forecast_token_mask"],
                tensors["forecast_positions"],
            )
        else:
            outputs = model(tensors["history"], tensors["history_mask"])
        losses = criterion(outputs, tensors)
        for metric in (
            "effective_forecast_token_fraction",
            "effective_forecast_feature_fraction",
        ):
            if metric in outputs:
                losses[metric] = outputs[metric]
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = tensors["history"].shape[0]
        examples += batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_size
    return {key: value / max(examples, 1) for key, value in totals.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the small dual-loss typhoon model")
    parser.add_argument("--integrated", required=True, help="Integrated pressure/track parquet or CSV")
    parser.add_argument("--distribution", required=True, help="spatial_distribution.csv from typnoon-disribution")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--track-steps", type=int, default=20)
    parser.add_argument("--max-highs", type=int, default=3)
    parser.add_argument("--output", default="small_dual_scale_model.pt")
    parser.add_argument("--distribution-weight", type=float, default=1.0)
    parser.add_argument("--track-weight", type=float, default=1.0)
    args = parser.parse_args(argv)

    integrated = pd.read_parquet(args.integrated) if args.integrated.endswith(".parquet") else pd.read_csv(args.integrated)
    integrated["time"] = pd.to_datetime(integrated["time"])
    base = TyphoonPressureDataset(integrated, history=args.history, horizon=args.track_steps, max_highs=args.max_highs)
    config = SmallModelConfig(
        input_dim=len(base.feature_cols), history_steps=args.history, local_track_steps=args.track_steps
    )
    lookup = SpatialDistributionLookup.from_csv(
        args.distribution, lat_bin_deg=config.lat_bin_deg, lon_bin_deg=config.lon_bin_deg
    )
    dataset = DualTargetDataset(base, lookup, config)
    if len(dataset) == 0:
        raise ValueError("No valid windows: reduce --history/--track-steps or check storm continuity")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallDualScaleModel(config).to(device)
    criterion = DualObjectiveLoss(DualLossConfig(
        distribution_weight=args.distribution_weight, local_track_weight=args.track_weight
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, loader, optimizer, criterion, device)
        print(f"epoch={epoch} " + " ".join(f"{key}={value:.5f}" for key, value in metrics.items()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.__dict__}, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
