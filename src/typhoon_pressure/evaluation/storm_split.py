"""Leakage-safe storm-level dataset splits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class StormSplitConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    def __post_init__(self):
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value <= 0 for value in fractions):
            raise ValueError("All split fractions must be positive")
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError("Split fractions must sum to 1")


def _read_table(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def validate_storm_split(manifest):
    required = {"storm_id", "split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Storm split manifest is missing columns: {sorted(missing)}")
    result = manifest.copy()
    result["storm_id"] = result["storm_id"].astype(str).str.strip()
    result["split"] = result["split"].astype(str).str.strip().str.lower()
    invalid = sorted(set(result["split"]).difference(SPLIT_NAMES))
    if invalid:
        raise ValueError(f"Unknown storm split labels: {invalid}")
    duplicated = result.loc[result["storm_id"].duplicated(keep=False), "storm_id"].unique()
    if len(duplicated):
        raise ValueError("Each storm_id must occur exactly once in the split manifest; duplicates: " + ", ".join(map(str, duplicated[:10])))
    return result.reset_index(drop=True)


def build_storm_split(integrated, config=StormSplitConfig()):
    required = {"storm_id", "time"}
    missing = required.difference(integrated.columns)
    if missing:
        raise ValueError(f"Integrated table is missing columns: {sorted(missing)}")
    frame = integrated.loc[:, ["storm_id", "time"]].copy()
    frame["storm_id"] = frame["storm_id"].astype(str).str.strip()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True).dt.tz_localize(None)
    frame = frame.dropna(subset=["time"]).drop_duplicates(["storm_id", "time"])
    storms = (
        frame.groupby("storm_id", sort=False)["time"]
        .agg(start_time="min", end_time="max", observation_count="size")
        .reset_index().sort_values(["start_time", "storm_id"], kind="stable").reset_index(drop=True)
    )
    count = len(storms)
    if count < 3:
        raise ValueError("At least three storms are required for train/validation/test splits")
    train_count = max(1, int(count * config.train_fraction))
    validation_count = max(1, int(count * config.validation_fraction))
    while train_count + validation_count >= count:
        if train_count > validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            raise ValueError("Unable to allocate a non-empty test split")
    storms["split"] = ["train"] * train_count + ["validation"] * validation_count + ["test"] * (count - train_count - validation_count)
    storms["split_order"] = range(count)
    return validate_storm_split(storms)


def load_storm_split(path):
    return validate_storm_split(_read_table(path))


class StormSplitSubset(Dataset):
    def __init__(self, dataset, manifest, split):
        normalized = validate_storm_split(manifest)
        split = str(split).lower()
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")
        allowed = set(normalized.loc[normalized["split"] == split, "storm_id"])
        if not allowed:
            raise ValueError(f"Storm split {split!r} is empty")
        if not hasattr(dataset, "storm_id_at"):
            raise TypeError("Dataset must implement storm_id_at(index) for leakage-safe splitting")
        available = {str(dataset.storm_id_at(index)) for index in range(len(dataset))}
        unassigned = available.difference(set(normalized["storm_id"]))
        if unassigned:
            raise ValueError("Dataset contains storms missing from the split manifest: " + ", ".join(sorted(unassigned)[:10]))
        selected = allowed.intersection(available)
        self.dataset = dataset
        self.split = split
        self.storm_ids = frozenset(selected)
        self.indices = [index for index in range(len(dataset)) if str(dataset.storm_id_at(index)) in selected]
        if not self.indices:
            raise ValueError(f"No dataset windows remain for split {split!r}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def storm_id_at(self, index):
        return str(self.dataset.storm_id_at(self.indices[index]))


def write_storm_split(manifest, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = validate_storm_split(manifest)
    if path.suffix.lower() == ".parquet":
        normalized.to_parquet(path, index=False)
    else:
        normalized.to_csv(path, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a chronological, leakage-safe storm-level split manifest")
    parser.add_argument("--integrated", required=True)
    parser.add_argument("--output", default="data/storm_split.csv")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args(argv)
    manifest = build_storm_split(_read_table(args.integrated), StormSplitConfig(args.train_fraction, args.validation_fraction, args.test_fraction))
    write_storm_split(manifest, args.output)
    summary = manifest.groupby("split")["storm_id"].count().reindex(SPLIT_NAMES, fill_value=0)
    print({"output": str(args.output), "storms": summary.to_dict()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
