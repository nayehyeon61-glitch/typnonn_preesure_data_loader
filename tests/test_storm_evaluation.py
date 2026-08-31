import math

import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from typhoon_pressure.evaluation.storm_split import StormSplitConfig, StormSplitSubset, build_storm_split, validate_storm_split
from typhoon_pressure.evaluation.weathernext import probabilistic_trajectory_metrics, survival_metrics


class StormDataset(Dataset):
    def __init__(self, storms): self.storms = list(storms)
    def __len__(self): return len(self.storms)
    def __getitem__(self, index): return {"storm_id": self.storms[index]}
    def storm_id_at(self, index): return self.storms[index]


def integrated_storms(count=10):
    return pd.DataFrame([
        {"storm_id": f"S{storm:02d}", "time": pd.Timestamp("2000-01-01") + pd.Timedelta(days=storm * 10, hours=6 * step)}
        for storm in range(count) for step in range(4) for _ in range(2)
    ])


def test_chronological_split_is_disjoint_and_deterministic():
    frame = integrated_storms()
    first = build_storm_split(frame, StormSplitConfig(0.6, 0.2, 0.2))
    second = build_storm_split(frame.sample(frac=1, random_state=8), StormSplitConfig(0.6, 0.2, 0.2))
    pd.testing.assert_frame_equal(first, second)
    groups = first.groupby("split")["storm_id"].apply(set).to_dict()
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    assert set(first.observation_count) == {4}


def test_subset_filters_and_rejects_leakage_contract_errors():
    manifest = build_storm_split(integrated_storms(6), StormSplitConfig(0.5, 1 / 6, 1 / 3))
    subset = StormSplitSubset(StormDataset(["S00", "S00", "S01", "S03", "S05"]), manifest, "train")
    assert {subset.storm_id_at(i) for i in range(len(subset))} == {"S00", "S01"}
    with pytest.raises(ValueError, match="missing from the split manifest"):
        StormSplitSubset(StormDataset(["UNKNOWN"]), manifest, "test")
    with pytest.raises(ValueError, match="exactly once"):
        validate_storm_split(pd.concat((manifest, manifest.iloc[[0]]), ignore_index=True))


def test_probabilistic_metrics_and_survival_scores():
    mean = torch.tensor([[[20.0, 130.0], [21.0, 131.0]]])
    covariance = torch.eye(2).reshape(1, 1, 2, 2).repeat(1, 2, 1, 1)
    samples = mean[:, None].repeat(1, 4, 1, 1)
    metrics = probabilistic_trajectory_metrics(mean, covariance, samples, mean.clone())
    assert torch.allclose(metrics["position_error_km"], torch.zeros(1, 2))
    assert torch.allclose(metrics["energy_score_km"], torch.zeros(1, 2))
    assert torch.allclose(metrics["gaussian_nll"], torch.full((1, 2), math.log(2 * math.pi) + math.log(1.0001)), atol=1e-5)
    assert torch.all(metrics["coverage_50"] == 1)
    survival = survival_metrics(torch.tensor([[0.8, 0.2]]), torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(survival["survival_brier"], torch.tensor([[0.04, 0.04]]))
