from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TyphoonPressureDataset(Dataset):
    """Sequence windows containing a typhoon and K surrounding high centres.

    State feature: [typhoon lat, lon, pressure, wind,
    high_1 dx, dy, pressure, anomaly, ..., high_K ...].
    Target: future [typhoon lat, lon, pressure, wind].
    """

    def __init__(self, integrated: pd.DataFrame, history: int, horizon: int, max_highs: int = 3):
        if min(history, horizon, max_highs) < 1:
            raise ValueError("history, horizon and max_highs must be positive")
        self.history, self.horizon, self.max_highs = history, horizon, max_highs
        index_cols = ["storm_id", "time"]
        typhoon_cols = ["typhoon_lat", "typhoon_lon", "typhoon_pressure_hpa", "typhoon_wind_kt"]
        base = integrated[index_cols + typhoon_cols].drop_duplicates(index_cols)
        high_values = ["high_dx_km", "high_dy_km", "high_pressure_hpa", "high_anomaly_hpa"]
        highs = integrated.dropna(subset=["high_rank"])
        highs = highs[highs.high_rank <= max_highs]
        wide = highs.pivot(index=index_cols, columns="high_rank", values=high_values)
        wide.columns = [f"{name}_{int(rank)}" for name, rank in wide.columns]
        self.feature_cols = typhoon_cols + [
            f"{name}_{rank}" for rank in range(1, max_highs + 1) for name in high_values
        ]
        frame = base.merge(wide.reset_index(), on=index_cols, how="left")
        for col in self.feature_cols:
            if col not in frame:
                frame[col] = np.nan
        self.groups = {
            str(storm_id): group.sort_values("time").reset_index(drop=True)
            for storm_id, group in frame.groupby("storm_id")
        }
        total = history + horizon
        self.windows = [
            (storm_id, start)
            for storm_id, group in self.groups.items()
            for start in range(len(group) - total + 1)
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        storm_id, start = self.windows[index]
        frame = self.groups[storm_id].iloc[start : start + self.history + self.horizon]
        raw = frame[self.feature_cols].to_numpy(np.float32)
        mask = np.isfinite(raw).astype(np.float32)
        values = np.nan_to_num(raw, nan=0.0)
        history = torch.from_numpy(values[: self.history])
        history_mask = torch.from_numpy(mask[: self.history])
        target = torch.from_numpy(values[self.history :, :4])
        target_mask = torch.from_numpy(mask[self.history :, :4])
        return {
            "history": history, "history_mask": history_mask,
            "target": target, "target_mask": target_mask,
            "storm_id": storm_id,
        }

