from __future__ import annotations

import pandas as pd


def build_integrated_dataset(track: pd.DataFrame, highs: pd.DataFrame) -> pd.DataFrame:
    """Left-join IBTrACS states with zero or more surrounding ERA5 highs."""
    required_track = {"storm_id", "time", "typhoon_lat", "typhoon_lon", "typhoon_pressure_hpa"}
    required_highs = {"storm_id", "time", "high_rank", "high_lat", "high_lon", "high_pressure_hpa"}
    if missing := required_track.difference(track.columns):
        raise ValueError(f"Track columns missing: {sorted(missing)}")
    if missing := required_highs.difference(highs.columns):
        raise ValueError(f"High-pressure columns missing: {sorted(missing)}")
    merged = track.merge(highs, on=["storm_id", "time"], how="left", validate="one_to_many")
    merged["pressure_difference_hpa"] = merged.high_pressure_hpa - merged.typhoon_pressure_hpa
    return merged.sort_values(["storm_id", "time", "high_rank"], na_position="last").reset_index(drop=True)

