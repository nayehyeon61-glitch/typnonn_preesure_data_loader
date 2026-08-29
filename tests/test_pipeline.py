import numpy as np
import pandas as pd
import xarray as xr

from typhoon_pressure.dataset import TyphoonPressureDataset
from typhoon_pressure.era5 import HighPressureConfig, extract_surrounding_highs
from typhoon_pressure.merge import build_integrated_dataset


def test_typhoon_and_surrounding_high_are_joined():
    times = pd.date_range("2025-08-01", periods=5, freq="6h")
    track = pd.DataFrame({
        "storm_id": ["2025001N12000"] * 5, "name": ["TEST"] * 5,
        "basin": ["WP"] * 5, "time": times,
        "typhoon_lat": [20, 21, 22, 23, 24], "typhoon_lon": [130, 131, 132, 133, 134],
        "typhoon_pressure_hpa": [995, 990, 985, 980, 982],
        "typhoon_wind_kt": [35, 40, 50, 60, 55], "nature": ["TS"] * 5,
    })
    lat = np.linspace(0, 60, 61)
    lon = np.linspace(100, 170, 71)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    fields = []
    for step in range(5):
        high_lon = 145 + step
        high = 18 * np.exp(-((yy - 35) ** 2 + (xx - high_lon) ** 2) / 35)
        fields.append((1010 + high) * 100)
    mslp = xr.DataArray(np.stack(fields), dims=("time", "lat", "lon"),
                        coords={"time": times, "lat": lat, "lon": lon})
    highs = extract_surrounding_highs(track, mslp, HighPressureConfig(
        radius_km=3000, local_window=7, background_window=17,
        min_anomaly_hpa=1, min_separation_km=300, max_highs=1,
    ))
    assert len(highs) == 5
    assert (highs.high_rank == 1).all()
    integrated = build_integrated_dataset(track, highs)
    assert integrated.high_pressure_hpa.notna().all()
    dataset = TyphoonPressureDataset(integrated, history=3, horizon=2, max_highs=1)
    sample = dataset[0]
    assert tuple(sample["history"].shape) == (3, 8)
    assert tuple(sample["target"].shape) == (2, 4)

