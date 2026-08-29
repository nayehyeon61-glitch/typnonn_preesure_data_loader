import numpy as np
import pandas as pd
import xarray as xr

from typhoon_pressure.dataset import TyphoonPressureDataset
from typhoon_pressure.era5 import HighPressureConfig, extract_surrounding_highs
from typhoon_pressure.merge import build_integrated_dataset
from typhoon_pressure.initial_condition import (
    CorrectionConfig,
    InitialConditionBuilder,
    StormObservation,
)
from typhoon_pressure.weathernext_adapter import make_weathernext_request


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


def test_auto_initial_condition_corrects_displaced_vortex():
    lat = np.linspace(10, 30, 41)
    lon = np.linspace(120, 150, 61)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    # Model vortex is centred at (20, 130), but IBTrACS locates it at (22, 133).
    model_mslp = 1010 - 25 * np.exp(-((yy - 20) ** 2 + (xx - 130) ** 2) / 8)
    state = xr.Dataset(
        {"msl": (("time", "latitude", "longitude"), model_mslp[None] * 100)},
        coords={
            "time": [pd.Timestamp("2025-08-01T00:00:00")],
            "latitude": lat,
            "longitude": lon,
        },
    )
    storm = StormObservation(
        storm_id="TEST", time=pd.Timestamp("2025-08-01T00:00:00"),
        lat=22, lon=133, pressure_hpa=975, wind_kt=70,
    )
    builder = InitialConditionBuilder(
        mode="auto",
        config=CorrectionConfig(
            position_threshold_km=100,
            pressure_threshold_hpa=5,
            search_radius_km=600,
        ),
    )
    condition = builder.build(state, storm)
    assert condition.correction_applied
    assert condition.applied_mode == "vortex_correction"
    assert condition.position_error_km > 100
    request = make_weathernext_request(condition, horizon_hours=360)
    assert request.tracker_seed["storm_id"] == "TEST"
    assert request.horizon_hours == 360


def test_initial_condition_can_preserve_two_weather_history_steps():
    times = pd.date_range("2025-07-31T18:00:00", periods=2, freq="6h")
    lat = np.linspace(10, 30, 21)
    lon = np.linspace(120, 150, 31)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    field = 1010 - 20 * np.exp(-((yy - 20) ** 2 + (xx - 130) ** 2) / 8)
    state = xr.Dataset(
        {"msl": (("time", "latitude", "longitude"), np.stack((field, field)) * 100)},
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    storm = StormObservation(
        storm_id="TEST",
        time=times[-1],
        lat=20,
        lon=130,
        pressure_hpa=990,
    )
    condition = InitialConditionBuilder(
        mode="tracker_seed",
        history_steps=2,
    ).build(state, storm)
    assert condition.atmospheric_state.sizes["time"] == 2
    assert condition.metadata()["input_history_steps"] == 2
    assert pd.Timedelta(
        condition.atmospheric_state.time.values[-1]
        - condition.atmospheric_state.time.values[-2]
    ) == pd.Timedelta("6h")
