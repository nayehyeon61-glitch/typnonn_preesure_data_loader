from __future__ import annotations

import argparse

import xarray as xr

from .era5 import HighPressureConfig, extract_surrounding_highs, normalize_era5_mslp
from .ibtracs import IBTrACSConfig, load_ibtracs
from .merge import build_integrated_dataset


def main():
    parser = argparse.ArgumentParser(description="Join IBTrACS tracks with ERA5 surrounding highs")
    parser.add_argument("--ibtracs", required=True)
    parser.add_argument("--era5", nargs="+", required=True)
    parser.add_argument("--output", default="integrated_typhoon_pressure.parquet")
    parser.add_argument("--basin", default="WP")
    parser.add_argument("--agency", default="TOKYO")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--radius-km", type=float, default=2500)
    parser.add_argument("--max-highs", type=int, default=3)
    args = parser.parse_args()

    track = load_ibtracs(args.ibtracs, IBTrACSConfig(
        basin=args.basin, agency=args.agency, start=args.start, end=args.end
    ))
    ds = xr.open_mfdataset(args.era5, combine="by_coords", chunks={"time": 24})
    highs = extract_surrounding_highs(
        track, normalize_era5_mslp(ds),
        HighPressureConfig(radius_km=args.radius_km, max_highs=args.max_highs),
    )
    integrated = build_integrated_dataset(track, highs)
    integrated.to_parquet(args.output, index=False)
    print(f"saved {len(integrated)} rows for {integrated.storm_id.nunique()} storms to {args.output}")


if __name__ == "__main__":
    main()

