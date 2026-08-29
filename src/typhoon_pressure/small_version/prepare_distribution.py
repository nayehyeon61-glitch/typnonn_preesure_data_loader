from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build IBTrACS distribution labels with typnoon-disribution")
    parser.add_argument("--ibtracs", required=True, help="IBTrACS CSV path or URL")
    parser.add_argument("--output-dir", default="data/distribution")
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--basins", nargs="*")
    parser.add_argument("--lat-bin-deg", type=float, default=5.0)
    parser.add_argument("--lon-bin-deg", type=float, default=5.0)
    args = parser.parse_args(argv)

    try:
        from typhoon_distribution import PipelineConfig, run_pipeline
    except ImportError as exc:
        raise SystemExit("Install the connection dependency with: pip install -e '.[small]'") from exc

    summary = run_pipeline(PipelineConfig(
        source=args.ibtracs,
        output_dir=args.output_dir,
        basins=tuple(args.basins) if args.basins else None,
        start_year=args.start_year,
        end_year=args.end_year,
        lat_bin_deg=args.lat_bin_deg,
        lon_bin_deg=args.lon_bin_deg,
        time_coordinate="calendar_month",
    ))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

