from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class IBTrACSConfig:
    basin: str | None = "WP"
    agency: str = "TOKYO"
    start: str | None = None
    end: str | None = None
    named_only: bool = False


def _read_source(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".nc", ".nc4"}:
        ds = xr.open_dataset(path)
        frame = ds.to_dataframe().reset_index()
        for col in frame.select_dtypes(include=[object]).columns:
            frame[col] = frame[col].map(
                lambda value: value.decode().strip() if isinstance(value, bytes) else value
            )
        return frame
    return pd.read_csv(path, low_memory=False, skiprows=[1])


def _numeric(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    available = [name for name in candidates if name in frame]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    result = pd.to_numeric(frame[available[0]], errors="coerce")
    for name in available[1:]:
        result = result.fillna(pd.to_numeric(frame[name], errors="coerce"))
    return result


def load_ibtracs(path: str | Path, config: IBTrACSConfig = IBTrACSConfig()) -> pd.DataFrame:
    """Load IBTrACS CSV/NetCDF into one row per storm observation.

    Pressure/wind prefer the selected agency, then WMO, then USA values.
    Output wind remains in knots and pressure in hPa.
    """
    raw = _read_source(path)
    raw.columns = [str(c).upper() for c in raw.columns]
    agency = config.agency.upper()
    required = {"SID", "ISO_TIME", "LAT", "LON"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"IBTrACS source is missing columns: {sorted(missing)}")

    out = pd.DataFrame({
        "storm_id": raw["SID"].astype(str).str.strip(),
        "name": raw.get("NAME", pd.Series("NOT_NAMED", index=raw.index)).astype(str).str.strip(),
        "basin": raw.get("BASIN", pd.Series("", index=raw.index)).astype(str).str.strip(),
        "time": pd.to_datetime(raw["ISO_TIME"], errors="coerce", utc=True).dt.tz_localize(None),
        "typhoon_lat": pd.to_numeric(raw["LAT"], errors="coerce"),
        "typhoon_lon": pd.to_numeric(raw["LON"], errors="coerce"),
        "typhoon_pressure_hpa": _numeric(raw, [f"{agency}_PRES", "WMO_PRES", "USA_PRES"]),
        "typhoon_wind_kt": _numeric(raw, [f"{agency}_WIND", "WMO_WIND", "USA_WIND"]),
        "nature": raw.get("NATURE", pd.Series("", index=raw.index)).astype(str).str.strip(),
    })
    out = out.dropna(subset=["storm_id", "time", "typhoon_lat", "typhoon_lon"])
    if config.basin:
        out = out[out.basin == config.basin.upper()]
    if config.start:
        out = out[out.time >= pd.Timestamp(config.start)]
    if config.end:
        out = out[out.time <= pd.Timestamp(config.end)]
    if config.named_only:
        out = out[~out.name.isin(["", "NAN", "NOT_NAMED", "UNNAMED"])]
    return out.sort_values(["storm_id", "time"]).drop_duplicates(["storm_id", "time"]).reset_index(drop=True)

