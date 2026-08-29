# Typhoon Pressure Data Loader

IBTrACS의 태풍 best-track과 ERA5 해면기압장(MSLP)을 결합하여, 태풍과 주변 고기압의 위치·기압 관계를 학습할 수 있는 통합 데이터셋을 만듭니다.

## 데이터 흐름

```text
IBTrACS CSV/NetCDF                    ERA5 NetCDF
storm_id, time, x/y, pressure        time, lat, lon, MSLP
          |                                  |
          +------------ time join -----------+
                             |
                 태풍 반경 R km의 고기압 검출
                             |
          typhoon state + surrounding high states
                             |
                   PyTorch sequence Dataset
```

## 1. IBTrACS trajectory loader

`load_ibtracs()`는 기관별 결측값을 고려해 선택한 기관(`TOKYO` 기본), WMO, USA 순서로 중심기압과 풍속을 보완합니다.

```python
from typhoon_pressure import IBTrACSConfig, load_ibtracs

tracks = load_ibtracs(
    "IBTrACS.ALL.v04r01.csv",
    IBTrACSConfig(basin="WP", agency="TOKYO", start="2020-01-01"),
)
```

출력: `storm_id, name, time, typhoon_lat, typhoon_lon, typhoon_pressure_hpa, typhoon_wind_kt`

## 2. ERA5 주변 고기압 추출

각 태풍 시각과 가장 가까운 ERA5 시각을 찾고, 기본 반경 2,500 km 안에서 MSLP 국소 최대이면서 주변 평균보다 1.5 hPa 이상 높은 중심을 찾습니다. 태풍 중심 250 km 안은 제외하고 고기압끼리 최소 400 km를 유지합니다.

```python
import xarray as xr
from typhoon_pressure import HighPressureConfig, extract_surrounding_highs

era5 = xr.open_mfdataset("era5_*.nc", combine="by_coords")
highs = extract_surrounding_highs(
    tracks, era5["msl"], HighPressureConfig(radius_km=2500, max_highs=3)
)
```

각 고기압은 절대 위치뿐 아니라 태풍 기준 `high_dx_km`, `high_dy_km`, 거리와 방위각을 가집니다.

## 3. 통합 및 DataLoader

```python
from torch.utils.data import DataLoader
from typhoon_pressure import build_integrated_dataset, TyphoonPressureDataset

integrated = build_integrated_dataset(tracks, highs)
dataset = TyphoonPressureDataset(integrated, history=8, horizon=20, max_highs=3)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

- `history=8`: 6시간 간격 기준 과거 48시간
- `horizon=20`: 미래 120시간
- `history`: 태풍 4개 feature + 고기압당 4개 feature
- `target`: 미래 태풍 `[lat, lon, pressure_hPa, wind_kt]`
- `history_mask`, `target_mask`: IBTrACS 결측값과 검출되지 않은 고기압 구분

## 4. WeatherNext 2 초기조건 보정 API

WeatherNext 2의 전체 초기 대기장은 HRES/ERA5에서 가져오고, IBTrACS는 태풍 중심의 tracker seed와 선택적 vortex 보정 자료로 사용합니다.

```python
from typhoon_pressure import (
    CorrectionConfig,
    InitialConditionBuilder,
    StormObservation,
    make_weathernext_request,
)

storm = StormObservation.from_series(tracks.iloc[0])

builder = InitialConditionBuilder(
    mode="auto",
    config=CorrectionConfig(
        position_threshold_km=100,
        pressure_threshold_hpa=5,
        search_radius_km=500,
        correction_radius_km=400,
    ),
)

condition = builder.build(
    atmospheric_state=hres_or_era5_state,
    storm=storm,
)

request = make_weathernext_request(
    condition,
    horizon_hours=360,
)
```

지원 모드:

| mode | 동작 |
|---|---|
| `tracker_seed` | 전체 대기장을 수정하지 않고 IBTrACS를 cyclone tracker seed로만 사용 |
| `vortex_correction` | IBTrACS 위치·기압을 기준으로 MSLP vortex를 이동·보정 |
| `auto` | 위치 오차 100 km 또는 중심기압 오차 5 hPa 초과 시에만 보정 |

`WeatherNextRequest`는 `initial_state`, `tracker_seed`, `horizon_hours`, 보정 metadata를 분리합니다. 실제 Google WN2 호출은 버전을 고정한 JAX/TPU runner에서 수행하도록 `WeatherNextRunner` 경계로 분리했습니다.

> 현재 vortex correction은 실험적인 MSLP-only 보정입니다. WN2 성능 실험에서는 `tracker_seed`를 baseline으로 두고, MSLP 보정 후 첫 rollout에서 바람·온도·습도장의 동역학적 균형이 유지되는지 반드시 비교해야 합니다.

## CLI

```bash
pip install -e '.[io,test]'

build-typhoon-pressure-data \
  --ibtracs IBTrACS.ALL.v04r01.csv \
  --era5 era5/2025_*.nc \
  --basin WP --agency TOKYO \
  --radius-km 2500 --max-highs 3 \
  --output integrated_typhoon_pressure.parquet

pytest
```

## 중요한 설계 조건

- IBTrACS CSV의 두 번째 행은 단위 행이므로 자동으로 건너뜁니다.
- 풍속은 기관마다 1분/10분 평균 정의가 다르므로 한 기관을 일관되게 선택해야 합니다.
- ERA5 고기압은 관측 label이 아니라 MSLP에서 계산한 파생 feature입니다.
- train/validation/test는 row가 아니라 `storm_id` 단위로 분리해야 같은 태풍이 양쪽에 포함되는 leakage를 막을 수 있습니다.
- 경도 180° 경계는 wrapped longitude 차이로 처리합니다.
