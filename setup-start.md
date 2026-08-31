# Setup Start

이 문서는 현재 WeatherNext + IBTrACS + GPT/Fusion 학습 시스템을 처음 실행할 때 필요한 순서를 정리합니다.

전체 흐름은 다음 5단계입니다.

```text
1. IBTrACS + ERA5 통합 데이터 생성
        ↓
2. Distribution target 생성
        ↓
3. WeatherNext 입력 병합·검증 → 실행 mode 선택 → token cache 생성
        ↓
4. GPT state cache 생성 (선택)
        ↓
5. WeatherNextFusionTransformer 학습
```

기본 `--execution-mode pretrained`에서는 WeatherNext가 frozen inference로 동작합니다. `api`는 원격 inference이며 역시 frozen입니다. WeatherNext 자체 학습은 명시적인 `trainable` mode와 사용자 제공 factory가 있을 때만 별도로 실행됩니다. 후단 dual-loss optimizer는 항상 5단계의 WeatherNextFusionTransformer를 학습합니다.

---

## 0. 환경 준비

현재 resolver/pipeline 구현은 `feature/weathernext-resolver` 브랜치에 있습니다.

```bash
git checkout feature/weathernext-resolver
pip install -e ".[io,small,gpt,weathernext,test]"
```

WeatherNext를 GPU에서 실행할 경우 현재 CUDA 환경에 맞는 JAX GPU 패키지도 별도로 설치해야 합니다.

---

# Step 1. IBTrACS + ERA5 통합 데이터 생성

IBTrACS의 태풍 trajectory와 ERA5의 주변 고기압 정보를 결합합니다.

```bash
build-typhoon-pressure-data \
  --ibtracs data/IBTrACS.ALL.v04r01.csv \
  --era5 data/era5_*.nc \
  --output data/integrated_typhoon_pressure.parquet \
  --basin WP \
  --agency TOKYO \
  --radius-km 2500 \
  --max-highs 3
```

출력 예시:

```text
data/
└── integrated_typhoon_pressure.parquet
```

이 파일은 이후 GPT state 생성과 Fusion Transformer 학습에 공통으로 사용됩니다.

---

# Step 2. Distribution target 생성

후단 모델의 dual objective 중 distribution loss를 위한 target을 생성합니다.

```bash
build-typhoon-distribution-targets \
  --ibtracs data/IBTrACS.ALL.v04r01.csv \
  --output-dir data/distribution \
  --start-year 1980 \
  --basins WP \
  --lat-bin-deg 5 \
  --lon-bin-deg 5
```

후단 학습에는 생성된 distribution CSV를 사용합니다.

예시:

```text
data/distribution/spatial_distribution.csv
```

Dual loss는 개념적으로 다음과 같습니다.

```text
L_total
  = distribution_weight * L_distribution
  + track_weight        * L_track
```

---

# Step 3. WeatherNext 입력 준비 + 전체 token cache 생성

이 단계가 WeatherNext resolver가 실제 pipeline에 연결되는 부분입니다.

실행 mode:

```text
pretrained: fine-tuned/local official/download checkpoint를 선택하고 frozen inference
api: --api-client-factory로 주입한 원격 forecast client 사용
trainable: --trainable-factory가 반환한 WeatherNext model/data로 fit 후 rollout
auto: 기존 fine-tuned → official → download → API fallback 우선순위
```

`pretrained`가 기본값이므로 공식 checkpoint를 지정하는 것만으로 WeatherNext가 학습되는 일은 없습니다.

전체 흐름:

```text
ERA5 / HRES + supplemental fields
        +
IBTrACS storm state
        ↓
WeatherNextInputPreparer
        ↓
InitialConditionBuilder
        ↓
WeatherNext Resolver
        ↓
Selected WeatherNext execution
        ↓
Rollout
        ↓
Forecast NetCDF
        ↓
WeatherNext tokenizer
        ↓
Token .npz + manifest.csv
```

## 3-0. 공식 입력 계약 해결

CLI는 기본적으로 입력 준비를 수행합니다. HRES/ERA5 source의 alias와 상대시간을 정규화하고, `t-6h`와 `t` 두 장을 선택한 뒤 다음 계약을 검사합니다.

| 항목 | WeatherNext2 계약 |
|---|---|
| 시간 | 6시간 간격 2개 시점 |
| pressure level | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa |
| grid | 전지구 0.25°, 721×1440 |
| 3D 변수 | temperature, geopotential, u/v wind, vertical velocity, specific humidity |
| surface/static | 2m temperature, MSLP, 10m u/v, SST, surface geopotential, land-sea mask |
| WN2 전용 | 100m u/v wind |

한 source에 없는 SST·정적장·100 m 바람은 반복 가능한 `--supplement-state`로 병합합니다. 결측 물리변수를 임의 생성하지 않으며, 끝까지 없으면 실행 전에 오류가 발생합니다.

```bash
prepare-weathernext-pipeline \
  --atmospheric-state data/hres_or_era5.zarr \
  --supplement-state data/era5_sst_static.zarr \
  --supplement-state data/hres_100m_wind.zarr \
  --jobs data/weathernext_jobs.csv \
  --execution-mode pretrained \
  --pretrained-checkpoint download/weathernext2.npz
```

정규 전지구 grid이지만 해상도만 다를 때는 `--regrid`를 명시할 수 있습니다. 지역 grid 또는 불완전한 source를 전지구 자료처럼 외삽하는 용도는 아닙니다. 100 m 바람을 준비하기 어렵고 태풍용 checkpoint를 사용할 경우 `--model-variant WeatherNextCyclones`를 선택하면 해당 두 변수는 요구하지 않습니다.

## 3-A. Fine-tuned weight가 있는 경우

```bash
prepare-weathernext-pipeline \
  --atmospheric-state data/weather_initial_state.nc \
  --storm-id TEST \
  --init-time 2025-08-01T00:00:00 \
  --lat 22.0 \
  --lon 133.0 \
  --pressure-hpa 975 \
  --wind-kt 70 \
  --execution-mode pretrained \
  --finetuned-checkpoint checkpoints/korea_finetuned.npz \
  --pretrained-checkpoint download/weathernext2.npz \
  --horizon-hours 360 \
  --initialization-mode auto \
  --forecast-dir data/weathernext_forecasts \
  --token-dir data/weathernext_tokens
```

`checkpoints/korea_finetuned.npz`가 존재하면 pretrained checkpoint보다 우선 사용됩니다.

## 3-B. Fine-tuned weight가 없고 pretrained weight를 사용할 경우

```bash
prepare-weathernext-pipeline \
  --atmospheric-state data/weather_initial_state.nc \
  --storm-id TEST \
  --init-time 2025-08-01T00:00:00 \
  --lat 22.0 \
  --lon 133.0 \
  --pressure-hpa 975 \
  --wind-kt 70 \
  --execution-mode pretrained \
  --pretrained-checkpoint download/weathernext2.npz \
  --horizon-hours 360 \
  --forecast-dir data/weathernext_forecasts \
  --token-dir data/weathernext_tokens
```

## 3-C. Local checkpoint가 없고 online checkpoint를 받을 경우

Google DeepMind가 공개한 `WeatherNext2_<2025` model 1 checkpoint를 사용하는 경우:

```bash
prepare-weathernext-pipeline \
  --atmospheric-state data/weather_initial_state.nc \
  --storm-id TEST \
  --init-time 2025-08-01T00:00:00 \
  --lat 22.0 \
  --lon 133.0 \
  --pressure-hpa 975 \
  --wind-kt 70 \
  --execution-mode pretrained \
  --checkpoint-url "https://storage.googleapis.com/dm_graphcast/weathernext2/params/WeatherNext2_%3C2025_model1.npz" \
  --download-dir download/weathernext \
  --horizon-hours 360 \
  --forecast-dir data/weathernext_forecasts \
  --token-dir data/weathernext_tokens
```

다운로드된 checkpoint는 cache되므로 이후 실행에서는 local weight처럼 재사용됩니다. URL의 `%3C`는 공식 파일명에 포함된 `<` 문자를 URL encoding한 것입니다. 운영용 ensemble은 `model1`부터 `model4`까지 공개되어 있으며, 위 예제는 단일 rollout을 위해 `model1`을 선택합니다.

공식 WN2 weight는 약 735 MB이고 0.25° full model이므로 충분한 accelerator memory가 필요합니다. 경량 실행 검증이 먼저 필요하면 공식 Mini checkpoint를 사용할 수 있습니다.

```text
https://storage.googleapis.com/dm_graphcast/weathernext2/params/WeatherNextCyclones_Mini_%3C2024.npz
```

Mini checkpoint를 사용할 때는 checkpoint 파일만 바꾸는 것이 아니라 해당 Mini model configuration과 입력 해상도도 함께 선택해야 하므로, 현재 기본값인 `--model-variant WeatherNext2`와 그대로 혼용하지 마십시오.

출력 예시:

```text
data/
├── weathernext_forecasts/
│   └── *.nc
│
└── weathernext_tokens/
    ├── manifest.csv
    └── <storm_id>__<init_time_ns>.npz
```

## 3-D. 여러 `(storm_id, init_time)`을 한 번에 처리

`--jobs` CSV/Parquet는 `storm_id, init_time, lat, lon`과 선택적인 `pressure_hpa, wind_kt` 열을 가집니다. Step 1의 integrated dataset을 그대로 전달하면 `--job-history`, `--job-horizon`, `--job-max-highs`로 downstream dataset window를 동일하게 재구성하고, 실제 학습 sample에 해당하는 초기시각만 자동 추출합니다.

```bash
prepare-weathernext-pipeline \
  --atmospheric-state data/hres_era5_full.zarr \
  --supplement-state data/weather_supplement.zarr \
  --jobs data/integrated_typhoon_pressure.parquet \
  --job-history 8 --job-horizon 20 --job-max-highs 3 \
  --execution-mode pretrained \
  --pretrained-checkpoint download/weathernext2.npz \
  --horizon-hours 360 \
  --forecast-dir data/weathernext_forecasts \
  --token-dir data/weathernext_tokens \
  --on-error continue
```

모델/checkpoint는 한 번만 load되고 모든 job에 재사용됩니다. 기본 resume mode는 token manifest에 이미 존재하는 key를 건너뜁니다. 다시 만들려면 `--no-resume`을 사용합니다.

---

# Step 4. GPT state cache 생성

GPT conditioning을 사용할 경우 실행합니다.

먼저 API key를 설정합니다.

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

그 다음:

```bash
build-gpt-state-cache \
  --integrated data/integrated_typhoon_pressure.parquet \
  --output-dir data/gpt_states \
  --history 8 \
  --track-steps 20 \
  --max-highs 3 \
  --on-error mask
```

출력:

```text
data/gpt_states/
├── manifest.csv
└── <storm_id>__<init_time_ns>.npz
```

WeatherNext token과 GPT state는 동일한 `(storm_id, init_time)` key를 사용하여 후단 Dataset에서 결합됩니다.

GPT conditioning을 사용하지 않을 경우 Step 4는 건너뛰고 Step 5에서 `--gpt-state-dir` 옵션을 제거하면 됩니다.

GPT Router를 사용할 경우 Step 4는 필수입니다. Step 5 시작 시 WeatherNext token manifest의 모든 key가 GPT cache에도 있는지 검사합니다. API 실패로 mask된 cache까지 금지하려면 `--require-valid-gpt-states`를 추가합니다.

---

# Step 5. WeatherNext + GPT Fusion Transformer 학습

이 단계에서 실제 trainable network를 학습합니다.

WeatherNext 자체의 weight는 이미 frozen forecast/token으로 분리되어 있으므로 optimizer에는 WeatherNext가 포함되지 않습니다.

```text
Frozen WeatherNext
        ↓
WeatherNext tokens ──────┐
                         │
IBTrACS history ─────────┼─→ WeatherNextFusionTransformer
                         │
GPT state ───────────────┘
                         ↓
             Distribution Loss
                    +
                Track Loss
                         ↓
                trained weight
```

GPT state를 포함하여 학습:

```bash
train-weathernext-transformer \
  --integrated data/integrated_typhoon_pressure.parquet \
  --distribution data/distribution/spatial_distribution.csv \
  --split-manifest data/storm_split.csv \
  --weathernext-token-dir data/weathernext_tokens \
  --gpt-state-dir data/gpt_states \
  --require-valid-gpt-states \
  --epochs 10 \
  --batch-size 8 \
  --history 8 \
  --track-steps 20 \
  --max-highs 3 \
  --max-forecast-tokens 720 \
  --model-dim 128 \
  --num-heads 8 \
  --num-layers 4 \
  --decoder-layers 2 \
  --input-mask-probability 0.15 \
  --distribution-weight 1.0 \
  --track-weight 1.0 \
  --output checkpoints/weathernext_transformer.pt
```

처음 pipeline 동작만 확인할 때는 GPT 없이 먼저 학습하는 것을 권장합니다.

```bash
train-weathernext-transformer \
  --integrated data/integrated_typhoon_pressure.parquet \
  --distribution data/distribution/spatial_distribution.csv \
  --split-manifest data/storm_split.csv \
  --weathernext-token-dir data/weathernext_tokens \
  --epochs 10 \
  --batch-size 8 \
  --output checkpoints/weathernext_transformer_no_gpt.pt
```

---

# 전체 명령 요약

```bash
# 0. Setup
git checkout feature/weathernext-resolver
pip install -e ".[io,small,gpt,weathernext,test]"

# 1. IBTrACS + ERA5
build-typhoon-pressure-data \
  --ibtracs data/IBTrACS.ALL.v04r01.csv \
  --era5 data/era5_*.nc \
  --output data/integrated_typhoon_pressure.parquet

# 2. Distribution target
build-typhoon-distribution-targets \
  --ibtracs data/IBTrACS.ALL.v04r01.csv \
  --output-dir data/distribution \
  --basins WP

# 3. WeatherNext pretrained frozen rollout + all tokens
prepare-weathernext-pipeline \
  --atmospheric-state data/hres_era5_full.zarr \
  --supplement-state data/weather_supplement.zarr \
  --jobs data/weathernext_jobs.parquet \
  --execution-mode pretrained \
  --pretrained-checkpoint download/weathernext2.npz

# 4. GPT state cache
export OPENAI_API_KEY="YOUR_API_KEY"
build-gpt-state-cache \
  --integrated data/integrated_typhoon_pressure.parquet \
  --output-dir data/gpt_states

# 5. Leakage-safe storm split
build-storm-split \
  --integrated data/integrated_typhoon_pressure.parquet \
  --output data/storm_split.csv

# 6. Fusion training
train-weathernext-transformer \
  --integrated data/integrated_typhoon_pressure.parquet \
  --distribution data/distribution/spatial_distribution.csv \
  --split-manifest data/storm_split.csv \
  --weathernext-token-dir data/weathernext_tokens \
  --gpt-state-dir data/gpt_states \
  --require-valid-gpt-states \
  --epochs 10 \
  --batch-size 8 \
  --output checkpoints/weathernext_transformer.pt
```

---

# Weight와 학습 구조 요약

```text
--execution-mode pretrained
        ↓
Fine-tuned checkpoint 존재
        ↓ YES
fine-tuned weight load
        ↓
      FREEZE

없으면
        ↓
Official pretrained load/download
        ↓
      FREEZE

        ↓
WeatherNext rollout
        ↓
Token cache
        ↓
================================================
               Gradient boundary
================================================
        ↓
IBTrACS + WeatherNext token + GPT state
        ↓
WeatherNextFusionTransformer
        ↓
TRAIN
        ↓
checkpoints/weathernext_transformer.pt
```

즉 WeatherNext checkpoint source를 바꾸더라도 후단 training command는 동일하게 유지됩니다.

`--execution-mode trainable`은 이 경계 앞에서 WeatherNext를 별도로 fit하는 선택입니다. token `.npz`로 저장된 뒤에는 autograd가 끊기므로 dual loss가 WeatherNext checkpoint까지 joint backpropagation하지는 않습니다. joint end-to-end fine-tuning이 필요하면 token cache 경계를 제거하는 별도 모델 통합이 필요합니다.
