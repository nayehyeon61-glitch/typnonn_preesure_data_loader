# Setup Start

이 문서는 현재 WeatherNext + IBTrACS + GPT/Fusion 학습 시스템을 처음 실행할 때 필요한 순서를 정리합니다.

전체 흐름은 다음 5단계입니다.

```text
1. IBTrACS + ERA5 통합 데이터 생성
        ↓
2. Distribution target 생성
        ↓
3. WeatherNext source 선택 → frozen rollout → token cache 생성
        ↓
4. GPT state cache 생성 (선택)
        ↓
5. WeatherNextFusionTransformer 학습
```

WeatherNext 자체는 3단계에서 checkpoint를 선택한 뒤 frozen 상태로 rollout만 수행합니다. 실제 optimizer 학습은 5단계의 WeatherNextFusionTransformer에서 수행됩니다.

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

# Step 3. WeatherNext frozen rollout + token cache 생성

이 단계가 WeatherNext resolver가 실제 pipeline에 연결되는 부분입니다.

Resolver 우선순위:

```text
1. fine-tuned checkpoint
2. local official/pretrained checkpoint
3. online checkpoint download
4. API fallback (Python application에서 client 주입 시)
```

선택된 WeatherNext weight는 frozen 상태로 사용되며 추가 fine-tuning은 발생하지 않습니다.

전체 흐름:

```text
ERA5 / HRES initial state
        +
IBTrACS storm state
        ↓
InitialConditionBuilder
        ↓
WeatherNext Resolver
        ↓
Frozen WeatherNext
        ↓
Rollout
        ↓
Forecast NetCDF
        ↓
WeatherNext tokenizer
        ↓
Token .npz + manifest.csv
```

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
  --pretrained-checkpoint download/weathernext2.npz \
  --horizon-hours 360 \
  --forecast-dir data/weathernext_forecasts \
  --token-dir data/weathernext_tokens
```

## 3-C. Local checkpoint가 없고 online checkpoint를 받을 경우

공개 checkpoint URL이 있는 경우:

```bash
prepare-weathernext-pipeline \
  --atmospheric-state data/weather_initial_state.nc \
  --storm-id TEST \
  --init-time 2025-08-01T00:00:00 \
  --lat 22.0 \
  --lon 133.0 \
  --pressure-hpa 975 \
  --wind-kt 70 \
  --checkpoint-url "https://.../weathernext2.npz" \
  --download-dir download/weathernext \
  --horizon-hours 360 \
  --forecast-dir data/weathernext_forecasts \
  --token-dir data/weathernext_tokens
```

다운로드된 checkpoint는 cache되므로 이후 실행에서는 local weight처럼 재사용됩니다.

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

### 중요

현재 `prepare-weathernext-pipeline`은 한 개의 `(storm_id, init_time)` sample을 처리합니다.

따라서 전체 training dataset에 대해서는 각 initialization sample별로 반복 실행하여 `data/weathernext_tokens/manifest.csv`를 충분히 채워야 합니다.

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
  --weathernext-token-dir data/weathernext_tokens \
  --gpt-state-dir data/gpt_states \
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

# 3. WeatherNext frozen rollout + token
prepare-weathernext-pipeline \
  --atmospheric-state data/weather_initial_state.nc \
  --storm-id TEST \
  --init-time 2025-08-01T00:00:00 \
  --lat 22 \
  --lon 133 \
  --pressure-hpa 975 \
  --finetuned-checkpoint checkpoints/korea_finetuned.npz \
  --pretrained-checkpoint download/weathernext2.npz

# Step 3은 training sample별 반복

# 4. GPT state cache
export OPENAI_API_KEY="YOUR_API_KEY"
build-gpt-state-cache \
  --integrated data/integrated_typhoon_pressure.parquet \
  --output-dir data/gpt_states

# 5. Fusion training
train-weathernext-transformer \
  --integrated data/integrated_typhoon_pressure.parquet \
  --distribution data/distribution/spatial_distribution.csv \
  --weathernext-token-dir data/weathernext_tokens \
  --gpt-state-dir data/gpt_states \
  --epochs 10 \
  --batch-size 8 \
  --output checkpoints/weathernext_transformer.pt
```

---

# Weight와 학습 구조 요약

```text
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
