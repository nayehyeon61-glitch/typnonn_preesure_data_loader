# Small Version: Dual-scale Typhoon Training

이 폴더는 두 저장소를 연결하는 경량 실험 구현입니다.

```text
typnoon-disribution
  spatial_distribution.csv (IBTrACS monthly Earth-grid probabilities)
                              |
                              v
WeatherNext 0-15 day xarray --> masked tokens --> GPT Router --> Transformer
                                                       |
Pressure Data Loader history --> GPT Structured Output cache
              |                        |
              +--> masked projection --> History FiLM --> GRU
                                                       |
                                      GRU + Transformer fusion
                                          |              |
                                          v              v
                         15-30 day future queries     East Asia track
                                          |
                              cross-attention decoder
                                          |
                               daily Earth-grid logits
                                          |              |
                                  soft-label CE      masked location MSE
                                          +------v-------+
                                               total loss
```

목적함수는 다음과 같습니다.

```text
L_total = lambda_distribution * L_CE
        + lambda_track * L_EastAsia_MSE
```

- `L_CE`: 초기시각에서 15–30일 뒤의 달(month)에 해당하는 IBTrACS 경험적 지구 격자분포와 예측 logits의 soft-label cross entropy
- `L_EastAsia_MSE`: 동아시아 `(0–60°N, 100–180°E)` 안에 존재하는 유효한 미래 태풍 위치만 사용한 거리 MSE
- 위치 loss는 km²를 `track_scale_km²`로 나누어 CE와 수치 규모를 맞춥니다. 로그에는 실제 `local_track_rmse_km`도 함께 표시됩니다.

현재 분포 target은 **월별 기후학적 분포**입니다. 따라서 30일 동안 동일한 태풍이 생존한다는 뜻이 아니라, 해당 미래 시점에 태풍이 나타날 공간확률을 학습합니다. 이후 WeatherNext ensemble이나 조건부 IBTrACS occurrence target을 확보하면 `SpatialDistributionLookup`만 교체하면 됩니다.

## WeatherNext Transformer 입력 mask

WeatherNext 전 지구 field는 기본적으로 10개 lead time과 `6×12` 공간 patch로 축약하여 최대 720개 token으로 만듭니다.

1. `forecast_feature_mask`: NaN이나 부분 결측을 변수 단위로 차단합니다.
2. `forecast_token_mask`: 모든 변수가 결측인 공간·시간 token을 attention에서 제외합니다.
3. `history_mask`: 기존 IBTrACS/ERA5 history 결측값을 0으로 만들고 mask 자체를 encoder 입력에 포함합니다.
4. `input_mask_probability`: 학습 중 유효 feature 일부를 무작위로 가려 결측과 입력 손상에 대한 강건성을 높입니다.
5. 항상 유효한 `CLS` token을 두어 한 sample의 forecast token이 전부 mask되어도 attention이 NaN을 만들지 않도록 합니다.

전체 0–15일 WeatherNext 출력은 예측 시점에 이미 주어진 입력이므로 temporal causal mask는 사용하지 않습니다. 미래 15–30일 target은 Transformer 입력에 포함되지 않습니다.

분포 branch에는 15일부터 30일까지 하나씩 총 16개의 learnable future query가 있습니다. 이 query들이 masked WeatherNext memory에 cross-attention한 뒤 각 날짜의 Earth-grid logits를 직접 출력합니다. 16개 날짜를 동시에 예측하므로 decoder query 사이에도 causal mask를 적용하지 않습니다.

## GPT state extraction

GPT는 수치 예측값을 대체하지 않고, 다음 10차원 synoptic state를 추가 feature로 제공합니다.

```text
eastward / northward steering
recurvature / intensification
subtropical-high / monsoon influence
East-Asia approach risk
track / intensity uncertainty
state confidence
```

GPT에는 태풍·주변 고기압 history의 최신값·변화량·유효 비율만 전달합니다. OpenAI Responses API Structured Outputs로 schema를 강제한 뒤 결과를 sample별로 한 번만 cache합니다. 같은 state가 FiLM의 scale/shift로 history representation을 조절하고, `GPTForecastRouter`의 token/channel gate로 WeatherNext representation을 routing합니다.

```text
raw history → masked projection → GPT scale/shift → dynamic history → GRU
WeatherNext token → GPT token/channel gates → routed token → Transformer
```

API 실패·거절·미생성 state는 `values=0, mask=0`으로 저장됩니다. 이 경우 scale과 shift는 0, 두 Router gate는 1이 되어 history와 WeatherNext token 모두 exact identity 경로로 작동합니다. `--gpt-state-dir` 자체를 생략하면 두 GPT adapter module은 생성되지 않습니다.

```bash
pip install -e '.[io,small,gpt]'

export OPENAI_API_KEY=...

build-gpt-state-cache \
  --integrated data/integrated_typhoon_pressure.parquet \
  --output-dir data/gpt_states \
  --model gpt-5.6 \
  --on-error mask
```

## 실행

```bash
pip install -e '.[io,small]'

build-typhoon-distribution-targets \
  --ibtracs data/IBTrACS.ALL.v04r01.csv \
  --basins WP \
  --output-dir data/distribution

train-small-typhoon-model \
  --integrated data/integrated_typhoon_pressure.parquet \
  --distribution data/distribution/spatial_distribution.csv \
  --history 8 \
  --track-steps 20 \
  --distribution-weight 1.0 \
  --track-weight 1.0 \
  --output checkpoints/small_dual_scale_model.pt
```

WeatherNext 출력 파일을 Transformer token으로 변환하고 fusion model을 학습하려면:

```bash
tokenize-weathernext-output \
  --forecast data/weathernext/TEST_20250101.nc \
  --storm-id TEST \
  --init-time 2025-01-01T12:00:00 \
  --output-dir data/weathernext_tokens

train-weathernext-transformer \
  --integrated data/integrated_typhoon_pressure.parquet \
  --distribution data/distribution/spatial_distribution.csv \
  --weathernext-token-dir data/weathernext_tokens \
  --gpt-state-dir data/gpt_states \
  --input-mask-probability 0.15 \
  --output checkpoints/weathernext_transformer.pt
```

`history=8`은 6시간 간격 48시간 입력이며, `track-steps=20`은 동아시아 경로 5일 target입니다. 전 지구 분포 head는 독립적으로 15일부터 30일까지 일 단위로 출력합니다.
