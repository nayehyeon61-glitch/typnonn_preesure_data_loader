# WeatherNext Checkpoint Download

이 디렉터리는 Google DeepMind의 공식 WeatherNext checkpoint를 내려받는 기본
위치입니다. 대형 `.npz` 파일과 생성된 `.metadata.json`은 Git에 커밋하지 않습니다.
공식 파일 목록은 [`official_checkpoints.json`](official_checkpoints.json)에 정리되어
있습니다.

## 1. 설치와 모델 목록

```bash
pip install -e '.[io,small,gpt,weathernext]'
download-weathernext-checkpoint --list
```

## 2. 공식 checkpoint 다운로드

로컬 검증에는 1° Mini를 먼저 권장합니다.

```bash
download-weathernext-checkpoint --model mini
```

운영 HRES 기반 WeatherNext2 member 1:

```bash
download-weathernext-checkpoint --model weathernext2
```

태풍 특화 member 1:

```bash
download-weathernext-checkpoint --model cyclones
```

정확한 파일명도 사용할 수 있습니다.

```bash
download-weathernext-checkpoint \
  --model 'WeatherNext2_<2025_model2.npz'
```

다운로드는 `.npz.part`에 먼저 기록한 뒤 완료 시 원자적으로 이름을 바꿉니다.
동시에 checkpoint 출처와 모델 구조를 기록한 sidecar가 생성됩니다.

```text
download/
├── README.md
├── official_checkpoints.json
├── WeatherNext2_<2025_model1.npz
└── WeatherNext2_<2025_model1.metadata.json
```

## 3. Pretrained rollout과 token cache

```bash
prepare-weathernext-tokens \
  --initial-state data/hres_history.nc \
  --checkpoint 'download/WeatherNext2_<2025_model1.npz' \
  --model-variant WeatherNext2 \
  --model-id 'WeatherNext2_<2025_model1' \
  --storm-id WP012026 \
  --init-time 2026-08-01T00:00:00 \
  --storm-lat 22.5 \
  --storm-lon 132.0 \
  --storm-pressure-hpa 975 \
  --storm-wind-kt 75 \
  --output-dir data/official_pretrained_tokens
```

이 명령은 checkpoint를 읽기 전용으로 로드하고, 0–15일 rollout을 실행한 뒤
Transformer token과 provenance를 `manifest.csv`에 저장합니다.

전체 integrated table에 대한 cache는 같은 명령에서 개별 storm 인자 대신
`--jobs data/integrated.parquet`을 사용합니다. 모든 `(storm_id, init_time)` coverage를
검사하며 중단 후 재실행하면 완성된 identity는 자동으로 resume합니다.

## 4. 후단 모델 학습

```bash
train-weathernext-transformer \
  --integrated data/integrated.parquet \
  --distribution data/spatial_distribution.csv \
  --weathernext-token-dir data/official_pretrained_tokens \
  --gpt-state-dir data/gpt_states \
  --require-checkpoint-kind official_pretrained \
  --distribution-weight 1.0 \
  --track-weight 1.0 \
  --output checkpoints/official_pretrained_fusion.pt
```

`--require-checkpoint-kind official_pretrained`은 fine-tuned 또는 출처가 기록되지 않은
token을 실수로 섞으면 학습을 중단합니다. 최종 `.pt`에는
`weathernext_provenance`가 함께 저장됩니다.

## Fine-tuned checkpoint 확인

`weather-me-fine_tune_weight.npz`와 그 옆의
`weather-me-fine_tune_weight.metadata.json`을 지정하면 같은 token 생성 경로를
사용합니다. metadata의 `checkpoint_kind=fine_tuned` 또는 `fine_tune_steps`를 읽어
fine-tuned 적용 여부를 기록합니다.

```bash
prepare-weathernext-tokens \
  --initial-state data/hres_history.nc \
  --checkpoint download/weather-me-fine_tune_weight.npz \
  --model-variant WeatherNext2 \
  --model-id regional-wn2-35N45N \
  --storm-id WP012026 \
  --init-time 2026-08-01T00:00:00 \
  --storm-lat 22.5 \
  --storm-lon 132.0 \
  --output-dir data/fine_tuned_tokens

train-weathernext-transformer \
  --integrated data/integrated.parquet \
  --distribution data/spatial_distribution.csv \
  --weathernext-token-dir data/fine_tuned_tokens \
  --gpt-state-dir data/gpt_states \
  --require-checkpoint-kind fine_tuned \
  --output checkpoints/fine_tuned_fusion.pt
```

WeatherNext/GPT에는 후단 loss gradient가 전달되지 않습니다. 학습 대상은 FiLM,
GRU, Transformer, fusion, distribution head와 track head입니다.
