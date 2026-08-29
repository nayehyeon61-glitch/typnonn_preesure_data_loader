# IBTrACS-only Evaluation

첫 evaluation 단계는 외부 모델 간 비교 없이, 모델 예측을 IBTrACS 관측과 직접 비교합니다.

실제 구현은 [`src/typhoon_pressure/evaluation/`](../src/typhoon_pressure/evaluation/)에 있습니다.

## Prediction contract

필수 열:

```text
storm_id, pred_lat, pred_lon
```

시간은 다음 중 한 형식을 사용합니다.

```text
time
```

또는

```text
init_time, lead_hours
```

선택 열:

```text
pred_pressure_hpa, pred_wind_kt, lead_hours, weathernext_backend
```

## CLI

```bash
evaluate-ibtracs-predictions \
  --predictions data/predictions.csv \
  --ibtracs data/IBTrACS.ALL.v04r01.csv \
  --basin WP \
  --agency TOKYO \
  --weathernext-backend pretrained \
  --output-dir evaluation/ibtracs
```

동아시아 target 위치만 평가하려면 `--east-asia-only`를 추가합니다.

## Outputs

| 파일 | 내용 |
|---|---|
| `matched_predictions.csv` | storm ID와 절대시간으로 결합한 예측·IBTrACS·sample별 오차 |
| `metrics_overall.json` | 전체 track·기압·풍속 metric |
| `metrics_by_lead.csv` | `lead_hours`가 있을 때 lead-time별 metric |
| `metrics_by_storm.csv` | 태풍별 metric |
| `metrics_by_backend.csv` | WeatherNext backend별 metric |

## WeatherNext type 비교

`trainable`, `pretrained`, `api`가 생성한 prediction을 하나의 표로 합치고 `weathernext_backend` 열을 유지하면, 동일한 IBTrACS 관측에 대해 backend별 metric을 계산합니다. 즉 현재 비교 기준은 모두 IBTrACS로 동일하며 backend만 달라집니다.

WeatherNext 실행 선택과 provenance 구조는 [`weathernext/README.md`](../weathernext/README.md)를 참고하십시오.

현재 비교 정답은 IBTrACS로 제한합니다. 이후 ensemble spread와 distribution calibration metric을 같은 evaluation interface 아래 추가할 수 있습니다.
