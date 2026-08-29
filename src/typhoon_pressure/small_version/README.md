# Small Version: Dual-scale Typhoon Training

이 폴더는 두 저장소를 연결하는 경량 실험 구현입니다.

```text
typnoon-disribution
  spatial_distribution.csv (IBTrACS monthly Earth-grid probabilities)
                              |
                              v
Typhoon Pressure Data Loader history --> shared GRU encoder
                                          |              |
                                          v              v
                               15-30 day grid logits   East Asia track
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

`history=8`은 6시간 간격 48시간 입력이며, `track-steps=20`은 동아시아 경로 5일 target입니다. 전 지구 분포 head는 독립적으로 15일부터 30일까지 일 단위로 출력합니다.

