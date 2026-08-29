# Typhoon Forecasting System Diagrams

이 폴더는 현재 구현된 태풍 장기예측 시스템을 Mermaid 구조도로 정리합니다.

## 그림 목록

| 문서 | 내용 |
|---|---|
| [01-overall-system.md](01-overall-system.md) | 데이터부터 두 loss와 역전파까지 전체 구조 |
| [02-weathernext-transformer.md](02-weathernext-transformer.md) | WeatherNext 0–15일 출력과 최대 mask Transformer |
| [03-gpt-dynamic-history.md](03-gpt-dynamic-history.md) | History → GPT → Dynamic History → GRU 직렬 구조 |
| [04-dual-loss-training.md](04-dual-loss-training.md) | 15–30일 분포 CE와 동아시아 경로 MSE |
| [05-gpt-cache-runtime.md](05-gpt-cache-runtime.md) | GPT API 호출, cache 생성과 실패·누락 처리 |

## 전체 흐름

```mermaid
flowchart TB
    DATA["IBTrACS + HRES/ERA5"]

    subgraph INPUT["입력 생성"]
        direction LR
        WN["WeatherNext 2<br/>0–15일 예보"]
        HIST["태풍·기압 History<br/>48시간"]
    end

    subgraph ENCODER["표현 학습"]
        direction LR
        TF["Masked Transformer<br/>WeatherNext memory"]
        GPT["GPT state → FiLM<br/>Dynamic History → GRU"]
    end

    FUSION["Shared Fusion Representation"]

    subgraph OUTPUT["Multi-task prediction"]
        direction LR
        DIST["15–30일<br/>전지구 분포"]
        TRACK["동아시아<br/>태풍 경로"]
    end

    subgraph LOSS["Joint objective"]
        direction LR
        CE["Soft-label CE"]
        MSE["Masked Track MSE"]
    end

    TOTAL["L_total = λ_dist L_CE + λ_track L_MSE"]

    DATA --> WN
    DATA --> HIST
    WN --> TF
    HIST --> GPT
    TF --> FUSION
    GPT --> FUSION
    FUSION --> DIST
    FUSION --> TRACK
    DIST --> CE
    TRACK --> MSE
    CE --> TOTAL
    MSE --> TOTAL
```

## 핵심 설정

| 항목 | 현재 구조 |
|---|---|
| History | 6시간 간격 8 step, 총 48시간 |
| WeatherNext 입력 | 0–15일 예보장 |
| WeatherNext token | 최대 10 × 6 × 12 = 720 tokens |
| 장기 분포 예측 | 15일부터 30일까지 16 future queries |
| 전지구 격자 | 위도 36 × 경도 72 = 2,592 cells |
| 동아시아 영역 | 0–60°N, 100–180°E |
| 경로 예측 | 6시간 간격 20 step, 총 5일 |
| GPT state | Structured Output 10차원, 사전 cache |
| 최종 목적함수 | λ_dist L_CE + λ_track L_MSE |

## 범례

- 외부·고정 단계: WeatherNext 2 실행, GPT API 호출, cache와 target 생성
- 학습 가능 단계: Transformer, FiLM adapter, GRU, fusion, decoder, track head
- Mask: feature, token, padding, random-input, history, GPT-state, distribution-day, track-valid/region
- Gradient 차단: WeatherNext 2 자체, GPT API와 GPT cache에는 역전파하지 않음
