# 01. Overall System

아래 구조는 `feature/weathernext-resolver`의 실제 실행 경로를 기준으로 합니다. WeatherNext와 GPT API는 downstream 학습 밖에서 실행되고, cache/token을 입력받는 FiLM·Router·GRU·Transformer·두 예측 head만 공동 학습됩니다.

```mermaid
flowchart TB
    subgraph SOURCE["1. Source and targets"]
        IB["IBTrACS<br/>track, wind, pressure, time"]
        ATM["HRES or ERA5<br/>atmospheric fields"]
        SUP["Supplement sources<br/>SST, static, 100 m wind"]
        DISTDATA["typnoon-disribution<br/>monthly Earth distribution"]
        DISTGT["Day 15–30 soft grid target<br/>B×16×2592"]
        TRACKGT["East Asia track target<br/>valid and region mask"]
    end

    IB --> DISTDATA
    DISTDATA --> DISTGT
    IB --> TRACKGT

    subgraph OFFLINE["2. Offline or upstream inference"]
        PREP["WeatherNextInputPreparer<br/>merge, 2 times, 13 levels, global grid"]
        INIT["InitialConditionBuilder<br/>tracker seed or vortex correction"]
        RES["Execution mode<br/>pretrained, API, or trainable"]
        ROLLOUT["Resolved WeatherNext rollout<br/>0–15 days"]
        TOKENIZE["Spatiotemporal tokenizer<br/>values, masks, positions"]
        SUMMARY["Deterministic history summary"]
        GPTAPI["GPT Structured Output API"]
        CACHE["10D semantic-state cache<br/>values and state mask"]
    end

    ATM --> PREP
    SUP --> PREP
    PREP --> INIT
    IB --> INIT
    INIT --> RES
    RES --> ROLLOUT
    ROLLOUT --> TOKENIZE
    IB --> SUMMARY
    ATM --> SUMMARY
    SUMMARY --> GPTAPI
    GPTAPI --> CACHE

    subgraph HISTORY["3-A. History branch"]
        HP["Masked history projection<br/>concat values and mask"]
        FILM["History FiLM<br/>γ, β from GPT state"]
        DYNAMIC["Dynamic history<br/>h'=(1+γ)h+β"]
        GRU["GRU history encoder"]
        HH["History hidden state"]
    end

    IB --> HP
    ATM --> HP
    CACHE --> FILM
    HP --> DYNAMIC
    FILM --> DYNAMIC
    DYNAMIC --> GRU
    GRU --> HH

    subgraph FORECAST["3-B. WeatherNext branch"]
        MASK["Training random feature mask<br/>plus missing-data masks"]
        PROJ["Value + effective mask + position<br/>projection to Z_WN"]
        CONTEXT["GPT context MLP"]
        TG["Token Gate<br/>from Z_WN + GPT context"]
        CG["Channel Gate<br/>from GPT context only"]
        ROUTED["Routed tokens<br/>Z̃_WN=Z_WN⊙g_token⊙g_channel"]
        TF["Transformer encoder<br/>padding mask + always-valid CLS"]
        MEM["Routed forecast memory and CLS"]
    end

    TOKENIZE --> MASK
    MASK --> PROJ
    CACHE --> CONTEXT
    PROJ --> TG
    CONTEXT --> TG
    CONTEXT --> CG
    PROJ --> ROUTED
    TG --> ROUTED
    CG --> ROUTED
    ROUTED --> TF
    MASK --> TF
    TF --> MEM

    subgraph OUTPUT["4. Prediction"]
        FUS["Fusion<br/>history hidden + forecast CLS"]
        QUERY["16 learned day queries<br/>day 15…30"]
        DEC["Cross-attention decoder<br/>queries attend forecast memory"]
        GRID["Global distribution logits"]
        TRACK["East Asia track head"]
        LATLON["Track latitude and longitude"]
    end

    HH --> FUS
    MEM --> FUS
    FUS --> QUERY
    QUERY --> DEC
    MEM --> DEC
    DEC --> GRID
    FUS --> TRACK
    TRACK --> LATLON

    subgraph LOSS["5. Joint objective"]
        CE["Soft-label distribution CE<br/>available-day mask"]
        MSE["Normalized track MSE in km<br/>valid and East-Asia mask"]
        TOTAL["L_total = λ_dist L_CE + λ_track L_MSE"]
    end

    GRID --> CE
    DISTGT --> CE
    LATLON --> MSE
    TRACKGT --> MSE
    CE --> TOTAL
    MSE --> TOTAL

    TOTAL -. "through both heads" .-> FUS
    TOTAL -. "shared gradient" .-> TF
    TOTAL -. "router gradient" .-> TG
    TOTAL -. "router gradient" .-> CG
    TOTAL -. "history gradient" .-> GRU
    TOTAL -. "FiLM gradient" .-> FILM
```

## 학습 경계

| 구성요소 | downstream 학습 상태 | 비고 |
|---|---:|---|
| Pretrained/API WeatherNext rollout | 고정 | 공식/fine-tuned checkpoint와 API는 inference-only |
| Trainable WeatherNext mode | upstream 별도 학습 | 명시적 factory로 fit 후 token 생성; dual loss와 autograd 연결 없음 |
| GPT API와 state cache | 고정 | 학습 loop에서 API를 호출하지 않음 |
| FiLM, GPTForecastRouter | 학습 | 두 loss의 gradient를 받음 |
| GRU, Transformer, Fusion, Decoder, Heads | 학습 | 공동 최적화 |

`--gpt-state-dir`을 지정하면 학습 시작 전에 모든 WeatherNext token key의 GPT cache 존재 여부를 검사합니다. 존재하지만 API 실패로 masked된 record는 FiLM `γ=β=0`, Router `g_token=g_channel=1`의 identity fallback이며, `--require-valid-gpt-states`로 이 경우도 오류 처리할 수 있습니다.
