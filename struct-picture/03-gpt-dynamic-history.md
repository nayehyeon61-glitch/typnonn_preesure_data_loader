# 03. GPT Semantic State: History FiLM + Forecast Router

```mermaid
flowchart TB
    RAW["Raw typhoon / pressure history<br/>B×8×F"]
    HMASK["History valid mask"]

    subgraph SUMMARY["Deterministic summary"]
        direction LR
        LATEST["Latest valid values"]
        CHANGE["Temporal changes"]
        FRACTION["Valid fractions"]
        HIGH["Surrounding-high context"]
    end

    RAW --> LATEST
    RAW --> CHANGE
    HMASK --> FRACTION
    RAW --> HIGH

    JSON["Restricted numerical JSON"]
    LATEST --> JSON
    CHANGE --> JSON
    FRACTION --> JSON
    HIGH --> JSON

    API["OpenAI Responses API<br/>Structured Output schema"]
    STATE["10D semantic state z_GPT<br/>steering·recurvature·risk·uncertainty"]
    CACHE["Per-sample GPT state cache"]

    JSON --> API
    API --> STATE
    STATE --> CACHE

    subgraph PATH1["Path A. History FiLM"]
        direction TB
        SMASK1["GPT state mask"]
        FILM["FiLM adapter"]
        PARAM["γ = 0.5 tanh γ_raw<br/>β = tanh β_raw"]
        DYNAMIC["h' = (1+γ)h + β"]
        GRU["Masked GRU Encoder"]
        HIDDEN["Typhoon hidden state"]
    end

    CACHE --> SMASK1
    CACHE --> FILM
    SMASK1 --> FILM
    FILM --> PARAM
    RAW --> DYNAMIC
    PARAM --> DYNAMIC
    DYNAMIC --> GRU
    HMASK --> GRU
    GRU --> HIDDEN

    subgraph PATH2["Path B. WeatherNext Router"]
        direction TB
        ZWN["Projected WeatherNext tokens Z_WN"]
        CTX["GPT context projection"]
        TG["Token Gate g_token"]
        CG["Channel Gate g_channel"]
        ROUTED["Z̃_WN = Z_WN ⊙ g_token ⊙ g_channel"]
        TF["Transformer Encoder"]
        WMEM["Routed WeatherNext memory"]
    end

    CACHE --> CTX
    ZWN --> TG
    CTX --> TG
    CTX --> CG
    ZWN --> ROUTED
    TG --> ROUTED
    CG --> ROUTED
    ROUTED --> TF
    TF --> WMEM

    MISSING["API 실패 / cache 누락<br/>values=0, mask=0"]
    ID1["FiLM identity<br/>γ=β=0"]
    ID2["Router identity<br/>g_token=g_channel=1"]
    MISSING --> ID1
    MISSING --> ID2
    ID1 --> DYNAMIC
    ID2 --> ROUTED

    FUSION["Fusion<br/>GRU hidden + WeatherNext CLS"]
    HIDDEN --> FUSION
    WMEM --> FUSION
```

GPT는 이제 단순한 별도 feature branch가 아닙니다. 동일한 semantic state가

1. 관측 history representation을 FiLM으로 조절하고,
2. WeatherNext forecast token의 공간·lead-time·latent-channel 중요도를 routing합니다.

즉 `z_GPT`는 현재 시스템에서 **conditioning state + routing policy source**로 사용됩니다.
