# 01. Overall System

현재 시스템은 WeatherNext 2의 물리적 forecast dynamics를 고정하고, GPT가 생성한 synoptic state를 두 경로에서 능동적으로 사용합니다.

1. History FiLM conditioning
2. WeatherNext Token / Channel Router

```mermaid
flowchart TB
    subgraph SOURCE["1. 원천 데이터"]
        direction LR
        IB["IBTrACS<br/>위치·풍속·기압·시간"]
        ERA["HRES / ERA5<br/>대기장·초기조건"]
        OBS["관측 정답<br/>미래 위치·경로"]
    end

    subgraph TARGETS["2. 전처리와 target 생성"]
        direction LR
        INIT["InitialConditionBuilder<br/>절대시간·공간 정렬"]
        DS["typnoon-disribution<br/>Earth distribution"]
        GT["15–30일 soft grid target<br/>36×72 cells"]
        TT["동아시아 track target<br/>valid + region mask"]
    end

    IB --> INIT
    ERA --> INIT
    IB --> DS
    OBS --> DS
    DS --> GT
    OBS --> TT

    subgraph WNB["3-A. Frozen WeatherNext dynamics"]
        direction TB
        RES["WeatherNext Resolver<br/>fine-tuned → pretrained → download → API"]
        WN["WeatherNext 2<br/>Frozen rollout 0–15일"]
        TOK["Spatiotemporal tokenizer<br/>최대 720 tokens"]
        FP["Value + mask + position projection"]
        ROUTED["GPT-routed WeatherNext tokens"]
        TE["Masked Transformer Encoder"]
        MEM["WeatherNext memory + CLS"]
    end

    INIT --> RES
    RES --> WN
    WN --> TOK
    TOK --> FP

    subgraph GPTB["3-B. GPT semantic-state branch"]
        direction TB
        HIS["태풍·기압 History<br/>48시간"]
        SUM["Deterministic Summary"]
        API["GPT Structured Output<br/>10-state cache"]
        ZGPT["Semantic State z_GPT<br/>steering·recurvature·risk·uncertainty"]
    end

    IB --> HIS
    ERA --> HIS
    HIS --> SUM
    SUM --> API
    API --> ZGPT

    subgraph ACTIVE["3-C. Active LLM control"]
        direction LR
        FILM["History FiLM<br/>γ, β"]
        TGR["Token Gate<br/>g_token"]
        CGR["Channel Gate<br/>g_channel"]
    end

    ZGPT --> FILM
    ZGPT --> TGR
    ZGPT --> CGR
    HIS --> FILM

    subgraph HISTORY["History dynamics representation"]
        direction TB
        DH["GPT-conditioned history<br/>h'=(1+γ)h+β"]
        GRU["Masked GRU Encoder"]
        HH["Typhoon hidden state"]
    end

    FILM --> DH
    HIS --> DH
    DH --> GRU
    GRU --> HH

    FP --> TGR
    FP --> CGR
    TGR --> ROUTED
    CGR --> ROUTED
    ROUTED --> TE
    TE --> MEM

    subgraph MODEL["4. 공유 표현과 예측"]
        direction TB
        FUS["Fusion<br/>WeatherNext CLS + GRU hidden"]
        Q["16 learned future queries<br/>day 15…30"]
        DEC["Cross-attention Decoder"]
        GRID["Earth-grid logits<br/>B×16×2592"]
        TH["East Asia Track Head"]
        TP["Track prediction<br/>B×20×2"]
    end

    MEM --> FUS
    HH --> FUS
    Q --> DEC
    MEM --> DEC
    FUS --> DEC
    DEC --> GRID
    FUS --> TH
    TH --> TP

    subgraph OBJECTIVE["5. Dual objective"]
        direction LR
        CE["Soft Distribution CE<br/>day mask"]
        MSE["Track MSE in km<br/>valid + region mask"]
        TOTAL["λ_dist L_CE<br/>+ λ_track L_MSE"]
    end

    GRID --> CE
    GT --> CE
    TP --> MSE
    TT --> MSE
    CE --> TOTAL
    MSE --> TOTAL

    TOTAL -. "gradient" .-> DEC
    TOTAL -. "gradient" .-> TH
    TOTAL -. "shared gradient" .-> FUS
    TOTAL -. "shared gradient" .-> TE
    TOTAL -. "router gradient" .-> TGR
    TOTAL -. "router gradient" .-> CGR
    TOTAL -. "shared gradient" .-> GRU
    TOTAL -. "FiLM gradient" .-> FILM
```

## 핵심 해석

```text
WeatherNext 2 = frozen physical forecast dynamics
GPT state     = semantic state
GPT Router    = semantic state → routing policy
Transformer   = routed WeatherNext representation learner
```

WeatherNext 2와 GPT API 자체에는 학습 gradient가 전달되지 않습니다. 반면 GPT state를 입력받는 FiLM adapter와 Token/Channel Router는 후단 loss에서 학습됩니다.
