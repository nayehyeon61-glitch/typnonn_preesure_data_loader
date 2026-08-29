# 02. WeatherNext Output to GPT-routed Transformer Input

```mermaid
flowchart TB
    IB["IBTrACS storm state"]
    ATM["HRES / ERA5 atmosphere"]
    BUILDER["InitialConditionBuilder<br/>시간·좌표·변수 정렬"]
    RES["WeatherNext Resolver<br/>fine-tuned → pretrained → download → API"]
    WN["WeatherNext 2 frozen inference<br/>0–15일 forecast"]
    XR["xarray<br/>lead × lat × lon × variable"]

    IB --> BUILDER
    ATM --> BUILDER
    BUILDER --> RES
    RES --> WN
    WN --> XR

    subgraph TOKENIZE["Tokenization"]
        direction TB
        LEAD["10 selected lead times"]
        PATCH["6×12 spatial patches"]
        TOKEN["최대 720 tokens<br/>values B×N×V"]
        POS["Position features B×N×6<br/>lead·lat·lon encoding"]
    end

    XR --> LEAD
    LEAD --> PATCH
    PATCH --> TOKEN
    PATCH --> POS

    subgraph MASKS["Input masks"]
        direction TB
        FEATURE["Feature mask<br/>변수별 NaN 차단"]
        TOKENMASK["Token mask<br/>전체 변수 결측 token 차단"]
        PAD["Padding mask"]
        RANDOM["Random valid-feature mask<br/>training p=0.15"]
    end

    TOKEN --> FEATURE
    FEATURE --> TOKENMASK
    TOKENMASK --> PAD
    PAD --> RANDOM

    PROJ["Forecast projection<br/>value + feature mask + position<br/>→ Z_WN"]
    TOKEN --> PROJ
    FEATURE --> PROJ
    POS --> PROJ

    GPT["GPT semantic state z_GPT<br/>10D + state mask"]

    subgraph ROUTER["GPTForecastRouter"]
        direction TB
        CTX["GPT context projection"]
        TG["Token Gate<br/>g_token ∈ R^(B×N×1)"]
        CG["Channel Gate<br/>g_channel ∈ R^(B×1×d)"]
        ROUTE["Z̃_WN = Z_WN ⊙ g_token ⊙ g_channel"]
        ID["GPT missing → gate = 1<br/>exact identity"]
    end

    GPT --> CTX
    CTX --> TG
    CTX --> CG
    PROJ --> TG
    PROJ --> ROUTE
    TG --> ROUTE
    CG --> ROUTE
    GPT -. "mask=0" .-> ID
    ID --> ROUTE

    subgraph ENCODER["Trainable WeatherNext representation"]
        direction TB
        CLS["Always-valid CLS token"]
        TF["Non-causal Transformer Encoder"]
        MEMORY["GPT-routed WeatherNext memory"]
        GLOBAL["CLS global representation"]
    end

    ROUTE --> CLS
    CLS --> TF
    RANDOM --> TF
    TF --> MEMORY
    TF --> GLOBAL
```

## Router 수식

```math
g_{token}=2\sigma(f_{token}(Z_{WN},z_{GPT}))
```

```math
g_{channel}=2\sigma(f_{channel}(z_{GPT}))
```

```math
\tilde Z_{WN}=Z_{WN}\odot g_{token}\odot g_{channel}
```

마지막 gate layer는 0으로 초기화하므로 학습 시작 시 `2σ(0)=1`입니다. 즉 기존 Transformer와 동일한 입력으로 시작하고, 학습을 통해 GPT state가 spatial/lead-time token 및 latent channel의 중요도를 점진적으로 조절합니다.
