# 02. WeatherNext Output to GPT-routed Transformer Input

이 그림은 mask가 적용되는 정확한 순서와 Router의 실제 입력 의존성을 보여줍니다.

```mermaid
flowchart TB
    INIT["Aligned initial condition"]
    RES["Priority resolver"]
    WN["Resolved WeatherNext inference<br/>0–15 days, frozen downstream"]
    XR["xarray forecast<br/>lead × lat × lon × variable"]

    INIT --> RES
    RES --> WN
    WN --> XR

    subgraph TOK["Tokenization"]
        SELECT["Up to 10 lead times"]
        PATCH["6×12 spatial patches"]
        VALUES["Raw token values<br/>B×N×V, N≤720"]
        POS["Position features<br/>B×N×6"]
        SOURCE_MASKS["Source feature and token masks"]
    end

    XR --> SELECT
    SELECT --> PATCH
    PATCH --> VALUES
    PATCH --> POS
    PATCH --> SOURCE_MASKS

    subgraph MASKING["Runtime masking before projection"]
        RANDOM["Random feature keep mask<br/>training only, p=0.15 masked"]
        EFFECTIVE_FEATURE["Effective feature mask<br/>source feature ∧ random keep"]
        ZERO["Masked values set to zero"]
        EFFECTIVE_TOKEN["Effective token mask<br/>source token ∧ any valid feature"]
        PADDING["Transformer padding mask<br/>CLS is always valid"]
    end

    SOURCE_MASKS --> EFFECTIVE_FEATURE
    RANDOM --> EFFECTIVE_FEATURE
    VALUES --> ZERO
    EFFECTIVE_FEATURE --> ZERO
    SOURCE_MASKS --> EFFECTIVE_TOKEN
    EFFECTIVE_FEATURE --> EFFECTIVE_TOKEN
    EFFECTIVE_TOKEN --> PADDING

    PROJ["Forecast projection<br/>zeroed values + effective feature mask + position<br/>→ Z_WN"]
    ZERO --> PROJ
    EFFECTIVE_FEATURE --> PROJ
    POS --> PROJ

    subgraph ROUTER["GPTForecastRouter"]
        GPT["GPT state values + mask"]
        CTX["Masked-state context MLP"]
        TG["Token gate<br/>from Z_WN + context"]
        CG["Channel gate<br/>from context only"]
        ROUTE["Z̃_WN = Z_WN ⊙ g_token ⊙ g_channel"]
        ID["No GPT field available<br/>both gates forced to 1"]
    end

    GPT --> CTX
    PROJ --> TG
    CTX --> TG
    CTX --> CG
    PROJ --> ROUTE
    TG --> ROUTE
    CG --> ROUTE
    GPT -. "all-zero state mask" .-> ID
    ID --> ROUTE

    subgraph ENCODER["Forecast representation"]
        CLS["Prepend always-valid CLS"]
        TF["Non-causal Transformer encoder"]
        MEMORY["Routed WeatherNext memory"]
        GLOBAL["CLS global state"]
    end

    ROUTE --> CLS
    CLS --> TF
    PADDING --> TF
    TF --> MEMORY
    TF --> GLOBAL
```

```math
g_{token}=2\sigma\!\left(f_{token}(Z_{WN}+f_{ctx}(z_{GPT},m_{GPT}))\right)
```

```math
g_{channel}=2\sigma\!\left(f_{channel}(f_{ctx}(z_{GPT},m_{GPT}))\right)
```

```math
\tilde Z_{WN}=Z_{WN}\odot g_{token}\odot g_{channel}
```

마지막 gate layer의 weight와 bias는 0으로 초기화되어 첫 forward는 `2σ(0)=1`입니다. Random mask는 Router 뒤가 아니라 projection 전에 적용되고, padding mask는 Router가 아니라 Transformer attention에 적용됩니다.
