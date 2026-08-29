# 03. GPT Semantic State: History FiLM + Forecast Router

GPT state는 같은 cache record에서 읽히지만 두 개의 서로 다른 trainable adapter를 제어합니다.

```mermaid
flowchart TB
    RAW["Typhoon and pressure history<br/>B×8×F"]
    HMASK["History feature mask"]

    subgraph EXTRACT["Offline GPT state extraction"]
        SUMMARY["Deterministic summary<br/>latest, change, valid fraction"]
        API["GPT Structured Output API"]
        CACHE["10D state cache<br/>values z_GPT and mask m_GPT"]
    end

    RAW --> SUMMARY
    HMASK --> SUMMARY
    SUMMARY --> API
    API --> CACHE

    subgraph PATH1["Path A. History FiLM"]
        ZERO["Invalid history values → 0"]
        HPROJ["History projection<br/>concat zeroed values and mask → h"]
        FILM["FiLM adapter<br/>concat masked z_GPT and m_GPT"]
        PARAM["γ=0.5 tanh γ_raw<br/>β=tanh β_raw"]
        DYNAMIC["h'=(1+γ)h+β"]
        GRU["Standard GRU over projected sequence"]
        HIDDEN["History hidden state"]
    end

    RAW --> ZERO
    HMASK --> ZERO
    ZERO --> HPROJ
    HMASK --> HPROJ
    CACHE --> FILM
    FILM --> PARAM
    HPROJ --> DYNAMIC
    PARAM --> DYNAMIC
    DYNAMIC --> GRU
    GRU --> HIDDEN

    subgraph PATH2["Path B. WeatherNext Router"]
        ZWN["Projected WeatherNext tokens Z_WN"]
        CTX["GPT context MLP<br/>concat masked z_GPT and m_GPT"]
        TG["Token gate<br/>f(Z_WN + context)"]
        CG["Channel gate<br/>f(context)"]
        ROUTED["Z̃_WN=Z_WN⊙g_token⊙g_channel"]
        TF["Transformer encoder"]
        WMEM["Routed memory and CLS"]
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

    MISSING["API failure, masked record, or cache miss<br/>z_GPT=0 and m_GPT=0"]
    ID1["FiLM identity<br/>γ=β=0"]
    ID2["Router identity<br/>g_token=g_channel=1"]
    MISSING --> ID1
    MISSING --> ID2
    ID1 --> DYNAMIC
    ID2 --> ROUTED

    FUSION["Fusion<br/>history hidden + forecast CLS"]
    HIDDEN --> FUSION
    WMEM --> FUSION
```

`history_mask`는 GRU의 `pack_padded_sequence`나 recurrent-step skip으로 사용되지 않습니다. 실제 구현은 결측값을 0으로 바꾸고 mask 자체를 projection 입력에 포함하므로, 정확한 표현은 **mask-aware history projection + standard GRU**입니다.

State mask 중 하나라도 유효하면 GPT adapter가 활성화되고, 부분 mask 자체도 context 입력에 포함됩니다. 모든 state field가 무효일 때만 두 adapter가 강제로 identity가 됩니다.
