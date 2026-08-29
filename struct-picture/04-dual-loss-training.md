# 04. Dual-loss Training and Gradient Paths

```mermaid
flowchart TB
    subgraph SHARED["Shared trainable representation"]
        FILM["History FiLM"]
        GRU["GRU history encoder"]
        ROUTER["GPTForecastRouter"]
        TF["WeatherNext Transformer"]
        MEM["Routed forecast memory"]
        FUSION["Fusion state"]
    end

    FILM --> GRU
    ROUTER --> TF
    TF --> MEM
    GRU --> FUSION
    MEM --> FUSION

    subgraph DIST["Loss 1. Day 15–30 global distribution"]
        QUERY["16 learned future queries<br/>fusion-conditioned"]
        DEC["Cross-attention decoder<br/>memory = routed forecast memory"]
        LOGIT["Grid logits<br/>B×16×2592"]
        TARGET["Monthly IBTrACS soft target q(d,c)"]
        DMASK["Available-day mask m_d"]
        CE["Soft-label cross entropy L_CE"]
    end

    FUSION --> QUERY
    QUERY --> DEC
    MEM --> DEC
    DEC --> LOGIT
    LOGIT --> CE
    TARGET --> CE
    DMASK --> CE

    subgraph TRACK["Loss 2. East Asia local track"]
        HEAD["Track regression head"]
        PRED["Predicted lat and lon<br/>B×20×2"]
        OBS["IBTrACS future lat and lon"]
        TMASK["Valid ∧ East-Asia region mask m_t"]
        KM["Tangent-plane error in km<br/>wrapped longitude"]
        MSE["Normalized track MSE L_MSE"]
        RMSE["Logging only<br/>track RMSE in km"]
    end

    FUSION --> HEAD
    HEAD --> PRED
    PRED --> KM
    OBS --> KM
    KM --> MSE
    TMASK --> MSE
    MSE --> RMSE

    TOTAL["L_total=λ_dist L_CE + λ_track L_MSE"]
    CE --> TOTAL
    MSE --> TOTAL

    TOTAL -. "distribution + track" .-> FUSION
    TOTAL -. "both losses via shared state" .-> GRU
    TOTAL -. "both losses via shared state" .-> FILM
    TOTAL -. "both losses via CLS or memory" .-> TF
    TOTAL -. "both losses via CLS or memory" .-> ROUTER
    TOTAL -. "distribution only" .-> DEC
    TOTAL -. "track only" .-> HEAD
```

```math
L_{CE}=-\frac{\sum_d m_d\sum_c q_{d,c}\log\operatorname{softmax}(z_{d})_c}{\sum_d m_d}
```

```math
L_{MSE}=\frac{\sum_t m_t(e_{x,t}^2+e_{y,t}^2)}{\sum_t m_t\,(500\text{ km})^2}
```

분포 branch는 routed memory에 직접 cross-attention하고 fusion state로 query를 조건화합니다. 경로 branch는 fusion state를 사용합니다. 따라서 두 loss 모두 Router와 Transformer에 도달하지만, 도달 경로는 서로 다릅니다. 분포는 memory와 CLS/fusion 두 경로, track은 CLS/fusion 경로를 사용합니다.

유효한 distribution day 또는 track point가 전혀 없는 batch는 해당 loss를 미분 가능한 0으로 반환합니다.
