# 04. Dual-loss Training with Adaptive Distribution Sampling

```mermaid
flowchart TB
    subgraph SHARED["Shared trainable representation"]
        FILM["History FiLM"]
        GRU["GRU history encoder"]
        ROUTER["GPTForecastRouter"]
        TF["WeatherNext Transformer"]
        MEM["Routed 0-15 day forecast memory"]
        FUSION["Fusion state"]
    end

    FILM --> GRU
    ROUTER --> TF
    TF --> MEM
    GRU --> FUSION
    MEM --> FUSION

    subgraph DIST["Loss 1. Day 15-30 adaptive stochastic distribution"]
        QUERY["16 learned future queries<br/>Day 15 ... Day 30"]
        DEC["Cross-attention future decoder"]
        FUT["Future states h_15:30"]
        GPTQ["GPT semantic state<br/>uncertainty / confidence"]
        QNET["Adaptive process-noise net"]
        CHOL["Cholesky L_t<br/>Q_t = L_t L_t^T"]
        MEAN["Recursive mean dynamics<br/>mu_t"]
        EPS["epsilon_t^k ~ N(0,I)"]
        SAMPLE["K recursive trajectories<br/>x_t^k = x_(t-1)^k + drift_t + L_t epsilon_t^k"]
        KDE["Differentiable global-grid KDE<br/>log-softmax + log-sum-exp mixture"]
        TARGET["Monthly IBTrACS soft target q(d,c)"]
        DMASK["Available-day mask m_d"]
        CE["Sampled distribution CE L_dist"]
    end

    FUSION --> QUERY
    QUERY --> DEC
    MEM --> DEC
    DEC --> FUT
    FUT --> QNET
    GPTQ --> QNET
    QNET --> CHOL
    FUT --> MEAN
    CHOL --> SAMPLE
    MEAN --> SAMPLE
    EPS --> SAMPLE
    SAMPLE --> KDE
    KDE --> CE
    TARGET --> CE
    DMASK --> CE

    subgraph TRACK["Loss 2. East Asia local track"]
        HEAD["Track regression head"]
        PRED["Predicted lat/lon<br/>B x 20 x 2"]
        OBS["IBTrACS future lat/lon"]
        TMASK["Valid and East-Asia mask m_t"]
        KM["Tangent-plane error in km<br/>wrapped longitude"]
        MSE["Normalized track MSE L_track"]
    end

    FUSION --> HEAD
    HEAD --> PRED
    PRED --> KM
    OBS --> KM
    KM --> MSE
    TMASK --> MSE

    TOTAL["L_total = lambda_dist L_dist + lambda_track L_track"]
    CE --> TOTAL
    MSE --> TOTAL

    TOTAL -. "sample-distribution gradient" .-> QNET
    TOTAL -. "sample-distribution gradient" .-> DEC
    TOTAL -. "both losses" .-> FUSION
    TOTAL -. "both losses" .-> GRU
    TOTAL -. "both losses" .-> FILM
    TOTAL -. "both losses" .-> TF
    TOTAL -. "both losses" .-> ROUTER
    TOTAL -. "track only" .-> HEAD
```

## Time-correlated sampling

For lead days `15, 16, ..., 30`, a sample member keeps the same identity through time:

```math
x_{15}^{(k)} = \mu_{15} + L_{15}\epsilon_{15}^{(k)}
```

```math
x_t^{(k)} = x_{t-1}^{(k)} + \Delta\mu_t + L_t\epsilon_t^{(k)},\qquad t>15
```

with

```math
Q_t=L_tL_t^\top,\qquad \epsilon_t^{(k)}\sim\mathcal N(0,I).
```

`L_t` is parameterized so `Q_t` is positive definite. The process-noise condition receives the future WeatherNext/fusion state and, when available, GPT semantic state. Thus the learned spread can adapt to forecast context and semantic uncertainty.

## Differentiable distribution supervision

The `K` sampled trajectories are softly rasterized to the same global grid as the existing IBTrACS distribution target. Instead of a non-differentiable histogram, each sample contributes a smooth spatial kernel. The mixture is evaluated stably in log space using `log-softmax` and `log-sum-exp`, producing `log p_sample(d,c)`.

```math
L_{dist}=-\frac{\sum_d m_d\sum_c q_{d,c}\log p_{sample}(d,c)}{\sum_d m_d}.
```

The local-track branch is unchanged:

```math
L_{track}=\frac{\sum_t m_t(e_{x,t}^2+e_{y,t}^2)}{\sum_t m_t\,(500\text{ km})^2}.
```

The current sampler is **Kalman-inspired adaptive process noise**, not yet a full Kalman measurement-update layer: it learns `Q_t` and propagates uncertainty recursively after the routed WeatherNext representation. A future extension can add explicit pseudo-observation covariance `R_t`, innovation, and Kalman gain.

The direct categorical `distribution_logits` output remains available for backward-compatible diagnostics, but the WeatherNext fusion model's primary distribution loss now uses the sampled trajectory distribution.
