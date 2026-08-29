# 06. GPT Forecast Router

```mermaid
flowchart TB
    GPTVAL["GPT values z_GPT<br/>B×G"]
    GPTMASK["GPT mask m_GPT<br/>B×G"]
    MASKED["masked_z = where(mask, z, 0)"]
    CONCAT["concat(masked_z, m_GPT)"]
    CTX["Context MLP<br/>B×d"]

    GPTVAL --> MASKED
    GPTMASK --> MASKED
    MASKED --> CONCAT
    GPTMASK --> CONCAT
    CONCAT --> CTX

    ZWN["Projected tokens Z_WN<br/>B×N×d"]

    subgraph GATES["Trainable gates"]
        ADD["Z_WN + context"]
        TGNET["Token-gate MLP"]
        TG["raw token gate<br/>2σ(logit), B×N×1"]
        CGNET["Channel-gate Linear"]
        CG["raw channel gate<br/>2σ(logit), B×1×d"]
    end

    ZWN --> ADD
    CTX --> ADD
    ADD --> TGNET
    TGNET --> TG
    CTX --> CGNET
    CGNET --> CG

    AVAILABLE["a = any(m_GPT)<br/>B×1×1"]
    FORCE_T["g_token = 1 + a(raw−1)"]
    FORCE_C["g_channel = 1 + a(raw−1)"]
    GPTMASK --> AVAILABLE
    AVAILABLE --> FORCE_T
    AVAILABLE --> FORCE_C
    TG --> FORCE_T
    CG --> FORCE_C

    MUL["Element-wise routing"]
    ROUTED["Z̃_WN<br/>B×N×d"]
    TF["Transformer encoder"]
    ZWN --> MUL
    FORCE_T --> MUL
    FORCE_C --> MUL
    MUL --> ROUTED
    ROUTED --> TF
```

```math
c=f_{ctx}(\operatorname{concat}(z_{GPT}\odot m_{GPT},m_{GPT}))
```

```math
g_{token}=1+a\left(2\sigma(f_{token}(Z_{WN}+c))-1\right)
```

```math
g_{channel}=1+a\left(2\sigma(f_{channel}(c))-1\right)
```

```math
\boxed{\tilde Z_{WN}=Z_{WN}\odot g_{token}\odot g_{channel}}
```

## 초기화와 학습

마지막 token-gate layer와 channel-gate layer의 weight/bias는 0으로 초기화됩니다. 첫 forward에서 두 gate는 정확히 1이며 기존 모델 입력을 보존합니다. 첫 backward에서는 두 마지막 layer가 gradient를 받고, optimizer step 이후 gate가 1에서 벗어나면서 앞단 context/token network에도 gradient가 전달됩니다.

Router 출력에는 active fraction, 유효 token의 평균 gate, channel 평균 gate, sample별 gate map이 포함됩니다. 모든 forecast token이 mask된 경우 token-gate 평균은 해석 가능한 identity 값 `1`로 기록합니다.
