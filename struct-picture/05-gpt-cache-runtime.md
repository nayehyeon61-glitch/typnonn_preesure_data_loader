# 05. GPT API, Cache, and Runtime Fallback

```mermaid
flowchart TB
    START["build-gpt-state-cache"]
    KEY{"OPENAI_API_KEY available?"}
    SUMMARY["Per-sample deterministic summary"]
    CALL["responses.parse<br/>schema-constrained request"]
    VALID{"output_parsed valid?"}
    OK["values=10D, mask=1<br/>status=ok"]
    POLICY{"--on-error"]
    MASKED["values=0, mask=0<br/>status=masked:error-type"]
    RAISE["Raise immediately"]
    CACHE["NPZ records + manifest.csv"]

    START --> KEY
    KEY -->|Yes| SUMMARY
    KEY -->|No| POLICY
    SUMMARY --> CALL
    CALL --> VALID
    VALID -->|Yes| OK
    VALID -->|No or API error| POLICY
    POLICY -->|mask| MASKED
    POLICY -->|raise| RAISE
    OK --> CACHE
    MASKED --> CACHE

    subgraph TRAIN["train-weathernext-transformer"]
        ARG{"--gpt-state-dir supplied?"}
        STORE["Load manifest and infer state_dim"]
        SAMPLE{"Sample cache record exists?"}
        STATE{"Any state-mask field valid?"}
        ACTIVE["FiLM active + Router active"]
        IDENTITY["FiLM identity + Router identity"]
        DISABLED["gpt_state_dim=0<br/>both GPT adapters not constructed"]
    end

    CACHE --> ARG
    ARG -->|No| DISABLED
    ARG -->|Yes| STORE
    STORE --> SAMPLE
    SAMPLE -->|No| IDENTITY
    SAMPLE -->|Yes| STATE
    STATE -->|Yes| ACTIVE
    STATE -->|No, including masked status| IDENTITY
```

| 상황 | History FiLM | Forecast Router | 의미 |
|---|---|---|---|
| `--gpt-state-dir` 없음 | module 없음 | module 없음 | GPT 기능 전체 비활성 |
| 정상 cache, mask 일부/전체 유효 | 활성 | 활성 | mask도 adapter 입력에 포함 |
| cache 누락 또는 all-zero mask | `γ=β=0` | `g_token=g_channel=1` | exact identity fallback |

GPT는 작동 여부를 그림의 `API 실패 / cache 누락` 문구만으로 판단할 수 없습니다. 실제 연결 확인은 cache 생성 출력에서 `completed > 0`, `masked = 0`, 그리고 `manifest.csv`의 `status=ok`를 함께 확인해야 합니다. 연결 진단 시에는 `--on-error raise`가 오류를 숨기지 않습니다.
