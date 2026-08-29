# 05. GPT API and Cache Runtime

```mermaid
flowchart TB
    START["build-gpt-state-cache 실행"]
    KEY{"OPENAI_API_KEY<br/>설정됨?"}
    SUMMARY["Sample별 deterministic summary"]
    CALL["responses.parse<br/>GPT Structured Output 호출"]
    VALID{"Schema-validated<br/>output_parsed 존재?"}
    OK["10-state values + mask<br/>status=ok"]
    POLICY{"--on-error 정책"}
    MASK["values=0, mask=0<br/>status=masked:error-type"]
    RAISE["오류 즉시 발생<br/>연결 검증용"]
    CACHE["data/gpt_states<br/>sample cache + manifest.csv"]

    START --> KEY
    KEY -->|Yes| SUMMARY
    KEY -->|No| POLICY
    SUMMARY --> CALL
    CALL --> VALID
    VALID -->|Yes| OK
    VALID -->|No 또는 API 실패| POLICY
    POLICY -->|mask| MASK
    POLICY -->|raise| RAISE
    OK --> CACHE
    MASK --> CACHE

    subgraph TRAINING["학습 시점"]
        direction TB
        ARG{"--gpt-state-dir<br/>지정됨?"}
        LOAD{"해당 sample<br/>cache 존재?"}
        ACTIVE["GPT conditioning 활성<br/>FiLM → Dynamic History"]
        IDENTITY["Zero state mask<br/>identity History → GRU"]
        DISABLED["gpt_state_dim=0<br/>GPT conditioning 비활성"]
    end

    CACHE --> ARG
    ARG -->|Yes| LOAD
    ARG -->|No| DISABLED
    LOAD -->|Yes, status=ok| ACTIVE
    LOAD -->|No 또는 masked| IDENTITY
```

GPT는 학습 loop 안에서 반복 호출하지 않습니다. 먼저 cache를 만들고 학습에서는 cache만 읽습니다. 최초 연결 검증에는 `--on-error raise`를 사용하고, `completed > 0`, `masked = 0`, `manifest.csv`의 `status=ok`로 확인합니다.
