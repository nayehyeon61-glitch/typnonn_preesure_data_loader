# Flow Matching + GPT-DoubleLoss integrated runtime

This document is the execution contract for the integrated system on
`feature/weathernext-resolver`.

## Architecture

```text
ERA5/HRES + IBTrACS
       |
       +--> frozen WeatherNext backend (360 h) -----+
       |                                             |
       +--> frozen Flow Matching backend ------------+--> forecast token cache
                                                      |   + provenance
GPT structured-state cache ---------------------------+--> GPTForecastRouter
                                                          token/channel gates
                                                               |
History --------------------------------------------------------+--> fusion
                                                                    |
                                  exact 360 h endpoint ----------> P15
                                  missing/wrong endpoint --------> learned fallback P15
                                                                    |
                                              AdaptiveDistributionSampler
                                              P15 + time-correlated Q_t
                                                                    |
                                     Day 15--30 sampled distributions
                                                                    |
                         storm-specific NLL + survival + local-track loss
```

## Non-negotiable 360 h / 720 h contract

`P15` means the distribution anchor at Day 15, therefore its semantic lead time
is **exactly 360 hours**. A monthly Flow checkpoint/rollout may legitimately have
a **720 hour** endpoint, but that coordinate is Day 30 and must never be reused
as P15.

The contract is enforced at three layers:

1. Token NPZ stores `endpoint_lead_hours` next to `endpoint_latlon`.
2. Forecast manifest/checkpoint provenance stores backend, checkpoint identity,
   forecast horizon and training checkpoint stores `required_endpoint_hours=360`.
3. Dataset and model runtime validate the endpoint lead. With the default
   `disable` policy, 720 h/missing endpoints keep their forecast tokens and GPT
   Router conditioning but their endpoint mask is set to zero; the sampler then
   uses its learned fallback initial mean. Strict experiments can use `error`
   policy to fail immediately.

Do not relabel or interpolate a 720 h endpoint as 360 h. To use a Flow endpoint
as P15, generate a real 360 h Flow rollout/checkpoint/token cache.

## Frozen backend selection

The forecast backend is an upstream frozen inference provider. Training the
GPT-DoubleLoss model must not update WeatherNext or Flow weights.

- WeatherNext: use a pretrained/fine-tuned checkpoint through the resolver and
  generate a 360 h token cache.
- Flow Matching: select the Flow checkpoint in the forecast resolver/backend and
  generate tokens with provenance. A 360 h Flow rollout can anchor P15; a 720 h
  monthly rollout is conditioning-only unless it also exposes a separately
  validated 360 h endpoint.
- Use `--require-forecast-backend flow_matching` or `pretrained` when an
  experiment must fail rather than accidentally use the other backend.

Never mix checkpoint/backend provenance inside one token directory. Rebuild to a
new cache directory when changing backend, checkpoint, tokenizer schema or
forecast horizon.

## GPT Router

Build the GPT state cache before GPT-enabled training. The structured state is
used twice: history FiLM conditioning and `GPTForecastRouter` token/channel
routing. Missing GPT state is identity/no-op; `--require-valid-gpt-states` makes
API/cache failures fatal instead.

## Adaptive distribution and losses

`AdaptiveDistributionSampler` learns a positive-definite initial covariance
`P15` and transition/process covariance `Q_t`. One sample index is propagated
through all lead days, producing coherent Day-15--30 trajectories rather than
independent daily samples. Survival probability converts conditional storm
location probabilities into unconditional distributions. The objective retains
storm-specific distribution NLL, survival supervision and local-track loss.
Train/validation/test splitting is storm-specific; never split windows from one
storm across partitions.

## Recommended execution

```bash
pip install -e '.[io,test,weathernext]'
pytest -q

# 1) Prepare ERA5/HRES -> forecast tokens using the selected frozen backend.
# The request used for a P15 anchor must have horizon_hours=360.
prepare-weathernext-pipeline ... --horizon-hours 360 --output-dir cache/forecast_360h

# 2) Optional GPT cache, with exactly matching storm/init keys.
build-gpt-state-cache ... --weathernext-token-dir cache/forecast_360h --output-dir cache/gpt_state

# 3) Train GPT-DoubleLoss fusion.
train-weathernext-transformer \
  --integrated data/integrated_typhoon_pressure.parquet \
  --weathernext-token-dir cache/forecast_360h \
  --gpt-state-dir cache/gpt_state \
  --split-manifest data/storm_split.csv \
  --require-forecast-backend pretrained \
  --output checkpoints/gpt_double_loss.pt
```

For Flow Matching replace the forecast cache with one generated from the frozen
Flow backend and use `--require-forecast-backend flow_matching`. If the cache is
720 h monthly Flow, the endpoint must be disabled/fallback for P15; prefer a
separate 360 h Flow cache for the integrated Day-15 experiment.

## Checkpoint and reproducibility checklist

Before comparing experiments record: forecast backend, forecast checkpoint path
and SHA256, checkpoint kind/format/release, forecast horizon, tokenizer
fingerprint/schema, initialization mode, required P15 endpoint lead, split
manifest, train/validation storm IDs, sampling config and random seed. The
training checkpoint already persists the forecast provenance and endpoint
requirement; keep the token manifest with the experiment artifact.

## Failure modes

- `720 h -> P15`: contract violation; endpoint is disabled or strict mode errors.
- legacy token cache without endpoint/provenance: rebuild the cache.
- mixed forecast provenance in one cache: rejected; use separate directories.
- GPT cache key mismatch: rebuild GPT cache from the same forecast/sample keys.
- WeatherNext/Flow checkpoint changed after tokenization: rebuild tokens; do not
  assume old cached tokens represent the new frozen model.
- Test leakage: evaluate only with the storm-specific test partition from the
  split manifest; never tune on test storms.
