"""Runtime endpoint contract for the Day-15 probabilistic anchor.

Forecast tokens may come from heterogeneous frozen backends.  The token sequence
is still useful when a backend only exposes a 720 h monthly endpoint, but that
endpoint must never be interpreted as the Day-15 (360 h) P15 anchor.  This
wrapper preserves forecast/GPT conditioning while disabling only the mismatched
endpoint so the sampler falls back to its learned initial state.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset


class EndpointContractDataset(Dataset):
    """Validate or disable endpoint anchors without discarding forecast tokens.

    ``mismatch_policy='disable'`` is the safe default for mixed WeatherNext/Flow
    experiments: the sample remains trainable, but ``weathernext_endpoint_mask``
    becomes zero and the adaptive sampler uses its learned fallback mean.

    ``mismatch_policy='error'`` fails immediately during construction if any
    cached endpoint is present at the wrong lead time.
    """

    TOLERANCE_HOURS = 1e-3

    def __init__(
        self,
        dataset: Dataset,
        *,
        required_endpoint_hours: float = 360.0,
        mismatch_policy: str = "disable",
    ):
        if required_endpoint_hours <= 0:
            raise ValueError("required_endpoint_hours must be positive")
        if mismatch_policy not in {"disable", "error"}:
            raise ValueError("mismatch_policy must be 'disable' or 'error'")
        self.dataset = dataset
        self.required_endpoint_hours = float(required_endpoint_hours)
        self.mismatch_policy = mismatch_policy
        self.contract_stats = self._scan_contract()

    def __len__(self):
        return len(self.dataset)

    def storm_id_at(self, index: int) -> str:
        if hasattr(self.dataset, "storm_id_at"):
            return str(self.dataset.storm_id_at(index))
        return str(self.dataset[index]["storm_id"])

    def _classify(self, sample: dict) -> tuple[bool, bool, float]:
        mask = sample.get("weathernext_endpoint_mask")
        lead = sample.get("weathernext_endpoint_lead_hours")
        available = bool(float(mask.item() if torch.is_tensor(mask) else (mask or 0.0)) > 0.0)
        lead_hours = float(lead.item() if torch.is_tensor(lead) else (lead or 0.0))
        exact = available and math.isfinite(lead_hours) and abs(
            lead_hours - self.required_endpoint_hours
        ) <= self.TOLERANCE_HOURS
        return available, exact, lead_hours

    def _scan_contract(self) -> dict[str, int]:
        exact = mismatched = missing = 0
        examples: list[str] = []
        for index in range(len(self.dataset)):
            sample = self.dataset[index]
            available, matches, lead_hours = self._classify(sample)
            if matches:
                exact += 1
            elif available:
                mismatched += 1
                if len(examples) < 5:
                    examples.append(
                        f"{sample.get('storm_id')}@{sample.get('init_time_ns')}: {lead_hours:g}h"
                    )
            else:
                missing += 1
        if self.mismatch_policy == "error" and mismatched:
            raise ValueError(
                f"Day-15 P15 requires exact {self.required_endpoint_hours:g}h endpoints; "
                f"found {mismatched} mismatched cached endpoint(s), examples={examples}"
            )
        return {"exact": exact, "mismatched": mismatched, "missing": missing}

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        available, matches, lead_hours = self._classify(sample)
        if available and not matches:
            if self.mismatch_policy == "error":
                raise ValueError(
                    f"Day-15 P15 requires exact {self.required_endpoint_hours:g}h endpoint, "
                    f"received {lead_hours:g}h"
                )
            # Preserve forecast/GPT tokens but make the endpoint unavailable to
            # AdaptiveDistributionSampler.  It then uses fallback_state_head.
            sample["weathernext_endpoint_mask"] = torch.tensor(0.0, dtype=torch.float32)
            sample["weathernext_endpoint_latlon"] = torch.zeros(2, dtype=torch.float32)
        sample["endpoint_contract_match"] = torch.tensor(float(matches), dtype=torch.float32)
        sample["endpoint_contract_disabled"] = torch.tensor(
            float(available and not matches), dtype=torch.float32
        )
        sample["required_endpoint_lead_hours"] = torch.tensor(
            self.required_endpoint_hours, dtype=torch.float32
        )
        return sample
