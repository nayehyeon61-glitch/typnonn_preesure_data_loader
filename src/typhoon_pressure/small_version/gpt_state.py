"""Optional GPT Structured-Output state extraction and cached training features."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field

from .config import GPTStateConfig
from .dataset import WeatherNextDualTargetDataset


STATE_FIELDS = (
    "steering_eastward_score",
    "steering_northward_score",
    "recurvature_score",
    "intensification_score",
    "subtropical_high_influence",
    "monsoon_influence",
    "east_asia_approach_risk",
    "track_uncertainty",
    "intensity_uncertainty",
    "confidence",
)


class GPTSynopticState(BaseModel):
    """Fixed schema returned by the OpenAI Responses API."""

    steering_eastward_score: float = Field(ge=-1.0, le=1.0)
    steering_northward_score: float = Field(ge=-1.0, le=1.0)
    recurvature_score: float = Field(ge=-1.0, le=1.0)
    intensification_score: float = Field(ge=-1.0, le=1.0)
    subtropical_high_influence: float = Field(ge=-1.0, le=1.0)
    monsoon_influence: float = Field(ge=-1.0, le=1.0)
    east_asia_approach_risk: float = Field(ge=0.0, le=1.0)
    track_uncertainty: float = Field(ge=0.0, le=1.0)
    intensity_uncertainty: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    def as_vector(self) -> np.ndarray:
        return np.asarray([getattr(self, field) for field in STATE_FIELDS], dtype=np.float32)


@dataclass(frozen=True)
class GPTStateRecord:
    values: np.ndarray
    mask: np.ndarray

    @classmethod
    def from_state(cls, state: GPTSynopticState) -> "GPTStateRecord":
        values = state.as_vector()
        return cls(values=values, mask=np.ones_like(values, dtype=np.float32))

    @classmethod
    def missing(cls, state_dim: int = len(STATE_FIELDS)) -> "GPTStateRecord":
        return cls(
            values=np.zeros(state_dim, dtype=np.float32),
            mask=np.zeros(state_dim, dtype=np.float32),
        )


def build_gpt_state_summary(
    sample: dict,
    history_feature_names: list[str] | tuple[str, ...],
) -> dict:
    """Summarize typhoon/high-pressure history before GPT dynamic conditioning."""
    history = sample["history"].detach().cpu().numpy()
    history_mask = sample["history_mask"].detach().cpu().numpy().astype(bool)
    history_summary = {}
    for index, name in enumerate(history_feature_names):
        valid = history_mask[:, index]
        if valid.any():
            sequence = history[:, index][valid]
            history_summary[name] = {
                "latest": round(float(sequence[-1]), 4),
                "change": round(float(sequence[-1] - sequence[0]), 4),
                "valid_fraction": round(float(valid.mean()), 4),
            }

    return {
        "storm_id": str(sample["storm_id"]),
        "init_time_ns": int(sample["init_time_ns"]),
        "typhoon_dynamic_history": history_summary,
    }


class OpenAIStateExtractor:
    """Call GPT once per initialization and return a schema-validated numeric state."""

    def __init__(self, config: GPTStateConfig = GPTStateConfig(), client=None):
        self.config = config
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client

    def extract(self, deterministic_summary: dict) -> GPTStateRecord:
        payload = json.dumps(deterministic_summary, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        response = self.client.responses.parse(
            model=self.config.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Extract a compact tropical-cyclone synoptic state from only the supplied numerical summary. "
                        "Do not invent unavailable observations. Use scores near zero and high uncertainty when evidence "
                        "is weak or masked. Positive steering scores indicate eastward or northward influence; negative "
                        "scores indicate the opposite. Interpret surrounding-high position and pressure features as "
                        "steering context when present. Return only the requested structured state."
                    ),
                },
                {"role": "user", "content": payload},
            ],
            text_format=GPTSynopticState,
        )
        if response.output_parsed is None:
            raise RuntimeError("GPT state extraction returned no parsed structured output")
        return GPTStateRecord.from_state(response.output_parsed)


def _safe_storm_id(storm_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(storm_id))
    if not safe:
        raise ValueError("storm_id produced an empty filename")
    return safe


def save_gpt_state(
    record: GPTStateRecord,
    output_dir: str | Path,
    *,
    storm_id: str,
    init_time_ns: int,
    status: str = "ok",
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_storm_id(storm_id)}__{int(init_time_ns)}.npz"
    path = output / filename
    np.savez_compressed(path, values=record.values, mask=record.mask, fields=np.asarray(STATE_FIELDS))
    manifest_path = output / "manifest.csv"
    row = pd.DataFrame([{
        "storm_id": str(storm_id), "init_time_ns": int(init_time_ns),
        "file": filename, "status": status,
    }])
    if manifest_path.exists():
        current = pd.read_csv(manifest_path)
        keep = ~(
            (current["storm_id"].astype(str) == str(storm_id))
            & (current["init_time_ns"].astype("int64") == int(init_time_ns))
        )
        row = pd.concat((current.loc[keep], row), ignore_index=True)
    row.to_csv(manifest_path, index=False)
    return path


class DirectoryGPTStateStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        manifest = pd.read_csv(self.directory / "manifest.csv")
        self.files = {
            (str(row.storm_id), int(row.init_time_ns)): self.directory / str(row.file)
            for row in manifest.itertuples(index=False)
        }
        if not self.files:
            raise ValueError("GPT state manifest is empty")
        first = self.load(*next(iter(self.files)))
        self.state_dim = len(first.values)

    def contains(self, storm_id: str, init_time_ns: int) -> bool:
        return (str(storm_id), int(init_time_ns)) in self.files

    def load(self, storm_id: str, init_time_ns: int) -> GPTStateRecord:
        key = (str(storm_id), int(init_time_ns))
        if key not in self.files:
            raise KeyError(f"No GPT state for {key}")
        with np.load(self.files[key], allow_pickle=False) as data:
            return GPTStateRecord(
                values=data["values"].astype(np.float32),
                mask=data["mask"].astype(np.float32),
            )

    def missing_keys(self, keys) -> list[tuple[str, int]]:
        return sorted(
            (str(storm_id), int(init_time_ns))
            for storm_id, init_time_ns in keys
            if not self.contains(storm_id, init_time_ns)
        )

    def validate_coverage(self, keys, *, require_valid: bool = False) -> dict[str, int]:
        """Require a cache entry for every WeatherNext token key before training."""
        normalized = [(str(storm_id), int(init_time_ns)) for storm_id, init_time_ns in keys]
        missing = self.missing_keys(normalized)
        if missing:
            preview = ", ".join(map(str, missing[:5]))
            raise ValueError(
                f"GPT state cache is missing {len(missing)} WeatherNext samples: {preview}. "
                "Run build-gpt-state-cache first with matching dataset/history settings."
            )
        masked = 0
        for key in normalized:
            record = self.load(*key)
            if record.values.shape != (self.state_dim,) or record.mask.shape != (self.state_dim,):
                raise ValueError(f"GPT state {key} has inconsistent dimensions")
            if not np.all(record.mask > 0):
                masked += 1
        if require_valid and masked:
            raise ValueError(
                f"GPT state cache contains {masked} masked/API-failure entries; "
                "rebuild the cache or omit --require-valid-gpt-states"
            )
        return {"keys": len(normalized), "masked": masked}


class WeatherNextGPTDualTargetDataset(WeatherNextDualTargetDataset):
    """Attach cached GPT state; absent samples become all-zero masked states."""

    def __init__(self, *args, gpt_state_store: DirectoryGPTStateStore, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpt_state_store = gpt_state_store

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.gpt_state_store.contains(sample["storm_id"], sample["init_time_ns"]):
            state = self.gpt_state_store.load(sample["storm_id"], sample["init_time_ns"])
        else:
            state = GPTStateRecord.missing(self.gpt_state_store.state_dim)
        sample["gpt_state_values"] = torch.from_numpy(state.values)
        sample["gpt_state_mask"] = torch.from_numpy(state.mask)
        return sample
