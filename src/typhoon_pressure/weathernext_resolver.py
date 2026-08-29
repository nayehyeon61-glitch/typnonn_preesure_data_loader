from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import xarray as xr

from .weathernext_backends import (
    WeatherNextBackendConfig,
    build_weathernext_runner,
)


class CheckpointOrigin(str, Enum):
    FINETUNED = "finetuned"
    OFFICIAL = "official"
    DOWNLOADED = "downloaded"
    API = "api"
    TRAINED = "trained"


class WeatherNextExecutionMode(str, Enum):
    AUTO = "auto"
    PRETRAINED = "pretrained"
    API = "api"
    TRAINABLE = "trainable"


class CheckpointDownloader(Protocol):
    def download(
        self,
        *,
        model_variant: str,
        release: str,
    ) -> Path:
        ...


@dataclass(frozen=True)
class WeatherNextSelectionConfig:
    """Priority-based frozen WeatherNext source selection.

    Resolution order:
    1) local fine-tuned checkpoint
    2) local official/pretrained checkpoint
    3) downloaded official checkpoint
    4) API forecast fallback

    ``pretrained`` always performs frozen inference. ``trainable`` is opt-in and
    requires an injected trainable model plus training data.
    """

    model_variant: str = "WeatherNext2"
    model_id: str = "WeatherNext2_<2025"
    release: str = "v0.3.0"
    finetuned_checkpoint: str | None = None
    pretrained_checkpoint: str | None = None
    allow_download: bool = True
    allow_api_fallback: bool = True
    api_provider: str | None = None
    execution_mode: WeatherNextExecutionMode | str = WeatherNextExecutionMode.AUTO
    training_kwargs: dict = None

    @property
    def mode(self) -> WeatherNextExecutionMode:
        return WeatherNextExecutionMode(self.execution_mode)


@dataclass
class ResolvedWeatherNext:
    """Resolved source that also satisfies the common rollout runner contract."""

    runner: object
    origin: CheckpointOrigin
    checkpoint: str | None
    frozen: bool = True

    def rollout(self, initial_state: xr.Dataset, horizon_hours: int) -> xr.Dataset:
        return self.runner.rollout(initial_state, horizon_hours)

    def provenance(self) -> dict[str, str | None]:
        attrs: dict[str, str | None] = {}
        base = getattr(self.runner, "provenance", None)
        if callable(base):
            attrs.update(base())
        attrs.update(
            {
                "weathernext_weight_origin": self.origin.value,
                "weathernext_resolved_checkpoint": self.checkpoint,
                "weathernext_frozen": str(self.frozen).lower(),
            }
        )
        return attrs


def _existing_checkpoint(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_file() else None


def _checkpoint_runner(config: WeatherNextSelectionConfig, path: Path):
    return build_weathernext_runner(
        WeatherNextBackendConfig(
            backend="pretrained",
            model_id=config.model_id,
            model_variant=config.model_variant,
            release=config.release,
            checkpoint=str(path),
        )
    )


def resolve_weathernext(
    config: WeatherNextSelectionConfig,
    *,
    downloader: CheckpointDownloader | None = None,
    api_client=None,
    trainable_model=None,
    training_data=None,
    fit_trainable: bool = True,
) -> ResolvedWeatherNext:
    """Resolve one explicit execution mode, or use the legacy auto priority."""

    if config.mode is WeatherNextExecutionMode.TRAINABLE:
        runner = build_weathernext_runner(
            WeatherNextBackendConfig(
                backend="trainable",
                model_id=config.model_id,
                model_variant=config.model_variant,
                release=config.release,
                training_kwargs=config.training_kwargs or {},
            ),
            trainable_model=trainable_model,
            training_data=training_data,
        )
        if fit_trainable:
            runner.fit()
        return ResolvedWeatherNext(
            runner=runner,
            origin=CheckpointOrigin.TRAINED,
            checkpoint=config.finetuned_checkpoint,
            frozen=False,
        )

    if config.mode is WeatherNextExecutionMode.API:
        if api_client is None:
            raise ValueError("api execution mode requires an injected api_client")
        if not config.api_provider:
            raise ValueError("api_provider is required for WeatherNext API mode")
        runner = build_weathernext_runner(
            WeatherNextBackendConfig(
                backend="api",
                model_id=config.model_id,
                model_variant=config.model_variant,
                release=config.release,
                api_provider=config.api_provider,
            ),
            api_client=api_client,
        )
        return ResolvedWeatherNext(
            runner=runner,
            origin=CheckpointOrigin.API,
            checkpoint=None,
            frozen=True,
        )

    finetuned = _existing_checkpoint(config.finetuned_checkpoint)
    if finetuned is not None:
        return ResolvedWeatherNext(
            runner=_checkpoint_runner(config, finetuned),
            origin=CheckpointOrigin.FINETUNED,
            checkpoint=str(finetuned),
        )

    pretrained = _existing_checkpoint(config.pretrained_checkpoint)
    if pretrained is not None:
        return ResolvedWeatherNext(
            runner=_checkpoint_runner(config, pretrained),
            origin=CheckpointOrigin.OFFICIAL,
            checkpoint=str(pretrained),
        )

    if config.allow_download and downloader is not None:
        downloaded = Path(
            downloader.download(
                model_variant=config.model_variant,
                release=config.release,
            )
        ).expanduser()
        if not downloaded.is_file():
            raise FileNotFoundError(
                "WeatherNext downloader did not produce a checkpoint: "
                f"{downloaded}"
            )
        downloaded = downloaded.resolve()
        return ResolvedWeatherNext(
            runner=_checkpoint_runner(config, downloaded),
            origin=CheckpointOrigin.DOWNLOADED,
            checkpoint=str(downloaded),
        )

    if (
        config.mode is WeatherNextExecutionMode.AUTO
        and config.allow_api_fallback
        and api_client is not None
    ):
        if not config.api_provider:
            raise ValueError("api_provider is required for WeatherNext API fallback")
        runner = build_weathernext_runner(
            WeatherNextBackendConfig(
                backend="api",
                model_id=config.model_id,
                model_variant=config.model_variant,
                release=config.release,
                api_provider=config.api_provider,
            ),
            api_client=api_client,
        )
        return ResolvedWeatherNext(
            runner=runner,
            origin=CheckpointOrigin.API,
            checkpoint=None,
        )

    attempted: list[str] = []
    if config.finetuned_checkpoint:
        attempted.append(f"finetuned={config.finetuned_checkpoint}")
    if config.pretrained_checkpoint:
        attempted.append(f"pretrained={config.pretrained_checkpoint}")
    if config.allow_download:
        attempted.append("download")
    if config.mode is WeatherNextExecutionMode.AUTO and config.allow_api_fallback:
        attempted.append("api")
    detail = ", ".join(attempted) if attempted else "no source configured"
    raise RuntimeError(f"No usable WeatherNext source found ({detail})")
