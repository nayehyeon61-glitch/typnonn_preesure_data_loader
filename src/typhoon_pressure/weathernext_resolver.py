from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from .weathernext_backends import (
    WeatherNextBackendConfig,
    build_weathernext_runner,
)


class CheckpointOrigin(str, Enum):
    FINETUNED = "finetuned"
    OFFICIAL = "official"
    DOWNLOADED = "downloaded"
    API = "api"


class CheckpointDownloader(Protocol):
    def download(
        self,
        *,
        model_variant: str,
        release: str,
    ) -> Path:
        ...


@dataclass
class WeatherNextSelectionConfig:

    model_variant: str = "WeatherNext2"
    model_id: str = "WeatherNext2_<2025"
    release: str = "v0.3.0"

    finetuned_checkpoint: str | None = None
    pretrained_checkpoint: str | None = None

    allow_download: bool = True
    allow_api_fallback: bool = True

    api_provider: str | None = None
@dataclass
class ResolvedWeatherNext:
    runner: object
    origin: CheckpointOrigin
    checkpoint: str | None


def resolve_weathernext(
    config: WeatherNextSelectionConfig,
    *,
    downloader: CheckpointDownloader | None = None,
    api_client=None,
) -> ResolvedWeatherNext:

    # 1. 우리가 fine-tuning한 weight
    if config.finetuned_checkpoint:

        path = Path(
            config.finetuned_checkpoint
        ).expanduser()

        if path.is_file():

            backend_config = WeatherNextBackendConfig(
                backend="pretrained",
                model_id=config.model_id,
                model_variant=config.model_variant,
                release=config.release,
                checkpoint=str(path),
            )

            runner = build_weathernext_runner(
                backend_config
            )

            return ResolvedWeatherNext(
                runner=runner,
                origin=CheckpointOrigin.FINETUNED,
                checkpoint=str(path),
            )

    # 2. local official pretrained
    if config.pretrained_checkpoint:

        path = Path(
            config.pretrained_checkpoint
        ).expanduser()

        if path.is_file():

            backend_config = WeatherNextBackendConfig(
                backend="pretrained",
                model_id=config.model_id,
                model_variant=config.model_variant,
                release=config.release,
                checkpoint=str(path),
            )

            runner = build_weathernext_runner(
                backend_config
            )

            return ResolvedWeatherNext(
                runner=runner,
                origin=CheckpointOrigin.OFFICIAL,
                checkpoint=str(path),
            )

    # 3. online에서 official checkpoint download
    if config.allow_download and downloader:

        path = downloader.download(
            model_variant=config.model_variant,
            release=config.release,
        )

        backend_config = WeatherNextBackendConfig(
            backend="pretrained",
            model_id=config.model_id,
            model_variant=config.model_variant,
            release=config.release,
            checkpoint=str(path),
        )

        runner = build_weathernext_runner(
            backend_config
        )

        return ResolvedWeatherNext(
            runner=runner,
            origin=CheckpointOrigin.DOWNLOADED,
            checkpoint=str(path),
        )

    # 4. 최종 API fallback
    if (
        config.allow_api_fallback
        and api_client is not None
    ):

        backend_config = WeatherNextBackendConfig(
            backend="api",
            model_id=config.model_id,
            model_variant=config.model_variant,
            release=config.release,
            api_provider=config.api_provider,
        )

        runner = build_weathernext_runner(
            backend_config,
            api_client=api_client,
        )

        return ResolvedWeatherNext(
            runner=runner,
            origin=CheckpointOrigin.API,
            checkpoint=None,
        )

    raise RuntimeError(
        "No usable WeatherNext source found."
    )
