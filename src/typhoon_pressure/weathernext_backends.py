from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import xarray as xr


class WeatherNextBackend(str, Enum):
    TRAINABLE = "trainable"
    PRETRAINED = "pretrained"
    FLOW_MATCHING = "flow_matching"
    API = "api"


class RolloutModel(Protocol):
    def rollout(self, initial_state: xr.Dataset, horizon_hours: int) -> xr.Dataset: ...


class TrainableRolloutModel(RolloutModel, Protocol):
    def fit(self, training_data: Any, **kwargs) -> Any: ...


class WeatherNextAPIClient(Protocol):
    def forecast(
        self,
        initial_state: xr.Dataset,
        horizon_hours: int,
        *,
        model_id: str,
    ) -> xr.Dataset: ...


@dataclass(frozen=True)
class WeatherNextBackendConfig:
    backend: WeatherNextBackend | str
    model_id: str = "WeatherNext2_<2025"
    release: str = "v0.3.0"
    checkpoint: str | None = None
    model_variant: str | None = None
    api_provider: str | None = None
    training_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def backend_type(self) -> WeatherNextBackend:
        try:
            return WeatherNextBackend(self.backend)
        except ValueError as exc:
            choices = ", ".join(item.value for item in WeatherNextBackend)
            raise ValueError(f"Unknown WeatherNext backend {self.backend!r}; choose {choices}") from exc

    def validate(self) -> None:
        if not self.model_id.strip():
            raise ValueError("WeatherNext model_id cannot be empty")
        if not self.release.strip():
            raise ValueError("Pin a WeatherNext release for reproducibility")
        if self.backend_type in {
            WeatherNextBackend.PRETRAINED,
            WeatherNextBackend.FLOW_MATCHING,
        } and not self.checkpoint:
            raise ValueError(
                f"{self.backend_type.value} backend requires a checkpoint identifier or path"
            )
        if self.backend_type is WeatherNextBackend.API and not self.api_provider:
            raise ValueError("api backend requires api_provider provenance")


class _RunnerMetadata:
    config: WeatherNextBackendConfig

    def provenance(self) -> dict[str, str | None]:
        return {
            "weathernext_backend": self.config.backend_type.value,
            "weathernext_model_id": self.config.model_id,
            "weathernext_model_variant": self.config.model_variant,
            "weathernext_release": self.config.release,
            "weathernext_checkpoint": self.config.checkpoint,
            "weathernext_api_provider": self.config.api_provider,
        }


@dataclass
class TrainableWeatherNextRunner(_RunnerMetadata):
    config: WeatherNextBackendConfig
    model: TrainableRolloutModel
    training_data: Any
    _prepared: bool = field(default=False, init=False)

    def fit(self) -> Any:
        result = self.model.fit(self.training_data, **self.config.training_kwargs)
        self._prepared = True
        return result

    def rollout(self, initial_state: xr.Dataset, horizon_hours: int) -> xr.Dataset:
        if not self._prepared:
            raise RuntimeError("trainable WeatherNext backend requires runner.fit() before rollout")
        return self.model.rollout(initial_state, horizon_hours)


@dataclass
class PretrainedWeatherNextRunner(_RunnerMetadata):
    config: WeatherNextBackendConfig
    model: RolloutModel

    def rollout(self, initial_state: xr.Dataset, horizon_hours: int) -> xr.Dataset:
        return self.model.rollout(initial_state, horizon_hours)


@dataclass
class FrozenFlowMatchingRunner:
    """Inference-only bridge to a monthly climate_diffusion checkpoint."""

    config: WeatherNextBackendConfig
    model: RolloutModel
    inference_only: bool = field(default=True, init=False)

    def provenance(self) -> dict[str, str | int | bool | None]:
        model_provenance = getattr(self.model, "provenance", None)
        values = dict(model_provenance()) if callable(model_provenance) else {}
        values.update(
            {
                "forecast_backend": "flow_matching",
                "forecast_checkpoint": self.config.checkpoint,
                "forecast_checkpoint_kind": "flow_matching",
                "forecast_release": self.config.release,
                "inference_only": True,
            }
        )
        return values

    def rollout(self, initial_state: xr.Dataset, horizon_hours: int) -> xr.Dataset:
        return self.model.rollout(initial_state, horizon_hours)


@dataclass
class APIWeatherNextRunner(_RunnerMetadata):
    config: WeatherNextBackendConfig
    client: WeatherNextAPIClient

    def rollout(self, initial_state: xr.Dataset, horizon_hours: int) -> xr.Dataset:
        return self.client.forecast(
            initial_state, horizon_hours, model_id=self.config.model_id
        )


def build_weathernext_runner(
    config: WeatherNextBackendConfig,
    *,
    trainable_model: TrainableRolloutModel | None = None,
    training_data: Any = None,
    pretrained_model: RolloutModel | None = None,
    api_client: WeatherNextAPIClient | None = None,
):
    """Select exactly one execution path while preserving one rollout contract."""
    config.validate()
    backend = config.backend_type
    if backend is WeatherNextBackend.TRAINABLE:
        if trainable_model is None or training_data is None:
            raise ValueError("trainable backend requires trainable_model and training_data")
        return TrainableWeatherNextRunner(config, trainable_model, training_data)
    if backend is WeatherNextBackend.PRETRAINED:
        if pretrained_model is None:
            from .weathernext_official import OfficialWeatherNextRunner

            pretrained_model = OfficialWeatherNextRunner(
                model_name=config.model_variant or config.model_id,
                checkpoint_path=config.checkpoint,
                release=config.release,
            )
        return PretrainedWeatherNextRunner(config, pretrained_model)
    if backend is WeatherNextBackend.FLOW_MATCHING:
        if pretrained_model is None:
            try:
                from climate_diffusion import FlowMatchingWeatherRunner
            except ImportError as exc:
                raise ImportError(
                    "flow_matching backend requires the climate-diffusion package"
                ) from exc
            pretrained_model = FlowMatchingWeatherRunner(config.checkpoint)
        return FrozenFlowMatchingRunner(config, pretrained_model)
    if api_client is None:
        raise ValueError("api backend requires api_client")
    return APIWeatherNextRunner(config, api_client)
