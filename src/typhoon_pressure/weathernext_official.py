"""Read-only inference adapter for official WeatherNext 2 checkpoints.

The training repository writes ``fgn.CheckPoint`` files.  This module loads
those parameters with the matching official model configuration and exposes the
small ``rollout(initial_state, horizon_hours)`` protocol used by this project.
No optimizer, gradient function, or training entry point exists in this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

SUPPORTED_RELEASE = "v0.3.0"


@dataclass(frozen=True)
class WeatherNextModelSpec:
    name: str
    config_name: str
    resolution_degrees: float
    aliases: tuple[str, ...]


MODEL_SPECS = (
    WeatherNextModelSpec(
        name="WeatherNext2",
        config_name="WeatherNext2",
        resolution_degrees=0.25,
        aliases=("weather-next", "weathernext", "weather-next2", "weathernext2"),
    ),
    WeatherNextModelSpec(
        name="WeatherNextCyclones",
        config_name="WeatherNextCyclones",
        resolution_degrees=0.25,
        aliases=("cyclone", "cyclones", "weather-next-cyclone", "weathernextcyclones"),
    ),
    WeatherNextModelSpec(
        name="WeatherNextCyclones_Mini",
        config_name="WeatherNextCyclones_Mini",
        resolution_degrees=1.0,
        aliases=("mini", "weather-next-mini", "weathernextcyclones-mini"),
    ),
)


COORDINATE_ALIASES = {
    "latitude": "lat",
    "longitude": "lon",
    "valid_time": "time",
}

VARIABLE_ALIASES = {
    "t": "temperature",
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
    "t2m": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "mslp": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "sst": "sea_surface_temperature",
    "lsm": "land_sea_mask",
}


def resolve_model_spec(model_name: str) -> WeatherNextModelSpec:
    """Resolve stable names plus user-facing weather-next/cyclone aliases."""
    value = model_name.strip().lower().replace("_", "-")
    for spec in MODEL_SPECS:
        candidates = {spec.name.lower().replace("_", "-"), *spec.aliases}
        if value in candidates:
            return spec
    for spec in sorted(
        MODEL_SPECS,
        key=lambda item: len(item.name),
        reverse=True,
    ):
        if value.startswith(spec.name.lower().replace("_", "-")):
            return spec
    supported = ", ".join(spec.name for spec in MODEL_SPECS)
    raise ValueError(
        f"Unsupported WeatherNext model {model_name!r}; choose {supported}"
    )


def _rename_known_aliases(dataset: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    for source, target in {**COORDINATE_ALIASES, **VARIABLE_ALIASES}.items():
        if (
            source in dataset.variables or source in dataset.dims
        ) and target not in dataset.variables and target not in dataset.dims:
            rename[source] = target
    return dataset.rename(rename)


def _absolute_times(dataset: xr.Dataset) -> np.ndarray:
    time = dataset.coords["time"]
    if np.issubdtype(time.dtype, np.datetime64):
        values = time.values.astype("datetime64[ns]")
    elif "datetime" in dataset.coords:
        datetimes = dataset.coords["datetime"]
        if datetimes.dims == ("time",):
            values = datetimes.values
        elif "batch" in datetimes.dims and "time" in datetimes.dims:
            values = datetimes.isel(batch=0).values
        else:
            raise ValueError(
                "datetime coordinate must have time or batch,time dimensions"
            )
        values = np.asarray(values).astype("datetime64[ns]")
    else:
        raise ValueError(
            "Initial state time must be datetime64 or have an absolute datetime coordinate"
        )
    if values.ndim != 1:
        raise ValueError("Initial state must use one shared time axis")
    return values


def _validate_grid(dataset: xr.Dataset, spec: WeatherNextModelSpec) -> None:
    for coordinate in ("lat", "lon", "level", "time"):
        if coordinate not in dataset.coords:
            raise ValueError(
                f"WeatherNext initial state requires {coordinate!r} coordinate"
            )
    for coordinate in ("lat", "lon"):
        values = np.asarray(dataset.coords[coordinate].values, dtype=float)
        if values.ndim != 1 or values.size < 2:
            raise ValueError(f"{coordinate} must be a one-dimensional global grid")
        step = float(np.median(np.abs(np.diff(values))))
        if not np.isclose(step, spec.resolution_degrees, atol=1e-5):
            raise ValueError(
                f"{spec.name} requires {spec.resolution_degrees} degree data; "
                f"received {coordinate} spacing {step}"
            )
    lat = np.asarray(dataset.lat.values, dtype=float)
    lon = np.mod(np.asarray(dataset.lon.values, dtype=float), 360.0)
    if lat.min() > -89.999 or lat.max() < 89.999:
        raise ValueError(
            "Official WeatherNext inference requires the full global latitude grid"
        )
    if np.unique(np.round(lon, 6)).size != dataset.sizes["lon"]:
        raise ValueError(
            "Longitude coordinates contain duplicates after wrapping to [0, 360)"
        )
    expected_lon = round(360.0 / spec.resolution_degrees)
    if dataset.sizes["lon"] != expected_lon:
        raise ValueError(
            f"{spec.name} requires {expected_lon} global longitude points; "
            f"received {dataset.sizes['lon']}"
        )


def _normalise_initial_state(
    initial_state: xr.Dataset,
    spec: WeatherNextModelSpec,
    task: Any,
) -> tuple[xr.Dataset, np.datetime64]:
    state = _rename_known_aliases(initial_state).copy()
    _validate_grid(state, spec)

    absolute_times = _absolute_times(state)
    order = np.argsort(absolute_times)
    state = state.isel(time=order)
    absolute_times = absolute_times[order]
    if absolute_times.size < 2:
        raise ValueError(
            "WeatherNext 2 requires two initial fields at -6h and 0h; "
            "build the condition with InitialConditionBuilder(history_steps=2)"
        )
    state = state.isel(time=slice(-2, None))
    absolute_times = absolute_times[-2:]
    spacing = pd.Timedelta(absolute_times[1] - absolute_times[0])
    if spacing != pd.Timedelta("6h"):
        raise ValueError(
            f"WeatherNext 2 initial fields must be 6 hours apart; received {spacing}"
        )

    levels = np.asarray(state.level.values, dtype=int)
    expected_levels = np.asarray(task.pressure_levels, dtype=int)
    if set(levels.tolist()) != set(expected_levels.tolist()):
        raise ValueError(
            "Pressure levels do not match the selected WeatherNext configuration: "
            f"expected={expected_levels.tolist()}, received={levels.tolist()}"
        )
    state = state.sel(level=list(task.pressure_levels))

    generated = set(task.forcing_variables)
    required = set(task.input_variables).difference(generated)
    missing = sorted(required.difference(state.data_vars))
    if missing:
        raise ValueError(
            "Initial state is missing WeatherNext input variables: "
            + ", ".join(missing)
        )

    state = state.assign_coords(time=("time", absolute_times))
    if "batch" not in state.dims:
        state = state.expand_dims(batch=[0])
    if state.sizes["batch"] != 1:
        raise ValueError("OfficialWeatherNextRunner currently supports batch size 1")
    return state, absolute_times[-1]


def _surface_template(dataset: xr.Dataset, task: Any) -> xr.DataArray:
    for name in task.input_variables:
        if name not in dataset:
            continue
        value = dataset[name]
        if (
            "time" in value.dims
            and "lat" in value.dims
            and "lon" in value.dims
            and "level" not in value.dims
        ):
            return value
    raise ValueError("Cannot infer a surface-grid template for WeatherNext targets")


def _prepare_rollout_data(
    initial_state: xr.Dataset,
    spec: WeatherNextModelSpec,
    task: Any,
    horizon_hours: int,
    data_utils: Any,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, np.datetime64]:
    state, init_time = _normalise_initial_state(initial_state, spec, task)
    input_times = _absolute_times(state)
    future_times = np.arange(
        init_time + np.timedelta64(6, "h"),
        init_time + np.timedelta64(horizon_hours + 6, "h"),
        np.timedelta64(6, "h"),
    ).astype("datetime64[ns]")
    all_times = np.concatenate((input_times, future_times))
    extended = state.assign_coords(time=("time", input_times)).reindex(time=all_times)

    surface = _surface_template(extended, task)
    for name in task.target_variables:
        if name not in extended:
            extended[name] = xr.zeros_like(surface, dtype=np.float32)

    relative_times = all_times - init_time
    datetime_values = np.broadcast_to(all_times, (1, all_times.size))
    extended = extended.assign_coords(
        time=("time", relative_times),
        datetime=(("batch", "time"), datetime_values),
    )
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        extended,
        target_lead_times=slice("6h", f"{horizon_hours}h"),
        input_variables=task.input_variables,
        target_variables=task.target_variables,
        forcing_variables=task.forcing_variables,
        pressure_levels=task.pressure_levels,
        input_duration=task.input_duration,
    )
    return inputs, targets * np.nan, forcings, init_time


def _metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".metadata.json")


def _validate_checkpoint_metadata(
    checkpoint_path: Path,
    spec: WeatherNextModelSpec,
    release: str,
) -> dict[str, Any]:
    path = _metadata_path(checkpoint_path)
    if not path.exists():
        return {}
    metadata = json.loads(path.read_text(encoding="utf-8"))
    recorded_model = metadata.get("model_name")
    if recorded_model and resolve_model_spec(recorded_model).name != spec.name:
        raise ValueError(
            f"Checkpoint was fine-tuned for {recorded_model}, not {spec.name}"
        )
    recorded_release = metadata.get("weathernext_release")
    if recorded_release and recorded_release != release:
        raise ValueError(
            f"Checkpoint release {recorded_release} does not match runner release {release}"
        )
    return metadata


def _checkpoint_kind(metadata: dict[str, Any]) -> str:
    recorded = metadata.get("checkpoint_kind")
    if recorded:
        return str(recorded)
    if metadata.get("fine_tune_steps") is not None:
        return "fine_tuned"
    if metadata.get("official_pretrained") is True:
        return "official_pretrained"
    return "pretrained_unknown"


def _import_official_modules() -> dict[str, Any]:
    try:
        import haiku as hk
        import jax
        from weathernext.utils import checkpoint, data_utils, fiddle_config_io, rollout
        from weathernext.weathernext2 import fgn
    except ImportError as exc:
        raise ImportError(
            "Official WeatherNext inference dependencies are missing. "
            "Install this project with: pip install -e '.[weathernext]'"
        ) from exc
    return {
        "hk": hk,
        "jax": jax,
        "checkpoint": checkpoint,
        "data_utils": data_utils,
        "fiddle_config_io": fiddle_config_io,
        "rollout": rollout,
        "fgn": fgn,
    }


class OfficialWeatherNextRunner:
    """Load official or fine-tuned WN2 parameters once and run inference only."""

    inference_only = True

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | Path,
        *,
        release: str = SUPPORTED_RELEASE,
        seed: int = 0,
    ):
        if release != SUPPORTED_RELEASE:
            raise ValueError(
                f"This adapter is pinned to {SUPPORTED_RELEASE}; received {release}"
            )
        self.spec = resolve_model_spec(model_name)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.release = release
        self.seed = int(seed)
        self.checkpoint_metadata = _validate_checkpoint_metadata(
            self.checkpoint_path, self.spec, release
        )
        self.checkpoint_kind = _checkpoint_kind(self.checkpoint_metadata)
        modules = _import_official_modules()
        self._jax = modules["jax"]
        self._rollout_module = modules["rollout"]
        self._data_utils = modules["data_utils"]
        fgn = modules["fgn"]

        config_path = f"weathernext2/configs/{self.spec.config_name}"
        self._config = modules["fiddle_config_io"].get_fiddle_config_by_name(
            config_path
        )
        self._configure_accelerator()
        with self.checkpoint_path.open("rb") as source:
            loaded = modules["checkpoint"].load(source, fgn.CheckPoint)
        self._params = self._jax.tree_util.tree_map(
            self._jax.lax.stop_gradient, loaded.params
        )

        inference_config = fgn.PredictorConfig(
            task=self._config.task,
            predictor_constructor=self._config.predictor_constructor,
            predictor_kwargs=self._config.predictor_kwargs,
            predictor_wrappers=self._config.predictor_wrappers[:-1],
        )
        hk = modules["hk"]

        @hk.transform
        def forward(inputs, targets_template, forcings):
            predictor = fgn.construct_predictor(inference_config)
            return predictor(
                inputs,
                targets_template=targets_template,
                forcings=forcings,
            )

        self._predict = self._jax.jit(
            lambda rng, inputs, targets, forcings: forward.apply(
                self._params, rng, inputs, targets, forcings
            )
        )

    def _configure_accelerator(self) -> None:
        backend = self._jax.default_backend()
        transformer_kwargs = self._config.predictor_kwargs["noisy_function_kwargs"][
            "mesh_model_ctor"
        ].keywords["transformer_kwargs"]
        if backend == "gpu":
            transformer_kwargs["attention_type"] = "triblockdiag_mha"
        elif backend == "tpu":
            transformer_kwargs.update(
                {
                    "block_q": 128,
                    "block_kv": 128,
                    "block_kv_compute": 128,
                    "block_q_dkv": 128,
                    "block_kv_dkv": 128,
                    "block_kv_dkv_compute": 128,
                }
            )

    def rollout(self, initial_state: xr.Dataset, horizon_hours: int) -> xr.Dataset:
        if horizon_hours <= 0 or horizon_hours > 360 or horizon_hours % 6:
            raise ValueError(
                "WeatherNext horizon must be a positive 6-hour multiple up to 360"
            )
        inputs, targets, forcings, init_time = _prepare_rollout_data(
            initial_state,
            self.spec,
            self._config.task,
            horizon_hours,
            self._data_utils,
        )
        prediction = self._rollout_module.chunked_prediction(
            predictor_fn=self._predict,
            rng=self._jax.random.PRNGKey(self.seed),
            inputs=inputs,
            targets_template=targets,
            forcings=forcings,
            num_steps_per_chunk=1,
        )
        prediction = self._jax.device_get(prediction)
        if "batch" in prediction.dims:
            if prediction.sizes["batch"] != 1:
                raise ValueError("Unexpected WeatherNext prediction batch size")
            prediction = prediction.isel(batch=0, drop=True)
        lead_times = prediction.coords["time"].values
        prediction = prediction.assign_coords(time=init_time + lead_times)
        return prediction.assign_attrs(
            {
                **prediction.attrs,
                "weathernext_inference_only": True,
                "weathernext_model_variant": self.spec.name,
                "weathernext_checkpoint_path": str(self.checkpoint_path),
                "weathernext_checkpoint_kind": self.checkpoint_kind,
            }
        )
