import inspect

import pytest

from typhoon_pressure.weathernext_official import (
    OfficialWeatherNextRunner,
    _validate_checkpoint_metadata,
    resolve_model_spec,
)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("weather-next", "WeatherNext2"),
        ("WeatherNext2_<2025", "WeatherNext2"),
        ("cyclone", "WeatherNextCyclones"),
        ("WeatherNextCyclones_<2025", "WeatherNextCyclones"),
        ("mini", "WeatherNextCyclones_Mini"),
        ("WeatherNextCyclones_Mini_<2024", "WeatherNextCyclones_Mini"),
    ],
)
def test_model_aliases_resolve_to_matching_architecture(alias, expected):
    assert resolve_model_spec(alias).name == expected


def test_checkpoint_metadata_rejects_wrong_architecture(tmp_path):
    checkpoint = tmp_path / "weather-me-fine_tune_weight.npz"
    checkpoint.touch()
    checkpoint.with_suffix(".metadata.json").write_text(
        '{"model_name": "WeatherNextCyclones_Mini", "weathernext_release": "v0.3.0"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not WeatherNext2"):
        _validate_checkpoint_metadata(
            checkpoint,
            resolve_model_spec("WeatherNext2"),
            "v0.3.0",
        )


def test_official_runner_has_no_training_entry_point():
    assert OfficialWeatherNextRunner.inference_only is True
    assert not hasattr(OfficialWeatherNextRunner, "fit")
    source = inspect.getsource(OfficialWeatherNextRunner)
    assert "value_and_grad" not in source
    assert "optimizer" not in source
