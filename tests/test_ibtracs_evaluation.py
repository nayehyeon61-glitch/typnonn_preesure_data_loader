import json

import numpy as np
import pandas as pd

from typhoon_pressure.evaluation import (
    IBTrACSEvaluationConfig,
    evaluate_ibtracs_predictions,
    write_ibtracs_evaluation,
)
from typhoon_pressure.utility.metrics import great_circle_distance_km


def _observations():
    return pd.DataFrame({
        "storm_id": ["A", "A", "B"],
        "time": pd.to_datetime([
            "2025-01-01 00:00", "2025-01-01 06:00", "2025-01-01 00:00"
        ]),
        "typhoon_lat": [20.0, 21.0, -10.0],
        "typhoon_lon": [130.0, 131.0, 140.0],
        "typhoon_pressure_hpa": [990.0, 985.0, 1000.0],
        "typhoon_wind_kt": [50.0, 60.0, 35.0],
    })


def test_dateline_track_distance_uses_wrapped_longitude():
    distance = great_circle_distance_km([0.0], [179.0], [0.0], [-179.0])
    assert 220.0 < distance[0] < 225.0


def test_ibtracs_evaluation_matches_absolute_time_and_optional_intensity():
    predictions = pd.DataFrame({
        "storm_id": ["A", "A"],
        "init_time": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "lead_hours": [0, 6],
        "pred_lat": [20.0, 22.0],
        "pred_lon": [130.0, 131.0],
        "pred_pressure_hpa": [992.0, 980.0],
        "pred_wind_kt": [48.0, 65.0],
        "weathernext_backend": ["pretrained", "pretrained"],
    })
    result = evaluate_ibtracs_predictions(predictions, _observations())
    assert result.overall["matched_count"] == 2
    assert result.overall["track_count"] == 2
    assert result.overall["pressure_hpa_mae"] == 3.5
    assert result.overall["wind_kt_bias"] == 1.5
    assert list(result.by_lead["lead_hours"]) == [0, 6]
    assert result.by_backend.iloc[0]["weathernext_backend"] == "pretrained"


def test_east_asia_filter_uses_ibtracs_target_position():
    predictions = pd.DataFrame({
        "storm_id": ["A", "B"],
        "time": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "pred_lat": [20.0, -10.0],
        "pred_lon": [130.0, 140.0],
    })
    result = evaluate_ibtracs_predictions(
        predictions, _observations(), IBTrACSEvaluationConfig(east_asia_only=True)
    )
    assert result.overall["matched_count"] == 1
    assert result.matched.iloc[0]["storm_id"] == "A"


def test_evaluation_writes_machine_readable_outputs(tmp_path):
    predictions = pd.DataFrame({
        "storm_id": ["A"], "time": ["2025-01-01"],
        "pred_lat": [20.5], "pred_lon": [130.0],
    })
    result = evaluate_ibtracs_predictions(predictions, _observations())
    write_ibtracs_evaluation(result, tmp_path)
    assert (tmp_path / "matched_predictions.csv").exists()
    assert (tmp_path / "metrics_by_storm.csv").exists()
    with (tmp_path / "metrics_overall.json").open(encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["matched_count"] == 1
    assert np.isfinite(saved["mean_track_error_km"])
