from types import SimpleNamespace

import numpy as np

from typhoon_pressure.small_version.config import GPTStateConfig
from typhoon_pressure.small_version.gpt_state import (
    DirectoryGPTStateStore,
    GPTStateRecord,
    GPTSynopticState,
    OpenAIStateExtractor,
    save_gpt_state,
)


class FakeResponses:
    def __init__(self, state):
        self.state = state
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_parsed=self.state)


def test_structured_gpt_state_is_cached_without_live_api(tmp_path):
    state = GPTSynopticState(
        steering_eastward_score=0.2,
        steering_northward_score=0.5,
        recurvature_score=0.7,
        intensification_score=-0.1,
        subtropical_high_influence=0.8,
        monsoon_influence=0.3,
        east_asia_approach_risk=0.6,
        track_uncertainty=0.4,
        intensity_uncertainty=0.5,
        confidence=0.75,
    )
    fake_responses = FakeResponses(state)
    client = SimpleNamespace(responses=fake_responses)
    record = OpenAIStateExtractor(GPTStateConfig(model="gpt-test"), client=client).extract({
        "history": {"typhoon_lat": {"latest": 20.0}},
        "weathernext_valid_token_fraction": 0.9,
    })
    assert fake_responses.request["text_format"] is GPTSynopticState
    assert fake_responses.request["model"] == "gpt-test"
    assert record.values.shape == (10,)
    assert np.all(record.mask == 1)
    save_gpt_state(record, tmp_path, storm_id="TEST", init_time_ns=123)
    loaded = DirectoryGPTStateStore(tmp_path).load("TEST", 123)
    np.testing.assert_allclose(loaded.values, record.values)


def test_missing_gpt_state_is_fully_masked():
    record = GPTStateRecord.missing()
    assert np.all(record.values == 0)
    assert np.all(record.mask == 0)
