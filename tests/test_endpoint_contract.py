import numpy as np
import pandas as pd
import pytest
import torch

from typhoon_pressure.small_version import SmallModelConfig, WeatherNextDualTargetDataset
from typhoon_pressure.small_version.weathernext_bridge import ForecastTokens


class _Base:
    def __init__(self):
        self.time=pd.Timestamp("2025-08-01T00:00:00"); self.groups={"TEST":pd.DataFrame({"time":pd.date_range(self.time,periods=130,freq="6h"),"typhoon_lat":np.linspace(20.,30.,130),"typhoon_lon":np.linspace(130.,145.,130)})}
    def __len__(self): return 1
    def storm_id_at(self,index): return "TEST"
    def __getitem__(self,index): return {"history":torch.zeros(2,4),"history_mask":torch.ones(2,4),"target":torch.tensor([[21.,131.],[22.,132.]]),"target_mask":torch.ones(2,2),"storm_id":"TEST","init_time_ns":int(self.time.value)}


class _Store:
    def __init__(self,lead_hours,endpoint_mask=True):
        self.tokens=ForecastTokens(values=np.zeros((1,4),dtype=np.float32),feature_mask=np.ones((1,4),dtype=np.float32),token_mask=np.ones(1,dtype=np.float32),positions=np.zeros((1,6),dtype=np.float32),feature_names=("mean_sea_level_pressure","10m_u_component_of_wind","10m_v_component_of_wind","2m_temperature"),endpoint_latlon=np.asarray([25.,140.],dtype=np.float32),endpoint_mask=endpoint_mask,endpoint_lead_hours=float(lead_hours))
    def contains(self,storm_id,init_time_ns): return True
    def load(self,storm_id,init_time_ns): return self.tokens


def _dataset(endpoint_lead_hours,policy="disable",endpoint_mask=True):
    config=SmallModelConfig(input_dim=4,history_steps=2,local_track_steps=2,distribution_start_day=15,distribution_end_day=16)
    return WeatherNextDualTargetDataset(_Base(),None,config,_Store(endpoint_lead_hours,endpoint_mask),max_forecast_tokens=2,forecast_input_dim=4,require_endpoint_lead_hours=360.,endpoint_mismatch_policy=policy)


def test_day15_endpoint_accepts_exact_360h():
    dataset=_dataset(360.); assert len(dataset)==1; sample=dataset[0]
    assert sample["weathernext_endpoint_mask"].item()==1.; assert sample["weathernext_endpoint_lead_hours"].item()==360.; assert dataset.endpoint_contract_stats["exact"]==1


def test_monthly_flow_720h_tokens_are_kept_but_endpoint_is_disabled():
    dataset=_dataset(720.); assert len(dataset)==1; sample=dataset[0]
    assert sample["forecast_token_mask"].sum().item()==1.; assert sample["weathernext_endpoint_mask"].item()==0.; assert sample["weathernext_endpoint_lead_hours"].item()==720.; assert torch.equal(sample["weathernext_endpoint_latlon"],torch.zeros(2)); assert dataset.endpoint_contract_stats["mismatched"]==1


def test_nearby_but_wrong_endpoint_is_disabled():
    dataset=_dataset(366.); assert len(dataset)==1; assert dataset[0]["weathernext_endpoint_mask"].item()==0.


def test_strict_policy_fails_immediately_on_720h_endpoint():
    with pytest.raises(ValueError,match="P15 requires exact 360h endpoint"):
        _dataset(720.,policy="error")


def test_missing_endpoint_uses_fallback_without_dropping_tokens():
    dataset=_dataset(0.,endpoint_mask=False); assert len(dataset)==1; sample=dataset[0]
    assert sample["forecast_token_mask"].sum().item()==1.; assert sample["weathernext_endpoint_mask"].item()==0.; assert dataset.endpoint_contract_stats["missing"]==1
