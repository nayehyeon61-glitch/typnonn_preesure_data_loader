from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import EastAsiaBounds, SmallModelConfig
from .weathernext_bridge import DirectoryForecastTokenStore


@dataclass
class SpatialDistributionLookup:
    probabilities: np.ndarray
    available_months: np.ndarray
    n_lat: int
    n_lon: int

    @classmethod
    def from_frame(cls, frame, *, lat_bin_deg=5.0, lon_bin_deg=5.0):
        required={"calendar_month","lat_bin","lon_bin","probability"}; missing=required.difference(frame.columns)
        if missing: raise ValueError(f"Distribution table is missing columns: {sorted(missing)}")
        n_lat=round(180.0/lat_bin_deg); n_lon=round(360.0/lon_bin_deg); dense=np.zeros((12,n_lat*n_lon),dtype=np.float32); available=np.zeros(12,dtype=bool)
        for row in frame.itertuples(index=False):
            month=int(row.calendar_month)-1; lat_bin=int(row.lat_bin); lon_bin=int(row.lon_bin)
            if 0<=month<12 and 0<=lat_bin<n_lat and 0<=lon_bin<n_lon: dense[month,lat_bin*n_lon+lon_bin]+=float(row.probability); available[month]=True
        totals=dense.sum(axis=1,keepdims=True); dense=np.divide(dense,totals,out=np.zeros_like(dense),where=totals>0)
        return cls(dense,available,n_lat,n_lon)

    @classmethod
    def from_csv(cls,path,**kwargs): return cls.from_frame(pd.read_csv(path),**kwargs)
    def targets(self,init_time_ns,lead_days):
        init_time=pd.Timestamp(init_time_ns,unit="ns"); months=np.asarray([(init_time+pd.Timedelta(days=day)).month-1 for day in lead_days]); return self.probabilities[months].copy(),self.available_months[months].astype(np.float32)


class DualTargetDataset(Dataset):
    def __init__(self,base_dataset,distribution,model_config,bounds=EastAsiaBounds()):
        if distribution is not None and (distribution.n_lat!=model_config.n_lat or distribution.n_lon!=model_config.n_lon): raise ValueError("Distribution grid and model grid do not match")
        self.base=base_dataset; self.distribution=distribution; self.config=model_config; self.bounds=bounds
    def __len__(self): return len(self.base)
    def storm_id_at(self,index):
        if hasattr(self.base,"storm_id_at"): return str(self.base.storm_id_at(index))
        if hasattr(self.base,"windows"): return str(self.base.windows[index][0])
        return str(self.base[index]["storm_id"])
    def _storm_future_targets(self,storm_id,init_time_ns):
        if not hasattr(self.base,"groups") or str(storm_id) not in self.base.groups: raise ValueError("Base dataset must expose complete per-storm groups")
        group=self.base.groups[str(storm_id)]; times=pd.DatetimeIndex(group["time"]); lat=pd.to_numeric(group["typhoon_lat"],errors="coerce").to_numpy(float); lon=pd.to_numeric(group["typhoon_lon"],errors="coerce").to_numpy(float); valid=np.isfinite(lat)&np.isfinite(lon)
        if not valid.any(): raise ValueError(f"Storm {storm_id!r} has no valid track positions")
        fixes={int(time.value):(float(lat[i]),float(lon[i])) for i,time in enumerate(times) if valid[i]}; final=max(fixes); init=pd.Timestamp(init_time_ns,unit="ns"); leads=len(self.config.lead_days)
        positions=np.zeros((leads,2),dtype=np.float32); position_mask=np.zeros(leads,dtype=np.float32); alive=np.zeros(leads,dtype=np.float32); alive_mask=np.zeros(leads,dtype=np.float32); grid=np.zeros((leads,self.config.n_cells),dtype=np.float32)
        for index,day in enumerate(self.config.lead_days):
            target=int((init+pd.Timedelta(days=day)).value)
            if target in fixes:
                la,lo=fixes[target]; positions[index]=(la,lo); position_mask[index]=alive[index]=alive_mask[index]=1.; lb=int(np.clip(np.floor((la+90)/self.config.lat_bin_deg),0,self.config.n_lat-1)); lob=int(np.clip(np.floor((lo%360)/self.config.lon_bin_deg),0,self.config.n_lon-1)); grid[index,lb*self.config.n_lon+lob]=1.
            elif target>final: alive_mask[index]=1.
        return positions,position_mask,alive,alive_mask,grid
    def __getitem__(self,index):
        sample=self.base[index]
        if sample["target"].shape[0]!=self.config.local_track_steps: raise ValueError("Base dataset horizon must equal local_track_steps")
        ft,fm,fa,fam,dt=self._storm_future_targets(sample["storm_id"],sample["init_time_ns"]); track=sample["target"][:,:2]; valid=sample["target_mask"][:,:2].all(dim=-1); lon360=torch.remainder(track[:,1],360.0); domain=(track[:,0]>=self.bounds.lat_min)&(track[:,0]<=self.bounds.lat_max)&(lon360>=self.bounds.lon_min)&(lon360<=self.bounds.lon_max)
        return {"history":sample["history"],"history_mask":sample["history_mask"],"distribution_target":torch.from_numpy(dt),"distribution_mask":torch.from_numpy(fm.copy()),"future_track_target":torch.from_numpy(ft),"future_track_mask":torch.from_numpy(fm),"future_alive_target":torch.from_numpy(fa),"future_alive_mask":torch.from_numpy(fam),"track_target":track,"track_mask":(valid&domain).to(torch.float32),"storm_id":sample["storm_id"],"init_time_ns":sample["init_time_ns"]}


class WeatherNextDualTargetDataset(DualTargetDataset):
    """Dual targets plus frozen forecast tokens with an exact P15 endpoint contract.

    A 720 h Flow endpoint does not remove the sample: forecast tokens remain
    available to the Transformer/GPT Router, while only the invalid Day-15
    endpoint anchor is disabled and the adaptive sampler falls back to its
    learned initial mean.
    """
    ENDPOINT_LEAD_TOLERANCE_HOURS=1e-3
    def __init__(self,base_dataset,distribution,model_config,forecast_store,*,max_forecast_tokens,forecast_input_dim,bounds=EastAsiaBounds(),require_endpoint_lead_hours=None,endpoint_mismatch_policy="disable"):
        super().__init__(base_dataset,distribution,model_config,bounds); self.forecast_store=forecast_store; self.max_forecast_tokens=max_forecast_tokens; self.forecast_input_dim=forecast_input_dim; self.require_endpoint_lead_hours=require_endpoint_lead_hours
        if endpoint_mismatch_policy not in {"disable","error"}: raise ValueError("endpoint_mismatch_policy must be disable or error")
        self.endpoint_mismatch_policy=endpoint_mismatch_policy; self.indices=[]; self.endpoint_contract_stats={"exact":0,"mismatched":0,"missing":0}
        for index in range(len(self.base)):
            sample=self.base[index]
            if not self.forecast_store.contains(sample["storm_id"],sample["init_time_ns"]): continue
            tokens=self.forecast_store.load(sample["storm_id"],sample["init_time_ns"])
            if require_endpoint_lead_hours is None: self.endpoint_contract_stats["exact"]+=int(tokens.endpoint_mask)
            else:
                matches=tokens.endpoint_mask and abs(float(tokens.endpoint_lead_hours)-float(require_endpoint_lead_hours))<=self.ENDPOINT_LEAD_TOLERANCE_HOURS
                if matches: self.endpoint_contract_stats["exact"]+=1
                elif tokens.endpoint_mask:
                    self.endpoint_contract_stats["mismatched"]+=1
                    if endpoint_mismatch_policy=="error": raise ValueError(f"P15 requires exact {require_endpoint_lead_hours:g}h endpoint; got {tokens.endpoint_lead_hours:g}h for {sample['storm_id']}@{sample['init_time_ns']}")
                else: self.endpoint_contract_stats["missing"]+=1
            self.indices.append(index)
    def __len__(self): return len(self.indices)
    def storm_id_at(self,index): return super().storm_id_at(self.indices[index])
    def __getitem__(self,index):
        sample=super().__getitem__(self.indices[index]); tokens=self.forecast_store.load(sample["storm_id"],sample["init_time_ns"])
        if tokens.values.shape[1]!=self.forecast_input_dim: raise ValueError(f"Forecast feature count {tokens.values.shape[1]} does not match forecast_input_dim={self.forecast_input_dim}")
        count=min(len(tokens.values),self.max_forecast_tokens); values=np.zeros((self.max_forecast_tokens,self.forecast_input_dim),dtype=np.float32); feature_mask=np.zeros_like(values); token_mask=np.zeros(self.max_forecast_tokens,dtype=np.float32); positions=np.zeros((self.max_forecast_tokens,6),dtype=np.float32)
        values[:count]=tokens.values[:count]; feature_mask[:count]=tokens.feature_mask[:count]; token_mask[:count]=tokens.token_mask[:count]; positions[:count]=tokens.positions[:count]
        endpoint_valid=bool(tokens.endpoint_mask)
        if self.require_endpoint_lead_hours is not None: endpoint_valid=endpoint_valid and abs(float(tokens.endpoint_lead_hours)-float(self.require_endpoint_lead_hours))<=self.ENDPOINT_LEAD_TOLERANCE_HOURS
        endpoint_latlon=tokens.endpoint_latlon if endpoint_valid else np.zeros(2,dtype=np.float32)
        sample.update({"forecast_values":torch.from_numpy(values),"forecast_feature_mask":torch.from_numpy(feature_mask),"forecast_token_mask":torch.from_numpy(token_mask),"forecast_positions":torch.from_numpy(positions),"weathernext_endpoint_latlon":torch.from_numpy(endpoint_latlon),"weathernext_endpoint_mask":torch.tensor(endpoint_valid,dtype=torch.float32),"weathernext_endpoint_lead_hours":torch.tensor(tokens.endpoint_lead_hours,dtype=torch.float32),"endpoint_contract_match":torch.tensor(endpoint_valid,dtype=torch.float32)})
        return sample