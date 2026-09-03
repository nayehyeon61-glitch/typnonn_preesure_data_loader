from __future__ import annotations

import torch
from torch import nn

from .config import (
    DistributionSamplingConfig,
    EastAsiaBounds,
    SmallModelConfig,
    TransformerConfig,
)
from .distribution_sampling import AdaptiveDistributionSampler


class SmallDualScaleModel(nn.Module):
    """One history encoder with global-distribution and local-track heads."""

    def __init__(self, config: SmallModelConfig, bounds: EastAsiaBounds = EastAsiaBounds()):
        super().__init__()
        self.config = config
        self.bounds = bounds
        self.input_projection = nn.Sequential(nn.Linear(config.input_dim * 2, config.hidden_dim), nn.LayerNorm(config.hidden_dim), nn.GELU())
        self.encoder = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.distribution_head = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, len(config.lead_days) * config.n_cells))
        self.track_head = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, config.local_track_steps * 2))

    def forward(self, history, history_mask):
        encoded_input = self.input_projection(torch.cat((history, history_mask), dim=-1))
        _, hidden = self.encoder(encoded_input)
        state = hidden[-1]
        distribution_logits = self.distribution_head(state).view(-1, len(self.config.lead_days), self.config.n_cells)
        raw_track = self.track_head(state).view(-1, self.config.local_track_steps, 2)
        unit_track = torch.sigmoid(raw_track)
        lat = self.bounds.lat_min + (self.bounds.lat_max - self.bounds.lat_min) * unit_track[..., 0]
        lon = self.bounds.lon_min + (self.bounds.lon_max - self.bounds.lon_min) * unit_track[..., 1]
        return {"distribution_logits": distribution_logits, "track_latlon": torch.stack((lat, lon), dim=-1)}


class GPTForecastRouter(nn.Module):
    """Use a structured GPT state to actively route frozen forecast tokens."""
    def __init__(self, gpt_state_dim: int, model_dim: int):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(gpt_state_dim * 2, model_dim), nn.LayerNorm(model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.token_gate = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 1))
        self.channel_gate = nn.Linear(model_dim, model_dim)
        nn.init.zeros_(self.token_gate[-1].weight); nn.init.zeros_(self.token_gate[-1].bias)
        nn.init.zeros_(self.channel_gate.weight); nn.init.zeros_(self.channel_gate.bias)

    def forward(self, forecast_tokens, gpt_state_values, gpt_state_mask):
        masked = torch.where(gpt_state_mask.bool(), gpt_state_values, torch.zeros_like(gpt_state_values))
        available = gpt_state_mask.bool().any(dim=-1, keepdim=True).to(forecast_tokens.dtype)
        context = self.context(torch.cat((masked, gpt_state_mask), dim=-1))
        raw_token_gate = 2.0 * torch.sigmoid(self.token_gate(forecast_tokens + context.unsqueeze(1)))
        raw_channel_gate = 2.0 * torch.sigmoid(self.channel_gate(context).unsqueeze(1))
        available_3d = available.unsqueeze(-1)
        token_gate = 1.0 + available_3d * (raw_token_gate - 1.0)
        channel_gate = 1.0 + available_3d * (raw_channel_gate - 1.0)
        return forecast_tokens * token_gate * channel_gate, token_gate, channel_gate, available


class WeatherNextFusionTransformer(nn.Module):
    """Fuse WeatherNext/Flow + GPT + history and sample day 15--30 trajectories."""
    ENDPOINT_TOLERANCE_HOURS = 1e-3

    def __init__(self, model_config, transformer_config, bounds=EastAsiaBounds(), sampling_config=DistributionSamplingConfig()):
        super().__init__()
        self.model_config=model_config; self.transformer_config=transformer_config; self.sampling_config=sampling_config; self.bounds=bounds
        hidden=model_config.hidden_dim
        self.history_projection=nn.Sequential(nn.Linear(model_config.input_dim*2,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.history_encoder=nn.GRU(hidden,hidden,batch_first=True)
        self.forecast_projection=nn.Sequential(nn.Linear(transformer_config.forecast_input_dim*2+6,transformer_config.model_dim),nn.LayerNorm(transformer_config.model_dim))
        self.cls_token=nn.Parameter(torch.zeros(1,1,transformer_config.model_dim)); nn.init.normal_(self.cls_token,std=.02)
        layer=nn.TransformerEncoderLayer(d_model=transformer_config.model_dim,nhead=transformer_config.num_heads,dim_feedforward=transformer_config.feedforward_dim,dropout=transformer_config.dropout,activation="gelu",batch_first=True,norm_first=True)
        self.forecast_encoder=nn.TransformerEncoder(layer,transformer_config.num_layers)
        self.gpt_history_conditioner=None; self.gpt_forecast_router=None
        if transformer_config.gpt_state_dim>0:
            self.gpt_history_conditioner=nn.Sequential(nn.Linear(transformer_config.gpt_state_dim*2,hidden*2),nn.LayerNorm(hidden*2),nn.GELU(),nn.Linear(hidden*2,hidden*2))
            self.gpt_forecast_router=GPTForecastRouter(transformer_config.gpt_state_dim,transformer_config.model_dim)
        self.fusion=nn.Sequential(nn.Linear(hidden+transformer_config.model_dim,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.history_to_forecast_dim=nn.Linear(hidden,transformer_config.model_dim)
        self.future_queries=nn.Parameter(torch.zeros(1,len(model_config.lead_days),transformer_config.model_dim)); nn.init.normal_(self.future_queries,std=.02)
        decoder_layer=nn.TransformerDecoderLayer(d_model=transformer_config.model_dim,nhead=transformer_config.num_heads,dim_feedforward=transformer_config.feedforward_dim,dropout=transformer_config.dropout,activation="gelu",batch_first=True,norm_first=True)
        self.future_decoder=nn.TransformerDecoder(decoder_layer,transformer_config.decoder_layers)
        self.distribution_projection=nn.Linear(transformer_config.model_dim,model_config.n_cells)
        self.distribution_sampler=AdaptiveDistributionSampler(model_config,transformer_config.model_dim,gpt_state_dim=transformer_config.gpt_state_dim,sampling_config=sampling_config)
        self.survival_hazard_head=nn.Linear(transformer_config.model_dim,1)
        self.track_head=nn.Linear(hidden,model_config.local_track_steps*2)

    def _apply_input_mask(self,values,feature_mask,token_mask):
        effective=feature_mask.bool(); p=self.transformer_config.input_mask_probability
        if self.training and p>0: effective=effective & (torch.rand_like(values)>=p)
        token=token_mask.bool() & effective.any(dim=-1)
        return torch.where(effective,values,torch.zeros_like(values)),effective.to(values.dtype),token

    def forward(self,history,history_mask,forecast_values,forecast_feature_mask,forecast_token_mask,forecast_positions,gpt_state_values=None,gpt_state_mask=None,weathernext_endpoint_latlon=None,weathernext_endpoint_mask=None,weathernext_endpoint_lead_hours=None):
        masked_history=torch.where(history_mask.bool(),history,torch.zeros_like(history))
        history_input=self.history_projection(torch.cat((masked_history,history_mask),dim=-1)); gpt_conditioning_fraction=None
        if self.gpt_history_conditioner is not None:
            if gpt_state_values is None or gpt_state_mask is None: raise ValueError("GPT state tensors are required when gpt_state_dim > 0")
            masked=torch.where(gpt_state_mask.bool(),gpt_state_values,torch.zeros_like(gpt_state_values)); gamma,beta=self.gpt_history_conditioner(torch.cat((masked,gpt_state_mask),dim=-1)).chunk(2,dim=-1)
            available=gpt_state_mask.bool().any(dim=-1,keepdim=True).to(history.dtype); gamma=.5*torch.tanh(gamma)*available; beta=torch.tanh(beta)*available
            history_input=history_input*(1+gamma.unsqueeze(1))+beta.unsqueeze(1); gpt_conditioning_fraction=available.mean().detach()
        _,history_hidden=self.history_encoder(history_input)
        masked_values,effective_feature_mask,effective_token_mask=self._apply_input_mask(forecast_values,forecast_feature_mask,forecast_token_mask)
        forecast_input=self.forecast_projection(torch.cat((masked_values,effective_feature_mask,forecast_positions),dim=-1))
        router_token_gate=router_channel_gate=router_active_fraction=None
        if self.gpt_forecast_router is not None:
            if gpt_state_values is None or gpt_state_mask is None: raise ValueError("GPT state tensors are required when gpt_state_dim > 0")
            forecast_input,router_token_gate,router_channel_gate,router_available=self.gpt_forecast_router(forecast_input,gpt_state_values,gpt_state_mask); router_active_fraction=router_available.mean().detach()
        cls=self.cls_token.expand(forecast_input.shape[0],-1,-1); forecast_input=torch.cat((cls,forecast_input),dim=1)
        padding_mask=torch.cat((torch.zeros((effective_token_mask.shape[0],1),dtype=torch.bool,device=effective_token_mask.device),~effective_token_mask),dim=1)
        forecast_memory=self.forecast_encoder(forecast_input,src_key_padding_mask=padding_mask); forecast_state=forecast_memory[:,0]
        state=self.fusion(torch.cat((history_hidden[-1],forecast_state),dim=-1))
        queries=self.future_queries.expand(history.shape[0],-1,-1)+self.history_to_forecast_dim(state).unsqueeze(1)
        future_states=self.future_decoder(tgt=queries,memory=forecast_memory,memory_key_padding_mask=padding_mask)
        distribution_logits=self.distribution_projection(future_states)

        # Defense in depth: even a custom dataset cannot feed a 720 h endpoint
        # into the Day-15 anchor. Wrong/missing lead metadata disables only the
        # anchor; frozen forecast tokens and GPT routing remain active.
        required_hours=float(self.model_config.distribution_start_day*24)
        contract_match=None
        if weathernext_endpoint_mask is not None:
            if weathernext_endpoint_lead_hours is None:
                contract_match=torch.zeros_like(weathernext_endpoint_mask,dtype=torch.bool)
            else:
                contract_match=(weathernext_endpoint_lead_hours.to(future_states.dtype)-required_hours).abs()<=self.ENDPOINT_TOLERANCE_HOURS
            weathernext_endpoint_mask=weathernext_endpoint_mask.to(future_states.dtype)*contract_match.to(future_states.dtype)

        sampling=self.distribution_sampler(future_states,gpt_state_values=gpt_state_values,gpt_state_mask=gpt_state_mask,weathernext_endpoint_latlon=weathernext_endpoint_latlon,weathernext_endpoint_mask=weathernext_endpoint_mask)
        survival_hazard_logits=self.survival_hazard_head(future_states).squeeze(-1)
        survival_probability=torch.cumsum(torch.nn.functional.logsigmoid(-survival_hazard_logits),dim=1).exp()
        raw_track=self.track_head(state).view(-1,self.model_config.local_track_steps,2); unit=torch.sigmoid(raw_track)
        lat=self.bounds.lat_min+(self.bounds.lat_max-self.bounds.lat_min)*unit[...,0]; lon=self.bounds.lon_min+(self.bounds.lon_max-self.bounds.lon_min)*unit[...,1]
        result={"distribution_logits":distribution_logits,"track_latlon":torch.stack((lat,lon),dim=-1),"survival_hazard_logits":survival_hazard_logits,"survival_probability":survival_probability,"no_storm_probability":1-survival_probability,"distribution_unconditional_probabilities":sampling["distribution_probabilities"]*survival_probability.unsqueeze(-1),"effective_forecast_token_fraction":effective_token_mask.float().mean().detach(),"effective_forecast_feature_fraction":effective_feature_mask.mean().detach(),**sampling}
        if contract_match is not None: result["endpoint_contract_match_fraction"]=contract_match.float().mean().detach()
        if gpt_conditioning_fraction is not None: result["gpt_history_conditioning_fraction"]=gpt_conditioning_fraction
        if router_token_gate is not None:
            valid=effective_token_mask.to(router_token_gate.dtype); values=router_token_gate.squeeze(-1); count=valid.sum(); mean=torch.where(count>0,(values*valid).sum()/count.clamp_min(1),torch.ones((),dtype=values.dtype,device=values.device))
            result.update({"gpt_forecast_router_active_fraction":router_active_fraction,"gpt_forecast_token_gate_mean":mean.detach(),"gpt_forecast_channel_gate_mean":router_channel_gate.mean().detach(),"gpt_forecast_token_gate":values.detach(),"gpt_forecast_channel_gate":router_channel_gate.squeeze(1).detach()})
        return result