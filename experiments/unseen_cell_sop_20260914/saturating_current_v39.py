"""Monotone saturating ohmic contribution with unchanged causal two-RC dynamics."""
import torch
from torch import nn
from relax_state_v36 import Encoder as BaseEncoder,load_pilot_data,STATE_NAMES
from nonlinear_q_v29 import response as base_response
from rest_loss_v37 import rest_delta,eligible_rest

CURRENT_SCALE=10.
SAT_MAX=.1


def saturate(current):return CURRENT_SCALE*torch.tanh(current/CURRENT_SCALE)


class Encoder(BaseEncoder):
    def __init__(self,n_features):
        super().__init__(n_features);state=torch.random.get_rng_state()
        self.saturation_head=nn.Linear(32,1);nn.init.zeros_(self.saturation_head.weight)
        with torch.no_grad():self.saturation_head.bias.fill_(float(torch.logit(torch.tensor(.0001/SAT_MAX))))
        torch.random.set_rng_state(state)
    def forward(self,x,history_time,history_current):
        p=super().forward(x,history_time,history_current)
        r=SAT_MAX*torch.sigmoid(self.saturation_head(self.net[:-1](x)))
        return torch.cat([p,r],dim=1)


def response(parameters,times,currents,query_times,query_currents):
    return base_response(parameters[:,:9],times,currents,query_times,query_currents)+parameters[:,9,None]*(saturate(query_currents)-saturate(currents[:,0,None]))


def cross_error(parameters,times,currents,targets,lengths):
    fractions=(torch.arange(8,device=times.device,dtype=times.dtype)+.5)/8
    inner=(fractions[None]*(lengths[:,None]-2)).long()+1
    query=torch.cat([inner,lengths[:,None]-1],1)
    qt=torch.gather(times,1,query);qi=torch.gather(currents,1,query);yy=torch.gather(targets,1,query)
    return torch.relu((response(parameters,times,currents,qt,qi)-yy).abs()-.010).square().mean(1)
