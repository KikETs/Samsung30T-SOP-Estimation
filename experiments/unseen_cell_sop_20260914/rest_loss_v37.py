"""Differentiable past-rest consistency supervision; inference stays V36."""
import torch
from relax_state_v36 import Encoder,response,load_pilot_data,STATE_NAMES


def eligible_rest(times,currents):
    begin=times[:,-1]-10.
    overlap=(times[:,1:]>begin[:,None])&(times[:,1:]>times[:,:-1])
    active=torch.any(overlap&(currents[:,:-1]!=0),dim=1)
    return (begin>=0)&~active&(currents[:,-1]==0)


def rest_delta(parameters,times,currents):
    begin=times[:,-1]-10.;right=torch.minimum(times[:,1:],begin[:,None])
    dt=(right-times[:,:-1]).clamp_min(0);lag=(begin[:,None]-right).clamp_min(0)
    result=torch.zeros_like(parameters[:,0])
    for resistance,tau in [(parameters[:,1],parameters[:,2]),(parameters[:,4],parameters[:,5])]:
        history=(currents[:,:-1]*torch.exp(-lag/tau[:,None])*(-torch.expm1(-dt/tau[:,None]))).sum(1)
        result=result-resistance*history*(-torch.expm1(-10./tau))
    return result
