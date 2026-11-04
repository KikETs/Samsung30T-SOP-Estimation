"""Causal state encoder and exact left-hold 1RC response, anchored before query."""
import numpy as np
import torch
from torch import nn


def response(parameters, times, currents, query_times, query_currents):
    """times/currents include pre-observation at t=0; all tensors are batched.

    Integrate intervals strictly before each query, allowing nonuniform and
    duplicate sample clocks. Instantaneous R0 uses current at query time.
    No query voltage is accepted. R0,R1,tau,p0,k are per-row state outputs.
    """
    r0,r1,tau,p0,k=parameters.unbind(-1)
    lo=times[:,:-1,None];hi=times[:,1:,None]
    q=query_times[:,None,:]
    duration=(torch.minimum(hi,q)-lo).clamp_min(0)
    lag=(q-torch.minimum(hi,q)).clamp_min(0)
    weight=torch.exp(-lag/tau[:,None,None])*(-torch.expm1(-duration/tau[:,None,None]))
    conv=(currents[:,:-1,None]*weight).sum(1)
    integral=(currents[:,:-1,None]*duration).sum(1)
    relax=-torch.expm1(-query_times/tau[:,None])
    return (r0[:,None]*(query_currents-currents[:,0,None])+
            r1[:,None]*conv-p0[:,None]*relax+k[:,None]*integral)


class Encoder(nn.Module):
    def __init__(self,n_features):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2*n_features,32),nn.SiLU(),
                               nn.Linear(32,32),nn.SiLU(),nn.Linear(32,5))
        # Start near modest positive resistances; no target-based initialization.
        nn.init.zeros_(self.net[-1].weight)
        start=torch.tensor([(.01-.0001)/.1999,(.005-.0001)/.1999,
                            (2.-.05)/59.95,.5,.0001/.005])
        with torch.no_grad(): self.net[-1].bias.copy_(torch.logit(start))

    def forward(self,x):
        q=torch.sigmoid(self.net(x))
        return torch.stack([.0001+.1999*q[:,0],.0001+.1999*q[:,1],
                            .05+59.95*q[:,2],-.2+.4*q[:,3],.005*q[:,4]],dim=1)


class StateScaler:
    def fit(self,x):
        self.medians=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
        filled=np.where(np.isfinite(x),x,self.medians)
        self.mean=filled.mean(0);self.std=filled.std(0)
        self.std=np.where(self.std>1e-8,self.std,1.)
        return self

    def transform(self,x):
        missing=~np.isfinite(x)
        return np.column_stack([(np.where(missing,self.medians,x)-self.mean)/self.std,missing]).astype(np.float32)
