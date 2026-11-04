"""Differentiable 1RC initial polarization from strictly past measured current."""
import numpy as np
import torch
from torch import nn
from run_envelope import HERE


def history_convolution(times,current,tau):
    dt=times[:,1:]-times[:,:-1]
    lag=times[:,-1,None]-times[:,1:]
    weights=torch.exp(-lag/tau[:,None])*(-torch.expm1(-dt/tau[:,None]))
    return (current[:,:-1]*weights).sum(1)


class Encoder(nn.Module):
    def __init__(self,n_features):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2*n_features,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,4))
        nn.init.zeros_(self.net[-1].weight)
        start=torch.tensor([(.01-.0001)/.1999,(.005-.0001)/.1999,(2.-.05)/59.95,.0001/.005])
        with torch.no_grad():self.net[-1].bias.copy_(torch.logit(start))

    def forward(self,x,history_time,history_current):
        q=torch.sigmoid(self.net(x));r0=.0001+.1999*q[:,0];r1=.0001+.1999*q[:,1];tau=.05+59.95*q[:,2];k=.005*q[:,3]
        p0=r1*history_convolution(history_time,history_current,tau)
        return torch.stack([r0,r1,tau,p0,k],1)


def load_histories(rows):
    with np.load(HERE/'polarization_history_data_v1/histories.npz') as z:
        t=z['time_s'];i=z['current_A'];offsets=z['offsets'];mapping=z['row_to_history']
        assert np.array_equal(z['row_ids'],[r['row_id'] for r in rows])
    width=int(np.diff(offsets).max());ht=np.empty((len(offsets)-1,width),np.float32);hi=ht.copy()
    for j,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        n=b-a;ht[j,:n]=t[a:b];ht[j,n:]=t[b-1];hi[j,:n]=i[a:b];hi[j,n:]=i[b-1]
    sign=np.array([-1. if r['direction']=='discharge' else 1. for r in rows],np.float32)
    return ht[mapping],hi[mapping]*sign[:,None]
