"""Strictly pre-decision temperature and relaxation features for V35."""
import numpy as np
import torch
from torch import nn
from run_envelope import HERE
from past_gain_v32 import Encoder as BaseEncoder,load_pilot_data as load_base,STATE_NAMES as BASE_NAMES,response

EXTRA_NAMES=['pre_temperature_5s_C','log1p_pre_rest_s','oriented_pre_slope_10s_V_per_s']
STATE_NAMES=BASE_NAMES+EXTRA_NAMES


def extract_prefix(t,current,voltage,temperature,sign):
    t=np.asarray(t,float);i=np.asarray(current,float);v=np.asarray(voltage,float);temp=np.asarray(temperature,float)
    if not(len(t)==len(i)==len(v)==len(temp)) or not len(t) or not np.isfinite(t).all():
        raise ValueError('Invalid past prefix')
    reset=np.flatnonzero(np.diff(t)<0);begin=int(reset[-1]+1) if len(reset) else 0
    t,i,v,temp=(z[begin:] for z in [t,i,v,temp])
    keep=np.r_[np.diff(t)>0,True];t,i,v,temp=(z[keep] for z in [t,i,v,temp])
    end=t[-1];recent=(t>=end-5)&np.isfinite(temp)
    temperature_value=float(np.median(temp[recent])) if recent.any() else np.nan
    active=np.flatnonzero((np.abs(i)>.1)|~np.isfinite(i))
    rest_start=int(active[-1]+1) if len(active) else 0
    rest=float(end-t[rest_start]) if rest_start<len(t) else 0.
    anchor=end-10.;a=int(np.searchsorted(t,anchor,side='right'))-1
    slope=sign*(v[-1]-np.interp(anchor,t,v))/10. if a>=0 and np.isfinite(v[a:]).all() else np.nan
    return np.array([temperature_value,np.log1p(rest),slope])


def load_pilot_data():
    rows,x,*rest=load_base()
    with np.load(HERE/'relax_state_data_v36/features.npz') as z:
        assert z['row_ids'].tolist()==[r['row_id'] for r in rows]
        extra=z['features']
    return rows,np.column_stack([x[:,:69],extra,x[:,69:]]),*rest


class RelaxInputLinear(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;state=torch.random.get_rng_state()
        self.extra=nn.Linear(6,32,bias=False);nn.init.zeros_(self.extra.weight)
        torch.random.set_rng_state(state)
    def forward(self,x):
        if x.shape[1]!=148:raise ValueError('Expected74 values and74 masks')
        base=torch.cat([x[:,:69],x[:,72:74],x[:,74:143],x[:,146:148]],1)
        extra=torch.cat([x[:,69:72],x[:,143:146]],1)
        return self.base(base)+self.extra(extra)


class Encoder(BaseEncoder):
    def __init__(self,n_features):
        if n_features!=74:raise ValueError('Expected74 state features')
        super().__init__(71);self.net[0]=RelaxInputLinear(self.net[0])
