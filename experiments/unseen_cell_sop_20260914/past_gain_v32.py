import numpy as np
import torch
from torch import nn
from run_envelope import HERE
from past_curve_v30 import load_pilot_data as load_curve, STATE_NAMES as CURVE_NAMES
from nonlinear_q_v29 import Encoder as BaseEncoder, response

GAIN_NAMES=[f'past_{h}s_fraction_{f:g}_apparent_gain_V_per_A' for h in (2,10) for f in (.1,.25,.5,.75)]
STATE_NAMES=CURVE_NAMES+GAIN_NAMES


def apparent_gain(curve_values, pre_current):
    curve=np.asarray(curve_values).reshape(-1,2,4,2)
    step=curve[:,:,:,1]-np.asarray(pre_current)[:,:,None]
    voltage=curve[:,:,:,0]
    valid=np.isfinite(step)&np.isfinite(voltage)&(np.abs(step)>=.25)
    return np.divide(voltage,step,out=np.full_like(voltage,np.nan),where=valid).reshape(-1,8)


def load_pilot_data():
    rows,x,*rest=load_curve()
    with np.load(HERE/'past_gain_data_v32/features.npz') as z:
        assert z['row_ids'].tolist()==[r['row_id'] for r in rows]
        gain=z['features']
    return rows,np.column_stack([x[:,:61],gain,x[:,61:]]),*rest


class GainInputLinear(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;state=torch.random.get_rng_state()
        self.extra=nn.Linear(48,32,bias=False);nn.init.zeros_(self.extra.weight)
        torch.random.set_rng_state(state)

    def forward(self,x):
        assert x.shape[1]==142
        base=torch.cat([x[:,:45],x[:,69:71],x[:,71:116],x[:,140:142]],1)
        extra=torch.cat([x[:,45:69],x[:,116:140]],1)
        return self.base(base)+self.extra(extra)


class Encoder(BaseEncoder):
    def __init__(self,n_features):
        if n_features!=71:raise ValueError('Expected71 state features')
        super().__init__(47);self.net[0]=GainInputLinear(self.net[0])
