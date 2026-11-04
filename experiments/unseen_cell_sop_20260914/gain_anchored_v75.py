"""Fixed common gain-conditioned resistance inference; no per-cell fitting."""
import numpy as np
import torch
from bounded_reserve_v57 import Encoder as BaseEncoder,utility_loss

def gain_anchor(raw_features):
    x=np.asarray(raw_features)
    if x.ndim!=2 or x.shape[1]!=74:raise ValueError('Expected original74 causal features')
    values=x[:,61:69];valid=np.isfinite(values)&(values>0)
    ordered=np.sort(np.where(valid,values,np.inf),axis=1);count=valid.sum(1)
    lo=np.maximum(0,(count-1)//2);hi=np.maximum(0,count//2)
    median=(ordered[np.arange(len(x)),lo]+ordered[np.arange(len(x)),hi])/2
    return np.where(count>0,median,np.nan)

def training_gain_reference(raw_training):
    g=gain_anchor(raw_training);valid=g[np.isfinite(g)]
    if not len(valid):raise ValueError('No positive causal training gain')
    return float(np.median(valid))

class Encoder(BaseEncoder):
    def __init__(self,n_features,gain_reference=1.):
        if not np.isfinite(gain_reference) or gain_reference<=0:raise ValueError('Invalid training-only gain reference')
        super().__init__(n_features)
        self.register_buffer('_gain_anchored_v75',torch.tensor([gain_reference],dtype=torch.float64))
    def forward(self,x,history_time,history_current,gain):
        p=super().forward(x,history_time,history_current)
        if gain.shape!=(len(p),):raise ValueError('Expected one causal gain per row')
        g=gain.to(dtype=torch.float64,device=p.device);g0=self._gain_anchored_v75[0]
        g=torch.where(torch.isfinite(g)&(g>0),g,g0);scale=g/g0
        if not bool(torch.isfinite(scale).all()):raise ValueError('Nonfinite gain scale')
        replacements={}
        for col,low,high in [(0,.0001,.2),(1,.00005,.1),(4,.00005,.1),(9,0.,.1)]:
            q=((p[:,col].double()-low)/(high-low)).clamp(0,1)
            sq=scale*q;denominator=(1-q)+sq
            mapped=low+(high-low)*sq/denominator
            replacements[col]=torch.where(scale==1,p[:,col],mapped.to(p.dtype))
        replacements[3]=p[:,3]*(replacements[1]/p[:,1]);replacements[6]=p[:,6]*(replacements[4]/p[:,4])
        return torch.stack([replacements.get(j,p[:,j]) for j in range(10)],1)
