"""Bound neural values and train at 90% reference, preserving physical inputs."""
import torch
from gap_invariant_v52 import Encoder as GapEncoder
from group_utility_v42 import sampled_peak,group_loss

VALUE_BOUND=5.
TRAINING_UTILITY_FRACTION=.9

def bounded_state(x):
    if x.ndim!=2 or x.shape[1]!=148:raise ValueError('Expected74 values and74 missing flags')
    return torch.cat([VALUE_BOUND*torch.tanh(x[:,:74]/VALUE_BOUND),x[:,74:]],dim=1)

class Encoder(GapEncoder):
    def __init__(self,n_features):
        super().__init__(n_features)
        self.register_buffer('_bounded_reserve_v57',torch.tensor([VALUE_BOUND,TRAINING_UTILITY_FRACTION]))
    def forward(self,x,history_time,history_current):
        return super().forward(bounded_state(x),history_time,history_current)

def utility_loss(p,pre,gap,horizon,reference,headroom,duration,groups):
    boundary=.1*torch.exp(-duration/p[:,2])+.1*torch.exp(-duration/p[:,5])
    residual=sampled_peak(p,pre,gap,horizon,TRAINING_UTILITY_FRACTION*reference)-(headroom-.02-boundary)+1e-6
    return group_loss(residual,groups)
