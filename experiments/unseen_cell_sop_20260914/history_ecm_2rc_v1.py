"""Two RC branches with causally integrated initial polarization."""
import torch
from torch import nn
from history_ecm_v1 import history_convolution
from differentiable_ecm_v1 import response as one_rc_response


class Encoder(nn.Module):
    def __init__(self,n_features):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2*n_features,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,6))
        nn.init.zeros_(self.net[-1].weight)
        initial=torch.tensor([(.01-.0001)/.1999,(.0025-.00005)/.09995,(.5-.05)/59.90,
                              (.0025-.00005)/.09995,(10.-.5-.0001)/(60.-.5-.0001),.0001/.005])
        with torch.no_grad():self.net[-1].bias.copy_(torch.logit(initial))

    def forward(self,x,history_time,history_current):
        q=torch.sigmoid(self.net(x));r0=.0001+.1999*q[:,0];r1=.00005+.09995*q[:,1]
        tau1=.05+59.90*q[:,2];r2=.00005+.09995*q[:,3]
        tau2=tau1+.0001+(60.-tau1-.0001)*q[:,4];k=.005*q[:,5]
        p01=r1*history_convolution(history_time,history_current,tau1)
        p02=r2*history_convolution(history_time,history_current,tau2)
        return torch.stack([r0,r1,tau1,p01,r2,tau2,p02,k],1)


def response(parameters,times,currents,query_times,query_currents):
    p=parameters;zero=torch.zeros_like(p[:,0])
    first=torch.stack([p[:,0],p[:,1],p[:,2],p[:,3],p[:,7]],1)
    second=torch.stack([zero,p[:,4],p[:,5],p[:,6],zero],1)
    return one_rc_response(first,times,currents,query_times,query_currents)+one_rc_response(second,times,currents,query_times,query_currents)
