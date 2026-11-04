"""Differentiable sampled-time peak and strict-majority group utility surrogate."""
import torch

def sampled_peak(p,pre,gap,horizon,current):
    dtype,device=p.dtype,p.device
    fractions=torch.cat([torch.linspace(0,1,33,dtype=dtype,device=device),torch.logspace(-4,0,17,dtype=dtype,device=device)])
    t=horizon[:,None]*fractions[None];q=pre[:,None]*gap[:,None]+current[:,None]*t
    future=p[:,0,None]*(current-pre)[:,None]+p[:,7,None]*q+p[:,8,None]*q*q.abs()+10*p[:,9,None]*(torch.tanh(current/10)-torch.tanh(pre/10))[:,None]
    past_t=gap[:,None]*torch.linspace(0,1,5,dtype=dtype,device=device)[None];past_q=pre[:,None]*past_t
    past=p[:,7,None]*past_q+p[:,8,None]*past_q*past_q.abs()
    for resistance,tau,initial in [(1,2,3),(4,5,6)]:
        decay=torch.exp(-gap/p[:,tau]);start=p[:,initial]*decay+p[:,resistance]*pre*(1-decay)
        future_decay=torch.exp(-t/p[:,tau,None])
        future=future+start[:,None]*future_decay+p[:,resistance,None]*current[:,None]*(1-future_decay)-p[:,initial,None]
        past_decay=torch.exp(-past_t/p[:,tau,None])
        past=past+p[:,initial,None]*(past_decay-1)+p[:,resistance,None]*pre[:,None]*(1-past_decay)
    return torch.cat([torch.zeros((len(p),1),dtype=dtype,device=device),past,future],1).max(1).values

def group_loss(residual,groups):
    losses=[]
    for indices in groups:
        if len(indices)==0:raise ValueError('Empty training group')
        median_upper=torch.sort(residual[indices],stable=True).values[len(indices)//2]
        losses.append(torch.relu(median_upper).square())
    if not losses:raise ValueError('No training groups')
    return torch.stack(losses).mean()

def utility_loss(p,pre,gap,horizon,reference,headroom,duration,groups):
    boundary=.1*torch.exp(-duration/p[:,2])+.1*torch.exp(-duration/p[:,5])
    residual=sampled_peak(p,pre,gap,horizon,.8*reference)-(headroom-.02-boundary)+1e-6
    return group_loss(residual,groups)
