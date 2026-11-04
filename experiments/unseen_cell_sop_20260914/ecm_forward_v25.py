"""Fixed 1RC encoder training, full waveform response and all-time SOP inverse."""
from collections import Counter
import numpy as np
import torch
from differentiable_ecm_v1 import Encoder,StateScaler,response


def constant_terms(parameters,pre,gap,horizon):
    r0,r1,tau,p0,k=np.asarray(parameters,float).T
    h=np.broadcast_to(np.asarray(horizon,float),r0.shape)
    if not np.isfinite(h).all() or (h<0).any():raise ValueError('Invalid horizon')
    decay=np.exp(-h/tau);predecay=np.exp(-gap/tau)
    a=r0+r1*(1-decay)+k*h
    b=-r0*pre-p0+k*pre*gap+(r1*pre*(1-predecay)+p0*predecay)*decay
    return a,b


def invert(parameters,pre,gap,horizon,headroom,cap,margin=0.):
    h=np.asarray(horizon,float)
    if not np.isfinite(h).all() or (h<=0).any():raise ValueError('Positive finite horizon required')
    head=np.asarray(headroom,float)-.020-margin
    if not np.isfinite(head).all() or not np.isfinite(cap) or cap<0 or not np.isfinite(margin) or margin<0:
        raise ValueError('Invalid current bound inputs')
    a0,b0=constant_terms(parameters,pre,gap,0.)
    ah,bh=constant_terms(parameters,pre,gap,horizon)
    current=np.clip(np.minimum((head-b0)/a0,(head-bh)/ah),0.,cap)
    zero_infeasible=np.maximum(b0,bh)>head
    return current,zero_infeasible


class Forward:
    def fit(self,x,t,i,y,lengths,cells,ids):
        torch.set_num_threads(1);torch.manual_seed(17);rng=np.random.default_rng(17)
        torch.use_deterministic_algorithms(True)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fit_ids=list(ids);self.fit_cells=sorted(set(cells));self.scaler=StateScaler().fit(x)
        self.negative_y_count=int((y[np.arange(len(y)),lengths-1]<0).sum())
        self.cap=float(1.5*np.max(i[np.arange(len(i)),lengths-1]))
        features=torch.tensor(self.scaler.transform(x),device=device)
        times=torch.tensor(t,device=device);current=torch.tensor(i,device=device)
        target=torch.tensor(y,device=device);lens=torch.tensor(lengths,device=device)
        counts=Counter(cells);weight=torch.tensor([len(x)/(len(counts)*counts[c]) for c in cells],device=device,dtype=torch.float32)
        model=Encoder(x.shape[1]).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        self.training_log=[]
        for epoch in range(80):
            order=rng.permutation(len(x));total=0.
            for off in range(0,len(order),512):
                ix=torch.tensor(order[off:off+512],device=device)
                inner=(torch.rand((len(ix),8),device=device)*(lens[ix,None]-2)).long()+1
                query=torch.cat([inner,lens[ix,None]-1],1)
                qt=torch.gather(times[ix],1,query);qi=torch.gather(current[ix],1,query)
                yy=torch.gather(target[ix],1,query)
                prediction=response(model(features[ix]),times[ix],current[ix],qt,qi)
                loss=((prediction-yy).square().mean(1)*weight[ix]).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss.detach())*len(ix)
            self.training_log.append(total/len(x))
        self.n_features=x.shape[1];self.state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        return self

    def parameters(self,x):
        model=Encoder(self.n_features);model.load_state_dict(self.state);model.eval()
        with torch.no_grad():return model(torch.from_numpy(self.scaler.transform(x))).numpy().astype(float)

    def trajectory(self,x,t,i):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        p=self.parameters(x);out=[]
        with torch.no_grad():
            for start in range(0,len(x),256):
                sl=slice(start,start+256)
                # Double precision physical calculation isolates float32 encoder roundoff.
                tt=torch.tensor(t[sl],device=device,dtype=torch.double)
                ii=torch.tensor(i[sl],device=device,dtype=torch.double)
                out.append(response(torch.tensor(p[sl],device=device),tt,ii,tt,ii).cpu().numpy())
        return np.concatenate(out)

    def sop(self,x,horizon,headroom,margin=0.):
        return invert(self.parameters(x),x[:,-2],x[:,-1],horizon,headroom,self.cap,margin)[0]

    def zero_infeasible(self,x,horizon,headroom,margin=0.):
        return invert(self.parameters(x),x[:,-2],x[:,-1],horizon,headroom,self.cap,margin)[1]
