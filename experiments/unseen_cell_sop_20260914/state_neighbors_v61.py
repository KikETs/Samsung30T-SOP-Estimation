"""Select calibration neighbors using the frozen model's causal state inputs."""
import numpy as np
import torch
from scipy.spatial.distance import cdist
from bounded_reserve_v57 import bounded_state

K=20
VOLTAGE_RADIUS=.10
TIE_ATOL=1e-12
TIE_RTOL=1e-12
KEEP=np.array([j for j in range(148) if j not in (73,147)])

def embed_standardized(z):
    z=np.asarray(z)
    if z.ndim!=2 or z.shape[1]!=148 or not np.isfinite(z).all():raise ValueError('Invalid standardized states')
    with torch.no_grad():result=bounded_state(torch.as_tensor(z)).numpy()
    return result[:,KEEP].astype(float)

def state_features(scaler,x):return embed_standardized(scaler.transform(x))

def state_margins(voltage,residual,states,query_voltage,query_states):
    voltage,residual,states,query_voltage,query_states=[np.asarray(a,float) for a in (voltage,residual,states,query_voltage,query_states)]
    if voltage.ndim!=1 or not len(voltage) or residual.shape!=voltage.shape or query_voltage.ndim!=1:raise ValueError('Invalid calibration arrays')
    if states.shape!=(len(voltage),146) or query_states.shape!=(len(query_voltage),146):raise ValueError('Invalid state shapes')
    if not all(np.isfinite(a).all() for a in [voltage,residual,states,query_voltage,query_states]):raise ValueError('Nonfinite state query')
    n=len(query_voltage);margins=np.empty(n);counts=np.empty(n,int);selected_count=np.empty(n,int);winner=np.empty(n,int);radii=np.empty(n);reasons=[]
    global_winner=int(np.argmax(residual))
    for start in range(0,n,128):
        stop=min(n,start+128);distances=cdist(query_states[start:stop],states,metric='sqeuclidean')
        for offset,qv in enumerate(query_voltage[start:stop]):
            j=start+offset;allowed=np.flatnonzero(abs(voltage-qv)<=VOLTAGE_RADIUS+1e-12);counts[j]=len(allowed)
            if qv<voltage.min() or qv>voltage.max():
                chosen=global_winner;selected_count[j]=len(voltage);radii[j]=-1.;why='outside_voltage_span'
            elif len(allowed)<K:
                chosen=global_winner;selected_count[j]=len(voltage);radii[j]=-1.;why='insufficient_neighbors'
            else:
                ds=distances[offset,allowed];radius=float(np.partition(ds,K-1)[K-1]);radii[j]=radius
                selected=allowed[ds<=radius+TIE_ATOL+TIE_RTOL*abs(radius)]
                chosen=int(selected[np.argmax(residual[selected])]);selected_count[j]=len(selected);why='state_neighbors'
            margins[j]=max(0.,float(residual[chosen]));winner[j]=chosen;reasons.append(why)
    return margins,counts,selected_count,winner,np.asarray(reasons),radii
