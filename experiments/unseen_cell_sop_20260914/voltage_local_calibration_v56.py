"""Calibration uses same-horizon calibration-cell rows and pre-decision voltage."""
import numpy as np

RADIUS_V=.10
MIN_NEIGHBORS=20

def local_margins(voltage, residual, query):
    voltage,residual,query=[np.asarray(a,float) for a in (voltage,residual,query)]
    if voltage.ndim!=1 or residual.shape!=voltage.shape or query.ndim!=1 or not len(voltage):
        raise ValueError('Invalid calibration/query arrays')
    if not all(np.isfinite(a).all() for a in [voltage,residual,query]):
        raise ValueError('Nonfinite calibration/query')
    order=np.argsort(voltage,kind='stable');v=voltage[order]
    lo=np.searchsorted(v,query-RADIUS_V-1e-12,side='left')
    hi=np.searchsorted(v,query+RADIUS_V+1e-12,side='right')
    count=hi-lo;global_winner=int(np.argmax(residual))
    margin=np.empty(len(query));winner=np.empty(len(query),int);reason=[]
    for j,q in enumerate(query):
        if q<v[0] or q>v[-1]:chosen=global_winner;why='outside_voltage_span'
        elif count[j]<MIN_NEIGHBORS:chosen=global_winner;why='insufficient_neighbors'
        else:
            candidates=order[lo[j]:hi[j]]
            chosen=int(candidates[np.argmax(residual[candidates])]);why='local'
        margin[j]=max(0.,float(residual[chosen]));winner[j]=chosen;reason.append(why)
    return margin,count,winner,np.asarray(reason)
