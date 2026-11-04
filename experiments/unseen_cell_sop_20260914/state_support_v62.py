"""Calibration-only leave-self-out neighborhoods define state support balls."""
import numpy as np
from scipy.spatial.distance import cdist

K=20
RADIUS_V=.10

def calibration_radii(voltage,states):
    voltage,states=np.asarray(voltage,float),np.asarray(states,float)
    if voltage.ndim!=1 or not len(voltage) or states.shape!=(len(voltage),146):raise ValueError('Invalid calibration states')
    if not np.isfinite(voltage).all() or not np.isfinite(states).all():raise ValueError('Nonfinite calibration states')
    radius=np.full(len(voltage),-1.);counts=np.zeros(len(voltage),int)
    for start in range(0,len(voltage),128):
        distance=cdist(states[start:start+128],states,metric='sqeuclidean')
        for k in range(len(distance)):
            j=start+k;mask=abs(voltage-voltage[j])<=RADIUS_V+1e-12;mask[j]=False;counts[j]=mask.sum()
            if counts[j]>=K:radius[j]=np.partition(distance[k,mask],K-1)[K-1]
    return radius,counts

def support_counts(voltage,states,radii,query_voltage,query_states):
    voltage,states,radii,query_voltage,query_states=[np.asarray(a,float) for a in (voltage,states,radii,query_voltage,query_states)]
    if voltage.ndim!=1 or not len(voltage) or states.shape!=(len(voltage),146) or radii.shape!=voltage.shape:raise ValueError('Invalid support arrays')
    if query_voltage.ndim!=1 or query_states.shape!=(len(query_voltage),146):raise ValueError('Invalid query arrays')
    if not all(np.isfinite(a).all() for a in [voltage,states,radii,query_voltage,query_states]):raise ValueError('Nonfinite support arrays')
    covers=np.zeros(len(query_voltage),int);witness=np.full(len(query_voltage),-1,int)
    for start in range(0,len(query_voltage),128):
        distance=cdist(query_states[start:start+128],states,metric='sqeuclidean')
        valid=(abs(query_voltage[start:start+128,None]-voltage[None,:])<=RADIUS_V+1e-12)&(radii[None,:]>=0)
        inside=valid&(distance<=radii[None,:]+1e-12+1e-12*abs(radii[None,:]))
        covers[start:start+len(distance)]=inside.sum(axis=1)
        for k in range(len(distance)):
            if inside[k].any():witness[start+k]=np.flatnonzero(inside[k])[0]
    return covers,witness
