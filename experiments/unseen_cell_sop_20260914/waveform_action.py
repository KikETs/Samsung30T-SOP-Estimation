"""Fixed waveform descriptors; planned constant actions need no future sensor data."""
import numpy as np

TIME_CONSTANTS=(.1,.5,2.,10.,60.)


def action_features(time,current):
    t=np.asarray(time,float);i=np.asarray(current,float)
    if t.ndim!=1 or i.shape!=t.shape or len(t)<2:raise ValueError('Need a sampled profile')
    if not np.isfinite(t).all() or not np.isfinite(i).all():raise ValueError('Nonfinite profile')
    dt=np.diff(t);duration=float(t[-1]-t[0])
    if duration<=0 or (dt<0).any():raise ValueError('Invalid profile clock')
    result=dict(duration_s=duration,endpoint_current_A=float(i[-1]),
        mean_current_A=float(np.sum(i[:-1]*dt)/duration),integral_current_As=float(np.sum(i[:-1]*dt)))
    # Integral exp(-(end-s)/lambda)/lambda * I(s) ds for left-sample holds.
    weights=np.exp(-(t[-1]-t[1:])[:,None]/np.array(TIME_CONSTANTS))*\
        (-np.expm1(-dt[:,None]/np.array(TIME_CONSTANTS)))
    linear=i[:-1]@weights;square=(i[:-1]**2)@weights
    for k,(a,b) in enumerate(zip(linear,square)):
        result[f'conv_{k}_A']=float(a);result[f'conv_square_{k}_A2']=float(b)
    return result


def constant_action(current_A,horizon_s):
    if not np.isfinite([current_A,horizon_s]).all() or current_A<0 or horizon_s<=0:
        raise ValueError('Constant action needs nonnegative current and positive horizon')
    out=dict(duration_s=float(horizon_s),endpoint_current_A=float(current_A),
        mean_current_A=float(current_A),integral_current_As=float(current_A*horizon_s))
    for k,decay in enumerate(TIME_CONSTANTS):
        weight=-np.expm1(-horizon_s/decay)
        out[f'conv_{k}_A']=float(current_A*weight);out[f'conv_square_{k}_A2']=float(current_A**2*weight)
    return out


def observed_prefix(time,current,voltage,start,endpoint):
    """The only raw-data access ends at endpoint, including endpoint voltage."""
    t=np.asarray(time[start:endpoint+1],float)
    i=np.asarray(current[start:endpoint+1],float)
    v=np.asarray(voltage[start:endpoint+1],float)
    if not np.isfinite(v).all():raise ValueError('Nonfinite voltage prefix')
    return t-t[0],i,v
