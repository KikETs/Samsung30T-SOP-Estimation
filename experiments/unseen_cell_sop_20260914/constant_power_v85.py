"""Constant-power SOP for both directions on the frozen oriented 2RC+charge+saturation model.

Discharge delegates to the verified V47 solver (voltage floor, 30 A cap). Charge mirrors it in the oriented frame:
V = c_ch(state) + drop(I) with c_ch = vpre - drop(pre) + (s1-p3) + (s2-p6) + k q + b q|q|, constraint V <= ceiling,
P = I*V strictly increasing in I. No future measured current or voltage is accepted."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from nonlinear_q_sop_v29 import _segment
from constant_power_v47 import drop,sop as discharge_sop,evaluate as discharge_evaluate

def charge_current(c,p,power,cap):
    if power<=0:return 0.
    hi=cap
    if c+drop(hi,p)<=0:raise ValueError('Invalid charge level')
    if hi*(c+drop(hi,p))<power:return cap  # continuation only beyond feasible domain
    return brentq(lambda i:i*(c+drop(i,p))-power,0.,hi,xtol=1e-10)

def charge_evaluate(p,vpre,pre,gap,horizon,power,current_cap=15.,voltage_ceiling=4.13,rtol=2e-7,max_step=None,return_profile=False):
    p=np.asarray(p,float)
    if p.shape!=(10,) or not np.isfinite(p).all() or p[0]<=0 or min(p[2],p[5])<=0 or (p[[1,4,7,8,9]]<0).any():raise ValueError('Invalid passive 2RC parameters')
    if not np.isfinite([vpre,pre,gap,horizon,power,current_cap,voltage_ceiling]).all() or min(gap,power,current_cap)<0 or min(horizon,voltage_ceiling)<=0:raise ValueError('Invalid planned state or constraint')
    previous=_segment(p[:9],(p[3],p[6]),0.,pre,gap,pre)[0];previous_max=vpre+max(0.,previous)
    if previous_max>voltage_ceiling+1e-10:return dict(feasible=False,status='pre_gap_voltage',completed_s=0.,max_voltage_V=previous_max,max_current_A=abs(pre))
    decay1=np.exp(-gap/p[2]);decay2=np.exp(-gap/p[5])
    y0=np.array([p[3]*decay1+p[1]*pre*(1-decay1),p[6]*decay2+p[4]*pre*(1-decay2),pre*gap])
    def c_at(y):
        s1,s2,q=y
        return vpre-drop(pre,p)+(s1-p[3])+(s2-p[6])+p[7]*q+p[8]*q*abs(q)
    def ceiling(c):
        if c>=voltage_ceiling:return 0.
        imax=current_cap
        if c+drop(imax,p)>voltage_ceiling:imax=brentq(lambda i:c+drop(i,p)-voltage_ceiling,0.,imax,xtol=1e-10)
        return imax*(c+drop(imax,p))
    def event(t,y):return ceiling(c_at(y))-power
    event.terminal=True;event.direction=-1
    def rhs(t,y):
        current=charge_current(c_at(y),p,power,current_cap)
        return [(p[1]*current-y[0])/p[2],(p[4]*current-y[1])/p[5],current]
    c0=c_at(y0);available=ceiling(c0)
    if available<power-1e-10:return dict(feasible=False,status='initial_power_limit',completed_s=0.,max_voltage_V=c0,max_current_A=0.)
    sol=solve_ivp(rhs,[0,horizon],y0,rtol=rtol,atol=rtol*1e-3,max_step=max_step if max_step is not None else horizon/80,events=event,dense_output=True)
    if not sol.success:raise RuntimeError(sol.message)
    times=np.unique(np.r_[sol.t,np.linspace(0,sol.t[-1],161)]);states=sol.sol(times).T
    cc=np.array([c_at(y) for y in states]);ii=np.array([charge_current(c,p,power,current_cap) for c in cc]);vv=cc+np.array([drop(i,p) for i in ii])
    finished=sol.t[-1]>=horizon-1e-8
    feasible=bool(finished and vv.max()<=voltage_ceiling+1e-8 and ii.max()<=current_cap+1e-8)
    result=dict(feasible=feasible,status='feasible' if feasible else 'constraint_event',completed_s=float(sol.t[-1]),max_voltage_V=float(max(vv.max(),previous_max)),max_current_A=float(ii.max()),
                max_power_residual_W=float(np.max(abs(ii*vv-power))),rhs_evaluations=int(sol.nfev))
    if return_profile:result.update(time_s=times.tolist(),voltage_V=vv.tolist(),current_A=ii.tolist())
    return result

def charge_sop(p,vpre,pre,gap,horizon,current_cap=15.,voltage_ceiling=4.13,power_tolerance_W=.002):
    zero=charge_evaluate(p,vpre,pre,gap,horizon,0.,current_cap,voltage_ceiling)
    if not zero['feasible']:return dict(power_W=0.,bracket_width_W=0.,zero_infeasible=True,evaluation=zero)
    lo=0.;hi=current_cap*voltage_ceiling*1.001
    upper=charge_evaluate(p,vpre,pre,gap,horizon,hi,current_cap,voltage_ceiling);assert not upper['feasible'],'Invalid upper power bracket'
    last=zero
    while hi-lo>power_tolerance_W:
        mid=(lo+hi)/2;r=charge_evaluate(p,vpre,pre,gap,horizon,mid,current_cap,voltage_ceiling)
        if r['feasible']:lo=mid;last=r
        else:hi=mid
    return dict(power_W=float(lo),bracket_width_W=float(hi-lo),zero_infeasible=False,evaluation=last)

def sop(p,vpre,pre,gap,horizon,direction,current_cap,voltage_limit,power_tolerance_W=.002):
    if direction=='discharge':return discharge_sop(p,vpre,pre,gap,horizon,current_cap=current_cap,voltage_floor=voltage_limit,power_tolerance_W=power_tolerance_W)
    if direction=='charge':return charge_sop(p,vpre,pre,gap,horizon,current_cap=current_cap,voltage_ceiling=voltage_limit,power_tolerance_W=power_tolerance_W)
    raise ValueError(direction)

def replay(p,vpre,pre,gap,power,horizon,direction,max_step):
    """Independent vectorized RK4/Newton replay for both directions (extends verify_constant_power_v47.replay)."""
    p=np.asarray(p,float);vpre=np.asarray(vpre,float);pre=np.asarray(pre,float);gap=np.asarray(gap,float);power=np.asarray(power,float)
    sign=-1. if direction=='discharge' else 1.
    d1=np.exp(-gap/p[:,2]);d2=np.exp(-gap/p[:,5])
    state=np.column_stack([p[:,3]*d1+p[:,1]*pre*(1-d1),p[:,6]*d2+p[:,4]*pre*(1-d2),pre*gap])
    offset=vpre-sign*(p[:,0]*pre+p[:,9]*10*np.tanh(pre/10))-sign*(p[:,3]+p[:,6])
    maximum_residual=0.
    def algebra(y):
        nonlocal maximum_residual
        c=offset+sign*(y[:,0]+y[:,1]+p[:,7]*y[:,2]+p[:,8]*y[:,2]*abs(y[:,2]))
        j=power/np.maximum(c,1e-6)
        for _ in range(60):
            tanh=np.tanh(j/10);dr=p[:,0]*j+p[:,9]*10*tanh;volts=c+sign*dr;f=j*volts-power
            derivative=volts+sign*j*(p[:,0]+p[:,9]*(1-tanh*tanh))
            if (derivative<=0).any():raise ValueError('Stable branch not available in independent replay')
            update=f/derivative;j=j-update
            if np.max(abs(update))<1e-12:break
        else:raise ValueError('Independent Newton did not converge')
        volts=c+sign*(p[:,0]*j+p[:,9]*10*np.tanh(j/10));maximum_residual=max(maximum_residual,float(np.max(abs(j*volts-power))))
        return j,volts
    def rhs(y):
        j,_=algebra(y);return np.column_stack([(p[:,1]*j-y[:,0])/p[:,2],(p[:,4]*j-y[:,1])/p[:,5],j])
    n=int(np.ceil(horizon/max_step));dt=horizon/n
    j,v=algebra(state);vext=v.copy();imax=j.copy()
    for _ in range(n):
        a=rhs(state);b=rhs(state+dt*a/2);c=rhs(state+dt*b/2);d=rhs(state+dt*c);state=state+dt*(a+2*b+2*c+d)/6
        j,v=algebra(state);vext=np.minimum(vext,v) if direction=='discharge' else np.maximum(vext,v);imax=np.maximum(imax,j)
    return vext,imax,maximum_residual
