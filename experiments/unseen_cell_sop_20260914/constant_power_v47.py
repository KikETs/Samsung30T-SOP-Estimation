"""Planned discharge constant-power solver for the frozen monotone 2RC model.

The algebraic current is the low-current stable branch of P=I*V. No future
measured current/voltage is accepted. Adaptive ODE event detection is numerical,
not an exact continuous-time certificate; independent refinement is required.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from nonlinear_q_sop_v29 import _segment


def drop(i,p):return p[0]*i+p[9]*10*np.tanh(i/10)


def solve_current(c,p,power):
    if power==0:return 0.
    if c<=0:return 0.
    # Restrict to V>=C/2. On this domain I*d(drop)/dI<=drop<=C/2,
    # so d(I*V)/dI>=0 for linear plus concave tanh voltage drop.
    hi=brentq(lambda i:drop(i,p)-c/2,0,c/(2*p[0]),xtol=1e-11)
    if power>=hi*c/2:return hi  # continuation only beyond feasible domain
    return brentq(lambda i:i*(c-drop(i,p))-power,0,hi,xtol=1e-10)


def evaluate(p,vpre,pre,gap,horizon,power,current_cap=30.,voltage_floor=2.57,
             rtol=2e-7,max_step=None,return_profile=False):
    p=np.asarray(p,float)
    if p.shape!=(10,) or not np.isfinite(p).all() or p[0]<=0 or min(p[2],p[5])<=0 or (p[[1,4,7,8,9]]<0).any():
        raise ValueError('Invalid passive 2RC parameters')
    if not np.isfinite([vpre,pre,gap,horizon,power,current_cap,voltage_floor]).all() or min(gap,power,current_cap)<0 or min(horizon,voltage_floor)<=0:
        raise ValueError('Invalid planned state or constraint')
    # Initial measured point and pre-switch constant-current interval.
    previous=_segment(p[:9],(p[3],p[6]),0.,pre,gap,pre)[0]
    previous_min=vpre-max(0.,previous)
    if previous_min<voltage_floor-1e-10:
        return dict(feasible=False,status='pre_gap_voltage',completed_s=0.,min_voltage_V=previous_min,max_current_A=abs(pre))
    decay1=np.exp(-gap/p[2]);decay2=np.exp(-gap/p[5])
    y0=np.array([p[3]*decay1+p[1]*pre*(1-decay1),p[6]*decay2+p[4]*pre*(1-decay2),pre*gap])

    def c_at(y):
        s1,s2,q=y
        return vpre+drop(pre,p)-(s1-p[3])-(s2-p[6])-p[7]*q-p[8]*q*abs(q)

    def ceiling(c):
        if c<=voltage_floor:return 0.
        if c>2*voltage_floor:return -1.  # explicitly unsupported branch domain
        imax=current_cap
        if c-drop(imax,p)<voltage_floor:
            imax=brentq(lambda i:c-drop(i,p)-voltage_floor,0,imax,xtol=1e-10)
        return imax*(c-drop(imax,p))

    def event(t,y):return ceiling(c_at(y))-power
    event.terminal=True;event.direction=-1

    def rhs(t,y):
        current=solve_current(c_at(y),p,power)
        return [(p[1]*current-y[0])/p[2],(p[4]*current-y[1])/p[5],current]

    c0=c_at(y0);available=ceiling(c0)
    if available<0:return dict(feasible=False,status='unsupported_stable_domain',completed_s=0.,min_voltage_V=c0,max_current_A=0.)
    if available<power-1e-10:return dict(feasible=False,status='initial_power_limit',completed_s=0.,min_voltage_V=c0,max_current_A=0.)
    sol=solve_ivp(rhs,[0,horizon],y0,rtol=rtol,atol=rtol*1e-3,
                  max_step=max_step if max_step is not None else horizon/80,
                  events=event,dense_output=True)
    if not sol.success:raise RuntimeError(sol.message)
    # Include solver nodes plus uniform points. This is a numerical audit, not
    # proof that no infinitesimal intermediate limit crossing is possible.
    times=np.unique(np.r_[sol.t,np.linspace(0,sol.t[-1],161)])
    states=sol.sol(times).T
    cc=np.array([c_at(y) for y in states])
    ii=np.array([solve_current(c,p,power) for c in cc]);vv=cc-np.array([drop(i,p) for i in ii])
    domain=bool((cc<=2*voltage_floor+1e-10).all())
    finished=sol.t[-1]>=horizon-1e-8
    feasible=bool(finished and domain and vv.min()>=voltage_floor-1e-8 and ii.max()<=current_cap+1e-8)
    result=dict(feasible=feasible,status='feasible' if feasible else 'constraint_event' if domain else 'unsupported_stable_domain',
                completed_s=float(sol.t[-1]),min_voltage_V=float(min(vv.min(),previous_min)),max_current_A=float(ii.max()),
                max_power_residual_W=float(np.max(abs(ii*vv-power))),rhs_evaluations=int(sol.nfev))
    if return_profile:result.update(time_s=times.tolist(),voltage_V=vv.tolist(),current_A=ii.tolist())
    return result


def sop(p,vpre,pre,gap,horizon,current_cap=30.,voltage_floor=2.57,power_tolerance_W=.002):
    zero=evaluate(p,vpre,pre,gap,horizon,0.,current_cap,voltage_floor)
    if not zero['feasible']:return dict(power_W=0.,bracket_width_W=0.,zero_infeasible=True,evaluation=zero)
    lo=0.;hi=current_cap*max(vpre+drop(pre,p)+abs(p[3])+abs(p[6]),voltage_floor)
    upper=evaluate(p,vpre,pre,gap,horizon,hi,current_cap,voltage_floor)
    assert not upper['feasible'],'Invalid upper power bracket'
    last=zero
    while hi-lo>power_tolerance_W:
        mid=(lo+hi)/2
        r=evaluate(p,vpre,pre,gap,horizon,mid,current_cap,voltage_floor)
        if r['feasible']:lo=mid;last=r
        else:hi=mid
    return dict(power_W=float(lo),bracket_width_W=float(hi-lo),zero_infeasible=False,evaluation=last)
