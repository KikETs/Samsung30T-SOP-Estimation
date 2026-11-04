from pathlib import Path
from zipfile import ZipFile
import csv, io, json, hashlib, time
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent
ZIP = ROOT / 'paper_package_lsad_20261013_r2.zip'
PREFIX = 'paper_package_lsad_20261013/'
OUT = ROOT / 'results'
CELLS = ['A1', 'A17', 'B']
TAUS = np.round(np.arange(0, .20001, .0025), 4)

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def scalar(x):
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x))

def load():
    source_manifest = json.loads((ROOT/'source_manifest.json').read_text())
    for name, info in source_manifest.items():
        assert hashlib.sha256((ROOT/'sources'/name).read_bytes()).hexdigest() == info['sha256']
    with ZipFile(ZIP) as z:
        assert z.testzip() is None
        for line in z.read(PREFIX+'MANIFEST.sha256').decode().splitlines():
            if not line or line.startswith('#'): continue
            h, name = line.split(None, 1)
            if PREFIX+name in z.namelist():
                assert hashlib.sha256(z.read(PREFIX+name)).hexdigest() == h
        rows = list(csv.DictReader(io.StringIO(z.read(PREFIX+'data/rows_tau5_with_delivery_posthoc.csv').decode('utf-8-sig'))))
        order = list(csv.DictReader(io.StringIO(z.read(PREFIX+'data/ad_trajectory.csv').decode('utf-8-sig'))))
        ls_curves = list(csv.DictReader(io.StringIO(z.read(PREFIX+'data/iso_tradeoff_posthoc.csv').decode('utf-8-sig'))))
    features = {r['row_id']: r for r in read_csv(ROOT/'sources/specaudit_v287/feats_fixed.csv')}
    assert len(features) == len(rows) == len(set(r['row_id'] for r in rows)) == 672
    assert set(features) == {r['row_id'] for r in rows} == {r['row_id'] for r in order}
    F = [features[r['row_id']] for r in rows]
    def arr(k): return np.array([float(r[k]) if r.get(k) not in ('', None) else np.nan for r in F])
    D = {k: arr(k) for k in F[0] if k not in ['row_id', 'sweep_id', 'last_sign']}
    D['direction'] = np.array([r['direction'] for r in rows])
    D['h'] = np.array([float(r['horizon_s']) for r in rows])
    ids = {r['row_id']: i for i,r in enumerate(rows)}
    orders = {c: np.array([ids[r['row_id']] for r in sorted([q for q in order if q['cell']==c], key=lambda q:int(q['order']))]) for c in CELLS}
    groups = {}
    for i,r in enumerate(rows):
        key=(r['evaluation_group'], r['direction'], float(r['temp_C']), float(r['horizon_s']))
        groups.setdefault(key, []).append(i)
    y=np.array([float(r['truth']) for r in rows]); assert np.all(y>0)
    return rows,D,y,orders,groups,ls_curves

def resistance_power(v, r, discharge):
    limit=np.where(discharge,2.55,4.15); imax=np.where(discharge,30.,15.)
    head=np.where(discharge,v-limit,limit-v)
    valid=np.isfinite(v)&np.isfinite(r)&(r>0)
    current=np.minimum(np.maximum(head,0)/np.where(valid,r,1),imax)
    voltage=v+np.where(discharge,-1,1)*current*r
    return np.where(valid,np.maximum(current*voltage,0),np.nan)

def fit_rc(t, recovery, load_duration):
    # Nonnegative linear least squares for every candidate tau; targets are past relaxation only.
    taus=np.logspace(-1,3,81)
    b=-np.expm1(-t[None,:]/taus[:,None])
    n=len(t); sx=b.sum(1); sxx=(b*b).sum(1); sy=recovery.sum(); sxy=b@recovery
    den=n*sxx-sx*sx
    amp=np.divide(n*sxy-sx*sy,den,out=np.zeros_like(den),where=den>1e-20)
    r0=(sy-amp*sx)/n
    candidates=np.stack([np.stack([r0,amp],1),np.stack([np.zeros_like(amp),np.maximum(sxy/sxx,0)],1),np.stack([np.full_like(amp,max(sy/n,0)),np.zeros_like(amp)],1)],1)
    pred=candidates[:,:,0,None]+candidates[:,:,1,None]*b[:,None,:]
    sse=((pred-recovery)**2).sum(2)
    sse[(candidates<0).any(2)]=np.inf
    i,j=np.unravel_index(np.argmin(sse),sse.shape)
    r0,a=candidates[i,j]; tau=taus[i]; r1=a/(-np.expm1(-load_duration/tau))
    return r0,r1,tau,a,float(np.sqrt(sse[i,j]/n))

def identify(D):
    n=len(D['h']); pars=np.full((n,5),np.nan); fitrmse=np.full(n,np.nan); times=np.array([1,2,5,10,30,60,120,240.])
    for i in range(n):
        y=np.array([D[f'Roff{int(t)}'][i] for t in times])
        ok=np.isfinite(y)&(y>0)&(times<=D['rest_s'][i]+1e-9)
        load=D['off_dur_s'][i]; current=D['I_off'][i]
        if ok.sum()<4 or not np.isfinite(load) or load<=0 or not np.isfinite(current) or abs(current)<1e-6:continue
        r0,r1,tau,a,rmse=fit_rc(times[ok],y[ok],load)
        z0=current*a*np.exp(-D['rest_s'][i]/tau)
        equilibrium=D['V_pre'][i]-z0
        if np.isfinite([r0,r1,tau,z0,equilibrium]).all() and equilibrium>0:
            pars[i]=[r0,r1,tau,z0,equilibrium];fitrmse[i]=rmse
    return pars,fitrmse

def simulate_feasible(P,pars,discharge,h,steps):
    r0,r1,tau,z,E=pars.T.copy();sign=np.where(discharge,-1.,1.)
    valid=np.ones(len(P),bool); dt=h/(steps-1); decay=np.exp(-dt/tau); half=np.exp(-.5*dt/tau)
    def electrical(pol):
        u=E+pol;disc=u*u+4*sign*r0*P
        v=.5*(u+np.sqrt(np.maximum(disc,0)))
        current=P/np.maximum(v,1e-12)
        good=(disc>=0)&(v>0)&np.where(discharge,(v>=2.55)&(current<=30),(v<=4.15)&(current<=15))
        return current,good
    for j in range(steps):
        current,good=electrical(z);valid &= good
        if j==steps-1:break
        mid=z*half+sign*r1*current*(1-half)
        current_mid,good_mid=electrical(mid);valid &= good_mid
        z=z*decay+sign*r1*current_mid*(1-decay)
    return valid

def rc_power(pars,discharge,h,steps=201):
    out=np.full(len(h),np.nan); mask=np.isfinite(pars).all(1)
    pp=pars[mask];dd=discharge[mask];hh=h[mask]
    low=np.zeros(len(pp)); high=np.where(dd,30*np.maximum(pp[:,4]+pp[:,3],0),15*4.15)*1.001
    zero=simulate_feasible(low,pp,dd,hh,steps)
    for _ in range(24):
        mid=(low+high)/2;good=simulate_feasible(mid,pp,dd,hh,steps)
        low=np.where(good,mid,low);high=np.where(good,high,mid)
    out[mask]=np.where(zero,low,0)
    return out

def independent_power(p,discharge,h):
    r0,r1,tau,z0,E=p; sign=-1 if discharge else 1
    def trial(power):
        def calc(z):
            disc=(E+z)**2+4*sign*r0*power
            v=.5*(E+z+np.sqrt(np.maximum(disc,0)))
            return v,power/np.maximum(v,1e-12),disc
        def rhs(t,z):
            _,current,_=calc(z)
            return (sign*r1*current-z)/tau
        sol=solve_ivp(rhs,(0,h),[z0],rtol=1e-9,atol=1e-11,dense_output=True)
        if not sol.success:return False
        v,i,disc=calc(sol.sol(np.linspace(0,h,1001))[0])
        return bool(np.all(disc>=0) and np.all((v>=2.55)&(i<=30) if discharge else (v<=4.15)&(i<=15)))
    lo=0.;hi=(30*max(E+z0,0) if discharge else 15*4.15)*1.001
    if not trial(0):return 0.
    for _ in range(24):
        mid=(lo+hi)/2
        if trial(mid):lo=mid
        else:hi=mid
    return lo

def calibrate(central,y,orders,tau):
    scales={};floor={};counts={}
    for c in CELLS:
        idx=np.concatenate([orders[d] for d in CELLS if d!=c]);idx=idx[np.isfinite(central[idx])&(central[idx]>0)]
        assert len(idx)>0
        grid=np.round(np.arange(.2,1.5001,.01),2)
        valid=(grid[:,None]*central[idx]>y[idx]+1e-9).mean(1)<=tau
        scales[c]=float(grid[valid][-1]) if valid.any() else .2
        floor[c]=not bool(valid.any());counts[c]=len(idx)
    return scales,floor,counts

def request(central,y,orders,scales,tau,adaptive):
    central=np.nan_to_num(central,nan=0);result=np.zeros(len(y))
    for c,idx in orders.items():
        logs=np.log(scales[c])
        for i in idx:
            result[i]=np.exp(logs)*central[i]
            if adaptive:logs=np.clip(logs+.05*(tau-float(result[i]/y[i]>1+1e-9)),np.log(.2),np.log(1.5))
    return result

def score(pred,y,groups):
    ratio=pred/y;exceed=ratio>1+1e-9;util=np.minimum(ratio,1);g=[]
    for key,idx in groups.items():
        rate=exceed[idx].mean();med=np.median(util[idx]);strict=not exceed[idx].any() and np.median(ratio[idx])>=.8
        g.append(dict(cell=key[0],direction=key[1],temp_C=key[2],horizon_s=key[3],n=len(idx),exceed=int(exceed[idx].sum()),int_rate=rate,util_q50=med,passed=bool(med>=.8 and rate<=.1),strict=bool(strict)))
    depths=ratio[exceed]-1
    return dict(n=len(y),exceed_rows=int(exceed.sum()),int_rate=float(exceed.mean()),util_q50=float(np.median(util)),closed_loop=sum(r['passed'] for r in g),original=sum(r['strict'] for r in g),depth_max=float(depths.max()) if len(depths) else 0,depth_p95=float(np.quantile(depths,.95)) if len(depths) else 0),g

def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=True)
    rows,D,y,orders,groups,ls_curves=load();discharge=D['direction']=='discharge'
    # Analytic/synthetic checks precede scoring of candidates.
    assert abs(resistance_power(np.array([3.7]),np.array([.05]),np.array([True]))[0]-2.55*23)<1e-10
    t=np.array([1,2,5,10,30,60,120,240.]);rec=.01+.02*(-np.expm1(-300/10))*(-np.expm1(-t/10))
    fit=fit_rc(t,rec,300);assert np.allclose(fit[:3],[.01,.02,10],rtol=1e-8,atol=1e-10)
    main_central=np.array([float(r['central']) for r in rows]); main_pred=request(main_central,y,orders,dict(A1=.76,A17=.82,B=.9),.05,True)
    assert np.max(np.abs(main_pred-np.array([float(r['pred']) for r in rows])))<1e-9
    mainscore,_=score(main_pred,y,groups);assert (mainscore['closed_loop'],mainscore['original'],mainscore['exceed_rows'])==(11,4,28)
    rh=np.where(D['h']==2,D['Roff2'],D['Roff10']);prh=np.where(D['h']==2,D['prev_R2'],D['prev_R10'])
    rest=resistance_power(D['V_pre'],rh,discharge)
    previous=resistance_power(D['V_pre'],prh,discharge);previous_fallback=~np.isfinite(previous);previous=np.where(previous_fallback,rest,previous)
    persistence=D['prev_P'].copy();persist_fallback=~np.isfinite(persistence)|(persistence<=0);persistence=np.where(persist_fallback,rest,persistence)
    pars,fitrmse=identify(D);rc201=rc_power(pars,discharge,D['h'],201);rc401=rc_power(pars,discharge,D['h'],401)
    maxdiff=float(np.nanmax(abs(rc201-rc401)));rc=rc401;points=401
    if maxdiff>.05:
        rc801=rc_power(pars,discharge,D['h'],801);maxdiff=float(np.nanmax(abs(rc401-rc801)));rc=rc801;points=801
    assert maxdiff<=.05, f'RC convergence failed: {maxdiff}'
    # Deterministic coverage of both directions, temperatures and sorted row identifiers.
    valid=np.flatnonzero(np.isfinite(rc));subset=valid[np.linspace(0,len(valid)-1,12,dtype=int)]
    checks=[]
    for i in subset:
        p=independent_power(pars[i],discharge[i],D['h'][i]);checks.append(dict(row_id=rows[i]['row_id'],candidate_W=float(rc[i]),independent_W=p,abs_difference_W=float(abs(rc[i]-p))))
    assert max(q['abs_difference_W'] for q in checks)<=.05
    rc_fallback=~np.isfinite(rc);rc=np.where(rc_fallback,rest,rc)
    candidates={'RINT_REST':rest,'RINT_PREVIOUS':previous,'RC1_RELAX':rc,'PERSISTENCE':persistence}
    primary=[];group_rows=[];request_rows=[];curves=[];scale_rows=[]
    for name,central in candidates.items():
        for tau in TAUS:
            scales,floor,counts=calibrate(central,y,orders,tau)
            for adaptive in [False,True]:
                pred=request(central,y,orders,scales,tau,adaptive);sm,G=score(pred,y,groups)
                curves.append(dict(method=name,policy='AD' if adaptive else 'fixed',tau=float(tau),**sm))
                if abs(tau-.05)<1e-10:
                    primary.append(dict(method=name,policy='AD' if adaptive else 'fixed',missing=int((~np.isfinite(central)).sum()),**sm))
                    for g in G:group_rows.append(dict(method=name,policy='AD' if adaptive else 'fixed',**g))
                    for i,r in enumerate(rows):request_rows.append(dict(method=name,policy='AD' if adaptive else 'fixed',row_id=r['row_id'],truth=y[i],central=float(central[i]),request=float(pred[i]),ratio=float(pred[i]/y[i])))
            if abs(tau-.05)<1e-10:
                for c in CELLS:scale_rows.append(dict(method=name,cell=c,scale=scales[c],calibration_rows=counts[c],floor_unsatisfied=floor[c]))
    # Ensure mesh differences do not change the pilot's primary decisions.
    coarse=np.where(np.isfinite(rc201),rc201,rest);fine=rc
    coarse_sc,_,_=calibrate(coarse,y,orders,.05);fine_sc,_,_=calibrate(fine,y,orders,.05)
    mesh_flags=[]
    for ad in [False,True]:
        a=request(coarse,y,orders,coarse_sc,.05,ad);b=request(fine,y,orders,fine_sc,.05,ad)
        sa,ga=score(a,y,groups);sb,gb=score(b,y,groups)
        mesh_flags.append(dict(policy='AD' if ad else 'fixed',exceedance_disagreements=int(np.count_nonzero((a/y>1+1e-9)!=(b/y>1+1e-9))),group_disagreements=sum(x['passed']!=q['passed'] or x['strict']!=q['strict'] for x,q in zip(ga,gb)),scales_equal=coarse_sc==fine_sc))
    assert all(q['exceedance_disagreements']==0 and q['group_disagreements']==0 and q['scales_equal'] for q in mesh_flags)
    iso=[]
    for name in candidates:
        for policy in ['fixed','AD']:
            eligible=[q for q in curves if q['method']==name and q['policy']==policy and q['exceed_rows']<=28]
            if not eligible:
                iso.append(dict(method=name,policy=policy,eligible_points=0,best_util_tau=None,best_util=None,exceed_rows=None,groups_at_best_util=None,maximum_groups=None));continue
            best=max(eligible,key=lambda q:q['util_q50'])
            iso.append(dict(method=name,policy=policy,eligible_points=len(eligible),best_util_tau=best['tau'],best_util=best['util_q50'],exceed_rows=best['exceed_rows'],groups_at_best_util=best['closed_loop'],maximum_groups=max(q['closed_loop'] for q in eligible)))
    fits=[dict(row_id=r['row_id'],R0=pars[i,0],R1=pars[i,1],tau_s=pars[i,2],polarization_initial_V=pars[i,3],equilibrium_V=pars[i,4],past_relaxation_rmse_ohm=fitrmse[i],fallback=bool(rc_fallback[i])) for i,r in enumerate(rows)]
    for name,data in [('primary.csv',primary),('groups.csv',group_rows),('requests.csv',request_rows),('curves.csv',curves),('scales.csv',scale_rows),('iso_budget.csv',iso),('rc_fits.csv',fits),('rc_independent_checks.csv',checks)]:write_csv(OUT/name,data)
    report=dict(zip_sha256=hashlib.sha256(ZIP.read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=672,groups=15,main_reproduction=mainscore,primary=primary,iso_budget=iso,fallback_counts=dict(RINT_PREVIOUS=int(previous_fallback.sum()),PERSISTENCE=int(persist_fallback.sum()),RC1_RELAX=int(rc_fallback.sum())),rc_solver=dict(final_points=points,max_mesh_power_difference_W=maxdiff,independent_max_difference_W=max(q['abs_difference_W'] for q in checks),mesh_decisions=mesh_flags),validation_boundary='Conditional numerical pilot on audited stored pre-query features. No raw-waveform feature re-extraction, no new-cell validation, no validation of RC model voltage predictions on measured held-out trajectories, no named-paper reproduction.',elapsed_seconds=time.perf_counter()-started)
    (OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=scalar),encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ['.csv','.json'] and p.name!='output_hashes.json'}
    (OUT/'output_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2,default=scalar))

if __name__=='__main__':main()
