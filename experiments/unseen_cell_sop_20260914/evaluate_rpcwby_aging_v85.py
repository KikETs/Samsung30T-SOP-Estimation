"""V85: frozen V57 raw fits and V61/V62 UYPYDJ-only calibrations evaluated on RPCWBY Test#1/#2 measured 10 s constant-power SOP.

No RPCWBY data enters training or calibration. Commands are constant power for 10 s with the test limits
(discharge 2.55 V floor / 30 A, charge 4.15 V ceiling / 15 A) plus the fixed 0.02 V guard, history boundary and,
for calibrated methods, the member margin. Truth is the measured end power of the final applied pulse (V84)."""
import argparse,json,time
from multiprocessing import Pool
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from run_envelope import HERE,digest,dump
from nested_data_v53 import load_data
from bounded_reserve_v57 import Encoder
from state_neighbors_v61 import state_features,state_margins
from state_support_v62 import calibration_radii,support_counts
from voltage_local_calibration_v56 import local_margins
from constant_power_v85 import sop

OUT=HERE/'rpcwby_aging_evaluation_v85';INP=HERE/'rpcwby_aging_inputs_v85';FITS=HERE/'results_bounded_reserve_v57';CACHE=HERE/'bounded_reserve_cache_v57'
LIMITS=dict(discharge=dict(cap=30.,limit=2.55,sign=-1),charge=dict(cap=15.,limit=4.15,sign=1))
GUARD=.02

def params_for(fit_id,X,hts,his,offsets,rows_idx):
    model=Encoder(74).eval();model.load_state_dict(torch.load(FITS/fit_id/'model.pt',map_location='cpu',weights_only=True));scaler=joblib.load(FITS/fit_id/'scaler.joblib')
    out=np.empty((len(rows_idx),10));unsupported=np.empty(len(rows_idx),int)
    with torch.no_grad():
        for k,j in enumerate(rows_idx):
            a,b=offsets[j],offsets[j+1];z=scaler.transform(X[j][None])
            p=model(torch.tensor(z),torch.tensor(hts[a:b][None],dtype=torch.float32),torch.tensor(his[a:b][None],dtype=torch.float32))
            out[k]=p.numpy()[0].astype(float);unsupported[k]=int(scaler.unsupported_channels(X[j][None]).sum())
    return out,scaler,unsupported

def command_task(args):
    p,vpre,pre,gap,direction,cap,limit=args
    r=sop(np.asarray(p),float(vpre),float(pre),float(gap),10.,direction,cap,limit);ev=r['evaluation']
    return dict(power_W=r['power_W'],zero_infeasible=bool(r['zero_infeasible']),status=ev.get('status'),extreme_voltage_V=ev.get('min_voltage_V',ev.get('max_voltage_V')),max_current_A=ev.get('max_current_A'),bracket_width_W=r['bracket_width_W'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folds',default='all');parser.add_argument('--temps',default='25,10');parser.add_argument('--workers',type=int,default=14);args=parser.parse_args()
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True);started=time.monotonic()
    inp=json.loads((INP/'audit.json').read_text())
    for n,h in inp['output_hashes'].items():assert digest(INP/n)==h,n
    R=pd.read_csv(INP/'inputs.csv');Z=np.load(INP/'inputs.npz');X=Z['features'];hts=Z['history_time_s'];his=Z['history_current_A'];offsets=Z['history_offsets']
    temps=[int(t) for t in args.temps.split(',')];R=R[R.temp_block_C.isin(temps)&R.table_linked]
    folds=json.loads((HERE/'nested_preparation_v53/folds.json').read_text());fits={r['fit_id']:r for r in json.loads((HERE/'nested_preparation_v53/fits.json').read_text())}
    if args.folds!='all':folds=[f for f in folds if f['outer_cell'] in args.folds.split(',')]
    rows,x,t,i,y,lengths,ht,hi=load_data()
    predictions=[];ledger=[];sources={str(INP/'audit.json'):digest(INP/'audit.json')}
    for fold in folds:
        d=fold['direction'];lim=LIMITS[d];tag=d+'_'+fold['outer_cell'];q=R[R.direction==d];idx=q.row_id.to_numpy();vpre=X[idx,0];pre=X[idx,72];gap=X[idx,73]
        boundary_of=lambda p,j:.1*np.exp(-hts[offsets[j+1]-1]/p[2])+.1*np.exp(-hts[offsets[j+1]-1]/p[5])
        members=[dict(fit_id=fold['raw_fit_id'],calibration_cell='',role='raw')]+[dict(m,role='member') for m in fold['members']]
        per_member={}
        for m in members:
            fid=m['fit_id'];assert fold['outer_cell'] not in fits[fid]['uypydj_cells']
            for n in ['model.pt','scaler.joblib','audit.json']:sources[str(FITS/fid/n)]=digest(FITS/fid/n)
            p,scaler,unsupported=params_for(fid,X,hts,his,offsets,idx);bound=np.array([boundary_of(p[k],j) for k,j in enumerate(idx)])
            margins={'none':np.zeros(len(idx))}
            if m['role']=='member':
                cell=m['calibration_cell'];assert cell not in fits[fid]['uypydj_cells']
                root=CACHE/fid;ca=json.loads((root/'audit.json').read_text())
                for n,h in ca['output_hashes'].items():assert digest(root/n)==h,n
                sources[str(root/'audit.json')]=digest(root/'audit.json')
                with np.load(root/'heldout.npz') as z:cidx=z['indices'];traj=z['trajectory']
                pos={int(j):k for k,j in enumerate(cidx)}
                ci=np.array([j for j,r in enumerate(rows) if r['cell']==cell and r['direction']==d and float(r['requested_horizon_s'])==10.])
                cv=(np.arange(t.shape[1])[None,:]<lengths[ci,None])&(np.arange(t.shape[1])[None,:]>0)
                residual=np.where(cv,y[ci]-traj[[pos[int(j)] for j in ci]],-np.inf).max(axis=1)
                cal_states=state_features(scaler,x[ci]);query_states=state_features(scaler,X[idx])
                v61,count,selected,winner,reason,radius=state_margins(x[ci,0],residual,cal_states,vpre,query_states)
                radii,_=calibration_radii(x[ci,0],cal_states);covers,_=support_counts(x[ci,0],cal_states,radii,vpre,query_states)
                baseline,_,_,_=local_margins(x[ci,0],residual,vpre)
                use_state=(reason=='state_neighbors')&(covers>0);v62=np.where(use_state,v61,baseline)
                margins['v61']=v61;margins['v62']=v62
                for k,j in enumerate(idx):ledger.append(dict(fold=tag,fit_id=fid,calibration_cell=cell,row_id=int(j),v61_margin_V=float(v61[k]),v62_margin_V=float(v62[k]),neighbor_reason=str(reason[k]),supporting_centers=int(covers[k]),nearby_rows=int(count[k]),global_margin_V=float(max(0.,residual.max()))))
            tasks={};results={}
            for name,mg in margins.items():
                for k,j in enumerate(idx):
                    # discharge floor = 2.55+guard+boundary+margin ; charge ceiling = 4.15-guard-boundary-margin
                    limit=lim['limit']+GUARD+bound[k]+mg[k] if d=='discharge' else lim['limit']-GUARD-bound[k]-mg[k]
                    key=(k,round(float(limit),9))
                    tasks.setdefault(key,(p[k],vpre[k],pre[k],gap[k],d,lim['cap'],float(limit)))
            keys=list(tasks)
            with Pool(args.workers) as pool:outs=pool.map(command_task,[tasks[k] for k in keys],chunksize=4)
            results=dict(zip(keys,outs))
            per_member[fid]=dict(p=p,bound=bound,unsupported=unsupported,margins=margins,results=results,role=m['role'],cell=m['calibration_cell'])
            print(tag,fid,m['role'],'commands',len(keys),'elapsed',round(time.monotonic()-started),flush=True)
        raw=per_member[fold['raw_fit_id']]
        for k,j in enumerate(idx):
            r=R[R.row_id==j].iloc[0];rec=dict(fold=tag,outer_cell=fold['outer_cell'],row_id=int(j),pulse_id=int(r.pulse_id),sheet=r.sheet,cycle=r.cycle,temp_block_C=int(r.temp_block_C),direction=d,soc_point=r.soc_point,
                truth_power_W=float(r.truth_power_W),truth_end_current_A=float(r.truth_end_current_A),truth_end_voltage_V=float(r.truth_end_voltage_V),converged=bool(r.converged),finite_features=int(r.finite_features),
                raw_fit_id=fold['raw_fit_id'],raw_unsupported_channels=int(raw['unsupported'][k]),raw_boundary_V=float(raw['bound'][k]))
            rr=raw['results'][(k,round(float(lim['limit']+GUARD+raw['bound'][k] if d=='discharge' else lim['limit']-GUARD-raw['bound'][k]),9))]
            rec.update(raw_power_W=rr['power_W'],raw_zero_infeasible=rr['zero_infeasible'],raw_status=rr['status'],raw_extreme_voltage_V=rr['extreme_voltage_V'],raw_max_current_A=rr['max_current_A'])
            for method in ['v61','v62']:
                vals=[];mgs=[]
                for fid,pm in per_member.items():
                    if pm['role']!='member':continue
                    mg=pm['margins'][method][k];limit=lim['limit']+GUARD+pm['bound'][k]+mg if d=='discharge' else lim['limit']-GUARD-pm['bound'][k]-mg
                    vals.append(pm['results'][(k,round(float(limit),9))]['power_W']);mgs.append(mg)
                rec[f'{method}_power_W']=float(min(vals));rec[f'{method}_max_margin_V']=float(max(mgs));rec[f'{method}_min_margin_V']=float(min(mgs))
            predictions.append(rec)
    P=pd.DataFrame(predictions)
    for method in ['raw','v61','v62']:
        P[f'{method}_over_W']=P[f'{method}_power_W']-P.truth_power_W;P[f'{method}_usable_pct']=100*P[f'{method}_power_W']/P.truth_power_W
    P.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(ledger).to_csv(OUT/'calibration_ledger.csv',index=False)
    summaries=[]
    def score(g,method,tol):
        over=g[f'{method}_over_W'];band=np.where(g.direction=='charge',.5*g.truth_end_voltage_V,0.)
        protocol_band=.04*g.truth_end_current_A+.1*g.truth_end_voltage_V
        return dict(n=len(g),exceed_strict=int((over>1e-9).sum()),exceed_tolerance=int((over>band+1e-9).sum()),exceed_beyond_protocol_band=int((over>protocol_band+1e-9).sum()),worst_over_W=float(max(0.,over.max())),
                    median_usable_pct=float(g[f'{method}_usable_pct'].median()),p05_usable_pct=float(g[f'{method}_usable_pct'].quantile(.05)),min_usable_pct=float(g[f'{method}_usable_pct'].min()),zero_outputs=int((g[f'{method}_power_W']<=1e-9).sum()),
                    mae_W=float(over.abs().mean()),bias_W=float(over.mean()))
    for method in ['raw','v61','v62']:
        for (fold,sheet,temp,d),g in P.groupby(['fold','sheet','temp_block_C','direction']):summaries.append(dict(method=method,fold=fold,sheet=sheet,temp_block_C=temp,direction=d,**score(g,method,None)))
        for (sheet,temp,d),g in P.groupby(['sheet','temp_block_C','direction']):summaries.append(dict(method=method,fold='all_folds_pooled',sheet=sheet,temp_block_C=temp,direction=d,**score(g,method,None)))
    S=pd.DataFrame(summaries);S.to_csv(OUT/'summary.csv',index=False)
    cyc=[]
    for method in ['raw','v61','v62']:
        for (sheet,temp,d,c),g in P.groupby(['sheet','temp_block_C','direction','cycle']):cyc.append(dict(method=method,sheet=sheet,temp_block_C=temp,direction=d,cycle=c,**score(g,method,None)))
    pd.DataFrame(cyc).to_csv(OUT/'cycle_summary.csv',index=False)
    names=['evaluate_rpcwby_aging_v85.py','constant_power_v85.py','constant_power_v47.py','state_neighbors_v61.py','state_support_v62.py','voltage_local_calibration_v56.py','bounded_reserve_v57.py','nested_preparation_v53/folds.json','nested_preparation_v53/fits.json','PROTOCOL_RPCWBY_AGING_EVAL_V85.md']
    sources.update({str(HERE/n):digest(HERE/n) for n in names})
    gate={}
    for method in ['raw','v61','v62']:
        pooled=S[(S.method==method)&(S.fold=='all_folds_pooled')]
        gate[method]=dict(groups=len(pooled),exceed_strict=int(pooled.exceed_strict.sum()),exceed_tolerance=int(pooled.exceed_tolerance.sum()),groups_at_80=int((pooled.median_usable_pct>=80).sum()),
                          groups_passing_both=int(((pooled.median_usable_pct>=80)&(pooled.exceed_tolerance==0)).sum()),min_median_usable_pct=float(pooled.median_usable_pct.min()))
    dump(OUT/'verification.json',dict(passed=True,rows=int(P.row_id.nunique()),predictions=len(P),folds=sorted(P.fold.unique()),temps=temps,gate=gate,truth='RPCWBY measured final constant-power pulse end power',
        surrogate_lin4_used=False,rpcwby_used_for_training_or_calibration=False,elapsed_s=time.monotonic()-started,goal_complete=False))
    dump(OUT/'audit.json',dict(source_hashes=sources,output_hashes={f.name:digest(f) for f in OUT.iterdir() if f.is_file()},goal_complete=False))
    print(json.dumps(gate,indent=1));print(S[S.fold=='all_folds_pooled'].to_string(index=False))
if __name__=='__main__':main()
