"""Fixed same-cell 2RC comparison with actual current-history initial states."""
import argparse
import time
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import torch
from run_envelope import HERE,dump,digest,write_csv
from waveform_forward_v20 import load_data,read_csv,STATE_NAMES
from differentiable_ecm_v1 import StateScaler
from history_ecm_v1 import load_histories
from nonlinear_q_v29 import Encoder,response
from constant_state_v26 import ConstantChannelScaler

FIT_CELLS=['BOOST','BOOST_REST','CC','CC_CELL2']
VALIDATION='BOOST_NEGPULSE'
OUTER='BOOST_NEGPULSE_1S'


def load_pilot_data():
    rows,x,action,y=load_data()
    initial=read_csv(HERE/'ecm_initial_data_v1/initial_samples.csv')
    assert [r['row_id'] for r in rows]==[r['row_id'] for r in initial]
    pre=np.array([float(r['pre_current_A']) for r in initial]);gap=np.array([float(r['pre_gap_s']) for r in initial])
    sign=np.array([-1. if r['direction']=='discharge' else 1. for r in rows]);pre*=sign
    x=np.column_stack([x,pre,gap])
    with np.load(HERE/'horizon_response_data_v1/profiles.npz') as z:
        offsets=z['offsets'];tt=z['time_s'];ii=z['current_A'];vv=z['voltage_V']
        assert np.array_equal(z['row_ids'],[r['row_id'] for r in rows])
    lengths=np.diff(offsets)+1;width=int(lengths.max());n=len(rows)
    t=np.zeros((n,width),np.float32);i=t.copy();target=t.copy()
    for j,r in enumerate(rows):
        a,b=offsets[j:j+2];end=lengths[j]
        t[j,1:end]=tt[a:b]+gap[j];t[j,end:]=t[j,end-1]
        i[j,0]=pre[j];i[j,1:end]=sign[j]*ii[a:b];i[j,end:]=i[j,end-1]
        target[j,1:end]=sign[j]*(vv[a:b].astype(float)-float(r['pre_voltage_V']))
        target[j,end:]=target[j,end-1]
    assert np.isfinite(t).all() and np.isfinite(i).all() and np.isfinite(target).all()
    np.testing.assert_allclose(target[np.arange(n),lengths-1],y,atol=1e-7)
    assert (np.diff(t,axis=1)>=0).all()
    return rows,x,t,i,target,lengths,action


def main():
    args=argparse.Namespace(variant='signed_square')
    torch.set_num_threads(1);torch.manual_seed(17);np.random.seed(17)
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=HERE/'results_nonlinear_q_pilot_v29';out.mkdir(exist_ok=False)
    source_names=['PROTOCOL_NONLINEAR_Q_V29.md','nonlinear_q_v29.py','nonlinear_q_sop_v29.py','test_nonlinear_q_v29.py','run_nonlinear_q_pilot_v29.py','PROTOCOL_RC_ABLATION_V28.md','rc_ablation_v28.py','test_rc_ablation_v28.py','run_rc_ablation_pilot_v28.py','results_history_2rc_pilot_v1/predictions.csv','results_history_2rc_pilot_v1/audit.json','PROTOCOL_HISTORY_2RC_PILOT_V1.md','history_ecm_2rc_v1.py','test_history_ecm_2rc_v1.py','run_history_2rc_pilot_v1.py','results_history_ecm_pilot_v1/predictions.csv','results_history_ecm_pilot_v1/audit.json','results_history_ecm_pilot_v1/verification.json','PROTOCOL_HISTORY_ECM_PILOT_V1.md','history_ecm_v1.py','test_history_ecm_v1.py','run_history_ecm_pilot_v1.py','constant_state_v26.py','polarization_history_data_v1/histories.npz','polarization_history_data_v1/audit.json','PROTOCOL_ECM_PINN_PILOT_V1.md','differentiable_ecm_v1.py','test_differentiable_ecm_v1.py',
        'run_ecm_pinn_pilot_v1.py','build_ecm_initial_data_v1.py','ecm_initial_data_v1/initial_samples.csv',
        'ecm_initial_data_v1/audit.json','horizon_response_data_v1/profiles.npz','horizon_response_data_v1/responses.csv',
        'horizon_response_data_v1/response_history_links.csv','response_data_v1/observed_pulses.csv',
        'waveform_forward_v20.py','concave_action_data_v1/actions.npz']
    sources={str(HERE/n):digest(HERE/n) for n in source_names}
    dump(out/'sources.json',dict(source_hashes=sources,fit_cells=FIT_CELLS,validation_cell=VALIDATION,
        excluded_outer_cell=OUTER,variant=args.variant,state_features=STATE_NAMES+['oriented_pre_current_A','pre_gap_s'],epochs=80,seed=17,
        device=str(device),torch_version=torch.__version__,goal_complete=False))
    rows,x,t,i,y,lengths,action=load_pilot_data()
    history_t,history_i=load_histories(rows)
    cells=np.array([r['cell'] for r in rows]);directions=np.array([r['direction'] for r in rows])
    with np.load(HERE/'concave_action_data_v1/actions.npz') as z: action18=np.column_stack([action,z['log_action']])
    all_records=[];summaries=[];logs=[];model_hashes={}
    for direction in ['discharge','charge']:
        torch.manual_seed(17);rng=np.random.default_rng(17)
        train=np.flatnonzero((directions==direction)&np.isin(cells,FIT_CELLS))
        val=np.flatnonzero((directions==direction)&(cells==VALIDATION));selected=np.r_[train,val]
        assert not np.isin(cells[selected],OUTER).any()
        scaler=ConstantChannelScaler(StateScaler().fit(x[train]),x[train]);features=torch.tensor(scaler.transform(x[selected]),device=device)
        times=torch.tensor(t[selected],device=device);current=torch.tensor(i[selected],device=device)
        history_time=torch.tensor(history_t[selected],device=device);history_current=torch.tensor(history_i[selected],device=device)
        target=torch.tensor(y[selected],device=device);lens=torch.tensor(lengths[selected],device=device)
        counts=Counter(cells[train]);weight=torch.tensor([len(train)/(len(FIT_CELLS)*counts[c]) for c in cells[train]],device=device,dtype=torch.float32)
        model=Encoder(x.shape[1]).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        limit=np.array([float(rows[j]['observed_voltage_limit']) for j in selected]);headroom=torch.tensor(limit-x[selected,0] if direction=='charge' else x[selected,0]-limit,device=device,dtype=torch.float32)
        started=time.monotonic()
        for epoch in range(80):
            order=rng.permutation(len(train));total=0.
            for off in range(0,len(order),512):
                ix=torch.tensor(order[off:off+512],device=device)
                # Eight sampled internal observed times and one endpoint per row.
                inner=(torch.rand((len(ix),8),device=device)*(lens[ix,None]-2)).long()+1
                query=torch.cat([inner,lens[ix,None]-1],1)
                qt=torch.gather(times[ix],1,query);qi=torch.gather(current[ix],1,query)
                yy=torch.gather(target[ix],1,query)
                prediction=response(model(features[ix],history_time[ix],history_current[ix]),times[ix],current[ix],qt,qi)
                error=(prediction-yy).square()
                loss=(error.mean(1)*weight[ix]).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss.detach())*len(ix)
            logs.append(dict(direction=direction,epoch=epoch+1,train_sampled_mse_V2=total/len(train)))
            if (epoch+1)%10==0:print(direction,'epoch',epoch+1,'sampled_MSE',total/len(train),'seconds',round(time.monotonic()-started,1),flush=True)
        torch.save(model.cpu().state_dict(),out/f'{direction}.pt');model.to(device);model.eval()
        joblib.dump(scaler,out/f'{direction}_scaler.joblib')
        dump(out/f'{direction}_fit.json',dict(fit_cells=FIT_CELLS,fit_ids=[rows[j]['row_id'] for j in train],
            validation_cell=VALIDATION,validation_ids=[rows[j]['row_id'] for j in val],excluded_outer_cell=OUTER,
            negative_training_endpoints=int((y[train,lengths[train]-1]<0).sum()),epochs=80))
        basepath=HERE/'results_history_2rc_pilot_v1/predictions.csv'
        one_records={r['row_id']:r for r in read_csv(basepath)}
        one_fit=__import__('json').loads((HERE/'results_history_2rc_pilot_v1'/f'{direction}_fit.json').read_text())
        assert one_fit['fit_ids']==[rows[j]['row_id'] for j in train]
        basepred=np.array([float(one_records[rows[j]['row_id']]['predicted_endpoint_V']) for j in selected])
        records=[]
        with torch.no_grad():
            for off in range(0,len(selected),256):
                sl=slice(off,off+256);params=model(features[sl],history_time[sl],history_current[sl])
                prediction=response(params,times[sl],current[sl],times[sl],current[sl]).cpu().numpy()
                param=params.cpu().numpy()
                for jj,j in enumerate(selected[sl]):
                    end=lengths[j];error=prediction[jj,1:end]-y[j,1:end]
                    records.append(dict(row_id=rows[j]['row_id'],cell=cells[j],direction=direction,
                        split='train' if off+jj<len(train) else 'validation',n_observed_samples=int(end-1),
                        target_endpoint_V=float(y[j,end-1]),predicted_endpoint_V=float(prediction[jj,end-1]),
                        baseline_two_rc_endpoint_V=float(basepred[off+jj]),trajectory_mae_V=float(abs(error).mean()),
                        trajectory_mse_V2=float((error*error).mean()),trajectory_worst_under_V=float(max(0.,-error.min())),
                        R0_ohm=float(param[jj,0]),R1_ohm=float(param[jj,1]),tau1_s=float(param[jj,2]),
                        initial_p1_V=float(param[jj,3]),R2_ohm=float(param[jj,4]),tau2_s=float(param[jj,5]),initial_p2_V=float(param[jj,6]),history_duration_s=float(history_t[j,-1]),
                        assumed_initial_boundary_effect_V=float(.1*np.exp(-float(history_t[j,-1])/float(param[jj,2]))+.1*np.exp(-float(history_t[j,-1])/float(param[jj,5]))),local_drift_V_per_As=float(param[jj,7]),nonlinear_b_V_per_As2=float(param[jj,8])))
        for split in ['train','validation']:
            rr=[r for r in records if r['split']==split]
            err=np.array([r['predicted_endpoint_V']-r['target_endpoint_V'] for r in rr])
            berr=np.array([r['baseline_two_rc_endpoint_V']-r['target_endpoint_V'] for r in rr])
            summary=dict(direction=direction,split=split,n=len(rr),endpoint_mae_V=float(abs(err).mean()),
                endpoint_rmse_V=float(np.sqrt(np.mean(err*err))),baseline_two_rc_endpoint_mae_V=float(abs(berr).mean()),
                trajectory_row_mean_mae_V=float(np.mean([r['trajectory_mae_V'] for r in rr])),
                trajectory_worst_under_V=max(r['trajectory_worst_under_V'] for r in rr))
            summaries.append(summary);print(summary,flush=True)
        all_records.extend(records);model_hashes[direction]=dict(model=digest(out/f'{direction}.pt'),scaler=digest(out/f'{direction}_scaler.joblib'),baseline_two_rc_comparison=digest(basepath))
        write_csv(out/'predictions.csv',all_records);write_csv(out/'summary.csv',summaries);write_csv(out/'training_log.csv',logs)
    assert all(digest(Path(p))==h for p,h in sources.items())
    dump(out/'audit.json',dict(summaries=summaries,rows=len(all_records),total_original_rows=len(rows),
        excluded_outer_rows=int((cells==OUTER).sum()),all_selected_rows_retained=len(all_records)==int((cells!=OUTER).sum()),
        no_outer_cell_evaluation=True,no_epoch_selection_on_validation=True,source_hashes_unchanged=True,
        model_hashes=model_hashes,output_hashes={n:digest(out/n) for n in ['predictions.csv','summary.csv','training_log.csv']},
        variant=args.variant,sop_evaluated=False,goal_complete=False))


if __name__=='__main__':main()
