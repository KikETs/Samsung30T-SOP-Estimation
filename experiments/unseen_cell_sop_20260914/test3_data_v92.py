"""V92 data: UYPYDJ + Mendeley (V50/V53) plus RPCWBY Test#3 constant-power discharge waveforms as one more weighted cell."""
from collections import Counter
import numpy as np
import pandas as pd
from run_envelope import HERE
from nested_data_v53 import MENDELEY,validate_cells,upper_targets,local_neighbors,cap_from_training
from multitemperature_data_v50 import load_data as load_base,pad

TEST3='RPCWBY_TEST3'

def load_test3():
    root=HERE/'test3_training_v92';rows=pd.read_csv(root/'responses.csv');z=np.load(root/'arrays.npz');assert z['row_ids'].tolist()==rows.row_id.tolist()
    lengths=np.diff(z['offsets']);width=int(lengths.max());n=len(rows);times=np.empty((n,width),np.float32);current=times.copy();target=times.copy()
    hlens=np.diff(z['history_offsets']);hwidth=int(hlens.max());ht=np.empty((n,hwidth),np.float32);hi=ht.copy()
    for j,r in rows.iterrows():
        a,b=z['offsets'][j:j+2];length=b-a;sign=-1 if r.direction=='discharge' else 1
        times[j,:length]=z['time_s'][a:b];times[j,length:]=z['time_s'][b-1];current[j,:length]=z['current_A'][a:b];current[j,length:]=z['current_A'][b-1]
        target[j,:length]=sign*(z['voltage_V'][a:b]-r.pre_voltage_V);target[j,length:]=target[j,length-1]
        a,b=z['history_offsets'][j:j+2];length=b-a;ht[j,:length]=z['history_time_s'][a:b];ht[j,length:]=z['history_time_s'][b-1];hi[j,:length]=z['history_current_A'][a:b];hi[j,length:]=z['history_current_A'][b-1]
    assert all(np.isfinite(a).all() for a in (times,current,target,ht,hi)) and (np.diff(times,axis=1)>=0).all() and (np.diff(ht,axis=1)>=0).all()
    return rows,z['features'],times,current,target,lengths,ht,hi

def load_data():
    rows,x,t,i,y,lengths,ht,hi=load_base();tr,tx,tt,ti,ty,tl,tht,thi=load_test3()
    width=max(t.shape[1],tt.shape[1]);hwidth=max(ht.shape[1],tht.shape[1]);rows=rows+tr.to_dict('records')
    assert len({r['row_id'] for r in rows})==len(rows)==58825+356
    return (rows,np.vstack([x,tx]),np.vstack([pad(t,width),pad(tt,width)]),np.vstack([pad(i,width),pad(ti,width)]),np.vstack([pad(y,width),pad(ty,width)]),np.r_[lengths,tl],np.vstack([pad(ht,hwidth),pad(tht,hwidth)]),np.vstack([pad(hi,hwidth),pad(thi,hwidth)]))

def train_indices(rows,direction,fit_cells):
    chosen=set(validate_cells(fit_cells))|{MENDELEY,TEST3}
    return np.array([j for j,r in enumerate(rows) if r['direction']==direction and r['cell'] in chosen])

def fit_weights(rows,indices,fit_cells):
    """Equal share per cell (UYPYDJ cells, Mendeley, Test#3); Mendeley and Test#3 shares split equally over their temperatures."""
    chosen=validate_cells(fit_cells);cell_names=np.array([rows[j]['cell'] for j in indices]);counts=Counter(cell_names)
    assert set(counts)<=set(chosen)|{MENDELEY,TEST3} and set(chosen)<=set(counts) and MENDELEY in counts
    temps={c:Counter(int(rows[j]['nominal_temperature_C']) for j in indices if rows[j]['cell']==c) for c in (MENDELEY,TEST3) if c in counts}
    ncell=len(counts);result=[]
    for j in indices:
        r=rows[j]
        if r['cell'] in temps:den=ncell*len(temps[r['cell']])*temps[r['cell']][int(r['nominal_temperature_C'])]
        else:den=ncell*counts[r['cell']]
        result.append(len(indices)/den)
    return np.asarray(result,dtype=np.float32)
