"""Training-ready padded tensors; preserve every extracted short waveform."""
import numpy as np
import pandas as pd
from run_envelope import HERE


def load_data():
    root=HERE/'mendeley_inputs_v49';rows=pd.read_csv(root/'responses.csv')
    z=np.load(root/'arrays.npz');assert z['row_ids'].tolist()==rows.row_id.tolist()
    lengths=np.diff(z['offsets']);width=int(lengths.max());n=len(rows)
    times=np.empty((n,width),np.float32);current=times.copy();target=times.copy()
    hlens=np.diff(z['history_offsets']);hwidth=int(hlens.max())
    ht=np.empty((n,hwidth),np.float32);hi=ht.copy()
    for j,r in rows.iterrows():
        a,b=z['offsets'][j:j+2];length=b-a;sign=-1 if r.direction=='discharge' else 1
        times[j,:length]=z['time_s'][a:b];times[j,length:]=z['time_s'][b-1]
        current[j,:length]=z['current_A'][a:b];current[j,length:]=z['current_A'][b-1]
        target[j,:length]=sign*(z['voltage_V'][a:b]-r.pre_voltage_V);target[j,length:]=target[j,length-1]
        a,b=z['history_offsets'][j:j+2];length=b-a
        ht[j,:length]=z['history_time_s'][a:b];ht[j,length:]=z['history_time_s'][b-1]
        hi[j,:length]=z['history_current_A'][a:b];hi[j,length:]=z['history_current_A'][b-1]
    assert np.isfinite([times,current,target]).all() and (np.diff(times,axis=1)>=0).all()
    assert np.isfinite([ht,hi]).all() and (np.diff(ht,axis=1)>=0).all()
    return rows,z['features'],times,current,target,lengths,ht,hi
