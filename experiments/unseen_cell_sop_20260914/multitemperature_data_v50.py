"""Append Mendeley as one physical cell, without changing original row data."""
from collections import Counter
import numpy as np
from saturating_current_v39 import load_pilot_data
from history_ecm_v1 import load_histories
from load_mendeley_v49 import load_data as load_mendeley

FIT_UYPYDJ=['BOOST','BOOST_REST','CC','CC_CELL2']
FIT_CELLS=FIT_UYPYDJ+['MENDELEY_SN001']
VALIDATION='BOOST_NEGPULSE'
OUTER='BOOST_NEGPULSE_1S'


def pad(a,width):
    return np.pad(a,((0,0),(0,width-a.shape[1])),mode='edge')


def load_data():
    rows,x,t,i,y,lengths,_=load_pilot_data();ht,hi=load_histories(rows)
    mr,mx,mt,mi,my,ml,mht,mhi=load_mendeley()
    width=max(t.shape[1],mt.shape[1]);hwidth=max(ht.shape[1],mht.shape[1])
    nold=len(rows);rows=rows+mr.to_dict('records')
    assert len({r['row_id'] for r in rows})==len(rows)==58825 and nold==57673
    return (rows,np.vstack([x,mx]),np.vstack([pad(t,width),pad(mt,width)]),
            np.vstack([pad(i,width),pad(mi,width)]),np.vstack([pad(y,width),pad(my,width)]),
            np.r_[lengths,ml],np.vstack([pad(ht,hwidth),pad(mht,hwidth)]),
            np.vstack([pad(hi,hwidth),pad(mhi,hwidth)]))


def fit_weights(rows,indices):
    cells=[rows[j]['cell'] for j in indices];counts=Counter(cells)
    assert set(counts)==set(FIT_CELLS)
    temperatures=Counter(int(rows[j]['nominal_temperature_C']) for j in indices if rows[j]['cell']=='MENDELEY_SN001')
    assert set(temperatures)=={-20,-10,0,10,25,40}
    result=[]
    for j in indices:
        r=rows[j]
        den=5*6*temperatures[int(r['nominal_temperature_C'])] if r['cell']=='MENDELEY_SN001' else 5*counts[r['cell']]
        result.append(len(indices)/den)
    weights=np.array(result,np.float32)
    for c in FIT_CELLS:
        np.testing.assert_allclose(weights[np.array(cells)==c].sum()/len(indices),.2,atol=1e-7)
    return weights
