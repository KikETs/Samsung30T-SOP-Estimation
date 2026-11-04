"""Separate causal state, observed action, and planned constant-action APIs."""
import csv
from collections import Counter
import numpy as np
from scipy.optimize import nnls
from sklearn.tree import DecisionTreeRegressor
from run_envelope import HERE
from waveform_action import TIME_CONSTANTS

LINEAR = ['endpoint_current_A'] + [f'conv_{k}_A' for k in range(5)]
STATE_BASE = ['pre_voltage_V', 'log_drive_r', 'drive_age'] + [f'charge_bin_{k}' for k in range(8)] + ['charge_band_s', 'charge_age']
PAST_FIELDS = ['pre_voltage_V', 'oriented_voltage_excursion_V', 'duration_s', 'age_s'] + LINEAR + ['endpoint_square_A2'] + [f'conv_square_{k}_A2' for k in range(5)]
STATE_NAMES = STATE_BASE + [f'past_{h}s_{k}' for h in (2,10) for k in PAST_FIELDS]
UNIT = 30.


def read_csv(path):
    return list(csv.DictReader(path.open()))


def action_matrix(rows):
    a = np.array([[float(r[k]) for k in LINEAR] + [float(r['endpoint_current_A'])**2] +
                  [float(r[f'conv_square_{k}_A2']) for k in range(5)] for r in rows])
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Nonnegative finite oriented action required')
    return a / np.array([UNIT]*6 + [UNIT**2]*6)


def make_state(row, old, histories, byid):
    """No access to current observed endpoint/current/target; prior targets allowed."""
    drive = float(old['drive_r'])
    end = float(old['drive_end_cycle'])
    values = [float(row['pre_voltage_V']), np.log(drive) if drive > 0 else np.nan,
              float(old['cycle'])-end if end >= 0 else np.nan]
    values += [float(old[f'charge_bin_{k}']) for k in range(8)]
    values += [float(old['charge_band_s']), float(old['charge_age'])]
    for horizon in (2,10):
        link = histories[row['row_id'], f"{row['direction']}_{horizon}s"]
        if link['available'] != 'True':
            values += [np.nan]*len(PAST_FIELDS)
            continue
        p = byid[link['source_row_id']]
        assert (p['cell'], p['source_file'], p['clock_segment']) == (row['cell'], row['source_file'], row['clock_segment'])
        assert int(p['endpoint_index']) < int(row['sample_index'])
        age = float(row['start_time_s'])-float(p['last_access_time_s'])
        assert 0 < age <= 7200 and abs(age-float(link['age_s'])) < 1e-7
        for key in PAST_FIELDS:
            values.append(age if key == 'age_s' else float(p['endpoint_current_A'])**2 if key == 'endpoint_square_A2' else float(p[key]))
    return values


def load_data():
    directory = HERE/'horizon_response_data_v1'
    rows = read_csv(directory/'responses.csv')
    byid = {r['row_id']:r for r in rows}
    old = {r['row_id']:r for r in read_csv(HERE/'response_data_v1/observed_pulses.csv')}
    histories = {(r['query_row_id'],r['channel']):r for r in read_csv(directory/'response_history_links.csv')}
    x = np.asarray([make_state(r, old[r['row_id']], histories, byid) for r in rows])
    assert x.shape == (57673,len(STATE_NAMES))
    return rows, x, action_matrix(rows), np.array([float(r['oriented_voltage_excursion_V']) for r in rows])


class Forward:
    def transform(self,x):
        missing = ~np.isfinite(x)
        return np.column_stack([np.where(missing,self.medians,x),missing.astype(float)])

    def fit(self,x,action,y,cells,ids,mean_current):
        assert len(x)==len(action)==len(y)==len(cells)==len(ids)
        self.fit_ids=list(ids); self.fit_cells=sorted(set(cells)); self.negative_y_count=int((y<0).sum())
        self.medians=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
        counts=Counter(cells);w=np.array([1/counts[c] for c in cells]);w*=len(w)/w.sum()
        self.tree=DecisionTreeRegressor(max_leaf_nodes=7,min_samples_leaf=100,random_state=17)
        self.tree.fit(self.transform(x),y/np.maximum(mean_current,.1),sample_weight=w)
        leaves=self.tree.apply(self.transform(x));self.coefficients={};self.kkt={}
        for leaf in np.unique(leaves):
            keep=leaves==leaf;a=action[keep];yy=y[keep];ww=w[keep]
            gram=a.T@(ww[:,None]*a);target=a.T@(ww*yy)
            ridge=max(.001*float(np.trace(gram)/12),1e-12)
            lower=np.linalg.cholesky(gram+ridge*np.eye(12))
            coef,_=nnls(lower.T,np.linalg.solve(lower,target),maxiter=2400)
            gradient=(gram+ridge*np.eye(12))@coef-target
            kkt=float(np.max(abs(np.where(coef>1e-10,gradient,np.minimum(gradient,0))))/max(1.,np.max(abs(target))))
            assert kkt<1e-7
            self.coefficients[int(leaf)]=coef;self.kkt[int(leaf)]=kkt
        self.cap=float(np.max(action[:,0])*UNIT*1.5)
        return self

    def coefficients_at(self,x):
        return np.array([self.coefficients[int(k)] for k in self.tree.apply(self.transform(x))])

    def observed(self,x,action):
        if action.shape!=(len(x),12) or not np.isfinite(action).all() or (action<0).any():
            raise ValueError('Invalid observed action')
        return np.sum(self.coefficients_at(x)*action,axis=1)

    def constant_coefficients(self,x,horizon):
        h=np.broadcast_to(np.asarray(horizon,float),(len(x),))
        if not np.isfinite(h).all() or (h<=0).any():raise ValueError('Invalid horizon')
        factors=np.column_stack([np.ones(len(x)), -np.expm1(-h[:,None]/np.array(TIME_CONSTANTS))])
        c=self.coefficients_at(x)
        return np.sum(c[:,:6]*factors,axis=1)/UNIT,np.sum(c[:,6:]*factors,axis=1)/UNIT**2

    def sop(self,x,horizon,headroom,margin=0.):
        a,b=self.constant_coefficients(x,horizon)
        h=np.maximum(np.asarray(headroom)-.020-margin,0.)
        den=a+np.sqrt(a*a+4*b*h)
        current=np.divide(2*h,den,out=np.zeros_like(h),where=den>1e-14)
        return np.minimum(current,self.cap)
