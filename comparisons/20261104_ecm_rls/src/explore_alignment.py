"""Exploratory look at candidate row-index conventions before the acceptance criteria are fixed (no model, no score).

Prints, for each convention, the pulse duration, power flatness, |V*I| vs truth_power_W and the end voltage vs limit.
"""
import csv
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
CACHE = EXP / 'cache'


def cache_path(member):
    return CACHE / (member.replace('/', '__').replace('#', '_') + '.npz')


def main():
    with open(EXP / 'inputs/bench_rows.csv', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    data = {}
    out = {c: [] for c in ('C0_line0', 'C1_line1', 'C2_datapoint')}
    for r in rows:
        m = r['member']
        if m not in data:
            d = np.load(cache_path(m))
            data[m] = {k: d[k] for k in d.files}
        D = data[m]
        n = len(D['t'])
        q, L = int(r['q_row']), int(r['last_row'])
        dis = r['direction'] == 'discharge'
        lim = 2.55 if dis else 4.15
        truth = float(r['truth_power_W'])
        h = float(r['horizon_s'])
        for conv in out:
            if conv == 'C0_line0':
                iq, iL = q, L
            elif conv == 'C1_line1':
                iq, iL = q - 1, L - 1
            else:
                a = np.flatnonzero(D['dp'] == q)
                b = np.flatnonzero(D['dp'] == L)
                if len(a) != 1 or len(b) != 1:
                    out[conv].append(dict(row_id=r['row_id'], ok_index=False))
                    continue
                iq, iL = int(a[0]), int(b[0])
            if not (1 <= iq <= iL < n - 1):
                out[conv].append(dict(row_id=r['row_id'], ok_index=False))
                continue
            I = D['I'][iq:iL + 1]
            V = D['V'][iq:iL + 1]
            t = D['t'][iq:iL + 1]
            P = np.abs(V * I)
            sgn_ok = bool(np.all(I < -0.02) if dis else np.all(I > 0.02))
            step = D['step'][iq:iL + 1]
            cyc = D['cyc'][iq:iL + 1]
            one_step = bool(np.all(step == step[0]) and np.all(cyc == cyc[0]))
            before_diff = bool(D['step'][iq - 1] != step[0] or D['cyc'][iq - 1] != cyc[0] or abs(D['I'][iq - 1]) < 0.02)
            after_diff = bool(D['step'][iL + 1] != step[0] or D['cyc'][iL + 1] != cyc[0])
            out[conv].append(dict(row_id=r['row_id'], ok_index=True, sgn_ok=sgn_ok, one_step=one_step,
                                  before_diff=before_diff, after_diff=after_diff, n=int(iL - iq + 1),
                                  dur_last=float(t[-1] - t[0]), dur_next=float(D['t'][iL + 1] - t[0]), h=h,
                                  p_rel_err=float(abs(P[-1] - truth) / truth), p_abs_err=float(abs(P[-1] - truth)),
                                  v_err=float(abs(V[-1] - lim)),
                                  flat_after1=float(np.max(np.abs(P[1:] / P[-1] - 1))) if len(P) > 1 else np.nan,
                                  flat_mid=float(np.max(np.abs(P[len(P) // 2:] / P[-1] - 1))),
                                  I_prev=float(D['I'][iq - 1])))
    summary = {}
    for conv, lst in out.items():
        ok = [x for x in lst if x['ok_index']]
        s = dict(n_index_ok=len(ok))
        if ok:
            def a(k):
                return np.array([x[k] for x in ok], dtype=float)
            p_ok = (a('p_rel_err') <= 0.005) | (a('p_abs_err') <= 0.05)
            s.update(sgn_ok=int(a('sgn_ok').sum()), one_step=int(a('one_step').sum()),
                     before_diff=int(a('before_diff').sum()), after_diff=int(a('after_diff').sum()),
                     power_ok=int(p_ok.sum()), v_ok=int((a('v_err') <= 0.04).sum()),
                     p_rel_err_max=float(a('p_rel_err').max()), v_err_max=float(a('v_err').max()))
            for h in (10.0, 2.0):
                hh = a('h') == h
                if hh.any():
                    s[f'h{int(h)}_dur_last_quantiles'] = np.quantile(a('dur_last')[hh], [0, .01, .5, .99, 1]).round(4).tolist()
                    s[f'h{int(h)}_dur_next_quantiles'] = np.quantile(a('dur_next')[hh], [0, .01, .5, .99, 1]).round(4).tolist()
                    s[f'h{int(h)}_n_quantiles'] = np.quantile(a('n')[hh], [0, .01, .5, .99, 1]).tolist()
            s['flat_after1_quantiles'] = np.quantile(a('flat_after1'), [0, .5, .9, .99, 1]).round(5).tolist()
            s['flat_mid_quantiles'] = np.quantile(a('flat_mid'), [0, .5, .9, .99, 1]).round(5).tolist()
            s['abs_I_prev_quantiles'] = np.quantile(np.abs(a('I_prev')), [0, .5, .9, .99, 1]).round(4).tolist()
        summary[conv] = s
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
