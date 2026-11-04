"""V322: compare the C engine's per-query output (host or board) with the Python references.
usage: python compare_host.py PREFIX [label]   (PREFIX_rows.csv, PREFIX_requests.csv from host_v322 or the board run)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent; HERE = OUT.parent
GK = ['evaluation_group', 'direction', 'temp_C', 'horizon_s']


def cmp(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.isnan(a), np.isnan(b); both = ~na & ~nb
    inf = both & (np.isinf(a) | np.isinf(b)); fin = both & ~inf
    d = np.abs(a[fin] - b[fin]); r = d / np.maximum(np.abs(b[fin]), 1e-300)
    return dict(nan_mismatch=int((na != nb).sum()), inf_mismatch=int((inf & (a != b)).sum()), max_abs=float(d.max()) if fin.any() else 0.0,
                max_rel=float(r.max()) if fin.any() else 0.0, n=int(fin.sum()))


def main():
    pre = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else pre
    C = pd.read_csv(f'{pre}_rows.csv'); Rq = pd.read_csv(f'{pre}_requests.csv')
    Q = pd.read_csv(OUT / 'queries_v322.csv', float_precision='round_trip')
    F = pd.read_csv(OUT / 'feats_fixed.csv', float_precision='round_trip').set_index('row_id')
    REF = pd.read_csv(OUT / 'ref_v322.csv', float_precision='round_trip').set_index('row_id')
    B = pd.read_csv(HERE / 'leakfree_v272/src/pybamm/out/rpcwby_bench/rows.csv').set_index('row_id')
    C = C.merge(Q[['query_id', 'row_id']], on='query_id').set_index('row_id')
    F = F.reindex(C.index); REF = REF.reindex(C.index); h0 = B.h0_row.reindex(C.index).to_numpy(float)
    res = {}
    names = ['V_pre', 'rest_s', 'I_end', 'last_dur_s', 'V_end', 'V_rest0', 'dt_rest0', 'slope_mVs', 'Ah_q', 'I_off', 'off_dur_s'] + \
            [f'Roff{t}' for t in (1, 2, 5, 10, 30, 60, 120, 240)] + ['sq_R0', 'sq_A', 'Rstep1', 'Rstep2', 'Rstep10', 'prev_gap_s']
    for p in ('prev_', 'prev2_'):
        names += [p + n for n in ('P', 'Vpre', 'I_last', 'V_last', 'dt_last', 'R1', 'R2', 'R10', 'Ah', 'I_off', 'off_dur_s', 'sq_R0', 'sq_A', 'Rstep1', 'Rstep2', 'Rstep10', 'rest_s')]
        names += [p + f'Roff{t}' for t in (1, 2, 5, 10, 30, 60, 120, 240)]
    feat = {}
    for n in names:
        if n not in F.columns:
            feat[n] = 'missing in Python'; continue
        feat[n] = cmp(C[n], F[n])
    feat['T_pre(vs T_pre_raw)'] = cmp(C.T_pre, F.T_pre_raw)
    for p in ('prev_', 'prev2_'):
        ck = np.where(C[p + 'valid'] == 1, C[p + 'k0'] + h0, np.nan)
        feat[p + 'k0_abs'] = cmp(ck, F[p + 'k0_abs'])
    py_sign = np.where(F.last_sign == 'chg', 1, np.where(F.last_sign == 'dis', -1, 0)); feat['last_sign'] = int((C.last_sign.to_numpy() != py_sign).sum())
    feat['valid'] = int(((C.valid == 1).to_numpy() != F.V_pre.notna().to_numpy()).sum())
    res['features'] = feat
    res['diagnostics'] = dict(a_zero=int(C.a_zero.sum()), rb_trunc=int(C.rb_trunc.sum()))
    regmap = {i * 3 + j: b + s for i, b in enumerate(['dis<=0.1', 'dis0.15-0.3', 'dis>=0.4', 'chg<=0.7', 'chg>=0.8']) for j, s in enumerate(['', '|first', '|firstcv'])}
    res['region_mismatch'] = int((C.reg.map(regmap).to_numpy() != REF.reg.to_numpy()).sum())
    res['predictions'] = {n: cmp(C['p_' + n], REF[n]) for n in ('prev', 'L4', 'M6', 'DN6', 'G1')}
    res['centre'] = cmp(C.centre, REF.central)
    Rq = Rq.merge(Q[['query_id', 'row_id']], on='query_id').set_index('row_id').reindex(C.index)
    res['scale'] = cmp(Rq.scale, REF.scale); res['request'] = cmp(Rq.request, REF.request)
    res['event_mismatch'] = int((Rq.event.to_numpy(bool) != REF.intervened.to_numpy(bool)).sum())
    G = REF.assign(req=Rq.request.to_numpy())
    r = G.req / G.truth; G['it'] = r > 1 + 1e-9; G['u'] = np.minimum(r, 1.0); G['rr'] = r
    A = G.groupby(GK).agg(n=('it', 'size'), k=('it', 'sum'), u=('u', 'median'), rq=('rr', 'median'))
    res['groups_closed_loop'] = int(((A.u >= 0.80) & (A.k / A.n <= 0.10)).sum()); res['groups_strict'] = int(((A.k == 0) & (A.rq >= 0.80)).sum())
    res['int_rate'] = float(G.it.mean()); res['util_q50'] = float(G.u.median())
    worst = sorted(((k, v['max_rel'], v['nan_mismatch']) for k, v in feat.items() if isinstance(v, dict)), key=lambda x: -(x[1] + 1e6 * x[2]))[:8]
    print(label, json.dumps({k: v for k, v in res.items() if k != 'features'}, indent=1))
    print('worst features (name, max_rel, nan_mismatch):', worst)
    print('feature scalars:', {k: v for k, v in feat.items() if not isinstance(v, dict)})
    json.dump(res, open(f'{pre}_compare.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
