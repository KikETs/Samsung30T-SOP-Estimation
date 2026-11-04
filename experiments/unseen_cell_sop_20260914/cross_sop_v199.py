"""V199: SOP-dataset cross configuration (PROTOCOL_CROSS_SOP_V199.md). For each evaluated SOP-dataset cell, the M3s transfer is fitted on the
aging-dataset SOC-step pairs plus the pulse pairs of the other SOP-dataset cells; margins come from leave-one-training-cell-out inside the fold."""
import json
from multiprocessing import Pool
import numpy as np, pandas as pd, torch
from run_envelope import HERE, dump
from relax_state_v36 import STATE_NAMES
from test3_data_v92 import load_test3
import bench_m3s_v196 as S
import score_v196 as S196

OUT = HERE / 'cross_sop_v199'; NM = list(STATE_NAMES); TC = NM.index('pre_temperature_5s_C')
LIMIT = S196.LIMIT; CAP = S196.CAP
FOLDS = {'A1': ['Test#2', 'B'], 'A17': ['Test#1', 'B'], 'B': ['Test#1', 'Test#2']}
CELL = {'A1': 'Test#1', 'A17': 'Test#2', 'B': 'B'}
MIN_ROWS = 5


def anchor(x, h):
    hh = 2 if h == 2. else 10
    return x[NM.index(f'past_{hh}s_pre_voltage_V')], x[NM.index(f'past_{hh}s_oriented_voltage_excursion_V')], abs(x[NM.index(f'past_{hh}s_endpoint_current_A')])


def with_features(P):
    P = P[(P.Iq > .5) & (P.Ia > .5) & (P.exq > 1e-4) & (P.exa > 1e-4)].copy()
    P['f1'] = np.where(P.d == 'discharge', 1 / (P.Vq - 2.5) - 1 / (P.Va - 2.5), 1 / (4.25 - P.Vq) - 1 / (4.25 - P.Va))
    P['f2'] = np.log(P.Iq / P.Ia); P['f3'] = P.f1 * (1000 / (P['T'] + 273.15) - 1000 / 298.15)
    P['target'] = np.log(P.exq / P.Iq) - np.log(P.exa / P.Ia); P['logRa'] = np.log(P.exa / P.Ia)
    return P


def sop_pairs():
    R = pd.read_csv(HERE / 'rpcwby_waveform_inputs_v87/inputs.csv'); Z = np.load(HERE / 'rpcwby_waveform_inputs_v87/inputs.npz')
    X = Z['features']; wv = Z['wave_oriented_excursion_V']; wi = Z['wave_oriented_current_A']; wo = Z['wave_offsets']; out = []
    for r in R[R.kind == 'constant_power'].itertuples():
        j = int(r.row_id); x = X[j]; Va, ea, Ia = anchor(x, 10.)
        out.append(dict(cell=r.sheet, d=r.direction, h=10., Vq=x[0], Va=Va, Iq=abs(wi[wo[j + 1] - 1]), Ia=Ia, exq=wv[wo[j + 1] - 1], exa=ea, T=x[TC], src_row=j))
    tr, tx, tt, ti, ty, tl, _, _ = load_test3()
    for j in range(len(tr)):
        h = float(tr.requested_horizon_s.iloc[j])
        if h not in (2., 10.): continue
        x = tx[j]; Va, ea, Ia = anchor(x, h)
        out.append(dict(cell='B', d='discharge', h=h, Vq=x[0], Va=Va, Iq=abs(ti[j, tl[j] - 1]), Ia=Ia, exq=ty[j, tl[j] - 1], exa=ea, T=x[TC], src_row=j))
    P = pd.DataFrame(out); P = P[np.isfinite(P.Va) & np.isfinite(P.exa)]
    return with_features(P)


def fit_w(parts):
    """parts: list of pair frames (one per source); each source gets equal total weight within (d, h)."""
    W = {}
    keys = set().union(*[set(map(tuple, p[['d', 'h']].drop_duplicates().to_numpy())) for p in parts if len(p)])
    for d, h in keys:
        A = []; b = []; w = []
        subs = [p[(p.d == d) & (p.h == h)] for p in parts]; subs = [s for s in subs if len(s)]
        for s in subs:
            A.append(S.design(s, True)); b.append(s.target.to_numpy()); w.append(np.full(len(s), 1. / (len(subs) * len(s))))
        A = np.vstack(A); b = np.concatenate(b); sw = np.sqrt(np.concatenate(w))
        W[(d, float(h))] = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]
    return W


def region(d, h, Vq, T, has_anchor):
    hh = 2. if h == 2. else 10.; cold = bool(T < 10)
    if not has_anchor: return f'{d}|{hh:g}|fallback_{"cold" if cold else "warm"}'
    if cold: return f'{d}|{hh:g}|cold'
    if d == 'discharge': r = 'dis>=3.40' if Vq >= 3.40 else ('dis3.29-3.40' if Vq >= 3.29 else 'dis<3.29')
    else: r = 'chg_head<0.2' if 4.15 - Vq < .2 else 'chg_head>=0.2'
    return f'{d}|{hh:g}|{r}'


def v196_margin(d, h, Vq, T, has_anchor):
    cold = bool(T < 10)
    if has_anchor: return S196.margin(d, h, Vq, True, 'M3') if cold else S196.m3_warm_margin(d, h, Vq)
    return S196.margin(d, h, Vq, cold, 'M2')


def has_anchor(x, h):
    Va, ea, Ia = anchor(x, h); return bool(np.isfinite(Va) and np.isfinite(ea) and ea > 1e-4 and Ia > .5)


def m2_power(p, x, h, d, m):
    lim = LIMIT[d] + m if d == 'discharge' else LIMIT[d] - m
    try: return float(S196.task((p, x[0], x[72], x[73], h, d, CAP[d], lim)))
    except Exception: return 0.


def power(a):
    kind, x, h, d, m, w, p = a
    if kind == 'M3': return S196.m3_sop(x, h, d, m, w)
    return m2_power(p, x, h, d, m)


def need(a):
    kind, x, h, d, w, p, truth = a
    f = lambda m: power((kind, x, h, d, m, w, p))
    if f(0.) <= truth + 1e-9: return 0.
    lo, hi = 0., 2.
    for _ in range(25):
        mid = (lo + hi) / 2
        if f(mid) <= truth + 1e-9: hi = mid
        else: lo = mid
    return hi


def judged():
    P = pd.read_csv(HERE / 'all6_measured_v178/predictions.csv')[['evaluation_group', 'sheet', 'direction', 'temp_C', 'horizon_s', 'row', 'soc_point', 'truth_power_W', 'truth_end_current_A', 'regime']]
    P['goal'] = (P.evaluation_group != 'B') | P.horizon_s.isin([2., 10.])
    return P


def inputs(group):
    Z = np.load(HERE / ('rpcwby_external_v46/inputs.npz' if group == 'B' else 'rpcwby_aging_inputs_v85/inputs.npz'))
    return Z['features'], Z['history_time_s'], Z['history_current_A'], Z['history_offsets']


def m2_for(group, d, rows):
    X, hts, his, off = inputs(group)
    return S196.m2_params(f'{d}__ALL6', X, hts, his, off, np.asarray(rows)) if len(rows) else np.empty((0, 10))


def rows_frame(P, group, pool_rows=False):
    """Judged voltage-limited goal rows of a group with region labels and (for fallback rows) M2 parameters."""
    X = inputs(group)[0]; V = P[(P.evaluation_group == group) & P.goal & ((P.regime == 'voltage_limited') | pool_rows)].copy()
    V['has_anchor'] = [has_anchor(X[int(r.row)], float(r.horizon_s)) for r in V.itertuples()]
    V['Vq'] = [X[int(r)][0] for r in V.row]; V['T'] = [X[int(r)][TC] for r in V.row]
    V['region'] = [region(r.direction, r.horizon_s, r.Vq, r.T, r.has_anchor) for r in V.itertuples()]
    m2 = {}
    for d in V.direction.unique():
        idx = V[(V.direction == d) & ~V.has_anchor].index
        if len(idx):
            prm = m2_for(group, d, V.loc[idx, 'row'].astype(int).tolist())
            for k, ix in enumerate(idx): m2[ix] = prm[k]
    V['m2'] = [m2.get(ix) for ix in V.index]
    return V, X


def main():
    OUT.mkdir(exist_ok=False); torch.set_num_threads(4)
    AG = S.pairs().reset_index(drop=True); SP = sop_pairs().reset_index(drop=True)
    print('aging pairs', len(AG), 'SOP pairs', SP.groupby(['cell', 'd', 'h']).size().to_dict(), flush=True)
    SP.to_csv(OUT / 'sop_pairs.csv', index=False)
    P = judged(); frames = {g: rows_frame(P, g) for g in FOLDS}
    shift_recs = []; margin_recs = []; weights = {}; preds = []
    pool = Pool(14)
    for g, train in FOLDS.items():
        Wf = fit_w([AG, SP[SP.cell.isin(train)]]); weights[g] = {f'{k[0]}|{k[1]:g}': v.tolist() for k, v in Wf.items()}
        for c in train:
            Wn = fit_w([AG, SP[SP.cell.isin([t for t in train if t != c])]])
            hg = [k for k, v in CELL.items() if v == c][0]; V, X = frames[hg]
            args = [('M3' if r.has_anchor else 'M2', X[int(r.row)], float(r.horizon_s), r.direction, Wn.get((r.direction, 2. if r.horizon_s == 2. else 10.)), r.m2, float(r.truth_power_W)) for r in V.itertuples()]
            sh = pool.map(need, args, chunksize=4)
            shift_recs.append(V.drop(columns=['m2']).assign(fold=g, heldout=c, shift_V=sh))
        H = pd.concat([s for s in shift_recs if s.fold.iloc[0] == g])
        for reg, q in H.groupby('region'):
            margin_recs.append(dict(fold=g, region=reg, n=len(q), cells='/'.join(sorted(q.heldout.unique())), max_V=float(q.shift_V.max()), p95_V=float(q.shift_V.quantile(.95)),
                                    zero_share=float((q.shift_V == 0).mean())))
        print('fold', g, 'weights', {k: np.round(v, 3).tolist() for k, v in weights[g].items() if k.endswith('|10')}, flush=True)
    M = pd.DataFrame(margin_recs); M.to_csv(OUT / 'margins.csv', index=False); pd.concat(shift_recs).to_csv(OUT / 'heldout_shifts.csv', index=False)
    dump(OUT / 'weights.json', weights)

    def margin_of(g, reg, d, h, Vq, T, anc, stat):
        q = M[(M.fold == g) & (M.region == reg)]
        if len(q) and int(q.n.iloc[0]) >= MIN_ROWS: return float(q[stat].iloc[0]), 'sop_heldout'
        return v196_margin(d, h, Vq, T, anc), 'v196_rule'

    for stat in ('max_V', 'p95_V'):
        for g in FOLDS:
            W = {(k.split('|')[0], float(k.split('|')[1])): np.array(v) for k, v in weights[g].items()}
            V, X = rows_frame(P, g, pool_rows=True)
            ms = [margin_of(g, r.region, r.direction, r.horizon_s, r.Vq, r.T, r.has_anchor, stat) for r in V.itertuples()]
            V['margin_V'] = [m for m, _ in ms]; V['margin_source'] = [s for _, s in ms]; V['method'] = np.where(V.has_anchor, 'M3', 'M2_fallback')
            args = [('M3' if r.has_anchor else 'M2', X[int(r.row)], float(r.horizon_s), r.direction, r.margin_V, W.get((r.direction, 2. if r.horizon_s == 2. else 10.)), r.m2) for r in V.itertuples()]
            V['power_W'] = pool.map(power, args, chunksize=4); V['margin_stat'] = stat
            preds.append(V.drop(columns=['m2']))
    D = pd.concat(preds, ignore_index=True); D.to_csv(OUT / 'predictions.csv', index=False)
    J = D[D.regime == 'voltage_limited'].copy(); J['over'] = J.power_W - J.truth_power_W > 1e-9; J['use'] = 100 * J.power_W / J.truth_power_W
    G = J.groupby(['margin_stat', 'evaluation_group', 'sheet', 'direction', 'temp_C', 'horizon_s']).agg(
        n=('row', 'size'), exceed=('over', 'sum'), median_usable_pct=('use', 'median'), fallback_rows=('method', lambda s: int((s != 'M3').sum())),
        fallback_exceed=('over', lambda s: int(s[J.loc[s.index, 'method'] != 'M3'].sum())), v196_rule_rows=('margin_source', lambda s: int((s == 'v196_rule').sum()))).reset_index()
    G['pass'] = (G.exceed == 0) & (G.median_usable_pct >= 80); G.to_csv(OUT / 'groups.csv', index=False)

    # auxiliary 1: measured-voltage-crossing loads allowed by the primary (max) model
    cross = crossed_check(D[D.margin_stat == 'max_V'], M, weights, pool, margin_of); cross.to_csv(OUT / 'crossed_loads.csv', index=False)
    pool.close()
    res = {}
    for stat, g in G.groupby('margin_stat'):
        res[stat] = dict(groups_passing=int(g['pass'].sum()), exceed=int(g.exceed.sum()), A_exceed=int(g[g.evaluation_group != 'B'].exceed.sum()),
                         B_exceed=int(g[g.evaluation_group == 'B'].exceed.sum()), groups_at_80=int((g.median_usable_pct >= 80).sum()),
                         fallback_exceed=int(g.fallback_exceed.sum()))
    res['primary'] = 'max_V'; res['crossed_loads'] = dict(n=int(len(cross)), allowed=int(cross.allowed.sum()), over40mV=int(cross.over40.sum()),
                                                          allowed_over40mV=int((cross.allowed & cross.over40).sum()))
    res.update(comparators='V196 1/15 exceed 159; V198 0/15 exceed 73; V193 0/15 exceed 109', adopted=False, goal_complete=False)
    dump(OUT / 'verification.json', res); pd.set_option('display.width', 250)
    print(M.round(4).to_string(index=False)); print(G.round(1).to_string(index=False)); print(json.dumps(res))


def crossed_check(D, M, weights, pool, margin_of):
    """All constant-power pulses (2/10 s, every pass) of each evaluated cell whose measured voltage went beyond the limit."""
    out = []
    R = pd.read_csv(HERE / 'rpcwby_waveform_inputs_v87/inputs.csv'); Z = np.load(HERE / 'rpcwby_waveform_inputs_v87/inputs.npz'); X = Z['features']
    wv = Z['wave_oriented_excursion_V']; wo = Z['wave_offsets']
    for r in R[R.kind == 'constant_power'].itertuples():
        j = int(r.row_id); head = X[j, 0] - LIMIT['discharge'] if r.direction == 'discharge' else LIMIT['charge'] - X[j, 0]; mx = float(wv[wo[j]:wo[j + 1]].max())
        if mx > head: out.append(dict(group='A1' if r.sheet == 'Test#1' else 'A17', src='v87', j=j, d=r.direction, h=10., temp=int(r.temp_block_C), power=abs(r.P_last_W), beyond_V=mx - head, is_final=bool(r.is_final)))
    tr, tx, tt, ti, ty, tl, _, _ = load_test3(); C = pd.read_csv(HERE / 'rpcwby_protocol_audit_v45/cp_pulses.csv')
    for j in range(len(tr)):
        h = float(tr.requested_horizon_s.iloc[j])
        if h not in (2., 10.): continue
        head = tx[j, 0] - LIMIT['discharge']; mx = float(ty[j, :tl[j]].max())
        if mx > head: out.append(dict(group='B', src='v46', j=j, d='discharge', h=h, temp=int(tr.nominal_temperature_C.iloc[j]), power=abs(C.power_W.iloc[j]), beyond_V=mx - head, is_final=bool(C.last_occurrence.iloc[j])))
    Q = pd.DataFrame(out)
    if not len(Q): return Q.assign(model_W=[], allowed=[], over40=[])
    ZA = np.load(HERE / 'rpcwby_waveform_inputs_v87/inputs.npz'); ZB = np.load(HERE / 'rpcwby_external_v46/inputs.npz'); args = []
    m2 = {}
    for (g, d), q in Q.groupby(['group', 'd']):
        Zs = ZB if g == 'B' else ZA; Xs = Zs['features']
        idx = [ix for ix in q.index if not has_anchor(Xs[int(Q.at[ix, 'j'])], Q.at[ix, 'h'])]
        if idx:
            prm = S196.m2_params(f'{d}__ALL6', Xs, Zs['history_time_s'], Zs['history_current_A'], Zs['history_offsets'], np.array([int(Q.at[ix, 'j']) for ix in idx]))
            for k, ix in enumerate(idx): m2[ix] = prm[k]
    Q['m2'] = [m2.get(ix) for ix in Q.index]
    for r in Q.itertuples():
        Xs = ZB['features'] if r.group == 'B' else ZA['features']; x = Xs[int(r.j)]; anc = has_anchor(x, r.h); reg = region(r.d, r.h, x[0], x[TC], anc)
        m, _ = margin_of(r.group, reg, r.d, r.h, x[0], x[TC], anc, 'max_V'); W = weights[r.group]
        w = np.array(W[f'{r.d}|{2 if r.h == 2. else 10}']) if anc else None
        args.append(('M3' if anc else 'M2', x, r.h, r.d, m, w, r.m2))
    Q['model_W'] = pool.map(power, args, chunksize=4); Q['allowed'] = Q.model_W >= Q.power - 1e-9; Q['over40'] = Q.beyond_V > .04
    return Q.drop(columns=['m2'])


if __name__ == '__main__':
    main()
