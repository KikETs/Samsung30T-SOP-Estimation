"""V201 (PROTOCOL_RELAX_V201.md) = V200 plus direct power transfer for anchored A charge rows. Base text of V200:
V200 (PROTOCOL_RELAX_V200.md): A-cell rows re-predicted with prefix relaxation features; every other row keeps the V199 primary prediction.
Model 1 = direct power transfer for anchored A discharge rows, model 2 = relaxation model for anchorless A charge rows. Training = the other A
cell only; derating factor = max predicted/truth over leave-one-round-out predictions inside the training cell."""
import json
import numpy as np, pandas as pd
from run_envelope import HERE, dump
import cross_sop_v199 as C

OUT = HERE / 'relax_transfer_v201'; NM = C.NM; TC = C.TC
OTHER = {'A1': ('Test#1', 'Test#2'), 'A17': ('Test#2', 'Test#1')}
CAP = dict(discharge=30., charge=15.)


def load():
    R = pd.read_csv(HERE / 'rpcwby_waveform_inputs_v87/inputs.csv'); X = np.load(HERE / 'rpcwby_waveform_inputs_v87/inputs.npz')['features']
    L = pd.read_csv(HERE / 'relax_inputs_v200/relax.csv').set_index('row_id')
    own = {}
    for m, g in R.groupby('member'):
        fr = g.first_row.to_numpy(); lr = g.last_row.to_numpy(); ids = g.row_id.to_numpy()
        for r in g.itertuples():
            if r.past_10s_end_index >= 0:
                k = np.flatnonzero((fr <= r.past_10s_end_index) & (lr >= r.past_10s_end_index)); own[r.row_id] = int(ids[k[0]]) if len(k) else -1
    return R, X, L, own


def rel(L, j):
    if j < 0 or j not in L.index or not bool(L.at[j, 'found']): return np.nan, np.nan, np.nan
    q = L.loc[j]; I = abs(q.step_I_A)
    return (q.pre_V - q.step_end_V) / I, (q.pre_V - q.rest30_V) / I, (q.rest1_V - q.step_end_V) / I


def feats(j, d, X, L, own):
    """Returns (model, phi, scale) or None. scale multiplies exp(phi.w) to give power."""
    x = X[j]; Vq = x[0]; T = x[TC]; tt = 1000 / (T + 273.15) - 1000 / 298.15
    anc = C.has_anchor(x, 10.)
    if d == 'discharge' and anc:
        Va = x[NM.index('past_10s_pre_voltage_V')]; ea = x[NM.index('past_10s_oriented_voltage_excursion_V')]; Ia = abs(x[NM.index('past_10s_endpoint_current_A')])
        hq = Vq - 2.55; ha = Va - 2.55; rtq, rsq, _ = rel(L, j); rta, rsa, _ = rel(L, own.get(j, -1))
        phi = np.array([1., np.log(hq / ha) if hq > 0 and ha > 0 else np.nan, np.log(ea / ha) if ha > 0 else np.nan, tt,
                        np.log(rtq / rta), np.log(rsq / rsa), np.log(rtq), np.log(ea / Ia)])
        if np.isfinite(phi).all(): return 'V200_direct', phi, Ia * (Va - ea)
        return None
    if d == 'charge' and anc:
        Va = x[NM.index('past_10s_pre_voltage_V')]; ea = x[NM.index('past_10s_oriented_voltage_excursion_V')]; Ia = abs(x[NM.index('past_10s_endpoint_current_A')])
        hq = 4.15 - Vq; ha = 4.15 - Va; rtq, rsq, _ = rel(L, j); rta, rsa, _ = rel(L, own.get(j, -1))
        phi = np.array([1., np.log(hq / ha) if hq > 0 and ha > 0 else np.nan, np.log(ea / ha) if ha > 0 else np.nan, tt,
                        np.log(rtq / rta), np.log(rsq / rsa), np.log(rtq), np.log(ea / Ia)])
        if np.isfinite(phi).all(): return 'V201_charge_direct', phi, Ia * (Va + ea)
        return None
    if d == 'charge' and not anc:
        hq = 4.15 - Vq; rt, rs, rj = rel(L, j)
        psi = np.array([1., np.log(hq) if hq > 0 else np.nan, tt, np.log(rt), np.log(rs), np.log(rj)])
        if np.isfinite(psi).all(): return 'V200_relax_fallback', psi, 1.
    return None


def main():
    OUT.mkdir(exist_ok=False)
    R, X, L, own = load()
    D = pd.read_csv(HERE / 'relax_transfer_v200/predictions.csv').drop(columns=['v200_model', 'k', 'pred_W'])
    R85 = pd.read_csv(HERE / 'rpcwby_aging_inputs_v85/inputs.csv')[['row_id', 'pulse_id', 'member']]
    A = D[D.evaluation_group != 'B'].merge(R85, left_on='row', right_on='row_id', how='left').merge(R[['row_id', 'pulse_id']].rename(columns={'row_id': 'r87'}), on='pulse_id', how='left')
    assert A.r87.notna().all() and len(A) == (D.evaluation_group != 'B').sum()
    A.index = D.index[D.evaluation_group != 'B']
    info = {ix: feats(int(r.r87), r.direction, X, L, own) for ix, r in A.iterrows()}
    D['v200_model'] = ''; D['k'] = np.nan; D['pred_W'] = np.nan; fits = {}; loro_recs = []
    for g, (ev, tr_cell) in OTHER.items():
        for model in ('V200_direct', 'V200_relax_fallback', 'V201_charge_direct'):
            T = A[(A.sheet == tr_cell) & (A.regime == 'voltage_limited') & np.array([info[ix] is not None and info[ix][0] == model for ix in A.index])]
            Phi = np.array([info[ix][1] for ix in T.index]); sc = np.array([info[ix][2] for ix in T.index]); y = np.log(T.truth_power_W.to_numpy() / sc)
            w = np.linalg.lstsq(Phi, y, rcond=None)[0]; ratios = []
            for m in T.member.unique():
                hold = (T.member == m).to_numpy(); w2 = np.linalg.lstsq(Phi[~hold], y[~hold], rcond=None)[0]
                r = np.exp(Phi[hold] @ w2 - y[hold]); ratios += list(r)
                loro_recs += [dict(fold=g, model=model, member=m, ratio=float(v)) for v in r]
            k = float(max(ratios)); fits[f'{g}|{model}'] = dict(train_cell=tr_cell, n=int(len(T)), rounds=int(T.member.nunique()), k=k, w=w.tolist(),
                                                                   in_sample_ratio_max=float(np.exp(Phi @ w - y).max()))
            E = A[(A.sheet == ev) & np.array([info[ix] is not None and info[ix][0] == model for ix in A.index])]
            for ix, r in E.iterrows():
                _, phi, s = info[ix]; pred = float(s * np.exp(phi @ w)); cmd = min(pred / k, CAP[r.direction] * r.Vq)
                D.loc[ix, ['v200_model', 'k', 'pred_W', 'power_W', 'method']] = [model, k, pred, cmd, model]
    D.to_csv(OUT / 'predictions.csv', index=False); pd.DataFrame(loro_recs).to_csv(OUT / 'loro_ratios.csv', index=False); dump(OUT / 'fits.json', fits)
    J = D[D.regime == 'voltage_limited'].copy(); J['over'] = J.power_W - J.truth_power_W > 1e-9; J['use'] = 100 * J.power_W / J.truth_power_W
    G = J.groupby(['evaluation_group', 'sheet', 'direction', 'temp_C', 'horizon_s']).agg(n=('row', 'size'), exceed=('over', 'sum'), median_usable_pct=('use', 'median'),
                                                                                     v200_rows=('v200_model', lambda s: int((s != '').sum())),
                                                                                     v200_exceed=('over', lambda s: int(s[J.loc[s.index, 'v200_model'] != ''].sum()))).reset_index()
    G['pass'] = (G.exceed == 0) & (G.median_usable_pct >= 80); G.to_csv(OUT / 'groups.csv', index=False)
    # measured-voltage-crossing loads: A pulses re-predicted where a V200 model applies
    Q = pd.read_csv(HERE / 'cross_sop_v199/crossed_loads.csv'); Q['v200_model'] = ''  # V201 re-predicts every A pulse where a V200/V201 model applies
    for ix, r in Q[Q.src == 'v87'].iterrows():
        f = feats(int(r.j), r.d, X, L, own)
        if f is None: continue
        ft = fits[f'{r.group}|{f[0]}']; pred = float(f[2] * np.exp(f[1] @ np.array(ft['w']))); Q.loc[ix, ['model_W', 'v200_model']] = [min(pred / ft['k'], CAP[r.d] * X[int(r.j), 0]), f[0]]
    Q['allowed'] = Q.model_W >= Q.power - 1e-9; Q.to_csv(OUT / 'crossed_loads.csv', index=False)
    res = dict(groups_passing=int(G['pass'].sum()), exceed=int(G.exceed.sum()), A_exceed=int(G[G.evaluation_group != 'B'].exceed.sum()), B_exceed=int(G[G.evaluation_group == 'B'].exceed.sum()),
               groups_at_80=int((G.median_usable_pct >= 80).sum()), v200_rows=int((J.v200_model != '').sum()), v200_exceed=int(G.v200_exceed.sum()),
               crossed_loads=dict(n=int(len(Q)), allowed=int(Q.allowed.sum()), allowed_over40mV=int((Q.allowed & Q.over40).sum()), v200_repredicted=int((Q.v200_model != '').sum())),
               fits={k: {kk: vv for kk, vv in v.items() if kk != 'w'} for k, v in fits.items()}, comparators='V200 1/15 exceed 13; V199 1/15 exceed 26', adopted=False, goal_complete=False)
    dump(OUT / 'verification.json', res); pd.set_option('display.width', 250); print(G.round(1).to_string(index=False)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
