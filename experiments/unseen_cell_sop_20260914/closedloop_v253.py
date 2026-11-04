"""V253 (PROTOCOL_CLOSEDLOOP_V253.md): per-row response model (fitted on the row's truth pulse) and an ideal voltage-clamp
simulation of the over-requests of V252; validated on the CCCV seed pulses of the A cells. usage: python closedloop_v253.py"""
import json
import numpy as np, pandas as pd
from scipy.optimize import nnls
from run_envelope import HERE
import closedloop_v251 as C1, closedloop_v252 as C2
import relax_transfer_v201 as V1

OUT = HERE / 'closedloop_v253'; MC = HERE / 'mamba_v211' / 'cache'
CELLMAP = {'A1': 'RPCWBY_TEST1', 'A17': 'RPCWBY_TEST2', 'B': 'RPCWBY_TEST3'}
TAUS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0); DT = 0.05
VLIM = {'discharge': 2.55, 'charge': 4.15}; ICAP = {'discharge': 30.0, 'charge': 15.0}


def lag(t, i, tau):
    """Unit-gain first-order lag of a piecewise-constant current sampled at t (current before t[0] = i[0])."""
    u, prev_t, out = 0.0, 0.0, []
    for tk, ik in zip(t, i):
        a = np.exp(-(tk - prev_t) / tau); u = u * a + ik * (1 - a); out.append(u); prev_t = tk
    return np.array(out)


def charge(t, i):
    return np.cumsum(i * np.diff(np.r_[0.0, t]))


def fit_ecm(t, i, dv):
    """(tau, R_a, R_b, k, rmse) for |dV| = R_a |I| + R_b u_tau + k Q."""
    best = None
    Q = charge(t, i)
    for tau in TAUS:
        cols = [i, lag(t, i, tau), Q] if len(t) >= 4 else [i, np.zeros(len(t)), Q]
        A = np.column_stack(cols); x, r = nnls(A, dv)
        rm = float(np.sqrt(np.mean((A @ x - dv) ** 2)))
        if best is None or rm < best[-1]:
            best = (tau, *x, rm)
        if len(t) < 4:
            break
    return best


def simulate(ecm, v0, d, P=None, I_cc=None, horizon=10.0):
    """Constant power P (or constant current I_cc) with a voltage clamp at the limit; returns (energy_J, P_end, t_clamp)."""
    tau, Ra, Rb, k = ecm[:4]; sgn = -1.0 if d == 'discharge' else 1.0; vl = VLIM[d]; cap = ICAP[d]
    u = q = E = 0.0; t_clamp = np.nan; p = 0.0; n = int(round(horizon / DT))
    for step in range(n):
        vx = v0 + sgn * (Rb * u + k * q)                     # open-circuit-like part (without the R_a drop)
        if I_cc is not None:
            im = min(I_cc, cap)
        elif d == 'discharge':
            disc = vx * vx - 4 * Ra * P
            im = (vx - np.sqrt(disc)) / (2 * Ra) if (Ra > 0 and disc >= 0) else (P / vx if Ra == 0 else cap)
        else:
            im = (-vx + np.sqrt(vx * vx + 4 * Ra * P)) / (2 * Ra) if Ra > 0 else P / vx
        im = min(max(im, 0.0), cap)
        v = vx + sgn * Ra * im
        if (d == 'discharge' and v < vl) or (d == 'charge' and v > vl):
            im = max((vx - vl) / Ra, 0.0) if d == 'discharge' else max((vl - vx) / Ra, 0.0)
            im = min(im, cap); v = vx + sgn * Ra * im
            if np.isnan(t_clamp):
                t_clamp = step * DT
        p = im * v; E += p * DT
        a = np.exp(-DT / tau); u = u * a + im * (1 - a); q += im * DT
    return E, p, t_clamp


def main():
    OUT.mkdir(exist_ok=True); res = {}
    # truth pulses and response models
    rows = []
    for g, cell in CELLMAP.items():
        z = np.load(MC / f'inputs_v238_{cell}.npz'); r = pd.read_csv(MC / f'inputs_v238_{cell}_rows.csv')
        for n in np.flatnonzero(r.goal.to_numpy()):
            m = z['Pm'][n] > 0; t = z['Pt'][n][m].astype(float); i = np.abs(z['Pi'][n][m].astype(float)); dv = np.abs(z['Pv'][n][m].astype(float))
            ecm = fit_ecm(t, i, dv); rr = r.iloc[n]
            rows.append(dict(row_id=rr.row_id, evaluation_group=g, direction=rr.direction, horizon_s=float(rr.horizon_s), V0=float(rr.V0),
                             tau=ecm[0], Ra=ecm[1], Rb=ecm[2], k=ecm[3], fit_rmse_V=ecm[4], n_samples=int(m.sum())))
    E = pd.DataFrame(rows)
    B = pd.read_csv(HERE / 'closedloop_v251/src/bench_rows.csv')[['row_id', 'temp_C', 'soc_point', 'truth_power_W']]
    E = E.merge(B, on='row_id'); assert len(E) == 672
    # fit check: simulate the truth power -> end voltage at the limit?
    endv = []
    for e in E.itertuples():
        ecm = (e.tau, e.Ra, e.Rb, e.k)
        _, pend, tcl = simulate(ecm, e.V0, e.direction, P=e.truth_power_W, horizon=e.horizon_s)
        endv.append(dict(clamped_at_truth=bool(np.isfinite(tcl)), clamp_time_at_truth=tcl, pend_over_truth=pend / e.truth_power_W))
    E = pd.concat([E, pd.DataFrame(endv)], axis=1)
    res['fit'] = dict(rmse_V_q50=float(E.fit_rmse_V.median()), rmse_V_q95=float(E.fit_rmse_V.quantile(.95)),
                      share_clamped_at_truth=float(E.clamped_at_truth.mean()), clamp_time_at_truth_q50=float(E.clamp_time_at_truth.median()),
                      pend_over_truth_q50=float(E.pend_over_truth.median()))
    E.to_csv(OUT / 'ecm_rows.csv', index=False)
    print('fit', json.dumps(res['fit']), flush=True)
    # validation on CCCV seeds (A cells)
    R, X, L, own = V1.load()
    seeds = R[R.kind == 'cccv_seed'][['member', 'direction', 'soc_counter', 'row_id', 'P_last_W', 'V_last_V', 'I_last_A']].rename(columns={'row_id': 'seed_row'})
    W = pd.read_csv(HERE / 'mamba_v211/precheck/history_windows.csv')[['row_id', 'member', 'query_first_row']]
    Rq = R[['member', 'first_row', 'soc_counter', 'direction']].rename(columns={'first_row': 'query_first_row'})
    link = W.merge(Rq, on=['member', 'query_first_row'], how='left')                 # query row -> its SOC counter
    V = E[E.evaluation_group != 'B'].merge(link[['row_id', 'member', 'soc_counter']], on='row_id', how='left').merge(seeds, on=['member', 'direction', 'soc_counter'], how='inner')
    val = []
    for v in V.itertuples():
        ecm = (v.tau, v.Ra, v.Rb, v.k); v0s = float(X[int(v.seed_row)][0])
        _, pend, tcl = simulate(ecm, v0s, v.direction, I_cc=ICAP[v.direction], horizon=10.0)
        val.append(dict(row_id=v.row_id, direction=v.direction, soc_point=v.soc_point, temp_C=v.temp_C, P_seed=abs(v.P_last_W), P_sim=pend,
                        seed_clamped=bool(abs(v.V_last_V - VLIM[v.direction]) < 0.01), sim_clamped=bool(np.isfinite(tcl))))
    Vd = pd.DataFrame(val); Vd['log_err'] = np.log(Vd.P_sim / Vd.P_seed); Vd.to_csv(OUT / 'seed_validation.csv', index=False)
    ae = Vd.log_err.abs()
    res['seed_validation'] = dict(n=int(len(Vd)), median_abs_log_err=float(ae.median()), q95_abs_log_err=float(ae.quantile(.95)),
                                  bias_median_log=float(Vd.log_err.median()), clamp_agreement=float((Vd.seed_clamped == Vd.sim_clamped).mean()),
                                  by_direction={d: dict(n=int(len(x)), median_abs=float(x.log_err.abs().median()), q95_abs=float(x.log_err.abs().quantile(.95)),
                                                        bias=float(x.log_err.median())) for d, x in Vd.groupby('direction')},
                                  credible=bool(ae.median() <= 0.10))
    print('seed validation', json.dumps(res['seed_validation']), flush=True)
    # application to the V252 requests (tau = 5 %)
    S252 = json.load(open(HERE / 'closedloop_v252/summary.json'))
    req = {'V242': C2.v_folds('relax_transfer_v242/predictions_primary_k_cross.csv', 'v242_pred_W'), 'sop_E3causal': C2.sop_folds('E3causal')}
    key = ['evaluation_group', 'direction', 'temp_C', 'horizon_s', 'soc_point']
    Bk = pd.read_csv(HERE / 'closedloop_v251/src/bench_rows.csv')[['row_id'] + key + ['truth_power_W']]
    out = {}
    for name, F in req.items():
        sc = {c: S252[name]['tau_5pct']['scales'][c]['scale'] for c in ('A1', 'A17', 'B')}
        if name == 'V242':
            Jv = C1.v_series('relax_transfer_v242/predictions_primary_k_cross.csv', 'v242_pred_W')
            D0 = pd.read_csv(HERE / 'relax_transfer_v242/predictions_primary_k_cross.csv'); D0 = D0[D0.regime == 'voltage_limited']
            Jv['soc_point'] = D0.soc_point.to_numpy(); Jv = Jv.merge(Bk.rename(columns={'truth_power_W': 'truth'}), on=key + ['truth'])
        else:
            Fo = F.copy(); Fo['row_id'] = pd.read_csv(HERE / 'closedloop_v251/src/folds_E3causal.csv').row_id.to_numpy()
            Jv = Fo[Fo.fold == Fo.evaluation_group].copy()
        Jv['req'] = [sc[g] * c for g, c in zip(Jv.evaluation_group, Jv.central)]
        Jv = Jv.merge(E[['row_id', 'tau', 'Ra', 'Rb', 'k', 'V0']], on='row_id', how='left'); assert Jv.Ra.notna().all() and len(Jv) == 672, name
        recs = []
        for j in Jv.itertuples():
            ratio = j.req / j.truth; Etruth = j.truth * j.horizon_s
            if ratio <= 1 + 1e-9:
                recs.append(dict(intervened=False, E_del=j.req * j.horizon_s, P_end=j.req, t_clamp=np.nan)); continue
            Ed, pend, tcl = simulate((j.tau, j.Ra, j.Rb, j.k), j.V0, j.direction, P=j.req, horizon=j.horizon_s)
            recs.append(dict(intervened=True, E_del=Ed, P_end=pend, t_clamp=tcl))
        Jv = pd.concat([Jv.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
        Jv['del_over_req'] = Jv.E_del / (Jv.req * Jv.horizon_s); Jv['del_over_truth'] = Jv.E_del / (Jv.truth * Jv.horizon_s)
        Jv['pend_over_truth'] = Jv.P_end / Jv.truth; Jv['lower_bound'] = np.minimum(1.0, Jv.truth / Jv.req)
        Jv.to_csv(OUT / f'rows_{name}.csv', index=False)
        it = Jv[Jv.intervened]
        q = lambda s: dict(q50=float(s.median()), q05=float(s.quantile(.05)), min=float(s.min())) if len(s) else None
        G = Jv.groupby(['evaluation_group', 'direction', 'temp_C', 'horizon_s']).agg(n=('row_id', 'size'), interventions=('intervened', 'sum'),
                                                                                   delivered_util_q50=('del_over_truth', lambda s: float(np.median(np.minimum(s, 1.0))))).reset_index()
        G.to_csv(OUT / f'groups_{name}.csv', index=False)
        out[name] = dict(scales=sc, interventions=int(len(it)), del_over_req=q(it.del_over_req), del_over_truth=q(it.del_over_truth),
                         pend_over_truth=q(it.pend_over_truth), clamp_time=q(it.t_clamp), lower_bound_del_over_req=q(it.lower_bound),
                         share_del_over_truth_ge_1=float((it.del_over_truth >= 1).mean()) if len(it) else None,
                         groups_delivered_util80=int((G.delivered_util_q50 >= 0.8).sum()))
        print(name, json.dumps(out[name]), flush=True)
    res['application'] = out
    json.dump(res, open(OUT / 'summary.json', 'w'), indent=1, default=lambda x: None)


if __name__ == '__main__':
    main()
