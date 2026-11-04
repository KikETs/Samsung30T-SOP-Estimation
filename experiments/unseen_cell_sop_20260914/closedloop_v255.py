"""V255 (PROTOCOL_CLOSEDLOOP_V255.md): closed-loop scales of the hybrid (sop E3causal for the queries of the event list, V242
for the short-rest queries) chosen per class from out-of-sample predictions of the other cells. usage: python closedloop_v255.py"""
import json
import numpy as np, pandas as pd
from run_envelope import HERE, dump
import closedloop_v251 as C1
from closedloop_v253 import simulate

OUT = HERE / 'closedloop_v255'; SRC = OUT / 'src'
CELLS = ('A1', 'A17', 'B'); GRID = np.round(np.arange(0.20, 1.501, 0.01), 2)
GROUP_INT, GROUP_UTIL = 0.10, 0.80
V252_SOP = dict(scale=0.91, int_rate=0.0774, util_q50=0.9147, groups=11)      # V252 sop E3causal, tau 5 % (reproduction target)
CONFIGS = [('1_sop_single_in', 'sop', 'single', 'in'), ('2_sop_single_out', 'sop', 'single', 'out'),
           ('3_hybrid_two_out', 'hybrid', 'two', 'out'), ('4_hybrid_class_in', 'hybrid', 'class', 'in'),
           ('5_hybrid_class_out', 'hybrid', 'class', 'out')]
MAIN = '5_hybrid_class_out'


def load():
    B = pd.read_csv(HERE / 'closedloop_v251/src/bench_rows.csv')
    E = pd.read_csv(SRC / 'events_v2.csv'); ev = set(zip(E.member, E.event_id))
    B['in_events'] = [(m, q) in ev for m, q in zip(B.member, B.q_row)]
    F = pd.read_csv(HERE / 'closedloop_v251/src/folds_E3causal.csv')
    P = np.exp(F.pivot(index='row_id', columns='fold', values='m').reindex(B.row_id))
    for c in CELLS:
        B[f'sop_{c}'] = P[c].to_numpy()                 # sop central of the row from the fold-c model (fitted without c)
    D = pd.read_csv(HERE / 'relax_transfer_v242/predictions_primary_k_cross.csv'); D = D[D.regime == 'voltage_limited']
    cen = D.pred_W.where(D.pred_W.notna(), D.power_W); cen = D.v242_pred_W.where(D.v242_pred_W.notna(), cen)   # V251 central
    D = pd.DataFrame({'sheet': D.sheet.to_numpy(), 'v178_row': D.row.to_numpy(), 'v242': cen.to_numpy()})
    B = B.merge(D, on=['sheet', 'v178_row'], how='left').rename(columns={'truth_power_W': 'truth'})
    assert len(B) == 672 and B.v242.notna().all() and B[[f'sop_{c}' for c in CELLS]].notna().all().all()
    return B


def classes(B, predictor, kind):
    """'short' = hybrid row outside the event list; otherwise 'all' (single), 'sop' (two) or the direction (class)."""
    if predictor == 'sop':
        return pd.Series('all', index=B.index)
    inner = B.direction if kind == 'class' else pd.Series({'single': 'all', 'two': 'sop'}[kind], index=B.index)
    return pd.Series(np.where(B.in_events, inner, 'short'), index=B.index)


def calibration(B, K, c, k, calib):
    """(pred, truth) of the calibration rows of evaluated cell c and class k."""
    if k == 'short':                                     # only B has short-rest rows: V242 on the other (A) cells' discharge rows
        m = B.evaluation_group.isin([d for d in ('A1', 'A17') if d != c]) & (B.direction == 'discharge')
        return B.v242[m].to_numpy(), B.truth[m].to_numpy()
    m = (B.evaluation_group != c) & (K == k)
    if calib == 'in':                                    # fold-c model on the other cells (in-sample, V252)
        pred = B.loc[m, f'sop_{c}'].to_numpy()
    else:                                                # each other cell d from its own fold-d model (out-of-sample)
        pred = np.array([B.at[i, f'sop_{g}'] for i, g in zip(B.index[m], B.evaluation_group[m])])
    return pred, B.truth[m].to_numpy()


def choose(pred, truth, tau):
    for s in GRID[::-1]:
        if np.mean(s * pred > truth + 1e-9) <= tau:
            return float(s), False
    return float(GRID[0]), True


def group_table(J, groups_all):
    summ, G = C1.closed_loop(J, J.pred.to_numpy())
    G = groups_all.merge(G, on=C1.GK, how='left')
    ok_u = G.util_q50 >= GROUP_UTIL; ok_i = G.int_rate <= GROUP_INT
    G['passed'] = ok_u & ok_i
    G['reason'] = np.select([G.passed, ~ok_i & ~ok_u, ~ok_i], ['pass', 'interventions+utilisation', 'interventions'], 'utilisation')
    return summ, G


def run(B, groups_all, name, predictor, kind, calib, tau):
    K = classes(B, predictor, kind); parts, scales = [], {}
    for c in CELLS:
        ev = B[B.evaluation_group == c].copy(); ev['cls'] = K[ev.index]
        ev['central'] = np.where((predictor == 'hybrid') & (ev.cls == 'short'), ev.v242, ev[f'sop_{c}'])
        ev['pred'] = np.nan
        for k in sorted(ev.cls.unique()):
            p, t = calibration(B, K, c, k, calib); s, floor = choose(p, t, tau)
            scales[f'{c}|{k}'] = dict(scale=s, floor_bound=floor, calibration_rows=int(len(p)), calibration_int_rate=float(np.mean(s * p > t + 1e-9)))
            ev.loc[ev.cls == k, 'pred'] = s * ev.loc[ev.cls == k, 'central']
        parts.append(ev)
    J = pd.concat(parts); assert len(J) == 672 and J.pred.notna().all()
    summ, G = group_table(J, groups_all); J['deployed'] = J.pred
    res = dict(name=name, predictor=predictor, classes=kind, calibration=calib, tau=tau, scales=scales,
               held_out=dict(int_rate=summ['int_rate'], depth_q95=summ['depth_q95'], depth_max=summ['depth_max'], util_q50=summ['util_q50']),
               groups_pass=int(G.passed.sum()), failed={f"{r.evaluation_group} {r.direction} {r.temp_C} C {r.horizon_s:g} s": r.reason
                                                        for r in G[~G.passed].itertuples()},
               open_loop_pass_exceed=list(C1.open_loop(J, 'deployed')))
    return res, G.assign(config=name, tau=tau), J


def deliver(J):
    """V253 response models and voltage-clamp simulation for the intervened rows of J."""
    E = pd.read_csv(HERE / 'closedloop_v253/ecm_rows.csv')[['row_id', 'tau', 'Ra', 'Rb', 'k', 'V0']]
    E.columns = ['row_id', 'e_tau', 'e_Ra', 'e_Rb', 'e_k', 'e_V0']
    J = J.merge(E, on='row_id', how='left'); assert len(J) == 672 and J.e_Ra.notna().all()
    recs = []
    for j in J.itertuples():
        if j.pred / j.truth <= 1 + 1e-9:
            recs.append(dict(intervened=False, E_del=j.pred * j.horizon_s, P_end=j.pred, t_clamp=np.nan)); continue
        Ed, pend, tcl = simulate((j.e_tau, j.e_Ra, j.e_Rb, j.e_k), j.e_V0, j.direction, P=j.pred, horizon=j.horizon_s)
        recs.append(dict(intervened=True, E_del=Ed, P_end=pend, t_clamp=tcl))
    J = pd.concat([J.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    J['del_over_req'] = J.E_del / (J.pred * J.horizon_s); J['del_over_truth'] = J.E_del / (J.truth * J.horizon_s)
    J['del_util'] = np.minimum(J.del_over_truth, 1.0)
    it = J[J.intervened]
    q = lambda s: dict(q50=float(s.median()), q05=float(s.quantile(.05)), min=float(s.min())) if len(s) else None
    G = J.groupby(C1.GK).agg(n=('row_id', 'size'), interventions=('intervened', 'sum'), del_util_q50=('del_util', 'median')).reset_index()
    G['int_rate'] = G.interventions / G.n
    out = dict(interventions=int(len(it)), del_over_req=q(it.del_over_req), del_over_truth=q(it.del_over_truth),
               share_del_over_truth_ge_1=float((it.del_over_truth >= 1).mean()) if len(it) else None, clamp_time=q(it.t_clamp),
               groups_delivered_util80=int((G.del_util_q50 >= GROUP_UTIL).sum()),
               groups_delivered_util80_int10=int(((G.del_util_q50 >= GROUP_UTIL) & (G.int_rate <= GROUP_INT)).sum()))
    return out, J, G


def main():
    OUT.mkdir(exist_ok=True)
    B = load(); groups_all = B[C1.GK].drop_duplicates().sort_values(C1.GK).reset_index(drop=True); assert len(groups_all) == 15
    print('rows outside the event list:', B.loc[~B.in_events, 'row_id'].tolist(), flush=True)
    res, allg = {}, []
    for name, predictor, kind, calib in CONFIGS:
        for tau in ((0.05, 0.10) if name == MAIN else (0.05,)):
            r, G, J = run(B, groups_all, name, predictor, kind, calib, tau); res[f'{name}|tau{int(tau * 100)}'] = r; allg.append(G)
            h = r['held_out']; sc = sorted({f"{k}={v['scale']:.2f}" for k, v in r['scales'].items()})
            print(f"{name:20s} tau {int(tau * 100):2d}%  groups {r['groups_pass']}/15  held-out int {100 * h['int_rate']:.2f}%  util {100 * h['util_q50']:.2f}%  "
                  f"depth q95/max {h['depth_q95']:.3f}/{h['depth_max']:.3f}  open-loop {r['open_loop_pass_exceed']}  scales {' '.join(sc)}", flush=True)
            for g, why in r['failed'].items():
                print(f'      failed: {g}: {why}', flush=True)
            if name == '1_sop_single_in':
                ok = (all(abs(v['scale'] - V252_SOP['scale']) < 1e-9 for v in r['scales'].values())
                      and abs(h['int_rate'] - V252_SOP['int_rate']) <= 5e-4 and abs(h['util_q50'] - V252_SOP['util_q50']) <= 5e-4
                      and r['groups_pass'] == V252_SOP['groups'])
                res['reproduction_V252'] = bool(ok); print('reproduction of V252 sop E3causal:', 'OK' if ok else 'FAILED', flush=True)
                if not ok:
                    dump(OUT / 'summary.json', res); raise SystemExit('V252 not reproduced; stopping (protocol)')
            if name == MAIN and tau == 0.05:
                d, Jd, Gd = deliver(J); res['delivery_main'] = d
                Jd.drop(columns=[c for c in Jd.columns if c.startswith('sop_')]).to_csv(OUT / 'rows_main.csv', index=False)
                Gd.to_csv(OUT / 'groups_delivery_main.csv', index=False)
                print('   delivery (V253 response models):', d, flush=True)
    G = pd.concat(allg); G.to_csv(OUT / 'groups.csv', index=False)
    m = res[f'{MAIN}|tau5']
    v = '15/15' if (m['groups_pass'] == 15 and m['held_out']['int_rate'] <= 0.10) else (
        'V242 이상' if (m['groups_pass'] >= 12 and m['held_out']['int_rate'] <= 0.10) else '개선 없음')
    res['verdict_main'] = v; print('verdict (main, tau 5 %):', v, flush=True)
    dis10 = G[(G.direction == 'discharge') & (G.temp_C == 10) & (G.horizon_s == 10) & (G.tau == 0.05)
              & G.config.isin(['4_hybrid_class_in', MAIN])][['config', 'evaluation_group', 'n', 'interventions', 'int_rate', 'util_q50', 'reason']]
    res['discharge_10C_in_vs_out'] = json.loads(dis10.round(4).to_json(orient='records')); print(dis10.round(3).to_string(index=False))
    dump(OUT / 'summary.json', res)


if __name__ == '__main__':
    main()
