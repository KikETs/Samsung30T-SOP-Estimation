"""V257 (PROTOCOL_CLOSEDLOOP_V257.md): the short-rest rows (B 25 C 2 s) of the V255 hybrid predicted by the V199 transfer
without its voltage margin (m = 0), the short-class scale recalibrated with the same definition on the A cells; the 13
margin-derated rows of the V242 baseline replaced the same way (secondary). usage: python closedloop_v257.py"""
import json
from multiprocessing import Pool
import numpy as np, pandas as pd
from run_envelope import HERE, dump
import closedloop_v255 as V5, closedloop_v256 as V6
import cross_sop_v199 as V199

OUT = HERE / 'closedloop_v257'
TAUS = V6.TAUS


def model_rows():
    """Voltage-limited goal rows of A1/A17/B with the V199 inputs, anchor flag, fold weights and M2 parameters."""
    P = V199.judged(); W = json.load(open(HERE / 'cross_sop_v199/weights.json')); out = []
    for g in V5.CELLS:
        V, X = V199.rows_frame(P, g, pool_rows=False)
        Wg = {(k.split('|')[0], float(k.split('|')[1])): np.array(v) for k, v in W[g].items()}
        for r in V.itertuples():
            out.append(dict(evaluation_group=g, sheet=r.sheet, v178_row=int(r.row), direction=r.direction, horizon_s=float(r.horizon_s),
                            has_anchor=bool(r.has_anchor), x=X[int(r.row)], w=Wg.get((r.direction, 2. if r.horizon_s == 2. else 10.)), m2=r.m2))
    return pd.DataFrame(out)


def powers(M, margins, pool):
    args = [('M3' if r.has_anchor else 'M2', r.x, r.horizon_s, r.direction, float(m), r.w, r.m2) for r, m in zip(M.itertuples(), margins)]
    return np.array(pool.map(V199.power, args, chunksize=4), dtype=float)


def run_main(B, groups_all, tau, short_eval, short_cal):
    """V255 main configuration (hybrid, per class, out-of-sample) with the short class given explicitly."""
    K = V5.classes(B, 'hybrid', 'class'); parts, scales = [], {}
    for c in V5.CELLS:
        ev = B[B.evaluation_group == c].copy(); ev['cls'] = K[ev.index]
        ev['central'] = np.where(ev.cls == 'short', short_eval.reindex(ev.index), ev[f'sop_{c}']); ev['pred'] = np.nan
        for k in sorted(ev.cls.unique()):
            p, t = short_cal if k == 'short' else V5.calibration(B, K, c, k, 'out')
            s, floor = V5.choose(p, t, tau); scales[f'{c}|{k}'] = dict(scale=s, floor_bound=floor, calibration_rows=int(len(p)))
            ev.loc[ev.cls == k, 'pred'] = s * ev.loc[ev.cls == k, 'central']
        parts.append(ev)
    J = pd.concat(parts); assert len(J) == 672 and J.pred.notna().all()
    summ, G = V5.group_table(J, groups_all)
    return J, summ, G, scales


def main():
    OUT.mkdir(exist_ok=True); res = {}
    B = V5.load(); groups_all = B[V5.C1.GK].drop_duplicates().sort_values(V5.C1.GK).reset_index(drop=True)
    M = model_rows().merge(B[['sheet', 'v178_row', 'row_id', 'truth', 'soc_point', 'temp_C']], on=['sheet', 'v178_row'], how='inner')
    assert len(M) == 672, len(M)
    D = pd.read_csv(HERE / 'relax_transfer_v242/predictions_primary_k_cross.csv'); D = D[D.regime == 'voltage_limited']
    D = D[['sheet', 'row', 'pred_W', 'v242_pred_W', 'power_W', 'margin_V', 'method']].rename(columns={'row': 'v178_row'})
    M = M.merge(D, on=['sheet', 'v178_row'], how='left'); assert len(M) == 672
    pool = Pool(8)
    # the 13 rows whose V251 central was the margin-derated command power
    F = M[M.v242_pred_W.isna() & M.pred_W.isna()].copy(); print('rows with a derated V251 central:', len(F), F.groupby(['evaluation_group', 'method']).size().to_dict(), flush=True)
    F['repro_W'] = powers(F, F.margin_V, pool); F['central_W'] = powers(F, np.zeros(len(F)), pool)
    err = float(np.max(np.abs(F.repro_W - F.power_W))); res['reproduction_13_rows_max_abs_W'] = err
    print(f'reproduction of power_W on the 13 rows: max |diff| = {err:.3g} W', flush=True)
    if not err <= 1e-6:
        dump(OUT / 'summary.json', res); raise SystemExit('13-row reproduction failed; stopping (protocol)')
    F[['row_id', 'evaluation_group', 'direction', 'temp_C', 'horizon_s', 'soc_point', 'method', 'margin_V', 'truth', 'power_W', 'central_W']].assign(
        power_over_truth=F.power_W / F.truth, central_over_truth=F.central_W / F.truth).to_csv(OUT / 'rows_13.csv', index=False)
    # short-class calibration rows: anchored A discharge rows, M3 without margin, fold weights of their own cell (out-of-sample)
    A = M[M.evaluation_group.isin(['A1', 'A17']) & (M.direction == 'discharge') & M.has_anchor].copy()
    A['central_W'] = powers(A, np.zeros(len(A)), pool)
    Vp = pd.read_csv(HERE / 'cross_sop_v199/predictions.csv'); Vp = Vp[Vp.margin_stat == 'max_V'][['sheet', 'row', 'margin_V', 'power_W']].rename(
        columns={'row': 'v178_row', 'margin_V': 'v199_margin', 'power_W': 'v199_power'})
    A = A.merge(Vp, on=['sheet', 'v178_row'], how='left'); A['v199_repro'] = powers(A, A.v199_margin, pool)
    res['extra_check_A_rows_vs_V199_max_abs_W'] = float(np.max(np.abs(A.v199_repro - A.v199_power)))
    print(f"extra check, A calibration rows vs V199 predictions: max |diff| = {res['extra_check_A_rows_vs_V199_max_abs_W']:.3g} W", flush=True)
    A[['row_id', 'evaluation_group', 'temp_C', 'soc_point', 'truth', 'central_W']].assign(ratio=A.central_W / A.truth).to_csv(OUT / 'calibration_short.csv', index=False)
    pool.close()
    short_ids = B.index[~B.in_events]; assert len(short_ids) == 3
    cen = F.set_index('row_id').central_W
    new_eval = pd.Series(np.nan, index=B.index); new_eval[short_ids] = cen.reindex(B.row_id[short_ids]).to_numpy()
    old_eval = pd.Series(np.nan, index=B.index); old_eval[short_ids] = B.v242[short_ids].to_numpy()
    mA = B.evaluation_group.isin(['A1', 'A17']) & (B.direction == 'discharge')
    old_cal = (B.v242[mA].to_numpy(), B.truth[mA].to_numpy()); new_cal = (A.central_W.to_numpy(), A.truth.to_numpy())
    S = B.loc[short_ids, ['row_id', 'soc_point', 'truth', 'v242']].assign(central_new=new_eval[short_ids])
    S['old_over_truth'] = S.v242 / S.truth; S['new_over_truth'] = S.central_new / S.truth; S.to_csv(OUT / 'rows_short.csv', index=False)
    print(S.round(4).to_string(index=False), flush=True)
    T6 = pd.read_csv(HERE / 'closedloop_v256/table.csv'); T6 = T6[T6.config == V5.MAIN].set_index('tau')
    rows, repro = [], []
    B2 = B.copy(); B2.loc[B2.row_id.isin(F.row_id), 'v242'] = F.set_index('row_id').central_W.reindex(B2.row_id[B2.row_id.isin(F.row_id)]).to_numpy()
    for tau in TAUS:
        tau = float(tau)
        _, so, Go, _ = run_main(B, groups_all, tau, old_eval, old_cal)
        ok = bool(int(Go.passed.sum()) == int(T6.at[tau, 'groups_pass']) and abs(so['int_rate'] - T6.at[tau, 'int_rate']) < 1e-12); repro.append(ok)
        J, s, G, sc = run_main(B, groups_all, tau, new_eval, new_cal)
        g2 = G[(G.evaluation_group == 'B') & (G.horizon_s == 2.)].iloc[0]
        sh = J.loc[short_ids]
        rows.append(dict(tau=tau, config='main_new_short', groups_pass=int(G.passed.sum()), int_rate=s['int_rate'], util_q50=s['util_q50'],
                         short_scale=sc['B|short']['scale'], short_interventions=int(g2.interventions), short_util_q50=float(g2.util_q50),
                         short_pred_over_truth=' '.join(f'{v:.3f}' for v in (sh.pred / sh.truth))))
        for name, BB in (('V242_old', B), ('V242_corrected', B2)):
            r = V6.v242(BB, groups_all, tau)
            rows.append(dict(tau=tau, config=name, groups_pass=r['groups_pass'], int_rate=r['held_out']['int_rate'], util_q50=r['held_out']['util_q50'],
                             failed='; '.join(r['failed'])))
        if tau == 0.05:
            d, Jd, _ = V5.deliver(J); res['delivery_tau5_short_rows'] = json.loads(Jd[Jd.row_id.isin(B.row_id[short_ids])][
                ['row_id', 'pred', 'truth', 'intervened', 'del_over_req', 'del_over_truth']].to_json(orient='records'))
    res['reproduction_V256_main_all_tau'] = bool(all(repro)); print('reproduction of V256 main with the old short class:', all(repro), flush=True)
    if not all(repro):
        dump(OUT / 'summary.json', res); raise SystemExit('V256 not reproduced; stopping (protocol)')
    Tt = pd.DataFrame(rows); Tt.to_csv(OUT / 'table.csv', index=False)
    print(Tt.drop(columns=['failed'], errors='ignore').round(4).to_string(index=False), flush=True)
    m = Tt[Tt.config == 'main_new_short'].groups_pass
    v = '15/15 (τ 범위 전체)' if (m == 15).all() else ('15/15 (일부 τ)' if (m == 15).any() else '변화 없음')
    res['verdict_main'] = v
    res['summary'] = {c: dict(groups_min=int(q.groups_pass.min()), groups_median=float(q.groups_pass.median()), groups_max=int(q.groups_pass.max()),
                              int_rate_max=float(q.int_rate.max())) for c, q in Tt.groupby('config')}
    dump(OUT / 'summary.json', res); print('verdict (main):', v); print(json.dumps(res['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
