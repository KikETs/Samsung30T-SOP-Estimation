"""V267 (PROTOCOL_LIGHT_V267.md): closed-loop result of the light ensemble (log median of the five local predictors prev, L4,
M6, DN6, G1; no physics identification) with the V256 main-configuration code (closedloop_v255.run, hybrid, classes from
the other cells' out-of-sample predictions, tau = 4-6 %). The V256 main result (E3causal, 14/14/14) must be reproduced
first. usage: python light_v267.py"""
import json
import numpy as np, pandas as pd
from run_envelope import HERE, dump
import closedloop_v255 as V5, closedloop_v256 as V6

OUT = HERE / 'light_v267'; SRC = OUT / 'src'
NAMES = ('prev', 'L4', 'M6', 'DN6', 'G1')


def light_folds():
    out = []
    for c in V5.CELLS:
        F = None
        for n in NAMES:
            P = pd.read_csv(SRC / f'local_{n}' / 'folds.csv'); P = P[P.fold == c][['row_id', 'pred_power_W']].rename(columns={'pred_power_W': n})
            F = P if F is None else F.merge(P, on='row_id', how='outer')
        L = np.log(F[list(NAMES)].to_numpy(float)); L[~np.isfinite(L)] = np.nan
        with np.errstate(all='ignore'):
            F['m'] = np.nanmedian(L, axis=1)
        F['n_local'] = np.isfinite(L).sum(1); F['fold'] = c; out.append(F[['fold', 'row_id', 'm', 'n_local']])
    return pd.concat(out)


def sweep(B, groups_all, configs):
    rows, res = [], {}
    for name, predictor, kind in configs:
        per = []
        for tau in V6.TAUS:
            r, G, J = V5.run(B, groups_all, name, predictor, kind, 'out', float(tau)); per.append(r)
            rows.append(dict(config=name, tau=float(tau), groups_pass=r['groups_pass'], int_rate=r['held_out']['int_rate'], util_q50=r['held_out']['util_q50']))
            if abs(tau - 0.05) < 1e-9:
                G.to_csv(OUT / f'groups_tau5_{name}.csv', index=False); J.to_csv(OUT / f'rows_tau5_{name}.csv', index=False); res[f'{name}_tau5'] = r
        gp = [p['groups_pass'] for p in per]; res[f'{name}_groups_tau_4_6'] = [int(min(gp)), float(np.median(gp)), int(max(gp))]
    return rows, res


def central_stats(B, col_fmt):
    """Open-loop statistics of the central prediction of the in-event rows, each row from its own fold."""
    ev = B[B.in_events]; cen = np.array([ev.at[i, col_fmt.format(g)] for i, g in zip(ev.index, ev.evaluation_group)])
    r = cen / ev.truth.to_numpy(); lr = np.log(r); q05, q50, q95 = np.quantile(lr, [.05, .5, .95])
    return dict(rows=int(len(r)), rel_abs_median_pct=float(100 * np.median(np.abs(r - 1))), rel_abs_q95_pct=float(100 * np.quantile(np.abs(r - 1), .95)),
                within_11pct=float(np.mean(np.abs(r - 1) <= .11)), bias=float(np.exp(q50)), over_tail=float(np.exp(q95 - q50)), under_tail=float(np.exp(q50 - q05)))


def main():
    OUT.mkdir(exist_ok=True); B = V5.load()
    groups_all = B[V5.C1.GK].drop_duplicates().sort_values(V5.C1.GK).reset_index(drop=True); assert len(groups_all) == 15
    # reproduction of the V256 main configuration with the E3causal centre
    _, ref = sweep(B, groups_all, [(V5.MAIN, 'hybrid', 'class')])
    rep = ref[f'{V5.MAIN}_groups_tau_4_6']; print('reproduction of V256 main:', rep, flush=True)
    if rep != [14, 14.0, 14]:
        raise SystemExit('V256 main configuration not reproduced; stopping (protocol)')
    full_stats = central_stats(B, 'sop_{}')
    # light centre
    Fl = light_folds(); Fl.to_csv(OUT / 'light_folds.csv', index=False)
    P = Fl.pivot(index='row_id', columns='fold', values='m').reindex(B.row_id)
    for c in V5.CELLS:
        B[f'sop_{c}'] = np.exp(P[c].to_numpy())
    miss = B.in_events & B[[f'sop_{c}' for c in V5.CELLS]].isna().any(axis=1)
    if miss.any():
        raise SystemExit(f'{int(miss.sum())} in-event rows without any local prediction; stopping (protocol)')
    rows, res = sweep(B, groups_all, [('light_hybrid_class_out', 'hybrid', 'class'), ('light_hybrid_two_out', 'hybrid', 'two')])
    pd.DataFrame(rows).to_csv(OUT / 'tau_range.csv', index=False)
    med = res['light_hybrid_class_out_groups_tau_4_6'][1]
    res.update(reproduction_v256_main=rep, reference_full_groups_tau_4_6=[14, 14, 14],
               verdict='가벼운 판으로 충분' if med >= 14 else ('가벼운 판 근접(1~2그룹 손실)' if med >= 12 else '물리 식별 필요'),
               central_open_loop=dict(full=full_stats, light=central_stats(B, 'sop_{}')),
               light_rows_missing_local_all=int((Fl.groupby('row_id').n_local.min() == 0).sum()))
    dump(OUT / 'summary.json', res)
    for k in ('light_hybrid_class_out', 'light_hybrid_two_out'):
        t = res[f'{k}_tau5']
        print(f"{k}: groups tau4-6 {res[f'{k}_groups_tau_4_6']} | tau5 int {100 * t['held_out']['int_rate']:.2f}% util {100 * t['held_out']['util_q50']:.1f}% "
              f"failed {t['failed']}", flush=True)
    print('verdict:', res['verdict'], '| central full', json.dumps(full_stats), '| light', json.dumps(res['central_open_loop']['light']), flush=True)


if __name__ == '__main__':
    main()
