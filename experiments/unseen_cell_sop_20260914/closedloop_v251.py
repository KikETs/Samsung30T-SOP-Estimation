"""V251 (PROTOCOL_CLOSEDLOOP_V251.md): existing SOP predictions re-scored in a closed-loop frame (an ideal voltage clamp turns
an over-request into an intervention). Descriptive; no model is fitted. usage: python closedloop_v251.py"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_envelope import HERE

OUT = HERE / 'closedloop_v251'; SRC = OUT / 'src'
GK = ['evaluation_group', 'direction', 'temp_C', 'horizon_s']; CELLS = ('A1', 'A17', 'B')
S_GRID = np.round(np.arange(0.50, 1.501, 0.01), 2)


def region(direction, soc):                       # tools/score_loco.py of sop_ident
    if direction == 'discharge':
        return 'dis<=0.1' if soc <= 0.1 + 1e-9 else ('dis0.15-0.3' if soc <= 0.3 + 1e-9 else 'dis>=0.4')
    return 'chg<=0.7' if soc <= 0.7 + 1e-9 else 'chg>=0.8'


def v_series(path, col=None):
    D = pd.read_csv(HERE / path); J = D[D.regime == 'voltage_limited'].copy()
    cen = J.pred_W.where(J.pred_W.notna(), J.power_W)
    if col is not None:
        cen = J[col].where(J[col].notna(), cen)
    return pd.DataFrame({**{k: J[k].to_numpy() for k in GK}, 'truth': J.truth_power_W.to_numpy(), 'deployed': J.power_W.to_numpy(), 'central': cen.to_numpy()})


def sop_ident(name):
    F = pd.read_csv(SRC / f'folds_{name}.csv')
    B = pd.read_csv(SRC / 'bench_rows.csv')[['row_id', 'evaluation_group', 'direction', 'temp_C', 'horizon_s', 'soc_point', 'truth_power_W']]
    F = F.merge(B, on='row_id'); F['ratio'] = F.pred_power_W / F.truth_power_W
    F['region'] = [region(d, s) for d, s in zip(F.direction, F.soc_point)]
    parts = []
    for c in CELLS:
        Fc = F[F.fold == c]; tr = Fc[Fc.evaluation_group != c]; te = Fc[Fc.evaluation_group == c]
        kr = tr.groupby('region').ratio.max().clip(lower=1.0); k = te.region.map(kr).fillna(float(tr.ratio.max())).to_numpy()
        parts.append(pd.DataFrame({**{g: te[g].to_numpy() for g in GK}, 'truth': te.truth_power_W.to_numpy(),
                                   'deployed': te.pred_power_W.to_numpy() / k, 'ensemble': te.pred_power_W.to_numpy(), 'central': np.exp(te.m.to_numpy())}))
    return pd.concat(parts, ignore_index=True)


def open_loop(J, col):
    r = J[col] / J.truth; G = J.assign(over=r > 1 + 1e-9, use=r).groupby(GK).agg(exceed=('over', 'sum'), use=('use', 'median'))
    return int(((G.exceed == 0) & (G.use >= 0.8)).sum()), int(G.exceed.sum())


def closed_loop(J, pred):
    r = pred / J.truth.to_numpy(); it = r > 1 + 1e-9; util = np.minimum(r, 1.0)
    G = J.assign(it=it, util=util, depth=np.where(it, r - 1, np.nan)).groupby(GK).agg(n=('it', 'size'), interventions=('it', 'sum'),
                                                                                      util_q50=('util', 'median'), depth_max=('depth', 'max')).reset_index()
    G['int_rate'] = G.interventions / G.n
    d = r[it] - 1
    summ = dict(int_rate=float(it.mean()), depth_q50=float(np.median(d)) if len(d) else 0.0, depth_q95=float(np.quantile(d, .95)) if len(d) else 0.0,
                depth_max=float(d.max()) if len(d) else 0.0, util_q50=float(np.median(util)), within11=float(np.mean(np.abs(r - 1) <= .11)),
                groups_util80_int5=int(((G.util_q50 >= 0.8) & (G.int_rate <= 0.05)).sum()),
                groups_util80_int10=int(((G.util_q50 >= 0.8) & (G.int_rate <= 0.10)).sum()),
                groups_util80_int0=int(((G.util_q50 >= 0.8) & (G.interventions == 0)).sum()))
    return summ, G


def main():
    OUT.mkdir(exist_ok=True)
    M = {'V201': v_series('relax_transfer_v201/predictions.csv'),
         'V242': v_series('relax_transfer_v242/predictions_primary_k_cross.csv', 'v242_pred_W'),
         'V245R': v_series('anchor_trend_v245/predictions_R.csv', 'v245_pred_W'),
         'sop_E3causal': sop_ident('E3causal'), 'sop_E2noprev': sop_ident('E2noprev'), 'sop_E2direction': sop_ident('E2direction')}
    res, curves, groups = {}, [], []
    for name, J in M.items():
        assert len(J) == 672, (name, len(J))
        res[name] = {'open_loop_deployed (pass, exceed)': open_loop(J, 'deployed')}
        for var in [v for v in ('deployed', 'ensemble', 'central') if v in J]:
            s, G = closed_loop(J, J[var].to_numpy()); res[name][var] = s
            groups.append(G.assign(method=name, variant=var))
        for sc in S_GRID:
            s, _ = closed_loop(J, sc * J.central.to_numpy()); curves.append(dict(method=name, scale=float(sc), **s))
    C = pd.DataFrame(curves); C.to_csv(OUT / 'tradeoff_curves.csv', index=False); pd.concat(groups).to_csv(OUT / 'groups.csv', index=False)
    for name in M:
        c = C[C.method == name]
        for lim in (0.05, 0.10):
            x = c[c.int_rate <= lim]
            res[name][f'at_int_rate_le_{int(lim * 100)}pct'] = (dict(scale=float(x.scale.max()), util_q50=float(x.loc[x.scale.idxmax(), 'util_q50']),
                                                                     groups_util80_int5=int(x.loc[x.scale.idxmax(), 'groups_util80_int5']),
                                                                     groups_util80_int10=int(x.loc[x.scale.idxmax(), 'groups_util80_int10']))
                                                                if len(x) else None)
    json.dump(res, open(OUT / 'summary.json', 'w'), indent=1)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name in M:
        c = C[C.method == name]; ax.plot(100 * c.int_rate, 100 * c.util_q50, label=name, lw=1.6)
        d = res[name]['deployed']; ax.scatter(100 * d['int_rate'], 100 * d['util_q50'], s=25)
    ax.axhline(80, color='gray', lw=0.8, ls='--'); ax.axvline(5, color='gray', lw=0.8, ls=':'); ax.axvline(10, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('intervention rate, % of 672 rows (prediction > truth)'); ax.set_ylabel('median utilisation, % (min(pred/truth, 1))')
    ax.set_xlim(0, 60); ax.set_ylim(40, 100); ax.legend(fontsize=8); ax.set_title('closed-loop trade-off (common scale on the central prediction; dots: deployed)')
    fig.tight_layout(); fig.savefig(OUT / 'tradeoff.png', dpi=130)
    for name, r in res.items():
        print(name, 'open-loop', r['open_loop_deployed (pass, exceed)'])
        for var in ('deployed', 'ensemble', 'central'):
            if var in r:
                s = r[var]; print(f'   {var:9s} int {100 * s["int_rate"]:5.1f}%  depth q50/q95/max {s["depth_q50"]:.3f}/{s["depth_q95"]:.3f}/{s["depth_max"]:.3f}  util {100 * s["util_q50"]:.1f}%  '
                                  f'±11% {100 * s["within11"]:.1f}%  groups util80 & int0/5/10%: {s["groups_util80_int0"]}/{s["groups_util80_int5"]}/{s["groups_util80_int10"]}')
        print('   at int<=5%:', r['at_int_rate_le_5pct'], ' at int<=10%:', r['at_int_rate_le_10pct'])


if __name__ == '__main__':
    main()
