"""V192 M3: previous-pulse anchor x fitted monotone SOC/current/temperature transfer (PROTOCOL_EXTRAPOLATION_BENCH_V192.md, M3)."""
import numpy as np, pandas as pd
from run_envelope import HERE, dump
from nested_data_v53 import load_data, MENDELEY, CELLS
from relax_state_v36 import STATE_NAMES

OUT = HERE / 'extrapolation_bench_v192'; NM = list(STATE_NAMES); CUT = 3.40


def table():
    rows, x, t, i, y, L, ht, hi = load_data(); n = len(rows)
    h = np.array([float(r['requested_horizon_s']) for r in rows]); d = np.array([r['direction'] for r in rows]); cell = np.array([r['cell'] for r in rows])
    yend = y[np.arange(n), L - 1]; iend = np.abs(i[np.arange(n), L - 1]); T = x[:, NM.index('pre_temperature_5s_C')]
    Va = np.full(n, np.nan); ea = Va.copy(); Ia = Va.copy()
    for hh in (2, 10):
        m = h == hh
        Va[m] = x[m, NM.index(f'past_{hh}s_pre_voltage_V')]; ea[m] = x[m, NM.index(f'past_{hh}s_oriented_voltage_excursion_V')]; Ia[m] = np.abs(x[m, NM.index(f'past_{hh}s_endpoint_current_A')])
    lim = np.array([float(r['observed_voltage_limit']) if r['cell'] != MENDELEY else (2.5 if r['direction'] == 'discharge' else 4.2) for r in rows])
    head = np.where(d == 'charge', lim - x[:, 0], x[:, 0] - lim)
    ymax = np.array([np.max(y[j, 1:L[j]]) for j in range(n)])
    D = pd.DataFrame(dict(cell=cell, d=d, h=h, Vq=x[:, 0], Va=Va, ea=ea, Ia=Ia, yend=yend, iend=iend, T=T, head=head, crossed=ymax > head + 1e-9))
    D['ok'] = np.isfinite(D.Va) & np.isfinite(D.ea) & (D.ea > 1e-4) & (D.Ia > .5) & (D.iend > .5) & (D.yend > 1e-4)
    f1d = 1 / (D.Vq - 2.5) - 1 / (D.Va - 2.5); f1c = 1 / (4.25 - D.Vq) - 1 / (4.25 - D.Va)
    D['f1'] = np.where(D.d == 'discharge', f1d, f1c); D['f2'] = np.log(D.iend / D.Ia); D['f3'] = D.f1 * (1000 / (D['T'] + 273.15) - 1000 / 298.15)
    D['target'] = np.log(D.yend / D.iend) - np.log(D.ea / D.Ia); D['logRa'] = np.log(D.ea / D.Ia)
    return D


def fit_predict(D, fit_mask, ev_mask):
    out = []
    for (dd, hh), g in D[ev_mask & D.ok].groupby(['d', 'h']):
        F = D[fit_mask & D.ok & (D.d == dd) & (D.h == hh)]
        A = np.column_stack([np.ones(len(F)), F.f1, F.f2, F.f3]); w, *_ = np.linalg.lstsq(A, F.target.to_numpy(), rcond=None)
        B = np.column_stack([np.ones(len(g)), g.f1, g.f2, g.f3]); pred = g.iend * np.exp(g.logRa + B @ w)
        gg = g.copy(); gg['pred'] = pred; gg['end_err_mV'] = 1e3 * (gg.yend - pred); gg['allowance'] = gg.crossed & (pred + .02 <= gg['head'] + 1e-9)
        gg['w'] = [np.round(w, 4).tolist()] * len(gg); out.append(gg)
    return pd.concat(out) if out else pd.DataFrame()


def summarize(E, keys):
    G = E.groupby(keys).agg(n=('end_err_mV', 'size'), median=('end_err_mV', 'median'), p90=('end_err_mV', lambda s: s.quantile(.9)), p95=('end_err_mV', lambda s: s.quantile(.95)),
                            over20=('end_err_mV', lambda s: float((s > 20).mean())), med_abs=('end_err_mV', lambda s: s.abs().median()), crossed=('crossed', 'sum'), allowance=('allowance', 'sum')).reset_index()
    G['allowance_rate'] = G.allowance / G.crossed.where(G.crossed > 0); return G


def main():
    OUT.mkdir(exist_ok=True); D = table(); res = []
    for outer in CELLS:   # bench B, discharge
        fit = (D.cell != outer) & (D.d == 'discharge') & (D.Vq >= CUT) & (D.Va >= CUT)
        ev = (D.d == 'discharge') & ((D.cell == outer) | (D.cell == MENDELEY))
        E = fit_predict(D, fit, ev); E['bench'] = 'B'; E['outer'] = outer
        E['subset'] = np.where(E.cell == outer, 'outer_', 'mendeley_') + np.where(E.Vq < CUT, 'below', 'inrange'); res.append(E)
    cold = (D.cell == MENDELEY) & (D['T'] < 10)   # bench C
    E = fit_predict(D, ~cold, cold | (D.cell == MENDELEY)); E['bench'] = 'C'; E['outer'] = 'ALL6'
    E['subset'] = np.where(E['T'] < 10, 'mendeley_cold', 'mendeley_warm'); res.append(E)
    E = pd.concat(res)
    B = E[E.bench == 'B']; mend = B[B.cell == MENDELEY].groupby(['d', 'h', 'Vq', 'yend', 'iend']).head(1)   # Mendeley rows appear in all 6 folds: keep one
    E2 = pd.concat([B[B.cell != MENDELEY], mend, E[E.bench == 'C']])
    E2.drop(columns=['w']).to_csv(OUT / 'bench_m3_rows.csv', index=False)
    G = summarize(E2.rename(columns={'d': 'direction', 'h': 'horizon_s'}), ['bench', 'direction', 'subset', 'horizon_s']); G['method'] = 'M3'
    G.to_csv(OUT / 'bench_m3_summary.csv', index=False)
    missing = int((~D.ok).sum())
    dump(OUT / 'bench_m3_verification.json', dict(rows_without_anchor=missing, total_rows=len(D), training_data_only=True, adopted=False, goal_complete=False))
    pd.set_option('display.width', 220); print('rows without usable anchor:', missing, 'of', len(D)); print(G.round(3).to_string(index=False))
    print('bench C weights:', E[E.bench == 'C'].groupby(['d', 'h']).w.first().to_dict())


if __name__ == '__main__':
    main()
