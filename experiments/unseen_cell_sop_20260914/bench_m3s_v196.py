"""V196: SOC-step pairs (anchor = max-current pulse of the previous SOC point) and the M3s transfer (PROTOCOL_SOC_STEP_PAIRS_V196.md)."""
import json
import numpy as np, pandas as pd
from run_envelope import HERE, dump
from nested_data_v53 import load_data, MENDELEY, CELLS
from relax_state_v36 import STATE_NAMES

OUT = HERE / 'soc_step_pairs_v196'; TC = list(STATE_NAMES).index('pre_temperature_5s_C'); CUT = 3.40


def pairs():
    rows, x, t, i, y, L, ht, hi = load_data(); n = len(rows)
    M = pd.DataFrame(dict(cell=[r['cell'] for r in rows], f=[r.get('source_file') for r in rows], seg=[r.get('clock_segment') for r in rows], d=[r['direction'] for r in rows],
                          h=[float(r['requested_horizon_s']) for r in rows], ts=pd.to_numeric(pd.Series([r.get('start_time_s') for r in rows]), errors='coerce'),
                          V=x[:, 0], T=x[:, TC], I=np.abs(i[np.arange(n), L - 1]), e=y[np.arange(n), L - 1]))
    lim = np.array([float(r['observed_voltage_limit']) if r['cell'] != MENDELEY else (2.5 if r['direction'] == 'discharge' else 4.2) for r in rows])
    M['head'] = np.where(M.d == 'charge', lim - M.V, M.V - lim); M['crossed'] = [bool(np.max(y[j, 1:L[j]]) > M['head'].iloc[j] + 1e-9) for j in range(n)]
    out = []
    for _, g in M.dropna(subset=['f', 'ts']).groupby(['cell', 'f', 'seg', 'd', 'h']):
        g = g.sort_values('ts'); step = (g.V.diff().abs() > .02) | (g.ts.diff() > 3000); g = g.assign(k=step.cumsum())
        for k, q in g.groupby('k'):
            prev = g[g.k == k - 1]
            if not len(prev): continue
            a = prev.loc[prev.I.idxmax()]
            for j, r in q.iterrows():
                out.append(dict(row=j, cell=r.cell, d=r.d, h=r.h, Vq=r.V, Va=a.V, Iq=r.I, Ia=a.I, exq=r.e, exa=a.e, T=r['T'], head=r['head'], crossed=r.crossed))
    P = pd.DataFrame(out); P = P[(P.Iq > .5) & (P.Ia > .5) & (P.exq > 1e-4) & (P.exa > 1e-4)].copy()
    P['f1'] = np.where(P.d == 'discharge', 1 / (P.Vq - 2.5) - 1 / (P.Va - 2.5), 1 / (4.25 - P.Vq) - 1 / (4.25 - P.Va))
    P['f2'] = np.log(P.Iq / P.Ia); P['f3'] = P.f1 * (1000 / (P['T'] + 273.15) - 1000 / 298.15)
    P['target'] = np.log(P.exq / P.Iq) - np.log(P.exa / P.Ia); P['logRa'] = np.log(P.exa / P.Ia)
    return P


def design(P, quad):
    cols = [np.ones(len(P)), P.f1, P.f2, P.f3] + ([P.f1 ** 2, P.f2 ** 2, P.f1 * P.f2] if quad else [])
    return np.column_stack(cols)


def fit(P):
    return {(dd, hh): np.linalg.lstsq(design(g, True), g.target.to_numpy(), rcond=None)[0] for (dd, hh), g in P.groupby(['d', 'h'])}


def predict(P, W, quad=True):
    out = np.full(len(P), np.nan)
    for (dd, hh), g in P.groupby(['d', 'h']):
        w = W.get((dd, hh))
        if w is None: continue
        pos = P.index.get_indexer(g.index); out[pos] = g.Iq * np.exp(g.logRa + design(g, quad) @ w)
    return out


def summarize(E):
    return E.groupby(['bench', 'method', 'd', 'region', 'h']).agg(n=('err', 'size'), median=('err', 'median'), p95=('err', lambda s: s.quantile(.95)),
                                                               over20=('err', lambda s: float((s > 20).mean())), crossed=('crossed', 'sum'), allowance=('allowance', 'sum')).reset_index()


def main():
    OUT.mkdir(exist_ok=True); P = pairs().reset_index(drop=True); print('SOC-step pairs', len(P), P.groupby(['d', 'h']).size().to_dict(), flush=True)
    old = {tuple([k.split('|')[0], float(k.split('|')[1])]): np.array(v) for k, v in json.loads((HERE / 'bench_selected_v193/verification.json').read_text())['m3_weights'].items()}
    def region(r):
        return (('dis>=3.40' if r.Vq >= CUT else ('dis3.29-3.40' if r.Vq >= 3.29 else 'dis<3.29')) if r.d == 'discharge' else ('chg_head<0.2' if 4.2 - r.Vq < .2 else 'chg_head>=0.2'))
    P['region'] = P.apply(region, axis=1); recs = []
    def add(E, bench, method, pred):
        e = E.copy(); e['err'] = 1e3 * (e.exq - pred); e['allowance'] = e.crossed & (pred + .02 <= e['head'] + 1e-9); e['bench'] = bench; e['method'] = method; recs.append(e)
    for outer in CELLS:
        tr = P[P.cell != outer]; te = P[P.cell == outer]; add(te, 'LOCO', 'M3s', predict(te, fit(tr))); add(te, 'LOCO', 'M3_old', predict(te, old, quad=False))
    tr = P[(P.d == 'discharge') & (P.Vq >= CUT) & (P.Va >= CUT)]; te = P[(P.d == 'discharge') & (P.Vq < CUT)]
    te = te.assign(region=np.where(te.cell == MENDELEY, 'mendeley<3.40', 'uypydj<3.40')); add(te, 'lowSOC', 'M3s', predict(te, fit(tr))); add(te, 'lowSOC', 'M3_old', predict(te, old, quad=False))
    cold = (P.cell == MENDELEY) & (P['T'] < 10); te = P[cold].assign(region='mendeley_cold'); add(te, 'cold', 'M3s', predict(te, fit(P[~cold]))); add(te, 'cold', 'M3_old', predict(te, old, quad=False))
    E = pd.concat(recs); E.to_csv(OUT / 'bench_rows.csv', index=False); G = summarize(E); G.to_csv(OUT / 'bench_summary.csv', index=False)
    Wall = fit(P); dump(OUT / 'weights_all.json', {f'{k[0]}|{k[1]:g}': v.tolist() for k, v in Wall.items()})
    dump(OUT / 'bench_verification.json', dict(pairs=len(P), training_data_only=True, adopted=False, goal_complete=False))
    pd.set_option('display.width', 220); print(G.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
