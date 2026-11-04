"""Independent check of V280 (post hoc, after the main run): recompute the inner scores, the selected tau and the scores of
the saved requests with separately written code (plain loops; only the calibration rows R_c come from the shared pipeline,
whose equality with strictfull_v273.cal_free_single was gate N2). usage: python verify/independent.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

V = Path(__file__).resolve().parent; OUT = V.parent; HERE = OUT.parent
sys.path.insert(0, str(HERE))
import nestedtau_v280 as N  # noqa: E402

S_GRID = [i / 100 for i in range(150, 19, -1)]                 # 1.50 .. 0.20, descending


def scale(p, t, tau):
    for s in S_GRID:
        if sum(1 for a, b in zip(p, t) if s * a > b + 1e-9) / len(p) <= tau:
            return s
    return S_GRID[-1]


def groups(rows, pred):
    """closed-loop and original statistics per (cell, direction, temp, horizon)"""
    g = {}
    for (cell, d, T, h, t), q in zip(rows, pred):
        g.setdefault((cell, d, T, h), []).append(q / t)
    out = {}
    for k, rs in g.items():
        n = len(rs); it = sum(1 for r in rs if r > 1 + 1e-9)
        u = float(np.median([min(r, 1.0) for r in rs])); m = float(np.median(rs))
        out[k] = dict(n=n, closed=(u >= 0.80 and it / n <= 0.10), original=(it == 0 and m >= 0.80), util=u, interventions=it)
    return out


def main():
    B0t = N.V5.load(); src = N.I6.sources(); X = N.baseline('P', src, B0t, 'P_verify')
    T = pd.read_csv(OUT / 'inner_scores.csv'); T = T[T.baseline == 'P']
    Sel = pd.read_csv(OUT / 'selected.csv'); Sel = Sel[Sel.baseline == 'P'].set_index(['grid', 'cell'])
    bad = []
    for g, grid in N.GRIDS.items():
        for c in N.CELLS:
            R = N.cal_rows(X, c); score = {}
            for tau in grid:
                tot = 0
                for k in sorted(set(R.evaluation_group)):
                    for v in sorted(set(R.evaluation_group)):
                        if k == v:
                            continue
                        K, W = R[R.evaluation_group == k], R[R.evaluation_group == v]
                        s = scale(K.p.tolist(), K.truth.tolist(), tau)
                        G = groups(list(zip(W.evaluation_group, W.direction, W.temp_C, W.horizon_s, W.truth)), (s * W.p).tolist())
                        npass = sum(x['closed'] for x in G.values()); tot += npass
                        ref = T[(T.grid == g) & (T.cell == c) & (np.abs(T.tau - tau) < 1e-12) & (T.cal == k) & (T.val == v)]
                        if len(ref) != 1 or abs(ref.scale.iloc[0] - s) > 1e-12 or int(ref.passed.iloc[0]) != npass:
                            bad.append(f'inner {g} {c} {tau} {k}->{v}')
                score[tau] = tot
            best = max(score.values()); tied = [t for t, v in score.items() if v == best]
            tau_c = min(tied, key=lambda t: (round(abs(t - 0.05), 6), t))
            if abs(Sel.loc[(g, c), 'tau'] - tau_c) > 1e-12:
                bad.append(f'selected {g} {c}: {tau_c} vs {Sel.loc[(g, c), "tau"]}')
            print(f'{g:4s} {c:3s} tau {100 * tau_c:.2f} %  inner best {best}  tied {[round(100 * t, 2) for t in tied]}')
    rows = list(zip(B0t.evaluation_group, B0t.direction, B0t.temp_C, B0t.horizon_s, B0t.truth))
    files = {'fixed5 (V278 B0)': HERE / 'compare_v278/P/requests_B0.csv', 'main': OUT / 'requests_P_main.csv', 'wide': OUT / 'requests_P_wide.csv'}
    req = {}
    for name, f in files.items():
        p = pd.read_csv(f, float_precision='round_trip').set_index('row_id').pred.reindex(B0t.row_id); req[name] = p
        G = groups(rows, p.tolist())
        print(f'{name:16s} closed loop {sum(x["closed"] for x in G.values())}/15  original {sum(x["original"] for x in G.values())}/15  groups {len(G)}')
    a1 = (B0t.evaluation_group == 'A1').to_numpy()
    r = (req['main'] / req['fixed5 (V278 B0)']).to_numpy()
    print('main / fixed5: A1 rows ratio min/max', r[a1].min(), r[a1].max(), '| other rows max |ratio - 1|', float(np.abs(r[~a1] - 1).max()))
    print('mismatches:', bad or 'none')


if __name__ == '__main__':
    main()
