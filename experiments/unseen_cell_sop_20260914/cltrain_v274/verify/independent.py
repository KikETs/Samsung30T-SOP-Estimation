"""Independent re-scoring of V274 N274 at every tau from the saved centres only (no project scoring code): one scale per
evaluated cell, calibration rows = the other cells' rows (discharge: centre of the third-cell model; charge: fold-c
centre), group pass = median min(pred/truth, 1) >= 0.80 and share pred > truth + 1e-9 <= 0.10."""
from pathlib import Path
import numpy as np, pandas as pd
E = str(Path(__file__).resolve().parents[2]) + '/'
B = pd.read_csv(E + 'closedloop_v251/src/bench_rows.csv')[['row_id', 'evaluation_group', 'direction', 'temp_C', 'horizon_s', 'truth_power_W']]
C = pd.read_csv(E + 'cltrain_v274/centres_full.csv'); S = pd.read_csv(E + 'cltrain_v274/strict_full.csv')
fold = {(f, r): np.exp(m) for f, r, m in zip(C.fold, C.row_id, C.m)}
strict = {(e, r): np.exp(m) for e, r, m in zip(S.train_cell, S.row_id, S.m)}
cells = ['A1', 'A17', 'B']; third = lambda c, d: [x for x in cells if x not in (c, d)][0]
grid = np.round(np.arange(0.20, 1.501, 0.01), 2)
T = pd.read_csv(E + 'cltrain_v274/tau_range.csv'); T = T[T.config == 'N274']
for tau in np.round(np.arange(0.04, 0.06001, 0.0025), 4):
    rows = []
    for c in cells:
        cal = B[B.evaluation_group != c]
        p = np.array([strict.get((third(c, d), r), np.nan) if k == 'discharge' else fold[(c, r)]
                      for d, r, k in zip(cal.evaluation_group, cal.row_id, cal.direction)])
        assert not (cal.evaluation_group == c).any()
        ok = np.isfinite(p); p, t = p[ok], cal.truth_power_W.to_numpy()[ok]
        s = next((x for x in grid[::-1] if np.mean(x * p > t + 1e-9) <= tau), grid[0])
        ev = B[B.evaluation_group == c].copy(); ev['pred'] = s * np.array([fold[(c, r)] for r in ev.row_id]); ev['s'] = s
        rows.append(ev)
    J = pd.concat(rows); J['ratio'] = J.pred / J.truth_power_W
    G = J.groupby(['evaluation_group', 'direction', 'temp_C', 'horizon_s']).apply(
        lambda g: pd.Series(dict(util=np.median(np.minimum(g.ratio, 1.0)), inter=np.mean(g.ratio > 1 + 1e-9))), include_groups=False)
    n = int(((G.util >= 0.80) & (G.inter <= 0.10)).sum())
    ref = int(T[np.isclose(T.tau, tau)].groups_pass.iloc[0])
    print(f'tau {tau:.4f}: independent {n:2d}  registered run {ref:2d}  scales {dict(J.groupby("evaluation_group").s.first())}', 'same' if n == ref else 'DIFFERENT')
