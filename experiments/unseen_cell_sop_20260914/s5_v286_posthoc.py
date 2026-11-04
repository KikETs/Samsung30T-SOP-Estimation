"""V286 post hoc (descriptive, not in the registered verdicts): tau 4-6 % (V256 range) and the V284 (gamma, tau*) grid for
L and F (the documented fallback order), B0 and AD requests on the 15 Samsung groups.
usage: python s5_v286_posthoc.py   (torch_env; after the V286 main run)"""
import numpy as np, pandas as pd
import s5_v286 as M
from run_envelope import dump

OUT = M.OUT
Bm = M.V5.load(); groups_all = Bm[M.GK].drop_duplicates().sort_values(M.GK).reset_index(drop=True)
D = M.CV.table_from('base', *M.I6.sources()); J = M.bench_J(Bm)
XL = M.build_A(D, Bm)
with M.fallback_fixed():
    XF = M.build_A(D, Bm)
rows, grid = [], []
for name, X in (('L', XL), ('F', XF)):
    fold = {(f, r): float(np.exp(m)) for f, r, m in zip(X['C'].fold, X['C'].row_id, X['C'].m)}
    Jc = J.copy(); Jc['central'] = [fold[(c, r)] for c, r in zip(Jc.evaluation_group, Jc.row_id)]
    for tau in [float(t) for t in M.V6.TAUS] if hasattr(M, 'V6') else []:
        pass
    import closedloop_v256 as V6
    for tau in [float(t) for t in V6.TAUS]:
        b0, sc = M.CV.arm_B0(X, tau, groups_all); s0 = {c: float(sc[c]) for c in M.CELLS}
        ad, _, _ = M.A4.adapt(Jc, s0, M.A4.GAMMA, M.A4.TAU, 'cell')
        for pol, req in (('B0', b0.reindex(Jc.row_id).to_numpy(float)), ('AD', ad)):
            sm, G = M.score_A(Jc, req)
            rows.append(dict(base=name, tau=tau, policy=pol, s0=str(s0), **{k: sm[k] for k in ('closed_loop', 'original', 'exceed_rows', 'int_rate', 'util_q50', 'depth_max')},
                             failed='; '.join(f'{x.evaluation_group} {x.direction} {x.temp_C} {x.horizon_s:g}s' for x in G[~G.passed].itertuples())))
    b0, sc = M.CV.arm_B0(X, 0.05, groups_all); s0 = {c: float(sc[c]) for c in M.CELLS}
    for g in M.A4.GRID_G:
        for t in M.A4.GRID_T:
            ad, _, _ = M.A4.adapt(Jc, s0, g, t, 'cell'); sm, _ = M.score_A(Jc, ad)
            grid.append(dict(base=name, gamma=g, tau_star=t, **{k: sm[k] for k in ('closed_loop', 'original', 'exceed_rows', 'int_rate', 'util_q50', 'depth_max')}))
T = pd.DataFrame(rows); T.to_csv(OUT / 'posthoc_tau_range.csv', index=False)
Gd = pd.DataFrame(grid); Gd.to_csv(OUT / 'posthoc_adapt_grid.csv', index=False)
pd.set_option('display.width', 250)
print(T.drop(columns=['s0']).to_string(index=False))
print(Gd.pivot_table(index=['gamma', 'tau_star'], columns='base', values='closed_loop').to_string())
summ = {f'{b} {p}': [int(T[(T.base == b) & (T.policy == p)].closed_loop.min()), float(T[(T.base == b) & (T.policy == p)].closed_loop.median()), int(T[(T.base == b) & (T.policy == p)].closed_loop.max())]
        for b in ('L', 'F') for p in ('B0', 'AD')}
print('tau 4-6 % (min / median / max):', summ)
dump(OUT / 'posthoc.json', dict(note='descriptive, after the registered run; not used for the verdicts', tau_4_6=summ))
