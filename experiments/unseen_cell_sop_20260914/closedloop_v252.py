"""V252 (PROTOCOL_CLOSEDLOOP_V252.md): the closed-loop scale chosen from the other cells only, checked on the held-out cell.
usage: python closedloop_v252.py"""
import json
import numpy as np, pandas as pd
from run_envelope import HERE
import closedloop_v251 as C1

OUT = HERE / 'closedloop_v252'; SRC = HERE / 'closedloop_v251' / 'src'
CELLS = ('A1', 'A17', 'B'); GRID = C1.S_GRID; TAUS = (0.05, 0.10)
BASE_GROUPS = {'sop_E3causal': 14, 'sop_E2noprev': 14, 'V242': 12, 'V201': 9}   # V251, scale from the evaluation data, int <= 10 %


def sop_folds(name):
    """Every fold's central predictions of every cell (fold c: the model fitted without c)."""
    F = pd.read_csv(SRC / f'folds_{name}.csv')
    B = pd.read_csv(SRC / 'bench_rows.csv')[['row_id'] + C1.GK + ['truth_power_W']]
    F = F.merge(B, on='row_id')
    return pd.DataFrame({'fold': F.fold, **{g: F[g] for g in C1.GK}, 'truth': F.truth_power_W, 'central': np.exp(F.m)})


def v_folds(path, col):
    """V files hold one out-of-cell prediction per row; fold c's calibration rows are the other cells' rows of the same file."""
    J = C1.v_series(path, col)
    return pd.concat([J.assign(fold=c) for c in CELLS], ignore_index=True)


def choose_scale(cal, tau):
    for s in GRID[::-1]:
        if np.mean(s * cal.central.to_numpy() > cal.truth.to_numpy() + 1e-9) <= tau:
            return float(s)
    return float(GRID[0])


def main():
    OUT.mkdir(exist_ok=True); res = {}
    M = {'sop_E3causal': sop_folds('E3causal'), 'sop_E2noprev': sop_folds('E2noprev'),
         'V242': v_folds('relax_transfer_v242/predictions_primary_k_cross.csv', 'v242_pred_W'), 'V201': v_folds('relax_transfer_v201/predictions.csv', None)}
    allg = []
    for name, F in M.items():
        res[name] = {}
        for tau in TAUS:
            parts, scales = [], {}
            for c in CELLS:
                Fc = F[F.fold == c]; cal = Fc[Fc.evaluation_group != c]; ev = Fc[Fc.evaluation_group == c].copy()
                s = choose_scale(cal, tau); scales[c] = dict(scale=s, calibration_rows=int(len(cal)),
                                                            calibration_int_rate=float(np.mean(s * cal.central > cal.truth + 1e-9)))
                ev['pred'] = s * ev.central; parts.append(ev)
            J = pd.concat(parts, ignore_index=True); assert len(J) == 672, (name, len(J))
            summ, G = C1.closed_loop(J, J.pred.to_numpy())
            J2 = J.assign(deployed=J.pred); ol = C1.open_loop(J2, 'deployed')
            allg.append(G.assign(method=name, tau=tau))
            v = dict(scales=scales, held_out=summ, open_loop_pass_exceed=ol)
            if tau == 0.05:
                v['holds'] = dict(int_rate_le_10pct=bool(summ['int_rate'] <= 0.10), groups_ok=bool(summ['groups_util80_int10'] >= BASE_GROUPS[name] - 2),
                                  baseline_groups=BASE_GROUPS[name])
                v['holds']['closed_loop_result_holds'] = bool(v['holds']['int_rate_le_10pct'] and v['holds']['groups_ok'])
            res[name][f'tau_{int(tau * 100)}pct'] = v
            print(f"{name:13s} tau {int(tau * 100):2d}%  scales {[round(scales[c]['scale'], 2) for c in CELLS]}  held-out int {100 * summ['int_rate']:.1f}%  "
                  f"depth q95/max {summ['depth_q95']:.3f}/{summ['depth_max']:.3f}  util {100 * summ['util_q50']:.1f}%  "
                  f"groups util80&int0/5/10%: {summ['groups_util80_int0']}/{summ['groups_util80_int5']}/{summ['groups_util80_int10']}  open-loop {ol}"
                  + (f"  holds={v['holds']['closed_loop_result_holds']}" if tau == 0.05 else ''), flush=True)
    pd.concat(allg).to_csv(OUT / 'groups.csv', index=False); json.dump(res, open(OUT / 'summary.json', 'w'), indent=1)
    G = pd.concat(allg); g = G[(G.method == 'sop_E3causal') & (G.tau == 0.05)]
    print(g[C1.GK + ['n', 'interventions', 'int_rate', 'util_q50', 'depth_max']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
