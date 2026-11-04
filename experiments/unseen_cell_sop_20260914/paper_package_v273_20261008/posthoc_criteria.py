"""Post hoc analyses of the V273 package (2026-10-08): the original open-loop criterion and trade-off curves.

Plan: POSTHOC_CRITERIA_PLAN.md (hash in posthoc_plan.sha256, written before any result was computed). Descriptive only:
saved predictions, no training, no selection. run() checks the three gates, computes A1-A4, writes the data files and
returns a summary for build_package_data.py.

  A1  registered tau = 5 % requests of N, N_light, N_hc counted with the original criterion
  A2  zero-exceedance calibration (tau = 0) of N and N_light, scored with both criteria
  A3  calibrated trade-off: tau = 0 ... 20 % (0.25 % steps) for N and N_light
  A4  oracle common-scale trade-off (s chosen on the evaluation data, descriptive): N, N_light, prior method (Chen M)

Original criterion = closedloop_v251.open_loop: a group passes with no row pred/truth > 1 + 1e-9 and median pred/truth
>= 0.80. Closed-loop criterion = closedloop_v255.group_table (median utilisation >= 0.80, intervention rate <= 0.10).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent; TOP = PKG.parent; DATA = PKG / 'data'
if str(TOP) not in sys.path:
    sys.path.insert(0, str(TOP))
import closedloop_v255 as V5  # noqa: E402

C1 = V5.C1; GK = C1.GK; CELLS = V5.CELLS; NAMES = ['prev', 'L4', 'M6', 'DN6', 'G1']
TAUS_REG = np.round(np.arange(0.04, 0.06001, 0.0025), 4)
TAUS_CURVE = np.round(np.arange(0.0, 0.20001, 0.0025), 4)
S_GRID = V5.GRID


class GateFailed(Exception):
    pass


def logmed(X):
    """log median of the finite positive entries of each row (as strictfull_v273.logmed)"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with np.errstate(all='ignore'):
            L = np.log(np.asarray(X, float)); L[~np.isfinite(L)] = np.nan
            return np.nanmedian(L, axis=1)


def third(c, d):
    return [x for x in CELLS if x not in (c, d)][0]


def tables():
    """(fold centres, third-cell-only centres) per configuration, as power"""
    C = pd.read_csv(TOP / 'strictfull_v273/centres_full.csv'); S = pd.read_csv(TOP / 'strictfull_v273/strict_full.csv')
    out = {}
    for cfg in ('N', 'N_light'):
        mc = C.m.to_numpy() if cfg == 'N' else logmed(C[NAMES].to_numpy(float))
        ms = S.m.to_numpy() if cfg == 'N' else logmed(S[NAMES].to_numpy(float))
        out[cfg] = ({(f, r): float(np.exp(v)) for f, r, v in zip(C.fold, C.row_id, mc)},
                    {(e, r): float(np.exp(v)) for e, r, v in zip(S.train_cell, S.row_id, ms)})
    return out


def calibration(B, c, fold, strict):
    """evaluated-cell-free calibration rows of cell c (V273): discharge by the third-cell model, charge by fold c"""
    R = B[B.evaluation_group != c]
    p = np.array([strict.get((third(c, d), r), np.nan) if k == 'discharge' else fold.get((c, r), np.nan)
                  for d, r, k in zip(R.evaluation_group, R.row_id, R.direction)], float)
    t = R.truth.to_numpy(); ok = np.isfinite(p)
    return p[ok], t[ok]


def orig_groups(J, groups_all):
    """original criterion per group (closedloop_v251.open_loop with the reasons)"""
    r = J.pred / J.truth
    G = J.assign(over=r > 1 + 1e-9, ratio=r).groupby(GK).agg(n=('over', 'size'), exceed=('over', 'sum'), ratio_q50=('ratio', 'median')).reset_index()
    G = groups_all.merge(G, on=GK, how='left')
    ok_e = G.exceed == 0; ok_u = G.ratio_q50 >= 0.8
    G['orig_pass'] = ok_e & ok_u
    G['orig_reason'] = np.select([G.orig_pass, ~ok_e & ~ok_u, ~ok_e], ['pass', 'exceedance+utilisation', 'exceedance'], 'utilisation')
    return G


def score(B, groups_all, pred):
    J = B[GK + ['row_id', 'truth']].assign(pred=pred)
    summ, G = V5.group_table(J, groups_all)
    op, oe = C1.open_loop(J, 'pred'); O = orig_groups(J, groups_all)
    assert op == int(O.orig_pass.sum()) and oe == int(O.exceed.sum())
    return dict(J=J, G=G, O=O, int_rate=summ['int_rate'], util_q50=summ['util_q50'], closed=int(G.passed.sum()), orig=op, exceed=oe)


def calibrated(B, groups_all, fold, strict, tau):
    scales = {}; pred = np.full(len(B), np.nan)
    for c in CELLS:
        p, t = calibration(B, c, fold, strict); s, floor = V5.choose(p, t, tau)
        m = (B.evaluation_group == c).to_numpy()
        pred[m] = s * np.array([fold[(c, r)] for r in B.row_id[m]])
        scales[c] = s
    assert np.isfinite(pred).all()
    return dict(score(B, groups_all, pred), scales=scales)


def failed(G, col, reason):
    return '; '.join(f'{x.evaluation_group} {x.direction} {x.temp_C} C {x.horizon_s:g} s ({getattr(x, reason)})'
                     for x in G[~G[col]].itertuples())


def gates(B, groups_all, chen):
    out = {}
    T = pd.read_csv(TOP / 'strictfull_v273/tau_range.csv')
    for cfg in ('N', 'N_light'):
        fold, strict = T_[cfg]
        R = T[T.config == cfg]; bad = []
        for tau in TAUS_REG:
            r = calibrated(B, groups_all, fold, strict, float(tau)); ref = R[np.isclose(R.tau, tau)].iloc[0]
            same = (r['closed'] == int(ref.groups_pass) and abs(r['int_rate'] - ref.int_rate) <= 1e-12
                    and all(abs(r['scales'][c] - ref[f'scale_{c}']) < 1e-9 for c in CELLS))
            if not same:
                bad.append(float(tau))
        out[f'1_{cfg}_tau_4_6'] = dict(ok=not bad, mismatched_tau=bad)
    r5 = calibrated(B, groups_all, *T_['N'], 0.05)
    ref = pd.read_csv(TOP / 'strictfull_v273/rows_tau5_N.csv')[['row_id', 'pred']].rename(columns={'pred': 'ref'})
    M = r5['J'].merge(ref, on='row_id', how='inner'); rel = float(np.max(np.abs(M.pred / M.ref - 1))) if len(M) else np.inf
    out['2_N_tau5_requests'] = dict(ok=bool(len(M) == 672 and rel <= 1e-12), rows=int(len(M)), max_rel_diff=rel)
    ok3 = bool(len(chen) == 672 and set(chen.row_id) == set(B.row_id) and np.isfinite(chen.central).all() and (chen.central >= 0).all())   # plan amendment 1: zero predictions are valid
    out['3_prior_method_centres'] = dict(ok=ok3, rows=int(len(chen)))
    return out


def run(write=True):
    global T_
    B = V5.load(); groups_all = B[GK].drop_duplicates().sort_values(GK).reset_index(drop=True); assert len(groups_all) == 15
    T_ = tables()
    chen = pd.read_csv(TOP / 'chen_v266/rows_tau5_M.csv')[['row_id', 'central']]
    G0 = gates(B, groups_all, chen)
    if not all(g['ok'] for g in G0.values()):
        raise GateFailed(json.dumps(G0))
    summary = dict(plan='POSTHOC_CRITERIA_PLAN.md', gates=G0)

    # A1: registered tau = 5 % requests with the original criterion
    a_rows, a1 = [], {}
    for k in ('N', 'N_light', 'N_hc'):
        R = pd.read_csv(TOP / f'strictfull_v273/rows_tau5_{k}.csv')
        J = R[GK + ['row_id', 'truth', 'pred']]; s = score(J, groups_all, J.pred.to_numpy())
        a1[k] = dict(groups_original=s['orig'], exceed_rows=s['exceed'], groups_closed_loop=s['closed'], failed_original=failed(s['O'], 'orig_pass', 'orig_reason'))
        a_rows.append(dict(analysis='A1 registered tau 5 % requests', config=k, tau=0.05, scale_A1=np.nan, scale_A17=np.nan, scale_B=np.nan,
                           exceed_rows=s['exceed'], int_rate=round(s['int_rate'], 6), util_q50=round(s['util_q50'], 6),
                           groups_original=s['orig'], groups_closed_loop=s['closed'], failed_original=a1[k]['failed_original']))
    summary['A1'] = a1

    # A2: zero-exceedance calibration (tau = 0)
    a2, g_rows = {}, []
    for cfg in ('N', 'N_light'):
        r = calibrated(B, groups_all, *T_[cfg], 0.0)
        a2[cfg] = dict(scales=r['scales'], exceed_rows=r['exceed'], int_rate=r['int_rate'], util_q50=r['util_q50'], groups_original=r['orig'],
                       groups_closed_loop=r['closed'], failed_original=failed(r['O'], 'orig_pass', 'orig_reason'),
                       failed_closed_loop=failed(r['G'], 'passed', 'reason'))
        a_rows.append(dict(analysis='A2 zero-exceedance calibration', config=cfg, tau=0.0, scale_A1=r['scales']['A1'], scale_A17=r['scales']['A17'],
                           scale_B=r['scales']['B'], exceed_rows=r['exceed'], int_rate=round(r['int_rate'], 6), util_q50=round(r['util_q50'], 6),
                           groups_original=r['orig'], groups_closed_loop=r['closed'], failed_original=a2[cfg]['failed_original']))
        M = r['G'][GK + ['n', 'interventions', 'int_rate', 'util_q50', 'passed', 'reason']].merge(
            r['O'][GK + ['exceed', 'ratio_q50', 'orig_pass', 'orig_reason']], on=GK)
        g_rows.append(M.assign(config=cfg).rename(columns={'passed': 'closed_loop_pass', 'reason': 'closed_loop_reason'}))
    summary['A2'] = a2

    # A3: calibrated trade-off over tau
    c_rows = []
    for cfg in ('N', 'N_light'):
        for tau in TAUS_CURVE:
            r = calibrated(B, groups_all, *T_[cfg], float(tau))
            c_rows.append(dict(config=cfg, tau=float(tau), scale_A1=r['scales']['A1'], scale_A17=r['scales']['A17'], scale_B=r['scales']['B'],
                               exceed_rows=r['exceed'], int_rate=round(r['int_rate'], 6), util_q50=round(r['util_q50'], 6),
                               groups_closed_loop=r['closed'], groups_original=r['orig']))
    C3 = pd.DataFrame(c_rows); a3 = {}
    for cfg, g in C3.groupby('config'):
        pick = lambda t: g[np.isclose(g.tau, t)].iloc[0]
        a3[cfg] = dict(groups_closed_loop_range=[int(g.groups_closed_loop.min()), int(g.groups_closed_loop.max())],
                       groups_original_range=[int(g.groups_original.min()), int(g.groups_original.max())],
                       at={f'{t:.2f}': dict(closed=int(pick(t).groups_closed_loop), original=int(pick(t).groups_original),
                                            int_rate=float(pick(t).int_rate), util_q50=float(pick(t).util_q50)) for t in (0.0, 0.05, 0.10, 0.20)})
    summary['A3'] = a3

    # A4: oracle common-scale trade-off (descriptive)
    cen = {'N': np.array([T_['N'][0][(c, r)] for c, r in zip(B.evaluation_group, B.row_id)]),
           'N_light': np.array([T_['N_light'][0][(c, r)] for c, r in zip(B.evaluation_group, B.row_id)]),
           'prior_method_M': chen.set_index('row_id').central.reindex(B.row_id).to_numpy(float)}
    o_rows, a4 = [], {}
    for name, x in cen.items():
        rows = []
        for s in S_GRID:
            r = score(B, groups_all, float(s) * x)
            rows.append(dict(predictor=name, s=float(s), exceed_rows=r['exceed'], int_rate=round(r['int_rate'], 6), util_q50=round(r['util_q50'], 6),
                             groups_closed_loop=r['closed'], groups_original=r['orig']))
        D = pd.DataFrame(rows); o_rows.append(D)
        z = D[D.exceed_rows == 0]; f5 = D[D.int_rate <= 0.05]
        pt = lambda d: (dict(s=float(d.s.max()), util_q50=float(d.loc[d.s.idxmax(), 'util_q50']), groups_closed_loop=int(d.loc[d.s.idxmax(), 'groups_closed_loop']),
                             groups_original=int(d.loc[d.s.idxmax(), 'groups_original'])) if len(d) else None)
        a4[name] = dict(zero_exceedance=pt(z), int_rate_le_5pct=pt(f5),
                        max_groups_closed_loop=int(D.groups_closed_loop.max()), max_groups_original=int(D.groups_original.max()))
    summary['A4'] = a4

    if write:
        pd.DataFrame(a_rows).to_csv(DATA / 'openloop_criterion_posthoc.csv', index=False)
        pd.concat(g_rows, ignore_index=True)[['config'] + GK + ['n', 'interventions', 'int_rate', 'util_q50', 'closed_loop_pass', 'closed_loop_reason',
                                               'exceed', 'ratio_q50', 'orig_pass', 'orig_reason']].to_csv(DATA / 'groups_tau0_posthoc.csv', index=False)
        C3.to_csv(DATA / 'tradeoff_calibrated_posthoc.csv', index=False)
        pd.concat(o_rows, ignore_index=True).to_csv(DATA / 'tradeoff_oracle_posthoc.csv', index=False)
        (DATA / 'posthoc_criteria_summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')
    return summary


if __name__ == '__main__':
    print(json.dumps(run(), indent=1, ensure_ascii=False))
