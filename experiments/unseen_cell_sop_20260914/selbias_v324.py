"""V324 analysis 2 (PROTOCOL_V324.md): optimism from choosing among variants evaluated on the same three cells.
Variants (768): 16 predictors (every subset of the V287 audit fixes R12, L1, S1, S2; none = L, all = Ls) x 48 request rules
(fixed scale at tau 4 / 5 / 6 %; adaptation from the tau 4 / 5 / 6 % starting scale with (gamma, tau*) on the 5 x 3 grid of
adapt_grid.csv). Main: Ls, adaptation, tau 5 %, gamma 0.05, tau* 0.05.
Bootstrap of the evaluated rows (B = 1000, seed 324): per group, sweeps with replacement, then rows within each drawn sweep
with replacement; windows in time order with copies adjacent, rows in time order. Predictions and starting scales never use
the evaluated cell's rows and are kept; the adaptation is rerun on the resampled sequence.
Gates G1 (reproduction), G2 (vectorised adaptation = adaptive_v284.adapt), G3 (group scoring = closedloop_v251).
usage: OMP_NUM_THREADS=1 python selbias_v324.py [--smoke]   (torch_env; --smoke: 2 predictors, B = 5, writes to v324/smoke, no summary)"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'mcu_port_v322'))
import v322_stubs  # noqa: E402
v322_stubs.install()
import closedloop_v251 as C1  # noqa: E402
import invariance_v276 as I6  # noqa: E402
import compare_v278 as CV  # noqa: E402
import adaptive_v284 as A4  # noqa: E402
import specaudit_v287 as SA  # noqa: E402

GK = C1.GK; CELLS = SA.CELLS; FIXES = SA.FIXES
TAUS = (0.04, 0.05, 0.06)
GRID = [(g, t) for g in (0.01, 0.02, 0.05, 0.1, 0.2) for t in (0.025, 0.05, 0.1)]
MAIN_PRED = tuple(FIXES); MAIN_MODE = ('AD', 0.05, 0.05, 0.05)
B_BOOT, SEED = 1000, 324


def pred_name(fx):
    return 'none (L)' if not fx else ('all (Ls)' if len(fx) == 4 else '+'.join(fx))


def modes():
    out = [('FIX', t, None, None) for t in TAUS]
    out += [('AD', t, g, ts) for t in TAUS for g, ts in GRID]
    return out


def mode_name(m):
    return f'FIX tau{m[1] * 100:g}' if m[0] == 'FIX' else f'AD tau{m[1] * 100:g} g{m[2]:g} t*{m[3]:g}'


def score_matrix(req, truth, grp, ng):
    """closed-loop and strict pass counts for every row of req (variants x rows); grp: group index of each column"""
    r = req / truth[None, :]
    it = r > 1 + 1e-9; util = np.minimum(r, 1.0)
    closed = np.zeros(req.shape[0], int); strict = np.zeros(req.shape[0], int)
    for g in range(ng):
        cols = np.flatnonzero(grp == g)
        mu = np.median(util[:, cols], axis=1); ir = it[:, cols].sum(axis=1) / len(cols)
        closed += (mu >= 0.80) & (ir <= 0.10)
        strict += (it[:, cols].sum(axis=1) == 0) & (np.median(r[:, cols], axis=1) >= 0.80)
    return closed, strict


def adapt_vec(seq_cells, cen_ad, ls0_ad, gam, tst, truth):
    """vectorised adaptive_v284.adapt ('cell' unit) over variants; seq_cells: per cell the row indices in processing order"""
    req = np.full((cen_ad.shape[0], sum(len(s) for s in seq_cells.values())), np.nan); pos = {}
    k = 0
    for c, seq in seq_cells.items():
        ls = ls0_ad[:, CELLS.index(c)].copy()
        for i in seq:
            s = np.exp(ls); r = s * cen_ad[:, i]
            e = (r / truth[i]) > 1 + 1e-9
            req[:, k] = r; pos[k] = i; k += 1
            ls = np.clip(ls + gam * (tst - e.astype(float)), A4.LO, A4.HI)
    return req, np.array([pos[j] for j in range(k)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    OUT = HERE / 'v324' / ('smoke' if a.smoke else ''); OUT.mkdir(parents=True, exist_ok=True)
    CV.TMP = HERE / 'v324' / 'tmp'
    t0 = time.time()
    Bm = pd.read_csv(HERE / 'closedloop_v251/src/bench_rows.csv').rename(columns={'truth_power_W': 'truth'})
    src = I6.sources(); J = SA.bench_J(Bm)
    W = pd.read_csv(HERE / 'mcu_port_v322/queries_v322.csv')[['row_id', 'window_id']]
    J = J.merge(W, on='row_id', how='left'); assert J.window_id.notna().all() and len(J) == 672
    truth = J.truth.to_numpy(float); cell_of = J.evaluation_group.to_numpy(); tkey = J.tkey.to_numpy(float)
    gtab = J[GK].drop_duplicates().sort_values(GK).reset_index(drop=True); ng = len(gtab)
    grp = J[GK].merge(gtab.reset_index(), on=GK, how='left')['index'].to_numpy()
    subsets = [tuple(f for f, on in zip(FIXES, bits) if on) for bits in itertools.product((0, 1), repeat=4)]
    if a.smoke:
        subsets = [(), MAIN_PRED]
    MODES = modes()
    # ---------------------------------------------------------------- predictors: centrals, starting scales, fixed requests
    cen = np.zeros((len(subsets), 672)); scale = np.zeros((len(subsets), len(TAUS), 3)); fixreq = np.zeros((len(subsets), len(TAUS), 672))
    for p, fx in enumerate(subsets):
        X = SA.build('v324_' + ('_'.join(fx) or 'none'), src, Bm, **{f: (f in fx) for f in FIXES})
        cen[p] = SA.central_own(X, J)
        for t, tau in enumerate(TAUS):
            b0, sc = SA.requests_B0(X, tau, J)
            fixreq[p, t] = b0; scale[p, t] = [sc[c]['scale'] for c in CELLS]
        print(f'built {pred_name(fx)} ({time.time() - t0:.0f} s)', flush=True)
    var = [(p, m) for p in range(len(subsets)) for m in MODES]
    names = [f'{pred_name(subsets[p])} | {mode_name(m)}' for p, m in var]
    is_ad = np.array([m[0] == 'AD' for _, m in var])
    ad_idx = np.flatnonzero(is_ad)
    cen_ad = cen[[var[v][0] for v in ad_idx]]
    ls0_ad = np.log(np.array([scale[var[v][0], TAUS.index(var[v][1][1])] for v in ad_idx]))
    gam = np.array([var[v][1][2] for v in ad_idx]); tst = np.array([var[v][1][3] for v in ad_idx])
    fix_rows = {v: fixreq[var[v][0], TAUS.index(var[v][1][1])] for v in np.flatnonzero(~is_ad)}
    # ---------------------------------------------------------------- observed requests and gates
    seq_obs = {c: [i for i in np.flatnonzero(cell_of == c)[np.argsort(tkey[cell_of == c], kind='stable')]] for c in CELLS}
    req_ad_seq, pos = adapt_vec(seq_obs, cen_ad, ls0_ad, gam, tst, truth)
    req_obs = np.zeros((len(var), 672))
    req_obs[ad_idx[:, None], pos[None, :]] = req_ad_seq
    for v, rr in fix_rows.items():
        req_obs[v] = rr
    gates = {}
    # G2: vectorised adaptation = adaptive_v284.adapt on the original order
    worst, flag_mis = 0.0, 0
    for j, v in enumerate(ad_idx):
        p, m = var[v]
        Jc = J.copy(); Jc['central'] = cen[p]
        ref, _, ev = A4.adapt(Jc, {c: scale[p, TAUS.index(m[1]), CELLS.index(c)] for c in CELLS}, m[2], m[3], 'cell')
        d = np.abs(req_obs[v] - ref) / np.maximum(np.abs(ref), 1e-300); worst = max(worst, float(np.max(np.where(ref == 0, np.abs(req_obs[v]), d))))
        flag_mis += int(((req_obs[v] / truth > 1 + 1e-9) != ev).sum())
    gates['G2'] = dict(max_rel=worst, flag_mismatch=flag_mis, ok=bool(worst <= 1e-12 and flag_mis == 0))
    # G3: scoring = closedloop_v251
    S_obs, T_obs = score_matrix(req_obs, truth, grp, ng)
    mis = 0
    for v in range(len(var)):
        s, _ = C1.closed_loop(J, req_obs[v]); o, _ = C1.open_loop(J.assign(_p=req_obs[v]), '_p')
        mis += int(s['groups_util80_int10'] != S_obs[v]) + int(o != T_obs[v])
    gates['G3'] = dict(mismatches=mis, ok=bool(mis == 0))
    # G1: reproduction
    vid = {(subsets[p], m): v for v, (p, m) in enumerate(var)}
    g1 = {}
    if not a.smoke:
        AC = pd.read_csv(HERE / 'paper_package_lsad_20261013/data/audit_components.csv')
        cfg = {'L (pre-audit implementation)': (), 'only R12': ('R12',), 'only L1': ('L1',), 'only S1': ('S1',), 'only S2': ('S2',), 'Ls (all fixes; main)': MAIN_PRED}
        bad = []
        for r in AC.itertuples():
            fx = cfg[r.config]
            got_b0 = int(S_obs[vid[(fx, ('FIX', 0.05, None, None))]]); got_ad = int(S_obs[vid[(fx, MAIN_MODE)]])
            if got_b0 != int(r.B0_tau5) or got_ad != int(r.AD_tau5):
                bad.append(dict(config=r.config, B0=(got_b0, int(r.B0_tau5)), AD=(got_ad, int(r.AD_tau5))))
        AG = pd.read_csv(HERE / 'paper_package_lsad_20261013/data/adapt_grid.csv')
        for r in AG.itertuples():
            v = vid[(MAIN_PRED, ('AD', 0.05, float(r.gamma), float(r.tau_star)))]
            if int(S_obs[v]) != int(r.closed_loop) or int(T_obs[v]) != int(r.original):
                bad.append(dict(grid=(r.gamma, r.tau_star), closed=(int(S_obs[v]), int(r.closed_loop)), strict=(int(T_obs[v]), int(r.original))))
        T = pd.read_csv(HERE / 'paper_package_lsad_20261013/data/ad_trajectory.csv', float_precision='round_trip').set_index('row_id').reindex(J.row_id)
        mr = float(np.max(np.abs(req_obs[vid[(MAIN_PRED, MAIN_MODE)]] - T.request.to_numpy()) / np.maximum(T.request.to_numpy(), 1e-300)))
        g1 = dict(mismatches=bad, main_request_max_rel=mr, ok=bool(not bad and mr <= 1e-9))
    gates['G1'] = g1
    print('gates', json.dumps(gates), flush=True)
    pd.DataFrame({'variant': names, 'predictor': [pred_name(subsets[p]) for p, _ in var], 'mode': [mode_name(m) for _, m in var],
                  'closed_obs': S_obs, 'strict_obs': T_obs}).to_csv(OUT / 'selbias_variants.csv', index=False)
    if not all(g.get('ok', True) for g in gates.values() if g):
        json.dump(dict(gates=gates), open(OUT / 'selbias_summary.json', 'w'), indent=1)
        print('a gate failed: stop'); return
    # ---------------------------------------------------------------- bootstrap
    rng = np.random.default_rng(SEED)
    win_rows = {w: np.flatnonzero(J.window_id.to_numpy() == w)[np.argsort(tkey[J.window_id.to_numpy() == w], kind='stable')]
                for w in J.window_id.unique()}
    grp_wins = {g: sorted(J.loc[grp == g, 'window_id'].unique(), key=lambda w: tkey[win_rows[w][0]]) for g in range(ng)}
    nB = 5 if a.smoke else B_BOOT
    SB = np.zeros((nB, len(var)), np.int16); TB = np.zeros((nB, len(var)), np.int16)
    for b in range(nB):
        blocks = {c: [] for c in CELLS}
        for g in range(ng):
            ws = grp_wins[g]
            for k, wi in enumerate(rng.integers(0, len(ws), size=len(ws))):
                w = ws[wi]; rows = win_rows[w]
                pick = np.sort(rng.integers(0, len(rows), size=len(rows)))
                blocks[cell_of[rows[0]]].append((tkey[rows[0]], g, k, rows[pick]))
        seq = {c: np.concatenate([x[3] for x in sorted(blocks[c], key=lambda x: (x[0], x[1], x[2]))]) for c in CELLS}
        req_ad_b, pos_b = adapt_vec(seq, cen_ad, ls0_ad, gam, tst, truth)
        req_b = np.zeros((len(var), len(pos_b)))
        req_b[ad_idx] = req_ad_b
        for v, rr in fix_rows.items():
            req_b[v] = rr[pos_b]
        SB[b], TB[b] = score_matrix(req_b, truth[pos_b], grp[pos_b], ng)
        if (b + 1) % 100 == 0:
            print(f'bootstrap {b + 1}/{nB} ({time.time() - t0:.0f} s)', flush=True)
    np.savez_compressed(OUT / 'selbias_boot.npz', closed=SB, strict=TB, closed_obs=S_obs, strict_obs=T_obs, names=np.array(names))
    if a.smoke:
        print('smoke ok', SB.shape, f'{time.time() - t0:.0f} s'); return
    # ---------------------------------------------------------------- summaries
    main_v = vid[(MAIN_PRED, MAIN_MODE)]

    def dist(x):
        return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), q025=float(np.quantile(x, .025)), q975=float(np.quantile(x, .975)))

    def optimism(fam, Sb, So):
        fam = np.asarray(fam); Ob = []
        for b in range(Sb.shape[0]):
            sb = Sb[b, fam]; win = fam[sb == sb.max()]
            Ob.append(float(sb.max() - So[win].mean()))
        Ob = np.array(Ob); smax = int(So[fam].max())
        return dict(variants=int(len(fam)), observed_max=smax, argmax=[names[v] for v in fam[So[fam] == smax]][:20], n_argmax=int((So[fam] == smax).sum()),
                    optimism=float(Ob.mean()), optimism_q025=float(np.quantile(Ob, .025)), optimism_q975=float(np.quantile(Ob, .975)),
                    corrected_max=float(smax - Ob.mean()), main_observed=int(So[main_v]))
    allv = np.arange(len(var))
    fam_a = [v for v in allv if subsets[var[v][0]] == MAIN_PRED]
    fam_b = [v for v in allv if var[v][1] == MAIN_MODE]
    res = dict(gates=gates, main=names[main_v], main_closed=dist(SB[:, main_v].astype(float)), main_strict=dist(TB[:, main_v].astype(float)),
               closed=dict(all=optimism(allv, SB, S_obs), ls_request_rules=optimism(fam_a, SB, S_obs), main_rule_predictors=optimism(fam_b, SB, S_obs)),
               strict=dict(all=optimism(allv, TB, T_obs), ls_request_rules=optimism(fam_a, TB, T_obs), main_rule_predictors=optimism(fam_b, TB, T_obs)))
    better = [v for v in allv if S_obs[v] > S_obs[main_v]]
    res['beats_main'] = dict(count=int(len(better)),
                             top=[dict(variant=names[v], closed_obs=int(S_obs[v]), p_boot_beats_main=float(np.mean(SB[:, v] > SB[:, main_v])),
                                       p_boot_ties=float(np.mean(SB[:, v] == SB[:, main_v])))
                                  for v in sorted(better, key=lambda v: -S_obs[v])[:15]],
                             p_beats_main_quantiles=[float(q) for q in np.quantile([np.mean(SB[:, v] > SB[:, main_v]) for v in better], (0, .25, .5, .75, 1))] if better else None)
    res['seconds'] = time.time() - t0
    json.dump(res, open(OUT / 'selbias_summary.json', 'w'), indent=1)
    print(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
