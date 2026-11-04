"""V278 (PROTOCOL_COMPARE_V278.md): one pre-fixed diagnostic comparison on a cleaned baseline.
  B0  the baseline: L = the five local predictors (V273 N_light computation), or P = V273 N with the i3n tables of the
      vT1h rebuild (--phys-dir); one scale per evaluated cell (V273 rule: discharge calibration rows by the third-cell-only
      model, charge rows by the fold-c model in sample), tau 5 % main;
  C1  discharge-only calibration control: B0 centres, discharge requests x a scale chosen on B0's discharge calibration
      rows only, charge requests = B0;
  P1  symmetric cross-calibration: discharge request = sqrt(s_ab f_a(x) * s_ba f_b(x)), f_e the cell-e-only centre, s_eg
      chosen on cell g's discharge rows predicted by f_e (tau per branch), charge requests = B0;
  P2  2 s residual: for the B 25 C 2 s rows only, prev / L4 / M6 of the fold-B model refitted on the training cells' 2 s
      response inside their 10 s truth pulses (target R2_q, current |q_I2|, reference current |prev_I2|, samples at
      2.0 +- 0.05 s), DN6 / G1 unchanged, request = B0's s_B x the new centre; everything else = B0.
  H   V273 N as saved (historical, its physics selection depended on the evaluation truths, V277).
Both criteria (closed loop; original: no exceedance and median ratio >= 0.80) on the same requests, per group.
usage: python compare_v278.py --baseline L --gates-only    (Q1, Q2; no comparison computed)
       python compare_v278.py --baseline L                 (main run, once; Q3 truth-change check inside)
       python compare_v278.py --baseline P --phys-dir DIR  (after the vT1h rebuild, once)"""
import os
import sys

READS = set()


def _audit(event, args):
    try:
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if isinstance(mode, str) and any(ch in mode for ch in 'wax+'):
            return
        if mode is None and isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            return
        READS.add(os.path.abspath(os.fsdecode(args[0])))
    except Exception:
        pass


sys.addaudithook(_audit)

import argparse  # noqa: E402
import hashlib  # noqa: E402
from pathlib import Path  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
from run_envelope import HERE, dump  # noqa: E402
import closedloop_v251 as C1, closedloop_v255 as V5, closedloop_v256 as V6  # noqa: E402
import leakfree_v272 as V2  # noqa: E402
import strictfull_v273 as S  # noqa: E402
import invariance_v276 as I6  # noqa: E402

OUT = HERE / 'compare_v278'; TMP = OUT / 'tmp'
CELLS = V5.CELLS; TAUS = [float(t) for t in V6.TAUS]; NAMES = V2.NAMES; TAU_MAIN = 0.05
RESIST = ('prev', 'L4', 'M6'); T2TOL = 0.05; SEED = 276
LOG = []


class Stop(Exception):
    pass


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def table_from(name, R, F, E, CV):
    """local_sop.table() (unchanged) on the given frames (invariance_v276.table_from with this experiment's folder)"""
    d = TMP / name; (d / 'out/feats').mkdir(parents=True, exist_ok=True)
    R.to_csv(d / 'rows.csv', index=False, float_format='%.17g'); F.to_csv(d / 'out/feats/relax.csv', index=False, float_format='%.17g')
    E.to_csv(d / 'events.csv', index=False, float_format='%.17g')
    if CV is not None:
        CV.to_csv(d / 'out/feats/cv.csv', index=False, float_format='%.17g')
    old = (V2.ls.BENCH, V2.ls.EV, V2.ls.ROOT)
    V2.ls.BENCH, V2.ls.EV, V2.ls.ROOT = d / 'rows.csv', d / 'events.csv', d
    try:
        D = V2.ls.table()
    finally:
        V2.ls.BENCH, V2.ls.EV, V2.ls.ROOT = old
    assert len(D) == 672
    return D


def phys_from(d):
    """S.phys_tables() on another folder of i3n tables (n_all, n_bA17, n_bA1)"""
    out, cov = {}, {}
    for t in ('n_all', 'n_bA17', 'n_bA1'):
        P = pd.read_csv(Path(d) / t / 'pred.csv')
        ok = (P.status == 'ok') & np.isfinite(P.pred_power_W) & (P.pred_power_W > 0)
        out[t] = P[ok].set_index('row_id').pred_power_W; cov[t] = dict(rows=int(len(P)), ok=int(ok.sum()))
    return out, cov


# ---------------------------------------------------------------- baseline
def build(name, base, src, Bm, phys_dir=None):
    D = table_from(name, *src)
    cell_of = D.set_index('row_id').evaluation_group.to_dict()
    TH = V2.boundaries(D, 'min_error')
    Fv = {n: V2.loco(D, n, TH) for n in NAMES}; Cv = V2.centres(Fv)
    Sv = pd.concat([V2.strict(D, n, TH) for n in NAMES], ignore_index=True)
    cov = None
    if base == 'L':
        C, Sc = Cv, S.light_strict_centres(Sv)
    else:
        phys, cov = phys_from(phys_dir)
        C, Sc = S.full_fold_centres(Cv, cell_of, phys), S.full_strict_centres(Sv, cell_of, phys)
    Bc = S.set_centres(Bm, C, name)
    return dict(D=D, TH=TH, Fv=Fv, C=C, Sc=Sc, Bc=Bc, sm=Sc.set_index(['train_cell', 'row_id']).m, cov=cov)


def arm_B0(X, tau, groups_all):
    fn = S.look(X['Bc'])
    r, _, J = S.score_single(X['Bc'], groups_all, tau, S.cal_free_single(S.dis_from(X['sm']), lambda c, d, rr: fn[(c, rr)]))
    return J.set_index('row_id').pred.astype(float), {c: v['scale'] for c, v in r['scales'].items()}


def arm_C1(X, b0, tau):
    Bc = X['Bc']; dis = S.dis_from(X['sm']); pred = b0.copy(); sc = {}
    for c in CELLS:
        R = Bc[(Bc.evaluation_group != c) & (Bc.direction == 'discharge')]
        p = np.array([dis(c, d, r) for d, r in zip(R.evaluation_group, R.row_id)], float); t = R.truth.to_numpy(); ok = np.isfinite(p)
        s, _ = V5.choose(p[ok], t[ok], tau); sc[c] = dict(scale_discharge=s, calibration_rows=int(ok.sum()))
        ev = Bc[(Bc.evaluation_group == c) & (Bc.direction == 'discharge')]
        pred.loc[ev.row_id.to_numpy()] = s * ev[f'sop_{c}'].to_numpy()
    return pred, sc


def arm_P1(X, b0, tau):
    Bc = X['Bc']; sm = X['sm']; pred = b0.copy(); sc = {}; fb = 0
    f = lambda e, r: float(np.exp(sm.get((e, r), np.nan)))
    for c in CELLS:
        a, b = [x for x in CELLS if x != c]; s = {}
        for e, g in ((a, b), (b, a)):
            R = Bc[(Bc.evaluation_group == g) & (Bc.direction == 'discharge')]
            p = np.array([f(e, r) for r in R.row_id], float); t = R.truth.to_numpy(); ok = np.isfinite(p)
            s[e], _ = V5.choose(p[ok], t[ok], tau); sc[f'{c}|{e}->{g}'] = dict(scale=s[e], calibration_rows=int(ok.sum()))
        ev = Bc[(Bc.evaluation_group == c) & (Bc.direction == 'discharge')]
        for r in ev.row_id:
            qa, qb = s[a] * f(a, r), s[b] * f(b, r)
            if np.isfinite(qa) and np.isfinite(qb):
                v = float(np.sqrt(qa * qb))
            elif np.isfinite(qa) or np.isfinite(qb):
                v = qa if np.isfinite(qa) else qb; fb += 1
            else:
                v = float(b0[r]); fb += 1
            pred[r] = v
    return pred, sc, fb


# ---------------------------------------------------------------- P2
def p2_sources(src, p2):
    """as-if-2 s training table: A rows with horizon 2 s, target R2_q (unchanged column), current |q_I2| (through
    truth_power_W = |q_I2| V_lim), reference current |prev_I2| (prev_I_last); a sample outside 2.0 +- 0.05 s removes the
    row's target (A) or its reference current (B 2 s rows)"""
    R, F, E, CV = (x.copy() if x is not None else None for x in src)
    P = p2.set_index('row_id')
    a = R.evaluation_group.isin(['A1', 'A17']).to_numpy()
    vlim = np.where(R.direction == 'charge', 4.15, 2.55)
    qi = R.row_id.map(P.q_I2).abs().to_numpy(); qt = R.row_id.map(P.q_t2).to_numpy()
    pt = F.row_id.map(P.prev_t2).to_numpy(); pi = F.row_id.map(P.prev_I2).to_numpy()
    R['horizon_s'] = R.horizon_s.astype(float)
    R.loc[a, 'horizon_s'] = 2.0
    R['truth_power_W'] = R.truth_power_W.astype(float)
    R.loc[a, 'truth_power_W'] = qi[a] * vlim[a]
    fa = F.row_id.isin(R.row_id[a]).to_numpy()
    bad_q = fa & ~(np.abs(F.row_id.map(P.q_t2).to_numpy() - 2.0) <= T2TOL)
    has_prev = np.isfinite(pt)
    bad_p = has_prev & ~(np.abs(pt - 2.0) <= T2TOL)
    F['R2_q'] = F.R2_q.astype(float)
    F.loc[bad_q | (fa & bad_p), 'R2_q'] = np.nan                        # training rows outside the sample-time rule
    F['prev_I_last'] = F.prev_I_last.astype(float)
    ok_p = has_prev & ~bad_p
    F.loc[ok_p, 'prev_I_last'] = pi[ok_p]                               # reference current = previous pulse at 2 s
    F.loc[has_prev & bad_p, 'prev_I_last'] = np.nan
    info = dict(train_rows=int(fa.sum()), train_rows_q_time_out=int(bad_q.sum()), rows_prev_time_out=int(bad_p.sum()),
                q_t2=pd.Series(qt[a]).describe().round(4).to_dict(), prev_t2=pd.Series(pt[has_prev]).describe().round(4).to_dict())
    return (R, F, E, CV), info


def arm_P2(X, src, p2, b0, b0_scales, Bm, base):
    s2, info = p2_sources(src, p2)
    D2 = table_from('P2_' + base, *s2)
    TH = V2.boundaries(D2, 'min_error')
    assert all(TH[('fold', 'B')][k]['t'] == X['TH'][('fold', 'B')][k]['t'] for k in TH[('fold', 'B')]), 'fold-B boundaries changed'
    Bc = X['Bc']; rows2 = Bc[(Bc.evaluation_group == 'B') & (Bc.horizon_s == 2.0)].row_id.tolist()
    new = {}
    for n in RESIST:
        F = V2.loco(D2, n, TH); F = F[(F.fold == 'B') & F.row_id.isin(rows2)].set_index('row_id').pred_power_W
        new[n] = F
    C = X['C']; Cb = C[(C.fold == 'B') & C.row_id.isin(rows2)].set_index('row_id').copy()
    cols = list(NAMES) + (['i3n'] if 'i3n' in Cb.columns else [])
    old = Cb[cols].copy()
    for n in RESIST:
        Cb[n] = new[n].reindex(Cb.index)
    m, k = S.logmed(Cb[cols].to_numpy(float))
    pred = b0.copy(); pred.loc[Cb.index] = b0_scales['B'] * np.exp(m)
    detail = pd.concat([old.add_suffix('_B0'), Cb[cols].add_suffix('_P2')], axis=1).assign(centre_B0=np.exp(C[(C.fold == 'B') & C.row_id.isin(rows2)].set_index('row_id').m), centre_P2=np.exp(m), n_pred=k)
    return pred, info, detail


# ---------------------------------------------------------------- scoring
def evaluate(Bm, pred, groups_all):
    J = Bm[['row_id', 'evaluation_group', 'direction', 'temp_C', 'horizon_s', 'truth']].copy()
    J['pred'] = pred.reindex(J.row_id).to_numpy(); assert J.pred.notna().all()
    summ, G = V5.group_table(J, groups_all)
    r = J.pred / J.truth
    O = J.assign(over=r > 1 + 1e-9, ratio=r).groupby(C1.GK).agg(exceed=('over', 'sum'), ratio_q50=('ratio', 'median'), ratio_max=('ratio', 'max')).reset_index()
    O['pass_original'] = (O.exceed == 0) & (O.ratio_q50 >= 0.80)
    G = G.merge(O, on=C1.GK, how='left')
    op, oe = C1.open_loop(J, 'pred'); assert op == int(G.pass_original.sum())
    return dict(closed_loop=int(G.passed.sum()), original=op, exceed_rows=oe, int_rate=summ['int_rate'], util_q50=summ['util_q50']), G


def run_all(X, src, Bm, p2, base, groups_all, taus):
    out = {}
    for tau in taus:
        b0, b0s = arm_B0(X, tau, groups_all)
        c1, c1s = arm_C1(X, b0, tau)
        p1, p1s, fb = arm_P1(X, b0, tau)
        out[tau] = dict(B0=(b0, b0s), C1=(c1, c1s), P1=(p1, p1s, fb))
        if abs(tau - TAU_MAIN) < 1e-12:
            out[tau]['P2'] = arm_P2(X, src, p2, b0, b0s, Bm, base)
    return out


# ---------------------------------------------------------------- registration
def check_hashes(listfile, base):
    bad, seen = [], set()
    for line in listfile.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        h, p = line.split(None, 1); p = p.strip(); seen.add(p); q = base / p
        if not q.is_file() or hashlib.sha256(q.read_bytes()).hexdigest() != h:
            bad.append(p)
    if bad:
        raise Stop(f'{listfile.name}: files changed or missing: {bad}')
    return seen


def project_inputs():
    root = str(HERE.parents[1]) + os.sep
    data = sorted(p for p in READS if p.startswith(root) and os.path.isfile(p) and not p.endswith(('.py', '.pyc'))
                  and '/compare_v278/tmp/' not in p and '/compare_v278/v278_prereg' not in p and '/invariance_v276/tmp/' not in p)
    mods = sorted({os.path.abspath(m.__file__) for m in list(sys.modules.values())
                   if isinstance(getattr(m, '__file__', None), str) and os.path.isabs(m.__file__) and m.__file__.startswith(root) and os.path.isfile(m.__file__)})
    return data, mods


def gate_q2(p2, F0):
    M = p2.merge(F0[['row_id', 'prev_R2', 'R2_q']], on='row_id', suffixes=('', '_relax'))
    a = I6.rel(M.prev_R2, M.prev_R2_relax); b = I6.rel(M.q_R2, M.R2_q)
    out = dict(rows=int(len(M)), prev_R2_max_rel=float(np.nanmax(a)), prev_R2_nan_mismatch=int((M.prev_R2.isna() != M.prev_R2_relax.isna()).sum()),
               q_R2_max_rel=float(np.nanmax(b)), q_R2_nan_mismatch=int((M.q_R2.isna() != M.R2_q.isna()).sum()))
    out['ok'] = bool(out['rows'] == 672 and out['prev_R2_max_rel'] <= 1e-9 and out['q_R2_max_rel'] <= 1e-9 and out['prev_R2_nan_mismatch'] == 0 and out['q_R2_nan_mismatch'] == 0)
    return out


def run(a, res):
    base = a.baseline; sub = OUT / base; sub.mkdir(parents=True, exist_ok=True)
    reg = set()
    if not a.gates_only:
        reg = check_hashes(OUT / 'v278_prereg_stage1.sha256', HERE) | check_hashes(OUT / 'v278_prereg.sha256', HERE)
        if base == 'P':
            reg |= check_hashes(OUT / 'v278_prereg_P.sha256', HERE)
        res['registration'] = dict(registered_files=len(reg), all_match=True); say('registration:', res['registration'])
    B0t = V5.load(); groups_all = B0t[C1.GK].drop_duplicates().sort_values(C1.GK).reset_index(drop=True)
    src = I6.sources(); p2 = pd.read_csv(OUT / 'p2_feats.csv')
    G = res['gates'] = {}
    G['Q2_p2_features'] = gate_q2(p2, src[1]); say('Q2 P2 features:', G['Q2_p2_features'])
    X = build(base, base, src, B0t, a.phys_dir)
    if base == 'L':
        b0, _ = arm_B0(X, TAU_MAIN, groups_all)
        sv = pd.read_csv(S.OUT / 'rows_tau5_N_light.csv').set_index('row_id')
        q1 = dict(max_rel_pred=float(np.nanmax(I6.rel(b0.reindex(sv.index), sv.pred))),
                  max_rel_central=float(np.nanmax(I6.rel(X['Bc'].set_index('row_id').apply(lambda r: r[f"sop_{r.evaluation_group}"], axis=1).reindex(sv.index), sv.central))))
        q1['ok'] = bool(q1['max_rel_pred'] <= 1e-12 and q1['max_rel_central'] <= 1e-12)
        G['Q1_baseline_L_equals_V273_N_light'] = q1; say('Q1 B0(L) = V273 N_light:', q1)
    else:
        res['physics_coverage'] = X['cov']; say('physics tables:', X['cov'])
    if a.gates_only:
        data, mods = project_inputs()
        (sub / 'inputs_read.txt').write_text('# data files read by the gates run\n' + ''.join(os.path.relpath(p, HERE) + '\n' for p in data)
                                             + '# project modules imported\n' + ''.join(os.path.relpath(p, HERE) + '\n' for p in mods))
        res['gates_ok'] = bool(all(g['ok'] for g in G.values())); say('gates', 'passed' if res['gates_ok'] else 'FAILED')
        return
    if not all(g['ok'] for g in G.values()):
        raise Stop('a gate failed; no comparison computed (protocol)')
    # ---- main computation
    R = run_all(X, src, B0t, p2, base, groups_all, TAUS)
    # Q3: truth change of each evaluated cell leaves its requests of every arm unchanged (tau 5 %)
    rng = np.random.default_rng(SEED); fac = I6.factors(src[0], rng); q3 = {}
    for c in CELLS:
        s1, B1, _ = I6.t1(src, B0t, c, fac)
        X1 = build(f'{base}_Q3_{c}', base, s1, B1, a.phys_dir)
        R1 = run_all(X1, s1, B1, p2, base, groups_all, [TAU_MAIN])[TAU_MAIN]
        ids = B0t.row_id[B0t.evaluation_group == c].to_numpy()
        for arm in ('B0', 'C1', 'P1', 'P2'):
            x, y = R[TAU_MAIN][arm][0].reindex(ids), R1[arm][0].reindex(ids)
            q3[f'{c}|{arm}'] = float(np.nanmax(I6.rel(x, y)))
    G['Q3_truth_change'] = dict(max_rel=q3, ok=bool(max(q3.values()) <= 1e-12)); say('Q3 truth change (max rel per cell|arm):', q3)
    if not G['Q3_truth_change']['ok']:
        raise Stop('Q3 failed: an evaluated cell\'s request depends on its truth (protocol: not reported)')
    # ---- scoring
    H = pd.read_csv(S.OUT / 'rows_tau5_N.csv').set_index('row_id').pred
    arms = {'H': H, 'B0': R[TAU_MAIN]['B0'][0], 'C1': R[TAU_MAIN]['C1'][0], 'P1': R[TAU_MAIN]['P1'][0], 'P2': R[TAU_MAIN]['P2'][0]}
    summ, groups = {}, []
    for k, p in arms.items():
        s, Gk = evaluate(B0t, p, groups_all); summ[k] = s; groups.append(Gk.assign(arm=k))
        p.rename('pred').to_csv(sub / f'requests_{k}.csv', float_format='%.17g')
    Gall = pd.concat(groups, ignore_index=True); Gall.to_csv(sub / 'groups_tau5.csv', index=False)
    res['tau5'] = summ
    res['scales_tau5'] = dict(B0=R[TAU_MAIN]['B0'][1], C1=R[TAU_MAIN]['C1'][1], P1=R[TAU_MAIN]['P1'][1], P1_fallback_rows=R[TAU_MAIN]['P1'][2])
    _, p2info, p2detail = R[TAU_MAIN]['P2']; res['P2'] = p2info; p2detail.to_csv(sub / 'p2_rows.csv', float_format='%.6g')
    rows = []
    for tau in TAUS:
        for k in ('B0', 'C1', 'P1'):
            s, _ = evaluate(B0t, R[tau][k][0], groups_all); rows.append(dict(tau=tau, arm=k, **s))
    T = pd.DataFrame(rows); T.to_csv(sub / 'tau_range.csv', index=False)
    res['tau_4_6'] = {k: [int(T[T.arm == k].closed_loop.min()), float(T[T.arm == k].closed_loop.median()), int(T[T.arm == k].closed_loop.max())] for k in ('B0', 'C1', 'P1')}
    # ---- verdicts (protocol 4)
    def g(arm):
        x = Gall[Gall.arm == arm].set_index(C1.GK); return x
    gc1, gp1, gb0, gp2 = g('C1'), g('P1'), g('B0'), g('P2')
    dis_fail_util = [i for i in gc1.index if i[1] == 'discharge' and not gc1.loc[i, 'passed'] and gc1.loc[i, 'reason'] == 'utilisation']
    newly_int = [i for i in gp1.index if not gp1.loc[i, 'passed'] and 'interventions' in gp1.loc[i, 'reason'] and gc1.loc[i, 'passed']]
    rescued = [i for i in dis_fail_util if gp1.loc[i, 'passed']]
    if summ['P1']['closed_loop'] <= summ['C1']['closed_loop']:
        h1 = '반증'
    elif rescued and not newly_int:
        h1 = '지지'
    else:
        h1 = '엇갈림'
    key = ('B', 'discharge', 25, 2.0)
    kb = [i for i in gp2.index if i[0] == 'B' and i[2] == 25 and float(i[3]) == 2.0][0]
    u0, u2, i2 = float(gb0.loc[kb, 'util_q50']), float(gp2.loc[kb, 'util_q50']), int(gp2.loc[kb, 'interventions'])
    h2 = '지지' if bool(gp2.loc[kb, 'passed']) else ('반증' if (u2 < u0 or i2 > 0) else '부분')
    res['verdicts'] = dict(H1_P1_vs_C1=h1, H1_rescued=[list(map(str, i)) for i in rescued], H1_newly_intervention=[list(map(str, i)) for i in newly_int],
                           H2_P2_B25C2s=h2, H2_util_B0=u0, H2_util_P2=u2, H2_interventions_P2=i2, calibration_effect_C1_minus_B0=summ['C1']['closed_loop'] - summ['B0']['closed_loop'])
    for k, s in summ.items():
        say(f"{k:3s} tau 5 %: closed loop {s['closed_loop']:2d}/15  original {s['original']:2d}/15  exceed rows {s['exceed_rows']:3d}  int {100 * s['int_rate']:.2f} %  util {100 * s['util_q50']:.1f} %")
    say('tau 4-6 % (closed loop min / median / max):', res['tau_4_6'])
    say('verdicts:', res['verdicts'])
    say('ci_note: 세 물리적 셀에서 구성된 조건별 기술통계이며, 15개 그룹을 독립 시행으로 간주한 일반화율 신뢰구간은 제시하지 않았다.')
    data, mods = project_inputs()
    res['unregistered_reads'] = sorted({os.path.relpath(p, HERE) for p in data + mods} - reg)
    say('files read but not registered:', res['unregistered_reads'] or 'none')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, choices=['L', 'P']); ap.add_argument('--phys-dir'); ap.add_argument('--gates-only', action='store_true')
    a = ap.parse_args()
    if a.baseline == 'P' and not a.phys_dir:
        ap.error('--phys-dir is required for baseline P')
    OUT.mkdir(exist_ok=True); stem = ('gates_only' if a.gates_only else 'run')
    res = dict(protocol='PROTOCOL_COMPARE_V278.md', baseline=a.baseline, mode=stem, phys_dir=a.phys_dir)
    try:
        run(a, res)
    except Stop as e:
        res['stopped'] = str(e); say('STOPPED:', e)
        raise SystemExit(1)
    finally:
        dump(OUT / a.baseline / f'{stem}.json', V2.clean(res))
        (OUT / a.baseline / f'{stem}.log').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
