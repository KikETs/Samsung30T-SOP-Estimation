"""V286 (PROTOCOL_S5_V286.md): S5 (RPCWBY Test#3 Samsung30T_old, channel 5) as a fourth cell.
  A  S5's eligible rows join every training set of the L pipeline (fold models, single-cell models, region boundaries);
     calibration rows, tau rule and evaluation unchanged (672 Samsung rows, 15 groups); requests B0 (V273 rule, tau 5 %)
     and AD (V284 C0 with s0 = the arm's B0 scales). Arms A2 (S5 2 s rows), A12 (S5 2 s and 10 s rows).
  B  the L pipeline fitted on the three Samsung cells (fold S5) predicts S5; initial scale by the V273 rule extended to a
     fourth cell; FIX / C0 (V284 AD) / C1 (per direction) in S5's time order; groups (set, direction, temperature,
     horizon) with n >= 3.
  C  fallback order of the local predictors as documented ("no earlier pulse: off basis"): a row with an earlier pulse
     but missing features keeps the prev basis with its finite features (local_sop2), and local_sop's prev basis without
     the R_off ratio is the previous pulse's R; B0 and AD on the 15 groups (arm F). Descriptive: A2F (A2 + F), B under F.
usage: python s5_v286.py --stage gates   (GA reproduction, GB S5 truth invariance + ACI copy, GC fallback scope)
       python s5_v286.py --stage main    (once, after the registration s5_v286/v286_prereg.sha256)"""
import argparse
import contextlib
import hashlib
import json
import warnings

import numpy as np, pandas as pd
from run_envelope import HERE, dump
import closedloop_v251 as C1, closedloop_v255 as V5
import leakfree_v272 as V2
import strictfull_v273 as S
import invariance_v276 as I6
import compare_v278 as CV
import adaptive_v284 as A4

OUT = HERE / 's5_v286'; XD = OUT / 'out'
CV.TMP = OUT / 'tmp'                                                    # after the imports (adaptive_v284 sets its own)
warnings.simplefilter('ignore')
ls, ls2 = V2.ls, V2.ls2
NAMES = V2.NAMES; CELLS = V5.CELLS; GK = C1.GK
TAU = 0.05; TOL = 1e-12; SEED = 286
MIN_GROUP, MIN_REST, MIN_LOAD_S = 3, 60.0, 5.0
FILE_ORDER = ['SOP_30T_July_11_0degC_10s_Channel_5_Wb_1.CSV', 'SOP_30T_July_12_25degC_10s_Channel_5_Wb_1.CSV',
              'SOP_30T_July_12_10degC_10s_Channel_5_Wb_1.CSV', 'SOP_30T_July_13_40degC_10s_Channel_5_Wb_1.CSV',
              'SOP_30T_July_19_25degC_2s_Channel_5_Wb_1.CSV', 'SOP_30T_July_20_0degC_2s_Channel_5_Wb_1.CSV',
              'SOP_30T_July_20_10degC_2s_Channel_5_Wb_1.CSV', 'SOP_30T_July_21_40degC_2s_Channel_5_Wb_1.CSV',
              'SOP_30T_Sep_26_-20degC_10s_Channel_5_Wb_1.CSV', 'SOP_30T_Sep_26_-20degC_2s_Channel_5_Wb_1.CSV',
              'SOP_30T_Oct_17_-10degC_2s_Channel_5_Wb_1.CSV', 'SOP_30T_Oct_18_-10degC_10s_Channel_5_Wb_1.CSV']
LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(all='ignore'):
        r = np.abs(a / b - 1)
    r[(a == b) | (np.isnan(a) & np.isnan(b))] = 0.0
    r[np.isnan(a) != np.isnan(b)] = np.inf
    return r


# ---------------------------------------------------------------- S5 rows (V269 eligibility, local_sop.table columns)
def derive(D):
    """predict_v269.derive: the derived columns of local_sop.table() with the row's own Vlim / Imax"""
    D = D.copy()
    D['chg'] = D.direction == 'charge'
    D['region'] = [ls.region(d, s) for d, s in zip(D.direction, D.soc_point)]
    h10 = D.horizon_s >= 5
    D['Roff'] = np.where(h10, D.Roff10, D.Roff2); D['Rstep'] = np.where(h10, D.Rstep10, D.Rstep2)
    D['prevRstep'] = np.where(h10, D.prev_Rstep10, D.prev_Rstep2); D['prevR'] = np.where(h10, D.prev_R10, D.prev_R2)
    D['prevRoff'] = np.where(h10, D.prev_Roff10, D.prev_Roff2); D['Rq'] = np.where(h10, D.R10_q, D.R2_q)
    D['Iq'] = D.truth_power_W / D.Vlim
    drift = 1e-3 * D.slope_mVs.fillna(0.0) * D.horizon_s
    D['H'] = np.where(D.chg, D.Vlim - D.V_pre - drift, D.V_pre + drift - D.Vlim)
    D['gap'] = np.where(D.chg, D.Vlim - D.V_max, D.V_min - D.Vlim)
    return D


def s5_rows(horizons):
    parts = []
    for h in horizons:
        R = pd.read_csv(XD / f'rows_S5_{h}s.csv'); F = pd.read_csv(XD / f'feats_S5_{h}s.csv')
        X = R.merge(F, on='row_id', how='left')
        has = X.note.isna() if 'note' in X else pd.Series(True, index=X.index)
        ok = X.voltage_limited.astype(bool) & has & (X.off_dur_s >= MIN_LOAD_S)
        X = X[ok].copy(); X['set'] = np.where(X.rest_s >= MIN_REST, 'main', 'secondary'); parts.append(X)
    X = pd.concat(parts, ignore_index=True)
    X['R10_q'] = np.where(X.horizon_s >= 5, X.R_h_final, np.nan)       # the truth pulse's own R: training target only
    X['R2_q'] = np.where(X.horizon_s < 5, X.R_h_final, np.nan); X['R1_q'] = np.nan
    X['evaluation_group'] = 'S5'
    X['ford'] = [FILE_ORDER.index(m.split('/')[-1]) for m in X.member]
    X['tkey'] = X.ford * 1e7 + X.q_row
    assert X.tkey.is_unique
    return derive(X)


def combine(D, X5):
    if X5 is None or not len(X5):
        return D.copy()
    return pd.concat([D, X5[[c for c in X5.columns if c in D.columns]]], ignore_index=True)


# ---------------------------------------------------------------- A: the L pipeline with S5 in every training set
def boundaries_aug(D):
    TH = {}
    for c in CELLS:
        TH[('fold', c)] = V2.learn(D[D.evaluation_group != c], 'min_error')
        TH[('strict', c)] = V2.learn(D[D.evaluation_group.isin([c, 'S5'])], 'min_error')
    return TH


def strict_aug(D0, name, TH):
    """V2.strict with S5 in the single-cell model's training rows (model e + S5 for the rows of the other Samsung cells)"""
    own = D0.set_index('row_id').evaluation_group; out = []
    for e in CELLS:
        D = D0.copy(); D['evaluation_group'] = np.where(D0.evaluation_group.isin([e, 'S5']), e, 'ZZ')
        D['region'] = V2.assign(D, TH[('strict', e)])
        F = V2.folds(name, D); F = F[F.fold == 'ZZ'].copy(); F['cell'] = F.row_id.map(own); F = F[~F.cell.isin([e, 'S5'])]
        out.append(F.assign(name=name, train_cell=e)[['name', 'train_cell', 'row_id', 'cell', 'pred_power_W']])
    return pd.concat(out, ignore_index=True)


def build_A(D, Bm):
    TH = boundaries_aug(D)
    Fv = {n: V2.loco(D, n, TH) for n in NAMES}
    Cv = V2.centres(Fv); Cv = Cv[Cv.row_id.isin(set(Bm.row_id))].reset_index(drop=True)
    Sv = pd.concat([strict_aug(D, n, TH) for n in NAMES], ignore_index=True)
    Sc = S.light_strict_centres(Sv); Bc = S.set_centres(Bm, Cv, 'V286')
    return dict(TH=TH, Fv=Fv, C=Cv, Sc=Sc, Bc=Bc, sm=Sc.set_index(['train_cell', 'row_id']).m)


def bench_J(Bm):
    bench = pd.read_csv(HERE / 'closedloop_v251/src/bench_rows.csv')[['row_id', 'member', 'q_row']]
    J = Bm[GK + ['row_id', 'truth']].merge(bench, on='row_id', how='left').reset_index(drop=True)
    assert len(J) == 672 and J.member.notna().all() and J.q_row.notna().all()
    J['ford'] = [A4.file_order(m) for m in J.member]; J['tkey'] = J.ford * 1e7 + J.q_row
    return J


def requests_A(X, J, groups_all):
    b0, sc = CV.arm_B0(X, TAU, groups_all)
    s0 = {c: float(sc[c]) for c in CELLS}
    fold = {(f, r): float(np.exp(m)) for f, r, m in zip(X['C'].fold, X['C'].row_id, X['C'].m)}
    Jc = J.copy(); Jc['central'] = [fold[(c, r)] for c, r in zip(Jc.evaluation_group, Jc.row_id)]
    ad, scale, clamp = A4.adapt(Jc, s0, A4.GAMMA, A4.TAU, 'cell')
    return dict(B0=b0.reindex(Jc.row_id).to_numpy(float), AD=ad, s0=s0, J=Jc, scale=scale, clamp=clamp)


def score_A(J, req):
    sm, G = A4.score(J, req)
    G = G.copy()
    r = req / J.truth.to_numpy()
    O = J.assign(over=r > 1 + 1e-9, ratio=r).groupby(GK).agg(exceed=('over', 'sum'), ratio_q50=('ratio', 'median')).reset_index()
    G = G.merge(O, on=GK, how='left'); G['pass_original'] = (G.exceed == 0) & (G.ratio_q50 >= 0.80)
    return sm, G


# ---------------------------------------------------------------- B: fold S5 (three Samsung cells) and S5's time order
def build_B(D, X5):
    Dall = combine(D, X5)
    TH5 = V2.learn(Dall[Dall.evaluation_group != 'S5'], 'min_error')
    Dr = Dall.copy(); Dr['region'] = V2.assign(Dr, TH5)
    F = {}
    for n in NAMES:
        f = V2.folds(n, Dr); F[n] = f[f.fold == 'S5'].reset_index(drop=True)
    C5 = V2.centres(F)
    return C5, TH5


def s0_S5(C5, C_loco, Bm):
    """V273 rule for a fourth cell: discharge calibration rows (Samsung) by their own cell's LOCO model (two training
    cells, out of cell), charge rows by the three-cell model in sample; tau 5 %"""
    cen5 = C5.set_index('row_id').m; loco = {(f, r): m for f, r, m in zip(C_loco.fold, C_loco.row_id, C_loco.m)}
    p = np.array([np.exp(loco.get((g, r), np.nan)) if d == 'discharge' else np.exp(cen5.get(r, np.nan))
                  for r, g, d in zip(Bm.row_id, Bm.evaluation_group, Bm.direction)], float)
    t = Bm.truth.to_numpy(); ok = np.isfinite(p)
    s, floor = V5.choose(p[ok], t[ok], TAU)
    return s, dict(scale=s, floor_bound=floor, calibration_rows=int(ok.sum()), dropped=int((~ok).sum()),
                   calibration_int_rate=float(np.mean(s * p[ok] > t[ok] + 1e-9)))


def adapt1(J, s0, gamma, tau, unit):
    """A4.adapt for one cell (state per cell or per direction), J in any order, time key tkey"""
    req = np.full(len(J), np.nan); sc = np.full(len(J), np.nan); ev = np.zeros(len(J), bool)
    cen, tru, dirs = J.central.to_numpy(), J.truth.to_numpy(), J.direction.to_numpy(); state = {}
    for i in np.argsort(J.tkey.to_numpy(), kind='stable'):
        key = 'cell' if unit == 'cell' else dirs[i]
        lg = state.get(key, np.log(s0)); s = float(np.exp(lg)); r = s * cen[i]
        e = bool(r / tru[i] > 1 + 1e-9)
        req[i], sc[i], ev[i] = r, s, e
        state[key] = float(np.clip(lg + gamma * (tau - float(e)), A4.LO, A4.HI))
    return req, sc, ev


def S5_J(X5, C5):
    cen5 = C5.set_index('row_id').m
    J = X5[['row_id', 'set', 'direction', 'temp_C', 'horizon_s', 'member', 'q_row', 'ford', 'tkey', 'soc_point']].copy()
    J['truth'] = X5.truth_power_W.to_numpy(); J['central'] = np.exp(cen5.reindex(J.row_id).to_numpy())
    J['evaluation_group'] = 'S5|' + J.set
    return J.reset_index(drop=True)


def score_S5(J, req):
    s, G = C1.closed_loop(J, req)
    r = req / J.truth.to_numpy()
    O = J.assign(over=r > 1 + 1e-9, ratio=r).groupby(GK).agg(exceed=('over', 'sum'), ratio_q50=('ratio', 'median')).reset_index()
    G = G.merge(O, on=GK, how='left')
    G['eligible'] = G.n >= MIN_GROUP; G['passed'] = (G.util_q50 >= 0.80) & (G.int_rate <= 0.10)
    G['pass_original'] = (G.exceed == 0) & (G.ratio_q50 >= 0.80); G['set'] = G.evaluation_group.str.split('|').str[1]
    out = dict(int_rate=s['int_rate'], util_q50=s['util_q50'], depth_max=s['depth_max'], exceed_rows=int((r > 1 + 1e-9).sum()), rows=int(len(J)))
    for st in ('main', 'secondary'):
        e = G[(G.set == st) & G.eligible]
        out[st] = dict(eligible=int(len(e)), passed=int(e.passed.sum()), original=int(e.pass_original.sum()),
                       small_groups=int(((G.set == st) & ~G.eligible).sum()))
    e = G[G.eligible]; out['all'] = dict(eligible=int(len(e)), passed=int(e.passed.sum()), original=int(e.pass_original.sum()))
    return out, G


def stats(J):
    r = J.central / J.truth; lr = np.log(r)
    return dict(rows=int(len(r)), centre_over_truth_median=float(np.exp(np.median(lr))), rel_abs_median_pct=float(100 * np.median(np.abs(r - 1))),
                q05=float(np.exp(np.quantile(lr, .05))), q95=float(np.exp(np.quantile(lr, .95))))


# ---------------------------------------------------------------- C: fallback order as documented
MISS = []


def basis_fb(D, kind):
    """local_sop.basis with the previous pulse's R alone when the R_off ratio is missing (off basis only without prevR)"""
    if kind == 'prev':
        b = D.prevR * D.Roff / D.prevRoff
        b = b.where(np.isfinite(b), D.prevR.where(np.isfinite(D.prevR) & (D.prevR > 0)))
        return b.where(np.isfinite(b), D.Roff), D.prev_I_last.abs().where(np.isfinite(b), D.I_end.abs())
    return ORIG['basis'](D, kind)


def run_fb(D, specs, first_split, direct=False, gap=False):
    """local_sop2.run with the documented fallback: a 'prev' row whose prediction is missing keeps the prev basis with its
    finite features (one fit per missing pattern on the same training rows) when it has an earlier pulse; only rows
    without one use the off basis"""
    fitf = ls2.fit_region_direct if direct else ls2.fit_region
    predf = ls2.predict_rows_direct if direct else ls2.predict_rows
    X = ls2.features(D)
    D = D.copy()
    D0 = D
    first = D0.prev_R10.isna()
    D0["reg"] = D0.region + np.where(first_split & first, np.where(D0.last_sign == "chg", "|firstcv", "|first"), "")
    parts, coefs = [], {}
    for c in sorted(D0.evaluation_group.unique()):
        D = ls.with_gap(D0, D0.evaluation_group != c) if gap else D0
        tr = D.evaluation_group != c
        for reg in D.reg.unique():
            base_reg = reg.split("|")[0]
            spec = specs.get(reg, specs.get(base_reg, specs.get("*")))
            kind, feats = spec
            m_tr = tr & (D.reg == reg)
            m_all = D.reg == reg
            if kind.endswith("!"):
                kind = kind[:-1]
                par = dict(c0=0.0, a=1.0, c={}, beta=0.0, n=0, fallback=True, fixed=True)
            else:
                par = fitf(D[m_tr], X[m_tr], kind, feats)
            coefs[f"{c}|{reg}"] = dict(par, kind=kind)
            P = predf(D[m_all], X[m_all], kind, par)
            if kind == "prev":
                miss = P.pred_power_W.isna()
                if miss.any():
                    Da, Xa = D[m_all], X[m_all]
                    early = (np.isfinite(Da.prevR) & (Da.prevR > 0) & np.isfinite(Da.prev_I_last) & (Da.prev_I_last.abs() > 0)).to_numpy()
                    pats = {}
                    for i in np.flatnonzero(miss.to_numpy() & early):
                        pats.setdefault(tuple(f for f in feats if np.isfinite(Xa[f].iat[i])), []).append(i)
                    for fs, idx in pats.items():
                        if len(fs) == len(feats):
                            continue                         # all features present: missing for another reason (off basis as before)
                        par_p = fitf(D[m_tr], X[m_tr], "prev", list(fs))
                        P.iloc[idx] = predf(Da.iloc[idx], Xa.iloc[idx], "prev", par_p).to_numpy()
                        MISS.extend((c, r) for r in Da.row_id.iloc[idx])
                    miss = P.pred_power_W.isna()
                    if miss.any():
                        par2 = fitf(D[m_tr], X[m_tr], "off", [])
                        P.loc[miss] = predf(D[m_all][miss.to_numpy()], X[m_all][miss.to_numpy()], "off", par2).to_numpy()
            P.insert(0, "fold", c)
            parts.append(pd.concat([D.loc[m_all, ["row_id", "evaluation_group"]], P], axis=1))
    F = pd.concat(parts)
    return F[F.fold == F.evaluation_group].drop(columns=["fold", "evaluation_group"]), coefs, F


ORIG = dict(basis=ls.basis, run=ls2.run)


@contextlib.contextmanager
def fallback_fixed():
    ls.basis, ls2.run = basis_fb, run_fb
    try:
        yield
    finally:
        ls.basis, ls2.run = ORIG['basis'], ORIG['run']


# ---------------------------------------------------------------- gates
def gate_A(D, Bm, J, groups_all):
    X = build_A(D, Bm); R = requests_A(X, J, groups_all)
    ref_b0 = pd.read_csv(HERE / 'compare_v278/L/requests_B0.csv', float_precision='round_trip').set_index('row_id').pred.reindex(J.row_id).to_numpy()
    ref_ad = pd.read_csv(HERE / 'adaptive_v284/requests_AD.csv', float_precision='round_trip').set_index('row_id').request.reindex(J.row_id).to_numpy()
    g = dict(B0_max_rel=float(np.max(rel(R['B0'], ref_b0))), AD_max_rel=float(np.max(rel(R['AD'], ref_ad))), s0=R['s0'])
    g['ok'] = bool(g['B0_max_rel'] <= TOL and g['AD_max_rel'] <= TOL)
    return g, X, R


def gate_B(D, X5, J):
    """S5 truths changed (x U(0.5, 2) per row, with the truth-derived columns recomputed) leave fold-S5 centres of the S5
    rows unchanged; adapt1 reproduces A4.adapt for each Samsung cell"""
    C5, _ = build_B(D, X5)
    rng = np.random.default_rng(SEED); f = rng.uniform(0.5, 2.0, len(X5))
    Y = X5.copy(); Y['truth_power_W'] = Y.truth_power_W * f; Y['R_h_final'] = Y.R_h_final * f
    Y['R10_q'] = Y.R10_q * f; Y['R2_q'] = Y.R2_q * f; Y['V_min'] = Y.V_min - 0.1 * f; Y['V_max'] = Y.V_max + 0.1 * f; Y['I_last'] = Y.I_last * f
    Y = derive(Y)
    C5b, _ = build_B(D, Y)
    a = C5.set_index('row_id').m.reindex(X5.row_id); b = C5b.set_index('row_id').m.reindex(X5.row_id)
    g = dict(S5_rows=int(len(X5)), S5_centres_finite=int(np.isfinite(a).sum()), truth_change_max_rel=float(np.max(rel(np.exp(a), np.exp(b)))))
    ref = pd.read_csv(HERE / 'adaptive_v284/requests_AD.csv', float_precision='round_trip').set_index('row_id')
    s0 = json.load(open(HERE / 'adaptive_v284/summary.json'))['s0']; d = []
    for c in CELLS:
        Jc = J[J.evaluation_group == c].copy(); Jc['central'] = ref.central.reindex(Jc.row_id).to_numpy()
        req, _, _ = adapt1(Jc, s0[c], A4.GAMMA, A4.TAU, 'cell'); d.append(float(np.max(rel(req, ref.request.reindex(Jc.row_id).to_numpy()))))
    g['adapt1_vs_V284_max_rel'] = max(d)
    g['ok'] = bool(g['truth_change_max_rel'] <= TOL and g['adapt1_vs_V284_max_rel'] <= TOL and g['S5_centres_finite'] == len(X5))
    return g, C5


def gate_C(D):
    """under the fix, local_sop2 fold predictions change only on rows recorded as 'earlier pulse, missing features'"""
    out = {}
    TH = V2.boundaries(D, 'min_error')
    for n in ('L4', 'M6', 'DN6', 'G1'):
        a = V2.loco(D, n, TH)
        MISS.clear()
        with fallback_fixed():
            b = V2.loco(D, n, TH)
        m = a.merge(b, on=['fold', 'row_id'], suffixes=('_a', '_b'))
        ch = m[rel(m.pred_power_W_a, m.pred_power_W_b) > 0]
        recorded = set(MISS)
        out[n] = dict(changed=int(len(ch)), recorded=int(len(recorded)), changed_outside_recorded=int(sum((f, r) not in recorded for f, r in zip(ch.fold, ch.row_id))),
                      changed_rows_by_cell=ch.merge(D[['row_id', 'evaluation_group']], on='row_id').groupby('evaluation_group').size().to_dict())
    a = V2.loco(D, 'prev', TH)
    with fallback_fixed():
        b = V2.loco(D, 'prev', TH)
    m = a.merge(b, on=['fold', 'row_id'], suffixes=('_a', '_b'))
    out['prev'] = dict(changed=int((rel(m.pred_power_W_a, m.pred_power_W_b) > 0).sum()),
                       basis_changed_rows=int((np.isfinite(D.prevR) & ~np.isfinite(D.prevR * D.Roff / D.prevRoff)).sum()))
    out['ok'] = bool(all(out[n]['changed_outside_recorded'] == 0 for n in ('L4', 'M6', 'DN6', 'G1')))
    return out


# ---------------------------------------------------------------- registration
def check_hashes():
    bad = []
    for line in (OUT / 'v286_prereg.sha256').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            h, p = line.split(None, 1); q = HERE / p.strip()
            if not q.is_file() or hashlib.sha256(q.read_bytes()).hexdigest() != h:
                bad.append(p.strip())
    if bad:
        raise SystemExit(f'registration mismatch: {bad}')


# ---------------------------------------------------------------- main
def arm_A(name, D, X5, Bm, J, groups_all, res, Gs, fixed=False):
    with (fallback_fixed() if fixed else contextlib.nullcontext()):
        X = build_A(combine(D, X5), Bm)
    R = requests_A(X, J, groups_all); res[name] = dict(s0=R['s0'], S5_rows_added=int(0 if X5 is None else len(X5)))
    rows = J[['row_id'] + GK + ['truth']].copy()
    for k in ('B0', 'AD'):
        sm, G = score_A(R['J'], R[k]); res[name][k] = sm
        res[name][k]['failed_groups'] = [f'{x.evaluation_group} {x.direction} {x.temp_C} C {x.horizon_s:g} s (u {x.util_q50:.3f}, it {x.int_rate:.3f})' for x in G[~G.passed].itertuples()]
        b2 = G[(G.evaluation_group == 'B') & (G.horizon_s == 2.0)].iloc[0]
        res[name][k]['B_2s'] = dict(util_q50=float(b2.util_q50), interventions=int(b2.interventions), passed=bool(b2.passed))
        Gs.append(G.assign(arm=name, policy=k)); rows[k] = R[k]
        say(f"{name:4s} {k}: closed loop {sm['closed_loop']:2d}/15  original {sm['original']:2d}  exceed {sm['exceed_rows']:3d}  int {100 * sm['int_rate']:.2f} %  "
            f"util {100 * sm['util_q50']:.1f} %  depth max {sm['depth_max']:.3f} | B 2 s u {b2.util_q50:.3f} it {int(b2.interventions)}")
    rows['central'] = R['J'].central.to_numpy(); rows['scale_AD'] = R['scale']
    b2c = R['J'][R['J'].horizon_s == 2.0]; res[name]['B_2s_centres'] = dict(zip(b2c.row_id, b2c.central.round(4)))
    rows.to_csv(OUT / f'requests_{name}.csv', index=False, float_format='%.10g')
    return X


def part_B(name, D, X5, Bm, C_loco, res, Gs, fixed=False):
    with (fallback_fixed() if fixed else contextlib.nullcontext()):
        C5, TH5 = build_B(D, X5)
    s0, cal = s0_S5(C5, C_loco, Bm); J = S5_J(X5, C5)
    assert np.isfinite(J.central).all()
    res[name] = dict(s0=cal, boundaries={k: v['t'] for k, v in TH5.items()},
                     open_loop={f'{st} {d}': stats(g) for (st, d), g in J.groupby(['set', 'direction'])}, open_loop_all=stats(J))
    out = J.copy()
    for pol, unit in (('FIX', None), ('C0', 'cell'), ('C1', 'dir')):
        if unit is None:
            req = s0 * J.central.to_numpy(); sc = np.full(len(J), s0); ev = req / J.truth.to_numpy() > 1 + 1e-9
        else:
            req, sc, ev = adapt1(J, s0, A4.GAMMA, A4.TAU, unit)
        sm, G = score_S5(J, req); res[name][pol] = sm
        res[name][pol]['failed_groups'] = [f'{x.set} {x.direction} {x.temp_C} C {x.horizon_s:g} s (n {x.n}, u {x.util_q50:.3f}, it {x.int_rate:.3f})'
                                           for x in G[G.eligible & ~G.passed].itertuples()]
        Gs.append(G.assign(arm=name, policy=pol)); out[f'req_{pol}'] = req; out[f'scale_{pol}'] = sc; out[f'clamp_{pol}'] = ev
        say(f"{name:4s} {pol}: main {sm['main']['passed']}/{sm['main']['eligible']} (orig {sm['main']['original']})  secondary {sm['secondary']['passed']}/{sm['secondary']['eligible']}  "
            f"all {sm['all']['passed']}/{sm['all']['eligible']}  int {100 * sm['int_rate']:.2f} %  util {100 * sm['util_q50']:.1f} %  exceed {sm['exceed_rows']}  depth max {sm['depth_max']:.3f}")
    up = []                                                             # descriptive: the fixed scale chosen on S5 itself (upper bound)
    for s in V5.GRID:
        sm, _ = score_S5(J, s * J.central.to_numpy()); up.append((sm['main']['passed'], sm['all']['passed'], float(s)))
    res[name]['oracle_fixed_scale_posthoc'] = dict(main_max=int(max(u[0] for u in up)), all_max=int(max(u[1] for u in up)),
                                                   scales_main_max=[u[2] for u in up if u[0] == max(x[0] for x in up)])
    out.sort_values('tkey').to_csv(OUT / f'S5_rows_{name}.csv', index=False, float_format='%.10g')
    return J


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=['gates', 'main']); a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    res = dict(protocol='PROTOCOL_S5_V286.md', stage=a.stage)
    if a.stage == 'main':
        check_hashes(); say('registration: ok')
    Bm = V5.load(); groups_all = Bm[GK].drop_duplicates().sort_values(GK).reset_index(drop=True)
    D = CV.table_from('base', *I6.sources()); J = bench_J(Bm)
    X5 = s5_rows((2, 10)); X2 = X5[X5.horizon_s == 2.0].reset_index(drop=True)
    res['S5_rows'] = {f'{st} {h:g} s': int(n) for (st, h), n in X5.groupby(['set', 'horizon_s']).size().items()}
    say('S5 eligible rows:', res['S5_rows'])
    gA, X0, R0 = gate_A(D, Bm, J, groups_all); res['GA'] = gA; say('GA (zero S5 rows = V278 L B0 and V284 AD):', gA)
    gB, _ = gate_B(D, X5, J); res['GB'] = gB; say('GB (S5 truth invariance, adapt1 = V284):', gB)
    gC = gate_C(D); res['GC'] = gC; say('GC (fallback scope):', gC)
    if a.stage == 'gates':
        dump(OUT / 'gates.json', V2.clean(res)); (OUT / 'gates.log').write_text('\n'.join(LOG) + '\n'); return
    if not (gA['ok'] and gB['ok'] and gC['ok']):
        dump(OUT / 'summary.json', V2.clean(res)); raise SystemExit('a gate failed (protocol: stop)')
    GA_, GB_, XS = [], [], {}
    for name, Xs, fx in (('A2', X2, False), ('A12', X5, False), ('F', None, True), ('A2F', X2, True)):
        XS[name] = arm_A(name, D, Xs, Bm, J, groups_all, res, GA_, fixed=fx)
    pd.concat(GA_, ignore_index=True).to_csv(OUT / 'groups_A.csv', index=False)
    for name, fx, Cl in (('B', False, X0['C']), ('BF', True, XS['F']['C'])):   # BF: the fixed predictors' LOCO centres for its calibration
        part_B(name, D, X5, Bm, Cl, res, GB_, fixed=fx)
    pd.concat(GB_, ignore_index=True).to_csv(OUT / 'groups_S5.csv', index=False)
    ref = json.load(open(HERE / 'adaptive_v284/summary.json'))['AD']
    res['reference_V284_AD'] = {k: ref[k] for k in ('closed_loop', 'original', 'exceed_rows', 'int_rate', 'util_q50', 'depth_max')}
    # verdicts (protocol)
    passed_V284 = set(map(tuple, pd.read_csv(HERE / 'adaptive_v284/groups_AD.csv').query('passed')[GK].astype(str).values))
    def kept(name):
        G = pd.read_csv(OUT / 'groups_A.csv').query('arm == @name and policy == "AD"')
        now = set(map(tuple, G[G.passed][GK].astype(str).values)); return passed_V284 <= now
    a2 = res['A2']['AD']
    if a2['B_2s']['passed'] and a2['closed_loop'] >= 13:
        vA = '고침'
    elif (a2['B_2s']['util_q50'] >= 0.6816 + 0.05) or (a2['B_2s']['passed'] and a2['closed_loop'] < 13):
        vA = '부분'
    else:
        vA = '효과 없음'
    b = res['B']; fixm, c0m = b['FIX']['main']['passed'], b['C0']['main']['passed']
    if b['FIX']['main']['passed'] == b['FIX']['main']['eligible']:
        vB = '판정 불가(천장)'
    elif c0m >= fixm + 2 and b['C0']['int_rate'] <= 0.10:
        vB = '외부 셀에서 유효'
    elif (c0m == fixm + 1 and b['C0']['int_rate'] <= 0.10) or (c0m >= fixm + 2):
        vB = '부분'
    else:
        vB = '효과 없음'
    f = res['F']['AD']
    if f['closed_loop'] >= 14 and kept('F'):
        vC = '개선'
    elif f['closed_loop'] >= 14 or (f['B_2s']['passed'] and f['closed_loop'] >= 13):
        vC = '부분'
    else:
        vC = '변화 없음 또는 악화'
    res['verdicts'] = dict(A_S5_training=vA, B_external_adaptation=vB, C_fallback=vC)
    say('verdicts:', res['verdicts'])
    dump(OUT / 'summary.json', V2.clean(res)); (OUT / 'run.log').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
