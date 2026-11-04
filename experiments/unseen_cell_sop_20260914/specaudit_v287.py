"""V287 (PROTOCOL_SPECAUDIT_V287.md, AUDIT_V287.md): the local predictors with every deviation of the frozen list fixed.
  R12  relax features: prev / prev2 pulses after >= 60 s rest (R1) searched from the last load sample (R2); rows whose
       chosen pulse changes take the fixed recomputation (specaudit_v287/feats_fixed.csv), all others keep relax.csv
  L1   local_sop.fit regresses on the measured end current |I_last| (current-limited exclusion unchanged)
  S1   local_sop2: '|firstcv' only when the last load was the CC-CV charge (charge ending at V_end >= 4.19 V)
  S2   local_sop2: the off-basis fallback only for rows without an earlier pulse
  E1   events R1 / R2: no recomputation (no predictor input or target changes; checked in G3)
Ls = all fixes. A row without a centre in its fold gets a request of 0 W (no estimate, no power); calibration rows
without a centre are dropped (V272). Requests B0 (V273 rule, tau 5 %) and AD (V284 C0).
usage: python specaudit_v287.py --stage gates   (G1-G4; no fixed configuration is scored)
       python specaudit_v287.py --stage main    (once, after specaudit_v287/v287_prereg_stage3.sha256)"""
import argparse
import contextlib
import hashlib
import inspect
import json
import warnings

import numpy as np, pandas as pd
from run_envelope import HERE, dump
import closedloop_v251 as C1, closedloop_v255 as V5, closedloop_v256 as V6
import leakfree_v272 as V2
import strictfull_v273 as S
import invariance_v276 as I6
import compare_v278 as CV
import adaptive_v284 as A4

OUT = HERE / 'specaudit_v287'
CV.TMP = OUT / 'tmp'
warnings.simplefilter('ignore')
ls, ls2 = V2.ls, V2.ls2
NAMES = V2.NAMES; CELLS = V5.CELLS; GK = C1.GK; TAU = 0.05; TOL = 1e-12
FIXES = ('R12', 'L1', 'S1', 'S2')
FLAGS = dict.fromkeys(FIXES, False)
ORIG = dict(fit=ls.fit, run=ls2.run)
S2_LOST = []
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


# ---------------------------------------------------------------- R12: spliced relax features
def spliced_relax(F0):
    """relax.csv with the prev_* / prev2_* columns of the rows whose chosen pulse changed under R1 / R2 taken from the
    fixed recomputation (NaN where the fixed search finds no pulse)"""
    Fo = pd.read_csv(OUT / 'feats_orig.csv').set_index('row_id'); Fx = pd.read_csv(OUT / 'feats_fixed.csv').set_index('row_id')
    F = F0.copy().set_index('row_id'); info = {}
    for pre in ('prev_', 'prev2_'):
        k = f'{pre}k0_abs'
        a, b = Fo[k].reindex(F.index), Fx[k].reindex(F.index)
        ch = ~((a == b) | (a.isna() & b.isna()))
        cols = [c for c in F.columns if c.startswith(pre) and (pre == 'prev2_' or not c.startswith('prev2_'))]
        for c in cols:
            v = Fx[c].reindex(F.index) if c in Fx.columns else pd.Series(np.nan, index=F.index)
            F.loc[ch, c] = v[ch].astype(float) if F[c].dtype != object else v[ch]
        info[pre] = dict(rows_changed=int(ch.sum()), lost_pulse=int((ch & b.isna()).sum()), gained_pulse=int((ch & a.isna()).sum()),
                         changed_rows=sorted(F.index[ch].tolist()), columns=len(cols))
    return F.reset_index()[F0.columns], info


# ---------------------------------------------------------------- L1, S1, S2: replacement functions
def fit_audit(T, kind, fit_beta):
    """local_sop.fit; L1: the regression current is the measured end current |I_last|"""
    B, Iref = ls.basis(T, kind)
    y = np.log(T.Rq / B)
    x = np.log((T.I_last.abs() if FLAGS['L1'] else T.Iq) / Iref)
    ok = np.isfinite(y) & np.isfinite(x) & (T.Iq < 0.98 * T.Imax)
    out = {}
    for r, m in T[ok].groupby("region").groups.items():
        yy, xx = y[m], x[m]
        if fit_beta and len(m) >= 8 and xx.std() > 0.05:
            beta, lk = np.polyfit(xx, yy, 1)
        else:
            beta, lk = 0.0, float(np.median(yy))
        out[r] = (float(np.exp(lk)), float(beta))
    return out


def run_audit(D, specs, first_split, direct=False, gap=False):
    """local_sop2.run; S1: firstcv only after the CC-CV charge (V_end >= 4.19 V); S2: off fallback only without an
    earlier pulse (rows with an earlier pulse and no prediction keep NaN)"""
    fitf = ls2.fit_region_direct if direct else ls2.fit_region
    predf = ls2.predict_rows_direct if direct else ls2.predict_rows
    X = ls2.features(D)
    D = D.copy()
    D0 = D
    first = D0.prev_R10.isna()
    cv = ((D0.last_sign == "chg") & (D0.V_end >= 4.19)) if FLAGS['S1'] else (D0.last_sign == "chg")
    D0["reg"] = D0.region + np.where(first_split & first, np.where(cv, "|firstcv", "|first"), "")
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
                if FLAGS['S2']:
                    Da = D[m_all]
                    noprev = ~(np.isfinite(Da.prevR) & np.isfinite(Da.prev_I_last)).to_numpy()
                    S2_LOST.extend((c, r) for r in Da.row_id[miss.to_numpy() & ~noprev])
                    miss = miss & noprev
                if miss.any():
                    par2 = fitf(D[m_tr], X[m_tr], "off", [])
                    P.loc[miss] = predf(D[m_all][miss.to_numpy()], X[m_all][miss.to_numpy()], "off", par2).to_numpy()
            P.insert(0, "fold", c)
            parts.append(pd.concat([D.loc[m_all, ["row_id", "evaluation_group"]], P], axis=1))
    F = pd.concat(parts)
    return F[F.fold == F.evaluation_group].drop(columns=["fold", "evaluation_group"]), coefs, F


@contextlib.contextmanager
def flags(**kw):
    old = dict(FLAGS); FLAGS.update({k: False for k in FIXES}); FLAGS.update(kw)
    ls.fit, ls2.run = fit_audit, run_audit
    try:
        yield
    finally:
        ls.fit, ls2.run = ORIG['fit'], ORIG['run']; FLAGS.clear(); FLAGS.update(old)


# ---------------------------------------------------------------- the L pipeline without the stop on missing centres
def set_centres_nostop(B, C):
    B = B.copy(); P = C.pivot(index='row_id', columns='fold', values='m').reindex(B.row_id)
    for c in CELLS:
        B[f'sop_{c}'] = np.exp(P[c].to_numpy())
    return B


def build(name, src, Bm, **kw):
    R, F, E, CVf = src
    if kw.get('R12'):
        F, _ = spliced_relax(F)
    D = CV.table_from(name, R, F, E, CVf)
    S2_LOST.clear()
    with flags(**kw):
        TH = V2.boundaries(D, 'min_error')
        Fv = {n: V2.loco(D, n, TH) for n in NAMES}; Cv = V2.centres(Fv)
        Sv = pd.concat([V2.strict(D, n, TH) for n in NAMES], ignore_index=True)
    Sc = S.light_strict_centres(Sv); Bc = set_centres_nostop(Bm, Cv)
    own = np.array([Bc.at[i, f'sop_{g}'] for i, g in zip(Bc.index, Bc.evaluation_group)])
    return dict(D=D, C=Cv, Sc=Sc, Bc=Bc, sm=Sc.set_index(['train_cell', 'row_id']).m, Fv=Fv,
                missing_own=int((~np.isfinite(own)).sum()), missing_any=int(Bc[[f'sop_{c}' for c in CELLS]].isna().any(axis=1).sum()),
                s2_lost=sorted(set(S2_LOST)))


def requests_B0(X, tau, J):
    """S.score_single's calibration (rows without a centre dropped) and requests; an evaluated row without a centre: 0 W"""
    fn = S.look(X['Bc']); cal = S.cal_free_single(S.dis_from(X['sm']), lambda c, d, rr: fn[(c, rr)])
    Bc = X['Bc']; pred = pd.Series(np.nan, index=Bc.row_id.to_numpy()); scales = {}
    for c in CELLS:
        p, t, dropped = cal(Bc, c); s, floor = V5.choose(p, t, tau)
        scales[c] = dict(scale=float(s), calibration_rows=int(len(p)), dropped_without_centre=int(dropped))
        ev = Bc[Bc.evaluation_group == c]
        pred.loc[ev.row_id.to_numpy()] = s * np.nan_to_num(ev[f'sop_{c}'].to_numpy(float), nan=0.0)
    return pred.reindex(J.row_id).to_numpy(float), scales


def central_own(X, J):
    fold = {(f, r): float(np.exp(m)) for f, r, m in zip(X['C'].fold, X['C'].row_id, X['C'].m)}
    return np.nan_to_num(np.array([fold.get((c, r), np.nan) for c, r in zip(J.evaluation_group, J.row_id)], float), nan=0.0)


def requests_all(X, J, tau, gamma=A4.GAMMA, tau_star=A4.TAU):
    b0, sc = requests_B0(X, tau, J)
    Jc = J.copy(); Jc['central'] = central_own(X, J)
    ad, scale, clamp = A4.adapt(Jc, {c: sc[c]['scale'] for c in CELLS}, gamma, tau_star, 'cell')
    return b0, ad, sc, Jc


def score(J, req):
    sm, G = A4.score(J, req)
    r = req / J.truth.to_numpy()
    O = J.assign(over=r > 1 + 1e-9, ratio=r).groupby(GK).agg(exceed=('over', 'sum'), ratio_q50=('ratio', 'median')).reset_index()
    G = G.merge(O, on=GK, how='left'); G['pass_original'] = (G.exceed == 0) & (G.ratio_q50 >= 0.80)
    sm['failed_groups'] = [f'{x.evaluation_group} {x.direction} {x.temp_C} C {x.horizon_s:g} s (u {x.util_q50:.3f}, it {x.int_rate:.3f})' for x in G[~G.passed].itertuples()]
    return sm, G


def bench_J(Bm):
    bench = pd.read_csv(HERE / 'closedloop_v251/src/bench_rows.csv')[['row_id', 'member', 'q_row']]
    J = Bm[GK + ['row_id', 'truth']].merge(bench, on='row_id', how='left').reset_index(drop=True)
    assert len(J) == 672
    J['ford'] = [A4.file_order(m) for m in J.member]; J['tkey'] = J.ford * 1e7 + J.q_row
    return J


# ---------------------------------------------------------------- gates
def gates(src, Bm, J):
    G = {}
    ref_b0 = pd.read_csv(HERE / 'compare_v278/L/requests_B0.csv', float_precision='round_trip').set_index('row_id').pred.reindex(J.row_id).to_numpy()
    ref_ad = pd.read_csv(HERE / 'adaptive_v284/requests_AD.csv', float_precision='round_trip').set_index('row_id').request.reindex(J.row_id).to_numpy()
    X0 = build('L_flags_off', src, Bm)                   # every fix off, through the replacement functions
    b0, ad, sc, _ = requests_all(X0, J, TAU)
    G['G1'] = dict(B0_max_rel=float(np.max(rel(b0, ref_b0))), AD_max_rel=float(np.max(rel(ad, ref_ad))), missing_centres=X0['missing_any'])
    G['G1']['ok'] = bool(G['G1']['B0_max_rel'] <= TOL and G['G1']['AD_max_rel'] <= TOL and X0['missing_any'] == 0)
    Fo = pd.read_csv(OUT / 'feats_orig.csv')
    g2a = I6.feature_gate(Fo.drop(columns=[c for c in Fo.columns if c.endswith('k0_abs')]), src[1])
    Fs, info = spliced_relax(src[1])
    cols = [c for c in src[1].columns if c != 'row_id']
    M = Fs.merge(src[1], on='row_id', suffixes=('', '_0'))
    changed = set()
    for c in cols:
        a, b = M[c], M[c + '_0']
        d = ~((a == b) | (a.isna() & b.isna()))
        changed |= set(M.row_id[d])
    allowed = set(info['prev_']['changed_rows']) | set(info['prev2_']['changed_rows'])
    G['G2'] = dict(recompute_vs_relax=g2a, splice=info | {'rows_with_any_change': len(changed), 'changed_outside_pulse_change': len(changed - allowed)})
    for pre in ('prev_', 'prev2_'):
        G['G2']['splice'][pre].pop('changed_rows')
    G['G2']['ok'] = bool(g2a['ok'] and not (changed - allowed))
    D = X0['D']
    src_ls = inspect.getsource(ls) + inspect.getsource(ls2)
    G['G3'] = dict(horizon2_rows=int((D.horizon_s < 5).sum()), horizon2_rows_with_event=int(((D.horizon_s < 5) & (D.R10_q.notna() | D.I_last.notna())).sum()),
                   R1_q_read_by_predictors='R1_q' in src_ls.replace('"R1_q"', '').replace("'R1_q'", ''))
    G['G3']['ok'] = bool(G['G3']['horizon2_rows_with_event'] == 0 and not G['G3']['R1_q_read_by_predictors'])
    Dc = D.copy(); Dc['region'] = V2.assign(Dc, V2.boundaries(D, 'min_error')[('fold', 'A1')])
    first = Dc.prev_R10.isna(); chg = Dc.last_sign == 'chg'
    tr = Dc.Rq.notna() & Dc.I_last.notna() & (Dc.Iq < 0.98 * Dc.Imax)
    G['G4'] = dict(S1_rows_firstcv_to_first=int((first & chg & ~(Dc.V_end >= 4.19)).sum()), S1_rows_firstcv_kept=int((first & chg & (Dc.V_end >= 4.19)).sum()),
                   L1_training_rows=int(tr.sum()), L1_current_ratio_I_last_over_Iq=dict(zip(('q01', 'q50', 'q99'), np.round(np.quantile((Dc.I_last.abs() / Dc.Iq)[tr], [.01, .5, .99]), 5).tolist())))
    XS2 = build('S2_only_gate', src, Bm, S2=True)
    G['G4']['S2_fold_rows_losing_off_fallback'] = len(XS2['s2_lost']); G['G4']['S2_rows'] = sorted({r for _, r in XS2['s2_lost']})
    G['ok'] = bool(G['G1']['ok'] and G['G2']['ok'] and G['G3']['ok'])
    return G, X0


def check_hashes():
    bad = []
    for f in ('v287_prereg_stage1.sha256', 'v287_prereg_stage2.sha256', 'v287_prereg_stage3.sha256'):
        for line in (OUT / f).read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                h, p = line.split(None, 1); q = HERE / p.strip()
                if not q.is_file() or hashlib.sha256(q.read_bytes()).hexdigest() != h:
                    bad.append(p.strip())
    if bad:
        raise SystemExit(f'registration mismatch: {bad}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=['gates', 'main']); a = ap.parse_args()
    OUT.mkdir(exist_ok=True); res = dict(protocol='PROTOCOL_SPECAUDIT_V287.md', audit='AUDIT_V287.md', stage=a.stage)
    if a.stage == 'main':
        check_hashes(); say('registration: ok')
    Bm = V5.load(); groups_all = Bm[GK].drop_duplicates().sort_values(GK).reset_index(drop=True)
    src = I6.sources(); J = bench_J(Bm)
    G, X0 = gates(src, Bm, J); res['gates'] = G
    for k in ('G1', 'G2', 'G3', 'G4'):
        say(k, json.dumps(V2.clean(G[k]), ensure_ascii=False)[:900])
    if a.stage == 'gates':
        dump(OUT / 'gates.json', V2.clean(res)); (OUT / 'gates.log').write_text('\n'.join(LOG) + '\n'); return
    if not G['ok']:
        dump(OUT / 'summary.json', V2.clean(res)); raise SystemExit('a gate failed (protocol: stop)')
    configs = [('Ls', dict(R12=True, L1=True, S1=True, S2=True))] + [(f'only_{k}', {k: True}) for k in FIXES]
    Gall, T = [], []
    for name, kw in configs:
        X = build(name, src, Bm, **kw)
        res[name] = dict(flags=kw, missing_centres_own_fold=X['missing_own'], rows_missing_any_fold=X['missing_any'], S2_fold_rows=len(X['s2_lost']))
        for tau in [float(t) for t in V6.TAUS]:
            b0, ad, sc, Jc = requests_all(X, J, tau)
            for pol, req in (('B0', b0), ('AD', ad)):
                sm, Gk = score(Jc, req)
                T.append(dict(config=name, tau=tau, policy=pol, **{k: sm[k] for k in ('closed_loop', 'original', 'exceed_rows', 'int_rate', 'util_q50', 'depth_max')}))
                if abs(tau - TAU) < 1e-12:
                    res[name][pol] = sm; Gall.append(Gk.assign(config=name, policy=pol))
                    if name == 'Ls':
                        pd.DataFrame({'row_id': J.row_id, 'request': req, 'central': Jc.central, 'truth': J.truth}).to_csv(OUT / f'requests_Ls_{pol}.csv', index=False, float_format='%.10g')
            if abs(tau - TAU) < 1e-12:
                res[name]['scales_tau5'] = {c: v['scale'] for c, v in sc.items()}; res[name]['calibration_tau5'] = sc
        s5 = res[name]
        say(f"{name:8s} B0 {s5['B0']['closed_loop']:2d}/15 (orig {s5['B0']['original']})  AD {s5['AD']['closed_loop']:2d}/15 (orig {s5['AD']['original']}, exceed {s5['AD']['exceed_rows']}, "
            f"int {100 * s5['AD']['int_rate']:.2f} %, util {100 * s5['AD']['util_q50']:.1f} %, depth {s5['AD']['depth_max']:.3f}) | scales {s5['scales_tau5']} | missing centres {s5['missing_centres_own_fold']}")
        if name == 'Ls':
            Jc = J.copy(); Jc['central'] = central_own(X, J)
            grid = []
            b0, _, sc, _ = requests_all(X, J, TAU)
            for g in A4.GRID_G:
                for t in A4.GRID_T:
                    adg, _, _ = A4.adapt(Jc, {c: sc[c]['scale'] for c in CELLS}, g, t, 'cell'); smg, _ = score(Jc, adg)
                    grid.append(dict(gamma=g, tau_star=t, closed_loop=smg['closed_loop'], original=smg['original'], int_rate=smg['int_rate'], util_q50=smg['util_q50']))
            pd.DataFrame(grid).to_csv(OUT / 'Ls_adapt_grid.csv', index=False)
            res['Ls']['adapt_grid'] = {f"{x['gamma']}|{x['tau_star']}": x['closed_loop'] for x in grid}
            b2 = J[J.horizon_s == 2.0].row_id.tolist()
            res['Ls']['B_2s_centres'] = {r: float(v) for r, v in zip(J.row_id, Jc.central) if r in b2}
    Tt = pd.DataFrame(T); Tt.to_csv(OUT / 'tau_range.csv', index=False)
    pd.concat(Gall, ignore_index=True).to_csv(OUT / 'groups_tau5.csv', index=False)
    res['tau_4_6'] = {f'{c} {p}': [int(g.closed_loop.min()), float(g.closed_loop.median()), int(g.closed_loop.max())] for (c, p), g in Tt.groupby(['config', 'policy'])}
    ref = dict(L_B0_tau5=9, L_AD_tau5=13, L_B0_tau_4_6=[9, 9.0, 9], L_AD_tau_4_6=[11, 12.0, 13])
    res['reference_L'] = ref
    GL = pd.read_csv(HERE / 'adaptive_v284/groups_AD.csv').set_index(GK); GLb = pd.read_csv(HERE / 'compare_v278/L/groups_tau5.csv').query('arm == "B0"').set_index(GK)
    Gl = pd.concat(Gall).query('config == "Ls"')
    for pol, Gref in (('AD', GL), ('B0', GLb)):
        g = Gl[Gl.policy == pol].set_index(GK)
        res['Ls'][f'{pol}_group_changes_vs_L'] = [f'{i[0]} {i[1]} {i[2]} C {i[3]:g} s: {"pass" if Gref.loc[i, "passed"] else "fail"} -> {"pass" if g.loc[i, "passed"] else "fail"} '
                                                  f'(u {Gref.loc[i, "util_q50"]:.3f} -> {g.loc[i, "util_q50"]:.3f}, it {int(Gref.loc[i, "interventions"])} -> {int(g.loc[i, "interventions"])})'
                                                  for i in g.index if bool(Gref.loc[i, 'passed']) != bool(g.loc[i, 'passed'])]
    a5, m5 = res['Ls']['AD']['closed_loop'], res['tau_4_6']['Ls AD'][1]
    res['summary_statement'] = '나아짐' if (a5 > 13 and m5 >= 12) or (a5 >= 13 and m5 > 12) else ('같음' if (a5 == 13 and m5 == 12) else ('나빠짐' if (a5 < 13 and m5 <= 12) or (a5 <= 13 and m5 < 12) else '엇갈림'))
    say('tau 4-6 %:', res['tau_4_6'])
    say('Ls group changes vs L (AD):', res['Ls']['AD_group_changes_vs_L'])
    say('Ls group changes vs L (B0):', res['Ls']['B0_group_changes_vs_L'])
    say('Ls B 2 s centres:', res['Ls']['B_2s_centres'])
    say('summary (Ls vs L):', res['summary_statement'])
    dump(OUT / 'summary.json', V2.clean(res)); (OUT / 'run.log').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
