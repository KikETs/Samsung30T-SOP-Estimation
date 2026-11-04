"""V320 step 2 (PROTOCOL_POSTHOC_LSAD_V320.md): post hoc analyses and information-flow checks of the manuscript
configuration Ls + AD (V287 Ls, all audit fixes; V284 AD at tau = 5 %), with the definitions of V276 (invariance tests),
V281 / V283 (post hoc analyses). No new model, rule or training; the only new definition is S02_AD (the scale-factor
sensitivity of V283 applied to the AD starting scales).
  gates  G1 the V287 Ls build reproduces requests_Ls_{B0,AD}.csv (relative 1e-9: saved with 10 significant digits,
            AMENDMENT_1), the intervention flags and group verdicts exactly, and the tau 4-6 % counts of tau_range.csv;
         G2 the step-1 ref features equal specaudit_v287/feats_fixed.csv (V276 feature gate);
         G3 V284 adapt on the recomputed centres and the Ls tau-5 % scales reproduces the recomputed AD requests (1e-12)
  the post hoc analyses and I2 / I3 use the gated recomputation (unrounded), not the rounded saved files (AMENDMENT_1)
  I1     V276 T1-T4 for the Ls centres, B0 requests at the nine tau values and scales of the evaluated cell
         (T1 / T3 against R0 = the V287 Ls build; T2 / T4 against Rref = Ls on the recomputed step-1 features)
  I2     AD causality: the request of every row is unchanged when the truths of that row and of every later row of its
         cell (processing order) are multiplied by U(0.5, 2)
  I3     AD reads the clamp event only: truths changed without changing any row's event leave every request unchanged
  post hoc on the tau-5 % requests: D (intervened rows, depth), E (V253 delivery simulation), S01 (thresholds),
         S02_AD, K6 (first queries), O10 (centre over-prediction tail), O2 (original criterion), D08_Ls, AD trajectory,
         T_Ls (local-predictor stage timing, three builds), X01_Ls (internal)
usage: OMP_NUM_THREADS=1 python posthoc_lsad_v320.py   (torch_env; once, after step 1 and posthoc_lsad_v320/v320_prereg.sha256)"""
import hashlib
import json
import time
import warnings

import numpy as np, pandas as pd
from run_envelope import HERE, dump
import closedloop_v251 as C1, closedloop_v255 as V5, closedloop_v256 as V6
import leakfree_v272 as V2
import invariance_v276 as I6
import compare_v278 as CV
import adaptive_v284 as A4
import specaudit_v287 as SA

warnings.simplefilter('ignore')
OUT = HERE / 'posthoc_lsad_v320'
CV.TMP = OUT / 'tmp'
GK = C1.GK; CELLS = V5.CELLS; TOL = 1e-12; SEED = 320; LOG = []
TOL_SAVED = 1e-9          # AMENDMENT_1: V287 saved the requests with 10 significant digits (rounding <= 5e-10 relative)
TAUS = [float(t) for t in V6.TAUS]
SAV = HERE / 'specaudit_v287'


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def check_hashes():
    bad = []
    for line in (OUT / 'v320_amend1.sha256').read_text().splitlines():          # AMENDMENT_1: the amended registration list
        if line.strip() and not line.startswith('#'):
            h, p = line.split(None, 1); q = HERE / p.strip()
            if not q.is_file() or hashlib.sha256(q.read_bytes()).hexdigest() != h:
                bad.append(p.strip())
    if bad:
        raise SystemExit(f'registration mismatch: {bad}')


def ls_build(name, src, Bm, recomputed=False):
    """V287 Ls (all fixes). recomputed: the relax table already holds the FIXED features of every row (no splice)"""
    return SA.build(name, src, Bm, R12=not recomputed, L1=True, S1=True, S2=True)


def b0_all_tau(X, J):
    """own-fold centres, B0 requests at the nine tau values and the scales (the V276 comparison frames)"""
    cen = SA.central_own(X, J); rows, sc = [], []
    for tau in TAUS:
        pred, scales = SA.requests_B0(X, tau, J)
        rows.append(pd.DataFrame({'row_id': J.row_id, 'evaluation_group': J.evaluation_group, 'central': cen, 'pred': pred, 'tau': tau}))
        sc += [dict(tau=tau, cell=c, scale=v['scale'], calibration_rows=v['calibration_rows']) for c, v in scales.items()]
    return pd.concat(rows, ignore_index=True), pd.DataFrame(sc)


def maxrel(a, b):
    return float(np.max(SA.rel(a, b)))


def ad_order(J, c):
    m = (J.evaluation_group == c).to_numpy()
    return J.index[m][np.argsort(J.loc[m, 'tkey'].to_numpy(), kind='stable')]


def groups_pass(J, req, u=0.80, i=0.10):
    _, G = C1.closed_loop(J.assign(pred=req), req)
    return int(((G.util_q50 >= u) & (G.int_rate <= i)).sum()), G


def main():
    OUT.mkdir(exist_ok=True)
    check_hashes(); say('registration: ok')
    res = dict(protocol='PROTOCOL_POSTHOC_LSAD_V320.md')
    Bm = V5.load(); src = I6.sources(); F0 = src[1]; J = SA.bench_J(Bm)
    S287 = json.load(open(SAV / 'summary.json'))
    s0 = {c: float(v) for c, v in S287['Ls']['scales_tau5'].items()}
    sav_b0 = pd.read_csv(SAV / 'requests_Ls_B0.csv', float_precision='round_trip').set_index('row_id').reindex(J.row_id)
    sav_ad = pd.read_csv(SAV / 'requests_Ls_AD.csv', float_precision='round_trip').set_index('row_id').reindex(J.row_id)

    # ---------------- gates
    G = res['gates'] = {}
    X0 = ls_build('R0', src, Bm)
    b0, ad, sc, Jc0 = SA.requests_all(X0, J, 0.05)
    tr = pd.read_csv(SAV / 'tau_range.csv'); tr = tr[tr.config == 'Ls']; bad = []
    for tau in TAUS:
        bb, aa, _, Jt = SA.requests_all(X0, J, tau)
        for pol, req in (('B0', bb), ('AD', aa)):
            sm, _ = SA.score(Jt, req); ref = tr[(tr.policy == pol) & np.isclose(tr.tau, tau)].iloc[0]
            if int(sm['closed_loop']) != int(ref.closed_loop):
                bad.append((pol, tau))
    tt = J.truth.to_numpy(); flag_mis, verdict_mis = {}, {}
    for pol, rec, sv in (('B0', b0, sav_b0.request.to_numpy()), ('AD', ad, sav_ad.request.to_numpy())):   # AMENDMENT_1: discrete outcomes exact
        flag_mis[pol] = int(((rec / tt > 1 + 1e-9) != (sv / tt > 1 + 1e-9)).sum())
        _, Gr = SA.score(Jc0, rec); _, Gv = SA.score(Jc0, sv)
        verdict_mis[pol] = int(((Gr.passed.to_numpy() != Gv.passed.to_numpy()) | (Gr.pass_original.to_numpy() != Gv.pass_original.to_numpy())).sum())
    G['G1'] = dict(B0_max_rel=maxrel(b0, sav_b0.request.to_numpy()), AD_max_rel=maxrel(ad, sav_ad.request.to_numpy()),
                   centre_max_rel=maxrel(Jc0.central.to_numpy(), sav_ad.central.to_numpy()), tolerance_saved=TOL_SAVED,
                   intervention_flag_mismatch=flag_mis, group_verdict_mismatch=verdict_mis, tau_4_6_mismatch=bad,
                   scales={c: v['scale'] for c, v in sc.items()})
    G['G1']['ok'] = bool(G['G1']['B0_max_rel'] <= TOL_SAVED and G['G1']['AD_max_rel'] <= TOL_SAVED and G['G1']['centre_max_rel'] <= TOL_SAVED and not bad
                         and not any(flag_mis.values()) and not any(verdict_mis.values()) and all(abs(G['G1']['scales'][c] - s0[c]) <= TOL for c in CELLS))
    Fref = pd.read_csv(OUT / 'feats_ref_fixed.csv'); FT2 = pd.read_csv(OUT / 'feats_T2_fixed.csv'); FT4 = pd.read_csv(OUT / 'feats_T4_fixed.csv')
    G['G2'] = I6.feature_gate(Fref, pd.read_csv(SAV / 'feats_fixed.csv'))
    Jc = J.copy(); Jc['central'] = Jc0.central.to_numpy()                       # AMENDMENT_1: recomputed (unrounded) centres
    req0, sc0, ev0 = A4.adapt(Jc, s0, A4.GAMMA, A4.TAU, 'cell')
    G['G3'] = dict(AD_max_rel_vs_recomputed=maxrel(req0, ad), AD_max_rel_vs_saved=maxrel(req0, sav_ad.request.to_numpy()))
    G['G3']['ok'] = bool(G['G3']['AD_max_rel_vs_recomputed'] <= TOL and G['G3']['AD_max_rel_vs_saved'] <= TOL_SAVED)
    for k in ('G1', 'G2', 'G3'):
        say(k, json.dumps(V2.clean(G[k]), ensure_ascii=False)[:600])
    if not all(G[k]['ok'] for k in ('G1', 'G2', 'G3')):
        dump(OUT / 'summary.json', V2.clean(res)); (OUT / 'run.log').write_text('\n'.join(LOG) + '\n')
        raise SystemExit('a gate failed (protocol: stop, nothing reported)')

    # ---------------- I1: V276 T1-T4 on Ls
    R0 = b0_all_tau(X0, J)
    src_ref = (src[0], I6.with_feats(F0, Fref, src[0].row_id.tolist()), src[2], src[3])
    Rref = b0_all_tau(ls_build('Rref', src_ref, Bm, recomputed=True), J)
    rng = np.random.default_rng(I6.SEED)                                    # the V276 factors and permutations
    fac = I6.factors(src[0], rng); Rs = src[0].sort_values('row_id')
    perm_soc = pd.Series(rng.permutation(Rs.soc_point.to_numpy()), index=Rs.row_id.to_numpy())
    perm_ds = pd.Series(rng.permutation(Rs.ds_query.to_numpy()), index=Rs.row_id.to_numpy())
    T = res['I1'] = {}; rows = []
    for c in CELLS:
        for t in ('T1', 'T3', 'T2', 'T4'):
            if t == 'T1':
                s, b, info = I6.t1(src, Bm, c, fac); out = b0_all_tau(ls_build(f'T1_{c}', s, b), J); ref, rn = R0, 'R0'
            elif t == 'T3':
                s, b, info = I6.t3(src, Bm, c, perm_soc, perm_ds); out = b0_all_tau(ls_build(f'T3_{c}', s, b), J); ref, rn = R0, 'R0'
            elif t == 'T2':
                s, b, info = I6.t2(src_ref, Bm, c, FT2); out = b0_all_tau(ls_build(f'T2_{c}', s, b, recomputed=True), J); ref, rn = Rref, 'Rref'
            else:
                s, b, info = I6.t4(src_ref, Bm, c, FT4); out = b0_all_tau(ls_build(f'T4_{c}', s, b, recomputed=True), J); ref, rn = Rref, 'Rref'
            k = f'{t}_{c}'; T[k] = dict(I6.compare(out, ref, c, TOL), against=rn, changed=info)
            T[k].pop('scales', None)
            rows.append(dict(test=t, cell=c, against=rn, rows=T[k]['rows'], max_rel_central=T[k]['max_rel_central'], max_rel_pred=T[k]['max_rel_pred'],
                             nan_mismatch=T[k]['nan_mismatch'], scales_equal=T[k]['scales_equal'], ok=T[k]['ok']))
            say(f"I1 {k:7s} vs {rn:4s}: rows {T[k]['rows']}  max rel central {T[k]['max_rel_central']:.2e}  pred {T[k]['max_rel_pred']:.2e}  "
                f"scales equal {T[k]['scales_equal']}  nan {T[k]['nan_mismatch']} -> {'pass' if T[k]['ok'] else 'FAIL'}")
    pd.DataFrame(rows).to_csv(OUT / 'invariance_tests.csv', index=False)
    res['I1_all_pass'] = bool(all(v['ok'] for v in T.values()))
    res['Rref_vs_R0_informative'] = {c: {k: v for k, v in I6.compare(Rref, R0, c, np.inf).items() if k != 'scales'} for c in CELLS}
    say('I1:', 'all pass' if res['I1_all_pass'] else 'FAILED', '| Rref vs R0 (informative):',
        {c: (f"{v['max_rel_central']:.1e}", f"{v['max_rel_pred']:.1e}") for c, v in res['Rref_vs_R0_informative'].items()})

    # ---------------- I2 / I3: AD causality and clamp-event-only
    rng = np.random.default_rng(SEED); viol = []; n = 0
    for c in CELLS:
        idx = ad_order(Jc, c)
        for k in range(len(idx)):
            J2 = Jc.copy(); J2.loc[idx[k:], 'truth'] = J2.loc[idx[k:], 'truth'].to_numpy() * rng.uniform(0.5, 2.0, len(idx) - k)
            r2, _, _ = A4.adapt(J2, s0, A4.GAMMA, A4.TAU, 'cell'); n += 1
            d = float(np.max(np.abs(r2[idx[:k + 1]] - req0[idx[:k + 1]])))
            if d != 0.0:
                viol.append(dict(cell=c, k=int(k), max_abs_diff=d))
    res['I2'] = dict(perturbations=n, violations=len(viol), ok=not viol); pd.DataFrame(viol, columns=['cell', 'k', 'max_abs_diff']).to_csv(OUT / 'ad_causality_violations.csv', index=False)
    J3 = Jc.copy(); u = rng.uniform(size=len(J3)); e = req0 / Jc.truth.to_numpy() > 1 + 1e-9
    J3['truth'] = Jc.truth.to_numpy() * np.where(e, 0.5 + 0.5 * u, 1.0 + u)
    r3, _, ev3 = A4.adapt(J3, s0, A4.GAMMA, A4.TAU, 'cell')
    res['I3'] = dict(events_equal=bool((ev3 == ev0).all()), max_abs_diff=float(np.max(np.abs(r3 - req0))))
    res['I3']['ok'] = bool(res['I3']['events_equal'] and res['I3']['max_abs_diff'] == 0.0)
    say('I2 AD causality:', res['I2'], '| I3 clamp event only:', res['I3'])

    # ---------------- post hoc on the tau-5 % requests
    req = {'AD': np.asarray(ad, float), 'B0': np.asarray(b0, float)}              # AMENDMENT_1: the gated recomputation (unrounded)
    cen = Jc0.central.to_numpy(); truth = J.truth.to_numpy()
    for pol, r in req.items():                                                      # D
        q = r / truth; it = q > 1 + 1e-9; dep = q[it] - 1
        res[f'D_{pol}'] = dict(intervened_rows=int(it.sum()), depth_q95=float(np.quantile(dep, .95)) if it.any() else None,
                               depth_max=float(dep.max()) if it.any() else None, request_zero_rows=int((r == 0).sum()))
    Jd = J[GK + ['row_id', 'truth']].assign(central=cen, pred=req['AD'])          # E
    d, Jdd, _ = V5.deliver(Jd)
    Jdd[['row_id'] + GK + ['truth', 'central', 'pred', 'intervened', 'del_over_req', 'del_over_truth', 't_clamp']].to_csv(OUT / 'delivery_rows_LsAD.csv', index=False)
    res['E_AD'] = d
    rows = []                                                                       # S01
    for pol, r in req.items():
        _, Gk = C1.closed_loop(Bm.assign(pred=r), r)
        for uu in (0.70, 0.75, 0.80, 0.85, 0.90):
            for ii in (0.05, 0.10, 0.15, 0.20):
                rows.append(dict(policy=pol, util_threshold=uu, int_threshold=ii, groups_pass=int(((Gk.util_q50 >= uu) & (Gk.int_rate <= ii)).sum())))
    pd.DataFrame(rows).to_csv(OUT / 'threshold_sensitivity.csv', index=False)
    res['S01'] = rows
    rows = []                                                                       # S02_AD
    for f in np.round(np.arange(0.96, 1.0401, 0.01), 2):
        rf, _, _ = A4.adapt(Jc, {c: float(f) * s0[c] for c in CELLS}, A4.GAMMA, A4.TAU, 'cell')
        g, _ = groups_pass(Bm, rf); rows.append(dict(factor=float(f), groups_pass=g))
    SF = pd.DataFrame(rows); SF.to_csv(OUT / 'start_scale_factor_AD.csv', index=False)
    res['S02_AD'] = dict(zip(SF.factor.astype(str), SF.groups_pass.astype(int)))
    RB = pd.read_csv(V2.ls.BENCH)                                                   # K6
    K = J[['row_id', 'direction', 'truth']].merge(RB[['row_id', 'context_pulses']], on='row_id', how='left')
    assert len(K) == 672 and K.context_pulses.notna().all()
    K['first_query'] = (K.context_pulses == 0).to_numpy(); rows = []
    for pol, r in req.items():
        K['pred'] = r
        for (fq, dr), g in list(K.groupby(['first_query', 'direction'])) + [((fq, 'all'), g) for fq, g in K.groupby('first_query')]:
            rr = (g.pred / g.truth).to_numpy(); it = rr > 1 + 1e-9
            rows.append(dict(policy=pol, first_query=bool(fq), direction=dr, rows=int(len(g)), interventions=int(it.sum()), int_rate=float(it.mean()),
                             util_q50=float(np.median(np.minimum(rr, 1))), ratio_q05=float(np.quantile(rr, .05)), ratio_q50=float(np.median(rr)),
                             ratio_q95=float(np.quantile(rr, .95)), ratio_max=float(rr.max())))
    pd.DataFrame(rows).to_csv(OUT / 'first_query_LsAD.csv', index=False)
    res['K6'] = rows
    ok = cen > 0; ratio = cen[ok] / truth[ok]; lr = np.log(ratio); w = np.flatnonzero(ok)[int(np.argmax(ratio))]   # O10
    res['O10_Ls'] = dict(rows_with_centre=int(ok.sum()), rows_without_centre=[str(x) for x in J.row_id[~ok]],
                         max=float(ratio.max()), where=f"{J.evaluation_group.iat[w]} {J.direction.iat[w]} {J.temp_C.iat[w]} C {J.horizon_s.iat[w]:g} s",
                         share_above_1_2=float((ratio > 1.2).mean()), rows_above_1_3=int((ratio > 1.3).sum()),
                         q50=float(np.exp(np.quantile(lr, .5))), q95=float(np.exp(np.quantile(lr, .95))), q99=float(np.exp(np.quantile(lr, .99))))
    for pol, r in req.items():                                                      # O2
        sm, Gs = SA.score(Jc, r)
        res[f'O2_{pol}'] = dict(original=int(sm['original']), exceed_rows=int(sm['exceed_rows']),
                                failed=[f"{x.evaluation_group} {x.direction} {x.temp_C} C {x.horizon_s:g} s ("
                                        + ', '.join(([f'exceedance {int(x.exceed)}'] if x.exceed > 0 else []) + ([f'median ratio {x.ratio_q50:.3f}'] if x.ratio_q50 < 0.80 else [])) + ')'
                                        for x in Gs[~Gs.pass_original].itertuples()])
        Gs.to_csv(OUT / f'groups_tau5_{pol}.csv', index=False)
    Fs, info = SA.spliced_relax(F0)                                                 # D08_Ls
    Mx = RB[['row_id', 'member', 'q_row', 'direction', 'truth_power_W']].merge(Fs[['row_id', 'prev_P']], on='row_id').sort_values(['member', 'q_row'])
    Mx['pt'] = Mx.groupby('member').truth_power_W.shift(1); Mx['pdir'] = Mx.groupby('member').direction.shift(1)
    pr = Mx[Mx.prev_P.notna() & Mx.pt.notna() & (Mx.pdir == Mx.direction)]
    res['D08_Ls'] = dict(pairs=int(len(pr)), max_abs_diff_W=float((pr.prev_P - pr.pt).abs().max()),
                         equal_3_decimals=int((np.round(pr.prev_P, 3) == np.round(pr.pt, 3)).sum()), rows_with_prev_features=int(Fs.prev_P.notna().sum()),
                         splice={k: {kk: vv for kk, vv in v.items() if kk != 'changed_rows'} for k, v in info.items()})
    tr_rows = []                                                                     # AD trajectory
    for c in CELLS:
        idx = ad_order(Jc, c)
        for k, i in enumerate(idx):
            tr_rows.append(dict(cell=c, order=k, row_id=Jc.row_id.iat[i], file=str(Jc.member.iat[i]).split('/')[-1], direction=Jc.direction.iat[i],
                                temp_C=Jc.temp_C.iat[i], horizon_s=Jc.horizon_s.iat[i], central=cen[i], truth=truth[i], scale=sc0[i], request=req0[i],
                                ratio=req0[i] / truth[i], intervened=bool(ev0[i])))
    pd.DataFrame(tr_rows).to_csv(OUT / 'ad_trajectory.csv', index=False, float_format='%.10g')
    trows = []                                                                       # T_Ls
    for k in range(3):
        c0, w0 = time.process_time(), time.perf_counter(); ls_build(f'Ls_rep{k}', src, Bm); c1, w1 = time.process_time(), time.perf_counter()
        trows.append(dict(rep=k, cpu_s=c1 - c0, wall_s=w1 - w0, rows=672, cpu_ms_per_row=1000 * (c1 - c0) / 672))
    Tt = pd.DataFrame(trows); Tt.to_csv(OUT / 'timing_Ls.csv', index=False)
    res['T_Ls'] = dict(cpu_ms_per_row=[round(x, 3) for x in Tt.cpu_ms_per_row], wall_s=[round(x, 3) for x in Tt.wall_s],
                       note='upper bound: fitting of all folds and single-cell models plus temp CSV I/O, divided by 672 rows; feature extraction not included')
    res['X01_Ls'] = dict(central_abs_rel_err_median_pct=float(100 * np.median(np.abs(ratio - 1))), status='internal only (open loop)')
    for k in ('D_AD', 'D_B0', 'E_AD', 'S02_AD', 'O10_Ls', 'O2_AD', 'O2_B0', 'D08_Ls', 'T_Ls', 'X01_Ls'):
        say(k, json.dumps(V2.clean(res[k]), ensure_ascii=False)[:700])
    for r in res['K6']:
        say('   K6', r)
    dump(OUT / 'summary.json', V2.clean(res)); (OUT / 'run.log').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
