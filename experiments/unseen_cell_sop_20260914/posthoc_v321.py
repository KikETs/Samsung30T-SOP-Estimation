"""V321 (PROTOCOL_POSTHOC_V321.md): post hoc analyses of the manuscript configuration Ls + AD.
No new model or rule; only the adaptation signal, the calibration target and an update mask are varied.
  gates  G1 the V287 Ls build reproduces requests_Ls_{B0,AD}.csv (relative 1e-9: saved with 10 significant digits), the
            intervention flags and the group verdicts exactly (the V320 AMENDMENT_1 gate)
         G2 closedloop_v255.deliver on the recomputed AD requests equals V320 E_AD (28 intervened rows, 16 clamped), and
            the all-row simulation of this script equals deliver on the intervened rows
         G3 adapt_signal with the benchmark signal reproduces adaptive_v284.adapt (difference 0)
         G4 the tau = 5 % points of the C curves equal the gated fixed-scale and AD requests (difference 0)
  B  AD driven by the simulated clamp of the V253 response model (every row simulated at its request)
  C  fixed-scale and AD curves over the calibration target tau = 0-20 % (step 0.25 %p) and the iso-exceedance comparison
  D  AD replay in which the three B discharge 25 C 2 s rows do not update the B scale (diagnostic)
usage: OMP_NUM_THREADS=1 python posthoc_v321.py   (torch_env; once, after posthoc_v321/v321_prereg.sha256)"""
import hashlib
import json
import warnings

import numpy as np, pandas as pd
from run_envelope import HERE, dump
import closedloop_v251 as C1, closedloop_v255 as V5
from closedloop_v253 import simulate
import leakfree_v272 as V2
import invariance_v276 as I6
import compare_v278 as CV
import adaptive_v284 as A4
import specaudit_v287 as SA

warnings.simplefilter('ignore')
OUT = HERE / 'posthoc_v321'
CV.TMP = OUT / 'tmp'
GK = C1.GK; CELLS = V5.CELLS; TOL = 1e-12; TOL_SAVED = 1e-9; LOG = []
SAV = HERE / 'specaudit_v287'; P320 = HERE / 'posthoc_lsad_v320'
TAU_GRID = np.round(np.arange(0.0, 0.20001, 0.0025), 4)
B2S = ('test3:325', 'test3:330', 'test3:335')


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def check_hashes():
    bad = []
    for line in (OUT / 'v321_prereg.sha256').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            h, p = line.split(None, 1); q = HERE / p.strip()
            if not q.is_file() or hashlib.sha256(q.read_bytes()).hexdigest() != h:
                bad.append(p.strip())
    if bad:
        raise SystemExit(f'registration mismatch: {bad}')


def stop(res, why):
    dump(OUT / 'summary.json', V2.clean(res)); (OUT / 'run.log').write_text('\n'.join(LOG) + '\n')
    raise SystemExit(why)


def ad_order(J, c):
    m = (J.evaluation_group == c).to_numpy()
    return J.index[m][np.argsort(J.loc[m, 'tkey'].to_numpy(), kind='stable')]


def adapt_signal(J, s0, gamma, tau, signal, skip=()):
    """adaptive_v284.adapt (unit 'cell') with the event e = signal(i, request) and no scale update after the rows in skip"""
    req = np.full(len(J), np.nan); sc = np.full(len(J), np.nan); ev = np.zeros(len(J), bool)
    cen = J.central.to_numpy(); rid = J.row_id.to_numpy()
    for c in CELLS:
        ls = np.log(s0[c])
        for i in ad_order(J, c):
            s = float(np.exp(ls)); r = s * cen[i]; e = bool(signal(i, r))
            req[i], sc[i], ev[i] = r, s, e
            if rid[i] not in skip:
                ls = float(np.clip(ls + gamma * (tau - float(e)), A4.LO, A4.HI))
    return req, sc, ev


def ecm_arrays(J):
    """V253 response models aligned with J (read as closedloop_v255.deliver reads them)"""
    E = pd.read_csv(HERE / 'closedloop_v253/ecm_rows.csv').set_index('row_id').reindex(J.row_id)
    assert E.Ra.notna().all() and (E.direction.to_numpy() == J.direction.to_numpy()).all()
    assert np.array_equal(E.horizon_s.to_numpy(float), J.horizon_s.to_numpy(float))
    return dict(ecm=list(zip(E.tau, E.Ra, E.Rb, E.k)), V0=E.V0.to_numpy(float), d=J.direction.to_numpy(), h=J.horizon_s.to_numpy(float))


def sim_one(M, i, P):
    return simulate(M['ecm'][i], M['V0'][i], M['d'][i], P=float(P), horizon=M['h'][i])


def sim_all(J, req, M):
    """every row simulated at its request: delivered energy, end power, first clamp time, delivered utilisation"""
    out = np.array([sim_one(M, i, req[i]) for i in range(len(J))], float)
    S = pd.DataFrame({'E_del': out[:, 0], 'P_end': out[:, 1], 't_clamp': out[:, 2]})
    S['clamp'] = np.isfinite(S.t_clamp.to_numpy())
    S['del_util'] = np.minimum(S.E_del.to_numpy() / (J.truth.to_numpy(float) * M['h']), 1.0)
    return S


def sim_groups(J, S):
    G = J[GK].assign(clamp=S.clamp.to_numpy(), del_util=S.del_util.to_numpy()).groupby(GK).agg(
        n=('clamp', 'size'), clamps=('clamp', 'sum'), del_util_q50=('del_util', 'median')).reset_index()
    G['clamp_rate'] = G.clamps / G.n
    G['sim_pass'] = (G.del_util_q50 >= 0.80) & (G.clamp_rate <= 0.10)
    return G


def disagreement(J, req, S, label):
    r = req / J.truth.to_numpy(float); bench = r > 1 + 1e-9; cl = S.clamp.to_numpy(); rows = []
    for d in ('charge', 'discharge', 'all'):
        m = np.ones(len(J), bool) if d == 'all' else (J.direction == d).to_numpy()
        bo, so = bench & ~cl & m, ~bench & cl & m
        rows.append(dict(requests=label, direction=d, rows=int(m.sum()), bench_events=int((bench & m).sum()), sim_clamps=int((cl & m).sum()),
                         both=int((bench & cl & m).sum()), bench_only=int(bo.sum()), sim_only=int(so.sum()),
                         bench_only_ratio_min=float(r[bo].min()) if bo.any() else np.nan, bench_only_ratio_max=float(r[bo].max()) if bo.any() else np.nan,
                         sim_only_ratio_min=float(r[so].min()) if so.any() else np.nan, sim_only_ratio_max=float(r[so].max()) if so.any() else np.nan))
    return rows


def label(x):
    return f'{x.evaluation_group} {x.direction} {x.temp_C} C {x.horizon_s:g} s'


def bench_summary(sm):
    return {k: sm[k] for k in ('closed_loop', 'original', 'exceed_rows', 'int_rate', 'util_q50', 'depth_max', 'failed_groups')}


def sim_summary(J, S, Gs):
    return dict(clamp_rows=int(S.clamp.sum()), clamp_rate=float(S.clamp.mean()), del_util_q50=float(np.median(S.del_util)),
                clamp_rows_by_direction={d: int(S.clamp[(J.direction == d).to_numpy()].sum()) for d in ('charge', 'discharge')},
                groups_sim_pass=int(Gs.sim_pass.sum()), failed_sim=[f'{label(x)} (delivered u {x.del_util_q50:.3f}, clamp {x.clamp_rate:.3f})' for x in Gs[~Gs.sim_pass].itertuples()])


def trajectory(J, sc, ev):
    out = {}
    for c in CELLS:
        idx = ad_order(J, c)
        out[c] = dict(start=float(sc[idx[0]]), end=float(sc[idx[-1]]), min=float(np.min(sc[idx])), max=float(np.max(sc[idx])),
                      events=int(ev[idx].sum()), queries=int(len(idx)))
    return out


def plot(F, A, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for D, lab, mk in ((F, 'fixed scale (tau grid)', 'o'), (A, 'online adaptation (tau* = tau)', 's')):
        ax[0].plot(100 * D.int_rate, 100 * D.util_q50, mk + '-', ms=3, lw=0.8, label=lab)
        ax[1].plot(100 * D.int_rate, D.closed_loop, mk + '-', ms=3, lw=0.8, label=lab)
    for D, mk in ((F, 'o'), (A, 's')):
        p = D[np.isclose(D.tau, 0.05)].iloc[0]
        ax[0].plot(100 * p.int_rate, 100 * p.util_q50, mk, ms=9, mfc='none', mec='k')
        ax[1].plot(100 * p.int_rate, p.closed_loop, mk, ms=9, mfc='none', mec='k')
    ax[0].set_xlabel('realised intervention rate (%)'); ax[0].set_ylabel('median utilisation (%)')
    ax[1].set_xlabel('realised intervention rate (%)'); ax[1].set_ylabel('groups passing (closed-loop criterion)')
    ax[0].set_title('open markers: tau = 5 %'); ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True); (OUT / 'tmp').mkdir(exist_ok=True)
    check_hashes(); say('registration: ok')
    res = dict(protocol='PROTOCOL_POSTHOC_V321.md')
    Bm = V5.load(); src = I6.sources(); J = SA.bench_J(Bm)
    S287 = json.load(open(SAV / 'summary.json'))
    s0 = {c: float(v) for c, v in S287['Ls']['scales_tau5'].items()}
    sav_b0 = pd.read_csv(SAV / 'requests_Ls_B0.csv', float_precision='round_trip').set_index('row_id').reindex(J.row_id)
    sav_ad = pd.read_csv(SAV / 'requests_Ls_AD.csv', float_precision='round_trip').set_index('row_id').reindex(J.row_id)

    # ---------------- gates
    G = res['gates'] = {}
    X0 = SA.build('R0', src, Bm, R12=True, L1=True, S1=True, S2=True)
    b0, ad, sc, Jc0 = SA.requests_all(X0, J, 0.05)
    tt = J.truth.to_numpy(float); flag_mis, verdict_mis = {}, {}
    for pol, rec, sv in (('B0', b0, sav_b0.request.to_numpy()), ('AD', ad, sav_ad.request.to_numpy())):
        flag_mis[pol] = int(((rec / tt > 1 + 1e-9) != (sv / tt > 1 + 1e-9)).sum())
        _, Gr = SA.score(Jc0, rec); _, Gv = SA.score(Jc0, sv)
        verdict_mis[pol] = int(((Gr.passed.to_numpy() != Gv.passed.to_numpy()) | (Gr.pass_original.to_numpy() != Gv.pass_original.to_numpy())).sum())
    G['G1'] = dict(B0_max_rel=float(np.max(SA.rel(b0, sav_b0.request.to_numpy()))), AD_max_rel=float(np.max(SA.rel(ad, sav_ad.request.to_numpy()))),
                   centre_max_rel=float(np.max(SA.rel(Jc0.central.to_numpy(), sav_ad.central.to_numpy()))), tolerance_saved=TOL_SAVED,
                   intervention_flag_mismatch=flag_mis, group_verdict_mismatch=verdict_mis, scales={c: v['scale'] for c, v in sc.items()})
    G['G1']['ok'] = bool(G['G1']['B0_max_rel'] <= TOL_SAVED and G['G1']['AD_max_rel'] <= TOL_SAVED and G['G1']['centre_max_rel'] <= TOL_SAVED
                         and not any(flag_mis.values()) and not any(verdict_mis.values()) and all(abs(G['G1']['scales'][c] - s0[c]) <= TOL for c in CELLS))
    Jc = J.copy(); Jc['central'] = Jc0.central.to_numpy(); tru = Jc.truth.to_numpy(float)
    M = ecm_arrays(Jc)
    Jd = J[GK + ['row_id', 'truth']].assign(central=Jc.central.to_numpy(), pred=ad)
    d, Jdd, _ = V5.deliver(Jd)
    E320 = json.load(open(P320 / 'summary.json'))['E_AD']
    S_ad = sim_all(Jc, ad, M); it = Jdd.intervened.to_numpy(bool)
    same = bool(np.array_equal(S_ad.E_del.to_numpy()[it], Jdd.E_del.to_numpy(float)[it])
                and np.array_equal(np.isfinite(Jdd.t_clamp.to_numpy(float)[it]), S_ad.clamp.to_numpy()[it]))
    G['G2'] = dict(E_AD_equal=V2.clean(d) == E320, intervened=int(it.sum()), clamped=int(np.isfinite(Jdd.t_clamp.to_numpy(float)[it]).sum()),
                   all_row_simulation_equals_deliver_on_intervened=same)
    G['G2']['ok'] = bool(G['G2']['E_AD_equal'] and G['G2']['intervened'] == 28 and G['G2']['clamped'] == 16 and same)
    req0, sc0, ev0 = A4.adapt(Jc, s0, A4.GAMMA, A4.TAU, 'cell')
    bench_sig = lambda i, r: r / tru[i] > 1 + 1e-9
    reqG, scG, evG = adapt_signal(Jc, s0, A4.GAMMA, A4.TAU, bench_sig)
    G['G3'] = dict(max_abs_vs_adapt=float(np.max(np.abs(reqG - req0))), scales_equal=bool(np.array_equal(scG, sc0)),
                   events_equal=bool(np.array_equal(evG, ev0)), adapt_vs_gated_AD=float(np.max(np.abs(req0 - ad))))
    G['G3']['ok'] = bool(G['G3']['max_abs_vs_adapt'] == 0.0 and G['G3']['scales_equal'] and G['G3']['events_equal'] and G['G3']['adapt_vs_gated_AD'] == 0.0)
    for k in ('G1', 'G2', 'G3'):
        say(k, json.dumps(V2.clean(G[k]), ensure_ascii=False)[:600])
    if not all(G[k]['ok'] for k in ('G1', 'G2', 'G3')):
        stop(res, 'a gate failed (protocol: stop, nothing reported)')

    # ---------------- C first (its gate G4 belongs to the gates)
    rows_f, rows_a, g4 = [], [], []
    for tau in TAU_GRID:
        tau = float(tau)
        bt, at, sct, _ = SA.requests_all(X0, J, tau, A4.GAMMA, tau)
        for pol, rq, rows in (('fixed', bt, rows_f), ('AD', at, rows_a)):
            sm, _ = SA.score(Jc, rq)
            rows.append(dict(tau=tau, **{f'scale_{c}': sct[c]['scale'] for c in CELLS}, closed_loop=int(sm['closed_loop']), original=int(sm['original']),
                             exceed_rows=int(sm['exceed_rows']), int_rate=float(sm['int_rate']), util_q50=float(sm['util_q50']), depth_max=float(sm['depth_max'])))
        if abs(tau - 0.05) < 1e-12:
            g4.append(dict(fixed_max_abs=float(np.max(np.abs(bt - b0))), AD_max_abs=float(np.max(np.abs(at - ad)))))
    G['G4'] = dict(points=len(g4), **(g4[0] if g4 else {}))
    G['G4']['ok'] = bool(len(g4) == 1 and g4[0]['fixed_max_abs'] == 0.0 and g4[0]['AD_max_abs'] == 0.0)
    say('G4', json.dumps(V2.clean(G['G4'])))
    if not G['G4']['ok']:
        stop(res, 'gate G4 failed (protocol: stop, nothing reported)')

    # ---------------- B: simulated-clamp signal
    reqB, scB, evB = adapt_signal(Jc, s0, A4.GAMMA, A4.TAU, lambda i, r: np.isfinite(sim_one(M, i, r)[2]))
    S_B = sim_all(Jc, reqB, M)
    assert np.array_equal(S_B.clamp.to_numpy(), evB)
    smA, GA = SA.score(Jc, ad); smB, GB = SA.score(Jc, reqB)
    GsA, GsB = sim_groups(Jc, S_ad), sim_groups(Jc, S_B)
    res['B'] = dict(
        AD=dict(benchmark=bench_summary(smA), simulated=sim_summary(Jc, S_ad, GsA), trajectory=trajectory(Jc, sc0, ev0)),
        SIG=dict(benchmark=bench_summary(smB), simulated=sim_summary(Jc, S_B, GsB), trajectory=trajectory(Jc, scB, evB)))
    keep = GK + ['n', 'interventions', 'int_rate', 'util_q50', 'depth_max', 'passed', 'exceed', 'ratio_q50', 'pass_original']
    gs = []
    for pol, Gb, Gs in (('AD', GA, GsA), ('SIG', GB, GsB)):
        gs.append(Gb[keep].merge(Gs[GK + ['clamps', 'clamp_rate', 'del_util_q50', 'sim_pass']], on=GK).assign(policy=pol))
    GG = pd.concat(gs, ignore_index=True); GG.to_csv(OUT / 'sigclamp_groups.csv', index=False)
    res['B']['group_changes'] = [f"{x.evaluation_group} {x.direction} {x.temp_C} C {x.horizon_s:g} s: benchmark {'pass' if a.passed else 'fail'} -> {'pass' if b.passed else 'fail'}"
                                 f" (u {a.util_q50:.3f} -> {b.util_q50:.3f}, it {int(a.interventions)} -> {int(b.interventions)})"
                                 for (_, a), (_, b), x in zip(GA.iterrows(), GB.iterrows(), GA.itertuples()) if bool(a.passed) != bool(b.passed)]
    R = Jc[['row_id'] + GK + ['truth', 'central']].assign(
        AD_scale=sc0, AD_request=ad, AD_bench_event=ad / tru > 1 + 1e-9, AD_sim_clamp=S_ad.clamp.to_numpy(), AD_del_util=S_ad.del_util.to_numpy(),
        SIG_scale=scB, SIG_request=reqB, SIG_bench_event=reqB / tru > 1 + 1e-9, SIG_sim_clamp=evB, SIG_del_util=S_B.del_util.to_numpy())
    R.to_csv(OUT / 'sigclamp_rows.csv', index=False, float_format='%.10g')
    dis = disagreement(Jc, ad, S_ad, 'AD') + disagreement(Jc, reqB, S_B, 'SIG')
    pd.DataFrame(dis).to_csv(OUT / 'signal_disagreement.csv', index=False); res['B']['disagreement'] = dis
    nb = smB['closed_loop']
    res['B']['reading'] = ('주 결과는 적응 신호의 정의에 민감하다' if nb != 11 else '이 응답 모델 안에서는 적응 신호의 정의에 민감하지 않다')
    say('B AD :', json.dumps(V2.clean(res['B']['AD']), ensure_ascii=False)[:900])
    say('B SIG:', json.dumps(V2.clean(res['B']['SIG']), ensure_ascii=False)[:900])
    say('B group changes:', res['B']['group_changes']); say('B reading:', res['B']['reading'])
    for r in dis:
        say('   disagreement', r)

    # ---------------- C: curves and iso-exceedance comparison
    F = pd.DataFrame(rows_f); A = pd.DataFrame(rows_a)
    F.to_csv(OUT / 'tradeoff_fixed.csv', index=False); A.to_csv(OUT / 'tradeoff_ad.csv', index=False)
    iso = []
    for a in A.itertuples():
        m = (F.int_rate <= a.int_rate + 1e-12).to_numpy(); u = (F.util_q50 >= a.util_q50 - 1e-12).to_numpy()
        row = dict(tau=a.tau, AD_int_rate=a.int_rate, AD_util_q50=a.util_q50, AD_closed_loop=a.closed_loop, AD_original=a.original, fixed_points_at_or_below=int(m.sum()))
        if m.any():
            Fm = F[m]; k = Fm.util_q50.to_numpy().argmax()
            row.update(fixed_best_util_q50=float(Fm.util_q50.max()), fixed_best_util_tau=float(Fm.tau.iloc[k]), fixed_best_util_int_rate=float(Fm.int_rate.iloc[k]),
                       fixed_best_closed_loop=int(Fm.closed_loop.max()), fixed_best_original=int(Fm.original.max()))
        if u.any():
            row.update(fixed_min_int_rate_for_util=float(F.int_rate[u].min()))
        iso.append(row)
    I = pd.DataFrame(iso)
    for col in ('fixed_best_util_q50', 'fixed_best_util_tau', 'fixed_best_util_int_rate', 'fixed_best_closed_loop', 'fixed_best_original', 'fixed_min_int_rate_for_util'):
        if col not in I.columns:
            I[col] = np.nan
    I.to_csv(OUT / 'iso_comparison.csv', index=False)
    main_pt = I[np.isclose(I.tau, 0.05)].iloc[0]
    have = I.fixed_points_at_or_below > 0
    res['C'] = dict(
        fixed_int_rate_range=[float(F.int_rate.min()), float(F.int_rate.max())], AD_int_rate_range=[float(A.int_rate.min()), float(A.int_rate.max())],
        fixed_closed_loop_max=int(F.closed_loop.max()), AD_closed_loop_max=int(A.closed_loop.max()),
        main=V2.clean(main_pt.to_dict()),
        AD_points=int(len(I)), AD_points_with_fixed_at_or_below=int(have.sum()),
        AD_util_above_fixed_frontier=int((I.AD_util_q50[have] > I.fixed_best_util_q50[have] + 1e-12).sum()),
        AD_groups_above_fixed_frontier=int((I.AD_closed_loop[have] > I.fixed_best_closed_loop[have]).sum()),
        AD_groups_equal_fixed_frontier=int((I.AD_closed_loop[have] == I.fixed_best_closed_loop[have]).sum()),
        AD_groups_below_fixed_frontier=int((I.AD_closed_loop[have] < I.fixed_best_closed_loop[have]).sum()),
        util_gap_pp_range=[float(100 * (I.AD_util_q50[have] - I.fixed_best_util_q50[have]).min()), float(100 * (I.AD_util_q50[have] - I.fixed_best_util_q50[have]).max())])
    mp = res['C']['main']
    res['C']['reading'] = ('적응의 이득이 초과를 더 허용한 것만으로 설명되지 않는다' if mp.get('fixed_best_util_q50') is not None and mp['AD_util_q50'] > mp['fixed_best_util_q50'] + 1e-12
                           else '이득은 초과와 사용률의 교환으로 설명된다')
    plot(F, A, OUT / 'iso_tradeoff.png')
    say('C', json.dumps(V2.clean({k: v for k, v in res['C'].items()}), ensure_ascii=False)[:1500])

    # ---------------- D: B 2 s rows do not update the B scale
    mB = ((Jc.evaluation_group == 'B') & (Jc.direction == 'discharge') & (Jc.temp_C == 25) & (Jc.horizon_s == 2)).to_numpy()
    idxB = ad_order(Jc, 'B')
    ok = sorted(Jc.row_id[mB]) == sorted(B2S) and tuple(Jc.row_id.to_numpy()[idxB[:3]]) == B2S
    res['D_rows_check'] = dict(group_rows=sorted(Jc.row_id[mB]), first_three_in_B_order=list(Jc.row_id.to_numpy()[idxB[:3]]), ok=bool(ok))
    if not ok:
        stop(res, 'D row check failed (protocol: stop)')
    reqD, scD, evD = adapt_signal(Jc, s0, A4.GAMMA, A4.TAU, bench_sig, skip=set(B2S))
    smD, GD = SA.score(Jc, reqD)
    cols = GK + ['n', 'interventions', 'int_rate', 'util_q50', 'passed', 'exceed', 'ratio_q50', 'pass_original']
    CB = pd.concat([GA[GA.evaluation_group == 'B'][cols].assign(policy='AD'), GD[GD.evaluation_group == 'B'][cols].assign(policy='AD_no_update_2s')], ignore_index=True)
    CB.to_csv(OUT / 'counterfactual_b2s_groups.csv', index=False)
    res['D'] = dict(AD=bench_summary(smA), no_update_2s=bench_summary(smD),
                    B_scale_AD=trajectory(Jc, sc0, ev0)['B'], B_scale_no_update=trajectory(Jc, scD, evD)['B'],
                    B_scale_at_first_10s_row=dict(AD=float(sc0[idxB[3]]), no_update=float(scD[idxB[3]])),
                    B_groups=V2.clean(CB.to_dict(orient='records')),
                    group_changes=[f"{label(x)}: {'pass' if a.passed else 'fail'} -> {'pass' if b.passed else 'fail'} (u {a.util_q50:.3f} -> {b.util_q50:.3f}, it {int(a.interventions)} -> {int(b.interventions)})"
                                   for (_, a), (_, b), x in zip(GA.iterrows(), GD.iterrows(), GA.itertuples()) if bool(a.passed) != bool(b.passed)])
    say('D', json.dumps(V2.clean({k: v for k, v in res['D'].items() if k != 'B_groups'}), ensure_ascii=False)[:1500])
    say(CB.to_string(index=False))
    dump(OUT / 'summary.json', V2.clean(res)); (OUT / 'run.log').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
