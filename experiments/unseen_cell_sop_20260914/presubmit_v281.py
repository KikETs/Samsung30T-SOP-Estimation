"""V281 (PROTOCOL_PRESUBMIT_V281.md): pre-submission checks of the frozen main configuration P (V278 B0, vT1h i3n,
tau 5 %) and its post hoc analyses. No new model, rule or training: saved outputs and code only.
  K1  lineage audit of P: code identity and absolute paths (presubmit_v281/remote_hashes.tsv, a read-only remote
      listing), held-rule re-derivation, map rest counts, i3n event counts against V273
  K2  V276 T2 / T4 against R0 and R0RT (re-count of the saved remote outputs)
  K3  training cells of the V280 inner predictions
  K4  output paths of the three B 25 C 2 s rows under P2 (post hoc, descriptive)
  K5  the post hoc analyses of the V273 package recomputed for P and L (same definitions and grids)
  K6  first queries against the other rows (post hoc, descriptive)
usage: python presubmit_v281.py   (torch_env; once, after the registration presubmit_v281/v281_prereg.sha256)"""
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

import hashlib  # noqa: E402
import re  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
from run_envelope import HERE, dump  # noqa: E402
import closedloop_v251 as C1, closedloop_v255 as V5  # noqa: E402
import leakfree_v272 as V2  # noqa: E402
import strictfull_v273 as S  # noqa: E402
import invariance_v276 as I6  # noqa: E402
import compare_v278 as CV  # noqa: E402
import nestedtau_v280 as N8  # noqa: E402
sys.path.insert(0, str(HERE / 'paper_package_v273_20261008'))
import posthoc_criteria as PH  # noqa: E402  (V273 package post hoc definitions, used unchanged)

OUT = HERE / 'presubmit_v281'
CV.TMP = OUT / 'tmp'                      # set after importing nestedtau_v280, which points it to its own folder
CELLS = V5.CELLS; GK = C1.GK; TOL = 1e-12
RO = HERE / 'rebuild_v279/remote_out'
PHYS_P = HERE / 'compare_v278/phys_vT1h'
MIRROR_RESTS = HERE.parent / 'physrebuild_vT1h/mirror_vT1h/Samsung30T/aging_refit/out/v2/rests.csv'
ADAPTED = {'tools/stage2/fit_rpt_ocv.py': 'd9945ada1ca88c1b27a56d09d5083bee0262d894f5fe04ee60eb787ef8c597e8',
           'tools/v2/pulses_uy2.py': '9d45fd90c030ea513a1a998abd232fce2b2b9c8511530055730d63365f3fa7d6',
           'tools/v2/thermo.py': '06549680cb2a1ca420322ae80730f5b7a3ca581a0204355e56401a782f956ca3'}    # V279 stage-4 registration
MONTHS = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'March': 3, 'Apr': 4, 'April': 4, 'May': 5, 'Jun': 6, 'June': 6, 'Jul': 7,
          'July': 7, 'Aug': 8, 'Sep': 9, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
LOG = []


class Stop(Exception):
    pass


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def relerr(x, y):
    return abs(float(x) - float(y)) / max(abs(float(y)), 1e-300) if x != y else 0.0


# ---------------------------------------------------------------- K1 lineage audit
def k1a():
    T = pd.read_csv(OUT / 'remote_hashes.tsv', sep='\t', dtype=str, keep_default_na=False)
    H, A, D, M = (T[T.kind == k] for k in ('hash', 'abspath', 'data', 'missing'))
    diff = H[H.sha_rebuild != H.sha_orig]
    only_copy = H[H.sha_orig == '-']
    is_adapted = (diff.dir == 'aging_refit') & diff.relpath.isin(list(ADAPTED))
    ad = H[(H.dir == 'aging_refit') & H.relpath.isin(list(ADAPTED))]
    adapted_ok = bool(len(ad) == 3 and all(ADAPTED[p] == h for p, h in zip(ad.relpath, ad.sha_rebuild)))
    tl = {}
    for line in (HERE / 'invariance_v276/remote/logs/tools.sha256').read_text().splitlines():
        if line.strip():
            h, p = line.split(None, 1); tl[os.path.basename(p.strip())] = h
    st = H[(H.dir == 'strictfull_v273') & H.relpath.str.startswith('tools/')]
    stm = {os.path.basename(p): h for p, h in zip(st.relpath, st.sha_rebuild)}
    tools_bad = sorted(k for k in set(tl) | set(stm) if tl.get(k) != stm.get(k))
    rests_local = sha(MIRROR_RESTS)
    rr = D[D.relpath == 'out/v2/rests.csv']
    data_ok = bool(len(D) == 3 and (D.sha_rebuild == D.sha_orig).all() and (D.sha_rebuild != '-').all()
                   and len(rr) == 1 and rr.sha_rebuild.iloc[0] == rests_local)
    H.assign(equal=H.sha_rebuild == H.sha_orig).to_csv(OUT / 'k1a_code.csv', index=False)
    out = dict(files_compared=int(len(H)), per_dir={k: int(v) for k, v in H.groupby('dir').size().items()},
               differing=diff[['dir', 'relpath']].values.tolist(), only_in_copy=only_copy[['dir', 'relpath']].values.tolist(),
               unexpected_differences=diff[~is_adapted][['dir', 'relpath']].values.tolist(), adapted_hashes_match=adapted_ok,
               i3n_tools_vs_v276_list=dict(listed=len(tl), in_copy=len(stm), mismatched=tools_bad),
               data_tables=D[['relpath', 'sha_rebuild', 'sha_orig']].values.tolist(), rests_local_sha=rests_local, data_ok=data_ok,
               abspath_hits=A[['dir', 'relpath']].values.tolist(), missing_dirs=M.dir.tolist())
    out['ok'] = bool(not out['missing_dirs'] and not out['unexpected_differences'] and not out['only_in_copy'] and adapted_ok
                     and not tools_bad and data_ok and not out['abspath_hits'])
    out['abspath_import_check_needed'] = bool(out['abspath_hits'])
    return out


def order_of(member, cycle):
    """time order of a file: the cycle for A files, month * 100 + day of the file name for B files (premodes.py docstring)"""
    if np.isfinite(cycle):
        return float(cycle)
    m = re.search(r'_(Jan|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|Aug|Sept|Sep|Oct|Nov|Dec)(\d+)_', member)
    return float(MONTHS[m.group(1)] * 100 + int(m.group(2)))


def k1b():
    TS = pd.read_csv(RO / 'aging_refit/out/v2/thermo/states.csv')
    P = pd.read_csv(RO / 'sop_ident_premodes/out/premodes/causal/rows.csv')
    TS['ord'] = [order_of(m, c) for m, c in zip(TS.member, TS.cycle)]
    info = TS.set_index('member')
    recs = []
    for r in P.itertuples():
        me = info.loc[r.member]
        cand = TS[(TS.cell == me.cell) & (TS.temp_C == 25.0) & (TS.ord < me.ord) & (TS.n >= 3)]
        ties = int((cand.ord == cand.ord.max()).sum()) if len(cand) else 0
        src = cand.loc[cand.ord.idxmax()] if len(cand) else None
        kind = str(r.rule).split()[0]
        if kind == 'held':
            ok = bool(src is not None and ties == 1 and r.rule == f"held {src.member.split('/')[-1][:40]}" and src.ord < me.ord
                      and relerr(r.LLI, src.LLI) <= TOL and relerr(r.LAM_p, src.LAM_p) <= TOL and r.cell == me.cell)
        else:
            ok = bool(src is None)
        recs.append(dict(row_id=r.row_id, cell=r.cell, rule=kind, row_order=me.ord, source=None if src is None else src.member,
                         source_order=np.nan if src is None else src.ord, ties=ties, ok=ok))
    D = pd.DataFrame(recs); D.to_csv(OUT / 'k1b_held.csv', index=False)
    return dict(rows=int(len(D)), held=int((D.rule == 'held').sum()), map=int((D.rule == 'map').sum()),
                failed=D[~D.ok].row_id.tolist(), ok=bool(D.ok.all()))


def k1c():
    P = pd.read_csv(RO / 'sop_ident_premodes/out/premodes/causal/rows.csv'); P = P[P.rule.astype(str).str.startswith('map')]
    RS = pd.read_csv(MIRROR_RESTS)
    recs = []
    for r in P.itertuples():
        g = RS[RS.member == r.member]
        before = int((g.rest_row < r.q_row).sum())
        recs.append(dict(row_id=r.row_id, member=r.member, q_row=int(r.q_row), n_rests=int(r.n_rests), rests_in_file=int(len(g)),
                         rests_before_query=before, ok=bool(len(g) > 0 and r.n_rests <= before)))
    D = pd.DataFrame(recs); D.to_csv(OUT / 'k1c_map.csv', index=False)
    return dict(rows=int(len(D)), failed=D[~D.ok].row_id.tolist(), ok=bool(len(D) > 0 and D.ok.all()))


def k1d():
    out, bad_all = {}, []
    for t in ('n_all', 'n_bA17', 'n_bA1'):
        a = pd.read_csv(PHYS_P / t / 'pred.csv'); b = pd.read_csv(S.RS / t / 'pred.csv')
        M = a.merge(b, on='row_id', suffixes=('_v1h', '_v8s'), how='outer', indicator=True)
        both = (M._merge == 'both') & (M.status_v1h == 'ok') & (M.status_v8s == 'ok')
        same = np.ones(len(M), bool)
        for c in ('n_events', 'sweep_id', 'direction', 'temp_C', 'horizon_s'):
            same &= (M[f'{c}_v1h'] == M[f'{c}_v8s']).to_numpy()
        bad = M[both.to_numpy() & ~same]
        bad_all += [f'{t}:{r}' for r in bad.row_id]
        out[t] = dict(rows_vT1h=int(len(a)), rows_vT8s=int(len(b)), both_ok=int(both.sum()), mismatched=int(len(bad)),
                      n_events_vT1h={int(k): int(v) for k, v in a.n_events.value_counts().sort_index().items()})
    pd.DataFrame(dict(mismatch=bad_all)).to_csv(OUT / 'k1d_events.csv', index=False)
    out['ok'] = bool(not bad_all)
    return out


# ---------------------------------------------------------------- K2, K3, K4
def k2():
    rd = lambda t: pd.read_csv(HERE / f'invariance_v276/remote/out/{t}/pred.csv')
    cols = ['row_id', 'status'] + I6.RCOLS
    R0, RT = rd('R0'), rd('R0RT'); rows = []
    for t, A in (('T2', rd('T2')), ('T4', rd('T4')), ('R0RT', RT)):
        for nm, ref in (('R0', R0), ('R0RT', RT)):
            if t == 'R0RT' and nm == 'R0RT':
                continue
            c = I6.remote_compare(A[cols], ref[cols], ['row_id'])
            rows.append(dict(test=t, against=nm, rows=c['rows'], status_equal=c['status_equal'], within_registered_1e9=c['ok'],
                             **{f'max_rel_{k}': c[f'max_rel_{k}'] for k in I6.RCOLS}))
    D = pd.DataFrame(rows); D.to_csv(OUT / 'k2_t2t4.csv', index=False)
    g = lambda t, nm: D[(D.test == t) & (D.against == nm)].iloc[0]
    zero = all(g(t, 'R0RT')[[f'max_rel_{k}' for k in I6.RCOLS]].max() == 0 and g(t, 'R0RT').status_equal for t in ('T2', 'T4'))
    return dict(table=D.round(12).to_dict('records'), T2_T4_vs_R0RT_all_zero=bool(zero),
                T2_vs_R0_max_rel_a=float(g('T2', 'R0').max_rel_a), T2_vs_R0_max_rel_pred=float(g('T2', 'R0').max_rel_pred_power_W),
                ok=bool(zero and not g('T2', 'R0').within_registered_1e9 and not g('T4', 'R0').within_registered_1e9))


def k3(X):
    sm = X['sm'].to_dict(); fn = S.look(X['Bc']); bad = 0; rows = []
    for c in CELLS:
        R = N8.cal_rows(X, c)
        for d, r, k, p in zip(R.evaluation_group, R.row_id, R.direction, R.p):
            ref = float(np.exp(sm[(S.third(c, d), r)])) if k == 'discharge' else fn[(c, r)]
            bad += int(p != ref)
        others = [x for x in CELLS if x != c]
        for k_ in others:
            v = [x for x in others if x != k_][0]
            Kr, Vr = R[R.evaluation_group == k_], R[R.evaluation_group == v]
            nk, nv = (Kr.direction == 'charge').sum(), (Vr.direction == 'charge').sum()
            rows.append(dict(evaluated=c, calibration_cell=k_, validation_cell=v,
                             cal_discharge_rows=int((Kr.direction == 'discharge').sum()), cal_discharge_model=f'{S.third(c, k_)} alone',
                             cal_charge_rows=int(nk), cal_charge_model=f'fold {c} ({k_} + {v}), in sample' if nk else '-',
                             val_discharge_rows=int((Vr.direction == 'discharge').sum()), val_discharge_model=f'{S.third(c, v)} alone',
                             val_charge_rows=int(nv), val_charge_model=f'fold {c} ({k_} + {v}), in sample' if nv else '-',
                             validation_cell_in_calibration_models='yes' if (S.third(c, k_) == v or nk) else 'no'))
    D = pd.DataFrame(rows); D.to_csv(OUT / 'k3_v280_training_cells.csv', index=False)
    return dict(rows_checked=int(sum(len(N8.cal_rows(X, c)) for c in CELLS)), mismatched=int(bad),
                validation_cell_in_calibration=D.validation_cell_in_calibration_models.tolist(), ok=bool(bad == 0))


def k4(B0t, F):
    P2 = pd.read_csv(HERE / 'compare_v278/P/p2_rows.csv').set_index('row_id')
    rq = {k: pd.read_csv(HERE / f'compare_v278/P/requests_{k}.csv', float_precision='round_trip').set_index('row_id').pred for k in ('B0', 'P2')}
    T = B0t.set_index('row_id').truth; Fi = F.set_index('row_id'); rows = []
    for r in P2.index:
        f = Fi.loc[r]; b = f.prev_R2 * f.Roff2 / f.prev_Roff2
        path = 'prev' if np.isfinite(b) else ('off' if np.isfinite(f.Roff2) else 'none')
        iref = abs(f.prev_I_last) if path == 'prev' else (abs(f.I_end) if path == 'off' else np.nan)
        x = P2.loc[r]
        rows.append(dict(row_id=r, rest_s=f.rest_s, Roff2=f.Roff2, prev_R2=f.prev_R2, prev_Roff2=f.prev_Roff2, basis_path=path, I_ref_A=iref,
                         **{f'{n}_{a}': x[f'{n}_{a}'] for a in ('B0', 'P2') for n in ('prev', 'L4', 'M6', 'DN6', 'G1')},
                         n_predictors_in_median=int(x.n_pred), centre_B0=x.centre_B0, centre_P2=x.centre_P2,
                         request_B0=rq['B0'][r], request_P2=rq['P2'][r], truth=T[r],
                         utilisation_B0=min(rq['B0'][r] / T[r], 1.0), utilisation_P2=min(rq['P2'][r] / T[r], 1.0)))
    D = pd.DataFrame(rows); D.to_csv(OUT / 'k4_p2_paths.csv', index=False)
    return dict(paths=dict(zip(D.row_id, D.basis_path)), rest_s=dict(zip(D.row_id, D.rest_s.round(3))),
                n_predictors=dict(zip(D.row_id, D.n_predictors_in_median)))


# ---------------------------------------------------------------- K5 post hoc (V273 package definitions)
def tables(X):
    fold = {(f, r): float(np.exp(m)) for f, r, m in zip(X['C'].fold, X['C'].row_id, X['C'].m)}
    strict = {(e, r): float(np.exp(m)) for e, r, m in zip(X['Sc'].train_cell, X['Sc'].row_id, X['Sc'].m)}
    return fold, strict


def k5(B, groups_all, XS):
    T_ = {k: tables(X) for k, X in XS.items()}; G = {}
    for k in XS:
        tr = pd.read_csv(HERE / f'compare_v278/{k}/tau_range.csv'); tr = tr[tr.arm == 'B0']; bad = []
        for tau in PH.TAUS_REG:
            r = PH.calibrated(B, groups_all, *T_[k], float(tau)); ref = tr[np.isclose(tr.tau, tau)].iloc[0]
            if not (r['closed'] == int(ref.closed_loop) and abs(r['int_rate'] - ref.int_rate) <= TOL):
                bad.append(float(tau))
        r5 = PH.calibrated(B, groups_all, *T_[k], 0.05)
        ref = pd.read_csv(HERE / f'compare_v278/{k}/requests_B0.csv', float_precision='round_trip').set_index('row_id').pred
        rel = float(np.max(np.abs(r5['J'].set_index('row_id').pred.reindex(ref.index) / ref - 1)))
        G[k] = dict(tau_4_6_mismatch=bad, tau5_requests_max_rel=rel, ok=bool(not bad and rel <= TOL))
    if not all(g['ok'] for g in G.values()):
        raise Stop(f'K5 gate failed: {G}')
    say('K5 gates:', G)
    req = {k: pd.read_csv(HERE / f'compare_v278/{k}/requests_B0.csv', float_precision='round_trip').set_index('row_id').pred.reindex(B.row_id).to_numpy()
           for k in XS}
    summ = dict(gates=G); a_rows, g_rows, c_rows, o_rows = [], [], [], []
    for k in XS:                                                              # A1: registered tau 5 % requests
        s = PH.score(B, groups_all, req[k])
        summ[f'A1_{k}'] = dict(groups_original=s['orig'], exceed_rows=s['exceed'], groups_closed_loop=s['closed'],
                               failed_original=PH.failed(s['O'], 'orig_pass', 'orig_reason'))
        a_rows.append(dict(analysis='A1 registered tau 5 % requests', config=k, tau=0.05, exceed_rows=s['exceed'], int_rate=s['int_rate'],
                           util_q50=s['util_q50'], groups_original=s['orig'], groups_closed_loop=s['closed']))
    for k in XS:                                                              # A2: tau = 0
        r = PH.calibrated(B, groups_all, *T_[k], 0.0)
        summ[f'A2_{k}'] = dict(scales=r['scales'], exceed_rows=r['exceed'], int_rate=r['int_rate'], util_q50=r['util_q50'],
                               groups_original=r['orig'], groups_closed_loop=r['closed'], top_group_util=float(r['G'].util_q50.max()),
                               failed_closed_loop=PH.failed(r['G'], 'passed', 'reason'))
        a_rows.append(dict(analysis='A2 zero-exceedance calibration', config=k, tau=0.0, **{f'scale_{c}': r['scales'][c] for c in CELLS},
                           exceed_rows=r['exceed'], int_rate=r['int_rate'], util_q50=r['util_q50'], groups_original=r['orig'], groups_closed_loop=r['closed']))
        M = r['G'][GK + ['n', 'interventions', 'int_rate', 'util_q50', 'passed', 'reason']].merge(r['O'][GK + ['exceed', 'ratio_q50', 'orig_pass', 'orig_reason']], on=GK)
        g_rows.append(M.assign(config=k))
    for k in XS:                                                              # A3: tau 0 ... 20 %
        for tau in PH.TAUS_CURVE:
            r = PH.calibrated(B, groups_all, *T_[k], float(tau))
            c_rows.append(dict(config=k, tau=float(tau), **{f'scale_{c}': r['scales'][c] for c in CELLS}, exceed_rows=r['exceed'],
                               int_rate=r['int_rate'], util_q50=r['util_q50'], groups_closed_loop=r['closed'], groups_original=r['orig']))
    C3 = pd.DataFrame(c_rows)
    for k, g in C3.groupby('config'):
        tm = g[g.groups_closed_loop == g.groups_closed_loop.max()].tau
        summ[f'A3_{k}'] = dict(closed_range=[int(g.groups_closed_loop.min()), int(g.groups_closed_loop.max())],
                               original_range=[int(g.groups_original.min()), int(g.groups_original.max())],
                               closed_max_tau=[float(tm.min()), float(tm.max()), int(len(tm))],
                               at={f'{100 * t:.2f}': [int(g[np.isclose(g.tau, t)].groups_closed_loop.iloc[0]), int(g[np.isclose(g.tau, t)].groups_original.iloc[0])]
                                   for t in (0.0, 0.05, 0.10, 0.20)})
    chen = pd.read_csv(HERE / 'chen_v266/rows_tau5_M.csv').set_index('row_id').central.reindex(B.row_id).to_numpy(float)
    cen = {k: np.array([T_[k][0][(c, r)] for c, r in zip(B.evaluation_group, B.row_id)]) for k in XS}
    cen['prior_method_M'] = chen
    for name, x in cen.items():                                               # A4: oracle common scale (descriptive)
        rows = []
        for s_ in PH.S_GRID:
            r = PH.score(B, groups_all, float(s_) * x)
            rows.append(dict(predictor=name, s=float(s_), exceed_rows=r['exceed'], int_rate=r['int_rate'], util_q50=r['util_q50'],
                             groups_closed_loop=r['closed'], groups_original=r['orig']))
        D = pd.DataFrame(rows); o_rows.append(D)
        z, f5 = D[D.exceed_rows == 0], D[D.int_rate <= 0.05]
        pt = lambda d: dict(s=float(d.s.max()), util_q50=float(d.loc[d.s.idxmax(), 'util_q50']), closed=int(d.loc[d.s.idxmax(), 'groups_closed_loop']),
                            original=int(d.loc[d.s.idxmax(), 'groups_original'])) if len(d) else None
        summ[f'A4_{name}'] = dict(zero_exceedance=pt(z), int_rate_le_5pct=pt(f5), max_closed=int(D.groups_closed_loop.max()), max_original=int(D.groups_original.max()))
    ratio = cen['P'] / B.truth.to_numpy(); w = int(np.argmax(ratio))           # O10: over-prediction tail of the P centres
    summ['O10_P'] = dict(max=float(ratio.max()), where=f"{B.evaluation_group.iat[w]} {B.direction.iat[w]} {B.temp_C.iat[w]} C {B.horizon_s.iat[w]:g} s",
                         share_above_1_2=float((ratio > 1.2).mean()), rows_above_1_3=int((ratio > 1.3).sum()))
    J = B[GK + ['row_id', 'truth']].assign(central=cen['P'], pred=req['P'])  # E: delivery simulation of the P intervened rows
    d, Jd, _ = V5.deliver(J)
    Jd[['row_id'] + GK + ['truth', 'central', 'pred', 'intervened', 'del_over_req', 'del_over_truth', 't_clamp']].to_csv(OUT / 'k5_delivery_rows.csv', index=False)
    summ['E_P'] = d
    rows = []                                                                 # S01: threshold sensitivity
    for k in XS:
        _, Gk = C1.closed_loop(B.assign(pred=req[k]), req[k])
        for u in (0.70, 0.75, 0.80, 0.85, 0.90):
            for i in (0.05, 0.10, 0.15, 0.20):
                rows.append(dict(config=k, util_threshold=u, int_threshold=i, groups_pass=int(((Gk.util_q50 >= u) & (Gk.int_rate <= i)).sum())))
    pd.DataFrame(rows).to_csv(OUT / 'k5_threshold_sensitivity.csv', index=False)
    rows = []                                                                 # S02: scale-factor sensitivity of P
    for f in np.round(np.arange(0.96, 1.0401, 0.01), 2):
        _, Gf = C1.closed_loop(B.assign(pred=f * req['P']), f * req['P'])
        rows.append(dict(factor=float(f), groups_pass=int(((Gf.util_q50 >= 0.80) & (Gf.int_rate <= 0.10)).sum())))
    SF = pd.DataFrame(rows); SF.to_csv(OUT / 'k5_scale_factor_sensitivity.csv', index=False)
    summ['S02_P'] = dict(zip(SF.factor.astype(str), SF.groups_pass.astype(int)))
    pd.DataFrame(a_rows).to_csv(OUT / 'k5_openloop_criterion.csv', index=False)
    pd.concat(g_rows, ignore_index=True).to_csv(OUT / 'k5_groups_tau0.csv', index=False)
    C3.to_csv(OUT / 'k5_tradeoff_calibrated.csv', index=False)
    pd.concat(o_rows, ignore_index=True).to_csv(OUT / 'k5_tradeoff_oracle.csv', index=False)
    return summ


# ---------------------------------------------------------------- K6
def k6(B, F):
    R = pd.read_csv(V2.ls.BENCH)
    reqP = pd.read_csv(HERE / 'compare_v278/P/requests_B0.csv', float_precision='round_trip').set_index('row_id').pred
    J = B[['row_id', 'direction', 'truth']].merge(R[['row_id', 'context_pulses']], on='row_id', how='left')
    assert len(J) == 672 and J.context_pulses.notna().all()
    J['pred'] = reqP.reindex(J.row_id).to_numpy(); J['first_query'] = J.context_pulses == 0
    rows = []
    for (fq, d), g in list(J.groupby(['first_query', 'direction'])) + [((fq, 'all'), g) for fq, g in J.groupby('first_query')]:
        r = (g.pred / g.truth).to_numpy(); it = r > 1 + 1e-9
        rows.append(dict(first_query=bool(fq), direction=d, rows=int(len(g)), interventions=int(it.sum()), int_rate=float(it.mean()),
                         util_q50=float(np.median(np.minimum(r, 1))), ratio_q05=float(np.quantile(r, .05)), ratio_q50=float(np.median(r)),
                         ratio_q95=float(np.quantile(r, .95)), ratio_max=float(r.max())))
    D = pd.DataFrame(rows); D.to_csv(OUT / 'k6_first_query.csv', index=False)
    M = R[['row_id', 'member', 'q_row', 'direction', 'truth_power_W']].merge(F[['row_id', 'prev_P']], on='row_id', how='left').sort_values(['member', 'q_row'])
    M['prev_truth'] = M.groupby('member').truth_power_W.shift(1); M['prev_dir'] = M.groupby('member').direction.shift(1)
    pair = M.prev_P.notna() & M.prev_truth.notna() & (M.prev_dir == M.direction)
    dev = (M.prev_P[pair] / M.prev_truth[pair] - 1).abs()
    return dict(table=D.round(6).to_dict('records'), rows_with_prev_P=int(M.prev_P.notna().sum()), first_query_rows=int(J.first_query.sum()),
                prev_P_pairs_same_file_direction=int(pair.sum()), prev_P_equal_previous_truth=int((dev <= 1e-6).sum()), max_abs_dev=float(dev.max()))


# ---------------------------------------------------------------- main
def check_hashes(listfile, base):
    bad = []
    for line in listfile.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        h, p = line.split(None, 1); q = base / p.strip()
        if not q.is_file() or sha(q) != h:
            bad.append(p.strip())
    if bad:
        raise Stop(f'{listfile.name}: files changed or missing: {bad}')


def run(res):
    check_hashes(OUT / 'v281_prereg.sha256', HERE); res['registration'] = 'all registered files match'; say('registration: ok')
    B = V5.load(); groups_all = B[GK].drop_duplicates().sort_values(GK).reset_index(drop=True); assert len(groups_all) == 15
    src = I6.sources(); F = src[1]
    K = res['checks'] = {}
    for name, fn in (('K1a', k1a), ('K1b', k1b), ('K1c', k1c), ('K1d', k1d)):
        K[name] = fn(); say(name, 'ok' if K[name]['ok'] else 'FAILED', {k: v for k, v in K[name].items() if k not in ('ok',)} if name != 'K1a' else
                           {k: K[name][k] for k in ('files_compared', 'per_dir', 'differing', 'only_in_copy', 'unexpected_differences', 'adapted_hashes_match',
                                                    'i3n_tools_vs_v276_list', 'data_ok', 'abspath_hits', 'missing_dirs')})
    K['K1e'] = dict(cited=['V278 Q3 (baseline P): truth change -> request difference 0 (three cells x four arms)',
                           'V280 N3 (baseline P): truth change -> tau_c, s_c and requests difference 0'])
    res['K1_ok'] = bool(all(K[k]['ok'] for k in ('K1a', 'K1b', 'K1c', 'K1d'))); say('K1 (lineage of P):', 'passed' if res['K1_ok'] else 'NOT passed')
    K['K2'] = k2(); say('K2:', {k: v for k, v in K['K2'].items() if k != 'table'})
    XS = {'P': CV.build('P', 'P', src, B, PHYS_P), 'L': CV.build('L', 'L', src, B, None)}
    K['K3'] = k3(XS['P']); say('K3:', K['K3'])
    K['K4'] = k4(B, F); say('K4:', K['K4'])
    K['K5'] = k5(B, groups_all, XS)
    for k, v in K['K5'].items():
        say('K5', k, v)
    K['K6'] = k6(B, F); say('K6:', {k: v for k, v in K['K6'].items() if k != 'table'})
    for r in K['K6']['table']:
        say('   ', r)
    say('ci_note: 세 물리적 셀에서 구성된 조건별 기술통계이며, 15개 그룹을 독립 시행으로 간주한 일반화율 신뢰구간은 제시하지 않았다.')
    root = str(HERE.parents[1]) + os.sep
    data = sorted(p for p in READS if p.startswith(root) and os.path.isfile(p) and not p.endswith(('.py', '.pyc')) and '/presubmit_v281/tmp/' not in p)
    (OUT / 'inputs_read.tsv').write_text('path\tsha256\n' + ''.join(f'{os.path.relpath(p, HERE)}\t{sha(p)}\n' for p in data))
    res['inputs_read'] = len(data)


def main():
    OUT.mkdir(exist_ok=True); res = dict(protocol='PROTOCOL_PRESUBMIT_V281.md')
    try:
        run(res)
    except Stop as e:
        res['stopped'] = str(e); say('STOPPED:', e)
        raise SystemExit(1)
    finally:
        dump(OUT / 'summary.json', V2.clean(res))
        (OUT / 'run.log').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
