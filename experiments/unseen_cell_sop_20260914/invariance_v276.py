"""V276 (PROTOCOL_INVARIANCE_V276.md): information-flow invariance tests of the frozen V273 N requests.
Local part (all 672 rows, every evaluated cell c): the V273 N computation (strictfull_v273 / leakfree_v272 functions,
i3n held at the saved V273 values) on model-input tables changed for c's rows only; the evaluator's query identities and
truths stay fixed. T1 truth x U(0.5, 2) (rows.csv truth, relax R*_q, the query events' I_last / V_min / V_max, the
calibration copy of the truth table); T2 nothing at or after the query (per-row features from raw with the query sample
removed, the query events dropped, every non-identity rows.csv column, cv.csv and the calibration truth blanked); T3
soc_point and ds_query permuted; T4 the earlier iterations of each row's own search removed from the raw. Compared:
c's centres and requests at the 9 tau values of V256 and c's scales; T1 / T3 against R0 (saved inputs), T2 / T4 against
R_ref (locally recomputed features). Remote part (i3n sample re-run, invariance_v276/remote): compared with --remote.
usage: python invariance_v276.py --gates-only   (G1 reproduction of V273 N, G2 recomputed features; no test computed)
       python invariance_v276.py                (main local run, once; stage-1 and stage-2 registrations checked)
       python invariance_v276.py --remote-gate  (G3: remote R0 = V273 n_all on the sample sweeps)
       python invariance_v276.py --remote       (remote T1-T4 against remote R0)"""
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
import json  # noqa: E402
from multiprocessing import get_context  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
from run_envelope import HERE, dump  # noqa: E402
import closedloop_v255 as V5, closedloop_v256 as V6  # noqa: E402
import leakfree_v272 as V2  # noqa: E402
import strictfull_v273 as S  # noqa: E402

OUT = HERE / 'invariance_v276'; TMP = OUT / 'tmp'; REM = OUT / 'remote'
CELLS = V5.CELLS; TAUS = [float(t) for t in V6.TAUS]; NAMES = V2.NAMES; SEED = 276
SRC_BENCH, SRC_EV, SRC_ROOT = V2.ls.BENCH, V2.ls.EV, V2.ls.ROOT          # the original model-input files of V273
IDENT = ['k', 'row_id', 'cell', 'member', 'direction', 'h0_row', 'temp_col', 'sheet', 'evaluation_group', 'temp_C',
         'horizon_s', 'v178_row', 'q_row']
NONIDENT = ['soc_point', 'offset', 'n', 'query_index', 'last_row', 'T_median', 'window_h', 'hidden_gaps_in_history',
            'segments_in_history', 'pulse_index', 'h0_index', 'blackout_index', 'truth_power_W', 'V0', 'I0',
            'context_pulses', 'ds_query']
RQ = ['R1_q', 'R2_q', 'R10_q']; QEV = ['I_last', 'V_min', 'V_max']
TOL_LOCAL, TOL_REMOTE = 1e-12, 1e-9
SAMPLE = {5: 'A1 charge', 40: 'A1 discharge', 73: 'A17 charge', 92: 'A17 discharge', 127: 'B discharge'}
LOG = []


class Stop(Exception):
    pass


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


# ---------------------------------------------------------------- model inputs
def sources():
    R = pd.read_csv(SRC_BENCH); F = pd.read_csv(SRC_ROOT / 'out/feats/relax.csv'); E = pd.read_csv(SRC_EV)
    cv = SRC_ROOT / 'out/feats/cv.csv'
    CV = pd.read_csv(cv) if cv.exists() else None
    assert sorted(R.columns) == sorted(IDENT + NONIDENT), 'rows.csv columns differ from the protocol lists'
    return R, F, E, CV


def table_from(name, R, F, E, CV):
    """local_sop.table() (unchanged code) on the given frames, written to TMP/name and read back"""
    d = TMP / name; (d / 'out/feats').mkdir(parents=True, exist_ok=True)
    R.to_csv(d / 'rows.csv', index=False, float_format='%.17g')
    F.to_csv(d / 'out/feats/relax.csv', index=False, float_format='%.17g')
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


def requests(args):
    """V273 N (strictfull_v273.run's main computation): centres and requests of every row and the scale of every
    evaluated cell at every tau, from the model-input frames and the calibration truth table Bm"""
    name, (R, F, E, CV), Bm = args
    D = table_from(name, R, F, E, CV)
    cell_of = D.set_index('row_id').evaluation_group.to_dict()
    TH = V2.boundaries(D, 'min_error')
    Fv = {n: V2.loco(D, n, TH) for n in NAMES}; Cv = V2.centres(Fv)
    Sv = pd.concat([V2.strict(D, n, TH) for n in NAMES], ignore_index=True)
    phys, _ = S.phys_tables()
    Cf = S.full_fold_centres(Cv, cell_of, phys); Sf = S.full_strict_centres(Sv, cell_of, phys)
    Bn = S.set_centres(Bm, Cf, name); sf = Sf.set_index(['train_cell', 'row_id']).m; fn = S.look(Bn)
    chg = lambda c, d, r: fn[(c, r)]
    groups_all = Bm[V5.C1.GK].drop_duplicates().sort_values(V5.C1.GK).reset_index(drop=True)
    rows, sc = [], []
    for tau in TAUS:
        r, _, J = S.score_single(Bn, groups_all, tau, S.cal_free_single(S.dis_from(sf), chg))
        sc += [dict(tau=tau, cell=c, scale=v['scale'], calibration_rows=v['calibration_rows']) for c, v in r['scales'].items()]
        rows.append(J[['row_id', 'evaluation_group', 'central', 'pred']].assign(tau=tau))
    return name, pd.concat(rows, ignore_index=True), pd.DataFrame(sc)


# ---------------------------------------------------------------- comparisons
def rel(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    with np.errstate(all='ignore'):
        r = np.abs(x - y) / np.maximum(np.abs(y), 1e-300)
    r[(x == y)] = 0.0
    r[np.isnan(x) & np.isnan(y)] = 0.0
    return r


def compare(test, ref, c, tol):
    T, Rf = test[0], ref[0]
    a = T[T.evaluation_group == c].merge(Rf[Rf.evaluation_group == c], on=['row_id', 'tau'], suffixes=('', '_ref'), how='outer', indicator=True)
    rc, rp = rel(a.central, a.central_ref), rel(a.pred, a.pred_ref)
    nan_mis = int((a.central.isna() != a.central_ref.isna()).sum() + (a.pred.isna() != a.pred_ref.isna()).sum())
    s = test[1][test[1].cell == c].merge(ref[1][ref[1].cell == c], on=['tau', 'cell'], suffixes=('', '_ref'))
    sc_eq = bool(len(s) == len(TAUS) and (s.scale == s.scale_ref).all())
    out = dict(rows=int((a._merge == 'both').sum()), unmatched=int((a._merge != 'both').sum()),
               max_rel_central=float(np.nanmax(rc)) if len(rc) else 0.0, max_rel_pred=float(np.nanmax(rp)) if len(rp) else 0.0,
               nan_mismatch=nan_mis, scales_equal=sc_eq, scales=s[['tau', 'scale', 'scale_ref']].to_dict('records'))
    out['ok'] = bool(out['unmatched'] == 0 and out['rows'] == len(TAUS) * int((test[0].evaluation_group == c).sum() / len(TAUS))
                     and out['max_rel_central'] <= tol and out['max_rel_pred'] <= tol and nan_mis == 0 and sc_eq)
    return out


def feature_gate(Fref, F0):
    """G2: recomputed relax features = relax.csv (numeric columns relative 1e-9, NaN mismatch 0, last_sign equal)"""
    cols = [c for c in F0.columns if c not in ('row_id', 'sweep_id', 'note', 'last_sign') + tuple(RQ) and pd.api.types.is_numeric_dtype(F0[c])]
    M = Fref.merge(F0, on='row_id', how='outer', suffixes=('', '_ref'), indicator=True)
    out = {}
    for c in cols:
        r = rel(M[c], M[c + '_ref']); out[c] = dict(max_rel=float(np.nanmax(r)), nan_mismatch=int((M[c].isna() != M[c + '_ref'].isna()).sum()))
    sign = bool((M.last_sign.fillna('') == M.last_sign_ref.fillna('')).all())
    missing = sorted(set(cols) - set(Fref.columns))
    ok = bool((M._merge == 'both').all() and len(M) == 672 and not missing and sign
              and all(v['max_rel'] <= 1e-9 and v['nan_mismatch'] == 0 for v in out.values()))
    return dict(rows=int(len(M)), columns=len(cols), max_rel=max(v['max_rel'] for v in out.values()),
                nan_mismatch=sum(v['nan_mismatch'] for v in out.values()), last_sign_equal=sign, missing_columns=missing, ok=ok)


def with_feats(F0, Fnew, rows):
    """relax table with the rows `rows` taken from Fnew (feature columns) and R*_q from F0"""
    F = F0.copy().set_index('row_id'); N = Fnew.set_index('row_id')
    cols = [c for c in F.columns if c not in RQ]
    for c in cols:                                   # column by column, so that numeric columns stay numeric
        v = N.loc[rows, c].to_numpy() if c in N.columns else np.full(len(rows), np.nan)
        if F[c].dtype != object and pd.Series(v).dtype == object:
            v = pd.to_numeric(pd.Series(v), errors='coerce').to_numpy()
        F.loc[rows, c] = v
    return F.reset_index()[F0.columns]


# ---------------------------------------------------------------- tests
def factors(R, rng):
    return pd.Series(rng.uniform(0.5, 2.0, len(R)), index=R.sort_values('row_id').row_id.to_numpy())


def t1(src, B0, c, fac):
    R, F, E, CV = (x.copy() if x is not None else None for x in src)
    m = R.evaluation_group == c; f = R.row_id.map(fac)
    R.loc[m, 'truth_power_W'] = R.loc[m, 'truth_power_W'] * f[m]
    fm = F.row_id.isin(R.row_id[m]); ff = F.row_id.map(fac)
    for col in RQ:
        F.loc[fm, col] = F.loc[fm, col] * ff[fm]
    qf = {(a, b): fac[r] for a, b, r in zip(R.member[m], R.q_row[m], R.row_id[m])}
    ek = [qf.get((a, b), np.nan) for a, b in zip(E.member, E.event_id)]; ek = pd.Series(ek, index=E.index); em = ek.notna()
    for col in QEV:
        E.loc[em, col] = E.loc[em, col] * ek[em]
    B = B0.copy(); bm = B.evaluation_group == c
    B.loc[bm, 'truth'] = B.loc[bm, 'truth'] * B.row_id[bm].map(fac)
    return (R, F, E, CV), B, dict(rows=int(m.sum()), relax_rows=int(fm.sum()), query_events=int(em.sum()), calib_rows=int(bm.sum()))


def t3(src, B0, c, perm_soc, perm_ds):
    R, F, E, CV = (x.copy() if x is not None else None for x in src)
    m = R.evaluation_group == c
    R.loc[m, 'soc_point'] = R.loc[m, 'row_id'].map(perm_soc).to_numpy()
    R.loc[m, 'ds_query'] = R.loc[m, 'row_id'].map(perm_ds).to_numpy()
    return (R, F, E, CV), B0.copy(), dict(rows=int(m.sum()), soc_changed=int((R.loc[m, 'soc_point'] != src[0].loc[m, 'soc_point']).sum()))


def t2(src, B0, c, FT2):
    R, F, E, CV = (x.copy() if x is not None else None for x in src)
    m = R.evaluation_group == c; ids = R.row_id[m].tolist()
    q = set(zip(R.member[m], R.q_row[m]))
    keep = [(a, b) not in q for a, b in zip(E.member, E.event_id)]
    n_ev = int(len(E) - sum(keep)); E = E[keep].copy()
    R[NONIDENT] = R[NONIDENT].astype(float) if R[NONIDENT].dtypes.ne(object).all() else R[NONIDENT]
    R.loc[m, NONIDENT] = np.nan
    F = with_feats(F, FT2, ids); F.loc[F.row_id.isin(ids), RQ] = np.nan
    sw = set(F.loc[F.row_id.isin(ids), 'sweep_id'].dropna().astype(int))
    if CV is not None:
        cm = CV.sweep_id.isin(sw); CV.loc[cm, [x for x in CV.columns if x != 'sweep_id']] = np.nan
    B = B0.copy(); bm = B.evaluation_group == c; B.loc[bm, 'truth'] = np.nan
    return (R, F, E, CV), B, dict(rows=len(ids), query_events_dropped=n_ev, cv_sweeps_blanked=len(sw), calib_rows=int(bm.sum()))


def t4(src, B0, c, FT4):
    R, F, E, CV = (x.copy() if x is not None else None for x in src)
    ids = R.row_id[R.evaluation_group == c].tolist()
    F = with_feats(F, FT4, ids)
    return (R, F, E, CV), B0.copy(), dict(rows=len(ids))


# ---------------------------------------------------------------- registration
def check_hashes(listfile, base):
    bad, seen = [], set()
    for line in listfile.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        h, p = line.split(None, 1); p = p.strip(); seen.add(p); qp = base / p
        if not qp.is_file() or hashlib.sha256(qp.read_bytes()).hexdigest() != h:
            bad.append(p)
    if bad:
        raise Stop(f'{listfile.name}: files changed or missing: {bad}')
    return seen


def project_inputs():
    root = str(HERE.parents[1]) + os.sep
    data = sorted(p for p in READS if p.startswith(root) and os.path.isfile(p) and not p.endswith(('.py', '.pyc'))
                  and '/invariance_v276/tmp/' not in p and '/invariance_v276/v276_prereg' not in p)
    mods = sorted({os.path.abspath(m.__file__) for m in list(sys.modules.values())
                   if isinstance(getattr(m, '__file__', None), str) and os.path.isabs(m.__file__) and m.__file__.startswith(root) and os.path.isfile(m.__file__)})
    return data, mods


# ---------------------------------------------------------------- local run
def run_local(gates_only, res):
    reg = set()
    if not gates_only:
        reg = check_hashes(OUT / 'v276_prereg_stage1.sha256', HERE) | check_hashes(OUT / 'v276_prereg.sha256', HERE)
        res['registration'] = dict(registered_files=len(reg), all_match=True); say('registration:', res['registration'])
    B0 = V5.load(); src = sources()
    S.phys_tables()                                  # read here too, so that the input list (registration) has the i3n tables
    res['model_input_columns'] = dict(rows=list(src[0].columns), relax=list(src[1].columns), events=list(src[2].columns),
                                      cv=list(src[3].columns) if src[3] is not None else None, calibration_truth_table=list(B0.columns))
    F0 = src[1]
    Fref = pd.read_csv(OUT / 'feats_ref.csv'); FT2 = pd.read_csv(OUT / 'feats_T2.csv'); FT4 = pd.read_csv(OUT / 'feats_T4.csv')
    G = res['gates'] = {}
    G['G2_features'] = feature_gate(Fref, F0); say('G2 recomputed features = relax.csv:', {k: v for k, v in G['G2_features'].items()})
    rd = pd.read_csv(OUT / 'reads.csv'); h0s = set(zip(src[0].member, src[0].h0_row))
    G['window_starts_at_h0'] = dict(calls=int(len(rd)), start_is_a_sweep_h0=int(sum((a, b) in h0s for a, b in zip(rd.member, rd.start))))
    G['window_starts_at_h0']['ok'] = G['window_starts_at_h0']['calls'] == G['window_starts_at_h0']['start_is_a_sweep_h0']
    ctx = get_context('fork')
    if gates_only:
        with ctx.Pool(1) as pool:
            R0 = {n: (r, s) for n, r, s in pool.map(requests, [('R0', src, B0)])}['R0']
    else:
        R0 = None
    # G1 always computed first (in the main run inside the pool below)
    saved = pd.read_csv(S.OUT / 'rows_tau5_N.csv'); tr = pd.read_csv(S.OUT / 'tau_range.csv'); tr = tr[tr.config == 'N']

    def gate_g1(R0p):
        a = R0p[0][np.isclose(R0p[0].tau, 0.05)].merge(saved[['row_id', 'central', 'pred']], on='row_id', suffixes=('', '_saved'), how='outer', indicator=True)
        sc = R0p[1].pivot(index='tau', columns='cell', values='scale')
        sv = tr.set_index('tau')[[f'scale_{c}' for c in CELLS]]; sv.columns = list(CELLS)
        sc_ok = bool(all(np.isclose(t, sv.index, rtol=0, atol=1e-12).any() for t in sc.index)
                     and all((sc.loc[t, c] == sv.loc[sv.index[np.isclose(sv.index, t, rtol=0, atol=1e-12)][0], c]) for t in sc.index for c in CELLS))
        g = dict(rows=int((a._merge == 'both').sum()), max_rel_central=float(np.nanmax(rel(a.central, a.central_saved))),
                 max_rel_pred=float(np.nanmax(rel(a.pred, a.pred_saved))), scales_equal_tau_range=sc_ok)
        g['ok'] = bool(g['rows'] == 672 and (a._merge == 'both').all() and g['max_rel_central'] <= TOL_LOCAL and g['max_rel_pred'] <= TOL_LOCAL and sc_ok)
        return g

    if gates_only:
        G['G1_reproduction'] = gate_g1(R0); say('G1 reproduction of V273 N:', G['G1_reproduction'])
        res['gates_ok'] = bool(all(g['ok'] for g in G.values()))
        data, mods = project_inputs()
        (OUT / 'inputs_read.txt').write_text('# data files read by the gates run\n' + ''.join(os.path.relpath(p, HERE) + '\n' for p in data)
                                             + '# project modules imported\n' + ''.join(os.path.relpath(p, HERE) + '\n' for p in mods))
        say(f'gates {"passed" if res["gates_ok"] else "FAILED"}; {len(data)} data files and {len(mods)} modules listed in invariance_v276/inputs_read.txt')
        return
    if not (G['G2_features']['ok'] and G['window_starts_at_h0']['ok']):
        raise Stop('gate G2 failed; no test computed (protocol)')
    # ---- tests (inputs built first, then every V273 N computation in a pool)
    rng = np.random.default_rng(SEED)
    fac = factors(src[0], rng)
    Rs = src[0].sort_values('row_id')
    perm_soc = pd.Series(rng.permutation(Rs.soc_point.to_numpy()), index=Rs.row_id.to_numpy())
    perm_ds = pd.Series(rng.permutation(Rs.ds_query.to_numpy()), index=Rs.row_id.to_numpy())
    src_ref = (src[0], with_feats(F0, Fref, src[0].row_id.tolist()), src[2], src[3])
    jobs, info = [('R0', src, B0), ('Rref', src_ref, B0)], {}
    for c in CELLS:
        for t, fn, base, extra in (('T1', t1, src, fac), ('T3', t3, src, None), ('T2', t2, src_ref, FT2), ('T4', t4, src_ref, FT4)):
            if t == 'T3':
                s, b, i = t3(base, B0, c, perm_soc, perm_ds)
            else:
                s, b, i = fn(base, B0, c, extra)
            jobs.append((f'{t}_{c}', s, b)); info[f'{t}_{c}'] = i
    say('jobs:', [j[0] for j in jobs])
    with ctx.Pool(7) as pool:
        out = {n: (r, s) for n, r, s in pool.map(requests, jobs, chunksize=1)}
    G['G1_reproduction'] = gate_g1(out['R0']); say('G1 reproduction of V273 N:', G['G1_reproduction'])
    if not G['G1_reproduction']['ok']:
        raise Stop('gate G1 failed; tests not judged (protocol)')
    res['gates_ok'] = True
    T = res['tests'] = {}
    for c in CELLS:
        for t, ref in (('T1', 'R0'), ('T3', 'R0'), ('T2', 'Rref'), ('T4', 'Rref')):
            k = f'{t}_{c}'; T[k] = dict(compare(out[k], out[ref], c, TOL_LOCAL), against=ref, changed=info[k])
            say(f"{k:7s} vs {ref:4s}: rows {T[k]['rows']}  max rel central {T[k]['max_rel_central']:.2e}  pred {T[k]['max_rel_pred']:.2e}  "
                f"scales equal {T[k]['scales_equal']}  nan {T[k]['nan_mismatch']}  -> {'pass' if T[k]['ok'] else 'FAIL'}")
    res['ref_vs_R0_informative'] = {c: compare(out['Rref'], out['R0'], c, np.inf) for c in CELLS}
    for c in CELLS:
        x = res['ref_vs_R0_informative'][c]; say(f"R_ref vs R0 ({c}, informative): max rel central {x['max_rel_central']:.2e}  pred {x['max_rel_pred']:.2e}  scales equal {x['scales_equal']}")
    res['local_all_pass'] = bool(all(v['ok'] for v in T.values()))
    say('local part:', 'all tests pass' if res['local_all_pass'] else 'FAILED: ' + ', '.join(k for k, v in T.items() if not v['ok']))
    for n, (r, s) in out.items():
        r.to_csv(OUT / f'requests_{n}.csv', index=False, float_format='%.17g'); s.to_csv(OUT / f'scales_{n}.csv', index=False)
    data, mods = project_inputs()
    rel_ = {os.path.relpath(p, HERE) for p in data + mods}
    res['unregistered_reads'] = sorted(rel_ - reg)
    say('files read but not registered:', res['unregistered_reads'] or 'none')


# ---------------------------------------------------------------- remote comparisons
RCOLS = ['pred_power_W', 'a', 'b', 'c_mOhm', 'n_events', 'soc0']


def remote_compare(A, Bref, key):
    M = A.merge(Bref, on=key, suffixes=('', '_ref'), how='outer', indicator=True)
    out = dict(rows=int((M._merge == 'both').sum()), unmatched=int((M._merge != 'both').sum()))
    for c in RCOLS:
        out[f'max_rel_{c}'] = float(np.nanmax(rel(M[c], M[c + '_ref']))) if len(M) else 0.0
        out[f'nan_mismatch_{c}'] = int((M[c].isna() != M[c + '_ref'].isna()).sum())
    out['status_equal'] = bool((M.status.astype(str) == M.status_ref.astype(str)).all())
    out['timeouts'] = int((M.status.astype(str) == 'timeout').sum() + (M.status_ref.astype(str) == 'timeout').sum())
    out['ok'] = bool(out['unmatched'] == 0 and out['status_equal'] and out['timeouts'] == 0
                     and all(out[f'max_rel_{c}'] <= TOL_REMOTE and out[f'nan_mismatch_{c}'] == 0 for c in RCOLS))
    return out


def run_remote(gate, res):
    R0 = pd.read_csv(REM / 'out/R0/pred.csv')
    if gate:
        V = pd.read_csv(S.RS / 'n_all' / 'pred.csv'); V = V[V.sweep_id.isin(SAMPLE)]
        g = remote_compare(R0[['row_id', 'status'] + RCOLS], V[['row_id', 'status'] + RCOLS], ['row_id'])
        g['rows_expected'] = int(len(V)); g['ok'] = bool(g['ok'] and g['rows'] == len(V) == 26)
        res['G3_remote_reproduction'] = g; say('G3 remote R0 = V273 n_all (sample sweeps):', g)
        return
    reg = check_hashes(OUT / 'v276_prereg_stage1.sha256', HERE) | check_hashes(OUT / 'v276_prereg.sha256', HERE)
    res['registration'] = dict(registered_files=len(reg), all_match=True)
    T = res['remote_tests'] = {}
    for t in ('T1', 'T3', 'T4'):
        A = pd.read_csv(REM / f'out/{t}/pred.csv')
        T[t] = remote_compare(A[['row_id', 'status'] + RCOLS], R0[['row_id', 'status'] + RCOLS], ['row_id'])
        T[t]['ok'] = bool(T[t]['ok'] and T[t]['rows'] == 26)
        say(f"remote {t}: rows {T[t]['rows']}  max rel pred {T[t]['max_rel_pred_power_W']:.2e}  a {T[t]['max_rel_a']:.2e}  status equal {T[t]['status_equal']}  -> {'pass' if T[t]['ok'] else 'FAIL'}")
    A = pd.read_csv(REM / 'out/T2/pred.csv')                     # one row per prefix job: the job's last row (q_k)
    T['T2'] = remote_compare(A[['row_id', 'status'] + RCOLS], R0[['row_id', 'status'] + RCOLS], ['row_id'])
    T['T2']['ok'] = bool(T['T2']['ok'] and T['T2']['rows'] == 26)
    say(f"remote T2 (prefix runs): rows {T['T2']['rows']}  max rel pred {T['T2']['max_rel_pred_power_W']:.2e}  a {T['T2']['max_rel_a']:.2e}  status equal {T['T2']['status_equal']}  -> {'pass' if T['T2']['ok'] else 'FAIL'}")
    res['remote_all_pass'] = bool(all(v['ok'] for v in T.values()))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--gates-only', action='store_true'); g.add_argument('--remote-gate', action='store_true'); g.add_argument('--remote', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    stem = 'gates_only' if a.gates_only else ('remote_gate' if a.remote_gate else ('remote' if a.remote else 'run'))
    res = dict(protocol='PROTOCOL_INVARIANCE_V276.md', mode=stem)
    try:
        if a.remote_gate or a.remote:
            run_remote(a.remote_gate, res)
        else:
            run_local(a.gates_only, res)
    except Stop as e:
        res['stopped'] = str(e); say('STOPPED:', e)
        raise SystemExit(1)
    finally:
        dump(OUT / f'{stem}.json', V2.clean(res))
        (OUT / f'{stem}.log').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
