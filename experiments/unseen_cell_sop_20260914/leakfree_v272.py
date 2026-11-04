"""V272 (PROTOCOL_LEAKFREE_V272.md): the light configuration without the two leaks found in the evidence audit.
D1: the five local predictors (prev, L4, M6, DN6, G1; sop_ident/run_local.sh settings, code and inputs copied to
    leakfree_v272/src) choose their coefficient region from the pre-query rest voltage V_pre instead of the test
    schedule's SOC set point: each model learns the three boundary voltages (discharge SOC 0.1|0.15 and 0.3|0.4, charge
    0.7|0.8) from the (V_pre, soc_point) of all rows of its own training cells; the evaluated cell's soc_point is not used.
D2: the calibration of evaluated cell c uses no model fitted with c: discharge rows of each other cell d from the models
    fitted on the third cell alone (V268 strict, boundaries from that cell), charge rows of the other cells from the
    fold-c model (in sample), short-rest class (B) from the fold-B M3 (A1/A17 SOP pairs, margin 0) on the A discharge rows.
Short-rest rows (B 25 C 2 s): V257 centre (fold-B M3, margin 0). Configurations R0 (V267 reproduction), R1, S1, S2,
P (main), P2 as in the protocol; tau = 4-6 %.
usage: python leakfree_v272.py --gates-only     (reproduction gates; no V_pre region is computed)
       python leakfree_v272.py                  (main run, once; requires leakfree_v272/v272_prereg.sha256 to match)"""
import os
import sys

READS = set()


def _audit(event, args):
    """files opened for reading (inputs of the run, listed for the pre-registration hashes)"""
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
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
from run_envelope import HERE, dump  # noqa: E402
import closedloop_v255 as V5, closedloop_v256 as V6, closedloop_v257 as V7, light_v267 as L7  # noqa: E402

OUT = HERE / 'leakfree_v272'; SRC = OUT / 'src'; TOOLS = SRC / 'sop_ident' / 'tools'
sys.path.insert(0, str(TOOLS))
import local_sop as ls, local_sop2 as ls2  # noqa: E402
assert os.path.dirname(os.path.abspath(ls.__file__)) == str(TOOLS) and os.path.dirname(os.path.abspath(ls2.__file__)) == str(TOOLS)
warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

CELLS = V5.CELLS; TAUS = V6.TAUS
NAMES = ('prev', 'L4', 'M6', 'DN6', 'G1')
CH = 'chg>=0.8@step:pdiff,slope,lT,sqA'                                  # sop_ident/run_local.sh, strict_v268 strict_local.py
SPECS = {'L4': (['*@prev:trend10,trend1,invT'], False, False),
         'M6': (['*@prev:trendS,trendS1,dlT,invT,sqA', CH, 'chg>=0.8|first@off:'], False, False),
         'DN6': (['*@prev:trendS,amp,ext,invT', CH, 'chg>=0.8|first@off:'], True, False),
         'G1': (['*@prev:ext', CH, 'chg>=0.8|first@off:', 'chg>=0.8|firstcv@off!:', 'dis>=0.4|firstcv@off!:'], True, True)}
BOUNDS = (('dis1', ('dis<=0.1',), ('dis0.15-0.3', 'dis>=0.4')),         # soc <= 0.1 | >= 0.15
          ('dis2', ('dis<=0.1', 'dis0.15-0.3'), ('dis>=0.4',)),         # soc <= 0.3 | >= 0.4
          ('chg', ('chg<=0.7',), ('chg>=0.8',)))                        # soc <= 0.7 | >= 0.8
SOC_BOUND = {'dis1': '0.1|0.15', 'dis2': '0.3|0.4', 'chg': '0.7|0.8'}
RULES = ('min_error', 'median_mid')                                    # main rule (P, S1), secondary rule (P2)
LOG = []


class Stop(Exception):
    pass


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


# ---------------------------------------------------------------- regions from the rest voltage
def threshold(lo, up, rule):
    """boundary voltage t between the lower and the upper group (V > t: upper). min_error: the neighbour midpoint with
    the fewest misassigned rows (ties: middle candidate, the lower one of the two middle ones for an even count, so the
    chosen t always has the minimum error); median_mid: midpoint of the two group medians. An empty group: undefined
    (t = -inf, every row of that direction above the boundary)."""
    lo = np.sort(np.asarray(lo, float)); up = np.sort(np.asarray(up, float))
    rec = dict(n_lower=int(len(lo)), n_upper=int(len(up)), lower_max=float(lo.max()) if len(lo) else np.nan,
               upper_min=float(up.min()) if len(up) else np.nan)
    if not len(lo) or not len(up):
        return dict(rec, defined=False, t=-np.inf, errors=np.nan, ties=0)
    if rule == 'median_mid':
        t, ties = 0.5 * (float(np.median(lo)) + float(np.median(up))), 1
    elif rule == 'min_error':
        v = np.unique(np.concatenate([lo, up]))
        if len(v) < 2:
            raise ValueError('all boundary rows share one voltage')
        cand = 0.5 * (v[1:] + v[:-1])
        err = (lo[None, :] > cand[:, None]).sum(1) + (up[None, :] <= cand[:, None]).sum(1)
        best = cand[err == err.min()]; t, ties = float(best[(len(best) - 1) // 2]), int(len(best))
    else:
        raise ValueError(rule)
    return dict(rec, defined=True, t=float(t), errors=int((lo > t).sum() + (up <= t).sum()), ties=ties)


def learn(T, rule, col='V_pre'):
    """the three boundaries from the training rows T: groups from their soc_point region, values from column col"""
    sreg = np.array([ls.region(d, s) for d, s in zip(T.direction, T.soc_point)]); v = T[col].to_numpy(float)
    return {b: threshold(v[np.isin(sreg, lo)], v[np.isin(sreg, up)], rule) for b, lo, up in BOUNDS}


def assign(D, th, col='V_pre'):
    v = D[col].to_numpy(float)
    dis = np.where(v <= th['dis1']['t'], 'dis<=0.1', np.where(v <= th['dis2']['t'], 'dis0.15-0.3', 'dis>=0.4'))
    chg = np.where(v <= th['chg']['t'], 'chg<=0.7', 'chg>=0.8')
    return np.where((D.direction == 'discharge').to_numpy(), dis, chg).astype(object)


def boundaries(D, rule, col='V_pre'):
    """('fold', c): the model without c (its two training cells); ('strict', e): the model fitted on e alone"""
    TH = {}
    for c in CELLS:
        TH[('fold', c)] = learn(D[D.evaluation_group != c], rule, col)
        TH[('strict', c)] = learn(D[D.evaluation_group == c], rule, col)
    return TH


# ---------------------------------------------------------------- local predictors
def folds(name, D):
    """all folds of predictor `name` over D's evaluation_group labels (strict_v268 strict_local.folds)"""
    if name == 'prev':
        out = []
        for c in sorted(D.evaluation_group.unique()):
            par = ls.fit(D[D.evaluation_group != c], 'prev', True)
            P = pd.concat([D[['row_id']], ls.predict(D, 'prev', par)], axis=1)
            out.append(P[['row_id', 'pred_power_W']].assign(fold=c))
        return pd.concat(out)
    spec, direct, gap = SPECS[name]
    _, _, F = ls2.run(D, ls2.parse_specs(spec), True, direct, gap)
    return F[['fold', 'row_id', 'pred_power_W']]


def loco(D0, name, TH=None, col='V_pre'):
    """fold c prediction of every row; TH None: soc_point regions, else regions from the boundaries of model c"""
    if TH is None:
        return folds(name, D0).reset_index(drop=True)
    out = []
    for c in CELLS:
        D = D0.copy(); D['region'] = assign(D, TH[('fold', c)], col)
        F = folds(name, D); out.append(F[F.fold == c])
    return pd.concat(out, ignore_index=True)


def strict(D0, name, TH=None, col='V_pre'):
    """the model fitted on cell e alone (labels rewritten to e / ZZ, fold ZZ) for the rows of the other cells"""
    own = D0.set_index('row_id').evaluation_group; out = []
    for e in CELLS:
        D = D0.copy(); D['evaluation_group'] = np.where(D0.evaluation_group == e, e, 'ZZ')
        if TH is not None:
            D['region'] = assign(D, TH[('strict', e)], col)
        F = folds(name, D); F = F[F.fold == 'ZZ'].copy(); F['cell'] = F.row_id.map(own); F = F[F.cell != e]
        out.append(F.assign(name=name, train_cell=e)[['name', 'train_cell', 'row_id', 'cell', 'pred_power_W']])
    return pd.concat(out, ignore_index=True)


def centres(Fs):
    """light centre per (fold, row): log median of the five local predictions (light_v267.light_folds)"""
    F = None
    for n in NAMES:
        P = Fs[n][['fold', 'row_id', 'pred_power_W']].rename(columns={'pred_power_W': n})
        F = P if F is None else F.merge(P, on=['fold', 'row_id'], how='outer')
    with np.errstate(all='ignore'):
        L = np.log(F[list(NAMES)].to_numpy(float)); L[~np.isfinite(L)] = np.nan
        F['m'] = np.nanmedian(L, axis=1)
    F['n_local'] = np.isfinite(L).sum(1)
    return F


def with_centres(B, C, label):
    B = B.copy(); P = C.pivot(index='row_id', columns='fold', values='m').reindex(B.row_id)
    for c in CELLS:
        B[f'sop_{c}'] = np.exp(P[c].to_numpy())
    miss = B.in_events & B[[f'sop_{c}' for c in CELLS]].isna().any(axis=1)
    if miss.any():
        raise Stop(f'{label}: {int(miss.sum())} in-event rows without a local centre (protocol: stop)')
    return B


def strict_centres(S):
    """log median of the five strict predictions per (train cell, row) (strict_v268.strict_centres without i3c)"""
    W = S.pivot_table(index=['train_cell', 'row_id'], columns='name', values='pred_power_W').reset_index()
    with np.errstate(all='ignore'):
        L = np.log(W[list(NAMES)].to_numpy(float)); L[~np.isfinite(L)] = np.nan
        W['centre'] = np.exp(np.nanmedian(L, axis=1))
    return W.set_index(['train_cell', 'row_id']).centre


# ---------------------------------------------------------------- short-rest class (M3, V257)
def m3(B):
    """fold-B M3 at margin 0: the three short-rest rows (V257 centre) and the anchored A discharge rows, the latter also
    with their own fold weights (V257 calibration, fitted with B's SOP pairs)"""
    M = V7.model_rows().merge(B[['sheet', 'v178_row', 'row_id', 'truth', 'in_events', 'temp_C']], on=['sheet', 'v178_row'], how='inner')
    assert len(M) == 672, len(M)
    W = json.load(open(HERE / 'cross_sop_v199/weights.json'))
    WB = {(k.split('|')[0], float(k.split('|')[1])): np.array(v) for k, v in W['B'].items()}
    S = M[~M.in_events].copy(); assert len(S) == 3 and (S.evaluation_group == 'B').all() and S.has_anchor.all()
    A = M[M.evaluation_group.isin(['A1', 'A17']) & (M.direction == 'discharge') & M.has_anchor].copy()
    Ab = A.copy(); Ab['w'] = [WB[(d, 2. if h == 2. else 10.)] for d, h in zip(Ab.direction, Ab.horizon_s)]
    with Pool(8) as pool:
        S['central_W'] = V7.powers(S, np.zeros(len(S)), pool)
        A['central_W'] = V7.powers(A, np.zeros(len(A)), pool)
        A['central_foldB_W'] = V7.powers(Ab, np.zeros(len(Ab)), pool)
    cols = ['row_id', 'evaluation_group', 'direction', 'temp_C', 'horizon_s', 'has_anchor', 'truth']
    return S[cols + ['central_W']].reset_index(drop=True), A[cols + ['central_W', 'central_foldB_W']].reset_index(drop=True)


# ---------------------------------------------------------------- closed loop
def third(c, d):
    return [x for x in CELLS if x not in (c, d)][0]


def cal_out(short_cal):
    """V255/V257 calibration: each other cell from its own fold model (fitted with c); short class given"""
    def f(B, K, c, k):
        if k == 'short':
            return short_cal[0], short_cal[1], 0
        p, t = V5.calibration(B, K, c, k, 'out')
        return p, t, 0
    return f


def cal_strict(Sc, charge, short_cal):
    """discharge: each other cell d from the strict centres of the third cell (rows without a centre dropped and
    counted); charge: V5.calibration 'out' (V268) or 'in' (fold-c model in sample, V272); short class given"""
    def f(B, K, c, k):
        if k == 'short':
            return short_cal[0], short_cal[1], 0
        if k == 'discharge':
            m = (B.evaluation_group != c) & (K == k); idx = B.index[m]
            p = np.array([Sc.get((third(c, B.at[i, 'evaluation_group']), B.at[i, 'row_id']), np.nan) for i in idx], dtype=float)
            t = B.truth[m].to_numpy(); ok = np.isfinite(p)
            return p[ok], t[ok], int((~ok).sum())
        p, t = V5.calibration(B, K, c, k, charge)
        if not np.isfinite(p).all():
            raise Stop(f'{charge} calibration of {c}|{k} has rows without a centre')
        return p, t, 0
    return f


def score(B, groups_all, tau, short_eval, cal):
    """V255 main configuration (hybrid, per class) with the short-rest centre and the calibration given"""
    K = V5.classes(B, 'hybrid', 'class'); parts, scales = [], {}
    for c in CELLS:
        ev = B[B.evaluation_group == c].copy(); ev['cls'] = K[ev.index]
        ev['central'] = np.where(ev.cls == 'short', short_eval.reindex(ev.index), ev[f'sop_{c}']); ev['pred'] = np.nan
        for k in sorted(ev.cls.unique()):
            p, t, dropped = cal(B, K, c, k); s, floor = V5.choose(p, t, tau)
            scales[f'{c}|{k}'] = dict(scale=s, floor_bound=floor, calibration_rows=int(len(p)), dropped_without_centre=dropped,
                                      calibration_int_rate=float(np.mean(s * p > t + 1e-9)))
            ev.loc[ev.cls == k, 'pred'] = s * ev.loc[ev.cls == k, 'central']
        parts.append(ev)
    J = pd.concat(parts); assert len(J) == 672 and J.pred.notna().all()
    summ, G = V5.group_table(J, groups_all)
    return dict(tau=float(tau), groups_pass=int(G.passed.sum()), int_rate=summ['int_rate'], util_q50=summ['util_q50'],
                depth_max=summ['depth_max'], scales=scales,
                failed={f'{x.evaluation_group} {x.direction} {x.temp_C} C {x.horizon_s:g} s': x.reason for x in G[~G.passed].itertuples()}), G, J


def sweep(B, groups_all, short_eval, cal):
    per, keep = [], None
    for tau in TAUS:
        r, G, J = score(B, groups_all, float(tau), short_eval, cal); per.append(r)
        if abs(tau - 0.05) < 1e-9:
            keep = (G, J)
    gp = [p['groups_pass'] for p in per]
    return dict(groups_tau_4_6=[int(min(gp)), float(np.median(gp)), int(max(gp))], int_rate_max=float(max(p['int_rate'] for p in per)),
                per_tau=per), keep


# ---------------------------------------------------------------- gates
def compare(mine, ref, keys, tol):
    F = mine.merge(ref, on=keys, how='outer', suffixes=('', '_ref'), indicator=True)
    both = int((F._merge == 'both').sum())
    with np.errstate(all='ignore'):
        r = (F.pred_power_W / F.pred_power_W_ref - 1).abs()
    r[(F.pred_power_W == 0) & (F.pred_power_W_ref == 0)] = 0.0
    nan_mis = int((F.pred_power_W.isna() != F.pred_power_W_ref.isna()).sum())
    mx = float(np.nanmax(r)) if r.notna().any() else 0.0
    return dict(rows=int(len(mine)), ref_rows=int(len(ref)), matched=both, max_rel_diff=mx, nan_mismatch=nan_mis,
                ok=bool(both == len(F) == len(mine) == len(ref) and mx <= tol and nan_mis == 0))


def gate_local(Fs):
    out = {n: compare(Fs[n], pd.read_csv(SRC / 'sop_ident/out/local' / n / 'folds.csv')[['fold', 'row_id', 'pred_power_W']],
                      ['fold', 'row_id'], 1e-9) for n in NAMES}
    return dict(out, ok=all(v['ok'] for v in out.values()))


def gate_strict(S):
    R = pd.read_csv(HERE / 'strict_v268/src/local_strict.csv')[['name', 'train_cell', 'row_id', 'pred_power_W']]
    out = {n: compare(S[S.name == n][['name', 'train_cell', 'row_id', 'pred_power_W']], R[R.name == n], ['name', 'train_cell', 'row_id'], 1e-9)
           for n in NAMES}
    return dict(out, ok=all(v['ok'] for v in out.values()))


def gate_m3(Sm, Am):
    R = pd.read_csv(HERE / 'closedloop_v257/rows_short.csv').set_index('row_id').central_new
    C = pd.read_csv(HERE / 'closedloop_v257/calibration_short.csv').set_index('row_id').central_W
    ds = float(np.max(np.abs(Sm.central_W.to_numpy() - R.reindex(Sm.row_id).to_numpy())))
    da = float(np.max(np.abs(Am.central_W.to_numpy() - C.reindex(Am.row_id).to_numpy())))
    same = set(Sm.row_id) == set(R.index) and set(Am.row_id) == set(C.index)
    return dict(short_rows=int(len(Sm)), short_max_abs_W=ds, A_rows=int(len(Am)), A_own_max_abs_W=da, same_rows=bool(same),
                A_foldB_over_own_median=float(np.median(Am.central_foldB_W / Am.central_W)),
                ok=bool(same and ds <= 1e-9 and da <= 1e-9))


def gate_tau(res, ref_csv, config):
    T = pd.read_csv(ref_csv); T = T[T.config == config]; rows, ok = [], True
    for p in res['per_tau']:
        q = T[np.isclose(T.tau, p['tau'], rtol=0, atol=1e-12)]
        if len(q) != 1:
            rows.append(dict(tau=p['tau'], missing=True)); ok = False; continue
        q = q.iloc[0]; di = abs(p['int_rate'] - q.int_rate); du = abs(p['util_q50'] - q.util_q50)
        good = bool(p['groups_pass'] == int(q.groups_pass) and di <= 1e-12)
        rows.append(dict(tau=p['tau'], groups=p['groups_pass'], groups_ref=int(q.groups_pass), int_rate_abs_diff=di, util_q50_abs_diff=du, ok=good))
        ok &= good
    return dict(reference=f'{ref_csv.relative_to(HERE)} ({config})', groups_tau_4_6=res['groups_tau_4_6'], per_tau=rows,
                ok=bool(ok and len(rows) == len(TAUS)))


def gate_machinery(D, Fsoc, Ssoc):
    """the V_pre region code on known answers, without any V_pre region: (a) threshold() on toy groups; (b) min-error
    boundaries learned with soc_point in place of V_pre separate the training rows without error (B alone: no charge
    boundary); (c) loco()/strict() with the fixed boundaries 0.125/0.35/0.75 on soc_point reproduce the soc_point-region
    predictions."""
    a = dict(separable=threshold([1, 2, 3], [4, 5], 'min_error'), tie=threshold([1, 3], [2, 4], 'min_error'),
             median_mid=threshold([1, 2, 3], [4, 6], 'median_mid'), empty=threshold([], [4, 5], 'min_error'))
    ok_a = bool(a['separable']['t'] == 3.5 and a['separable']['errors'] == 0 and a['tie']['t'] == 1.5 and a['tie']['ties'] == 2
                and a['tie']['errors'] == 1 and a['median_mid']['t'] == 3.5 and not a['empty']['defined'])
    TH = boundaries(D, 'min_error', 'soc_point'); b = {}
    for (kind, cell), th in TH.items():
        for bd, x in th.items():
            b[f'{kind}|{cell}|{bd}'] = dict(defined=x['defined'], t=x['t'] if x['defined'] else None, errors=x['errors'] if x['defined'] else None)
    undefined = sorted(k for k, v in b.items() if not v['defined'])
    ok_b = bool(undefined == ['strict|B|chg'] and all(v['errors'] == 0 for v in b.values() if v['defined']))
    fix = {bd: dict(t=t, defined=True) for bd, t in (('dis1', 0.125), ('dis2', 0.35), ('chg', 0.75))}
    TF = {(kind, c): fix for kind in ('fold', 'strict') for c in CELLS}
    c_loc = {n: compare(loco(D, n, TF, 'soc_point'), Fsoc[n], ['fold', 'row_id'], 1e-12) for n in NAMES}
    S2 = pd.concat([strict(D, n, TF, 'soc_point') for n in NAMES], ignore_index=True)
    c_str = {n: compare(S2[S2.name == n], Ssoc[Ssoc.name == n], ['name', 'train_cell', 'row_id', 'cell'], 1e-12) for n in NAMES}
    ok_c = all(v['ok'] for v in c_loc.values()) and all(v['ok'] for v in c_str.values())
    return dict(a_toy=dict(ok=ok_a, **{k: dict(t=v['t'] if v['defined'] else None, errors=v['errors'] if v['defined'] else None, ties=v['ties'])
                                        for k, v in a.items()}),
                b_soc_as_voltage=dict(ok=ok_b, undefined=undefined, boundaries=b),
                c_fixed_boundaries=dict(ok=bool(ok_c), loco=c_loc, strict=c_str), ok=bool(ok_a and ok_b and ok_c))


# ---------------------------------------------------------------- records
def threshold_rows(TH, rule):
    recs = []
    for (kind, cell), th in TH.items():
        cells = [x for x in CELLS if x != cell] if kind == 'fold' else [cell]
        for bd, x in th.items():
            recs.append(dict(rule=rule, model=kind, model_cell=cell, training_cells='+'.join(cells), boundary=bd, soc_boundary=SOC_BOUND[bd],
                             defined=x['defined'], t_V=x['t'] if x['defined'] else np.nan, n_lower=x['n_lower'], n_upper=x['n_upper'],
                             lower_max_V=x['lower_max'], upper_min_V=x['upper_min'], errors=x['errors'], ties=x['ties']))
    return recs


def region_changes(D, TH, rule):
    soc = D.region.to_numpy(); recs = []
    for (kind, cell), th in TH.items():
        v = assign(D, th)
        for (g, d), idx in D.groupby(['evaluation_group', 'direction']).groups.items():
            ii = D.index.get_indexer(idx); ch = v[ii] != soc[ii]
            role = ('predict' if g == cell else 'train') if kind == 'fold' else ('train' if g == cell else 'predict')
            tr = pd.Series([f'{x}->{y}' for x, y in zip(soc[ii][ch], v[ii][ch])], dtype=object).value_counts()
            recs.append(dict(rule=rule, model=kind, model_cell=cell, row_cell=g, role=role, direction=d, n_rows=int(len(ii)),
                             n_changed=int(ch.sum()), transitions='; '.join(f'{k}: {n}' for k, n in tr.items())))
    return recs


def project_inputs():
    root = str(HERE.parents[1]) + os.sep; skip = str(OUT / 'v272_prereg.sha256')
    data = sorted(p for p in READS if p.startswith(root) and os.path.isfile(p) and not p.endswith(('.py', '.pyc')) and p != skip)
    mods = sorted({os.path.abspath(m.__file__) for m in list(sys.modules.values())            # (torch._classes/_ops carry a
                   if isinstance(getattr(m, '__file__', None), str) and os.path.isabs(m.__file__)  # relative dummy __file__)
                   and m.__file__.startswith(root) and os.path.isfile(m.__file__)})
    return data, mods


def check_prereg():
    f = OUT / 'v272_prereg.sha256'
    if not f.exists():
        raise Stop('leakfree_v272/v272_prereg.sha256 missing: gates first, then registration (protocol)')
    bad, seen = [], set()
    for line in f.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        h, p = line.split(None, 1); p = p.strip(); seen.add(p); q = HERE / p
        if not q.is_file() or hashlib.sha256(q.read_bytes()).hexdigest() != h:
            bad.append(p)
    if bad:
        raise Stop(f'registered files changed or missing: {bad}')
    return dict(file='leakfree_v272/v272_prereg.sha256', files_checked=len(seen), all_match=True), seen


# ---------------------------------------------------------------- main
def run(gates_only, res):
    if not gates_only:
        res['prereg'], registered = check_prereg(); say('pre-registration hashes:', res['prereg'])
    B = V5.load(); groups_all = B[V5.C1.GK].drop_duplicates().sort_values(V5.C1.GK).reset_index(drop=True); assert len(groups_all) == 15
    D = ls.table(); assert len(D) == 672 and set(D.row_id) == set(B.row_id) and D.V_pre.notna().all()
    G = res['gates'] = {}
    Fsoc = {n: loco(D, n) for n in NAMES}; G['1_local_folds'] = gate_local(Fsoc)
    say('gate 1 local folds (soc_point regions) vs sop_ident folds.csv:', {n: (v['max_rel_diff'], v['nan_mismatch']) for n, v in G['1_local_folds'].items() if n != 'ok'}, 'ok' if G['1_local_folds']['ok'] else 'FAILED')
    Ssoc = pd.concat([strict(D, n) for n in NAMES], ignore_index=True); G['2_strict'] = gate_strict(Ssoc)
    say('gate 2 strict (soc_point regions) vs strict_v268 local_strict.csv:', {n: (v['max_rel_diff'], v['nan_mismatch']) for n, v in G['2_strict'].items() if n != 'ok'}, 'ok' if G['2_strict']['ok'] else 'FAILED')
    Sm, Am = m3(B); G['3_m3'] = gate_m3(Sm, Am); say('gate 3 M3 vs V257:', G['3_m3'])
    Bsoc = with_centres(B, centres(Fsoc), 'soc_point regions')
    mA = B.evaluation_group.isin(['A1', 'A17']) & (B.direction == 'discharge')
    v255_cal = (B.v242[mA].to_numpy(), B.truth[mA].to_numpy())
    R0, R0keep = sweep(Bsoc, groups_all, B.v242, cal_out(v255_cal))
    G['4_R0_vs_V267'] = gate_tau(R0, HERE / 'light_v267/tau_range.csv', 'light_hybrid_class_out')
    say('gate 4 R0 vs V267 light:', G['4_R0_vs_V267']['groups_tau_4_6'], 'ok' if G['4_R0_vs_V267']['ok'] else 'FAILED')
    Scs = strict_centres(Ssoc)
    X, _ = sweep(Bsoc, groups_all, B.v242, cal_strict(Scs, 'out', v255_cal))
    G['5_light_strict_vs_V268'] = gate_tau(X, HERE / 'strict_v268/tau_range.csv', 'light_strict')
    say('gate 5 light strict vs V268:', G['5_light_strict_vs_V268']['groups_tau_4_6'], 'ok' if G['5_light_strict_vs_V268']['ok'] else 'FAILED')
    G['6_machinery'] = gate_machinery(D, Fsoc, Ssoc)
    say('gate 6 region machinery:', {k: v['ok'] for k, v in G['6_machinery'].items() if isinstance(v, dict)}, 'ok' if G['6_machinery']['ok'] else 'FAILED')
    res['gates_ok'] = bool(all(g['ok'] for g in G.values()))
    if not res['gates_ok']:
        raise Stop('a reproduction gate failed; no V_pre region computed (protocol)')
    if gates_only:
        data, mods = project_inputs()
        (OUT / 'inputs_read.txt').write_text('# data files read by the gates run\n' + ''.join(os.path.relpath(p, HERE) + '\n' for p in data)
                                             + '# project modules imported\n' + ''.join(os.path.relpath(p, HERE) + '\n' for p in mods))
        say(f'gates passed; {len(data)} data files and {len(mods)} project modules listed in leakfree_v272/inputs_read.txt')
        return
    # -------- main computation (everything computed before anything is written or printed)
    TH = {r: boundaries(D, r) for r in RULES}
    Fv = {r: {n: loco(D, n, TH[r]) for n in NAMES} for r in RULES}
    Sv = {r: pd.concat([strict(D, n, TH[r]) for n in NAMES], ignore_index=True) for r in RULES}
    Cs, Cv = centres(Fsoc), {r: centres(Fv[r]) for r in RULES}
    Bv = {r: with_centres(B, Cv[r], f'V_pre regions ({r})') for r in RULES}
    Scv = {r: strict_centres(Sv[r]) for r in RULES}
    sv = pd.Series(np.nan, index=B.index); short_ids = B.index[~B.in_events]
    sv[short_ids] = Sm.set_index('row_id').central_W.reindex(B.row_id[short_ids]).to_numpy()
    own, fb = (Am.central_W.to_numpy(), Am.truth.to_numpy()), (Am.central_foldB_W.to_numpy(), Am.truth.to_numpy())
    plan = {'R1': (Bsoc, sv, cal_out(own)), 'S1': (Bv['min_error'], sv, cal_out(own)),
            'S2': (Bsoc, sv, cal_strict(Scs, 'in', fb)), 'P': (Bv['min_error'], sv, cal_strict(Scv['min_error'], 'in', fb)),
            'P2': (Bv['median_mid'], sv, cal_strict(Scv['median_mid'], 'in', fb))}
    CFG, KEEP = {'R0': R0}, {'R0': R0keep}
    for name, (Bc, se, cal) in plan.items():
        CFG[name], KEEP[name] = sweep(Bc, groups_all, se, cal)
    # -------- records
    rows = []
    for name, r in CFG.items():
        for p in r['per_tau']:
            rows.append(dict(config=name, tau=p['tau'], groups_pass=p['groups_pass'], int_rate=p['int_rate'], util_q50=p['util_q50'], depth_max=p['depth_max'],
                             **{f'scale_{k}': v['scale'] for k, v in p['scales'].items()}, **{f'cal_rows_{k}': v['calibration_rows'] for k, v in p['scales'].items()},
                             **{f'dropped_{k}': v['dropped_without_centre'] for k, v in p['scales'].items()}, failed='; '.join(f'{g} ({w})' for g, w in p['failed'].items())))
    pd.DataFrame(rows).to_csv(OUT / 'tau_range.csv', index=False)
    for name, (Gt, Jt) in KEEP.items():
        Gt.to_csv(OUT / f'groups_tau5_{name}.csv', index=False); Jt.to_csv(OUT / f'rows_tau5_{name}.csv', index=False)
    tag = {'min_error': 'vpre_main', 'median_mid': 'vpre_secondary'}
    pd.DataFrame([x for r in RULES for x in threshold_rows(TH[r], tag[r])]).to_csv(OUT / 'thresholds.csv', index=False)
    RC = pd.DataFrame([x for r in RULES for x in region_changes(D, TH[r], tag[r])]); RC.to_csv(OUT / 'region_changes.csv', index=False)
    Cs.to_csv(OUT / 'centres_soc.csv', index=False)
    for r in RULES:
        Cv[r].to_csv(OUT / f'centres_{tag[r]}.csv', index=False); Sv[r].to_csv(OUT / f'strict_{tag[r]}.csv', index=False)
    Ssoc.to_csv(OUT / 'strict_soc.csv', index=False); Sm.to_csv(OUT / 'm3_short.csv', index=False); Am.to_csv(OUT / 'm3_calibration_A.csv', index=False)
    # -------- verdicts (protocol)
    med = {k: v['groups_tau_4_6'][1] for k, v in CFG.items()}; p = med['P']
    res['configs'] = CFG
    res['verdict_P'] = '누설 없이도 가벼운 판 수준 유지' if p >= 12 else ('누설 제거 비용 1~2그룹' if p >= 10 else '지금 결과가 누설에 크게 기댄다')
    res['D1_effect'] = dict(S1_minus_R1=med['S1'] - med['R1'], verdict='구간 표지 영향 작음' if abs(med['S1'] - med['R1']) <= 1 else '구간 표지 영향 있음')
    res['D2_effect'] = dict(S2_minus_R1=med['S2'] - med['R1'])
    res['rule_sensitivity'] = dict(P2_minus_P=med['P2'] - med['P'])
    res['central_open_loop'] = dict(soc=L7.central_stats(Bsoc, 'sop_{}'), **{tag[r]: L7.central_stats(Bv[r], 'sop_{}') for r in RULES})
    ev = RC[(RC.model == 'fold') & (RC.role == 'predict')]
    res['region_changes_evaluated_rows'] = {f'{t}|{g}|{d}': f'{c}/{n}' for t, g, d, c, n in zip(ev.rule, ev.row_cell, ev.direction, ev.n_changed, ev.n_rows)}
    for name in CFG:
        say(f"{name:3s} groups tau 4-6 % {CFG[name]['groups_tau_4_6']}  int max {100 * CFG[name]['int_rate_max']:.2f} %")
    t5 = {k: [x for x in v['per_tau'] if abs(x['tau'] - 0.05) < 1e-9][0] for k, v in CFG.items()}
    for name in CFG:
        x = t5[name]; sc = ' '.join('%s=%.2f' % (k, v['scale']) for k, v in x['scales'].items())
        say(f"  tau 5 % {name}: groups {x['groups_pass']} int {100 * x['int_rate']:.2f} % util {100 * x['util_q50']:.1f} % scales {sc} failed {list(x['failed'])}")
    say('verdict P:', res['verdict_P'], '| D1:', res['D1_effect'], '| D2:', res['D2_effect'], '| rule:', res['rule_sensitivity'])
    data, mods = project_inputs()
    res['unregistered_reads'] = sorted({os.path.relpath(p, HERE) for p in data + mods} - registered)
    say('files read but not registered:', res['unregistered_reads'] or 'none')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gates-only', action='store_true'); a = ap.parse_args()
    OUT.mkdir(exist_ok=True); stem = 'gates_only' if a.gates_only else 'run'
    res = dict(protocol='PROTOCOL_LEAKFREE_V272.md', mode=stem)
    try:
        run(a.gates_only, res)
    except Stop as e:
        res['stopped'] = str(e); say('STOPPED:', e)
        raise SystemExit(1)
    finally:
        dump(OUT / ('gates_only.json' if a.gates_only else 'summary.json'), clean(res))
        (OUT / f'{stem}.log').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
