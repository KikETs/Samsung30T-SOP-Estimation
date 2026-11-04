"""ECM-RLS candidates on the 672-row SOP benchmark (PROTOCOL.md, PROTOCOL_AMENDMENTS.md). Run on the remote server.

    python -I -B src/run_all.py --stage gates      # main-configuration reproduction + synthetic and solver checks
    python -I -B src/run_all.py --stage identify   # RLS passes, causal voltage check, hyperparameter choice
    python -I -B src/run_all.py --stage score      # SOP, post hoc fidelity, convergence, scores, invariance

Candidates: ECM-1RC-RLS and ECM-2RC-RLS as pre-registered (src/ecm_rls.py, unchanged) and ECM-1RC-RLS-A1 and
ECM-2RC-RLS-A1 of amendment 1 (src/ecm_rls_a1.py).
"""
import argparse
import json
import pickle
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ecm_rls as M                                                       # noqa: E402
import ecm_rls_a1 as A                                                    # noqa: E402
from evaluation import (CELLS, ID_COLUMNS, TAUS, load_benchmark, main_reproduction, pilot_functions,  # noqa: E402
                        read_csv, sha256, write_csv)

EXP = Path(__file__).resolve().parents[1]
OUT = EXP / 'results'
WORK = EXP / 'work'
CACHE = EXP / 'cache'
LAMS = M.LAMBDAS
CANDIDATES = {'ECM-1RC-RLS': dict(n_rc=1, version='v1'), 'ECM-2RC-RLS': dict(n_rc=2, version='v1'),
              'ECM-1RC-RLS-A1': dict(n_rc=1, version='a1'), 'ECM-2RC-RLS-A1': dict(n_rc=2, version='a1')}
CONFIGS = {'v1': tuple((lam, False) for lam in LAMS), 'a1': A.CONFIGS}
STEPS_FINAL = 801
SEED = 20261104
WORKERS = 8


def cache_path(member):
    return CACHE / (member.replace('/', '__').replace('#', '_') + '.npz')


def config_label(cfg):
    return f'lam={cfg[0]:g},kappa={"on" if cfg[1] else "off"}'


def tie_rank(cfg):
    """Preferred among tied configurations: kappa off first, then the larger lambda (v1 has kappa off only)."""
    return (cfg[1], -cfg[0])


def check_prereg():
    pre = json.loads((OUT / 'prereg.json').read_text(encoding='utf-8'))
    now = sha256(EXP / 'PROTOCOL.md')
    assert now == pre['sha256']['PROTOCOL.md'], 'PROTOCOL.md changed after freezing'
    assert sha256(EXP / 'src/ecm_rls.py') == pre['sha256']['src/ecm_rls.py'], 'pre-registered predictor changed'
    amend = None
    p = OUT / 'prereg_amendment1.json'
    if p.exists():
        amend = json.loads(p.read_text(encoding='utf-8'))
        for name, h in amend['sha256'].items():
            assert sha256(EXP / name) == h, f'{name} changed after amendment 1 was frozen'
    return pre, now, amend


# ----------------------------------------------------------------------------------------------- predictor side
def pass_task(args):
    """Worker: one causal pass. Only Test_Time, Current and Voltage of the member enter the reader."""
    version, n_rc, member, h0, snaps, trunc = args
    d = np.load(cache_path(member))
    t, I, V = d['t'], d['I'], d['V']
    if trunc is not None:
        t, I, V = t[:trunc].copy(), I[:trunc].copy(), V[:trunc].copy()
    reader = M.Reader({member: dict(t=t, I=I, V=V)})
    run = M.run_pass if version == 'v1' else A.run_pass
    out = run(reader, member, h0, snaps, n_rc, LAMS)
    return dict(member=member, h0=h0, n_rc=n_rc, version=version, out=out, log=reader.log, n_avail=len(t))


def query_table(table):
    """Projection of a row table onto the query identification the predictor may receive."""
    return [{k: r[k] for k in ID_COLUMNS} for r in table]


def group_snaps(queries, check):
    """Snapshot indices per (member, h0): q of every row and p0 of its check pulse; labels name the purpose."""
    groups = {}
    for r in queries:
        key = (r['member'], int(r['h0_row']))
        g = groups.setdefault(key, {})
        g.setdefault(int(r['q_row']), []).append(('q', r['row_id']))
        c = check.get(r['row_id'])
        if c is not None:
            g.setdefault(c[0], []).append(('p', r['row_id']))
    return {k: sorted((s, tuple(v)) for s, v in g.items()) for k, g in groups.items()}


def run_passes(queries, check, pool, version, n_rc):
    tasks = [(version, n_rc, m, h0, snaps, None) for (m, h0), snaps in group_snaps(queries, check).items()]
    res = list(pool.map(pass_task, tasks, chunksize=1))
    snap_q, snap_p, logs = {}, {}, []
    for r in res:
        logs.append(r['log'])
        for s, lst in r['out'].items():
            for kind, rid in lst[0]['label']:
                (snap_q if kind == 'q' else snap_p)[rid] = lst
    return snap_q, snap_p, logs


def params(snap, version):
    """R0, R, kappa, E_rls of a snapshot (v1 stores them in theta = [E, R0, R...])."""
    if version == 'v1':
        th = snap['theta']
        return float(th[1]), th[2:], 0.0, float(th[0])
    return snap['R0'], snap['R'], snap['kappa'], snap['E_rls']


def sop_for(snaps_by_row, rows, version, n_rc, cfg_index, steps):
    """Central SOP for every row from its snapshot at q with configuration cfg_index[i]."""
    picked = [snaps_by_row[r['row_id']][cfg_index[i]] for i, r in enumerate(rows)]
    dis = np.array([r['direction'] == 'discharge' for r in rows])
    h = np.array([float(r['horizon_s']) for r in rows])
    if version == 'v1':
        E, r0, R, TAU, Z = M.snapshot_arrays(picked, n_rc)
        return M.rc_power_n(E, r0, R, TAU, Z, dis, h, steps), picked
    E, r0, R, TAU, Z, K = A.snapshot_arrays(picked, n_rc)
    return A.rc_power_nk(E, r0, R, TAU, Z, K, dis, h, steps), picked


# ------------------------------------------------------------------------------------------------ evaluator side
_D = {}


def raw(member):
    if member not in _D:
        d = np.load(cache_path(member))
        _D[member] = {k: d[k] for k in ('t', 'I', 'V')}
    return _D[member]


def load_check_pulses():
    out = {}
    for r in read_csv(OUT / 'check_pulses.csv'):
        if r['check_pulse_first'] != '':
            out[r['row_id']] = (int(r['check_pulse_first']), int(r['check_pulse_last']))
    return out


def pulse_errors(snap, version, member, a, b):
    """Voltage error of an open-loop prediction of rows a..b from a snapshot taken at a (rows < a)."""
    if snap['status'] == 'missing':
        return None
    D = raw(member)
    t, I, V = D['t'][a:b + 1], D['I'][a:b + 1], D['V'][a:b + 1]
    vh = (M.simulate_measured if version == 'v1' else A.simulate_measured)(snap, t, I)
    err = vh - V
    return dict(end_err_V=float(err[-1]), max_abs_err_V=float(np.max(np.abs(err))), n=len(t))


def check2(rows, snap_p, check, version):
    out = []
    for r in rows:
        c = check.get(r['row_id'])
        if c is None:
            continue
        for ci, cfg in enumerate(CONFIGS[version]):
            s = snap_p[r['row_id']][ci]
            e = pulse_errors(s, version, r['member'], c[0], c[1])
            out.append(dict(row_id=r['row_id'], cell=r['evaluation_group'], direction=r['direction'],
                            temp_C=float(r['temp_C']), horizon_s=float(r['horizon_s']), config=ci,
                            config_label=config_label(cfg), lam=cfg[0], kappa_on=cfg[1], status=s['status'],
                            end_err_V=e['end_err_V'] if e else np.nan, max_abs_err_V=e['max_abs_err_V'] if e else np.nan))
    return out


def select_config(c2, version):
    cfgs = CONFIGS[version]
    sel, table = {}, {}
    for c in CELLS:
        by_row = {}
        for x in c2:
            if x['cell'] != c:
                by_row.setdefault(x['row_id'], {})[x['config']] = x['end_err_V']
        common = [rid for rid, d in by_row.items() if len(d) == len(cfgs) and all(np.isfinite(v) for v in d.values())]
        med = {ci: float(np.median([abs(by_row[rid][ci]) for rid in common])) for ci in range(len(cfgs))}
        best = min(med.values())
        choice = min((ci for ci in med if med[ci] <= best + 1e-12), key=lambda ci: tie_rank(cfgs[ci]))
        sel[c] = choice
        table[c] = dict(rows_used=len(common), median_abs_end_err_V={config_label(cfgs[ci]): v for ci, v in med.items()},
                        selected=config_label(cfgs[choice]))
    return sel, table


def summarize_voltage(recs, candidate, check, selected=None):
    out = []
    keys = sorted({(x['config'], x['cell'], x['direction'], x['temp_C'], x['horizon_s']) for x in recs})
    for ci, cell, d, T, h in keys:
        g = [x for x in recs if (x['config'], x['cell'], x['direction'], x['temp_C'], x['horizon_s']) == (ci, cell, d, T, h)]
        ok = np.array([x['end_err_V'] for x in g if np.isfinite(x['end_err_V'])])
        mx = np.array([x['max_abs_err_V'] for x in g if np.isfinite(x['max_abs_err_V'])])
        out.append(dict(candidate=candidate, check=check, config=g[0]['config_label'], lam=g[0]['lam'],
                        kappa_on=g[0]['kappa_on'], selected_for_this_cell=(selected is not None and selected.get(cell) == ci),
                        cell=cell, direction=d, temp_C=T, horizon_s=h, rows=len(g), missing=len(g) - len(ok),
                        median_abs_end_mV=1e3 * float(np.median(np.abs(ok))) if len(ok) else np.nan,
                        mean_abs_end_mV=1e3 * float(np.mean(np.abs(ok))) if len(ok) else np.nan,
                        p95_abs_end_mV=1e3 * float(np.quantile(np.abs(ok), .95)) if len(ok) else np.nan,
                        median_signed_end_mV=1e3 * float(np.median(ok)) if len(ok) else np.nan,
                        median_max_abs_mV=1e3 * float(np.median(mx)) if len(mx) else np.nan,
                        max_max_abs_mV=1e3 * float(np.max(mx)) if len(mx) else np.nan))
    return out


def identify(table, pool, check):
    """Full predictor pipeline up to the hyperparameter choice (no SOP, no score)."""
    queries = query_table(table)
    res = {}
    for cand, spec in CANDIDATES.items():
        snap_q, snap_p, logs = run_passes(queries, check, pool, spec['version'], spec['n_rc'])
        c2 = check2(table, snap_p, check, spec['version'])
        sel, sel_table = select_config(c2, spec['version'])
        res[cand] = dict(snap_q=snap_q, snap_p=snap_p, logs=logs, check2=c2, cfg_sel=sel, cfg_table=sel_table)
    return res


def central_from(res_c, rows, spec, steps=STEPS_FINAL, cfg_fixed=None):
    idx = [cfg_fixed if cfg_fixed is not None else res_c['cfg_sel'][r['evaluation_group']] for r in rows]
    central, picked = sop_for(res_c['snap_q'], rows, spec['version'], spec['n_rc'], idx, steps)
    status = np.array([p['status'] for p in picked])
    return central, status, picked


# --------------------------------------------------------------------------------------------------------- stages
def stage_gates():
    check_prereg()
    B = load_benchmark()
    F, src = pilot_functions()
    rep = main_reproduction(B, F)
    import checks
    import checks_a1
    syn = checks.main()
    syn_a1 = checks_a1.main()
    gates = dict(main_reproduction=dict(AD=rep['AD']['score'], fixed=rep['fixed']['score'],
                                        AD_request_max_abs_diff_vs_stored=rep['AD_request_max_abs_diff_vs_stored'],
                                        passed=rep['gate_passed']),
                 truth_matches_ad_trajectory_max_abs=B['truth_matches_ad'], synthetic_and_solver=syn['all_passed'],
                 synthetic_and_solver_a1=syn_a1['all_passed'])
    (OUT / 'gates.json').write_text(json.dumps(gates, indent=1, default=float), encoding='utf-8')
    (OUT / 'evaluation_functions_from_run_pilot.py').write_text(
        '# Byte-identical source segments of run_pilot.py (SHA-256 fe7e60ce...4a15e), extracted with ast.\n\n'
        + '\n\n\n'.join(src[k] for k in ('calibrate', 'request', 'score')) + '\n', encoding='utf-8')
    print(json.dumps(gates, indent=1, default=float))
    assert rep['gate_passed'] and syn['all_passed'] and syn_a1['all_passed'], 'gate failed: nothing is scored'


def stage_identify():
    _, _, amend = check_prereg()
    assert amend is not None, 'amendment 1 must be frozen before this stage'
    gates = json.loads((OUT / 'gates.json').read_text(encoding='utf-8'))
    assert gates['main_reproduction']['passed'] and gates['synthetic_and_solver'] and gates['synthetic_and_solver_a1']
    B = load_benchmark()
    rows = B['rows']
    check = load_check_pulses()
    started = time.time()
    with ProcessPoolExecutor(WORKERS) as pool:
        res = identify(rows, pool, check)
    WORK.mkdir(exist_ok=True)
    with open(WORK / 'identify_main.pkl', 'wb') as f:
        pickle.dump(res, f)
    vv = []
    for cand, r in res.items():
        vv += summarize_voltage(r['check2'], cand, 'causal_previous_pulse', r['cfg_sel'])
        write_csv(OUT / f'voltage_rows_causal_{cand}.csv', r['check2'])
    write_csv(OUT / 'voltage_validation_causal.csv', vv)
    sel = {cand: dict(selected={c: config_label(CONFIGS[CANDIDATES[cand]['version']][ci]) for c, ci in r['cfg_sel'].items()},
                      table=r['cfg_table']) for cand, r in res.items()}
    (OUT / 'lambda_selection.json').write_text(json.dumps(dict(
        rule='per candidate and evaluation cell: smallest median |end-of-pulse error| of the causal check on the other '
             'two cells over rows with an estimate for every configuration; ties to kappa off, then the larger lambda',
        candidates=sel, elapsed_s=time.time() - started), indent=1), encoding='utf-8')
    print(json.dumps(sel, indent=1))


def scores_for(central, F, B):
    y, orders, groups = B['y'], B['orders'], B['groups']
    out = {}
    for tau in TAUS:
        scales, floor, counts = F['calibrate'](central, y, orders, tau)
        for adaptive in (False, True):
            pred = F['request'](central, y, orders, scales, tau, adaptive)
            sm, G = F['score'](pred, y, groups)
            out[(float(tau), 'AD' if adaptive else 'fixed')] = dict(score=sm, groups=G, pred=pred, scales=scales)
    return out


def decisions(central, F, B):
    """Exceedance flags, group verdicts and scales at tau = 5 % for both policies."""
    y, orders, groups = B['y'], B['orders'], B['groups']
    scales, _, _ = F['calibrate'](central, y, orders, .05)
    d = {'scales': scales}
    for adaptive in (False, True):
        pred = F['request'](central, y, orders, scales, .05, adaptive)
        sm, G = F['score'](pred, y, groups)
        d['AD' if adaptive else 'fixed'] = dict(exceed=(pred / y > 1 + 1e-9), passed=[g['passed'] for g in G],
                                                strict=[g['strict'] for g in G])
    return d


def same_decisions(a, b):
    return bool(a['scales'] == b['scales'] and all(np.array_equal(a[p]['exceed'], b[p]['exceed'])
                                                   and a[p]['passed'] == b[p]['passed'] and a[p]['strict'] == b[p]['strict']
                                                   for p in ('fixed', 'AD')))


def audit_logs(logs, rows, check):
    """Every read before a query snapshot lies in [h0, q); before a check snapshot in [h0, p0)."""
    info = {r['row_id']: (int(r['h0_row']), int(r['q_row'])) for r in rows}
    n_snap = viol = n_reads = 0
    for log in logs:
        lo_min, hi_max = None, None
        for ev in log:
            if ev[0] == 'read':
                n_reads += 1
                lo_min = ev[2] if lo_min is None else min(lo_min, ev[2])
                hi_max = ev[3] if hi_max is None else max(hi_max, ev[3])
            else:
                for kind, rid in ev[3]:
                    h0, q = info[rid]
                    bound = q if kind == 'q' else check[rid][0]
                    n_snap += 1
                    if lo_min is None or lo_min < h0 or hi_max > bound:
                        viol += 1
    return dict(logs=len(logs), reads=n_reads, snapshots_checked=n_snap, violations=viol)


def compare(cl, st, ref_cl, ref_st):
    fin = np.isfinite(ref_cl) & np.isfinite(cl)
    nan_mismatch = int(np.sum(np.isfinite(ref_cl) != np.isfinite(cl)))
    rel = float(np.max(np.abs(cl[fin] - ref_cl[fin]) / np.maximum(np.abs(ref_cl[fin]), 1e-300))) if fin.any() else 0.0
    return rel, nan_mismatch, bool(np.array_equal(st, ref_st))


def stage_score():
    pre, proto_hash, amend = check_prereg()
    assert amend is not None
    started = time.time()
    B = load_benchmark()
    F, _ = pilot_functions()
    rows, y, groups = B['rows'], B['y'], B['groups']
    check = load_check_pulses()
    with open(WORK / 'identify_main.pkl', 'rb') as f:
        res = pickle.load(f)
    import checks
    import checks_a1
    dis = np.array([r['direction'] == 'discharge' for r in rows])
    primary, group_rows, request_rows, curves, iso, sens, ident_rows, vv_post, inv_rows = [], [], [], [], [], [], [], [], []
    conv, indep, counts, final_central, final_status = {}, {}, {}, {}, {}
    for cand, spec in CANDIDATES.items():
        r = res[cand]
        version = spec['version']
        cfgs = CONFIGS[version]
        c201, _, _ = central_from(r, rows, spec, 201)
        c401, _, _ = central_from(r, rows, spec, 401)
        central, status, picked = central_from(r, rows, spec, STEPS_FINAL)
        final_central[cand], final_status[cand] = central, status
        d201, d401, d801 = decisions(c201, F, B), decisions(c401, F, B), decisions(central, F, B)
        fin = np.isfinite(central)
        conv[cand] = dict(max_abs_diff_201_401_W=float(np.max(np.abs(c201[fin] - c401[fin]))) if fin.any() else 0.0,
                          max_abs_diff_401_801_W=float(np.max(np.abs(c401[fin] - central[fin]))) if fin.any() else 0.0,
                          decisions_equal_201_801=same_decisions(d201, d801),
                          decisions_equal_401_801=same_decisions(d401, d801))
        conv[cand]['passed'] = bool(conv[cand]['max_abs_diff_401_801_W'] <= 0.05 and conv[cand]['decisions_equal_401_801'])
        valid = np.flatnonzero(np.isfinite(central))
        sub = valid[np.linspace(0, len(valid) - 1, 12, dtype=int)]
        ind = []
        for i in sub:
            p = picked[i]
            r0_, R_, k_, _ = params(p, version)
            h_ = float(rows[i]['horizon_s'])
            if version == 'v1':
                ref = checks.independent_power_n(p['E_anchor'], r0_, R_, p['tau'], p['z0'], bool(dis[i]), h_)
            else:
                ref = checks_a1.independent_power_nk(p['E_anchor'], r0_, R_, p['tau'], p['z0'], k_, bool(dis[i]), h_)
            ind.append(dict(row_id=rows[i]['row_id'], candidate_W=float(central[i]), independent_W=ref,
                            abs_diff_W=float(abs(central[i] - ref))))
        indep[cand] = dict(rows=ind, max_abs_diff_W=max(x['abs_diff_W'] for x in ind),
                           passed=bool(max(x['abs_diff_W'] for x in ind) <= 0.05))
        counts[cand] = dict(missing=int((status == 'missing').sum()), fallback=int((status == 'fallback').sum()),
                            ok=int((status == 'ok').sum()), zero_predictions=int(np.sum(central == 0)),
                            missing_reasons=sorted({p.get('reason', '') for p in picked if p['status'] == 'missing'}))
        sel_label = {c: config_label(cfgs[ci]) for c, ci in r['cfg_sel'].items()}
        for i, rr in enumerate(rows):
            p = picked[i]
            has = p['status'] != 'missing'
            if has:
                r0_, R_, k_, e_ = params(p, version)
            ident_rows.append(dict(candidate=cand, row_id=rr['row_id'], cell=rr['evaluation_group'], direction=rr['direction'],
                                   temp_C=rr['temp_C'], horizon_s=rr['horizon_s'], config=sel_label[rr['evaluation_group']],
                                   status=p['status'], tau_s='/'.join(f'{v:.4g}' for v in p['tau']) if has else '',
                                   R0_mOhm=1e3 * r0_ if has else np.nan,
                                   R_mOhm='/'.join(f'{1e3 * v:.4g}' for v in R_) if has else '',
                                   kappa_V_per_Ah=k_ if has else np.nan, E_rls_V=e_ if has else np.nan,
                                   E_anchor_V=p.get('E_anchor', np.nan),
                                   z0_mV='/'.join(f'{1e3 * v:.4g}' for v in p['z0']) if has else '',
                                   V_last_V=p['V_last'], n_updates=p['n_updates'], n_cap=p['n_cap'],
                                   n_valid_members=p['n_valid_members'], central_W=float(central[i])))
        post = []
        for i, rr in enumerate(rows):
            e = pulse_errors(picked[i], version, rr['member'], int(rr['q_row']), int(rr['last_row']))
            ci = r['cfg_sel'][rr['evaluation_group']]
            post.append(dict(row_id=rr['row_id'], cell=rr['evaluation_group'], direction=rr['direction'],
                             temp_C=float(rr['temp_C']), horizon_s=float(rr['horizon_s']), config=ci,
                             config_label=config_label(cfgs[ci]), lam=cfgs[ci][0], kappa_on=cfgs[ci][1],
                             status=picked[i]['status'], end_err_V=e['end_err_V'] if e else np.nan,
                             max_abs_err_V=e['max_abs_err_V'] if e else np.nan))
        write_csv(OUT / f'voltage_rows_posthoc_query_{cand}.csv', post)
        vv_post += summarize_voltage(post, cand, 'posthoc_query_pulse', r['cfg_sel'])
        S = scores_for(central, F, B)
        for (tau, pol), v in S.items():
            curves.append(dict(candidate=cand, policy=pol, tau=tau, **v['score'], scale_A1=v['scales']['A1'],
                               scale_A17=v['scales']['A17'], scale_B=v['scales']['B'], label='curve (descriptive)'))
            if abs(tau - .05) < 1e-12:
                primary.append(dict(candidate=cand, amended=version == 'a1', policy=pol, **v['score'],
                                    missing=counts[cand]['missing'], fallback=counts[cand]['fallback'],
                                    zero_predictions=counts[cand]['zero_predictions'],
                                    scale_A1=v['scales']['A1'], scale_A17=v['scales']['A17'], scale_B=v['scales']['B'],
                                    config_A1=sel_label['A1'], config_A17=sel_label['A17'], config_B=sel_label['B'],
                                    solver_points=STEPS_FINAL))
                for g in v['groups']:
                    group_rows.append(dict(candidate=cand, policy=pol, **g))
                for i, rr in enumerate(rows):
                    request_rows.append(dict(candidate=cand, policy=pol, row_id=rr['row_id'], cell=rr['evaluation_group'],
                                             direction=rr['direction'], temp_C=rr['temp_C'], horizon_s=rr['horizon_s'],
                                             truth=y[i], central=float(central[i]), request=float(v['pred'][i]),
                                             ratio=float(v['pred'][i] / y[i]), exceed=bool(v['pred'][i] / y[i] > 1 + 1e-9),
                                             status=status[i], config=sel_label[rr['evaluation_group']]))
        for pol in ('fixed', 'AD'):
            el = [x for x in curves if x['candidate'] == cand and x['policy'] == pol and x['exceed_rows'] <= 28]
            if not el:
                iso.append(dict(candidate=cand, policy=pol, eligible_points=0, best_util_tau='', best_util='',
                                exceed_rows='', groups_at_best_util='', strict_at_best_util='', maximum_groups='',
                                selected_on_evaluation_data=True, label='post hoc'))
                continue
            best = max(el, key=lambda q: q['util_q50'])
            iso.append(dict(candidate=cand, policy=pol, eligible_points=len(el), best_util_tau=best['tau'],
                            best_util=best['util_q50'], exceed_rows=best['exceed_rows'], groups_at_best_util=best['closed_loop'],
                            strict_at_best_util=best['original'], maximum_groups=max(q['closed_loop'] for q in el),
                            selected_on_evaluation_data=True, label='post hoc'))
        for ci, cfg in enumerate(cfgs):
            cl, st, _ = central_from(r, rows, spec, STEPS_FINAL, cfg_fixed=ci)
            scales, _, _ = F['calibrate'](cl, y, B['orders'], .05)
            for adaptive in (False, True):
                pred = F['request'](cl, y, B['orders'], scales, .05, adaptive)
                sm, _ = F['score'](pred, y, groups)
                sens.append(dict(candidate=cand, config_all_cells=config_label(cfg), policy='AD' if adaptive else 'fixed',
                                 **sm, missing=int((st == 'missing').sum()), fallback=int((st == 'fallback').sum()),
                                 label='post hoc sensitivity, not used for any choice'))
    # ------------------------------------------------------------------------------------- invariance T1-T4
    rng = np.random.default_rng(SEED)
    mult = rng.uniform(0.5, 2.0, len(rows))
    t1 = [dict(r, truth_power_W=str(float(r['truth_power_W']) * mult[i])) for i, r in enumerate(rows)]
    perm = rng.permutation(len(rows))
    lab = ('soc_point', 'ds_query', 'V0', 'I0', 'pulse_index', 'blackout_index', 'context_pulses')
    soh = rng.uniform(0.7, 1.0, len(rows))
    t3 = [dict(r, **{k: rows[perm[i]][k] for k in lab}, soh=str(soh[i])) for i, r in enumerate(rows)]
    with ProcessPoolExecutor(WORKERS) as pool:
        tests = {'T1_truth_scaled': identify(t1, pool, check), 'T3_labels_permuted': identify(t3, pool, check)}
        iso_res = {}
        for cand, spec in CANDIDATES.items():
            tasks = [(spec['version'], spec['n_rc'], r['member'], int(r['h0_row']), [(int(r['q_row']), (('q', r['row_id']),))],
                      int(r['q_row'])) for r in rows]
            iso_res[cand] = list(pool.map(pass_task, tasks, chunksize=4))
    for name, tres in tests.items():
        for cand, spec in CANDIDATES.items():
            cl, st, _ = central_from(tres[cand], rows, spec, STEPS_FINAL)
            rel, nm, same = compare(cl, st, final_central[cand], final_status[cand])
            eq = tres[cand]['cfg_sel'] == res[cand]['cfg_sel']
            inv_rows.append(dict(test=name, candidate=cand, rows=len(rows), max_rel_diff_central=rel, nan_mismatches=nm,
                                 status_equal=same, config_selection_equal=eq,
                                 passed=bool(rel <= 1e-12 and nm == 0 and same and eq)))
    for cand, spec in CANDIDATES.items():
        snap_iso = {}
        for rr, task in zip(rows, iso_res[cand]):
            assert task['n_avail'] == int(rr['q_row'])
            snap_iso[rr['row_id']] = task['out'][int(rr['q_row'])]
        idx = [res[cand]['cfg_sel'][r['evaluation_group']] for r in rows]
        cl, picked = sop_for(snap_iso, rows, spec['version'], spec['n_rc'], idx, STEPS_FINAL)
        st = np.array([p['status'] for p in picked])
        rel, nm, same = compare(cl, st, final_central[cand], final_status[cand])
        th_diff = max(float(np.max(np.abs(a['theta'] - b['theta']))) if 'theta' in a and 'theta' in b else 0.0
                      for rr in rows for a, b in zip(snap_iso[rr['row_id']], res[cand]['snap_q'][rr['row_id']]))
        st_all = all(a['status'] == b['status'] for rr in rows
                     for a, b in zip(snap_iso[rr['row_id']], res[cand]['snap_q'][rr['row_id']]))
        inv_rows.append(dict(test='T2_rows_at_or_after_q_deleted', candidate=cand, rows=len(rows), max_rel_diff_central=rel,
                             nan_mismatches=nm, status_equal=same and st_all, config_selection_equal=True,
                             max_abs_theta_diff_all_configs=th_diff,
                             passed=bool(rel <= 1e-12 and nm == 0 and same and st_all and th_diff <= 1e-12)))
        a_main = audit_logs(res[cand]['logs'], rows, check)
        a_iso = audit_logs([t['log'] for t in iso_res[cand]], rows, check)
        a_t = [audit_logs(tests[n][cand]['logs'], rows, check) for n in tests]
        viol = a_main['violations'] + a_iso['violations'] + sum(x['violations'] for x in a_t)
        inv_rows.append(dict(test='T4_read_log', candidate=cand, rows=len(rows), max_rel_diff_central=0.0, nan_mismatches=0,
                             status_equal=True, config_selection_equal=True, reads_main=a_main['reads'],
                             snapshots_checked_main=a_main['snapshots_checked'],
                             snapshots_checked_isolated=a_iso['snapshots_checked'],
                             snapshots_checked_T1_T3=sum(x['snapshots_checked'] for x in a_t),
                             violations=viol, passed=bool(viol == 0)))
    # ------------------------------------------------------------------------------------------------ outputs
    write_csv(OUT / 'primary.csv', primary)
    write_csv(OUT / 'groups.csv', group_rows)
    write_csv(OUT / 'requests.csv', request_rows)
    write_csv(OUT / 'curves.csv', curves)
    write_csv(OUT / 'iso_budget.csv', iso)
    write_csv(OUT / 'lambda_sensitivity_posthoc.csv', sens)
    write_csv(OUT / 'identification_rows.csv', ident_rows)
    keys = sorted({k for x in inv_rows for k in x}, key=lambda k: (k not in ('test', 'candidate', 'passed'), k))
    write_csv(OUT / 'invariance.csv', [{k: x.get(k, '') for k in keys} for x in inv_rows])
    causal = read_csv(OUT / 'voltage_validation_causal.csv')
    write_csv(OUT / 'voltage_validation.csv', causal + [{k: str(v) for k, v in x.items()} for x in vv_post])
    gates = json.loads((OUT / 'gates.json').read_text(encoding='utf-8'))
    syn = json.loads((OUT / 'checks_synthetic.json').read_text(encoding='utf-8'))
    syn_a1 = json.loads((OUT / 'checks_synthetic_a1.json').read_text(encoding='utf-8'))
    code = {p.name: sha256(p) for p in sorted((EXP / 'src').glob('*.py'))}
    report = dict(
        protocol_sha256=proto_hash, protocol_sha256_at_freeze=pre['sha256']['PROTOCOL.md'], protocol_frozen_utc=pre['frozen_utc'],
        protocol_unchanged=proto_hash == pre['sha256']['PROTOCOL.md'],
        amendment1=dict(frozen_utc=amend['frozen_utc'], sha256=amend['sha256']),
        code_sha256_now=code, code_sha256_at_freeze={k: v for k, v in pre['sha256'].items() if k.startswith('src/')},
        preregistered_predictor_unchanged=code['ecm_rls.py'] == pre['sha256']['src/ecm_rls.py'],
        inputs_sha256={p.name: sha256(p) for p in sorted((EXP / 'inputs').glob('*'))},
        run_pilot_sha256=sha256(EXP / 'inputs/run_pilot_source.py'),
        cache_manifest_sha256=sha256(CACHE / 'manifest.json'),
        alignment=json.loads((OUT / 'alignment.json').read_text(encoding='utf-8')),
        gates=gates, synthetic_and_solver_checks=syn, synthetic_and_solver_checks_a1=syn_a1,
        hyperparameter_selection=json.loads((OUT / 'lambda_selection.json').read_text(encoding='utf-8')),
        prediction_counts=counts, solver_convergence=conv, independent_solver=indep,
        invariance=inv_rows, primary=primary, iso_budget=iso,
        all_checks_passed=bool(all(c['passed'] for c in conv.values()) and all(c['passed'] for c in indep.values())
                               and all(x['passed'] for x in inv_rows) and gates['main_reproduction']['passed']
                               and syn['all_passed'] and syn_a1['all_passed']),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         host=platform.node()),
        boundary='Offline benchmark on logged data. One fixed classical ECM-RLS implementation of the method family '
                 '(pre-registered) and its amended version; not a reproduction of a specific published method and not '
                 'the performance of ECM methods in general.',
        elapsed_score_stage_s=time.time() - started)
    (OUT / 'validation.json').write_text(json.dumps(report, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
                                         encoding='utf-8')
    hashes = {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.suffix in ('.csv', '.json') and p.name not in
              ('output_hashes.json', 'independent_validation.json')}
    (OUT / 'output_hashes.json').write_text(json.dumps(hashes, indent=1), encoding='utf-8')
    print(json.dumps(dict(primary=primary, counts=counts, convergence=conv,
                          independent={k: v['max_abs_diff_W'] for k, v in indep.items()},
                          invariance=[(x['test'], x['candidate'], x['passed']) for x in inv_rows],
                          all_checks_passed=report['all_checks_passed']), indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['gates', 'identify', 'score'], required=True)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {'gates': stage_gates, 'identify': stage_identify, 'score': stage_score}[a.stage]()


if __name__ == '__main__':
    main()
