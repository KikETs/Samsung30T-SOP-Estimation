"""V273 (PROTOCOL_STRICTFULL_V273.md): the full configuration under the strict principle. Information-flow leaks are
removed and the designs chosen on these three cells are reverted to their neutral defaults where one exists:
- predictor N: log median of the five local predictors (V272 rest-voltage regions, main rule) and the neutral-prior
  physics identification i3n (double layer 1.0, no prior law; remote run, strictfull_v273/src_remote) for every row
  (no hybrid rule); B rows' i3n uses the base modes of the A cells allowed in the context (n_all / n_bA17 / n_bA1);
- one scale per evaluated cell (V252 frame), tau = 5 % main (4-6 % reported), chosen on the other cells' rows predicted
  without the evaluated cell: discharge rows by the third-cell-only models, charge rows by the fold-c model in sample.
Descriptive (no verdict): N_light (the five local predictors only), N_hc (hybrid rule and direction classes restored:
a design leak, shown only for the cost of the reversion).
usage: python strictfull_v273.py --gates-only    (local gates 1-4; nothing of N is computed)
       python strictfull_v273.py                 (main run, once; registration, remote hashes and remote gates checked)"""
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
import numpy as np, pandas as pd  # noqa: E402
from run_envelope import HERE, dump  # noqa: E402
import closedloop_v255 as V5, closedloop_v256 as V6, light_v267 as L7  # noqa: E402
import leakfree_v272 as V2  # noqa: E402

OUT = HERE / 'strictfull_v273'; RS = OUT / 'src_remote'; V2OUT = HERE / 'leakfree_v272'
CELLS = V5.CELLS; TAUS = V6.TAUS; NAMES = V2.NAMES; TAU_MAIN = 0.05
PHYS_FOLD = {'B': 'n_all', 'A1': 'n_bA17', 'A17': 'n_bA1'}      # B rows in fold c: base modes without c
PHYS_STRICT = {'A17': 'n_bA17', 'A1': 'n_bA1'}                     # B rows by the model of cell e alone (c = third)
LOG = []


class Stop(Exception):
    pass


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def logmed(X):
    with np.errstate(all='ignore'):
        L = np.log(np.asarray(X, float)); L[~np.isfinite(L)] = np.nan
        return np.nanmedian(L, axis=1), np.isfinite(L).sum(1)


# ---------------------------------------------------------------- physics
def phys_tables():
    """i3n predictions per table: row_id -> power (status ok, finite, > 0) and the coverage"""
    out, cov = {}, {}
    for t in ('n_all', 'n_bA17', 'n_bA1'):
        P = pd.read_csv(RS / t / 'pred.csv')
        ok = (P.status == 'ok') & np.isfinite(P.pred_power_W) & (P.pred_power_W > 0)
        out[t] = P[ok].set_index('row_id').pred_power_W
        cov[t] = dict(rows=int(len(P)), ok=int(ok.sum()), status=P.status.value_counts().to_dict())
    return out, cov


def with_phys(F, cell_of, phys, table_of):
    """add the physics column to a frame of local predictions and the log median of all six (E3causal rule)"""
    F = F.copy()
    F['i3n'] = [phys[table_of(k, cell_of[r])].get(r, np.nan) for k, r in zip(F.key, F.row_id)]
    F['m'], F['n_pred'] = logmed(F[list(NAMES) + ['i3n']].to_numpy(float))
    return F


def full_fold_centres(Cv, cell_of, phys):
    """fold c centre of every row: five local fold-c predictions and i3n in the context of fold c"""
    F = Cv[['fold', 'row_id'] + list(NAMES)].assign(key=Cv.fold)
    return with_phys(F, cell_of, phys, lambda c, cell: PHYS_FOLD[c] if cell == 'B' else 'n_all').drop(columns='key')


def full_strict_centres(Sv, cell_of, phys):
    """centre of each other-cell row by the models of cell e alone: five strict local predictions and i3n"""
    W = Sv.pivot_table(index=['train_cell', 'row_id'], columns='name', values='pred_power_W').reset_index()
    for n in NAMES:
        if n not in W:
            W[n] = np.nan
    F = W[['train_cell', 'row_id'] + list(NAMES)].assign(key=W.train_cell)
    return with_phys(F, cell_of, phys, lambda e, cell: PHYS_STRICT[e] if cell == 'B' else 'n_all').drop(columns='key')


def light_strict_centres(Sv):
    W = Sv.pivot_table(index=['train_cell', 'row_id'], columns='name', values='pred_power_W').reset_index()
    W['m'], W['n_pred'] = logmed(W[list(NAMES)].to_numpy(float))
    return W


def set_centres(B, C, label):
    """B with sop_<c> = exp(fold-c centre) for every row; every row needs a centre in every fold (V267 rule)"""
    B = B.copy(); P = C.pivot(index='row_id', columns='fold', values='m').reindex(B.row_id)
    for c in CELLS:
        B[f'sop_{c}'] = np.exp(P[c].to_numpy())
    miss = B[[f'sop_{c}' for c in CELLS]].isna().any(axis=1)
    if miss.any():
        raise Stop(f'{label}: {int(miss.sum())} rows without a centre in some fold (protocol: stop)')
    return B


# ---------------------------------------------------------------- closed loop, one scale per evaluated cell
def third(c, d):
    return [x for x in CELLS if x not in (c, d)][0]


def look(Bx):
    """(fold c, row_id) -> fold-c centre"""
    return {(c, r): v for c in CELLS for r, v in zip(Bx.row_id, Bx[f'sop_{c}'])}


def dis_from(S):
    """discharge row of cell d, evaluated cell c: the centre of the model fitted on the third cell alone (S: log centre
    by (train cell, row_id))"""
    m = S.to_dict()
    return lambda c, d, r: float(np.exp(m.get((third(c, d), r), np.nan)))


def cal_out_single(B, c):
    K = pd.Series('all', index=B.index); p, t = V5.calibration(B, K, c, 'all', 'out')
    return p, t, 0


def cal_free_single(dis, chg):
    """discharge rows of cell d: dis(c, d, row_id); charge rows: chg(c, d, row_id); rows without a centre dropped"""
    def f(B, c):
        m = B.evaluation_group != c; R = B[m]
        p = np.array([dis(c, d, r) if k == 'discharge' else chg(c, d, r) for d, r, k in zip(R.evaluation_group, R.row_id, R.direction)], float)
        t = R.truth.to_numpy(); ok = np.isfinite(p)
        return p[ok], t[ok], int((~ok).sum())
    return f


def score_single(B, groups_all, tau, cal):
    parts, scales = [], {}
    for c in CELLS:
        ev = B[B.evaluation_group == c].copy()
        p, t, dropped = cal(B, c); s, floor = V5.choose(p, t, tau)
        scales[c] = dict(scale=s, floor_bound=floor, below_v252_grid=bool(s < 0.50 - 1e-9), calibration_rows=int(len(p)),
                         dropped_without_centre=dropped, calibration_int_rate=float(np.mean(s * p > t + 1e-9)))
        ev['cls'] = 'all'; ev['central'] = ev[f'sop_{c}']; ev['pred'] = s * ev.central
        parts.append(ev)
    J = pd.concat(parts); assert len(J) == 672 and J.pred.notna().all()
    summ, G = V5.group_table(J, groups_all)
    return dict(tau=float(tau), groups_pass=int(G.passed.sum()), int_rate=summ['int_rate'], util_q50=summ['util_q50'],
                depth_max=summ['depth_max'], scales=scales,
                failed={f'{x.evaluation_group} {x.direction} {x.temp_C} C {x.horizon_s:g} s': x.reason for x in G[~G.passed].itertuples()}), G, J


def sweep(fn):
    per, keep = [], None
    for tau in TAUS:
        r, G, J = fn(float(tau)); per.append(r)
        if abs(tau - TAU_MAIN) < 1e-9:
            keep = (r, G, J)
    gp = [p['groups_pass'] for p in per]
    return dict(groups_tau5=keep[0]['groups_pass'], groups_tau_4_6=[int(min(gp)), float(np.median(gp)), int(max(gp))],
                int_rate_max=float(max(p['int_rate'] for p in per)), per_tau=per), keep


# ---------------------------------------------------------------- gates
def gate_v272(Cv, Sv):
    """gate 1: V272 rest-voltage local predictions recomputed = the saved V272 outputs"""
    C0 = pd.read_csv(V2OUT / 'centres_vpre_main.csv'); S0 = pd.read_csv(V2OUT / 'strict_vpre_main.csv'); out = {}
    for n in NAMES:
        a = Cv[['fold', 'row_id', n]].rename(columns={n: 'pred_power_W'}); b = C0[['fold', 'row_id', n]].rename(columns={n: 'pred_power_W'})
        out[f'fold_{n}'] = V2.compare(a, b, ['fold', 'row_id'], 1e-12)
        out[f'strict_{n}'] = V2.compare(Sv[Sv.name == n][['name', 'train_cell', 'row_id', 'pred_power_W']],
                                        S0[S0.name == n][['name', 'train_cell', 'row_id', 'pred_power_W']], ['name', 'train_cell', 'row_id'], 1e-12)
    return dict(out, ok=all(v['ok'] for v in out.values()))


def gate_e3(Csoc):
    """gate 2: E3causal centre from the soc_point local five and i3c (x1.8)"""
    I = pd.read_csv(HERE / 'light_v267/src/i3c_pred.csv')[['row_id', 'pred_power_W']].rename(columns={'pred_power_W': 'i3c'})
    F = Csoc.merge(I, on='row_id', how='left'); F['m6'], _ = logmed(F[list(NAMES) + ['i3c']].to_numpy(float))
    R = pd.read_csv(HERE / 'closedloop_v251/src/folds_E3causal.csv')[['fold', 'row_id', 'm']].rename(columns={'m': 'm_ref'})
    X = F.merge(R, on=['fold', 'row_id'], how='outer', indicator=True)
    d = float(np.nanmax(np.abs(X.m6 - X.m_ref))); nan_mis = int((X.m6.isna() != X.m_ref.isna()).sum())
    ok = bool((X._merge == 'both').all() and len(X) == 2016 and d <= 1e-12 and nan_mis == 0)
    return dict(rows=int(len(X)), max_abs_log_diff=d, nan_mismatch=nan_mis, ok=ok), F[['fold', 'row_id', 'm6']].rename(columns={'m6': 'm'})


def gate_tau(res, config):
    T = pd.read_csv(HERE / 'closedloop_v256/table.csv'); T = T[T.config == config]; rows, ok = [], True
    for p in res['per_tau']:
        q = T[np.isclose(T.tau, p['tau'], rtol=0, atol=1e-12)]
        if len(q) != 1:
            rows.append(dict(tau=p['tau'], missing=True)); ok = False; continue
        q = q.iloc[0]; di = abs(p['int_rate'] - q.int_rate); good = bool(p['groups_pass'] == int(q.groups_pass) and di <= 1e-12)
        rows.append(dict(tau=p['tau'], groups=p['groups_pass'], groups_ref=int(q.groups_pass), int_rate_abs_diff=di, ok=good)); ok &= good
    return dict(reference=f'closedloop_v256/table.csv ({config})', groups_tau_4_6=res['groups_tau_4_6'], per_tau=rows, ok=bool(ok and len(rows) == len(TAUS)))


def same_results(a, b):
    return all(x['groups_pass'] == y['groups_pass'] and abs(x['int_rate'] - y['int_rate']) <= 1e-12 and abs(x['util_q50'] - y['util_q50']) <= 1e-12
               for x, y in zip(a['per_tau'], b['per_tau']))


def gate_remote():
    """gate 5: the remote reproduction and table checks, from the copied remote logs"""
    rc = (RS / 'logs/repro_check.txt').read_text().strip(); tc = json.loads((RS / 'out_premodes/tables_check.json').read_text())
    fz = (RS / 'logs/frozen_check.txt').read_text()
    rep_ok = 'max rel diff' in rc and float(rc.split('max rel diff')[1].split(';')[0]) <= 1e-9 and 'status equal True' in rc
    frozen_ok = 'aging_refit v2 manifest OK' in fz and 'pybamm v1 manifest OK' in fz and fz.strip().endswith(': 0')
    return dict(repro=rc, tables=tc, frozen=fz.strip().splitlines(), ok=bool(rep_ok and tc.get('ok') and frozen_ok))


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
    data = sorted(p for p in READS if p.startswith(root) and os.path.isfile(p) and not p.endswith(('.py', '.pyc')) and '/strictfull_v273/v273_prereg' not in p)
    mods = sorted({os.path.abspath(m.__file__) for m in list(sys.modules.values())
                   if isinstance(getattr(m, '__file__', None), str) and os.path.isabs(m.__file__) and m.__file__.startswith(root) and os.path.isfile(m.__file__)})
    return data, mods


# ---------------------------------------------------------------- main
def run(gates_only, res):
    if not gates_only:
        reg = check_hashes(OUT / 'v273_prereg_stage1.sha256', HERE) | check_hashes(OUT / 'v273_prereg.sha256', HERE)
        rem = check_hashes(RS / 'REMOTE.sha256', RS)
        res['registration'] = dict(registered_files=len(reg), remote_files=len(rem), all_match=True); say('registration and remote hashes:', res['registration'])
    B = V5.load(); groups_all = B[V5.C1.GK].drop_duplicates().sort_values(V5.C1.GK).reset_index(drop=True); assert len(groups_all) == 15
    D = V2.ls.table(); assert len(D) == 672 and set(D.row_id) == set(B.row_id)
    cell_of = D.set_index('row_id').evaluation_group.to_dict()
    G = res['gates'] = {}
    # gate 1: V272 rest-voltage local predictions
    TH = V2.boundaries(D, 'min_error')
    Fv = {n: V2.loco(D, n, TH) for n in NAMES}; Cv = V2.centres(Fv)
    Sv = pd.concat([V2.strict(D, n, TH) for n in NAMES], ignore_index=True)
    G['1_v272_vpre_local'] = gate_v272(Cv, Sv); say('gate 1 V272 rest-voltage local predictions:', 'ok' if G['1_v272_vpre_local']['ok'] else 'FAILED')
    # gate 2: E3causal centre from its components
    Csoc = V2.centres({n: V2.loco(D, n) for n in NAMES})
    G['2_e3causal'], E3 = gate_e3(Csoc); say('gate 2 E3causal centre:', G['2_e3causal'])
    # gate 3: V256 sop single out with the recomputed E3causal centre
    Be = set_centres(B, E3, 'E3causal')
    ref, _ = sweep(lambda tau: score_single(Be, groups_all, tau, cal_out_single))
    G['3_v256_sop_single_out'] = gate_tau(ref, '2_sop_single_out'); say('gate 3 V256 2_sop_single_out:', G['3_v256_sop_single_out']['groups_tau_4_6'], 'ok' if G['3_v256_sop_single_out']['ok'] else 'FAILED')
    # gate 4: the evaluated-cell-free calibration path fed with the out-of-sample centres
    le = look(Be); own = lambda c, d, r: le[(d, r)]
    pl, _ = sweep(lambda tau: score_single(Be, groups_all, tau, cal_free_single(own, own)))
    G['4_plumbing'] = dict(ok=bool(same_results(pl, ref))); say('gate 4 plumbing:', G['4_plumbing'])
    if not gates_only:
        G['5_remote'] = gate_remote(); say('gate 5 remote:', G['5_remote']['repro'], '| tables ok', G['5_remote']['tables'].get('ok'), '| frozen', G['5_remote']['frozen'])
    res['gates_ok'] = bool(all(g['ok'] for g in G.values()))
    if not res['gates_ok']:
        raise Stop('a gate failed; N not computed (protocol)')
    if gates_only:
        data, mods = project_inputs()
        (OUT / 'inputs_read.txt').write_text('# data files read by the gates run\n' + ''.join(os.path.relpath(p, HERE) + '\n' for p in data)
                                             + '# project modules imported\n' + ''.join(os.path.relpath(p, HERE) + '\n' for p in mods))
        say(f'local gates passed; {len(data)} data files and {len(mods)} project modules listed in strictfull_v273/inputs_read.txt')
        return
    # -------- main computation (everything computed before anything is written or printed)
    phys, cov = phys_tables()
    Cf = full_fold_centres(Cv, cell_of, phys); Sf = full_strict_centres(Sv, cell_of, phys); Sl = light_strict_centres(Sv)
    Bn = set_centres(B, Cf, 'N'); Bl = set_centres(B, Cv, 'N_light')
    sf = Sf.set_index(['train_cell', 'row_id']).m; sl = Sl.set_index(['train_cell', 'row_id']).m
    fn, fl = look(Bn), look(Bl)
    chg = lambda Fx: (lambda c, d, r: Fx[(c, r)])
    CFG, KEEP = {}, {}
    CFG['N'], KEEP['N'] = sweep(lambda tau: score_single(Bn, groups_all, tau, cal_free_single(dis_from(sf), chg(fn))))
    CFG['N_light'], KEEP['N_light'] = sweep(lambda tau: score_single(Bl, groups_all, tau, cal_free_single(dis_from(sl), chg(fl))))
    # N_hc: hybrid rule and direction classes restored (design leak; V272 P structure with the N centres)
    M3s = pd.read_csv(V2OUT / 'm3_short.csv'); M3a = pd.read_csv(V2OUT / 'm3_calibration_A.csv')
    se = pd.Series(np.nan, index=B.index); sid = B.index[~B.in_events]
    se[sid] = M3s.set_index('row_id').central_W.reindex(B.row_id[sid]).to_numpy()
    Bh = Bn.copy(); Bh.loc[:, 'in_events'] = B.in_events
    Sc = np.exp(sf)
    CFG['N_hc'], KEEP['N_hc'] = sweep(lambda tau: V2.score(Bh, groups_all, tau, se, V2.cal_strict(Sc, 'in', (M3a.central_foldB_W.to_numpy(), M3a.truth.to_numpy())))[:3])
    stats = dict(N=L7.central_stats(Bn.assign(in_events=True), 'sop_{}'), N_light=L7.central_stats(Bl.assign(in_events=True), 'sop_{}'))
    # -------- records
    rows = []
    for name, r in CFG.items():
        for p in r['per_tau']:
            sc = p['scales']; flat = {f'scale_{k}': v['scale'] for k, v in sc.items()}
            flat.update({f'cal_rows_{k}': v['calibration_rows'] for k, v in sc.items()}); flat.update({f'dropped_{k}': v['dropped_without_centre'] for k, v in sc.items()})
            rows.append(dict(config=name, tau=p['tau'], groups_pass=p['groups_pass'], int_rate=p['int_rate'], util_q50=p['util_q50'], depth_max=p['depth_max'],
                             **flat, failed='; '.join(f'{g} ({w})' for g, w in p['failed'].items())))
    pd.DataFrame(rows).to_csv(OUT / 'tau_range.csv', index=False)
    for name, (r, Gt, Jt) in KEEP.items():
        Gt.to_csv(OUT / f'groups_tau5_{name}.csv', index=False); Jt.to_csv(OUT / f'rows_tau5_{name}.csv', index=False)
    Cf.to_csv(OUT / 'centres_full.csv', index=False); Sf.to_csv(OUT / 'strict_full.csv', index=False)
    n5 = CFG['N']['groups_tau5']
    res['configs'] = {k: dict(groups_tau5=v['groups_tau5'], groups_tau_4_6=v['groups_tau_4_6'], int_rate_max=v['int_rate_max'], per_tau=v['per_tau']) for k, v in CFG.items()}
    res['verdict_N'] = ('엄격한 원칙에서도 12그룹 이상' if n5 >= 12 else ('정보 흐름 누설 제거판(V272 P) 수준' if n5 >= 9
                        else '되돌린 설계와 누설 경로가 지금 결과의 상당 부분을 만들었다'))
    res['name'] = '정보 흐름 누설을 없애고 중립 기본값으로 되돌린 판(설계 누설 남음)'
    res['physics_coverage'] = cov
    res['rows_physics_used'] = dict(fold=int(Cf.i3n.notna().sum()), strict=int(Sf.i3n.notna().sum()))
    res['central_open_loop'] = stats
    for name in CFG:
        x = KEEP[name][0]; sc = ' '.join('%s=%.2f' % (k, v['scale']) for k, v in x['scales'].items())
        say(f"{name:7s} tau 5 %: groups {x['groups_pass']}  int {100 * x['int_rate']:.2f} %  util {100 * x['util_q50']:.1f} %  scales {sc}  failed {list(x['failed'])}")
        say(f"        tau 4-6 %: {CFG[name]['groups_tau_4_6']}  int max {100 * CFG[name]['int_rate_max']:.2f} %")
    say('verdict N (tau 5 %):', res['verdict_N'], '| name:', res['name'])
    say('physics coverage:', json.dumps(cov, ensure_ascii=False))
    data, mods = project_inputs()
    rel = {os.path.relpath(p, HERE) for p in data + mods}
    res['unregistered_reads'] = sorted(x for x in rel - reg if not x.startswith('strictfull_v273/src_remote/'))
    say('files read but not registered:', res['unregistered_reads'] or 'none')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gates-only', action='store_true'); a = ap.parse_args()
    OUT.mkdir(exist_ok=True); stem = 'gates_only' if a.gates_only else 'run'
    res = dict(protocol='PROTOCOL_STRICTFULL_V273.md', mode=stem)
    try:
        run(a.gates_only, res)
    except Stop as e:
        res['stopped'] = str(e); say('STOPPED:', e)
        raise SystemExit(1)
    finally:
        dump(OUT / ('gates_only.json' if a.gates_only else 'summary.json'), V2.clean(res))
        (OUT / f'{stem}.log').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
