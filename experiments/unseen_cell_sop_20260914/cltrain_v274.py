"""V274 (PROTOCOL_CLTRAIN_V274.md): V273's strict-principle full configuration N with the training cells' current-limited
SOP rows (cltrain_v274/features_cl.csv, cltrain_v274_features.py) added to the training of the three log-resistance local
predictors (prev, L4, M6): each row is one (Rq, Iq) observation at the current cap, admitted to the fit although
Iq >= 0.98 Imax (the only change to local_sop.fit / local_sop2.fit_region). DN6 and G1 (direct fits; the target of a
current-limited row is censored), the physics i3n (V273 remote output) and the closed loop (one scale per evaluated
cell, evaluated-cell-free calibration, tau 5 % main) are V273's.
usage: python cltrain_v274.py --gates-only    (gates; nothing with the current-limited rows is scored)
       python cltrain_v274.py                 (main run, once; registration checked)"""
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
import json  # noqa: E402
from contextlib import contextmanager  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import strictfull_v273 as S3  # noqa: E402

HERE, dump = S3.HERE, S3.dump
V2, V5, L7 = S3.V2, S3.V5, S3.L7
ls, ls2 = V2.ls, V2.ls2
OUT = HERE / 'cltrain_v274'; V2S = HERE / 'leakfree_v272/src'
CELLS, NAMES, LOGR = V5.CELLS, V2.NAMES, ('prev', 'L4', 'M6')
LOG = []


class Stop(Exception):
    pass


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


# ---------------------------------------------------------------- fits with the current-limited rows admitted
def fit_cl(T, kind, fit_beta):
    """local_sop.fit with the rows flagged cl admitted although Iq >= 0.98 Imax (otherwise unchanged)"""
    B, Iref = ls.basis(T, kind)
    y = np.log(T.Rq / B)
    x = np.log(T.Iq / Iref)
    ok = np.isfinite(y) & np.isfinite(x) & ((T.Iq < 0.98 * T.Imax) | T.cl)
    out = {}
    for r, m in T[ok].groupby("region").groups.items():
        yy, xx = y[m], x[m]
        if fit_beta and len(m) >= 8 and xx.std() > 0.05:
            beta, lk = np.polyfit(xx, yy, 1)
        else:
            beta, lk = 0.0, float(np.median(yy))
        out[r] = (float(np.exp(lk)), float(beta))
    return out


def fit_region_cl(T, X, kind, feats):
    """local_sop2.fit_region with the rows flagged cl admitted although Iq >= 0.98 Imax (otherwise unchanged)"""
    B, Iref = ls2.bas(T, kind)
    y = np.log(T.Rq / B)
    xi = np.log(T.Iq / Iref)
    A = np.column_stack([np.ones(len(T))] + [X[f].to_numpy() for f in feats] + [xi.to_numpy()])
    ok = np.isfinite(y.to_numpy()) & np.isfinite(A).all(axis=1) & ((T.Iq < 0.98 * T.Imax) | T.cl).to_numpy()
    if ok.sum() < max(12, 4 * A.shape[1]):
        return dict(c0=float(np.nanmedian(y[ok])) if ok.any() else 0.0, c={}, beta=0.0, n=int(ok.sum()), fallback=True)
    coef, *_ = np.linalg.lstsq(A[ok], y.to_numpy()[ok], rcond=None)
    if not -0.9 <= coef[-1] <= 1.0:                     # I * R_hat(I) must rise with I: refit with beta clipped
        coef[-1] = float(np.clip(coef[-1], -0.9, 1.0))
        c2, *_ = np.linalg.lstsq(A[ok][:, :-1], y.to_numpy()[ok] - coef[-1] * A[ok][:, -1], rcond=None)
        coef[:-1] = c2
    res = y.to_numpy()[ok] - A[ok] @ coef
    return dict(c0=float(coef[0]), c=dict(zip(feats, map(float, coef[1:-1]))), beta=float(coef[-1]), n=int(ok.sum()),
                sd=float(res.std()), fallback=False)


@contextmanager
def patched():
    f0, r0 = ls.fit, ls2.fit_region
    ls.fit, ls2.fit_region = fit_cl, fit_region_cl
    try:
        yield
    finally:
        ls.fit, ls2.fit_region = f0, r0


# ---------------------------------------------------------------- tables
def cl_table(D0):
    """the current-limited rows with a target, in the layout of local_sop.table (training rows only): the derived
    columns as local_sop.table except Iq (the measured current at the R_h sample) and gap (not defined)"""
    X = pd.read_csv(OUT / 'features_cl.csv')
    X['R1_q'], X['R2_q'], X['R10_q'] = X.R1, X.R2, X.R10
    X['T_pre'] = X.T_pre.fillna(X.T_pre_raw)
    X = X.merge(pd.read_csv(V2S / 'sop_ident/out/feats/cv.csv'), on='sweep_id', how='left')
    D = pd.DataFrame({c: (X[c] if c in X else np.nan) for c in D0.columns})
    D['chg'] = D.direction == 'charge'
    D['Vlim'] = np.where(D.chg, 4.15, 2.55)
    D['Imax'] = np.where(D.chg, 15.0, 30.0)
    h10 = D.horizon_s >= 5
    D['Roff'] = np.where(h10, D.Roff10, D.Roff2)
    D['Rstep'] = np.where(h10, D.Rstep10, D.Rstep2)
    D['prevRstep'] = np.where(h10, D.prev_Rstep10, D.prev_Rstep2)
    D['prevR'] = np.where(h10, D.prev_R10, D.prev_R2)
    D['prevRoff'] = np.where(h10, D.prev_Roff10, D.prev_Roff2)
    D['Rq'] = np.where(h10, D.R10_q, D.R2_q)
    D['Iq'] = np.abs(np.where(h10, X.I_R10, X.I_R2))
    drift = 1e-3 * D.slope_mVs.fillna(0.0) * D.horizon_s
    D['H'] = np.where(D.chg, D.Vlim - D.V_pre - drift, D.V_pre + drift - D.Vlim)
    D['gap'] = np.nan
    D['region'] = None
    D['cl'] = True
    keep = np.isfinite(D.Rq) & np.isfinite(D.Iq) & (D.Iq > 0) & D.V_pre.notna()
    return D[keep].reset_index(drop=True)


def local_preds(Db, De, TH):
    """V_pre fold and strict predictions on the benchmark rows: prev, L4, M6 trained on De (benchmark + current-limited
    rows, admitted by the patched fits), DN6 and G1 on Db; the region boundaries TH come from the benchmark rows"""
    bench = set(Db.row_id)
    with patched():
        Fv = {n: V2.loco(De if n in LOGR else Db, n, TH) for n in NAMES}
        Sv = pd.concat([V2.strict(De if n in LOGR else Db, n, TH) for n in NAMES], ignore_index=True)
    Fv = {n: F[F.row_id.isin(bench)].reset_index(drop=True) for n, F in Fv.items()}
    return Fv, Sv[Sv.row_id.isin(bench)].reset_index(drop=True)


def configs(B, groups_all, cell_of, phys, Fv, Sv):
    Cv = V2.centres(Fv)
    Cf = S3.full_fold_centres(Cv, cell_of, phys); Sf = S3.full_strict_centres(Sv, cell_of, phys); Sl = S3.light_strict_centres(Sv)
    Bn = S3.set_centres(B, Cf, 'N274'); Bl = S3.set_centres(B, Cv, 'N274_light')
    sf = Sf.set_index(['train_cell', 'row_id']).m; sl = Sl.set_index(['train_cell', 'row_id']).m
    fn, fl = S3.look(Bn), S3.look(Bl)
    chg = lambda Fx: (lambda c, d, r: Fx[(c, r)])
    CFG, KEEP = {}, {}
    CFG['N274'], KEEP['N274'] = S3.sweep(lambda tau: S3.score_single(Bn, groups_all, tau, S3.cal_free_single(S3.dis_from(sf), chg(fn))))
    CFG['N274_light'], KEEP['N274_light'] = S3.sweep(lambda tau: S3.score_single(Bl, groups_all, tau, S3.cal_free_single(S3.dis_from(sl), chg(fl))))
    return CFG, KEEP, Cv, Cf, Sf, Bn, Bl


# ---------------------------------------------------------------- main
def run(gates_only, res):
    if not gates_only:
        reg = S3.check_hashes(OUT / 'v274_prereg.sha256', HERE)
        res['registration'] = dict(registered_files=len(reg), all_match=True); say('registration:', res['registration'])
    G = res['gates'] = {}
    fg = json.loads((OUT / 'features_gates.json').read_text())
    G['1_features'] = dict(ok=bool(fg['ok']), relax_rows=fg['relax_rows'], events_rows=fg['events_rows'],
                           relax_max_rel=max(v['max_rel'] for v in fg['relax'].values()), events_max_rel=max(v['max_rel'] for v in fg['events'].values()))
    say('gate 1 features:', G['1_features'])
    S3.check_hashes(S3.RS / 'REMOTE.sha256', S3.RS); S3.check_hashes(S3.OUT / 'v273_prereg.sha256', HERE)
    G['3_v273_inputs'] = dict(ok=True); say('gate 3 V273 remote output and registered inputs unchanged: ok')
    B = V5.load(); groups_all = B[V5.C1.GK].drop_duplicates().sort_values(V5.C1.GK).reset_index(drop=True)
    Db = V2.ls.table(); assert len(Db) == 672 and set(Db.row_id) == set(B.row_id)
    Db = Db.assign(cl=False); cell_of = Db.set_index('row_id').evaluation_group.to_dict()
    TH = V2.boundaries(Db.drop(columns='cl'), 'min_error')
    phys, cov = S3.phys_tables()
    # gate 2: without the current-limited rows the patched pipeline reproduces V272 and V273 N
    Fv0, Sv0 = local_preds(Db, Db, TH)
    g272 = S3.gate_v272(V2.centres(Fv0), Sv0)
    CFG0, _, _, _, _, _, _ = configs(B, groups_all, cell_of, phys, Fv0, Sv0)
    T3 = pd.read_csv(S3.OUT / 'tau_range.csv'); T3 = T3[T3.config == 'N']; rows, ok3 = [], True
    for p in CFG0['N274']['per_tau']:
        q = T3[np.isclose(T3.tau, p['tau'], rtol=0, atol=1e-12)].iloc[0]
        good = bool(p['groups_pass'] == int(q.groups_pass) and abs(p['int_rate'] - q.int_rate) <= 1e-12); ok3 &= good
        rows.append(dict(tau=p['tau'], groups=p['groups_pass'], groups_v273=int(q.groups_pass), ok=good))
    G['2_without_cl'] = dict(v272_local=g272['ok'], v273_N=rows, ok=bool(g272['ok'] and ok3 and len(rows) == 9))
    say('gate 2 without current-limited rows: V272 local', g272['ok'], '| V273 N per tau', [r['groups'] for r in rows], 'ok' if G['2_without_cl']['ok'] else 'FAILED')
    res['gates_ok'] = bool(all(g['ok'] for g in G.values()))
    if not res['gates_ok']:
        raise Stop('a gate failed; nothing with the current-limited rows computed (protocol)')
    Dc = cl_table(Db); res['cl_training_rows'] = {f'{g}|{d}': int(n) for (g, d), n in Dc.groupby(['evaluation_group', 'direction']).size().items()}
    say('current-limited training rows:', res['cl_training_rows'])
    if gates_only:
        data, mods = S3.project_inputs()
        data = sorted(set(data) | {p for p in READS if p.startswith(str(HERE.parents[1]) + os.sep) and os.path.isfile(p) and not p.endswith(('.py', '.pyc'))
                                   and '/cltrain_v274/v274_prereg' not in p})
        (OUT / 'inputs_read.txt').write_text('# data files read by the gates run\n' + ''.join(os.path.relpath(p, HERE) + '\n' for p in data)
                                             + '# project modules imported\n' + ''.join(os.path.relpath(p, HERE) + '\n' for p in mods))
        say(f'gates passed; {len(data)} data files and {len(mods)} modules listed in cltrain_v274/inputs_read.txt')
        return
    # -------- main computation (everything computed before anything is written or printed)
    De = pd.concat([Db, Dc], ignore_index=True)
    Fv1, Sv1 = local_preds(Db, De, TH)
    CFG, KEEP, Cv, Cf, Sf, Bn, Bl = configs(B, groups_all, cell_of, phys, Fv1, Sv1)
    rows = []
    for name, r in CFG.items():
        for p in r['per_tau']:
            sc = p['scales']
            rows.append(dict(config=name, tau=p['tau'], groups_pass=p['groups_pass'], int_rate=p['int_rate'], util_q50=p['util_q50'], depth_max=p['depth_max'],
                             **{f'scale_{k}': v['scale'] for k, v in sc.items()}, **{f'cal_rows_{k}': v['calibration_rows'] for k, v in sc.items()},
                             **{f'dropped_{k}': v['dropped_without_centre'] for k, v in sc.items()}, failed='; '.join(f'{g} ({w})' for g, w in p['failed'].items())))
    pd.DataFrame(rows).to_csv(OUT / 'tau_range.csv', index=False)
    for name, (r, Gt, Jt) in KEEP.items():
        Gt.to_csv(OUT / f'groups_tau5_{name}.csv', index=False); Jt.to_csv(OUT / f'rows_tau5_{name}.csv', index=False)
    Cf.to_csv(OUT / 'centres_full.csv', index=False); Sf.to_csv(OUT / 'strict_full.csv', index=False)
    n5 = CFG['N274']['groups_tau5']
    res['configs'] = {k: dict(groups_tau5=v['groups_tau5'], groups_tau_4_6=v['groups_tau_4_6'], int_rate_max=v['int_rate_max'], per_tau=v['per_tau']) for k, v in CFG.items()}
    res['verdict'] = '전류 제한 행 학습으로 개선' if n5 >= 13 else ('변화 없음' if n5 == 12 else '나빠짐')
    res['name'] = '정보 흐름 누설을 없애고 중립 기본값으로 되돌린 판 + 전류 제한 행 학습(설계 누설 남음)'
    res['central_open_loop'] = dict(N274=L7.central_stats(Bn.assign(in_events=True), 'sop_{}'), N274_light=L7.central_stats(Bl.assign(in_events=True), 'sop_{}'))
    for name in CFG:
        x = KEEP[name][0]; sc = ' '.join('%s=%.2f' % (k, v['scale']) for k, v in x['scales'].items())
        say(f"{name:10s} tau 5 %: groups {x['groups_pass']}  int {100 * x['int_rate']:.2f} %  util {100 * x['util_q50']:.1f} %  scales {sc}  failed {list(x['failed'])}")
        say(f"           tau 4-6 %: {CFG[name]['groups_tau_4_6']}  int max {100 * CFG[name]['int_rate_max']:.2f} %")
    say('verdict (tau 5 %, V273 N = 12):', res['verdict'], '| name:', res['name'])
    data, mods = S3.project_inputs()
    rel = {os.path.relpath(p, HERE) for p in data + mods} | {os.path.relpath(p, HERE) for p in READS
                                                             if p.startswith(str(HERE.parents[1]) + os.sep) and os.path.isfile(p) and not p.endswith('.pyc')}
    res['unregistered_reads'] = sorted(x for x in rel - reg if not x.startswith(('strictfull_v273/src_remote/', 'cltrain_v274/v274_prereg')))
    say('files read but not registered:', res['unregistered_reads'] or 'none')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gates-only', action='store_true'); a = ap.parse_args()
    OUT.mkdir(exist_ok=True); stem = 'gates_only' if a.gates_only else 'run'
    res = dict(protocol='PROTOCOL_CLTRAIN_V274.md', mode=stem)
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
