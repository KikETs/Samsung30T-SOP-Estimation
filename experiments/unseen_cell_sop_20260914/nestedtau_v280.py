"""V280 (PROTOCOL_NESTEDTAU_V280.md): the intervention target tau chosen per evaluated cell c from c's calibration rows
alone (the V278 B0 rows: the other two cells, discharge by the third-cell-only model, charge by the fold-c model in
sample), by leaving one calibration cell out inside them: scale on cell k's rows, closed-loop groups of cell v's rows,
inner score = passed groups of both directions; the highest score wins, ties go to the tau nearest 5 %, then the smaller.
The request of c is the B0 request at tau_c. A sensitivity analysis: the main configuration (tau 5 % fixed) is unchanged.
  P  V278 B0 with the vT1h i3n tables (main);  L  V278 B0, five local predictors;  H  V273 N (vT8s tables, historical).
usage: python nestedtau_v280.py --gates-only   (N1-N3; no selected tau, inner score or comparison printed)
       python nestedtau_v280.py                (main run, once)"""
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
import numpy as np, pandas as pd  # noqa: E402
from run_envelope import HERE, dump  # noqa: E402
import closedloop_v251 as C1, closedloop_v255 as V5, closedloop_v256 as V6  # noqa: E402
import leakfree_v272 as V2  # noqa: E402
import strictfull_v273 as S  # noqa: E402
import invariance_v276 as I6  # noqa: E402
import compare_v278 as CV  # noqa: E402

OUT = HERE / 'nestedtau_v280'
CV.TMP = OUT / 'tmp'                                                     # model-input copies of this experiment
CELLS = V5.CELLS; TAU_A = 0.05; SEED = 280; TOL = 1e-12
GRIDS = {'main': [float(t) for t in V6.TAUS],                            # V256: 4.00-6.00 %, 9 values
         'wide': [float(t) for t in np.round(np.arange(0.01, 0.10001, 0.0025), 4)]}    # 1.00-10.00 %, 37 values
PHYS = {'P': HERE / 'compare_v278/phys_vT1h', 'L': None, 'H': S.RS}
REF = {'P': HERE / 'compare_v278/P/requests_B0.csv', 'L': HERE / 'compare_v278/L/requests_B0.csv', 'H': S.OUT / 'rows_tau5_N.csv'}
LOG = []


class Stop(Exception):
    pass


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def baseline(kind, src, Bm, name=None):
    return CV.build(name or kind, 'L' if kind == 'L' else 'P', src, Bm, PHYS[kind])


# ---------------------------------------------------------------- selection
def cal_rows(X, c):
    """B0 calibration rows of evaluated cell c (strictfull_v273.cal_free_single with the charge rule of V278 arm_B0),
    with their cell and group keys; rows without a centre dropped"""
    Bc = X['Bc']; dis = S.dis_from(X['sm']); fn = S.look(Bc)
    R = Bc[Bc.evaluation_group != c][['row_id'] + C1.GK + ['truth']].copy()
    R['p'] = np.array([dis(c, d, r) if k == 'discharge' else fn[(c, r)] for d, r, k in zip(R.evaluation_group, R.row_id, R.direction)], float)
    return R[np.isfinite(R.p.to_numpy())]


def inner(R, tau):
    """leave one calibration cell out inside R: scale on cell k's rows (B0 rule), closed-loop groups of cell v's rows"""
    out, cells = [], sorted(R.evaluation_group.unique())
    for k in cells:
        for v in cells:
            if v == k:
                continue
            K, W = R[R.evaluation_group == k], R[R.evaluation_group == v]
            s, floor = V5.choose(K.p.to_numpy(), K.truth.to_numpy(), tau)
            _, G = C1.closed_loop(W, s * W.p.to_numpy())
            ok = ((G.util_q50 >= V5.GROUP_UTIL) & (G.int_rate <= V5.GROUP_INT)).to_numpy()
            out.append(dict(cal=k, val=v, cal_rows=int(len(K)), val_rows=int(len(W)), scale=s, floor_bound=floor, groups=int(len(G)),
                            passed=int(ok.sum()), failed='; '.join(f'{x.direction} {x.temp_C:g} C {x.horizon_s:g} s' for x, o in zip(G.itertuples(), ok) if not o)))
    return out


def select(R, grid):
    """tau with the highest inner score; ties: nearest to 5 %, then the smaller"""
    T = pd.DataFrame([dict(tau=tau, **x) for tau in grid for x in inner(R, tau)])
    sc = T.groupby('tau').passed.sum(); best = int(sc.max())
    tied = [float(t) for t in sc.index if sc[t] == best]
    tau = min(tied, key=lambda t: (round(abs(t - TAU_A), 6), t))
    return tau, T, dict(score=best, tied=tied, score_by_tau={f'{100 * t:.2f}': int(v) for t, v in sc.items()})


def outer(X, taus):
    """B0 request of every evaluated cell c at its tau: s_c(tau_c) from all of R_c x the fold-c centre"""
    Bc = X['Bc']; pred = pd.Series(np.nan, index=Bc.row_id.to_numpy()); sc = {}
    for c in CELLS:
        R = cal_rows(X, c); s, floor = V5.choose(R.p.to_numpy(), R.truth.to_numpy(), taus[c])
        ev = Bc[Bc.evaluation_group == c]
        pred.loc[ev.row_id.to_numpy()] = s * ev[f'sop_{c}'].to_numpy()
        sc[c] = dict(tau=taus[c], scale=s, floor_bound=floor, calibration_rows=int(len(R)))
    assert pred.notna().all()
    return pred, sc


# ---------------------------------------------------------------- gates
def gates(src, B0t, groups_all):
    G, X = {}, {}
    for kind in PHYS:
        X[kind] = Xk = baseline(kind, src, B0t)
        pred, _ = outer(Xk, {c: TAU_A for c in CELLS})
        ref = pd.read_csv(REF[kind], float_precision='round_trip').set_index('row_id').pred
        r = float(np.nanmax(I6.rel(pred.reindex(ref.index), ref)))
        G[f'N1_{kind}'] = dict(max_rel=r, rows=int(len(ref)), ok=bool(r <= TOL and len(ref) == 672 and ref.index.isin(pred.index).all()))
        fn = S.look(Xk['Bc']); cal = S.cal_free_single(S.dis_from(Xk['sm']), lambda c, d, rr: fn[(c, rr)])
        rows_equal, bad = True, []
        for c in CELLS:
            p, t, _ = cal(Xk['Bc'], c); R = cal_rows(Xk, c)
            rows_equal &= bool(np.array_equal(p, R.p.to_numpy()) and np.array_equal(t, R.truth.to_numpy()))
        for tau in GRIDS['main']:
            _, b0s = CV.arm_B0(Xk, tau, groups_all)
            for c in CELLS:
                R = cal_rows(Xk, c); s, _ = V5.choose(R.p.to_numpy(), R.truth.to_numpy(), tau)
                if s != b0s[c]:
                    bad.append(f'{tau}|{c}')
        G[f'N2_{kind}'] = dict(rows_equal=rows_equal, scale_mismatch=bad, ok=bool(rows_equal and not bad))
    fac = I6.factors(src[0], np.random.default_rng(SEED))
    for kind in PHYS:
        q = {}
        for c in CELLS:
            s1, B1, _ = I6.t1(src, B0t, c, fac)
            X1 = baseline(kind, s1, B1, f'{kind}_N3_{c}')
            ids = B0t.row_id[B0t.evaluation_group == c].to_numpy()
            for g, grid in GRIDS.items():
                t0, _, _ = select(cal_rows(X[kind], c), grid); t1, _, _ = select(cal_rows(X1, c), grid)
                p0, a0 = outer(X[kind], {x: t0 for x in CELLS}); p1, a1 = outer(X1, {x: t1 for x in CELLS})
                q[f'{c}|{g}'] = dict(tau_equal=bool(t0 == t1), scale_equal=bool(a0[c]['scale'] == a1[c]['scale']),
                                     max_rel=float(np.nanmax(I6.rel(p0.reindex(ids), p1.reindex(ids)))))
        G[f'N3_{kind}'] = dict(cells=q, ok=bool(all(v['tau_equal'] and v['scale_equal'] and v['max_rel'] <= TOL for v in q.values())))
    for k, v in G.items():
        say(k, 'ok' if v['ok'] else 'FAILED', {a: b for a, b in v.items() if a not in ('ok', 'cells')} or '',
            '' if 'cells' not in v else f"max rel {max(x['max_rel'] for x in v['cells'].values()):.3g}, tau and scale equal: "
            f"{all(x['tau_equal'] and x['scale_equal'] for x in v['cells'].values())}")
    return G, X


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
    data = sorted(p for p in READS if p.startswith(root) and os.path.isfile(p) and not p.endswith(('.py', '.pyc'))
                  and '/nestedtau_v280/tmp/' not in p and '/nestedtau_v280/v280_prereg' not in p)
    mods = sorted({os.path.abspath(m.__file__) for m in list(sys.modules.values())
                   if isinstance(getattr(m, '__file__', None), str) and os.path.isabs(m.__file__) and m.__file__.startswith(root) and os.path.isfile(m.__file__)})
    return data, mods


# ---------------------------------------------------------------- main run
def changed(Ga, Gb):
    a = Ga.set_index(C1.GK).passed; b = Gb.set_index(C1.GK).passed.reindex(a.index)
    return [f'{i[0]} {i[1]} {i[2]} C {i[3]:g} s: {"pass" if x else "fail"} -> {"pass" if y else "fail"}' for i, x, y in zip(a.index, a, b) if bool(x) != bool(y)]


def run(gates_only, res):
    reg = set()
    if not gates_only:
        reg = check_hashes(OUT / 'v280_prereg_stage1.sha256', HERE) | check_hashes(OUT / 'v280_prereg.sha256', HERE)
        res['registration'] = dict(registered_files=len(reg), all_match=True); say('registration:', res['registration'])
    B0t = V5.load(); groups_all = B0t[C1.GK].drop_duplicates().sort_values(C1.GK).reset_index(drop=True)
    src = I6.sources()
    res['gates'], X = gates(src, B0t, groups_all)
    res['gates_ok'] = bool(all(g['ok'] for g in res['gates'].values())); say('gates', 'passed' if res['gates_ok'] else 'FAILED')
    if gates_only:
        data, mods = project_inputs()
        (OUT / 'inputs_read.txt').write_text('# data files read by the gates run\n' + ''.join(os.path.relpath(p, HERE) + '\n' for p in data)
                                             + '# project modules imported\n' + ''.join(os.path.relpath(p, HERE) + '\n' for p in mods))
        return
    if not res['gates_ok']:
        raise Stop('a gate failed; no selection reported (protocol)')
    # ---- selection and scoring
    inner_rows, sel_rows, grp_rows, summ, chg = [], [], [], {}, {}
    for kind in PHYS:
        Xk = X[kind]
        p5, sc5 = outer(Xk, {c: TAU_A for c in CELLS}); s5, G5 = CV.evaluate(B0t, p5, groups_all)
        summ[f'{kind}|fixed5'] = dict(**s5, scales=sc5); grp_rows.append(G5.assign(baseline=kind, rule='fixed5'))
        for g, grid in GRIDS.items():
            taus, info = {}, {}
            for c in CELLS:
                taus[c], T, info[c] = select(cal_rows(Xk, c), grid)
                inner_rows.append(T.assign(baseline=kind, grid=g, cell=c))
            p, sc = outer(Xk, taus); s, Gk = CV.evaluate(B0t, p, groups_all)
            summ[f'{kind}|{g}'] = dict(**s, scales=sc, inner=info); grp_rows.append(Gk.assign(baseline=kind, rule=g))
            chg[f'{kind}|{g}'] = changed(G5, Gk)
            p.rename('pred').to_csv(OUT / f'requests_{kind}_{g}.csv', index_label='row_id', float_format='%.17g')
            for c in CELLS:
                sel_rows.append(dict(baseline=kind, grid=g, cell=c, tau=taus[c], scale=sc[c]['scale'], scale_fixed5=sc5[c]['scale'],
                                     inner_score=info[c]['score'], tied=' '.join(f'{100 * t:.2f}' for t in info[c]['tied']), calibration_rows=sc[c]['calibration_rows']))
    pd.concat(inner_rows, ignore_index=True).to_csv(OUT / 'inner_scores.csv', index=False)
    pd.DataFrame(sel_rows).to_csv(OUT / 'selected.csv', index=False)
    pd.concat(grp_rows, ignore_index=True).to_csv(OUT / 'groups.csv', index=False)
    res['summary'] = summ; res['changed_groups_vs_fixed5'] = chg
    for k, s in summ.items():
        t = ' '.join(f"{c} {100 * v['tau']:.2f} % s {v['scale']:.2f}" for c, v in s['scales'].items())
        say(f"{k:8s} closed loop {s['closed_loop']:2d}/15  original {s['original']:2d}/15  int {100 * s['int_rate']:.2f} %  util {100 * s['util_q50']:.1f} %  | {t}")
        if 'inner' in s:
            for c, v in s['inner'].items():
                say(f"           inner {c}: best {v['score']} (tied {v['tied']}), by tau {v['score_by_tau']}")
        if k in chg:
            say('           changed groups vs fixed 5 %:', chg[k] or 'none')
    n, n5 = summ['P|main']['closed_loop'], summ['P|fixed5']['closed_loop']
    res['verdict'] = dict(N_nested=n, N_fixed5=n5, label='중첩 τ 쪽이 많음' if n > n5 else ('같음' if n == n5 else '중첩 τ 쪽이 적음'),
                          main_configuration='unchanged (tau 5 % fixed, V252)')
    say('verdict (P, main grid):', res['verdict'])
    say('ci_note: 세 물리적 셀에서 구성된 조건별 기술통계이며, 15개 그룹을 독립 시행으로 간주한 일반화율 신뢰구간은 제시하지 않았다.')
    data, mods = project_inputs()
    res['unregistered_reads'] = sorted({os.path.relpath(p, HERE) for p in data + mods} - reg)
    say('files read but not registered:', res['unregistered_reads'] or 'none')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gates-only', action='store_true'); a = ap.parse_args()
    OUT.mkdir(exist_ok=True); stem = 'gates_only' if a.gates_only else 'run'
    res = dict(protocol='PROTOCOL_NESTEDTAU_V280.md', mode=stem, grids={k: [100 * t for t in v] for k, v in GRIDS.items()})
    try:
        run(a.gates_only, res)
    except Stop as e:
        res['stopped'] = str(e); say('STOPPED:', e)
        raise SystemExit(1)
    finally:
        dump(OUT / f'{stem}.json', V2.clean(res))
        (OUT / f'{stem}.log').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
