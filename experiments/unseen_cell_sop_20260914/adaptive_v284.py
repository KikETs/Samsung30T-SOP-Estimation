"""V284 (PROTOCOL_ADAPTIVE_V284.md): online adaptation of the derating scale from binary voltage-clamp feedback
(adaptive conformal inference applied to the log scale) on top of the L predictions (V278 baseline L).
The only feedback is e_k = 1[request_k > reference SOP_k] of queries already executed, in time order per cell.
usage: python adaptive_v284.py   (torch_env; once, after the registration adaptive_v284/v284_prereg.sha256)"""
import hashlib
import re
import sys

import numpy as np, pandas as pd
from run_envelope import HERE, dump
import closedloop_v251 as C1, closedloop_v255 as V5
import leakfree_v272 as V2
import invariance_v276 as I6
import compare_v278 as CV

OUT = HERE / 'adaptive_v284'
CV.TMP = OUT / 'tmp'
GK = C1.GK; CELLS = V5.CELLS; TOL = 1e-12
LO, HI = np.log(0.20), np.log(1.50)
GAMMA, TAU = 0.05, 0.05
GRID_G = (0.01, 0.02, 0.05, 0.10, 0.20); GRID_T = (0.025, 0.05, 0.10)
MONTHS = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'March': 3, 'Apr': 4, 'April': 4, 'May': 5, 'Jun': 6, 'June': 6, 'Jul': 7,
          'July': 7, 'Aug': 8, 'Sep': 9, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def check_hashes():
    bad = []
    for line in (OUT / 'v284_prereg.sha256').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            h, p = line.split(None, 1); q = HERE / p.strip()
            if not q.is_file() or hashlib.sha256(q.read_bytes()).hexdigest() != h:
                bad.append(p.strip())
    if bad:
        raise SystemExit(f'registration mismatch: {bad}')


def file_order(member):
    """A files: the ageing cycle in the file name; B files: month * 100 + day in the file name"""
    m = re.search(r'Cycle(\d+)', member)
    if m:
        return float(m.group(1))
    m = re.search(r'_(Jan|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|Aug|Sept|Sep|Oct|Nov|Dec)(\d+)_', member)
    return float(MONTHS[m.group(1)] * 100 + int(m.group(2)))


def adapt(J, s0, gamma, tau, unit):
    """requests with the scale updated after each executed query from the clamp event alone"""
    req = np.full(len(J), np.nan); sc = np.full(len(J), np.nan); ev = np.zeros(len(J), bool)
    cen, tru = J.central.to_numpy(), J.truth.to_numpy(); dirs = J.direction.to_numpy()
    for c in CELLS:
        state = {}
        for i in J.index[(J.evaluation_group == c).to_numpy()][np.argsort(J.loc[J.evaluation_group == c, 'tkey'].to_numpy(), kind='stable')]:
            key = c if unit == 'cell' else (c, dirs[i])
            ls = state.get(key, np.log(s0[c]))
            s = float(np.exp(ls)); r = s * cen[i]
            e = bool(r / tru[i] > 1 + 1e-9)                     # the benchmark's intervention test = ideal clamp engaged
            req[i], sc[i], ev[i] = r, s, e
            state[key] = float(np.clip(ls + gamma * (tau - float(e)), LO, HI))
    return req, sc, ev


def score(J, req):
    s, G = C1.closed_loop(J.assign(pred=req), req)
    orig, exc = C1.open_loop(J.assign(pred=req), 'pred')
    G = G.copy(); G['passed'] = (G.util_q50 >= 0.80) & (G.int_rate <= 0.10)
    return dict(closed_loop=s['groups_util80_int10'], original=orig, exceed_rows=exc, int_rate=s['int_rate'], util_q50=s['util_q50'],
                depth_q95=s['depth_q95'], depth_max=s['depth_max']), G


def main():
    OUT.mkdir(exist_ok=True)
    check_hashes(); say('registration: ok')
    B = V5.load(); src = I6.sources()
    X = CV.build('L', 'L', src, B, None)
    fold = {(f, r): float(np.exp(m)) for f, r, m in zip(X['C'].fold, X['C'].row_id, X['C'].m)}
    bench = pd.read_csv(HERE / 'closedloop_v251/src/bench_rows.csv')[['row_id', 'member', 'q_row']]
    J = B[GK + ['row_id', 'truth']].merge(bench, on='row_id', how='left').reset_index(drop=True)
    assert len(J) == 672 and J.member.notna().all() and J.q_row.notna().all()
    J['central'] = [fold[(c, r)] for c, r in zip(J.evaluation_group, J.row_id)]
    J['ford'] = [file_order(m) for m in J.member]
    J['tkey'] = J.ford * 1e7 + J.q_row                                  # file order first, then row in the file
    assert J.groupby('evaluation_group').tkey.apply(lambda x: x.is_unique).all()
    run = __import__('json').load(open(HERE / 'compare_v278/L/run.json'))
    s0 = {c: float(run['scales_tau5']['B0'][c]) for c in CELLS}
    res = dict(protocol='PROTOCOL_ADAPTIVE_V284.md', s0=s0, gamma=GAMMA, tau_star=TAU)

    # G0 gate: no adaptation reproduces the registered L requests
    ref = pd.read_csv(HERE / 'compare_v278/L/requests_B0.csv', float_precision='round_trip').set_index('row_id').pred.reindex(J.row_id).to_numpy()
    g0, _, _ = adapt(J, s0, 0.0, TAU, 'cell')
    rel = float(np.max(np.abs(g0 / ref - 1)))
    res['gate'] = dict(max_rel_vs_registered_L=rel, ok=bool(rel <= TOL)); say('G0 gate:', res['gate'])
    if not res['gate']['ok']:
        dump(OUT / 'summary.json', V2.clean(res)); (OUT / 'run.log').write_text('\n'.join(LOG) + '\n')
        raise SystemExit('gate failed')
    res['B0_L'], _ = score(J, ref)

    for arm, unit in (('AD', 'cell'), ('AD_dir', 'cell_dir')):
        req, sc, ev = adapt(J, s0, GAMMA, TAU, unit)
        sm, G = score(J, req)
        out = J[['row_id', 'evaluation_group', 'direction', 'temp_C', 'horizon_s', 'member', 'q_row', 'ford', 'truth', 'central']].assign(scale=sc, request=req, clamp=ev)
        out.to_csv(OUT / f'requests_{arm}.csv', index=False); G.to_csv(OUT / f'groups_{arm}.csv', index=False)
        traj = out.sort_values(['evaluation_group', 'ford', 'q_row'])
        tr = {c: dict(start=float(g.scale.iloc[0]), end=float(g.scale.iloc[-1]), min=float(g.scale.min()), max=float(g.scale.max()),
                      clamps=int(g.clamp.sum()), queries=int(len(g)), clamp_rate=float(g.clamp.mean())) for c, g in traj.groupby('evaluation_group')}
        traj[['evaluation_group', 'direction', 'member', 'q_row', 'scale', 'clamp']].to_csv(OUT / f'scale_trajectory_{arm}.csv', index=False)
        sm['per_cell'] = tr
        sm['failed_groups'] = [f'{x.evaluation_group} {x.direction} {x.temp_C} C {x.horizon_s:g} s (u {x.util_q50:.3f}, it {x.int_rate:.3f})' for x in G[~G.passed].itertuples()]
        res[arm] = sm; say(arm, {k: v for k, v in sm.items() if k != 'per_cell'}); say('   per cell', tr)

    a = res['AD']
    if a['closed_loop'] >= 11 and a['int_rate'] <= 0.10:
        verdict = '적응 유효'
    elif a['closed_loop'] >= 10:
        verdict = '부분'
    else:
        verdict = '효과 없음'
    res['verdict'] = verdict; say('verdict (AD):', verdict)

    rows = []                                                           # SENS: descriptive only
    for g in GRID_G:
        for t in GRID_T:
            req, _, _ = adapt(J, s0, g, t, 'cell')
            sm, _ = score(J, req)
            rows.append(dict(gamma=g, tau_star=t, **{k: sm[k] for k in ('closed_loop', 'original', 'exceed_rows', 'int_rate', 'util_q50', 'depth_max')}))
    S = pd.DataFrame(rows); S.to_csv(OUT / 'sensitivity.csv', index=False)
    say('sensitivity (descriptive):'); say(S.to_string(index=False))
    dump(OUT / 'summary.json', V2.clean(res))
    (OUT / 'run.log').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
