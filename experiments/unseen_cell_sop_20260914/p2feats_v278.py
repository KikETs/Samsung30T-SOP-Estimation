"""V278 step 1 (PROTOCOL_COMPARE_V278.md, P2): the 2 s sample of each benchmark row's query pulse and of its previous query
pulse, from the local RPCWBY zips (raw stand-in of V276, nothing masked):
  previous pulse: compare_v278/src/p2_relax_feats.one (sop_ident relax_feats.one plus the current and time of the sample
                  used for R1 / R2 / R10) -> prev_R2, prev_I2, prev_t2;
  query pulse:    cltrain_v274_features.onset (events.py's per-onset formulas) at q -> q_R2, q_I2, q_t2 (training target
                  of the 2 s residual fit; read only for the training cells' rows).
Gates (in compare_v278.py): prev_R2 = relax.csv prev_R2 and q_R2 = relax.csv R2_q (relative 1e-9).
Output: compare_v278/p2_feats.csv
usage: python p2feats_v278.py"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'compare_v278'; SRC = OUT / 'src'; SHIM = HERE / 'invariance_v276' / 'src' / 'shim_mask'
BENCH = HERE / 'leakfree_v272/src/pybamm/out/rpcwby_bench/rows.csv'; SWEEPS = HERE / 'cltrain_v274/src/sweeps.csv'
REST_A = 0.02


def _init():
    for p in (str(HERE), str(SRC), str(SHIM)):
        if p not in sys.path:
            sys.path.insert(0, p)


def member_job(args):
    _init()
    import raw
    import p2_relax_feats as rf
    import cltrain_v274_features as cf
    assert Path(raw.__file__).resolve().parent == SHIM and Path(rf.__file__).resolve().parent == SRC
    member, sweeps = args
    raw.MASK[:] = []
    out = []
    for sid, h0, rows in sweeps:
        end = max(r['q_row'] for r in rows)
        F = {f['row_id']: f for f in rf.one((sid, member, h0, end, [dict(row_id=r['row_id'], q_row=r['q_row'], direction=r['direction']) for r in rows]))}
        last = max(r['last_row'] for r in rows)
        W = raw.window(member, h0, last + 1, ['t', 'st', 'cyc', 'step', 'I', 'V', 'T_cell'])
        t, I, V, st, T = W.t.to_numpy(), W.I.to_numpy(), W.V.to_numpy(), W.st.to_numpy(), W.T_cell.to_numpy()
        key = W.cyc.to_numpy() * 100000 + W.step.to_numpy(); rest = np.abs(I) < REST_A
        for r in rows:
            f = F.get(r['row_id'], {}); o = cf.onset(t, I, V, st, key, rest, T, int(r['q_row']) - h0) or {}
            out.append(dict(row_id=r['row_id'], prev_R2=f.get('prev_R2', np.nan), prev_I2=f.get('prev_I2', np.nan), prev_t2=f.get('prev_t2', np.nan),
                            prev_I_last=f.get('prev_I_last', np.nan), prev_dt_last=f.get('prev_dt_last', np.nan),
                            q_R2=o.get('R2', np.nan), q_I2=o.get('I_R2', np.nan), q_t2=o.get('t_R2', np.nan)))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    B = pd.read_csv(BENCH); SW = pd.read_csv(SWEEPS)
    B = B.merge(SW[['member', 'h0_row', 'direction', 'sweep_id']], on=['member', 'h0_row', 'direction'], how='left', validate='many_to_one')
    assert len(B) == 672 and B.sweep_id.notna().all()
    jobs = []
    for m, g in B.groupby('member'):
        jobs.append((m, [(int(s), int(h), g2[['row_id', 'q_row', 'last_row', 'direction']].to_dict('records')) for (s, h), g2 in g.groupby(['sweep_id', 'h0_row'])]))
    res = []
    with ProcessPoolExecutor(8, initializer=_init) as ex:
        for o in ex.map(member_job, jobs):
            res += o
    P = pd.DataFrame(res); assert len(P) == 672 and P.row_id.is_unique
    P.to_csv(OUT / 'p2_feats.csv', index=False, float_format='%.17g')
    print('p2_feats.csv', len(P), 'rows | prev_I2 finite', int(np.isfinite(P.prev_I2).sum()), '| q_I2 finite', int(np.isfinite(P.q_I2).sum()), flush=True)


if __name__ == '__main__':
    main()
