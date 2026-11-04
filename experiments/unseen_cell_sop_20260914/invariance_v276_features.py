"""V276 step 1 (PROTOCOL_INVARIANCE_V276.md 4.1-4.2): local-predictor features of the 672 benchmark rows recomputed
from the local RPCWBY zips with sop_ident relax_feats.one (cltrain_v274/src copy, sha256 = the remote original) through
the masking stand-in of raw.py (invariance_v276/src/shim_mask/raw.py):
  ref  one call per sweep on [h0, last q of the sweep] (the V274 / sop_ident rule), nothing removed;
  T2   one call per row on [h0, q] with the query's first sample q removed (NaN): nothing at or after the query;
  T4   one call per sweep with every non-final pulse of each row's own search removed over [first_row, last_row]
       (V157 row_keys: same search_key, is_final False; 1,618 pulses).
Every window read is logged (member, start, end) for the h0 check. Nothing here reads a truth power.
Output: invariance_v276/feats_ref.csv, feats_T2.csv, feats_T4.csv, reads.csv
usage: python invariance_v276_features.py"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'invariance_v276'; SHIM = OUT / 'src' / 'shim_mask'; V274 = HERE / 'cltrain_v274' / 'src'
BENCH = HERE / 'leakfree_v272/src/pybamm/out/rpcwby_bench/rows.csv'
KEYS = HERE.parent / 'same_search_exclusion_v157_20260921/keys/row_keys.csv'


def _init():
    for p in (str(V274), str(SHIM)):                     # SHIM ends first: relax_feats' `import raw` finds the stand-in
        if p not in sys.path:
            sys.path.insert(0, p)


def member_job(args):
    """kind, member, items = [(sweep_id, h0, end, rows, mask)]: relax_feats.one per item with raw.MASK = mask"""
    _init()
    import raw
    import relax_feats as rf
    assert Path(raw.__file__).resolve().parent == SHIM and Path(rf.__file__).resolve().parent == V274
    kind, member, items = args
    feats = []
    raw.READS.clear()
    for sid, h0, end, rows, mask in items:
        raw.MASK[:] = mask
        feats += rf.one((sid, member, h0, end, rows))
    raw.MASK[:] = []
    return kind, feats, [(kind, m, a, b) for m, a, b in raw.READS]


def bench():
    B = pd.read_csv(BENCH)
    SW = pd.read_csv(V274 / 'sweeps.csv')
    B = B.merge(SW[['member', 'h0_row', 'direction', 'sweep_id']], on=['member', 'h0_row', 'direction'], how='left', validate='many_to_one')
    assert len(B) == 672 and B.sweep_id.notna().all()
    K = pd.read_csv(KEYS)
    F = K[K.is_final == True][['member', 'first_row', 'search_key']]
    B = B.merge(F, left_on=['member', 'q_row'], right_on=['member', 'first_row'], how='left', validate='one_to_one').drop(columns='first_row')
    assert B.search_key.notna().all()
    NF = K[K.is_final != True][['search_key', 'first_row', 'last_row']]
    iters = {k: list(zip(g.first_row.astype(int), g.last_row.astype(int))) for k, g in NF.groupby('search_key')}
    B['iters'] = [iters.get(k, []) for k in B.search_key]
    return B, int(len(NF))


def jobs(B):
    out = []
    for m, g in B.groupby('member'):
        ref, t2, t4 = [], [], []
        for (sid, h0), s in g.groupby(['sweep_id', 'h0_row']):
            rows = s[['row_id', 'q_row', 'direction']].to_dict('records')
            end = int(s.q_row.max())
            ref.append((int(sid), int(h0), end, rows, []))
            t4.append((int(sid), int(h0), end, rows, sorted({x for it in s.iters for x in it})))
            for r in rows:
                t2.append((int(sid), int(h0), int(r['q_row']), [r], [(int(r['q_row']), int(r['q_row']))]))
        out += [('ref', m, ref), ('T2', m, t2), ('T4', m, t4)]
    return out


def main():
    OUT.mkdir(exist_ok=True)
    B, n_iter = bench()
    print('benchmark rows', len(B), '| earlier iterations of the rows\' searches (all rows of row_keys, not final):', n_iter, flush=True)
    res = {'ref': [], 'T2': [], 'T4': []}; reads = []
    with ProcessPoolExecutor(6, initializer=_init) as ex:
        for kind, f, r in ex.map(member_job, jobs(B)):
            res[kind] += f; reads += r
    for kind, f in res.items():
        Fk = pd.DataFrame(f)
        assert len(Fk) == 672 and Fk.row_id.is_unique, kind
        Fk.to_csv(OUT / f'feats_{kind}.csv', index=False, float_format='%.17g')
        print(kind, 'rows', len(Fk), flush=True)
    R = pd.DataFrame(reads, columns=['kind', 'member', 'start', 'end'])
    R.to_csv(OUT / 'reads.csv', index=False)
    # first row read by any call for a row's sweep (member, h0) must be >= h0 (the windows start at h0)
    h = B.groupby(['member', 'h0_row']).size().reset_index()[['member', 'h0_row']]
    x = R.merge(h, left_on=['member', 'start'], right_on=['member', 'h0_row'], how='left')
    print('window calls', len(R), '| calls starting at a sweep h0:', int(x.h0_row.notna().sum()), flush=True)


if __name__ == '__main__':
    main()
