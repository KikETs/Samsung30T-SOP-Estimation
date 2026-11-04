"""V287 (AUDIT_V287.md, R1 / R2) step 1: relax features of the 672 benchmark rows recomputed from the local RPCWBY zips
with the audit copy specaudit_v287/src/relax_feats_audit.py through the V276 raw stand-in (invariance_v276/src/shim_mask,
nothing masked), once with FIXED = False (original behaviour, gate G2a against the saved relax.csv) and once with
FIXED = True (R1, R2 fixed). Both record the start rows of the chosen prev / prev2 pulses. Nothing here reads a truth power.
Output: specaudit_v287/feats_orig.csv, feats_fixed.csv
usage: python specaudit_v287_features.py"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'specaudit_v287'; SRC = OUT / 'src'; SHIM = HERE / 'invariance_v276' / 'src' / 'shim_mask'; V274 = HERE / 'cltrain_v274' / 'src'
BENCH = HERE / 'leakfree_v272/src/pybamm/out/rpcwby_bench/rows.csv'


def _init():
    for p in (str(SRC), str(SHIM)):                      # SHIM first on the path: relax_feats_audit's `import raw` finds it
        if p not in sys.path:
            sys.path.insert(0, p)


def member_job(args):
    _init()
    import raw
    import relax_feats_audit as rf
    assert Path(raw.__file__).resolve().parent == SHIM and Path(rf.__file__).resolve().parent == SRC
    kind, member, items = args
    rf.FIXED = (kind == 'fixed'); raw.MASK[:] = []
    feats = []
    for sid, h0, end, rows in items:
        feats += rf.one((sid, member, h0, end, rows))
    return kind, feats


def main():
    B = pd.read_csv(BENCH)
    SW = pd.read_csv(V274 / 'sweeps.csv')
    B = B.merge(SW[['member', 'h0_row', 'direction', 'sweep_id']], on=['member', 'h0_row', 'direction'], how='left', validate='many_to_one')
    assert len(B) == 672 and B.sweep_id.notna().all()
    jobs = []
    for m, g in B.groupby('member'):
        items = []
        for (sid, h0), s in g.groupby(['sweep_id', 'h0_row']):
            items.append((int(sid), int(h0), int(s.q_row.max()), s[['row_id', 'q_row', 'direction']].to_dict('records')))
        jobs += [('orig', m, items), ('fixed', m, items)]
    res = {'orig': [], 'fixed': []}
    with ProcessPoolExecutor(6, initializer=_init) as ex:
        for kind, f in ex.map(member_job, jobs):
            res[kind] += f
    for kind, f in res.items():
        F = pd.DataFrame(f)
        assert len(F) == 672 and F.row_id.is_unique, kind
        F.to_csv(OUT / f'feats_{kind}.csv', index=False, float_format='%.17g')
        print(kind, 'rows', len(F), flush=True)


if __name__ == '__main__':
    main()
