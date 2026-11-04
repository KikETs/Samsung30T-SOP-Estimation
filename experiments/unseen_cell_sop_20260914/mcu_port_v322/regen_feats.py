"""V322 step 1b: recompute specaudit_v287/feats_{orig,fixed}.csv with the frozen V287 step-1 code (specaudit_v287_features
member_job, unchanged) but 3 worker processes instead of 6 (user limit). Writes to mcu_port_v322/ first; the files are copied
to specaudit_v287/ only if their SHA-256 equals the hashes recorded before the cleanup.
usage: OMP_NUM_THREADS=1 python regen_feats.py   (torch_env)"""
import hashlib, shutil, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parents[1]; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import specaudit_v287_features as F  # noqa: E402
EXPECTED = {'orig': '74109bf897337912185149a7ced35d12c77bf6278b1af7f330b24333f19a2344',   # MANIFEST r1/r2, v320_prereg
            'fixed': '45bdc3fc2d816e000002e8dae6a11547104c404b5c68e534e1313c4c5d0cdbd7'}


def main():
    B = pd.read_csv(F.BENCH); SW = pd.read_csv(F.V274 / 'sweeps.csv')
    B = B.merge(SW[['member', 'h0_row', 'direction', 'sweep_id']], on=['member', 'h0_row', 'direction'], how='left', validate='many_to_one')
    assert len(B) == 672 and B.sweep_id.notna().all()
    jobs = []
    for m, g in B.groupby('member'):
        items = []
        for (sid, h0), s in g.groupby(['sweep_id', 'h0_row']):
            items.append((int(sid), int(h0), int(s.q_row.max()), s[['row_id', 'q_row', 'direction']].to_dict('records')))
        jobs += [('orig', m, items), ('fixed', m, items)]
    res = {'orig': [], 'fixed': []}
    with ProcessPoolExecutor(3, initializer=F._init) as ex:
        for kind, f in ex.map(F.member_job, jobs):
            res[kind] += f
    ok = True
    for kind, f in res.items():
        D = pd.DataFrame(f); assert len(D) == 672 and D.row_id.is_unique, kind
        p = OUT / f'feats_{kind}.csv'; D.to_csv(p, index=False, float_format='%.17g')
        h = hashlib.sha256(p.read_bytes()).hexdigest(); same = h == EXPECTED[kind]; ok &= same
        print(kind, 'rows', len(D), 'sha256', h[:16], 'matches recorded' if same else 'DIFFERS from recorded', flush=True)
    if ok:
        for kind in res:
            tgt = F.OUT / f'feats_{kind}.csv'
            if tgt.exists():
                raise SystemExit(f'target exists: {tgt}')
            shutil.copyfile(OUT / f'feats_{kind}.csv', tgt)
        print('restored to', F.OUT)


if __name__ == '__main__':
    main()
