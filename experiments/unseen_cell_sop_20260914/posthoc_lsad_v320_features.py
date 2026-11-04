"""V320 step 1 (PROTOCOL_POSTHOC_LSAD_V320.md): relax features of the 672 benchmark rows recomputed with the V287 audit
extractor (specaudit_v287/src/relax_feats_audit.py, FIXED = True: deviations R1 and R2 fixed, as in Ls) through the V276
masking stand-in of raw.py, with the V276 masks (invariance_v276_features.bench / jobs, unchanged):
  ref  one call per sweep on [h0, last q of the sweep], nothing removed;
  T2   one call per row on [h0, q] with the query's first sample q removed: nothing at or after the query;
  T4   one call per sweep with every non-final pulse of each row's own search removed.
Three worker processes. Nothing here reads a truth power. The process CPU time of every 'ref' job is
recorded (feature-extraction cost of Ls, one measurement).
Output: posthoc_lsad_v320/feats_ref_fixed.csv, feats_T2_fixed.csv, feats_T4_fixed.csv, reads.csv, feature_cpu.json
usage: OMP_NUM_THREADS=1 python posthoc_lsad_v320_features.py   (once, after posthoc_lsad_v320/v320_prereg.sha256)"""
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'posthoc_lsad_v320'; SRC = HERE / 'specaudit_v287' / 'src'; SHIM = HERE / 'invariance_v276' / 'src' / 'shim_mask'
sys.path.insert(0, str(HERE))
import invariance_v276_features as F6  # noqa: E402  (bench() and jobs(): the V276 rows, sweeps and masks)


def _init():
    for p in (str(SRC), str(SHIM)):                      # SHIM first on the path: relax_feats_audit's `import raw` finds it
        if p not in sys.path:
            sys.path.insert(0, p)


def member_job(args):
    """kind, member, items = [(sweep_id, h0, end, rows, mask)]: relax_feats_audit.one (FIXED) per item with raw.MASK = mask"""
    _init()
    import raw
    import relax_feats_audit as rf
    assert Path(raw.__file__).resolve().parent == SHIM and Path(rf.__file__).resolve().parent == SRC
    kind, member, items = args
    rf.FIXED = True
    feats = []
    raw.READS.clear()
    c0 = time.process_time()
    for sid, h0, end, rows, mask in items:
        raw.MASK[:] = mask
        feats += rf.one((sid, member, h0, end, rows))
    cpu = time.process_time() - c0
    raw.MASK[:] = []
    return kind, member, feats, [(kind, m, a, b) for m, a, b in raw.READS], cpu


def check_hashes():
    bad = []
    for line in (OUT / 'v320_prereg.sha256').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            h, p = line.split(None, 1); q = HERE / p.strip()
            if not q.is_file() or hashlib.sha256(q.read_bytes()).hexdigest() != h:
                bad.append(p.strip())
    if bad:
        raise SystemExit(f'registration mismatch: {bad}')


def main():
    check_hashes(); print('registration: ok', flush=True)
    B, n_iter = F6.bench()
    print('benchmark rows', len(B), '| earlier iterations of the rows\' searches:', n_iter, flush=True)
    res = {'ref': [], 'T2': [], 'T4': []}; reads = []; cpu = {}
    with ProcessPoolExecutor(3, initializer=_init) as ex:
        for kind, member, f, r, c in ex.map(member_job, F6.jobs(B)):
            res[kind] += f; reads += r; cpu.setdefault(kind, {})[member] = c
    for kind, f in res.items():
        Fk = pd.DataFrame(f)
        assert len(Fk) == 672 and Fk.row_id.is_unique, kind
        Fk.sort_values('row_id').to_csv(OUT / f'feats_{kind}_fixed.csv', index=False, float_format='%.17g')
        print(kind, 'rows', len(Fk), flush=True)
    pd.DataFrame(reads, columns=['kind', 'member', 'start', 'end']).to_csv(OUT / 'reads.csv', index=False)
    tot = sum(cpu['ref'].values())
    info = dict(ref_cpu_s=tot, rows=672, ref_cpu_ms_per_row=1000 * tot / 672, members=len(cpu['ref']),
                note='process CPU time of the ref jobs (one call per sweep, nothing masked), summed over the worker processes; one measurement')
    (OUT / 'feature_cpu.json').write_text(json.dumps(info, indent=1) + '\n')
    print('feature extraction (ref):', info, flush=True)


if __name__ == '__main__':
    main()
