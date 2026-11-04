"""V322b: gates and numbers of a board run of the pure-float32 build (PROTOCOL_MCU_V322B.md).
usage: OMP_NUM_THREADS=1 python summary_v322b.py PREFIX OUT_JSON [--pilot]   (torch_env)
       PREFIX = the board run prefix (PREFIX_run.json, _rows.csv, _requests.csv, _timing_queries.csv, and _compare.json from
       compare_host.py). --pilot: the counts of G1 come from the run itself and G2 skips the group counts (a subset of rows)."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
FW = OUT / 'fw_v322b/Build'
N_SAMPLES, N_Q = 1902037, 672
SOFT_D_SECTIONS = ('SOP object, soft-double calls:', 'harness object (Build/Src/main.o), soft-double calls:',
                   'linked image (Build/sop_v322b.elf), soft-double helpers present:')


def size_txt(p):
    rows = [ln.split() for ln in Path(p).read_text().splitlines()[1:] if ln.strip()]
    return {Path(r[5]).name: dict(text=int(r[0]), data=int(r[1]), bss=int(r[2])) for r in rows}


def map_lib_bytes(p):
    """flash bytes (.text* and .rodata* input sections placed in flash) per library, from a GNU ld map file"""
    s = Path(p).read_text(); s = s[s.index('Linker script and memory map'):]
    pat = re.compile(r'^ (\.(?:text|rodata)\S*)\s*\n?\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(\S+)$', re.M)
    out = {}
    for name, addr, size, path in pat.findall(s):
        if int(addr, 16) < 0x08000000:
            continue
        m = re.search(r'(lib[a-z_]+)\.a\(', path)
        key = m.group(1) if m else Path(path).name
        out[key] = out.get(key, 0) + int(size, 16)
    return out


def soft_double_none(p):
    """True if each of the three soft-double listings of double_check.txt is '  none'"""
    L = Path(p).read_text().splitlines()
    res = {}
    for h in SOFT_D_SECTIONS:
        i = L.index(h)
        res[h] = L[i + 1].strip() == 'none'
    return res


def same_prec_diff(board, host):
    B = pd.read_csv(board, float_precision='round_trip').set_index('query_id')
    H = pd.read_csv(host, float_precision='round_trip').set_index('query_id').reindex(B.index)
    assert list(B.columns) == list(H.columns)
    out, ncell = {}, 0
    for c in B.columns:
        a, b = B[c].to_numpy(float), H[c].to_numpy(float)
        diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        if diff.any():
            fin = diff & ~np.isnan(a) & ~np.isnan(b)
            r = np.abs(a[fin] - b[fin]) / np.maximum(np.abs(b[fin]), 1e-300)
            out[c] = dict(n=int(diff.sum()), nan_pattern=int((np.isnan(a) != np.isnan(b)).sum()), max_rel=float(r.max()) if fin.any() else None)
            ncell += int(diff.sum())
    return dict(cells_differing=ncell, cells_total=int(B.size), columns=out)


def stats(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), median=float(np.median(x)), p99=float(np.quantile(x, 0.99)), max=float(x.max()))


def main():
    pre, outp = sys.argv[1], Path(sys.argv[2]); pilot = '--pilot' in sys.argv[3:]
    pre = str(pre if Path(pre).is_absolute() else OUT / pre)
    run = json.load(open(f'{pre}_run.json')); cmpj = json.load(open(f'{pre}_compare.json'))
    hz = run['info_start']['core_hz']
    elf_now = hashlib.sha256((OUT / run['elf']).read_bytes()).hexdigest()
    r = dict(prefix=pre, pilot=pilot, elf_sha256=run['elf_sha256'], elapsed_s=run['elapsed_s'])
    # gates
    nq, ns = (run['queries'], run['samples']) if pilot else (N_Q, N_SAMPLES)
    g1 = run['queries'] == nq and run['samples'] == ns and run['sample_cycles']['count'] == ns
    nan_pred = sum(v['nan_mismatch'] for v in cmpj['predictions'].values()) + cmpj['centre']['nan_mismatch']
    g2 = cmpj['event_mismatch'] == 0 and cmpj['region_mismatch'] == 0 and nan_pred == 0
    if not pilot:
        g2 = g2 and cmpj['groups_closed_loop'] == 11 and cmpj['groups_strict'] == 4
    g3 = cmpj['centre']['max_rel'] <= 1e-3 and cmpj['request']['max_rel'] <= 1e-3 and cmpj['request']['nan_mismatch'] == 0
    sd = soft_double_none(FW / 'double_check.txt')
    g4 = all(sd.values()) and elf_now == run['elf_sha256']
    r['gates'] = dict(G1=bool(g1), G2=bool(g2), G3=bool(g3), G4=bool(g4), tol=1e-3, soft_double_none=sd, elf_matches_checked=elf_now == run['elf_sha256'])
    r['parity_python'] = dict(event_mismatch=cmpj['event_mismatch'], region_mismatch=cmpj['region_mismatch'], nan_pred_centre=nan_pred,
                              groups_closed_loop=cmpj['groups_closed_loop'], groups_strict=cmpj['groups_strict'], int_rate=cmpj['int_rate'],
                              util_q50=cmpj['util_q50'], centre_max_rel=cmpj['centre']['max_rel'], request_max_rel=cmpj['request']['max_rel'],
                              scale_max_rel=cmpj['scale']['max_rel'], pred_max_rel={k: v['max_rel'] for k, v in cmpj['predictions'].items()},
                              feat_worst=sorted(((k, v['max_rel']) for k, v in cmpj['features'].items() if isinstance(v, dict)), key=lambda x: -x[1])[:5],
                              feat_nan_mismatch=sum(v['nan_mismatch'] for v in cmpj['features'].values() if isinstance(v, dict)),
                              diagnostics=cmpj['diagnostics'])
    r['vs_host_pure_float'] = dict(rows=same_prec_diff(f'{pre}_rows.csv', OUT / 'v322b/host/host_b_rows.csv'),
                                   requests=same_prec_diff(f'{pre}_requests.csv', OUT / 'v322b/host/host_b_requests.csv'))
    # timing
    sc = run['sample_cycles']; h = np.array(sc['hist']); w = sc['hist_bin_cycles']; c = np.cumsum(h) / sc['count']
    pct = {f'p{q * 100:g}': int((np.searchsorted(c, q) + 1) * w) for q in (0.5, 0.99, 0.999)}       # upper edge of the bin
    r['per_sample_cycles'] = dict(mean=sc['mean'], max=sc['max'], hist_over=sc['hist_over'], **{k + '_upper': v for k, v in pct.items()})
    r['per_sample_us'] = {k: v / hz * 1e6 for k, v in r['per_sample_cycles'].items() if k != 'hist_over'}
    T = pd.read_csv(f'{pre}_timing_queries.csv')
    assert len(T) == nq and T[['cyc_ad_req', 'cyc_ad_event']].notna().all().all()
    T['total'] = T.cyc_feat + T.cyc_pred + T.cyc_ad_req + T.cyc_ad_event
    r['per_query_cycles'] = {k: stats(T[k]) for k in ('cyc_feat', 'cyc_pred', 'cyc_ad_req', 'cyc_ad_event', 'total')}
    r['per_query_us'] = {k: {q: v / hz * 1e6 for q, v in d.items()} for k, d in r['per_query_cycles'].items()}
    V3 = json.load(open(OUT / 'full/summary_v322.json'))                               # the same sample times as stream_v322.bin
    r['cpu_share_at_median_dt'] = sc['mean'] / hz / V3['stream']['dt_s']['median']
    # memory
    sz = size_txt(FW / 'size.txt')['sop_v322b.elf']; so = size_txt(FW / 'size_objs.txt'); info = run['info_start']
    lib = map_lib_bytes(FW / 'sop_v322b.map')
    r['memory_bytes'] = dict(flash_total=sz['text'] + sz['data'], ram_static=sz['data'] + sz['bss'], sop_code=so['sop_v322b.o']['text'],
                             libm=lib.get('libm', 0), libgcc=lib.get('libgcc', 0), coef_3folds=so['coef_v322b.o']['text'] + so['coef_v322b.o']['data'],
                             coef_1fold=info['sizeof_coef_3folds'] // 3,
                             sop_state=info['sizeof_eng'] + info['sizeof_feats'] + info['sizeof_pred_out'] + info['sizeof_ad'],
                             sizeof=dict(eng=info['sizeof_eng'], feats=info['sizeof_feats'], pred_out=info['sizeof_pred_out'], ad=info['sizeof_ad']),
                             stack_highwater=run['stack_hw_end'], cyc_empty_interval=info['cyc_empty_interval'])
    m = r['memory_bytes']
    m['one_cell_flash'] = m['sop_code'] + m['libm'] + m['libgcc'] + m['coef_1fold']; m['one_cell_ram'] = m['sop_state'] + m['stack_highwater']
    # the V322 mixed-precision float build, for comparison (same parser for its map)
    v = V3['float']; lib0 = map_lib_bytes(OUT / 'fw/Build/float/sop_v322.map')
    r['v322_mixed_float'] = dict(per_sample_us_mean=v['per_sample_us']['mean'], per_sample_us_max=v['per_sample_us']['max'],
                                 per_query_us={k: v['per_query_us'][k]['mean'] for k in v['per_query_us']},
                                 per_query_us_max={k: v['per_query_us'][k]['max'] for k in v['per_query_us']},
                                 memory_bytes=dict(v['memory_bytes'], libm=lib0.get('libm', 0), libgcc=lib0.get('libgcc', 0)))
    json.dump(r, open(outp, 'w'), indent=1)
    print('gates', r['gates'])
    print('python parity', {k: r['parity_python'][k] for k in ('event_mismatch', 'region_mismatch', 'nan_pred_centre', 'groups_closed_loop', 'groups_strict',
                                                                  'centre_max_rel', 'request_max_rel')})
    print('vs host pure float: differing cells rows', r['vs_host_pure_float']['rows']['cells_differing'], 'requests',
          r['vs_host_pure_float']['requests']['cells_differing'])
    print('per sample us', {k: round(x, 3) for k, x in r['per_sample_us'].items()})
    print('per query us mean', {k: round(d['mean'], 2) for k, d in r['per_query_us'].items()}, '| max', {k: round(d['max'], 2) for k, d in r['per_query_us'].items()})
    print('memory', r['memory_bytes'])
    print('V322 mixed float', r['v322_mixed_float'])


if __name__ == '__main__':
    main()
