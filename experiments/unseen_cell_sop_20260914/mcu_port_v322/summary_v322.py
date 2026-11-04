"""V322: gates and numbers of the registered board runs (PROTOCOL_MCU_V322.md) -> full/summary_v322.json
usage: OMP_NUM_THREADS=1 python summary_v322.py   (torch_env)"""
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
FULL = OUT / 'full'
N_SAMPLES, N_Q = 1902037, 672


def size_txt(p):
    rows = [ln.split() for ln in Path(p).read_text().splitlines()[1:] if ln.strip()]
    return {Path(r[5]).name: dict(text=int(r[0]), data=int(r[1]), bss=int(r[2])) for r in rows}


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
    res = {}
    S = (OUT / 'stream_v322.bin').read_bytes(); nw = struct.unpack_from('<I', S, 4)[0]; p = 8; dts = []
    for _ in range(nw):
        wid, cell, ns, nq = struct.unpack_from('<4I', S, p); p += 16 + 16 * nq
        t = np.frombuffer(S, dtype=np.dtype([('t', '<i8'), ('st', '<u4'), ('key', '<i4'), ('I', '<i4'), ('V', '<i4'), ('T', '<f8')]), count=ns, offset=p)['t']
        dts.append(np.diff(t)); p += 32 * ns
    dt = np.concatenate(dts) / 1000.0
    res['stream'] = dict(windows=nw, samples=int(sum(len(d) + 1 for d in dts)), dt_s=dict(min=float(dt.min()), p05=float(np.quantile(dt, 0.05)),
                                                                                       median=float(np.median(dt)), p95=float(np.quantile(dt, 0.95))))
    for pr in ('float', 'double'):
        run = json.load(open(FULL / f'board_{pr}_run.json')); cmpj = json.load(open(FULL / f'board_{pr}_compare.json'))
        hz = run['info_start']['core_hz']
        r = dict(elf_sha256=run['elf_sha256'], elapsed_s=run['elapsed_s'])
        # gates
        g1 = run['queries'] == N_Q and run['samples'] == N_SAMPLES and run['sample_cycles']['count'] == N_SAMPLES
        nan_pred = sum(v['nan_mismatch'] for v in cmpj['predictions'].values()) + cmpj['centre']['nan_mismatch']
        g2 = cmpj['event_mismatch'] == 0 and cmpj['region_mismatch'] == 0 and nan_pred == 0 and cmpj['groups_closed_loop'] == 11 and cmpj['groups_strict'] == 4
        tol = 1e-9 if pr == 'double' else 1e-3
        g3 = cmpj['centre']['max_rel'] <= tol and cmpj['request']['max_rel'] <= tol and cmpj['request']['nan_mismatch'] == 0
        r['gates'] = dict(G1=bool(g1), G2=bool(g2), G3=bool(g3), tol=tol)
        r['parity_python'] = dict(event_mismatch=cmpj['event_mismatch'], region_mismatch=cmpj['region_mismatch'], nan_pred_centre=nan_pred,
                                  groups_closed_loop=cmpj['groups_closed_loop'], groups_strict=cmpj['groups_strict'], int_rate=cmpj['int_rate'],
                                  util_q50=cmpj['util_q50'], centre_max_rel=cmpj['centre']['max_rel'], request_max_rel=cmpj['request']['max_rel'],
                                  scale_max_rel=cmpj['scale']['max_rel'], pred_max_rel={k: v['max_rel'] for k, v in cmpj['predictions'].items()},
                                  feat_worst=sorted(((k, v['max_rel']) for k, v in cmpj['features'].items() if isinstance(v, dict)), key=lambda x: -x[1])[:5],
                                  feat_nan_mismatch=sum(v['nan_mismatch'] for v in cmpj['features'].values() if isinstance(v, dict)),
                                  diagnostics=cmpj['diagnostics'])
        x = 'f' if pr == 'float' else 'd'
        r['vs_host_same_prec'] = dict(rows=same_prec_diff(FULL / f'board_{pr}_rows.csv', OUT / f'host/host_{x}_rows.csv'),
                                      requests=same_prec_diff(FULL / f'board_{pr}_requests.csv', OUT / f'host/host_{x}_requests.csv'))
        # timing
        sc = run['sample_cycles']; h = np.array(sc['hist']); w = sc['hist_bin_cycles']; c = np.cumsum(h) / sc['count']
        pct = {f'p{q * 100:g}': int((np.searchsorted(c, q) + 1) * w) for q in (0.5, 0.99, 0.999)}       # upper edge of the bin
        r['per_sample_cycles'] = dict(mean=sc['mean'], max=sc['max'], hist_over=sc['hist_over'], **{k + '_upper': v for k, v in pct.items()})
        r['per_sample_us'] = {k: v / hz * 1e6 for k, v in r['per_sample_cycles'].items() if k != 'hist_over'}
        T = pd.read_csv(FULL / f'board_{pr}_timing_queries.csv')
        assert len(T) == N_Q and T[['cyc_ad_req', 'cyc_ad_event']].notna().all().all()
        T['total'] = T.cyc_feat + T.cyc_pred + T.cyc_ad_req + T.cyc_ad_event
        r['per_query_cycles'] = {c: stats(T[c]) for c in ('cyc_feat', 'cyc_pred', 'cyc_ad_req', 'cyc_ad_event', 'total')}
        r['per_query_us'] = {c: {k: v / hz * 1e6 for k, v in d.items()} for c, d in r['per_query_cycles'].items()}
        r['cpu_share_at_median_dt'] = sc['mean'] / hz / res['stream']['dt_s']['median']
        # memory
        sz = size_txt(OUT / f'fw/Build/{pr}/size.txt')['sop_v322.elf']; so = size_txt(OUT / f'fw/Build/{pr}/size_objs.txt')
        info = run['info_start']
        r['memory_bytes'] = dict(flash_total=sz['text'] + sz['data'], ram_static=sz['data'] + sz['bss'], sop_code=so['sop_v322.o']['text'],
                                 coef_3folds=so['coef_v322.o']['text'] + so['coef_v322.o']['data'], coef_1fold=info['sizeof_coef_3folds'] // 3,
                                 sop_state=info['sizeof_eng'] + info['sizeof_feats'] + info['sizeof_pred_out'] + info['sizeof_ad'],
                                 sizeof=dict(eng=info['sizeof_eng'], feats=info['sizeof_feats'], pred_out=info['sizeof_pred_out'], ad=info['sizeof_ad']),
                                 stack_highwater=run['stack_hw_end'], cyc_empty_interval=info['cyc_empty_interval'])
        res[pr] = r
    json.dump(res, open(FULL / 'summary_v322.json', 'w'), indent=1)
    for pr in ('float', 'double'):
        r = res[pr]
        print(pr, 'gates', r['gates'], '| groups', r['parity_python']['groups_closed_loop'], r['parity_python']['groups_strict'],
              '| centre/request rel', f"{r['parity_python']['centre_max_rel']:.2e}", f"{r['parity_python']['request_max_rel']:.2e}",
              '| vs host differing cells', r['vs_host_same_prec']['rows']['cells_differing'], 'requests', r['vs_host_same_prec']['requests']['cells_differing'])
        print('   per sample us', {k: round(v, 2) for k, v in r['per_sample_us'].items()}, '| per query total us',
              {k: round(v, 1) for k, v in r['per_query_us']['total'].items()}, '| memory', r['memory_bytes'])


if __name__ == '__main__':
    main()
