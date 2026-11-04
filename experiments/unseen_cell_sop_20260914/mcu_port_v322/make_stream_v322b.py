"""V322b: stream_v322.bin -> stream_v322b.bin. The same windows, queries and integer samples; the cell temperature goes
from float64 to float32 (the only change), so the board never receives a double. Record: int64 t_ms, uint32 st_ms,
int32 key, int32 I (1e-5 A), int32 V (1e-4 V), float32 T, uint32 0 (pad) = 32 bytes. Also checks that every quantity
the pure-float engine keeps as an integer fits its type (window-relative time in int32 ms, step time in int32).
usage: python make_stream_v322b.py   (writes stream_v322b.bin and stream_v322b_audit.json)"""
import json
import struct
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
R8 = np.dtype([('t', '<i8'), ('st', '<u4'), ('key', '<i4'), ('I', '<i4'), ('V', '<i4'), ('T', '<f8')])
R4 = np.dtype([('t', '<i8'), ('st', '<u4'), ('key', '<i4'), ('I', '<i4'), ('V', '<i4'), ('T', '<f4'), ('pad', '<u4')])
assert R8.itemsize == 32 and R4.itemsize == 32


def main():
    S = (OUT / 'stream_v322.bin').read_bytes()
    assert S[:4] == b'V322'
    nw = struct.unpack_from('<I', S, 4)[0]; p = 8
    out = [b'V32B', struct.pack('<I', nw)]
    aud = dict(windows=nw, samples=0, queries=0, max_rel_ms=0, max_st_ms=0, max_abs_dT=0.0, max_rel_dT=0.0, max_dt_ms=0, min_dt_ms=None,
               max_abs_I_e5=0, max_V_e4=0, zero_dt=0, neg_dt=0)
    for _ in range(nw):
        wid, cell, ns, nq = struct.unpack_from('<4I', S, p)
        hdr = S[p: p + 16 + 16 * nq]; p += 16 + 16 * nq
        r = np.frombuffer(S, dtype=R8, count=ns, offset=p); p += 32 * ns
        rel = r['t'] - r['t'][0]; dt = np.diff(r['t'])
        aud['max_rel_ms'] = max(aud['max_rel_ms'], int(rel.max())); aud['max_st_ms'] = max(aud['max_st_ms'], int(r['st'].max()))
        aud['max_dt_ms'] = max(aud['max_dt_ms'], int(dt.max())); aud['min_dt_ms'] = int(dt.min()) if aud['min_dt_ms'] is None else min(aud['min_dt_ms'], int(dt.min()))
        aud['zero_dt'] += int((dt == 0).sum()); aud['neg_dt'] += int((dt < 0).sum())     # repeated time stamps exist in the raw files
        aud['max_abs_I_e5'] = max(aud['max_abs_I_e5'], int(np.abs(r['I']).max())); aud['max_V_e4'] = max(aud['max_V_e4'], int(r['V'].max()))
        w = np.zeros(ns, dtype=R4)
        for c in ('t', 'st', 'key', 'I', 'V'):
            w[c] = r[c]
        w['T'] = r['T'].astype(np.float32)
        d = np.abs(w['T'].astype(np.float64) - r['T'])
        aud['max_abs_dT'] = max(aud['max_abs_dT'], float(d.max()))
        aud['max_rel_dT'] = max(aud['max_rel_dT'], float((d / np.maximum(np.abs(r['T']), 1e-300)).max()))
        assert np.isfinite(w['T']).all()
        out += [hdr, w.tobytes()]
        aud['samples'] += ns; aud['queries'] += nq
    assert p == len(S)
    # the engine's integer ranges: window-relative time and step time in int32 ms; |I| < 2^24 and V < 2^24 so (float) is exact
    aud['checks'] = dict(rel_time_int32=aud['max_rel_ms'] < 2 ** 31 - 10 ** 6, step_time_int32=aud['max_st_ms'] < 2 ** 31 - 1,
                         I_exact_in_float=aud['max_abs_I_e5'] < 2 ** 24, V_exact_in_float=aud['max_V_e4'] < 2 ** 24, time_nondecreasing=aud['neg_dt'] == 0)
    assert all(aud['checks'].values()), aud
    (OUT / 'stream_v322b.bin').write_bytes(b''.join(out))
    json.dump(aud, open(OUT / 'stream_v322b_audit.json', 'w'), indent=1)
    print(json.dumps(aud, indent=1))


if __name__ == '__main__':
    main()
