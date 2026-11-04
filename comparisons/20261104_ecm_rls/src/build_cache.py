"""Parse the RPCWBY member CSVs used by the 672-row benchmark into a numeric cache (run on the remote server).

The zips are opened read-only. For every member named in inputs/bench_rows.csv the script stores, in file-line
order (header excluded), Data_Point, Test_Time(s), Step_Time(s), Cycle_Index, Step_Index, Current(A), Voltage(V)
and the temperature column named in temp_col. The predictor reads only Test_Time, Current and Voltage; the other
columns serve the evaluator-side alignment checks. No other column is parsed (in particular no MetaCode_MV_UD registers, no
Power(W), no capacity columns). Hashes of the zips and of every decompressed member are written to
cache/manifest.json.

    python -I -B src/build_cache.py --workers 8
"""
import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
RAW = EXP.parents[1] / 'raw' / 'RPCWBY'
ZIPS = {'1_Test#1': '1_Test_1.zip', '2_Test#2': '2_Test_2.zip', 'Test#3': '3_Test_3.zip'}
COLS = {'Data_Point': 'dp', 'Test_Time(s)': 't', 'Step_Time(s)': 'st', 'Cycle_Index': 'cyc', 'Step_Index': 'step',
        'Current(A)': 'I', 'Voltage(V)': 'V'}
CACHE = EXP / 'cache'


def cache_path(member):
    return CACHE / (member.replace('/', '__').replace('#', '_') + '.npz')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def one(args):
    member, temp_col = args
    with zipfile.ZipFile(RAW / ZIPS[member.split('/')[0]]) as z:
        info = z.getinfo(member)
        data = z.read(info)
    sha = hashlib.sha256(data).hexdigest()
    text_head = data[:65536].decode('utf-8-sig', errors='replace')
    header = text_head.splitlines()[0].split(',')
    assert temp_col in header, (member, temp_col)
    temp_cols = [c for c in header if c.startswith('Aux_Temperature')]
    lines = data.split(b'\n')
    if lines and lines[-1].strip() == b'':
        lines = lines[:-1]
    n_lines = len(lines) - 1
    blank = sum(1 for x in lines[1:] if x.strip() == b'')
    df = pd.read_csv(io.BytesIO(data), usecols=list(COLS) + [temp_col], encoding='utf-8-sig', low_memory=False,
                     skip_blank_lines=False)
    assert len(df) == n_lines, (member, len(df), n_lines)
    out = {}
    for src, dst in COLS.items():
        out[dst] = pd.to_numeric(df[src], errors='coerce').to_numpy()
    out['T'] = pd.to_numeric(df[temp_col], errors='coerce').to_numpy(dtype=float)
    for k in ('dp', 'cyc', 'step'):
        assert np.isfinite(out[k]).all(), (member, k)
        out[k] = out[k].astype(np.int64)
    for k in ('t', 'st', 'I', 'V'):
        out[k] = out[k].astype(float)
    nan_tiv = int(sum((~np.isfinite(out[k])).sum() for k in ('t', 'I', 'V')))
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cache_path(member).with_suffix('.part.npz')
    np.savez(tmp, **out)
    tmp.rename(cache_path(member))
    dt = np.diff(out['t'])
    return dict(member=member, zip=ZIPS[member.split('/')[0]], crc32=f'{info.CRC:08x}', bytes=info.file_size,
                sha256=sha, n_rows=int(len(df)), blank_lines=int(blank), temp_col=temp_col,
                temperature_columns_in_file=temp_cols, nan_t_I_V=nan_tiv,
                dp_equals_line_plus_1=bool(np.array_equal(out['dp'], np.arange(1, len(df) + 1))),
                negative_dt=int((dt < 0).sum()), zero_dt=int((dt == 0).sum()),
                cache_file=cache_path(member).name, cache_sha256=sha256_file(cache_path(member)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    started = time.time()
    with open(EXP / 'inputs/bench_rows.csv', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    jobs = sorted({(r['member'], r['temp_col']) for r in rows})
    assert len({m for m, _ in jobs}) == len(jobs), 'one temperature column per member'
    zips = {}
    for name in sorted(set(ZIPS.values())):
        p = RAW / name
        zips[name] = dict(bytes=p.stat().st_size, sha256=sha256_file(p), md5=md5_file(p))
    with ProcessPoolExecutor(a.workers) as ex:
        members = list(ex.map(one, jobs))
    manifest = dict(raw_dir=str(RAW), doi='10.5683/SP3/RPCWBY', license='CC BY 4.0', zips=zips, members=members,
                    columns_parsed=list(COLS) + ['<temp_col>'], elapsed_s=time.time() - started)
    (CACHE / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(n_members=len(members), elapsed_s=manifest['elapsed_s'],
                          all_dp_line_plus_1=all(m['dp_equals_line_plus_1'] for m in members),
                          blank=sum(m['blank_lines'] for m in members), nan=sum(m['nan_t_I_V'] for m in members),
                          negative_dt=sum(m['negative_dt'] for m in members), zips=zips), indent=1))


if __name__ == '__main__':
    main()
