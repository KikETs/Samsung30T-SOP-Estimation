"""V322b board run: flash the NUCLEO-H563ZI with fw_v322b/Build/sop_v322b.elf (pure float32) and replay stream_v322b.bin
through it over USART3 (ST-LINK VCP). The board runs the same C engine as host_v322b (feature streaming, the five local
predictors, log median, scale adaptation); the PC only sends raw samples, query positions and, for the adaptation, the
binary event (request > reference SOP) computed from the board's own request, as host_v322b does. The board replies in
float32 and int64 (charge accumulator); this script converts the accumulator to Ah, as host_v322b does.
usage: OMP_NUM_THREADS=1 python board_run_v322b.py --out PREFIX [--pilot N] [--no-flash]   (torch_env: pyserial)
writes PREFIX_rows.csv, PREFIX_requests.csv (host_v322 format, for compare_host.py), PREFIX_timing_queries.csv,
PREFIX_timing_batches.csv, PREFIX_run.json (board info, sample-cycle histogram, totals, hashes)"""
import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import serial

OUT = Path(__file__).resolve().parent
CLI = 'STM32_Programmer_CLI'  # STM32CubeProgrammer command-line tool, found on PATH
MAGIC = 0x42323356                              # 'V32B'
Q_REPLY = 424
NQ = 672
MAX_BATCH = 256
HIST_N, HIST_W = 1024, 64
INTS = ['I_off', 'off_dur_s', 'Roff1', 'Roff2', 'Roff5', 'Roff10', 'Roff30', 'Roff60', 'Roff120', 'Roff240', 'has_fit', 'sq_R0', 'sq_A',
        'Rstep1', 'Rstep2', 'Rstep10']
PULSE = ['valid', 'P', 'Vpre', 'I_last', 'V_last', 'dt_last', 'R1', 'R2', 'R10', 'Ah', 'k0', 'has_int']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def header():
    h = ['query_id', 'valid', 'a_zero', 'rb_trunc', 'V_pre', 'T_pre', 'rest_s', 'I_end', 'last_dur_s', 'V_end', 'V_rest0', 'dt_rest0', 'slope_mVs',
         'Ah_q', 'last_sign'] + INTS
    for p in ('prev_', 'prev2_'):
        h += [p + x for x in PULSE] + [p + x for x in INTS] + [p + 'rest_s']
    return h + ['prev_gap_s', 'base', 'reg', 'p_prev', 'p_L4', 'p_M6', 'p_DN6', 'p_G1', 'centre', 'nfinite']


def fmt(v):
    return repr(float(v)) if isinstance(v, float) else str(int(v))


def ah_of(ah2):
    """2 x discharged charge in 1e-5 A x ms -> Ah (the same double operations as host_v322b)"""
    return float(ah2) * 0.5e-8 / 3600.0


def decode_query(b):
    """the board's QUERY reply (424 bytes: 5 uint32, 16 int32, 3 int64, 79 float32) -> (header, row in host_v322 column order)"""
    hd = struct.unpack_from('<5I', b, 0)
    iv = struct.unpack_from('<16i', b, 20)
    ah = struct.unpack_from('<3q', b, 84)
    f = list(struct.unpack_from('<79f', b, 108))
    valid, a_zero, rb_trunc, last_sign, base, reg, nfinite, pv_valid, pv_k0, pv_hint, p2_valid, p2_k0, p2_hint, fit, pv_fit, p2_fit = iv
    nan = float('nan')
    ahq = ah_of(ah[0]) if (valid and not a_zero) else nan                 # as host_v322b (the V322 engine left Ah_q NaN there)
    pv_ah = ah_of(ah[1]) if pv_valid else nan
    p2_ah = ah_of(ah[2]) if p2_valid else nan

    def interr(x, has_fit):
        return list(x[0:10]) + [has_fit] + list(x[10:15])

    def pulse(x, valid_, ahv, k0, hint, has_fit):
        return [valid_] + list(x[0:8]) + [ahv, k0, hint] + interr(x[8:23], has_fit) + [x[23]]
    row = [valid, a_zero, rb_trunc] + f[0:9] + [ahq, last_sign] + interr(f[9:24], fit)
    row += pulse(f[24:48], pv_valid, pv_ah, pv_k0, pv_hint, pv_fit) + pulse(f[48:72], p2_valid, p2_ah, p2_k0, p2_hint, p2_fit)
    row += [f[72], base, reg] + f[73:78] + [f[78], nfinite]
    return hd, row


class Board:
    def __init__(self, port, baud):
        self.p = serial.Serial(port, baud, timeout=10)
        time.sleep(0.3); self.p.reset_input_buffer()

    def xfer(self, out, n):
        self.p.write(out)
        b = self.p.read(n)
        if len(b) != n:
            raise IOError(f'short reply: {len(b)} of {n} bytes after command 0x{out[0]:02x}')
        if struct.unpack_from('<I', b, 0)[0] != MAGIC:
            raise IOError(f'bad magic after command 0x{out[0]:02x}')
        return b


def flash(elf):
    r = subprocess.run([CLI, '-c', 'port=SWD', '-w', str(elf), '-v', '-rst'], capture_output=True, text=True)
    ok = r.returncode == 0 and 'Download verified successfully' in r.stdout
    if not ok:
        print(r.stdout[-3000:], r.stderr[-2000:]); raise SystemExit('flash failed')
    time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--pilot', type=int, default=0, help='stop after the window in which this many queries are reached')
    ap.add_argument('--no-flash', action='store_true')
    ap.add_argument('--port', default='/dev/ttyACM0')
    ap.add_argument('--baud', type=int, default=921600)
    a = ap.parse_args()
    elf = OUT / 'fw_v322b/Build/sop_v322b.elf'
    pre = Path(a.out) if Path(a.out).is_absolute() else OUT / a.out
    if not a.no_flash:
        flash(elf)
    bd = Board(a.port, a.baud)
    t_start = time.time()
    info = struct.unpack('<11I', bd.xfer(bytes([0x70]), 44))
    info = dict(zip(['magic', 'version', 'sizeof_real', 'core_hz', 'sizeof_eng', 'sizeof_feats', 'sizeof_pred_out', 'sizeof_ad',
                     'sizeof_coef_3folds', 'stack_hw_at_start', 'cyc_empty_interval'], info))
    assert info['magic'] == MAGIC and info['version'] == 2 and info['sizeof_real'] == 4, info
    print('board', info, flush=True)
    S = (OUT / 'stream_v322b.bin').read_bytes(); mv = memoryview(S)
    assert S[:4] == b'V32B'
    nw = struct.unpack_from('<I', S, 4)[0]; pos = 8
    rows, tq, tb = {}, {}, []
    nsamp = 0
    for w in range(nw):
        wid, cell, ns, nq = struct.unpack_from('<4I', S, pos); pos += 16
        qs = [struct.unpack_from('<4I', S, pos + 16 * k) for k in range(nq)]; pos += 16 * nq
        base_s = pos; pos += 32 * ns
        bd.xfer(bytes([0x71]) + struct.pack('<II', cell, wid), 8)
        i, k = 0, 0
        while True:
            while k < nq and qs[k][0] == i:
                off, qid, dirn, hms = qs[k]
                b = bd.xfer(bytes([0x73]) + struct.pack('<III', qid, dirn, hms), Q_REPLY)
                hd, row = decode_query(b)
                assert hd[1] == qid, (hd, qid)
                rows[qid] = row
                tq[qid] = dict(query_id=qid, window_id=wid, cell=cell, offset=off, cyc_feat=hd[2], cyc_pred=hd[3], stack_hw=hd[4])
                k += 1
            if i == ns:
                break
            nxt = qs[k][0] if k < nq else ns
            n = min(MAX_BATCH, nxt - i)
            chunk = bytes(mv[base_s + 32 * i: base_s + 32 * (i + n)])
            r = struct.unpack('<5I', bd.xfer(bytes([0x72]) + struct.pack('<I', n) + chunk, 20))
            if r[1] != n or r[4] != (sum(chunk) & 0xFFFFFFFF):
                raise IOError(f'batch error window {wid} at {i}: {r}')
            tb.append((wid, i, n, r[2], r[3]))
            i += n; nsamp += n
        assert k == nq
        if w % 10 == 0 or w == nw - 1:
            print(f'window {w + 1}/{nw}: queries {len(rows)}, samples {nsamp}, {time.time() - t_start:.0f} s', flush=True)
        if a.pilot and len(rows) >= a.pilot:
            break
    # adaptation, per cell in the processing order; in a pilot only the processed prefix
    A = (OUT / 'ad_v322.bin').read_bytes(); p = 0; req_rows = []
    truth_of = {}
    for c in range(3):
        n = struct.unpack_from('<I', A, p)[0]; p += 4
        items = [struct.unpack_from('<Id', A, p + 12 * j) for j in range(n)]; p += 12 * n
        done = [qid for qid, _ in items if qid in rows]
        if not done:
            continue
        assert [qid for qid, _ in items[:len(done)]] == done, 'processed queries are not a prefix of the adaptation order'
        bd.xfer(bytes([0x74]) + struct.pack('<I', c), 8)
        for order, (qid, truth) in enumerate(items[:len(done)]):
            b = bd.xfer(bytes([0x75]) + struct.pack('<I', qid), 24)
            _, q2, cyc, _ = struct.unpack_from('<4I', b, 0); scale, req = struct.unpack_from('<2f', b, 16)
            assert q2 == qid
            ev = int((req / truth) > 1 + 1e-9)
            r2 = struct.unpack('<2I', bd.xfer(bytes([0x76]) + struct.pack('<I', ev), 8))
            req_rows.append((qid, order, scale, req, ev)); truth_of[qid] = truth
            tq[qid].update(cyc_ad_req=cyc, cyc_ad_event=r2[1])
    st = bd.xfer(bytes([0x77]), 24 + 4 * HIST_N)
    _, cnt, mx, lo, hi, over = struct.unpack_from('<6I', st, 0); hist = list(struct.unpack_from(f'<{HIST_N}I', st, 24))
    info_end = struct.unpack('<11I', bd.xfer(bytes([0x70]), 44))
    elapsed = time.time() - t_start
    # outputs
    H = header()
    with open(f'{pre}_rows.csv', 'w') as f:
        f.write(','.join(H) + '\n')
        for qid in sorted(rows):
            f.write(','.join([str(qid)] + [fmt(v) for v in rows[qid]]) + '\n')
    with open(f'{pre}_requests.csv', 'w') as f:
        f.write('query_id,order,scale,request,event\n')
        for qid, order, scale, req, ev in req_rows:
            f.write(f'{qid},{order},{scale!r},{req!r},{ev}\n')
    cols = ['query_id', 'window_id', 'cell', 'offset', 'cyc_feat', 'cyc_pred', 'cyc_ad_req', 'cyc_ad_event', 'stack_hw']
    with open(f'{pre}_timing_queries.csv', 'w') as f:
        f.write(','.join(cols) + '\n')
        for qid in sorted(tq):
            f.write(','.join(str(tq[qid].get(c, '')) for c in cols) + '\n')
    with open(f'{pre}_timing_batches.csv', 'w') as f:
        f.write('window_id,start,n,cyc_sum,cyc_max\n')
        for x in tb:
            f.write(','.join(map(str, x)) + '\n')
    tot = lo + (hi << 32)
    run = dict(prec='float32 (pure)', pilot=a.pilot, elf=str(elf.relative_to(OUT)), elf_sha256=sha(elf), stream_sha256=sha(OUT / 'stream_v322b.bin'),
               ad_sha256=sha(OUT / 'ad_v322.bin'), info_start=info, stack_hw_end=info_end[9], queries=len(rows), samples=nsamp,
               sample_cycles=dict(count=cnt, total=tot, mean=tot / max(cnt, 1), max=mx, hist_bin_cycles=HIST_W, hist_over=over, hist=hist),
               batches=len(tb), elapsed_s=elapsed, flashed=not a.no_flash)
    assert cnt == nsamp, (cnt, nsamp)
    json.dump(run, open(f'{pre}_run.json', 'w'), indent=1)
    fq = np.array([[tq[q]['cyc_feat'], tq[q]['cyc_pred'], tq[q].get('cyc_ad_req', 0), tq[q].get('cyc_ad_event', 0)] for q in sorted(tq)], float)
    print(f'done: {len(rows)} queries, {nsamp} samples, {elapsed:.0f} s; per-sample cycles mean {tot / max(cnt, 1):.0f} max {mx}; '
          f'per-query cycles (feat, pred, ad_req, ad_event) mean {fq.mean(0).round(0).tolist()} max {fq.max(0).tolist()}; '
          f'stack high-water {info_end[9]} B', flush=True)


if __name__ == '__main__':
    main()
