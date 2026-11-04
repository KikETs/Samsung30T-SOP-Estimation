"""V281 K1f (PROTOCOL_PRESUBMIT_V281_K1F.md): copy of V276 inv_run.py with one change, the job list: R0, T1, T3 on the
five sample sweeps (15 jobs), no stage argument. One process group per job, at most --workers at a time, a hard
wall-clock limit per job (SIGKILL to the group), and a memory watch (MemAvailable < 10 GB: every running group is killed
and the run stops). Started by run_k1f_remote.sh with the V279 step-8 environment, from rebuild_v279/aging_refit/scripts.
Output: inv_k1f/out/<variant>/parts/<job>.csv (+ .log) and out/<variant>/pred.csv (merged)."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools' / 'inv_ident.py'
SAMPLE = {5: 4, 40: 5, 73: 4, 92: 5, 127: 8}          # sweep -> rows (protocol 3)


def mem_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) / 1048576
    return 1e9


def jobs():
    return [(v, s, None) for v in ('R0', 'T1', 'T3') for s in SAMPLE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=15)
    ap.add_argument('--timeout', type=float, default=5400.0)
    a = ap.parse_args()
    todo = jobs(); running = {}; done = []; killed = []; t0 = time.time()
    while todo or running:
        if mem_gb() < 10:
            for name, (p, ts, log) in running.items():
                os.killpg(p.pid, signal.SIGKILL); p.wait(); log.close()
            print(f'{time.strftime("%T")} MemAvailable < 10 GB: every job killed, run stopped', flush=True)
            sys.exit(2)
        while todo and len(running) < a.workers:
            v, s, k = todo.pop(0)
            name = f'{v}_{s}' + (f'_k{k}' if k else '')
            part = ROOT / 'out' / v / 'parts' / f'{name}.csv'; part.parent.mkdir(parents=True, exist_ok=True)
            log = open(part.with_suffix('.log'), 'w')
            cmd = [sys.executable, str(TOOL), '--variant', v, '--sweep', str(s), '--out', str(part)] + (['--k', str(k)] if k else [])
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            running[name] = (p, time.time(), log)
        time.sleep(2)
        for name, (p, ts, log) in list(running.items()):
            if p.poll() is not None:
                log.close(); del running[name]; done.append((name, p.returncode))
                print(f'{time.strftime("%T")} {name} exit {p.returncode} in {time.time() - ts:.0f} s; {len(done)} done, {len(killed)} killed, {time.time() - t0:.0f} s', flush=True)
            elif time.time() - ts > a.timeout:
                os.killpg(p.pid, signal.SIGKILL); p.wait(); log.close(); del running[name]; killed.append(name)
                print(f'{time.strftime("%T")} {name} KILLED at the {a.timeout:.0f} s limit', flush=True)
    for v in sorted({j[0] for j in jobs()}):
        parts = sorted((ROOT / 'out' / v / 'parts').glob('*.csv'))
        if parts:
            pd.concat([pd.read_csv(x) for x in parts], ignore_index=True).to_csv(ROOT / 'out' / v / 'pred.csv', index=False, float_format='%.17g')
    bad = [n for n, rc in done if rc != 0]
    print(f'EXIT k1f: {len(done)} jobs done, {len(bad)} non-zero exits {bad}, {len(killed)} killed {killed}', flush=True)


if __name__ == '__main__':
    main()
