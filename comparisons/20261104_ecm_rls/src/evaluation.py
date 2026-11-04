"""Benchmark rows, processing order, groups, and the study's calibrate / request / score functions.

calibrate, request and score are taken verbatim (byte-identical source text) from the pilot script
comparisons/20261014_baseline_pilot/run_pilot.py, copied to inputs/run_pilot_source.py and pinned by SHA-256.
They are extracted with ast and executed in a namespace holding only numpy and CELLS.
"""
import ast
import csv
import hashlib
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
PILOT = EXP / 'inputs/run_pilot_source.py'
PILOT_SHA256 = 'fe7e60ce5c461694b5dea801a20148a9988682d451825a360f12ad53df94a15e'
BENCH_SHA256 = '1b316c7aada2731ce5270e86e4377945cfb97262e21500f0d676c71560f74841'
AD_SHA256 = 'b86b43a432845d7dac51e2cca2a8adec43cbe9ffdd53a708aeb41f847a82369d'
CELLS = ['A1', 'A17', 'B']
TAUS = np.round(np.arange(0, .20001, .0025), 4)
ID_COLUMNS = ('row_id', 'evaluation_group', 'member', 'direction', 'h0_row', 'q_row', 'horizon_s')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def pilot_functions():
    """Return ({name: function}, {name: source text}) for calibrate, request, score (+ the pilot's 1RC solver)."""
    raw = PILOT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PILOT_SHA256, 'run_pilot.py copy changed'
    src = raw.decode('utf-8')
    tree = ast.parse(src)
    cells = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == 'CELLS' for t in n.targets)]
    assert len(cells) == 1 and ast.literal_eval(cells[0].value) == CELLS
    seg = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    names = ('calibrate', 'request', 'score', 'simulate_feasible', 'rc_power')
    ns = {'np': np, 'CELLS': list(CELLS)}
    for name in names:
        exec(compile(seg[name], f'run_pilot.py:{name}', 'exec'), ns)
    return {k: ns[k] for k in names}, {k: seg[k] for k in names}


def load_benchmark():
    """Rows in bench_rows.csv order, reference SOP y, per-cell processing orders and the 15 groups."""
    assert sha256(EXP / 'inputs/bench_rows.csv') == BENCH_SHA256
    assert sha256(EXP / 'inputs/ad_trajectory.csv') == AD_SHA256
    rows = read_csv(EXP / 'inputs/bench_rows.csv')
    ad = read_csv(EXP / 'inputs/ad_trajectory.csv')
    assert len(rows) == len(ad) == 672 and len({r['row_id'] for r in rows}) == 672
    assert {r['row_id'] for r in rows} == {r['row_id'] for r in ad}
    ids = {r['row_id']: i for i, r in enumerate(rows)}
    orders = {c: np.array([ids[r['row_id']] for r in sorted([q for q in ad if q['cell'] == c], key=lambda q: int(q['order']))])
              for c in CELLS}
    assert sorted(np.concatenate(list(orders.values())).tolist()) == list(range(672))
    for c in CELLS:
        assert all(rows[i]['evaluation_group'] == c for i in orders[c])
    groups = {}
    for i, r in enumerate(rows):
        key = (r['evaluation_group'], r['direction'], float(r['temp_C']), float(r['horizon_s']))
        groups.setdefault(key, []).append(i)
    assert len(groups) == 15
    y = np.array([float(r['truth_power_W']) for r in rows])
    assert np.all(y > 0)
    ad_by = {r['row_id']: r for r in ad}
    truth_ad = np.array([float(ad_by[r['row_id']]['truth']) for r in rows])
    return dict(rows=rows, ad=ad, ad_by=ad_by, ids=ids, orders=orders, groups=groups, y=y,
                truth_matches_ad=float(np.max(np.abs(truth_ad - y))))


def main_reproduction(B, F):
    """Main configuration from ad_trajectory.csv: central, start scales 0.76/0.82/0.90, adaptation at tau 5 %."""
    rows, y, orders, groups, ad_by = B['rows'], B['y'], B['orders'], B['groups'], B['ad_by']
    central = np.array([float(ad_by[r['row_id']]['central']) for r in rows])
    stored = np.array([float(ad_by[r['row_id']]['request']) for r in rows])
    scales = dict(A1=.76, A17=.82, B=.9)
    out = {}
    for adaptive in (True, False):
        pred = F['request'](central, y, orders, scales, .05, adaptive)
        sm, G = F['score'](pred, y, groups)
        out['AD' if adaptive else 'fixed'] = dict(score=sm, groups=G, pred=pred)
    ad_pred = out['AD']['pred']
    out['AD_request_max_abs_diff_vs_stored'] = float(np.max(np.abs(ad_pred - stored)))
    s = out['AD']['score']
    out['gate_passed'] = bool((s['closed_loop'], s['original'], s['exceed_rows']) == (11, 4, 28))
    out['central'] = central
    return out
