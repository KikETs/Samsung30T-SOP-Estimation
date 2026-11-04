"""Independent recount (check 6). Standard library only; imports neither the predictor nor the copied evaluation
functions. From results/requests.csv, inputs/bench_rows.csv and inputs/ad_trajectory.csv it
  1. recomputes every score of results/primary.csv and results/groups.csv,
  2. re-derives the tau = 5 % starting scales from the saved central predictions (other two cells only),
  3. replays the fixed and adapted request sequences in the processing order and compares them with the saved
     requests,
  4. verifies results/output_hashes.json.
"""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
CELLS = ['A1', 'A17', 'B']


def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def quantile(values, q):
    """Linear interpolation between order statistics (numpy's default)."""
    v = sorted(values)
    if not v:
        return 0.0
    pos = (len(v) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def main():
    bench = {r['row_id']: r for r in read(ROOT / 'inputs/bench_rows.csv')}
    key = {k: (r['evaluation_group'], r['direction'], float(r['temp_C']), float(r['horizon_s'])) for k, r in bench.items()}
    truth = {k: float(r['truth_power_W']) for k, r in bench.items()}
    ad = read(ROOT / 'inputs/ad_trajectory.csv')
    order = {c: [r['row_id'] for r in sorted((x for x in ad if x['cell'] == c), key=lambda x: int(x['order']))] for c in CELLS}
    requests = read(OUT / 'requests.csv')
    primary = read(OUT / 'primary.csv')
    groups_rep = read(OUT / 'groups.csv')
    errors, arms = [], []
    for rep in primary:
        cand, pol = rep['candidate'], rep['policy']
        rr = [r for r in requests if r['candidate'] == cand and r['policy'] == pol]
        ids = [r['row_id'] for r in rr]
        if not (len(rr) == 672 and len(set(ids)) == 672 and set(ids) == set(bench)):
            errors.append([cand, pol, 'row set'])
            continue
        req = {r['row_id']: float(r['request']) for r in rr}
        central = {r['row_id']: float(r['central']) for r in rr}
        ratios = {k: req[k] / truth[k] for k in req}
        # 1. scores
        exceed = sum(1 for v in ratios.values() if v > 1 + 1e-9)
        util = statistics.median(min(v, 1.0) for v in ratios.values())
        grouped = defaultdict(list)
        for k, v in ratios.items():
            grouped[key[k]].append(v)
        passed = strict = 0
        gstats = {}
        for g, a in grouped.items():
            e = sum(1 for v in a if v > 1 + 1e-9)
            u = statistics.median(min(v, 1.0) for v in a)
            p = u >= .8 and e / len(a) <= .1
            s = e == 0 and statistics.median(a) >= .8
            passed += p
            strict += s
            gstats[g] = (len(a), e, u, p, s)
        depths = [v - 1 for v in ratios.values() if v > 1 + 1e-9]
        actual = dict(exceed_rows=exceed, int_rate=exceed / 672, util_q50=util, closed_loop=passed, original=strict,
                      depth_max=max(depths) if depths else 0.0, depth_p95=quantile(depths, .95) if depths else 0.0)
        for f, v in actual.items():
            if abs(v - float(rep[f])) > 1e-9 * max(1.0, abs(v)):
                errors.append([cand, pol, f, v, rep[f]])
        for g in (x for x in groups_rep if x['candidate'] == cand and x['policy'] == pol):
            k = (g['cell'], g['direction'], float(g['temp_C']), float(g['horizon_s']))
            n, e, u, p, s = gstats[k]
            if (n, e, p, s) != (int(g['n']), int(g['exceed']), g['passed'] == 'True', g['strict'] == 'True') or abs(u - float(g['util_q50'])) > 1e-12:
                errors.append([cand, pol, 'group', k])
        # 2. starting scales at tau = 5 % from the other two cells
        scales = {}
        for c in CELLS:
            idx = [k for d in CELLS if d != c for k in order[d] if math.isfinite(central[k]) and central[k] > 0]
            best = 0.2
            for j in range(20, 151):
                s = j / 100
                frac = sum(1 for k in idx if s * central[k] > truth[k] + 1e-9) / len(idx)
                if frac <= .05:
                    best = s
            scales[c] = best
            if abs(best - float(rep[f'scale_{c}'])) > 1e-12:
                errors.append([cand, pol, 'scale', c, best, rep[f'scale_{c}']])
        # 3. replay of the request sequence (fixed or adapted, tau* = 5 %)
        worst = 0.0
        for c in CELLS:
            logs = math.log(scales[c])
            for k in order[c]:
                cen = central[k] if math.isfinite(central[k]) else 0.0
                r = math.exp(logs) * cen
                worst = max(worst, abs(r - req[k]))
                if pol == 'AD':
                    e = 1.0 if r / truth[k] > 1 + 1e-9 else 0.0
                    logs = min(max(logs + .05 * (.05 - e), math.log(.2)), math.log(1.5))
        if worst > 1e-9:
            errors.append([cand, pol, 'replay', worst])
        arms.append(dict(candidate=cand, policy=pol, **actual, scales=scales, replay_max_abs_diff_W=worst))
    hashes = json.loads((OUT / 'output_hashes.json').read_text(encoding='utf-8'))
    for name, h in hashes.items():
        if hashlib.sha256((OUT / name).read_bytes()).hexdigest() != h:
            errors.append(['hash', name])
    result = dict(arms=len(arms), request_rows=len(requests), errors=errors, verified_output_hashes=len(hashes),
                  passed=not errors, scores=arms)
    (OUT / 'independent_validation.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result, indent=1))
    assert not errors


if __name__ == '__main__':
    main()
