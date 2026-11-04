"""Raw-data alignment of the 672 benchmark rows (evaluator side; runs before any model or score).

Acceptance of a row-index convention requires, for every row:
  a. every sample q..L has the current sign of the stated direction and |I| >= 0.02 A
  b. q..L lie in one (Cycle_Index, Step_Index); sample q-1 is in another step or at rest
  c. Step_Time at L is within 0.1 s of the stated duration h (the logger may skip samples inside a step)
  d. constant power: |V*I| of every sample q..L within 2 % of |V*I| at L
  e. |V*I| at L equals truth_power_W within 0.5 % or 0.05 W
  f. V at L within 0.04 V of the stated limit (2.55 V discharge, 4.15 V charge)
Conventions tested: C0 = 0-based data line (header excluded), C1 = 1-based data line, C2 = Data_Point value.
Also: history excerpt cross-check, sweep structure, full-charge start at h0, sampling inside [h0, q), pulse inventory.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
CACHE = EXP / 'cache'
OUT = EXP / 'results'
REST_A = 0.02


def cache_path(member):
    return CACHE / (member.replace('/', '__').replace('#', '_') + '.npz')


_D = {}


def raw(member):
    if member not in _D:
        d = np.load(cache_path(member))
        _D[member] = {k: d[k] for k in d.files}
    return _D[member]


def index_of(conv, D, r):
    if conv == 'C0':
        return r
    if conv == 'C1':
        return r - 1
    k = np.flatnonzero(D['dp'] == r)
    return int(k[0]) if len(k) == 1 else None


def check_row(conv, r):
    D = raw(r['member'])
    n = len(D['t'])
    iq, iL = index_of(conv, D, int(r['q_row'])), index_of(conv, D, int(r['last_row']))
    res = dict(row_id=r['row_id'], convention=conv)
    if iq is None or iL is None or not (1 <= iq <= iL < n):
        res.update(index_ok=False)
        return res
    dis = r['direction'] == 'discharge'
    lim = 2.55 if dis else 4.15
    h = float(r['horizon_s'])
    truth = float(r['truth_power_W'])
    sl = slice(iq, iL + 1)
    I, V = D['I'][sl], D['V'][sl]
    P = np.abs(V * I)
    a = bool(np.all(I <= -REST_A) if dis else np.all(I >= REST_A))
    same = (D['cyc'][sl] == D['cyc'][iq]) & (D['step'][sl] == D['step'][iq])
    b = bool(same.all() and ((D['cyc'][iq - 1], D['step'][iq - 1]) != (D['cyc'][iq], D['step'][iq])
                             or abs(D['I'][iq - 1]) < REST_A))
    c = bool(abs(D['st'][iL] - h) <= 0.1)
    d = bool(np.max(np.abs(P / P[-1] - 1)) <= 0.02) if P[-1] > 0 else False
    e = bool(abs(P[-1] - truth) <= max(0.005 * truth, 0.05))
    f = bool(abs(V[-1] - lim) <= 0.04)
    res.update(index_ok=True, a_sign=a, b_one_step=b, c_duration=c, d_constant_power=d, e_power=e, f_voltage=f,
               all_ok=a and b and c and d and e and f, n_samples=int(iL - iq + 1), step_time_last=float(D['st'][iL]),
               step_time_first=float(D['st'][iq]), power_last=float(P[-1]), truth=truth,
               power_rel_err=float(abs(P[-1] - truth) / truth), v_last=float(V[-1]), v_err=float(abs(V[-1] - lim)),
               max_power_dev=float(np.max(np.abs(P / P[-1] - 1))) if P[-1] > 0 else np.nan,
               gap_prev_s=float(D['t'][iq] - D['t'][iq - 1]))
    return res


def full_charges(D):
    """Maximal runs of I >= 0.02 A lasting >= 1800 s whose voltage reaches >= 4.15 V: (first, last) indices."""
    chg = D['I'] >= REST_A
    out = []
    k = 0
    n = len(chg)
    edges = np.flatnonzero(np.diff(np.r_[0, chg.astype(np.int8), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        e -= 1
        if D['t'][e] - D['t'][s] >= 1800 and D['V'][s:e + 1].max() >= 4.15:
            out.append((int(s), int(e)))
    return out


def pulses(D, lo, hi):
    """Complete step runs inside [lo, hi): one (cyc, step), |I| >= 0.02 A of one sign, Step_Time at the last sample
    in [0.5, 12.5] s, preceded by a sample of another step or at rest. Returns (first, last, sign, step_time_last)."""
    I = D['I'][lo:hi]
    key = D['cyc'][lo:hi] * 1000000 + D['step'][lo:hi]
    load = np.abs(I) >= REST_A
    out = []
    j = 0
    m = len(I)
    while j < m:
        if not load[j]:
            j += 1
            continue
        s = j
        while j + 1 < m and load[j + 1] and key[j + 1] == key[s] and np.sign(I[j + 1]) == np.sign(I[s]):
            j += 1
        e = j
        j += 1
        if e == m - 1:
            continue                       # may continue at or after hi: not complete inside the window
        stl = D['st'][lo + e]
        prev_other = s == 0 or not load[s - 1] or key[s - 1] != key[s]
        if 0.5 <= stl <= 12.5 and prev_other:
            out.append((lo + s, lo + e, int(np.sign(I[s])), float(stl)))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    with open(EXP / 'inputs/bench_rows.csv', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    conv_summary = {}
    per_row = {}
    for conv in ('C0', 'C1', 'C2'):
        res = [check_row(conv, r) for r in rows]
        per_row[conv] = res
        ok = [x for x in res if x['index_ok']]
        conv_summary[conv] = dict(index_ok=len(ok), **{k: int(sum(x[k] for x in ok)) for k in
                                  ('a_sign', 'b_one_step', 'c_duration', 'd_constant_power', 'e_power', 'f_voltage',
                                   'all_ok')})
    accepted = [c for c in ('C0', 'C1', 'C2') if conv_summary[c]['all_ok'] == len(rows)]
    C0 = per_row['C0']
    with open(OUT / 'alignment_rows.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(C0[0]))
        w.writeheader()
        w.writerows(C0)
    # excerpt cross-check (row rpcwby:Test#1:1219)
    with open(EXP / 'inputs/history_A1_excerpt.csv', encoding='utf-8-sig', newline='') as f:
        ex = list(csv.DictReader(f))
    r1219 = next(r for r in rows if r['row_id'] == 'rpcwby:Test#1:1219')
    D = raw(r1219['member'])
    fr = np.array([int(x['file_row']) for x in ex])
    t_rel = np.array([float(x['time_s']) for x in ex])
    cur = np.array([float(x['current_A']) for x in ex])
    vol = np.array([float(x['voltage_V']) for x in ex])
    tem = np.array([float(x['temperature_C']) for x in ex])
    excerpt = dict(row_id=r1219['row_id'], samples=len(ex), file_row_min=int(fr.min()), file_row_max=int(fr.max()),
                   h0_row=int(r1219['h0_row']), q_row=int(r1219['q_row']),
                   current_exact=bool(np.array_equal(cur, D['I'][fr])),
                   voltage_exact=bool(np.array_equal(vol, D['V'][fr])),
                   temperature_exact=bool(np.array_equal(tem, D['T'][fr])),
                   time_max_abs_diff_s=float(np.max(np.abs(t_rel - (D['t'][fr] - D['t'][fr[0]])))),
                   alt_C1_current_exact=bool(np.array_equal(cur, D['I'][fr - 1])),
                   alt_C1_voltage_exact=bool(np.array_equal(vol, D['V'][fr - 1])))
    excerpt['match'] = bool(excerpt['current_exact'] and excerpt['voltage_exact'] and excerpt['temperature_exact']
                            and excerpt['time_max_abs_diff_s'] <= 1e-6)
    # sweep structure: groups of rows sharing (member, h0_row)
    groups = defaultdict(list)
    for r in rows:
        groups[(r['member'], int(r['h0_row']))].append(r)
    sweep_rows = []
    by_member = defaultdict(list)
    for (m, h0), g in groups.items():
        g = sorted(g, key=lambda r: int(r['q_row']))
        D = raw(m)
        fc = full_charges(D)
        start_fc = [s for s, e in fc if s <= int(g[0]['q_row'])]
        last_fc_start = max(start_fc) if start_fc else None
        inner_fc = [s for s, e in fc if h0 < s < int(g[-1]['q_row'])]
        earlier_pulses_inside = all(int(g[i]['last_row']) < int(g[i + 1]['q_row']) for i in range(len(g) - 1))
        q_last = int(g[-1]['q_row'])
        dt = np.diff(D['t'][h0:q_last])
        sweep_rows.append(dict(member=m, h0_row=h0, n_rows=len(g), directions='/'.join(sorted({r['direction'] for r in g})),
                               cell=g[0]['evaluation_group'], q_first=int(g[0]['q_row']), q_last=q_last,
                               h0_is_last_full_charge_start=last_fc_start == h0, full_charges_inside=len(inner_fc),
                               h0_current_A=float(D['I'][h0]), prev_sample_rest=bool(abs(D['I'][h0 - 1]) < REST_A),
                               earlier_query_pulses_end_before_next_query=earlier_pulses_inside,
                               negative_dt_in_window=int((dt < 0).sum()), zero_dt_in_window=int((dt == 0).sum()),
                               window_samples=q_last - h0, window_hours=float((D['t'][q_last - 1] - D['t'][h0]) / 3600),
                               dt_median=float(np.median(dt)), dt_p99=float(np.quantile(dt, .99)), dt_max=float(dt.max())))
        by_member[m].append((h0, int(g[0]['q_row']), q_last, int(g[-1]['last_row'])))
    interleave = 0
    for m, lst in by_member.items():
        lst.sort()
        for a, b in zip(lst, lst[1:]):
            if not a[3] < b[0]:
                interleave += 1
    with open(OUT / 'sweeps.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0]))
        w.writeheader()
        w.writerows(sweep_rows)
    # pulse inventory for the causal voltage check: last complete same-direction pulse in [h0, q)
    inv = []
    for r in rows:
        D = raw(r['member'])
        h0, q = int(r['h0_row']), int(r['q_row'])
        sgn = -1 if r['direction'] == 'discharge' else 1
        ps = [p for p in pulses(D, h0, q) if p[2] == sgn and p[3] >= 1.5]
        last = ps[-1] if ps else None
        inv.append(dict(row_id=r['row_id'], cell=r['evaluation_group'], direction=r['direction'],
                        n_same_direction_pulses=len(ps), check_pulse_first=last[0] if last else '',
                        check_pulse_last=last[1] if last else '', check_pulse_step_time=last[3] if last else '',
                        is_previous_query_pulse=bool(last and any(x['member'] == r['member'] and int(x['q_row']) == last[0]
                                                                   for x in rows))))
    with open(OUT / 'check_pulses.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(inv[0]))
        w.writeheader()
        w.writerows(inv)
    st = np.array([x['check_pulse_step_time'] for x in inv if x['check_pulse_step_time'] != ''], dtype=float)
    summary = dict(conventions=conv_summary, accepted=accepted,
                   accepted_definition='row r = 0-based position of a data line in the member CSV, header excluded'
                   if accepted == ['C0'] else None,
                   C0_max_power_rel_err=max(x['power_rel_err'] for x in C0),
                   C0_max_v_err=max(x['v_err'] for x in C0),
                   C0_max_power_dev_within_pulse=max(x['max_power_dev'] for x in C0),
                   C0_step_time_last_range=[min(x['step_time_last'] for x in C0), max(x['step_time_last'] for x in C0)],
                   C0_step_time_first_quantiles=np.quantile([x['step_time_first'] for x in C0], [0, .01, .5, .99, 1]).tolist(),
                   C0_gap_prev_quantiles=np.quantile([x['gap_prev_s'] for x in C0], [0, .01, .5, .99, 1]).tolist(),
                   C0_n_samples_quantiles=np.quantile([x['n_samples'] for x in C0], [0, .01, .5, .99, 1]).tolist(),
                   excerpt=excerpt,
                   sweeps=dict(n_groups=len(groups), rows_per_group=np.quantile([len(g) for g in groups.values()], [0, .5, 1]).tolist(),
                               mixed_direction_groups=sum('/' in s['directions'] for s in sweep_rows),
                               h0_is_last_full_charge_start=sum(s['h0_is_last_full_charge_start'] for s in sweep_rows),
                               groups_with_full_charge_inside=sum(s['full_charges_inside'] > 0 for s in sweep_rows),
                               interleaving_group_pairs=interleave,
                               earlier_pulses_end_before_next_query=sum(s['earlier_query_pulses_end_before_next_query'] for s in sweep_rows),
                               negative_dt_in_windows=sum(s['negative_dt_in_window'] for s in sweep_rows),
                               window_samples_quantiles=np.quantile([s['window_samples'] for s in sweep_rows], [0, .5, 1]).tolist(),
                               window_hours_quantiles=np.quantile([s['window_hours'] for s in sweep_rows], [0, .5, 1]).tolist(),
                               dt_median_range=[min(s['dt_median'] for s in sweep_rows), max(s['dt_median'] for s in sweep_rows)],
                               dt_max_max=max(s['dt_max'] for s in sweep_rows)),
                   check_pulses=dict(rows_with_pulse=int(sum(x['n_same_direction_pulses'] > 0 for x in inv)),
                                     rows_without_pulse=int(sum(x['n_same_direction_pulses'] == 0 for x in inv)),
                                     pulse_is_previous_query=int(sum(x['is_previous_query_pulse'] for x in inv)),
                                     step_time_quantiles=np.quantile(st, [0, .05, .5, .95, 1]).tolist() if len(st) else None),
                   cache_manifest_sha256=hashlib.sha256((CACHE / 'manifest.json').read_bytes()).hexdigest())
    (OUT / 'alignment.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
