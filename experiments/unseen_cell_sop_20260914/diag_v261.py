"""V261 (PROTOCOL_DIAG_V261.md): why the implied SOC capacity of the A cells is ~0.967 x the 1C capacity table. Per SOC-move
step (1C discharge ending at F_SOC >= threshold) of every A-cell SOP file: Ah from the cycler counter, Ah from our own
integration of the raw current, the denominator C = Ah / threshold, and the discharge capacity of the previous cycle.
Reads raw/RPCWBY/*.zip only. usage: python diag_v261.py"""
import io, json, re, sys, zipfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from run_envelope import HERE, dump
sys.path.insert(0, str(HERE / 'mamba_v211'))
import build_cache_v211 as BC

OUT = HERE / 'diag_v261'
ZIP = {'Test#1': BC.RPC['RPCWBY_TEST1'], 'Test#2': BC.RPC['RPCWBY_TEST2']}
COLS = ['Test_Time(s)', 'Cycle_Index', 'Step_Index', 'Current(A)', 'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)']


def move_steps():
    """protocol -> {raw Step_Index: (threshold, set current A)} for the 1C SOC-move steps (raw index = protocol step + 1)."""
    out = {}
    for p in json.load(open(HERE / 'rpcwby_aging_sop_v84/protocol_definitions.json')):
        name = p['member'].split('/')[-1]; m = {}
        for st in p['steps']:
            thr = [float(re.search(r'F_SOC>=([0-9.]+)', l['test']).group(1)) for l in st['limits'] if re.search(r'F_SOC>=([0-9.]+)', l['test'])]
            if thr and st['control'] == 'Current(A)' and 'disch' in st['label'].lower():
                m[st['step'] + 1] = (thr[0], float(st['value']), st['label'])
        out[name] = m
    return out


def segments(step):
    ch = np.flatnonzero(np.r_[True, step[1:] != step[:-1]]); return list(zip(ch, np.r_[ch[1:] - 1, len(step) - 1]))


def scan(job):
    sheet, member, cycle, cap_table, MS = job                                      # member = path inside the zip (same as the bench member)
    d = pd.read_csv(io.BytesIO(zipfile.ZipFile(ZIP[sheet]).read(member)), usecols=lambda c: c.replace('﻿', '').strip() in COLS, low_memory=False)
    d.columns = [c.replace('﻿', '').strip() for c in d.columns]
    t = d['Test_Time(s)'].to_numpy(float); I = d['Current(A)'].to_numpy(float); st = d['Step_Index'].to_numpy(int); cy = d['Cycle_Index'].to_numpy(int)
    DC = d['Discharge_Capacity(Ah)'].to_numpy(float); CC = d['Charge_Capacity(Ah)'].to_numpy(float)
    segs = segments(st)
    # protocol = the one whose move steps appear with their set current most often
    score = {}
    for name, m in MS.items():
        score[name] = sum(1 for a, b in segs if st[a] in m and b - a >= 3 and abs(np.median(I[a:b + 1]) - m[st[a]][1]) < 0.05)
    prot = max(score, key=score.get); m = MS[prot]; rows = []
    for a, b in segs:
        if st[a] not in m or b - a < 3 or a == 0 or b + 1 >= len(t):
            continue
        thr, iset, label = m[st[a]]
        if abs(np.median(I[a:b + 1]) - iset) >= 0.05 or cy[a] != cy[b] or np.any(np.diff(DC[a:b + 1]) < -1e-9):
            continue
        ah_counter = (DC[b] - DC[a]) - (CC[b] - CC[a])                             # net discharged Ah, cycler counters, samples a..b
        ah_ours = float(-np.sum(I[a:b] * np.diff(t[a:b + 1])) / 3600.0)            # left rectangle as ah_cc over the same span (raw test time)
        rows.append(dict(sheet=sheet, member=member, cycle=cycle, protocol=prot, step=int(st[a]), label=label, threshold=thr, i_set=iset,
                         cycle_index=int(cy[a]), n=int(b - a + 1), ah_counter=float(ah_counter), ah_ours=ah_ours,
                         c_denom_counter=float(ah_counter / thr), c_denom_ours=float(ah_ours / thr)))
    R = pd.DataFrame(rows)
    f = dict(sheet=sheet, member=member, cycle=cycle, protocol=prot, protocol_score=score, cap_table=cap_table, move_steps=int(len(R)))
    if len(R):
        c0 = int(R.cycle_index.min()); f['sop_cycle_index'] = c0
        f['c_denom_counter'] = float(R.c_denom_counter.median()); f['c_denom_ours'] = float(R.c_denom_ours.median())
        f['ours_over_counter'] = float((R.ah_ours / R.ah_counter).median())
        prev = cy == c0 - 1
        if prev.any():
            sp = sorted(set(st[prev].tolist()))
            f.update(prev_cycle_dc_max=float(DC[prev].max()), prev_cycle_cc_max=float(CC[prev].max()), prev_cycle_steps=' '.join(map(str, sp)))
        same = cy == c0; f['sop_cycle_first_steps'] = ' '.join(map(str, sorted(set(st[same].tolist()))[:12]))
    return f, R


def main():
    OUT.mkdir(exist_ok=True)
    MS = move_steps(); print({k: {s: v[:2] for s, v in m.items()} for k, m in MS.items()}, flush=True)
    B = pd.read_csv(HERE / 'closedloop_v251/src/bench_rows.csv'); A = B[B.evaluation_group != 'B'][['sheet', 'member']].drop_duplicates()
    Map = pd.read_csv(HERE / 'rpcwby_aging_sop_v84/cycle_file_map.csv')[['member', 'cycle']].drop_duplicates('member')
    Cap = pd.read_csv(HERE / 'rpcwby_aging_sop_v84/table_capacity.csv')
    A = A.merge(Map, on='member', how='left')
    A['cap_table'] = [float(np.interp(r.cycle, Cap[Cap.sheet == r.sheet].sort_values('cycle').cycle, Cap[Cap.sheet == r.sheet].sort_values('cycle').capacity_Ah)) for r in A.itertuples()]
    names = {sh: set(zipfile.ZipFile(zp).namelist()) for sh, zp in ZIP.items()}
    A['in_zip'] = [m in names[sh] for sh, m in zip(A.sheet, A.member)]; print('files', len(A), 'found in zip', int(A.in_zip.sum()), flush=True)
    jobs = [(r.sheet, r.member, int(r.cycle), float(r.cap_table), MS) for r in A.itertuples() if r.in_zip]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(scan, jobs))
    F = pd.DataFrame([f for f, _ in res]); S = pd.concat([r for _, r in res if len(r)], ignore_index=True)
    F['denom_over_table'] = F.c_denom_counter / F.cap_table
    F['denom_over_prev_dc'] = F.c_denom_counter / F.prev_cycle_dc_max if 'prev_cycle_dc_max' in F else np.nan
    V = pd.read_csv(HERE / 'joint_v260/implied_capacity.csv')[['member', 'q_impl']]
    F = F.merge(V, on='member', how='left')
    F['denom_over_q_impl'] = F.c_denom_counter / F.q_impl
    S.to_csv(OUT / 'steps.csv', index=False); F.drop(columns=['protocol_score']).to_csv(OUT / 'files.csv', index=False)
    q = lambda s: dict(n=int(s.notna().sum()), median=float(s.median()), min=float(s.min()), max=float(s.max())) if s.notna().any() else None
    summ = dict(protocols=F.protocol.value_counts().to_dict(), steps=int(len(S)),
                H1_ours_over_counter=q(F.ours_over_counter), H2_denom_over_table=q(F.denom_over_table), denom_over_q_impl=q(F.denom_over_q_impl),
                H3_denom_over_prev_cycle_dc=q(F.denom_over_prev_dc))
    summ['H1_rejected'] = bool(summ['H1_ours_over_counter'] and abs(summ['H1_ours_over_counter']['median'] - 1) <= 0.005)
    h2 = summ['H2_denom_over_table']; dq = summ['denom_over_q_impl']
    summ['H2_accepted'] = bool(h2 and 0.965 <= h2['median'] <= 0.970 and dq and abs(dq['median'] - 1) <= 0.01)
    h3 = summ['H3_denom_over_prev_cycle_dc']; summ['H3_accepted'] = bool(h3 and abs(h3['median'] - 1) <= 0.01)
    dump(OUT / 'summary.json', summ); print(json.dumps(summ, indent=1), flush=True)
    pd.set_option('display.width', 250)
    print(F.sort_values(['sheet', 'cycle'])[['sheet', 'cycle', 'protocol', 'move_steps', 'cap_table', 'c_denom_counter', 'q_impl', 'denom_over_table',
                                            'ours_over_counter', 'prev_cycle_dc_max', 'prev_cycle_cc_max', 'prev_cycle_steps']].round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
