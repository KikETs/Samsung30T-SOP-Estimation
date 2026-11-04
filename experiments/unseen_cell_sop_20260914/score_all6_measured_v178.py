"""V178: train on all six UYPYDJ cells (+Mendeley; V164 ALL6 models), test on the RPCWBY measured SOP
(main evaluation). Raw constant-power commands, no calibration yet. Groups follow the measured-track gate (V100/V127):
A = Test#1/#2 10 s (sheet x direction x temperature), B = Test#3-2024 discharge last-occurrence pulses (temperature x horizon; 2/10 s goal,
30 s diagnostic). Only voltage-limited truths are judged; exceed = predicted - truth > 1e-9 W; pass = 0 exceed and median >= 80%."""
import json
from multiprocessing import Pool
import joblib, numpy as np, pandas as pd, torch
from run_envelope import HERE, digest, dump
from bounded_reserve_v57 import Encoder
from constant_power_v85 import sop
from evaluate_rpcwby_aging_v85 import LIMITS, INP as V85INP

FITS = HERE / 'all_six_training_v164'; OUT = HERE / 'all6_measured_v178'
V46 = HERE / 'rpcwby_external_v46'; C45P = HERE / 'rpcwby_protocol_audit_v45/cp_pulses.csv'
B_LIMIT = dict(cap=30., limit=2.55)


def params(fit_id, X, hts, his, offsets, idx):
    m = Encoder(74).eval(); m.load_state_dict(torch.load(FITS / fit_id / 'model.pt', map_location='cpu', weights_only=True)); sc = joblib.load(FITS / fit_id / 'scaler.joblib')
    out = np.empty((len(idx), 10))
    with torch.no_grad():
        for k, j in enumerate(idx):
            a, b = offsets[j], offsets[j + 1]
            out[k] = m(torch.tensor(sc.transform(X[j][None])), torch.tensor(hts[a:b][None], dtype=torch.float32), torch.tensor(his[a:b][None], dtype=torch.float32)).numpy()[0]
    return out


def task(a):
    p, vpre, pre, gap, h, d, cap, lim = a
    r = sop(np.asarray(p), float(vpre), float(pre), float(gap), float(h), d, cap, lim)
    return r['power_W'], bool(r['zero_infeasible'])


def main():
    OUT.mkdir(exist_ok=False); torch.set_num_threads(4)
    sources = {str(p): digest(p) for p in [FITS / f'{d}__ALL6' / n for d in ('discharge', 'charge') for n in ('model.pt', 'scaler.joblib')]}
    sources.update({str(HERE / n): digest(HERE / n) for n in ('score_all6_measured_v178.py', 'constant_power_v85.py', 'evaluate_rpcwby_aging_v85.py')})
    inp = json.loads((V85INP / 'audit.json').read_text())
    for n, h in inp['output_hashes'].items(): assert digest(V85INP / n) == h, n
    R = pd.read_csv(V85INP / 'inputs.csv'); Z = np.load(V85INP / 'inputs.npz'); X = Z['features']; H = (Z['history_time_s'], Z['history_current_A'], Z['history_offsets'])
    R = R[R.table_linked & R.temp_block_C.isin([25, 10])]
    I46 = pd.read_csv(V46 / 'inputs.csv'); Z46 = np.load(V46 / 'inputs.npz'); C45 = pd.read_csv(C45P)
    assert len(I46) == len(C45) == 356 and (I46[['member', 'first_row', 'last_row']].to_numpy() == C45[['member', 'first_row', 'last_row']].to_numpy()).all()
    X46 = Z46['features']; H46 = (Z46['history_time_s'], Z46['history_current_A'], Z46['history_offsets']); assert X46.shape[1] == 74
    recs = []
    with Pool(14) as pool:
        for d in ('discharge', 'charge'):
            q = R[R.direction == d]; idx = q.row_id.to_numpy(); lim = LIMITS[d]
            p = params(f'{d}__ALL6', X, *H, idx)
            res = pool.map(task, [(p[k], X[j, 0], X[j, 72], X[j, 73], 10., d, lim['cap'], lim['limit']) for k, j in enumerate(idx)], chunksize=8)
            for (_, r), (pw, z) in zip(q.iterrows(), res):
                recs.append(dict(evaluation_group='A1' if r.sheet == 'Test#1' else 'A17', sheet=r.sheet, direction=d, temp_C=int(r.temp_block_C), horizon_s=10.,
                                 row=int(r.row_id), soc_point=r.soc_point, truth_power_W=r.truth_power_W, truth_end_current_A=r.truth_end_current_A,
                                 regime='current_limited' if r.truth_end_current_A >= lim['cap'] - .1 else 'voltage_limited', raw_power_W=pw, raw_zero=z))
        last = np.flatnonzero(I46.last_occurrence.to_numpy(bool)); c = C45.iloc[last]
        temp = np.where(np.isfinite(c.filename_temp_C.to_numpy(float)), c.filename_temp_C.to_numpy(float), 25.)
        p = params('discharge__ALL6', X46, *H46, last)
        res = pool.map(task, [(p[k], X46[j, 0], X46[j, 72], X46[j, 73], float(c.horizon_s.iloc[k]), 'discharge', B_LIMIT['cap'], B_LIMIT['limit'])
                              for k, j in enumerate(last)], chunksize=4)
        for k, (pw, z) in enumerate(res):
            r = c.iloc[k]; iend = abs(float(r.current_A)); vend = float(r.voltage_V)
            recs.append(dict(evaluation_group='B', sheet='Test#3_2024', direction='discharge', temp_C=int(temp[k]), horizon_s=float(r.horizon_s), row=int(last[k]),
                             soc_point=float(r.soc), truth_power_W=abs(float(r.power_W)), truth_end_current_A=iend,
                             regime='current_limited' if iend >= 29.9 else ('voltage_limited' if abs(vend - 2.55) <= .04 else 'unconverged'), raw_power_W=pw, raw_zero=z))
    P = pd.DataFrame(recs); P['raw_over_W'] = P.raw_power_W - P.truth_power_W; P['raw_usable_pct'] = 100 * P.raw_power_W / P.truth_power_W
    P.to_csv(OUT / 'predictions.csv', index=False)
    V = P[P.regime == 'voltage_limited']
    S = V.groupby(['evaluation_group', 'sheet', 'direction', 'temp_C', 'horizon_s']).agg(n=('row', 'size'), exceed=('raw_over_W', lambda s: int((s > 1e-9).sum())),
                                                                                     median_usable_pct=('raw_usable_pct', 'median'), zero_outputs=('raw_zero', 'sum')).reset_index()
    S['goal'] = (S.evaluation_group != 'B') | S.horizon_s.isin([2., 10.]); S['pass'] = (S.exceed == 0) & (S.median_usable_pct >= 80)
    S.to_csv(OUT / 'groups.csv', index=False)
    G = S[S.goal]
    res = dict(scheme='train all six UYPYDJ (+Mendeley), test RPCWBY measured SOP', method='raw (no calibration)', goal_groups=len(G), goal_rows=int(G.n.sum()),
               groups_passing=int(G['pass'].sum()), exceed_total=int(G.exceed.sum()), groups_at_80=int((G.median_usable_pct >= 80).sum()),
               A=dict(groups=int((G.evaluation_group != 'B').sum()), passing=int(G[G.evaluation_group != 'B']['pass'].sum()), exceed=int(G[G.evaluation_group != 'B'].exceed.sum())),
               B=dict(groups=int((G.evaluation_group == 'B').sum()), passing=int(G[G.evaluation_group == 'B']['pass'].sum()), exceed=int(G[G.evaluation_group == 'B'].exceed.sum())),
               comparator='V127 5/15', source_hashes=sources, adopted=False, goal_complete=False)
    dump(OUT / 'verification.json', res)
    pd.set_option('display.width', 200); print(S.round(1).to_string(index=False)); print(json.dumps({k: v for k, v in res.items() if k != 'source_hashes'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
