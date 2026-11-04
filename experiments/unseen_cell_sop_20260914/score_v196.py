"""V196 scoring: M3s (SOC-step pair transfer) for every row (fallback V193 M2), margins from the V196 bench (PROTOCOL_SOC_STEP_PAIRS_V196.md)."""
import json
from multiprocessing import Pool
import numpy as np, pandas as pd, torch, joblib
from run_envelope import HERE, dump
from gain_anchored_v75 import Encoder, gain_anchor
from constant_power_v85 import sop
from relax_state_v36 import STATE_NAMES
import bench_m3_v192 as B3

OUT = HERE / 'soc_step_pairs_v196' / 'scoring'; FITS = HERE / 'm2_all6_v193'; NM = list(STATE_NAMES); TC = NM.index('pre_temperature_5s_C')
LIMIT = dict(discharge=2.55, charge=4.15); CAP = dict(discharge=30., charge=15.)


BS = pd.read_csv(HERE / 'soc_step_pairs_v196/bench_summary.csv'); BS = BS[BS.method == 'M3s']
def bs(bench, d, region, hh):
    r = BS[(BS.bench == bench) & (BS.d == d) & (BS.region == region) & (BS.h == hh)]; return max(0., float(r.p95.iloc[0]) / 1000.)
def m3_warm_margin(d, h, preV):
    hh = 2. if h == 2. else 10.
    if d == 'discharge' and preV < 3.29: return bs('lowSOC', d, 'mendeley<3.40', hh)
    reg = ('dis>=3.40' if preV >= 3.40 else 'dis3.29-3.40') if d == 'discharge' else ('chg_head<0.2' if 4.15 - preV < .2 else 'chg_head>=0.2')
    return bs('LOCO', d, reg, hh)

def margin(d, h, preV, cold, method):
    hh = 2. if h == 2. else 10.
    if cold and method == 'M3':
        return bs('cold', d, 'mendeley_cold', hh)
    if cold:
        return {10.: .4482, 2.: .3981}[hh] if d == 'discharge' else {10.: .0543, 2.: .0665}[hh]
    if d == 'discharge':
        if preV >= 3.40: return {10.: .0273, 2.: .0120}[hh]
        if preV >= 3.29: return {10.: .1012, 2.: .1004}[hh]
        return {10.: .2462, 2.: .1062}[hh]
    if hh == 2.: return .0096
    return .0130 if (4.15 - preV) < .2 else .0154


def m3_weights():
    J = json.loads((HERE / 'soc_step_pairs_v196/weights_all.json').read_text())
    return {(k.split('|')[0], float(k.split('|')[1])): np.array(v) for k, v in J.items()}


def m3_sop(xr, h, d, m, w):
    hh = 2 if h == 2. else 10
    Va = xr[NM.index(f'past_{hh}s_pre_voltage_V')]; ea = xr[NM.index(f'past_{hh}s_oriented_voltage_excursion_V')]; Ia = abs(xr[NM.index(f'past_{hh}s_endpoint_current_A')])
    if not (np.isfinite(Va) and np.isfinite(ea) and ea > 1e-4 and Ia > .5): return None
    Vq = xr[0]; T = xr[TC]; Ra = ea / Ia
    f1 = (1 / (Vq - 2.5) - 1 / (Va - 2.5)) if d == 'discharge' else (1 / (4.25 - Vq) - 1 / (4.25 - Va)); f3 = f1 * (1000 / (T + 273.15) - 1000 / 298.15)
    def feasible(P):
        I = P / Vq
        for _ in range(40):
            f2 = np.log(max(I, 1e-6) / Ia); e = I * Ra * np.exp(w[0] + w[1] * f1 + w[2] * f2 + w[3] * f3 + w[4] * f1 ** 2 + w[5] * f2 ** 2 + w[6] * f1 * f2)
            V = Vq - e if d == 'discharge' else Vq + e
            if V <= .5: return False
            I = P / V
        return ((V - m >= LIMIT[d]) if d == 'discharge' else (V + m <= LIMIT[d])) and I <= CAP[d] + 1e-9
    lo, hi = 0., 400.
    for _ in range(35):
        mid = (lo + hi) / 2
        if feasible(mid): lo = mid
        else: hi = mid
    return lo


def m2_params(fid, X, hts, his, off, idx):
    m = Encoder(74, 1.).eval(); m.load_state_dict(torch.load(FITS / fid / 'model.pt', map_location='cpu', weights_only=True)); sc = joblib.load(FITS / fid / 'scaler.joblib')
    out = np.empty((len(idx), 10)); g = gain_anchor(X[idx])
    with torch.no_grad():
        for k, j in enumerate(idx):
            a, b = off[j], off[j + 1]
            out[k] = m(torch.tensor(sc.transform(X[j][None])), torch.tensor(hts[a:b][None], dtype=torch.float32), torch.tensor(his[a:b][None], dtype=torch.float32), torch.tensor(g[k:k + 1])).numpy()[0]
    return out


def task(a):
    p, vpre, pre, gap, h, d, cap, lim = a
    return sop(np.asarray(p), float(vpre), float(pre), float(gap), float(h), d, cap, lim)['power_W']


def main():
    OUT.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    W = m3_weights(); print('M3 weights', {f'{k[0]}|{k[1]:g}': np.round(v, 4).tolist() for k, v in W.items()}, flush=True)
    P = pd.read_csv(HERE / 'all6_measured_v178/predictions.csv')[['evaluation_group', 'sheet', 'direction', 'temp_C', 'horizon_s', 'row', 'soc_point', 'truth_power_W', 'truth_end_current_A', 'regime']]
    ZA = np.load(HERE / 'rpcwby_aging_inputs_v85/inputs.npz'); ZB = np.load(HERE / 'rpcwby_external_v46/inputs.npz')
    P['method'] = ''; P['margin_V'] = np.nan; P['power_W'] = np.nan; tasks = []
    for (grp, d), sub in P.groupby([P.evaluation_group == 'B', 'direction']):
        Z = ZB if grp else ZA; X = Z['features']; idx = sub.row.to_numpy()
        prm = m2_params(f'{d}__ALL6', X, Z['history_time_s'], Z['history_current_A'], Z['history_offsets'], idx)
        for k, (ix, r) in enumerate(sub.iterrows()):
            xr = X[int(r.row)]; cold = bool(xr[TC] < 10); h = float(r.horizon_s)
            m = margin(d, h, xr[0], True, 'M3') if cold else m3_warm_margin(d, h, xr[0])
            v = m3_sop(xr, h, d, m, W[(d, 2. if h == 2. else 10.)])
            if v is not None:
                P.loc[ix, ['method', 'margin_V', 'power_W']] = ['M3', m, v]; continue
            m = margin(d, h, xr[0], cold, 'M2'); P.loc[ix, ['method', 'margin_V']] = ['M2_fallback', m]
            lim = LIMIT[d] + m if d == 'discharge' else LIMIT[d] - m
            tasks.append((ix, (prm[k], xr[0], xr[72], xr[73], h, d, CAP[d], lim)))
    with Pool(14) as pool:
        res = pool.map(task, [t for _, t in tasks], chunksize=8)
    for (ix, _), v in zip(tasks, res): P.loc[ix, 'power_W'] = v
    P.to_csv(OUT / 'predictions.csv', index=False)
    V = P[P.regime == 'voltage_limited'].copy(); V['over'] = V.power_W - V.truth_power_W > 1e-9; V['use'] = 100 * V.power_W / V.truth_power_W
    G = V.groupby(['evaluation_group', 'sheet', 'direction', 'temp_C', 'horizon_s']).agg(n=('row', 'size'), exceed=('over', 'sum'), median_usable_pct=('use', 'median'),
                                                                                     methods=('method', lambda s: '/'.join(f'{k}:{v}' for k, v in s.value_counts().items()))).reset_index()
    G['goal'] = (G.evaluation_group != 'B') | G.horizon_s.isin([2., 10.]); G['pass'] = (G.exceed == 0) & (G.median_usable_pct >= 80)
    G.to_csv(OUT / 'groups.csv', index=False); g = G[G.goal]
    res = dict(groups_passing=int(g['pass'].sum()), exceed=int(g.exceed.sum()), A_exceed=int(g[g.evaluation_group != 'B'].exceed.sum()), B_exceed=int(g[g.evaluation_group == 'B'].exceed.sum()),
               groups_at_80=int((g.median_usable_pct >= 80).sum()), m3_weights={f'{k[0]}|{k[1]:g}': np.round(v, 5).tolist() for k, v in W.items()},
               comparators='V195 0/15 exceed 224; V193 0/15 exceed 109; V180 0/15 exceed 117', adopted=False, goal_complete=False)
    dump(OUT / 'verification.json', res); pd.set_option('display.width', 220); print(G.round(1).to_string(index=False)); print(json.dumps({k: v for k, v in res.items() if k != 'm3_weights'}))


if __name__ == '__main__':
    main()
