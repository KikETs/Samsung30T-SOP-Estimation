"""V211 pilot bench (PROTOCOL_MAMBA_V211.md): held-out UYPYDJ cell, the V184 HPPC pulses, same target waveforms as V75.

usage: python eval_bench_v211.py <fold_name> <held-out cell>
Residual per row = max over valid samples of (measured limit-direction shift - predicted shift), V184 definition.
Also a report-only diagnostic: the last 60 s of every constant-current full discharge (0.5C/1C/2C) of the held-out cell.
"""
import json, sys
import numpy as np, pandas as pd, torch
import model_v211 as M, data_v211 as D

BENCH = D.HERE / 'bench'


def encode_at(model, s, starts, qs, dev, max_batch_steps=600_000):
    """Encoder output at q for histories [start, q]; batched by length. Returns (n, D_MODEL) numpy."""
    n = len(qs); out = np.zeros((n, M.D_MODEL), np.float32); order = np.argsort(qs - starts)
    i = 0
    while i < n:
        L = int(np.ceil((qs[order[i]] - starts[order[i]] + 1) / M.CHUNK) * M.CHUNK); j = i
        while j < n:
            Lj = int(np.ceil((qs[order[j]] - starts[order[j]] + 1) / M.CHUNK) * M.CHUNK)
            if Lj * (j - i + 1) > max_batch_steps and j > i: break
            L = max(L, Lj); j += 1
        idx = order[i:j]; B = len(idx)
        arr = {k: np.zeros((B, L), np.float32) for k in ('V', 'I', 'T', 'dt')}
        for b, r in enumerate(idx):
            a, q = starts[r], qs[r]
            for k in ('V', 'I', 'T', 'dt'): arr[k][b, :q - a + 1] = s[k][a:q + 1]
            arr['dt'][b, 0] = 1.
        with torch.no_grad():
            y = model.encode(*(torch.tensor(arr[k], device=dev) for k in ('V', 'I', 'T', 'dt')))
        out[idx] = y[torch.arange(B, device=dev), torch.tensor(qs[idx] - starts[idx], device=dev)].cpu().numpy()
        i = j
    return out


def main():
    fold, held = sys.argv[1], sys.argv[2]
    run = D.HERE / 'runs' / fold; dev = torch.device('cuda')
    model = M.Model().to(dev); model.load_state_dict(torch.load(run / 'weights_final.pt', map_location=dev)); model.eval()
    R = pd.read_csv(BENCH / 'bench_rows.csv'); P = np.load(BENCH / 'bench_profiles.npz')
    R['bench_row'] = np.arange(len(R))
    R = R[(R.cell == held) & R.matched].copy()
    s = D.load(held); cs = D.full_charge_starts(s)
    # bench cache_index is a position in the uncleaned cache; map it to the cleaned series (attempt01 used it directly and
    # shifted every CC row after the dropped 60 A glitch by one sample, so the history ended on the first pulse sample)
    raw = R.cache_index.to_numpy(int); q = np.searchsorted(s['raw_index'], raw)
    kept = (q < len(s['raw_index'])) & (s['raw_index'][np.minimum(q, len(s['raw_index']) - 1)] == raw)
    R = R[kept].copy(); q = q[kept]; assert np.allclose(s['V'][q], R.pre_voltage_V.to_numpy(), atol=.002)
    st = np.array([D.history_start(s, k, cs) for k in q])
    R['hist_start'] = st; R['hist_h'] = (s['t'][q] - s['t'][st]) / 3600; R['hist_from_charge'] = np.isin(st, cs)
    uq, inv = np.unique(np.stack([st, q], 1), axis=0, return_inverse=True)
    Y = encode_at(model, s, uq[:, 0], uq[:, 1], dev)[inv.ravel()]
    j = R.bench_row.to_numpy(); t, i, y, L = P['t'][j], P['i'][j], P['y'][j], P['L'][j]
    sign = np.where(R.direction.to_numpy() == 'discharge', -1., 1.)
    w = t.shape[1]; valid = (np.arange(w)[None, :] < L[:, None]) & (np.arange(w)[None, :] > 0)
    T = lambda a: torch.tensor(a, device=dev, dtype=torch.float32)
    with torch.no_grad():
        p = model.params(T(Y), torch.tensor(sign > 0, device=dev))
        pred = M.response(p, T(t[:, 1:]), T(sign[:, None] * i[:, 1:]), T(sign * i[:, 0]), T(valid[:, 1:].astype(np.float32))).cpu().numpy()
    pred_o = sign[:, None] * pred
    resid = np.where(valid[:, 1:], y[:, 1:] - pred_o, -np.inf).max(1)
    end = L - 1
    R['mamba_residual_V'] = resid; R['mamba_end_error_V'] = pred_o[np.arange(len(R)), end - 1] - y[np.arange(len(R)), end]
    for k, v in p.items(): R[f'p_{k}'] = v.cpu().numpy()
    R.to_csv(run / f'bench_{held}.csv', index=False)
    summ = {}
    for (d, h), g in R.groupby(['direction', 'requested_horizon_s']):
        g = g[np.isfinite(g.v75_heldout_residual_V)]
        st_ = lambda x: dict(median=float(np.median(x)), q95=max(0., float(np.quantile(x, .95))), max=max(0., float(np.max(x))))
        summ[f'{d}_{int(h)}s'] = dict(n=int(len(g)), mamba=st_(g.mamba_residual_V), v75=st_(g.v75_heldout_residual_V))
    summ['history_h'] = dict(median=float(R.hist_h.median()), max=float(R.hist_h.max()), from_charge=int(R.hist_from_charge.sum()))
    # report-only: last 60 s of constant-current full discharges
    V, I, tt = s['V'], s['I'], s['t']
    ends = np.flatnonzero((V <= 2.52) & (I < -.5) & np.r_[(I[1:] > -.5) | (s['seg'][1:] != s['seg'][:-1]), True])
    rows = []
    for e in ends:
        a = int(np.searchsorted(tt, tt[e] - 60., 'left'))
        if a >= e - 2 or s['seg'][a] != s['seg'][e]: continue
        seg_I = I[a:e + 1]; med = np.median(seg_I)
        if np.any(np.abs(seg_I - med) > .05 * abs(med) + .05): continue
        rows.append((a, e, float(med)))
    if rows:
        a_ = np.array([r[0] for r in rows]); e_ = np.array([r[1] for r in rows])
        st2 = np.array([D.history_start(s, k, cs) for k in a_]); Y2 = encode_at(model, s, st2, a_, dev)
        ww = int((e_ - a_).max()); tau = np.zeros((len(rows), ww), np.float32); Iw = np.zeros_like(tau); m = np.zeros_like(tau)
        for r, (a, e, _) in enumerate(rows):
            k = e - a; tau[r, :k] = tt[a + 1:e + 1] - tt[a]; Iw[r, :k] = I[a + 1:e + 1]; m[r, :k] = 1
            if k < ww: tau[r, k:] = tau[r, k - 1] + np.arange(1, ww - k + 1)
        with torch.no_grad():
            p2 = model.params(T(Y2), torch.zeros(len(rows), dtype=torch.bool, device=dev))
            pr = M.response(p2, T(tau), T(Iw), T(I[a_]), T(m)).cpu().numpy()
        k_ = e_ - a_ - 1; meas = V[e_] - V[a_]
        err = pr[np.arange(len(rows)), k_] - meas          # > 0: predicted drop smaller than measured (optimistic)
        pd.DataFrame(dict(start=a_, end=e_, current_A=[r[2] for r in rows], measured_dV=meas, predicted_dV=pr[np.arange(len(rows)), k_],
                          optimistic_error_V=err)).to_csv(run / f'discharge_end_{held}.csv', index=False)
        summ['discharge_end_60s'] = dict(n=len(rows), median=float(np.median(err)), q95=float(np.quantile(err, .95)), max=float(err.max()))
    json.dump(summ, open(run / f'bench_summary_{held}.json', 'w'), indent=1); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
