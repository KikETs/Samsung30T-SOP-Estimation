"""V213 pilot bench (PROTOCOL_MAMBA_V213.md). usage: python eval_bench_v213.py <fold_name> <held-out cell>

The held-out cell is one lifelong stream (natural gaps as gap steps, no random blackouts).
  primary    30 h blackout: the state is run to 30 h before the latest full-charge start that precedes the pulse, the next
             30 h are skipped with one gap step at that charge start, and the stream then runs to the pulse's pre sample
             (the RPCWBY rule: blackout, then the recharge H0, then the sweep).
  continuous report only: the state is run without any blackout.
Residual per row and all statistics exactly as eval_bench_v211 / pilot_verdict_v211."""
import json, sys
import numpy as np, pandas as pd, torch
import model_v213 as M, data_v211 as D, data_v213 as S

BENCH = D.HERE / 'bench'; BLACKOUT_S = 30 * 3600.; SUB = 262144


def run(model, st, a, e, h, dev, gap_first=None, want=None):
    """Run stream samples [a, e] from state h (list or None); optional override of the first sample's dt/gap. Returns
    final state and the encoder outputs at the stream positions in `want` (dict pos -> row)."""
    out = {}
    pos = a
    while pos <= e:
        b = min(pos + SUB - 1, e); n = b - pos + 1; L = int(np.ceil(n / M.CHUNK) * M.CHUNK)
        arr = {k: np.zeros((1, L), np.float32) for k in ('V', 'I', 'T', 'dt')}; gp = np.zeros((1, L), bool)
        for k in ('V', 'I', 'T', 'dt'): arr[k][0, :n] = st[k][pos:b + 1]
        gp[0, :n] = st['gap'][pos:b + 1]
        if pos == a and gap_first is not None: arr['dt'][0, 0] = gap_first; gp[0, 0] = True
        with torch.no_grad():
            y, h = model.encode(*(torch.tensor(arr[k], device=dev) for k in ('V', 'I', 'T', 'dt')), torch.tensor(gp, device=dev), h0=h)
        if want is not None:
            sel = [p for p in want if pos <= p <= b]
            if sel: yy = y[0, torch.tensor([p - pos for p in sel], device=dev)].cpu().numpy(); out.update({p: yy[j] for j, p in enumerate(sel)})
        pos = b + 1
    return h, out


def residuals(model, R, Y, P, dev):
    j = R.bench_row.to_numpy(); t, i, y, L = P['t'][j], P['i'][j], P['y'][j], P['L'][j]
    sign = np.where(R.direction.to_numpy() == 'discharge', -1., 1.)
    w = t.shape[1]; valid = (np.arange(w)[None, :] < L[:, None]) & (np.arange(w)[None, :] > 0)
    T = lambda a: torch.tensor(a, device=dev, dtype=torch.float32)
    with torch.no_grad():
        p = model.params(T(Y), torch.tensor(sign > 0, device=dev))
        pred = M.response(p, T(t[:, 1:]), T(sign[:, None] * i[:, 1:]), T(sign * i[:, 0]), T(valid[:, 1:].astype(np.float32))).cpu().numpy()
    return np.where(valid[:, 1:], y[:, 1:] - sign[:, None] * pred, -np.inf).max(1)


def main():
    fold, held = sys.argv[1], sys.argv[2]
    run_dir = D.HERE / 'runs' / fold; dev = torch.device('cuda')
    model = M.Model().to(dev); model.load_state_dict(torch.load(run_dir / 'weights_final.pt', map_location=dev)); model.eval()
    R = pd.read_csv(BENCH / 'bench_rows.csv'); P = np.load(BENCH / 'bench_profiles.npz'); R['bench_row'] = np.arange(len(R))
    R = R[(R.cell == held) & R.matched].copy()
    base = S.base(held); st = S.stream(base)                                  # no blackout: stream index == base index
    raw = R.cache_index.to_numpy(int); q = np.searchsorted(base['raw_index'], raw)
    kept = (q < len(base['raw_index'])) & (base['raw_index'][np.minimum(q, len(base['raw_index']) - 1)] == raw)
    R = R[kept].copy(); q = q[kept]; assert np.allclose(st['V'][q], R.pre_voltage_V.to_numpy(), atol=.002)
    t = st['t']; cs = base['cs']
    k = np.searchsorted(cs, q, 'right') - 1; assert (k >= 0).all()
    c_of = cs[k]; b_of = np.searchsorted(t, t[c_of] - BLACKOUT_S, 'left')
    R['charge_start'] = c_of; R['blackout_start'] = b_of; R['hours_after_charge'] = (t[q] - t[c_of]) / 3600
    # continuous pass (report only) and the states at every blackout start, in one sweep
    need_states = sorted(set(int(b) - 1 for b in b_of if b > 0)); want = set(int(x) for x in q)
    states = {}; h = None; pos = 0; ycont = {}
    for e in need_states + [len(t) - 1]:
        if e < pos: continue
        h, out = run(model, st, pos, e, h, dev, want=want); ycont.update(out)
        states[e] = [x.clone() for x in h]; pos = e + 1
    # primary: per charge start, state before the blackout, one gap step at the charge start, then run to the pulses
    Yp = np.zeros((len(R), M.D_MODEL), np.float32)
    for (c, b), g in R.assign(q=q).groupby(['charge_start', 'blackout_start']):
        h0 = states.get(int(b) - 1) if b > 0 else None
        gap_len = float(t[c] - t[b - 1]) if b > 0 else 1.
        _, out = run(model, st, int(c), int(g.q.max()), h0, dev, gap_first=gap_len if b > 0 else None, want=set(int(x) for x in g.q))
        Yp[np.flatnonzero(R.index.isin(g.index))] = np.stack([out[int(x)] for x in g.q])
    Yc = np.stack([ycont[int(x)] for x in q])
    R['mamba_residual_V'] = residuals(model, R, Yp, P, dev); R['mamba_continuous_residual_V'] = residuals(model, R, Yc, P, dev)
    R.to_csv(run_dir / f'bench_{held}.csv', index=False)
    st_ = lambda x: dict(median=float(np.median(x)), q95=max(0., float(np.quantile(x, .95))), max=max(0., float(np.max(x))))
    summ = {f'{d}_{int(hh)}s': dict(n=int(len(g)), mamba=st_(g.mamba_residual_V), continuous=st_(g.mamba_continuous_residual_V), v75=st_(g.v75_heldout_residual_V))
            for (d, hh), g in R[np.isfinite(R.v75_heldout_residual_V)].groupby(['direction', 'requested_horizon_s'])}
    summ['hours_after_charge'] = dict(median=float(R.hours_after_charge.median()), max=float(R.hours_after_charge.max()))
    json.dump(summ, open(run_dir / f'bench_summary_{held}.json', 'w'), indent=1); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
