"""Diagnostic SOP scoring of the Mamba models on the 15 RPCWBY groups (user request 2026-09-23). NOT a pre-registered test:
all three models failed their pilot gate; the numbers are diagnostics and cannot support adoption.

Models: the all-cell fits (six UYPYDJ cells + Mendeley) of V211, V213, V214 with their pilot recipes.
Margins (V184 style): the limit moves inward by m; arms zero / q95 / max of the pooled held-out residuals of that
version's two pilot folds (CC, BOOST_REST) per direction and pulse length (two folds, not six: stated in the report).
Inputs: V211 / V214 read [H0, pulse start) (after the last hidden gap if one lies inside); V214's SOH is the all-cell ridge
on the 1C charges the row may read (before its blackout or after its H0; median of the last five). V213 runs the cell's whole stream; the stretch from the
first step of the query's own search to H0 is one gap step.
SOP: constant power P, 0.1 s grid, current I = P / V by fixed point (6 passes); the largest P whose voltage stays inside
the moved limit over the horizon and whose current stays under the cap (30 A discharge, 15 A charge); 40 bisections.
Judged rows, truth, exceed (> 1e-9 W), usage and pass as in every earlier version (V157 key table, goal rows)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import data_v211 as D, data_v213 as S2, soh_v214 as SOH
import model_v211 as M1, model_v213 as M3, model_v214 as M4
import eval_bench_v211 as E1, eval_bench_v213 as E3

HERE = D.HERE; OUT = HERE / 'sop_diagnostic_v211'
KEYS = D.HERE.parents[1] / 'same_search_exclusion_v157_20260921/keys/row_keys.csv'
CELLS = ('RPCWBY_TEST1', 'RPCWBY_TEST2', 'RPCWBY_TEST3')
LIMIT = dict(discharge=2.55, charge=4.15); CAP = dict(discharge=30., charge=15.); DT = .1


def margins(tag):
    runs = {'v211': 'pilot_heldout', 'v213': 'v213_heldout', 'v214': 'v214_heldout'}[tag]
    R = pd.concat([pd.read_csv(HERE / 'runs' / f'{runs}_{c}' / f'bench_{c}.csv') for c in ('CC', 'BOOST_REST')])
    out = {}
    for (d, h), g in R.groupby(['direction', 'requested_horizon_s']):
        r = g.mamba_residual_V.to_numpy(); out[(d, float(h))] = dict(zero=0., q95=max(0., float(np.quantile(r, .95))), max=max(0., float(r.max())))
    return out


def load_stream(cell):
    z = np.load(HERE / 'cache' / f'rpcwby_stream_{cell}.npz'); ok = z['ok']
    s = dict(t=z['t'][ok], V=z['V'][ok], I=z['I'][ok], T=z['T'][ok], seg=z['segment'][ok], raw_index=np.flatnonzero(ok), name=cell)
    dt = np.r_[1., np.diff(s['t'])]; dt[np.r_[True, s['seg'][1:] != s['seg'][:-1]]] = 1.; s['dt'] = dt.astype(np.float32)
    s['cs'] = D.full_charge_starts(s); return s


def sop(model, Y, rows, m_of, dev):
    """Bisection on constant power for every row. rows: DataFrame with direction, horizon_s, V0, I0. Returns W."""
    n = len(rows); d = rows.direction.to_numpy(); h = rows.horizon_s.to_numpy(float)
    sign = np.where(d == 'discharge', -1., 1.)
    lim = np.where(d == 'discharge', 2.55 + np.asarray(m_of), 4.15 - np.asarray(m_of)); cap = np.where(d == 'discharge', 30., 15.)
    w = int(round(h.max() / DT)); tau = np.tile(np.arange(1, w + 1) * DT, (n, 1)); mask = (tau <= h[:, None] + 1e-9).astype(np.float32)
    T_ = lambda a: torch.tensor(a, device=dev, dtype=torch.float32)
    with torch.no_grad():
        p = model.params(T_(Y), torch.tensor(sign > 0, device=dev))
    V0 = rows.V0.to_numpy(float); I0 = rows.I0.to_numpy(float)
    lo = np.zeros(n); hi = cap * V0 * 1.2

    def feasible(P):
        V = np.repeat(V0[:, None], w, 1)
        for _ in range(6):
            I = sign[:, None] * P[:, None] / np.maximum(V, 1.)
            with torch.no_grad():
                V = V0[:, None] + model_response(p, T_(tau), T_(I), T_(I0), T_(mask)).cpu().numpy()
        Ifin = sign[:, None] * P[:, None] / np.maximum(V, 1.)
        vio = np.where(d[:, None] == 'discharge', V < lim[:, None] - 1e-9, V > lim[:, None] + 1e-9) & (mask > 0)
        return ~vio.any(1) & ((np.abs(Ifin) <= cap[:, None] + 1e-9) | (mask == 0)).all(1)

    ok0 = feasible(np.full(n, 1e-6)); lo = np.where(ok0, 1e-6, 0.)
    for _ in range(40):
        mid = .5 * (lo + hi); f = feasible(mid); lo = np.where(f, mid, lo); hi = np.where(f, hi, mid)
    return np.where(ok0, lo, 0.)


def model_response(p, tau, I, I0, mask):
    return M1.response(p, tau, I, I0, mask)


def score(pred, J):
    J = J.copy(); J['pred_W'] = pred; J['over'] = J.pred_W - J.truth_power_W > 1e-9; J['use'] = 100 * J.pred_W / J.truth_power_W
    G = J.groupby(['evaluation_group', 'sheet', 'direction', 'temp_C', 'horizon_s']).agg(n=('row_id', 'size'), exceed=('over', 'sum'), median_usable_pct=('use', 'median')).reset_index()
    G['pass'] = (G.exceed == 0) & (G.median_usable_pct >= 80)
    ev = G.evaluation_group.astype(str)
    return G, dict(groups=int(len(G)), passing=int(G['pass'].sum()), exceed=int(G.exceed.sum()), A_exceed=int(G[ev != 'B'].exceed.sum()),
                   B_exceed=int(G[ev == 'B'].exceed.sum()), groups_at_80=int((G.median_usable_pct >= 80).sum()),
                   passing_groups=[f'{r.evaluation_group} {r.direction} {r.temp_C} {r.horizon_s:g}s' for r in G[G['pass']].itertuples()])


def main():
    OUT.mkdir(exist_ok=True); dev = torch.device('cuda')
    K = pd.read_csv(KEYS); K = K[K.goal].copy()
    R = pd.read_csv(HERE / 'cache' / 'rpcwby_stream_rows.csv').merge(K[['row_id', 'sheet', 'evaluation_group', 'temp_C', 'horizon_s', 'truth_power_W']], on='row_id')
    assert len(R) == 672 and R[['pulse_index', 'h0_index', 'blackout_index']].notna().all().all()
    Z = np.load(SOH.RIDGE_DATA); Xr, yr, cr = Z['X'].astype(float), Z['y'].astype(float), Z['cell'].astype(str)
    ridge_all, _ = SOH.fit_cells(Xr, yr, cr, D.UYPYDJ)
    Ys = {k: [] for k in ('v211', 'v213', 'v214')}; frames = []; report = {}
    models = {}
    for tag, mod, run in (('v211', M1, 'all6_v211'), ('v213', M3, 'all6_v213'), ('v214', M4, 'all6_v214')):
        m = mod.Model().to(dev); m.load_state_dict(torch.load(HERE / 'runs' / run / 'weights_final.pt', map_location=dev)); m.eval(); models[tag] = m
    for cell in CELLS:
        s = load_stream(cell); g = R[R.cell == cell].copy()
        mp = lambda raw: np.searchsorted(s['raw_index'], raw.astype(int))
        pi, hi_, bo = mp(g.pulse_index.to_numpy()), mp(g.h0_index.to_numpy()), mp(g.blackout_index.to_numpy())
        assert (s['raw_index'][pi] == g.pulse_index.to_numpy()).all()
        q = pi - 1; seg = s['seg']
        seg_start = np.array([np.searchsorted(seg, seg[k], 'left') for k in q]); st = np.maximum(hi_, seg_start)
        g['t0'] = q; g['V0'] = s['V'][q]; g['I0'] = s['I'][q]; g['hist_h'] = (s['t'][q] - s['t'][st]) / 3600; g['hist_cut_by_gap'] = seg_start > hi_
        # V214 SOH input: ridge on 1C charges before the query, none inside the query's own blackout window
        det = SOH.detect(s); av = np.array([a for a, _, _ in det]); est = np.array([float(SOH._pred(ridge_all, f[None])[0]) for _, f, _ in det]) if det else np.array([])
        soh_arr = np.zeros(len(s['t']), np.float32); has_arr = np.zeros(len(s['t']), bool)
        for k, a in enumerate(av): soh_arr[a:] = np.median(est[max(0, k - SOH.K_MEDIAN + 1):k + 1]); has_arr[a:] = True
        inside = [int(((av >= b) & (av < h)).sum()) for b, h in zip(bo, hi_)] if len(av) else [0] * len(g)
        report[cell] = dict(one_c_charges=int(len(det)), detections_inside_own_blackout=int(sum(inside)), rows=int(len(g)),
                            hist_cut_by_hidden_gap=int(g.hist_cut_by_gap.sum()), soh_at_queries=[float(soh_arr[q][has_arr[q]].min()) if has_arr[q].any() else None,
                                                                                                  float(soh_arr[q][has_arr[q]].max()) if has_arr[q].any() else None],
                            queries_with_soh=int(has_arr[q].sum()))
        # Test#3 recharges between sweeps at 3 A, so 1C runs can lie inside a query's own blackout (attempt01 stopped here):
        # the SOH input of each row is rebuilt from the runs it may read, i.e. before its blackout or after its H0
        row_soh = []
        for b, h, a0, qq in zip(bo, hi_, st, q):
            ok_ = (av < b) | ((av >= h) & (av <= qq)) if len(av) else np.zeros(0, bool)
            aa, ee = av[ok_], est[ok_]; soh_w = np.zeros(qq - a0 + 1, np.float32); has_w = np.zeros(qq - a0 + 1, bool)
            for k in range(len(aa)):
                pos = max(int(aa[k]) - a0, 0)
                if aa[k] <= qq: soh_w[pos:] = np.median(ee[max(0, k - SOH.K_MEDIAN + 1):k + 1]); has_w[pos:] = True
            row_soh.append((soh_w, has_w))
        report[cell]['queries_with_admissible_soh'] = int(sum(h_[-1] for _, h_ in row_soh))
        Ys['v211'].append(E1.encode_at(models['v211'], s, st, q, dev))
        Ys['v214'].append(encode_at_v214(models['v214'], s, st, q, dev, row_soh))
        Ys['v213'].append(lifelong(models['v213'], s, bo, hi_, q, dev))
        frames.append(g)
    J = pd.concat(frames).reset_index(drop=True); res = {}
    for tag in ('v211', 'v213', 'v214'):
        Y = np.concatenate(Ys[tag]); mg = margins(tag); res[tag] = {'margins_V': {f'{k[0]}|{k[1]:g}s': v for k, v in mg.items()}}
        for arm in ('zero', 'q95', 'max'):
            m_of = [mg[(d, 2. if h == 2. else 10.)][arm] for d, h in zip(J.direction, J.horizon_s)]
            pred = np.zeros(len(J))
            for hh in sorted(J.horizon_s.unique()):
                sel = (J.horizon_s == hh).to_numpy(); pred[sel] = sop(models[tag], Y[sel], J[sel], np.asarray(m_of)[sel], dev)
            G, sm = score(pred, J); res[tag][arm] = sm
            G.to_csv(OUT / f'groups_{tag}_{arm}.csv', index=False); J.assign(pred_W=pred).to_csv(OUT / f'predictions_{tag}_{arm}.csv', index=False)
            print(tag, arm, sm['passing'], '/', sm['groups'], 'exceed', sm['exceed'], '(A', sm['A_exceed'], 'B', sm['B_exceed'], ') at80', sm['groups_at_80'], sm['passing_groups'], flush=True)
    res['inputs'] = report; res['note'] = 'diagnostic after failed pilot gates; margins from two folds; not adoptable'
    json.dump(res, open(OUT / 'summary.json', 'w'), indent=1)


def encode_at_v214(model, s, starts, qs, dev, row_soh):
    """eval_bench_v211.encode_at with the two SOH inputs, given per row over its own window."""
    n = len(qs); out = np.zeros((n, M1.D_MODEL), np.float32); order = np.argsort(qs - starts)
    for i in range(0, n, 16):
        idx = order[i:i + 16]; L = int(np.ceil(max(qs[idx] - starts[idx] + 1) / M1.CHUNK) * M1.CHUNK); B_ = len(idx)
        arr = {k: np.zeros((B_, L), np.float32) for k in ('V', 'I', 'T', 'dt', 'soh')}; arr['has'] = np.zeros((B_, L), bool)
        for b, r in enumerate(idx):
            a, q = starts[r], qs[r]
            for k in ('V', 'I', 'T', 'dt'): arr[k][b, :q - a + 1] = s[k][a:q + 1]
            arr['soh'][b, :q - a + 1], arr['has'][b, :q - a + 1] = row_soh[r]
            arr['dt'][b, 0] = 1.
        with torch.no_grad():
            y = model.encode(*(torch.tensor(arr[k], device=dev) for k in ('V', 'I', 'T', 'dt', 'soh', 'has')))
        out[idx] = y[torch.arange(B_, device=dev), torch.tensor(qs[idx] - starts[idx], device=dev)].cpu().numpy()
    return out


def lifelong(model, s, bo, h0, q, dev):
    """V213: whole stream with gap steps at segment starts; per query group the state before its blackout, one gap step at
    H0, then the sweep up to the query."""
    st = S2.stream(dict(t=s['t'], V=s['V'], I=s['I'], T=s['T'], seg=s['seg'])); t = st['t']
    need = sorted(set(int(b) - 1 for b in bo if b > 0)); states = {-1: None}; h = None; pos = 0
    for e in need:
        h, _ = E3.run(model, st, pos, e, h, dev); states[e] = [x.clone() for x in h]; pos = e + 1
    out = np.zeros((len(q), M3.D_MODEL), np.float32)
    for (b, hh), idx in pd.Series(range(len(q))).groupby([bo, h0]).groups.items():
        idx = np.asarray(list(idx)); qq = q[idx]
        gap = float(t[hh] - t[b - 1]) if b > 0 else None   # blackout from the very first sample: start cold at H0
        _, o = E3.run(model, st, int(hh), int(qq.max()), states[int(b) - 1], dev, gap_first=gap, want=set(int(x) for x in qq))
        out[idx] = np.stack([o[int(x)] for x in qq])
    return out


if __name__ == '__main__':
    main()
