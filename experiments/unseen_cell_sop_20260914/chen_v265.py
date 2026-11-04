"""V265 (PROTOCOL_CHEN_V265.md): the reference paper's SOP method (Chen et al., J. Energy Storage 175 (2026) 123023: LSTM
voltage model + constant-power binary search) under this project's rules, scored with the V255/V256 closed-loop code.
  - leave one RPCWBY cell out; training = Mendeley Mixed 1-8 + 200 s windows from the SOP periods of the two other cells
    (0.5 x the drive windows, the August reproduction's best ratio); early stopping on the named Mendeley cycles
  - SOC input = rated-axis coulomb count from the end of the last full charge (no SOC label, no cycler SOC column)
  - the query state is read from the sweep's recharge start H0 (all same-search repetitions lie before H0) to the pulse
  - training recipe = Samsung30T/analysis/lstm_voltage.py (head 512/256, lr 1e-4, batch 5000 as 4 x 1250 with the exact
    RMSE gradient, MultiStepLR at 20/40/60/80 % of 1000 epochs, patience 20 checks x 10 epochs, seed 5, fp32)
usage: python chen_v265.py pilot | train <A1|A17|B> | infer <A1|A17|B> | infer_r | score"""
import json, sys, time
import numpy as np, pandas as pd, torch
from run_envelope import HERE, ANALYSIS, dump
import lstm_voltage as LV                                   # Samsung30T/analysis (put on sys.path by run_envelope)
from data30t import DRIVE_CYCLES, mendeley_files
sys.path.insert(1, str(HERE / 'mamba_v211'))
import score_sop_mamba_v211 as SS, ekf_v217 as K
import closedloop_v255 as V5

OUT = HERE / 'chen_v265'
CELLS = {'A1': 'RPCWBY_TEST1', 'A17': 'RPCWBY_TEST2', 'B': 'RPCWBY_TEST3'}
MENDELEY = ANALYSIS.parent / 'raw' / 'Mendeley'
AUG_MODEL = ANALYSIS / 'runs' / 'lstm_ratio05_e1000.pt'
EPOCHS, PATIENCE, VAL_EVERY, MICRO, SOP_RATIO = 1000, 20, 10, 1250, 0.5
SPAN_AFTER, P_MAX, N_BISECT, Q_AH = 600, 200.0, 30, 3.0
LIMITS = {'discharge': (2.55, 30.0), 'charge': (4.15, 15.0)}
TAUS = np.round(np.arange(0.04, 0.06001, 0.0025), 4)
REF = {'proposed (V256 main)': [14, 14, 14], 'V242 (V257 corrected)': [12, 12, 13]}


# ---------------------------------------------------------------- streams and inputs
def cc_soc(s):
    """Rated-axis coulomb count: 1.0 at the end of every full charge, backfilled over that charge, carried forward to the next
    full-charge start inside the segment; right-rectangle increments I_k (t_k - t_k-1); NaN before a segment's first full charge."""
    t, I, seg = s['t'], s['I'].astype(np.float64), s['seg']
    dq = np.r_[0.0, I[1:] * np.diff(t)]; dq[np.r_[True, seg[1:] != seg[:-1]]] = 0.0
    Q = np.cumsum(dq) / 3600.0
    soc = np.full(len(t), np.nan); cs, ce = K.charge_runs(s)
    for j, (a, e) in enumerate(zip(cs, ce)):
        nxt = cs[j + 1] if j + 1 < len(cs) else len(t)
        hi = min(nxt, int(np.searchsorted(seg, seg[e], side='right')))
        soc[a:hi] = 1.0 + (Q[a:hi] - Q[e]) / Q_AH
    return soc, Q


def carry(s):
    """Inference only (protocol): a segment that starts without a full charge continues the previous segment's last SOC as if
    no current flowed during the break."""
    soc, Q, seg = s['soc'].copy(), s['Q'], s['seg']
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    for a in starts[1:]:
        if np.isfinite(soc[a]) or not np.isfinite(soc[a - 1]):
            continue
        b = int(np.searchsorted(seg, seg[a], side='right')); fin = np.flatnonzero(np.isfinite(soc[a:b]))
        e = a + (fin[0] if len(fin) else b - a)
        soc[a:e] = soc[a - 1] + (Q[a:e] - Q[a]) / Q_AH
    return soc


def stream(cell):
    s = SS.load_stream(CELLS[cell]); s['Tf'] = K.fill_T(s['T'])
    s['soc'], s['Q'] = cc_soc(s); s['soc_inf'] = carry(s)
    return s


def rows_of(cell, s):
    """Benchmark rows of a cell with stream positions: bo (blackout start), h0, qp (pulse start), pre = qp - 1."""
    R = pd.read_csv(HERE / 'mamba_v211/cache/rpcwby_stream_rows.csv'); R = R[R.cell == CELLS[cell]].copy()
    B = V5.load()[['row_id', 'evaluation_group', 'direction', 'temp_C', 'horizon_s', 'truth']]
    R = R.drop(columns=['direction']).merge(B, on='row_id', how='left'); assert R.truth.notna().all()
    pos = lambda v: np.searchsorted(s['raw_index'], v.to_numpy().astype(np.int64))
    R['bo'], R['h0'], R['qp'] = pos(R.blackout_index), pos(R.h0_index), pos(R.pulse_index); R['pre'] = R.qp - 1
    return R.reset_index(drop=True)


def grid(s, a, b, soc_key='soc'):
    """1 Hz grid over stream positions a..b (one segment): linear interpolation of V, I, T, SOC; P = V I."""
    t = s['t']; tg = t[a] + np.arange(int(np.floor(t[b] - t[a])) + 1, dtype=np.float64); sl = slice(a, b + 1)
    g = {k: np.interp(tg, t[sl], s[src][sl].astype(np.float64)) for k, src in (('V', 'V'), ('I', 'I'), ('T', 'Tf'), ('SOC', soc_key))}
    g['t'] = tg; g['P'] = g['V'] * g['I']; return g


def sop_windows(cells, n_take, rng):
    """n_take 200 s windows (1 Hz) ending inside the SOP periods of the given cells, uniformly over all eligible window ends.
    SOP period of a cell = union over its benchmark rows of [blackout start, pulse start + 600 samples)."""
    pieces, ends, info, owner = [], [], {}, []
    for c in cells:
        s = stream(c); R = rows_of(c, s); n = len(s['t'])
        mask = np.zeros(n, bool)
        for a, q in zip(np.minimum(R.bo, R.h0), R.qp):
            mask[a:min(q + SPAN_AFTER, n)] = True
        cut = np.r_[True, (mask[1:] != mask[:-1]) | (s['seg'][1:] != s['seg'][:-1])]
        st = np.flatnonzero(cut); en = np.r_[st[1:] - 1, n - 1]; m_win = 0; m_pts = 0
        for a, b in zip(st, en):
            if not mask[a] or b - a < 10:
                continue
            g = grid(s, a, b); feats = np.stack([g['SOC'], g['T'], g['P']], 1).astype(np.float32); ok = np.isfinite(feats).all(1) & np.isfinite(g['V'])
            bad = np.cumsum(~ok); k = np.arange(LV.WINDOW - 1, len(ok))
            good = k[(bad[k] - np.r_[0, bad][k - LV.WINDOW + 1]) == 0]         # all 200 points of the window finite
            if len(good):
                pieces.append((feats, g['V'].astype(np.float32))); owner.append(c); ends.append(np.stack([np.full(len(good), len(pieces) - 1), good], 1))
                m_win += len(good); m_pts += len(ok)
        info[c] = dict(sop_period_points=int(m_pts), eligible_windows=int(m_win))
    E = np.concatenate(ends); sel = np.sort(rng.choice(len(E), size=min(n_take, len(E)), replace=False)); E = E[sel]
    X = np.stack([pieces[p][0][k - LV.WINDOW + 1:k + 1] for p, k in E]); Y = np.array([pieces[p][1][k] for p, k in E], np.float32)
    own = np.array(owner)[E[:, 0]]
    info['taken'] = int(len(E)); info['taken_by_cell'] = {c: int((own == c).sum()) for c in cells}
    info['taken_I_abs_gt_20A_at_end'] = float(np.mean(np.abs(X[:, -1, 2] / np.maximum(Y, 1.0)) > 20.0))
    return X, Y, info


def mendeley_windows():
    files = mendeley_files(str(MENDELEY), kinds=DRIVE_CYCLES)
    tr = [(p, i) for p, i in files if i['tag'].startswith('Mixed')]; va = [(p, i) for p, i in files if not i['tag'].startswith('Mixed')]
    Xd, Yd, _ = LV.build_windows(tr); Xva, Yva, mva = LV.build_windows(va)
    return Xd, Yd, Xva, Yva, mva, len(tr), len(va)


# ---------------------------------------------------------------- training
def train(fold, epochs=EPOCHS, out=OUT, tag=None):
    torch.manual_seed(LV.SEED); np.random.seed(LV.SEED); dev = 'cuda'; out.mkdir(parents=True, exist_ok=True); tag = tag or fold
    t0 = time.time(); Xd, Yd, Xva, Yva, mva, ntr, nva = mendeley_windows()
    Xs, Ys, sinfo = sop_windows([c for c in CELLS if c != fold], int(len(Xd) * SOP_RATIO), np.random.default_rng(LV.SEED))
    Xtr = np.concatenate([Xd, Xs]); Ytr = np.concatenate([Yd, Ys])
    print(f'[{tag}] windows: drive {len(Xd):,} + SOP {len(Xs):,} = {len(Xtr):,}, val {len(Xva):,} ({time.time() - t0:.0f} s); {sinfo}', flush=True)
    sx, sy = LV.MinMax(Xtr), LV.MinMax(Ytr.reshape(-1, 1))
    Xtr_t = torch.from_numpy(sx(Xtr)).to(dev); Ytr_t = torch.from_numpy(sy(Ytr.reshape(-1, 1)).ravel()).to(dev); Xva_t = torch.from_numpy(sx(Xva))
    del Xtr
    model = LV.VoltageLSTM().to(dev); nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LV.LR); miles = [int(epochs * f) for f in (0.2, 0.4, 0.6, 0.8)]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=miles, gamma=LV.GAMMA)
    print(f'[{tag}] model {nparam:,} parameters, milestones {miles}, patience {PATIENCE} x {VAL_EVERY}', flush=True)
    ckpt = out / f'lstm_{tag}.pt'; best, best_ep, bad, n, log, last_ep = float('inf'), -1, 0, len(Xtr_t), [], 0
    for ep in range(1, epochs + 1):
        last_ep = ep
        te = time.time(); model.train(); perm = torch.randperm(n).to(dev); tot = torch.zeros((), device=dev)
        for k in range(0, n, LV.BATCH):
            b = perm[k:k + LV.BATCH]; N = len(b); opt.zero_grad(set_to_none=True); sse = torch.zeros((), device=dev)
            for j in range(0, N, MICRO):                          # exact RMSE gradient over the 5000-window batch
                bb = b[j:j + MICRO]; part = ((model(Xtr_t[bb]) - Ytr_t[bb]) ** 2).sum(); (part / N).backward(); sse += part.detach()
            rmse = torch.sqrt(sse / N)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(0.5 / rmse)
            opt.step(); tot += rmse * N
        tot = tot.item(); sched.step()
        if ep % VAL_EVERY == 0 or ep == 1:
            v = val_rmse(model, Xva_t, Yva, sy, dev); flag = ''
            if v < best - 1e-6:
                best, best_ep, bad, flag = v, ep, 0, ' *best'
                torch.save({'model': model.state_dict(), 'sx': sx.to_dict(), 'sy': sy.to_dict(), 'window': LV.WINDOW, 'fold': fold}, ckpt)
            else:
                bad += 1
            log.append(dict(epoch=ep, train_rmse_scaled=tot / n, val_rmse_mV=v, lr=sched.get_last_lr()[0], sec=time.time() - te))
            print(f'[{tag}] ep {ep:4d} train {tot / n:.5f} val {v:7.2f} mV lr {sched.get_last_lr()[0]:.1e} {time.time() - te:.1f} s/ep '
                  f'peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB{flag}', flush=True)
            if bad >= PATIENCE:
                print(f'[{tag}] early stop at epoch {ep}', flush=True); break
    # best weights: validation predictions per condition (saved), as the paper reports them
    model.load_state_dict(torch.load(ckpt)['model']); model.eval()
    pv = predict_windows(model, Xva_t, sy, dev); cond = pd.DataFrame({'temp_C': [m[0] for m in mva], 'cycle': [m[1] for m in mva], 'err': pv - Yva})
    per = cond.groupby(['temp_C', 'cycle']).err.apply(lambda e: float(np.sqrt(np.mean(e ** 2))) * 1e3).rename('rmse_mV').reset_index()
    np.savez_compressed(out / f'val_pred_{tag}.npz', pred=pv, truth=Yva, temp=cond.temp_C.to_numpy(), cycle=cond.cycle.to_numpy())
    per.to_csv(out / f'val_conditions_{tag}.csv', index=False); pd.DataFrame(log).to_csv(out / f'train_log_{tag}.csv', index=False)
    summ = dict(fold=fold, best_val_rmse_pooled_mV=best, val_rmse_condition_mean_mV=float(per.rmse_mV.mean()), best_epoch=best_ep,
                epochs_run=last_ep, params=nparam, windows=dict(drive=int(len(Xd)), sop=int(len(Xs)), val=int(len(Xva))),
                sop_info=sinfo, minutes=(time.time() - t0) / 60, peak_gpu_GB=torch.cuda.max_memory_allocated() / 1e9)
    dump(out / f'train_summary_{tag}.json', summ); print(f'[{tag}] done: {json.dumps(summ)}', flush=True)
    return summ


def predict_windows(model, X_t, sy, dev):
    with torch.no_grad():
        p = [model(X_t[k:k + LV.VAL_CHUNK].to(dev)).cpu().numpy() for k in range(0, len(X_t), LV.VAL_CHUNK)]
    return np.concatenate(p) * sy.rng[0] + sy.lo[0]


def val_rmse(model, X_t, Y, sy, dev):
    model.eval(); return float(np.sqrt(np.mean((predict_windows(model, X_t, sy, dev) - Y) ** 2))) * 1e3


# ---------------------------------------------------------------- SOP search
class Scaler:
    def __init__(self, d):
        self.lo = np.asarray(d['lo'], np.float64); self.rng = np.where(np.asarray(d['hi']) - self.lo == 0, 1.0, np.asarray(d['hi']) - self.lo)


def load_model(path, dev='cuda'):
    ck = torch.load(path, map_location=dev); m = LV.VoltageLSTM().to(dev); m.load_state_dict(ck['model']); m.eval()
    return m, Scaler(ck['sx']), Scaler(ck['sy'])


def step(m, sx, sy, x, state, dev):
    """One or more 1 Hz steps: x (n, k, 3) physical (SOC, T, P) -> voltages (n, k) and the new state."""
    xs = torch.tensor((x - sx.lo) / sx.rng, dtype=torch.float32, device=dev)
    out, st = m.lstm(xs, state)
    v = m.head(out).squeeze(-1).double().cpu().numpy() * sy.rng[0] + sy.lo[0]
    return v, st


@torch.no_grad()
def search(m, sx, sy, h, c, soc0, T0, direction, L, dv, dev):
    """Largest |P| in [0, P_MAX] whose L-second constant-power pulse keeps the final model voltage and current inside the
    limits (paper Sec. 2.2; bisection). h, c: (2, n, 256) states before the pulse; dv: voltage offset added (arm O) or 0."""
    n = len(soc0); dis = direction == 'discharge'; sign = np.where(dis, -1.0, 1.0)
    vlim = np.where(dis, LIMITS['discharge'][0], LIMITS['charge'][0]); imax = np.where(dis, LIMITS['discharge'][1], LIMITS['charge'][1])
    Lmax = int(L.max()); lo = np.zeros(n); hi = np.full(n, P_MAX)

    def feasible(pm):
        P = sign * pm; soc = soc0.copy(); st = (h.clone(), c.clone()); Vf = np.full(n, np.nan); If = np.full(n, np.nan)
        for k in range(1, Lmax + 1):
            v, st = step(m, sx, sy, np.stack([soc, T0, P], 1)[:, None, :], st, dev); v = v[:, 0] + dv
            I = P / np.maximum(v, 1.0); last = L == k; Vf[last] = v[last]; If[last] = I[last]
            soc = soc + I / 3600.0 / Q_AH
        return np.where(dis, Vf >= vlim, Vf <= vlim) & (np.abs(If) <= imax)

    for _ in range(N_BISECT):
        mid = (lo + hi) / 2; ok = feasible(mid); lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
    return lo


@torch.no_grad()
def infer_cell(cell, ckpt, rows=None, arms=('M', 'W', 'O'), dev='cuda'):
    """SOP of the benchmark rows of a cell from one checkpoint. M: state from H0 (or the query's segment start), continuous;
    W: zero state over the last (200 - L) s before the pulse; O: M plus the last pre-pulse (measured - model) voltage."""
    m, sx, sy = load_model(ckpt, dev); s = stream(cell); R = rows_of(cell, s)
    if rows is not None:
        R = R[R.row_id.isin(rows)].reset_index(drop=True)
    seg = s['seg']; seg_start = lambda p: int(np.searchsorted(seg, seg[p], side='left'))
    R['start'] = [h if seg[h] == seg[p] else seg_start(p) for h, p in zip(R.h0, R.pre)]; R['start_kind'] = np.where(R.start == R.h0, 'H0', 'segment')
    n = len(R); H = torch.zeros(2, n, 256, device=dev); C = torch.zeros(2, n, 256, device=dev)
    soc_q, T_q, v_meas, v_model, hours, Lr = (np.full(n, np.nan) for _ in range(6))
    WH, WC = torch.zeros(2, n, 256, device=dev), torch.zeros(2, n, 256, device=dev)
    for st0, g in R.groupby('start'):
        last = int(g.pre.max()); G = grid(s, st0, last, 'soc_inf'); X = np.stack([G['SOC'], G['T'], G['P']], 1)
        assert np.isfinite(X).all(), (cell, st0)
        state = None; done = 0
        for i in g.sort_values('pre').index:
            k = int(np.floor(s['t'][R.at[i, 'pre']] - s['t'][st0]))       # last grid point at or before the pre-pulse sample
            if k + 1 > done:
                v, state = step(m, sx, sy, X[done:k + 1][None], state, dev); vlast = v[0, -1]; done = k + 1
            H[:, i], C[:, i] = state[0][:, 0], state[1][:, 0]
            soc_q[i], T_q[i], v_meas[i], v_model[i] = G['SOC'][k], G['T'][k], G['V'][k], vlast
            hours[i] = (G['t'][k] - G['t'][0]) / 3600.0; Lr[i] = R.at[i, 'horizon_s']
            if 'W' in arms:                                                # 200 s window ending at the pulse end
                a = max(0, k + 1 - (LV.WINDOW - int(round(Lr[i]))))
                _, wst = step(m, sx, sy, X[a:k + 1][None], None, dev); WH[:, i], WC[:, i] = wst[0][:, 0], wst[1][:, 0]
    L = np.round(Lr).astype(int); d = R.direction.to_numpy(); zero = np.zeros(n)
    out = R[['row_id', 'evaluation_group', 'direction', 'temp_C', 'horizon_s', 'truth', 'start_kind']].copy()
    out['hours_from_start'], out['soc_q'], out['T_q'], out['v_meas'], out['v_model'] = hours, soc_q, T_q, v_meas, v_model
    out['soc_carried'] = ~np.isfinite(s['soc'][R.pre.to_numpy()])          # protocol: SOC continued across a recording break
    if 'M' in arms:
        out['pred_M'] = search(m, sx, sy, H, C, soc_q, T_q, d, L, zero, dev)
    if 'W' in arms:
        out['pred_W'] = search(m, sx, sy, WH, WC, soc_q, T_q, d, L, zero, dev)
    if 'O' in arms:
        out['pred_O'] = search(m, sx, sy, H, C, soc_q, T_q, d, L, v_meas - v_model, dev)
    return out


# ---------------------------------------------------------------- scoring
def tails(lr):
    q05, q50, q95 = np.quantile(lr, [0.05, 0.5, 0.95])
    return dict(bias=float(np.exp(q50)), over=float(np.exp(q95 - q50)), under=float(np.exp(q50 - q05)))


def closed(Bt, col, kind, tau, groups_all):
    parts, scales = [], {}
    for c in V5.CELLS:
        ev = Bt[Bt.evaluation_group == c].copy(); ev['cls'] = ev.direction if kind == 'class' else 'all'
        ev['central'] = ev[col]; ev['pred'] = np.nan
        for k in sorted(ev.cls.unique()):
            mk = (Bt.evaluation_group != c) & ((Bt.direction == k) if kind == 'class' else True)
            p, t = Bt.loc[mk, col].to_numpy(), Bt.loc[mk, 'truth'].to_numpy(); s, floor = V5.choose(p, t, tau)
            scales[f'{c}|{k}'] = dict(scale=s, floor_bound=floor, calibration_rows=int(mk.sum()), calibration_int_rate=float(np.mean(s * p > t + 1e-9)))
            ev.loc[ev.cls == k, 'pred'] = s * ev.loc[ev.cls == k, 'central']
        parts.append(ev)
    J = pd.concat(parts); assert len(J) == 672 and J.pred.notna().all()
    summ, G = V5.group_table(J, groups_all)
    return dict(tau=float(tau), groups_pass=int(G.passed.sum()), int_rate=summ['int_rate'], util_q50=summ['util_q50'], depth_q95=summ['depth_q95'],
                depth_max=summ['depth_max'], scales=scales, failed={f'{r.evaluation_group} {r.direction} {r.temp_C} C {r.horizon_s:g} s': r.reason
                                                                    for r in G[~G.passed].itertuples()}), G, J


def open_loop(Bt, col):
    r = Bt[col] / Bt.truth; z = int((Bt[col] <= 0).sum()); lr = np.log(np.maximum(r, 1e-3)); e = Bt[col] - Bt.truth
    rm = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))
    out = dict(rows=int(len(Bt)), zero_predictions=z, rel_bias_median_pct=float(100 * (r.median() - 1)), rel_abs_median_pct=float(100 * (r - 1).abs().median()),
               rel_abs_q95_pct=float(100 * (r - 1).abs().quantile(.95)), over_truth_frac=float((r > 1).mean()), tails=tails(lr),
               rmse_W=rm(e), rmse_W_by_cell={c: rm(e[Bt.evaluation_group == c]) for c in V5.CELLS if (Bt.evaluation_group == c).any()})
    b = Bt.evaluation_group == 'B'
    out['rmse_W_B_by_temp'] = {f'{T:g}C {h:g}s': rm(e[b & (Bt.temp_C == T) & (Bt.horizon_s == h)]) for T, h in
                               sorted(set(zip(Bt.temp_C[b], Bt.horizon_s[b])))}
    return out


def score():
    P = pd.concat([pd.read_csv(OUT / f'pred_{c}.csv') for c in CELLS]); B = V5.load()
    Bt = B[['row_id', 'evaluation_group', 'direction', 'temp_C', 'horizon_s', 'truth']].merge(P[['row_id', 'pred_M', 'pred_W', 'pred_O']], on='row_id', how='left')
    assert len(Bt) == 672 and Bt[['pred_M', 'pred_W', 'pred_O']].notna().all().all()
    groups_all = B[V5.C1.GK].drop_duplicates().sort_values(V5.C1.GK).reset_index(drop=True); assert len(groups_all) == 15
    res = {'reference_groups_tau_4_6': REF}; rows = []
    for arm, col, kind in (('M', 'pred_M', 'class'), ('M1', 'pred_M', 'single'), ('W', 'pred_W', 'class'), ('O', 'pred_O', 'class')):
        per = []
        for tau in TAUS:
            r, G, J = closed(Bt, col, kind, tau, groups_all); per.append(r)
            rows.append(dict(arm=arm, tau=float(tau), groups_pass=r['groups_pass'], int_rate=r['int_rate'], util_q50=r['util_q50']))
            if abs(tau - 0.05) < 1e-9:
                G.assign(arm=arm).to_csv(OUT / f'groups_tau5_{arm}.csv', index=False); J.to_csv(OUT / f'rows_tau5_{arm}.csv', index=False)
                res[f'{arm}_tau5'] = r
        gp = [p['groups_pass'] for p in per]
        res[f'{arm}_groups_tau_4_6'] = [int(min(gp)), float(np.median(gp)), int(max(gp))]
        res[f'{arm}_open_loop'] = open_loop(Bt, col)
    pd.DataFrame(rows).to_csv(OUT / 'tau_range.csv', index=False)
    R = OUT / 'pred_R.csv'
    if R.exists():
        Rr = pd.read_csv(R).merge(Bt[['row_id', 'pred_M']], on='row_id', how='left')
        res['R_vs_M_on_B10s'] = {'R (paper setting, own-cell SOP in training)': open_loop(Rr.assign(evaluation_group='B'), 'pred_R'),
                                 'M (fold B, rule-compliant)': open_loop(Rr.assign(evaluation_group='B'), 'pred_M')}
    med = res['M_groups_tau_4_6'][1]
    res['verdict'] = '제안 방법 우위' if med < 14 else '우위 없음'
    tr = {c: json.load(open(OUT / f'train_summary_{c}.json')) for c in CELLS}
    res['training'] = {c: dict(val_pooled_mV=t['best_val_rmse_pooled_mV'], val_condition_mean_mV=t['val_rmse_condition_mean_mV'], best_epoch=t['best_epoch'],
                               windows=t['windows']) for c, t in tr.items()}
    res['reproduction_flag'] = '재현 수준 미달' if any(t['best_val_rmse_pooled_mV'] > 35 for t in tr.values()) else 'ok'
    dump(OUT / 'summary.json', res)
    print(json.dumps({k: v for k, v in res.items() if k.endswith('tau_4_6') or k in ('verdict', 'reproduction_flag', 'training')}, ensure_ascii=False, indent=1))
    for arm in ('M', 'M1', 'W', 'O'):
        t = res[f'{arm}_tau5']; o = res[f'{arm}_open_loop']
        print(f"{arm}: tau5 groups {t['groups_pass']} int {100 * t['int_rate']:.2f}% util {100 * t['util_q50']:.1f}% | open loop bias {o['rel_bias_median_pct']:+.1f}% "
              f"|err| med {o['rel_abs_median_pct']:.1f}% q95 {o['rel_abs_q95_pct']:.1f}% over-tail {o['tails']['over']:.3f} RMSE {o['rmse_W']:.2f} W "
              f"by cell {json.dumps({k: round(v, 2) for k, v in o['rmse_W_by_cell'].items()})} zeros {o['zero_predictions']}", flush=True)
    if 'R_vs_M_on_B10s' in res:
        for k, o in res['R_vs_M_on_B10s'].items():
            print(f"{k}: |err| med {o['rel_abs_median_pct']:.1f}% q95 {o['rel_abs_q95_pct']:.1f}% bias {o['rel_bias_median_pct']:+.1f}% RMSE {o['rmse_W']:.2f} W")


# ---------------------------------------------------------------- commands
def pilot():
    """Engineering checks only (protocol): fold-B data, 5 training epochs, and the search on ten A1 rows (a training cell of
    fold B, so no B truth is read)."""
    out = OUT / 'pilot'; summ = train('B', epochs=5, out=out, tag='pilotB')
    s = stream('A1'); R = rows_of('A1', s); pick = R.groupby('direction').head(5).row_id.tolist()
    t0 = time.time(); P = infer_cell('A1', out / 'lstm_pilotB.pt', rows=pick)
    P.to_csv(out / 'pilot_search_A1.csv', index=False)
    print(P[['row_id', 'direction', 'temp_C', 'start_kind', 'hours_from_start', 'soc_q', 'T_q', 'v_meas', 'v_model', 'truth', 'pred_M', 'pred_W', 'pred_O']]
          .round(3).to_string(index=False), f'\nsearch {time.time() - t0:.1f} s for {len(P)} rows', flush=True)
    for c in CELLS:                                               # input coverage of every query (no truth read)
        s = stream(c); R = rows_of(c, s); seg = s['seg']; bad = []; kinds = {'H0': 0, 'segment': 0}
        for h, p, rid in zip(R.h0, R.pre, R.row_id):
            st0 = h if seg[h] == seg[p] else int(np.searchsorted(seg, seg[p], side='left')); kinds['H0' if st0 == h else 'segment'] += 1
            G = grid(s, st0, p, 'soc_inf')
            if not (np.isfinite(G['SOC']).all() and np.isfinite(G['T']).all() and np.isfinite(G['P']).all()):
                bad.append(rid)
        q = np.array([s['soc_inf'][p] for p in R.pre])
        print(f'{c}: rows {len(R)} start {kinds} non-finite inputs {len(bad)} {bad[:5]} | SOC at query min {np.nanmin(q):.3f} max {np.nanmax(q):.3f}', flush=True)


def main():
    cmd = sys.argv[1]
    if cmd == 'pilot':
        pilot()
    elif cmd == 'train':
        train(sys.argv[2])
    elif cmd == 'infer':
        c = sys.argv[2]; P = infer_cell(c, OUT / f'lstm_{c}.pt'); P.to_csv(OUT / f'pred_{c}.csv', index=False); print(P.describe().T.to_string())
    elif cmd == 'infer_r':
        s = stream('B'); R = rows_of('B', s); pick = R[R.horizon_s == 10].row_id.tolist()
        P = infer_cell('B', AUG_MODEL, rows=pick, arms=('M',)).rename(columns={'pred_M': 'pred_R'}); P.to_csv(OUT / 'pred_R.csv', index=False)
        print(len(P), 'rows', flush=True)
    elif cmd == 'score':
        score()


if __name__ == '__main__':
    main()
