"""V266 (PROTOCOL_CHEN_V266.md): the Chen 2026 reproduction (V265) pushed to the SOP level before comparing again.
Changes against V265: every fifth sweep (H0 group, index % 5 == 2) of each training cell is held out; checkpoints are chosen by
the SOP error on those held-out rows (200 s window form, same bisection), patience 20 checks; candidate recipes C1 (half of
the SOP windows end inside a constant-power pulse), C2 (C1 + current-weighted RMSE), C3 (C1 with SOP windows = 1.0 x drive).
R2 = the paper's own setting (Mendeley Mixed + Test#3 2/30 s SOP, both channels) with the chosen recipe, scored on B 10 s.
usage: python chen_v266.py pilot <cand> | train <fold> <cand> | infer <fold> | r2 <cand> <epochs> | score <cand>"""
import json, sys, time
import numpy as np, pandas as pd, torch
import chen_v265 as C
from run_envelope import HERE, dump
from data30t import test3_windows_source

LV, V5 = C.LV, C.V5
OUT = HERE / 'chen_v266'
CANDS = {'C1': dict(f_pulse=0.5, weight=False, ratio=0.5), 'C2': dict(f_pulse=0.5, weight=True, ratio=0.5),
         'C3': dict(f_pulse=0.5, weight=False, ratio=1.0)}
TEST3_ZIP = C.ANALYSIS.parent / 'raw' / 'RPCWBY' / '3_Test_3.zip'
GATE_ERR, GATE_DROP = 0.10, (0.85, 1.15)


def clean(o):
    """NaN/inf -> None so that run_envelope.dump (allow_nan=False) never stops a finished run."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------------------------------------------------------- data
def plateaus(t, P, I):
    """Constant-power plateaus (amendment 1): runs of |I| > 0.2 A in which the power changes by <= max(2 %, 0.05 W) from sample
    to sample, lasting 0.9-31 s with |mean P| > 0.3 W. Returns (start, end) times; start = first plateau sample - 1 s (Arbin
    logs a step's first sample about 1 s after the step begins, V261)."""
    load = np.abs(I) > 0.2; dP = np.abs(np.diff(P)); tol = np.maximum(0.02 * np.abs(P[:-1]), 0.05)
    start = load & np.r_[True, ~load[:-1] | (dP > tol)]
    rid = np.cumsum(start) - 1; idx = np.flatnonzero(load)
    if not len(idx):
        return np.zeros(0), np.zeros(0)
    r = rid[idx]; first = idx[np.r_[True, r[1:] != r[:-1]]]; last = idx[np.r_[r[1:] != r[:-1], True]]
    cnt = last - first + 1; dur = t[last] - t[first]
    mean = np.add.reduceat(P[idx], np.flatnonzero(np.r_[True, r[1:] != r[:-1]])) / cnt
    keep = (cnt >= 2) & (dur >= 0.9) & (dur <= 31.0) & (np.abs(mean) > 0.3)
    return t[first[keep]] - 1.0, t[last[keep]]


def in_intervals(tg, a, b):
    """Grid times inside any [a, b] interval (a sorted)."""
    if not len(a):
        return np.zeros(len(tg), bool)
    j = np.searchsorted(a, tg, side='right') - 1; ok = j >= 0
    return ok & (tg <= b[np.where(ok, j, 0)])


def held_out(R):
    """Every fifth H0 group in time order (index % 5 == 2) is held out of training for internal validation."""
    g = np.sort(R.h0.unique()); return R.h0.isin(set(g[2::5])).to_numpy()


def sop_windows(cells, n_take, f_pulse, rng):
    """200 s windows from the SOP periods of the training cells minus their held-out sweeps; round(f_pulse n_take) windows end
    on a pulse sample, the rest uniformly over all eligible ends (both without replacement)."""
    pieces, ends, pend, owner, info = [], [], [], [], {}
    for c in cells:
        s = C.stream(c); R = C.rows_of(c, s); n = len(s['t']); hold = held_out(R)
        mask, excl = np.zeros(n, bool), np.zeros(n, bool)
        for a, q, h in zip(np.minimum(R.bo, R.h0), R.qp, hold):
            (excl if h else mask)[a:min(q + C.SPAN_AFTER, n)] = True
        mask &= ~excl
        cut = np.r_[True, (mask[1:] != mask[:-1]) | (s['seg'][1:] != s['seg'][:-1])]
        st = np.flatnonzero(cut); en = np.r_[st[1:] - 1, n - 1]; nw = npw = 0
        for a, b in zip(st, en):
            if not mask[a] or b - a < 10:
                continue
            g = C.grid(s, a, b); feats = np.stack([g['SOC'], g['T'], g['P']], 1).astype(np.float32)
            ok = np.isfinite(feats).all(1) & np.isfinite(g['V']); bad = np.cumsum(~ok); k = np.arange(LV.WINDOW - 1, len(ok))
            good = k[(bad[k] - np.r_[0, bad][k - LV.WINDOW + 1]) == 0]
            if not len(good):
                continue
            pa, pb = plateaus(s['t'][a:b + 1], (s['V'][a:b + 1] * s['I'][a:b + 1]).astype(np.float64), s['I'][a:b + 1].astype(np.float64))
            pm = in_intervals(g['t'], pa, pb); pieces.append((feats, g['V'].astype(np.float32), np.abs(g['I']).astype(np.float32))); owner.append(c)
            pid = len(pieces) - 1; ends.append(np.stack([np.full(len(good), pid), good], 1)); gp = good[pm[good]]
            pend.append(np.stack([np.full(len(gp), pid), gp], 1)); nw += len(good); npw += len(gp)
        info[c] = dict(eligible_windows=int(nw), pulse_end_windows=int(npw), held_out_sweeps=int(len(np.unique(R.h0[hold]))),
                       train_sweeps=int(len(np.unique(R.h0[~hold]))))
    E, Ep = np.concatenate(ends), np.concatenate(pend)
    n_p = min(int(round(f_pulse * n_take)), len(Ep)); n_u = n_take - n_p
    sel = np.concatenate([Ep[rng.choice(len(Ep), size=n_p, replace=False)], E[rng.choice(len(E), size=n_u, replace=False)]])
    X = np.stack([pieces[p][0][k - LV.WINDOW + 1:k + 1] for p, k in sel]); Y = np.array([pieces[p][1][k] for p, k in sel], np.float32)
    Iend = np.array([pieces[p][2][k] for p, k in sel], np.float32); own = np.array(owner)[sel[:, 0]]
    info.update(taken=int(len(sel)), taken_pulse_end=int(n_p), taken_by_cell={c: int((own == c).sum()) for c in cells},
                frac_I_end_gt_20A=float(np.mean(Iend > 20)))
    return X, Y, Iend, info


def r2_windows(n_take, f_pulse, rng):
    """The paper's setting: Test#3 2 s and 30 s SOP runs (both channels, dataset SOC column) as in the August reproduction."""
    runs = test3_windows_source(str(TEST3_ZIP), pulse_lengths=(2, 30)); pieces, ends, pend = [], [], []
    for r in runs:
        feats = np.stack([r['SOC'], r['T'], r['P']], 1).astype(np.float32); V = r['V'].astype(np.float32); I = np.abs(r['P'] / np.maximum(r['V'], 1.0))
        good = np.arange(LV.WINDOW - 1, len(V)); tt = np.arange(len(V), dtype=float); pa, pb = plateaus(tt, r['P'].astype(np.float64), I.astype(np.float64))
        pm = in_intervals(tt, pa, pb)
        pieces.append((feats, V, I.astype(np.float32))); pid = len(pieces) - 1
        ends.append(np.stack([np.full(len(good), pid), good], 1)); gp = good[pm[good]]; pend.append(np.stack([np.full(len(gp), pid), gp], 1))
    E, Ep = np.concatenate(ends), np.concatenate(pend)
    n_p = min(int(round(f_pulse * n_take)), len(Ep)); n_u = min(n_take - n_p, len(E))
    sel = np.concatenate([Ep[rng.choice(len(Ep), size=n_p, replace=False)], E[rng.choice(len(E), size=n_u, replace=False)]])
    X = np.stack([pieces[p][0][k - LV.WINDOW + 1:k + 1] for p, k in sel]); Y = np.array([pieces[p][1][k] for p, k in sel], np.float32)
    Iend = np.array([pieces[p][2][k] for p, k in sel], np.float32)
    return X, Y, Iend, dict(runs=len(runs), eligible_windows=int(len(E)), pulse_end_windows=int(len(Ep)), taken=int(len(sel)), taken_pulse_end=int(n_p))


def internal_val(cells):
    """Held-out sweeps' benchmark rows of the training cells, as 200 s window inputs on a grid aligned to the pre-pulse sample,
    plus the measured pulse inputs (for the drop ratio)."""
    recs = []
    for c in cells:
        s = C.stream(c); R = C.rows_of(c, s); R = R[held_out(R)]; seg = s['seg']
        for r in R.itertuples():
            a = r.h0 if seg[r.h0] == seg[r.pre] else int(np.searchsorted(seg, seg[r.pre], side='left')); b = min(r.qp + 60, len(s['t']) - 1)
            f = lambda tq, k: np.interp(tq, s['t'][a:b + 1], s[k][a:b + 1].astype(np.float64))
            L = int(round(r.horizon_s)); tp = s['t'][r.pre]
            tq = tp + np.arange(-(LV.WINDOW - L) + 1, 1, dtype=float); tpul = tp + np.arange(1, L + 1, dtype=float)
            X = np.stack([f(tq, 'soc_inf'), f(tq, 'Tf'), f(tq, 'V') * f(tq, 'I')], 1); Xp = np.stack([f(tpul, 'soc_inf'), f(tpul, 'Tf'), f(tpul, 'V') * f(tpul, 'I')], 1)
            recs.append(dict(cell=c, row_id=r.row_id, direction=r.direction, L=L, truth=float(r.truth), X=X, Xp=Xp, Vpre=float(f(tq[-1:], 'V')[0]), Vend=float(f(tpul[-1:], 'V')[0])))
    return recs


@torch.no_grad()
def internal_eval(m, sx, sy, IV, dev='cuda'):
    """SOP error and drop ratio on the internal validation rows (window form)."""
    out = []
    for L in sorted({r['L'] for r in IV}):
        rs = [r for r in IV if r['L'] == L]; X = np.stack([r['X'] for r in rs]); Xp = np.stack([r['Xp'] for r in rs])
        v_pre, st = C.step(m, sx, sy, X, None, dev); d = np.array([r['direction'] for r in rs])
        pred = C.search(m, sx, sy, st[0], st[1], X[:, -1, 0].copy(), X[:, -1, 1].copy(), d, np.full(len(rs), L), np.zeros(len(rs)), dev)
        v_pul, _ = C.step(m, sx, sy, Xp, st, dev)
        for i, r in enumerate(rs):
            out.append(dict(cell=r['cell'], row_id=r['row_id'], direction=r['direction'], L=L, truth=r['truth'], pred=float(pred[i]),
                            drop_ratio=float((v_pul[i, -1] - v_pre[i, -1]) / (r['Vend'] - r['Vpre'])) if abs(r['Vend'] - r['Vpre']) > 1e-3 else np.nan))
    D = pd.DataFrame(out); D['abs_err'] = (D.pred / D.truth - 1).abs()
    summ = dict(err_med=float(D.abs_err.median()), err_med_by_dir={k: float(v) for k, v in D.groupby('direction').abs_err.median().items()},
                bias_med_by_dir={k: float(v) for k, v in D.assign(r=D.pred / D.truth - 1).groupby('direction').r.median().items()},
                drop_by_dir={k: float(v) for k, v in D.groupby('direction').drop_ratio.median().items()}, rows=int(len(D)))
    summ['gate'] = bool(summ['err_med'] <= GATE_ERR and all(GATE_DROP[0] <= v <= GATE_DROP[1] for v in summ['drop_by_dir'].values()))
    return summ, D


# ---------------------------------------------------------------- training
def fit(tag, out, Xtr, Ytr, Itr, cfg, epochs, stop_at, IV, Xva, Yva, mva, info, fold, fixed=False):
    """V265 training loop with the exact (optionally current-weighted) RMSE gradient over 4 x 1250 micro-batches; checkpoint
    chosen by the internal-validation SOP error (patience 20 checks), or the last epoch when fixed=True (R2)."""
    torch.manual_seed(LV.SEED); np.random.seed(LV.SEED); dev = 'cuda'; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    sx, sy = LV.MinMax(Xtr), LV.MinMax(Ytr.reshape(-1, 1))
    Xt = torch.from_numpy(sx(Xtr)).to(dev); Yt = torch.from_numpy(sy(Ytr.reshape(-1, 1)).ravel()).to(dev); Xva_t = torch.from_numpy(sx(Xva))
    Wt = torch.from_numpy((1.0 + (Itr / 10.0) ** 2) if cfg['weight'] else np.ones(len(Itr))).float().to(dev)
    model = LV.VoltageLSTM().to(dev); opt = torch.optim.Adam(model.parameters(), lr=LV.LR)
    miles = [int(epochs * f) for f in (0.2, 0.4, 0.6, 0.8)]; sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=miles, gamma=LV.GAMMA)
    S = lambda d: C.Scaler(d.to_dict()); ckpt = out / f'lstm_{tag}.pt'; n = len(Xt); best, best_ep, bad, log, last_ep = np.inf, -1, 0, [], 0
    print(f'[{tag}] windows {n:,} (val {len(Xva):,}); {json.dumps(info)}; milestones {miles}; stop at {stop_at}; fixed {fixed}', flush=True)
    for ep in range(1, stop_at + 1):
        last_ep = ep; te = time.time(); model.train(); perm = torch.randperm(n).to(dev); tot = torch.zeros((), device=dev)
        for k in range(0, n, LV.BATCH):
            b = perm[k:k + LV.BATCH]; wsum = Wt[b].sum(); opt.zero_grad(set_to_none=True); sse = torch.zeros((), device=dev)
            for j in range(0, len(b), C.MICRO):
                bb = b[j:j + C.MICRO]; part = (Wt[bb] * (model(Xt[bb]) - Yt[bb]) ** 2).sum(); (part / wsum).backward(); sse += part.detach()
            rmse = torch.sqrt(sse / wsum)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(0.5 / rmse)
            opt.step(); tot += rmse * len(b)
        sched.step()
        if ep % C.VAL_EVERY == 0 or ep == 1 or ep == stop_at:
            model.eval(); v = C.val_rmse(model, Xva_t, Yva, sy, dev); rec = dict(epoch=ep, train=(tot / n).item(), val_rmse_mV=v, lr=sched.get_last_lr()[0], sec=time.time() - te)
            state = {'model': model.state_dict(), 'sx': sx.to_dict(), 'sy': sy.to_dict(), 'window': LV.WINDOW, 'fold': fold, 'epoch': ep}
            if IV is not None:
                summ, _ = internal_eval(model, S(sx), S(sy), IV, dev); rec.update(err_med=summ['err_med'], gate=summ['gate'],
                                                                                **{f'drop_{k}': v_ for k, v_ in summ['drop_by_dir'].items()},
                                                                                **{f'err_{k}': v_ for k, v_ in summ['err_med_by_dir'].items()})
                flag = ''
                if summ['err_med'] < best - 1e-9:
                    best, best_ep, bad, flag = summ['err_med'], ep, 0, ' *best'; torch.save(state, ckpt)
                else:
                    bad += 1
            else:
                flag = ''; torch.save(state, ckpt); best_ep = ep
            log.append(rec)
            print(f'[{tag}] ep {ep:4d} ' + ' '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in rec.items() if k != 'epoch') + flag, flush=True)
            if IV is not None and bad >= C.PATIENCE:
                print(f'[{tag}] early stop at epoch {ep}', flush=True); break
    pd.DataFrame(log).to_csv(out / f'train_log_{tag}.csv', index=False)
    ck = torch.load(ckpt); model.load_state_dict(ck['model']); model.eval()
    pv = C.predict_windows(model, Xva_t, sy, dev); cond = pd.DataFrame({'temp_C': [m_[0] for m_ in mva], 'cycle': [m_[1] for m_ in mva], 'err': pv - Yva})
    per = cond.groupby(['temp_C', 'cycle']).err.apply(lambda e: float(np.sqrt(np.mean(e ** 2))) * 1e3)
    np.savez_compressed(out / f'val_pred_{tag}.npz', pred=pv, truth=Yva)
    summ = dict(tag=tag, fold=fold, cand=cfg, best_epoch=best_ep, epochs_run=last_ep, val_rmse_pooled_mV=C.val_rmse(model, Xva_t, Yva, sy, dev),
                val_rmse_condition_mean_mV=float(per.mean()), windows=int(n), info=info, minutes=(time.time() - t0) / 60)
    if IV is not None:
        isum, D = internal_eval(model, C.Scaler(sx.to_dict()), C.Scaler(sy.to_dict()), IV, dev); D.to_csv(out / f'internal_val_{tag}.csv', index=False)
        summ['internal_at_best'] = isum
    dump(out / f'train_summary_{tag}.json', clean(summ)); print(f'[{tag}] done {json.dumps(summ)}', flush=True)
    return summ


def train(fold, cand, stop_at=C.EPOCHS, out=OUT, tag=None):
    cfg = CANDS[cand]; tr = [c for c in C.CELLS if c != fold]
    Xd, Yd, Xva, Yva, mva, _, _ = C.mendeley_windows(); Id = np.abs(Xd[:, -1, 2] / np.maximum(Yd, 1.0))
    Xs, Ys, Is, info = sop_windows(tr, int(len(Xd) * cfg['ratio']), cfg['f_pulse'], np.random.default_rng(LV.SEED))
    IV = internal_val(tr); info['internal_rows'] = len(IV)
    return fit(tag or fold, out, np.concatenate([Xd, Xs]), np.concatenate([Yd, Ys]), np.concatenate([Id, Is]), cfg, C.EPOCHS, stop_at, IV, Xva, Yva, mva, info, fold)


def train_r2(cand, n_epochs, out=OUT):
    cfg = CANDS[cand]; Xd, Yd, Xva, Yva, mva, _, _ = C.mendeley_windows(); Id = np.abs(Xd[:, -1, 2] / np.maximum(Yd, 1.0))
    Xs, Ys, Is, info = r2_windows(int(len(Xd) * cfg['ratio']), cfg['f_pulse'], np.random.default_rng(LV.SEED))
    return fit('R2', out, np.concatenate([Xd, Xs]), np.concatenate([Yd, Ys]), np.concatenate([Id, Is]), cfg, C.EPOCHS, n_epochs, None, Xva, Yva, mva, info, 'R2', fixed=True)


# ---------------------------------------------------------------- scoring
def score(cand):
    P = pd.concat([pd.read_csv(OUT / f'pred_{c}.csv') for c in C.CELLS]); B = V5.load()
    Bt = B[['row_id', 'evaluation_group', 'direction', 'temp_C', 'horizon_s', 'truth']].merge(P[['row_id', 'pred_M', 'pred_W', 'pred_O']], on='row_id', how='left')
    assert len(Bt) == 672 and Bt[['pred_M', 'pred_W', 'pred_O']].notna().all().all()
    groups_all = B[V5.C1.GK].drop_duplicates().sort_values(V5.C1.GK).reset_index(drop=True); assert len(groups_all) == 15
    res = {'candidate': cand, 'recipe': CANDS[cand], 'reference_groups_tau_4_6': C.REF}; rows = []
    for arm, col, kind in (('M', 'pred_M', 'class'), ('M1', 'pred_M', 'single'), ('W', 'pred_W', 'class'), ('O', 'pred_O', 'class')):
        per = []
        for tau in C.TAUS:
            r, G, J = C.closed(Bt, col, kind, tau, groups_all); per.append(r)
            rows.append(dict(arm=arm, tau=float(tau), groups_pass=r['groups_pass'], int_rate=r['int_rate'], util_q50=r['util_q50']))
            if abs(tau - 0.05) < 1e-9:
                G.assign(arm=arm).to_csv(OUT / f'groups_tau5_{arm}.csv', index=False); J.to_csv(OUT / f'rows_tau5_{arm}.csv', index=False); res[f'{arm}_tau5'] = r
        gp = [p['groups_pass'] for p in per]; res[f'{arm}_groups_tau_4_6'] = [int(min(gp)), float(np.median(gp)), int(max(gp))]
        res[f'{arm}_open_loop'] = C.open_loop(Bt, col)
    pd.DataFrame(rows).to_csv(OUT / 'tau_range.csv', index=False)
    tr = {c: json.load(open(OUT / f'train_summary_{c}.json')) for c in C.CELLS}
    res['training'] = {c: dict(best_epoch=t['best_epoch'], epochs_run=t['epochs_run'], val_pooled_mV=t['val_rmse_pooled_mV'], internal=t['internal_at_best'])
                       for c, t in tr.items()}
    res['gate_all_folds'] = bool(all(t['internal_at_best']['gate'] for t in tr.values()))
    res['verdict'] = ('제안 방법 우위' if res['M_groups_tau_4_6'][1] < 14 else '우위 없음') + ('' if res['gate_all_folds'] else ' (재현 관문 미달)')
    rp = OUT / 'pred_R2.csv'
    if rp.exists():
        R2 = pd.read_csv(rp).merge(Bt[['row_id', 'pred_M']], on='row_id', how='left'); r = R2.pred_R2 / R2.truth
        res['R2'] = dict(open_loop=C.open_loop(R2.assign(evaluation_group='B'), 'pred_R2'), M_same_rows=C.open_loop(R2.assign(evaluation_group='B'), 'pred_M'),
                         label='SOP 수준 재현' if float((r - 1).abs().median()) <= 0.10 else 'SOP 수준 재현 미달', paper_rmse_10s_W=1.87)
    dump(OUT / 'summary.json', clean(res))
    for arm in ('M', 'M1', 'W', 'O'):
        t = res[f'{arm}_tau5']; o = res[f'{arm}_open_loop']
        print(f"{arm}: groups tau4-6 {res[f'{arm}_groups_tau_4_6']} | tau5 int {100 * t['int_rate']:.2f}% util {100 * t['util_q50']:.1f}% | bias {o['rel_bias_median_pct']:+.1f}% "
              f"|err| med {o['rel_abs_median_pct']:.1f}% q95 {o['rel_abs_q95_pct']:.1f}% tail {o['tails']['over']:.3f} RMSE {o['rmse_W']:.2f} W {json.dumps({k: round(v, 2) for k, v in o['rmse_W_by_cell'].items()})}", flush=True)
    print('verdict:', res['verdict'], '| gates', {c: v['internal']['gate'] for c, v in res['training'].items()}, flush=True)
    if 'R2' in res:
        print('R2:', res['R2']['label'], json.dumps({k: res['R2']['open_loop'][k] for k in ('rel_bias_median_pct', 'rel_abs_median_pct', 'rmse_W')}), flush=True)


def main():
    cmd = sys.argv[1]
    if cmd == 'count':                                           # pulse-end window counts only (no training)
        X, Y, I, info = sop_windows(['A1', 'A17'], 65799, 0.5, np.random.default_rng(LV.SEED)); print('fold B', json.dumps(info))
        for f in (['A17', 'B'], ['A1', 'B']):
            X, Y, I, info = sop_windows(f, 65799, 0.5, np.random.default_rng(LV.SEED)); print(f, json.dumps(info))
        X, Y, I, info = r2_windows(65799, 0.5, np.random.default_rng(LV.SEED)); print('R2', json.dumps(info))
    elif cmd == 'smoke':                                           # all code paths, 2 epochs, no results used
        train('B', 'C2', stop_at=2, out=OUT / 'smoke', tag='smoke_B'); train_r2('C2', 2, out=OUT / 'smoke')
        c = 'B'; P = C.infer_cell(c, OUT / 'smoke' / 'lstm_smoke_B.pt', rows=None); print('infer rows', len(P), P[['pred_M', 'pred_W', 'pred_O']].notna().all().to_dict())
    elif cmd == 'pilot':
        train('B', sys.argv[2], stop_at=300, out=OUT / 'pilot', tag=f'B_{sys.argv[2]}')
    elif cmd == 'train':
        train(sys.argv[2], sys.argv[3])
    elif cmd == 'infer':
        c = sys.argv[2]; P = C.infer_cell(c, OUT / f'lstm_{c}.pt'); P.to_csv(OUT / f'pred_{c}.csv', index=False)
        print(P[['truth', 'pred_M', 'pred_W', 'pred_O']].describe().T.to_string(), flush=True)
    elif cmd == 'r2':
        train_r2(sys.argv[2], int(sys.argv[3]))
        s = C.stream('B'); R = C.rows_of('B', s); pick = R[R.horizon_s == 10].row_id.tolist()
        P = C.infer_cell('B', OUT / 'lstm_R2.pt', rows=pick, arms=('M',)).rename(columns={'pred_M': 'pred_R2'}); P.to_csv(OUT / 'pred_R2.csv', index=False)
    elif cmd == 'score':
        score(sys.argv[2])


if __name__ == '__main__':
    main()
