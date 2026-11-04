"""V214 step 1: causal Ridge-SOH series for every stream of a pilot fold (PROTOCOL_MAMBA_V214.md).

Estimator: the paper's partial-charge dQ/dV ridge (analysis/soh_ridge.py): dQ/dV of a 1C constant-current
charge on 64 bins over 3.55-4.05 V, standardised inputs, alpha by leave-one-cell-out over the fitting cells only.
Training data: the paper's curve set (analysis/cache/soh_charge.npz, 1C characterisation charges, true SOH labels).
Per fold (held-out cell h):
  outer model  fitted on the five other UYPYDJ cells  -> SOH inputs of the held-out cell and of the Mendeley cell;
  inner model  for each training cell c, fitted on the four cells other than h and c -> SOH inputs of c, so the
               sequence model trains on estimates with the error it will meet on an unseen cell.
Detection in a cleaned series: a run of samples with charging current 2.85-3.15 A that starts at <= 3.55 V and reaches
>= 4.05 V, median cell temperature 20-30 C (the ridge was fitted at 25 C). Charge is integrated from the current. The
estimate becomes available at the sample where the run first reaches 4.05 V; the input is the median of the last five
run estimates, held until the next run."""
import json, sys
from pathlib import Path
import numpy as np
import data_v211 as D

RIDGE_DATA = Path(__file__).resolve().parents[3] / 'analysis/cache/soh_charge.npz'
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
V_LO, V_HI, N = 3.55, 4.05, 64


def _fit(X, y, a):
    mu, sd = X.mean(0), X.std(0) + 1e-8; Z = (X - mu) / sd; ym = y.mean()
    return np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - ym)), float(ym), mu, sd


def _pred(m, X):
    w, b, mu, sd = m; return ((X - mu) / sd) @ w + b


def fit_cells(X, y, cell, cells):
    sel = np.isin(cell, list(cells)); Xs, ys, cs = X[sel], y[sel], cell[sel]
    best, best_a = np.inf, ALPHAS[0]
    for a in ALPHAS:
        errs = [_pred(_fit(Xs[cs != c], ys[cs != c], a), Xs[cs == c]) - ys[cs == c] for c in sorted(set(cs))]
        r = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
        if r < best: best, best_a = r, a
    return _fit(Xs, ys, best_a), best_a


def features(V, I, t):
    """curve_features of the paper on one run: dQ/dV on the 3.55-4.05 V grid, charge from the integrated current."""
    q = np.r_[0., np.cumsum(I[1:] * np.diff(t))] / 3600.
    o = np.argsort(V, kind='stable'); Vs, qs = V[o], q[o]; Vs, k = np.unique(Vs, return_index=True); qs = qs[k]
    if len(Vs) < 40 or Vs.min() > V_LO or Vs.max() < V_HI: return None
    g = np.linspace(V_LO, V_HI, N + 1); return (np.diff(np.interp(g, Vs, qs)) / np.diff(g)).astype(np.float64)


def detect(s):
    """[(availability index, features, median temperature)] for the 1C runs of a cleaned series."""
    I, V, T, t, seg = s['I'], s['V'], s['T'], s['t'], s['seg']
    on = (I >= 2.85) & (I <= 3.15); out = []
    idx = np.flatnonzero(on)
    if not len(idx): return out
    brk = np.r_[True, (np.diff(idx) > 1) | (seg[idx][1:] != seg[idx][:-1])]
    starts = idx[brk]; ends = np.r_[idx[np.flatnonzero(brk)[1:] - 1], idx[-1]]
    for a, e in zip(starts, ends):
        if V[a] > V_LO or V[e] < V_HI: continue
        Tm = float(np.nanmedian(T[a:e + 1])) if np.isfinite(T[a:e + 1]).any() else np.nan
        if not (20. <= Tm <= 30.): continue
        f = features(V[a:e + 1].astype(float), I[a:e + 1].astype(float), t[a:e + 1])
        if f is None: continue
        avail = a + int(np.argmax(V[a:e + 1] >= V_HI))
        out.append((avail, f, Tm))
    return out


K_MEDIAN = 5  # causal median of the last 5 run estimates (raw estimates jump 2-3 %p between charges)


def hold(n, det, model):
    soh = np.zeros(n, np.float32); has = np.zeros(n, bool); est = []
    for avail, f, _ in sorted(det, key=lambda r: r[0]):
        est.append(float(_pred(model, f[None])[0]))
        soh[avail:] = float(np.median(est[-K_MEDIAN:])); has[avail:] = True
    return soh, has


def main():
    held = sys.argv[1]; out = D.HERE / 'runs' / f'v214_soh_{held}.npz'; assert not out.exists()
    Z = np.load(RIDGE_DATA); X, y, cell = Z['X'].astype(np.float64), Z['y'].astype(np.float64), Z['cell'].astype(str)
    train = [c for c in D.UYPYDJ if c != held]
    outer, a_out = fit_cells(X, y, cell, train)
    inner = {c: fit_cells(X, y, cell, [k for k in train if k != c]) for c in train}
    rep = dict(held_out=held, alpha_outer=a_out, alpha_inner={c: v[1] for c, v in inner.items()},
               outer_rmse_on_held_out_curves=(float(np.sqrt(np.mean((_pred(outer, X[cell == held]) - y[cell == held]) ** 2)))
                                              if held in D.UYPYDJ else None),   # held 'NONE': the all-cell deployment fit
               inner_rmse_on_own_curves={c: float(np.sqrt(np.mean((_pred(inner[c][0], X[cell == c]) - y[cell == c]) ** 2))) for c in train})
    arrays = {}; det_rep = {}
    for name in D.UYPYDJ + D.MENDELEY:
        s = D.load(name); det = detect(s)
        model = inner[name][0] if name in inner else outer
        soh, has = hold(len(s['t']), det, model); arrays[f'{name}__soh'] = soh; arrays[f'{name}__has'] = has
        d = dict(detections=len(det), first_available_h=float((s['t'][det[0][0]] - s['t'][0]) / 3600) if det else None,
                 held_fraction=float(has.mean()))
        if name in D.UYPYDJ and det:   # extraction check: nearest paper curve for every detected run of this cell
            Xc = X[cell == name]; F = np.stack([f for _, f, _ in det])
            dist = np.sqrt(((F[:, None, :] - Xc[None]) ** 2).mean(-1)) / np.sqrt((Xc ** 2).mean())
            d['paper_curves'] = int(len(Xc)); d['median_rel_distance_to_nearest_paper_curve'] = float(np.median(dist.min(1)))
            d['paper_curves_matched_within_2pct'] = int((dist.min(0) < .02).sum())
        det_rep[name] = d
    rep['streams'] = det_rep
    np.savez(out, **arrays); json.dump(rep, open(out.with_suffix('.json'), 'w'), indent=1); print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
