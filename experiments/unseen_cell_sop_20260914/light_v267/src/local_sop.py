"""sop_ident step 3b: model-free local SOP from the query's allowed window (relax_feats.py), leave one cell out.

    python tools/local_sop.py [--variants off prev ...]

Basis (out/feats/relax.csv, all from rows < q of the query's own sweep):
  The truth of a voltage-limited row is reproduced within about 3 % by P = V_lim * |V_lim - V_pre| / R_h, where R_h is
  the query pulse's own (V(h) - V_pre) / I(h) (diagnostic check, 2026-10-04). So the task is to predict R_h of the query
  at its own current. Estimates of R_h (h = horizon, 10 s or 2 s):
    off:  R_off,h of the interruption that precedes the pre-query rest (same SOC, ~300 s earlier; the 3 A / 1 A
          SOC-adjust discharge, or the CC-CV end at 0.15 A for the first query of a sweep)
    prev: R_h of the previous query pulse of the sweep (previous SOC point) times R_off,h / R_off,h(previous point)
  R_hat(I) = basis * kappa * (I / I_ref) ** beta, I_ref = |I_off| (off) or the previous pulse's current (prev);
  kappa, beta per region (score_loco regions) fitted on the OTHER cells' rows (log least squares on the measured R_h
  of their query pulses at their measured end current), so the evaluated cell's truths never enter its prediction.
  I solves I = H / R_hat(I), H = |V_lim - V_pre| -/+ the end-of-rest drift over the horizon; if I > I_max
  (15 A charge, 30 A discharge) the row is current-limited: P = I_max * (V_pre -/+ I_max * R_hat(I_max)).
Output: out/local/<variant>/pred.csv (row_id, pred_power_W, R_hat, I_hat, kappa, beta); folds.csv (fold c = the fit
without cell c for every row: the training cells in sample, for a deployment-faithful derating k)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT.parent / "pybamm/out/rpcwby_bench/rows.csv"
EV = ROOT.parent / "aging_refit/out/v2/events.csv"


def region(direction, soc):
    if direction == "discharge":
        return "dis<=0.1" if soc <= 0.1 + 1e-9 else ("dis0.15-0.3" if soc <= 0.3 + 1e-9 else "dis>=0.4")
    return "chg<=0.7" if soc <= 0.7 + 1e-9 else "chg>=0.8"


def table():
    R = pd.read_csv(BENCH)
    F = pd.read_csv(ROOT / "out/feats/relax.csv")
    E = pd.read_csv(EV)
    D = R.merge(F, on="row_id").merge(E[["member", "event_id", "I_last", "V_min", "V_max", "T_pre"]],
                                       left_on=["member", "q_row"], right_on=["member", "event_id"], how="left")
    D["T_pre"] = D.T_pre.fillna(D.T_pre_raw)                     # rows without an event: the cell at q - 1
    cv = ROOT / "out/feats/cv.csv"                               # the sweep's CC-CV recharge (cv_feats.py)
    if cv.exists():
        D = D.merge(pd.read_csv(cv), on="sweep_id", how="left")
    D["chg"] = D.direction == "charge"
    D["Vlim"] = np.where(D.chg, 4.15, 2.55)
    D["Imax"] = np.where(D.chg, 15.0, 30.0)
    D["region"] = [region(d, s) for d, s in zip(D.direction, D.soc_point)]
    h10 = D.horizon_s >= 5
    D["Roff"] = np.where(h10, D.Roff10, D.Roff2)
    D["Rstep"] = np.where(h10, D.Rstep10, D.Rstep2)
    D["prevRstep"] = np.where(h10, D.prev_Rstep10, D.prev_Rstep2)
    D["prevR"] = np.where(h10, D.prev_R10, D.prev_R2)
    D["prevRoff"] = np.where(h10, D.prev_Roff10, D.prev_Roff2)
    D["Rq"] = np.where(h10, D.R10_q, D.R2_q)                   # diagnostic / training target only
    D["Iq"] = D.truth_power_W / D.Vlim                           # training target only (end current of the truth)
    drift = 1e-3 * D.slope_mVs.fillna(0.0) * D.horizon_s
    D["H"] = np.where(D.chg, D.Vlim - D.V_pre - drift, D.V_pre + drift - D.Vlim)
    # training only: how far the truth pulse ended from the limit (the applied power comes from the tester's earlier
    # probe, so it leaves a gap; median 2.6 mV for charge rows) [V], > 0 = inside the limit
    D["gap"] = np.where(D.chg, D.Vlim - D.V_max, D.V_min - D.Vlim)
    return D


def with_gap(D, train_mask):
    """Headroom and end voltage with the median truth gap of the training rows of each direction (voltage-limited)."""
    D = D.copy()
    for chg in (True, False):
        m = train_mask & (D.chg == chg) & (D.Iq < 0.98 * D.Imax)
        g = float(np.nanmedian(D.loc[m, "gap"])) if m.any() else 0.0
        sel = D.chg == chg
        D.loc[sel, "H"] = D.loc[sel, "H"] - g
        D.loc[sel, "Vlim"] = D.loc[sel, "Vlim"] - (g if chg else -g)
    return D


def basis(D, kind):
    if kind == "off":
        return D.Roff, D.I_end.abs()
    if kind == "prev":
        b = D.prevR * D.Roff / D.prevRoff
        return b.where(np.isfinite(b), D.Roff), D.prev_I_last.abs().where(np.isfinite(b), D.I_end.abs())
    raise ValueError(kind)


def fit(T, kind, fit_beta):
    B, Iref = basis(T, kind)
    y = np.log(T.Rq / B)
    x = np.log(T.Iq / Iref)
    ok = np.isfinite(y) & np.isfinite(x) & (T.Iq < 0.98 * T.Imax)
    out = {}
    for r, m in T[ok].groupby("region").groups.items():
        yy, xx = y[m], x[m]
        if fit_beta and len(m) >= 8 and xx.std() > 0.05:
            beta, lk = np.polyfit(xx, yy, 1)
        else:
            beta, lk = 0.0, float(np.median(yy))
        out[r] = (float(np.exp(lk)), float(beta))
    return out


def predict(D, kind, par):
    B, Iref = basis(D, kind)
    res = []
    for i, r in enumerate(D.itertuples()):
        kap, beta = par.get(r.region, (1.0, 0.0))
        b, ir = B.iat[i], Iref.iat[i]
        if not (np.isfinite(b) and b > 0 and np.isfinite(ir) and ir > 0) or r.H <= 0:
            res.append((np.nan if r.H > 0 else 0.0, np.nan, np.nan))
            continue
        Rh = lambda I: b * kap * (max(I, 1e-3) / ir) ** beta
        I = r.H / Rh(ir)
        for _ in range(50):
            I_new = r.H / Rh(I)
            if abs(I_new - I) < 1e-6 * max(1.0, I):
                break
            I = 0.5 * (I + I_new)
        if I > r.Imax:
            Vend = r.V_pre - r.Imax * Rh(r.Imax) if not r.chg else r.V_pre + r.Imax * Rh(r.Imax)
            res.append((r.Imax * Vend, Rh(r.Imax), r.Imax))
        else:
            res.append((r.Vlim * I, Rh(I), I))
    return pd.DataFrame(res, columns=["pred_power_W", "R_hat", "I_hat"], index=D.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["off0", "off", "prev0", "prev"])
    a = ap.parse_args()
    D = table()
    for v in a.variants:
        kind, fb = v.rstrip("0"), not v.endswith("0")
        parts, folds = [], []
        for c in sorted(D.evaluation_group.unique()):
            par = fit(D[D.evaluation_group != c], kind, fb)
            P = predict(D, kind, par)                       # fold c: every row from the fit without cell c
            P["kappa"] = [par.get(r, (np.nan, np.nan))[0] for r in D.region]
            P["beta"] = [par.get(r, (np.nan, np.nan))[1] for r in D.region]
            P = pd.concat([D[["row_id"]], P], axis=1)
            m = (D.evaluation_group == c).to_numpy()
            parts.append(P[m])
            folds.append(P.assign(fold=c))
        out = ROOT / "out/local" / v
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(parts).to_csv(out / "pred.csv", index=False)
        pd.concat(folds).to_csv(out / "folds.csv", index=False)
        print(f"wrote {out / 'pred.csv'}")


if __name__ == "__main__":
    main()
