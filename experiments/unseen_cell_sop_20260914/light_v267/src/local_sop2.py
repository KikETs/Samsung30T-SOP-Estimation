"""sop_ident step 3c: local SOP with a log-linear model of the query's R_h, leave one cell out (extends local_sop.py).

    python tools/local_sop2.py --name NAME --spec REGION@BASIS:FEAT,FEAT,... [REGION@...] [--subregion-first]

Per region (score_loco regions; with --subregion-first the first query of a sweep, i.e. no earlier query pulse in
its direction, is its own region "<region>|first", or "<region>|firstcv" when the last load was the CC-CV charge):
  log R_hat(I) = log basis + c0 + sum_j c_j x_j + beta log(I / I_ref)
  basis off  = R_off,h (interruption before the pre-query rest),            I_ref = |I_off|
  basis prev = R_h of the previous query pulse,                              I_ref = its end current
  basis prevamp = prev R_h * exp(ampT) (ampT below),                    I_ref = the previous pulse's end current
  a basis written with "!" (e.g. off!) is not fitted: I = H / basis (the CV-end first queries: a small-current response
  at the same SOC, so no learned factor; no training cell has such rows for A1)
  basis step = Rstep_h of the interruption (sqrt(t) superposition with the load duration, relax_feats.py), I_ref = |I_off|
  features (all from rows < q): trend10 = log(R_off,10 / R_off,10 of the previous point), trend1 (same at 1 s),
    diff = log(R_off,10 / R_off,1), pdiff = log(prev R_10 / prev R_1), prat = log(prev R_10 / prev R_off,10),
    slope = end-of-rest dV/dt [mV/s], late = log(R_off,240 / R_off,10), invT = 1000 / (T_pre + 273.15) - 3.354,
    lRoff = log(R_off,10 / 0.03), trendS / trendS1 = log(Rstep_h / previous Rstep_h) (at h / 1 s),
    lT = log(load duration / 200 s), dlT = log(duration / previous duration), sqA = log(diffusion share of Rstep10),
    obs = log(prev R_h / prev2 R_h), trS2 = log(prev Rstep_h / prev2 Rstep_h), ext = obs * clip(dAh ratio, 0, 3),
    amp = trendS * clip(obs / trS2, 0, 3) (the pulse / interruption trend ratio of the previous step; 1 if |trS2| <= 0.05),
    ampT = trendS * clip((obs trS2 + 0.02) / (trS2^2 + 0.02), 0, 3) (that ratio shrunk towards 1),
    ltcv / ltcc / ltau = log of the sweep's CV duration / CC duration / late CV decay time (cv_feats.py)
  c, beta: ordinary least squares on the OTHER cells' rows of the region (target: the measured R_h of their query
  pulses, current: their truth end current; current-limited rows left out). A region with fewer than
  max(12, 4 * n_coef) training rows falls back to basis * median ratio.
  I solves I * R_hat(I) = H in closed form (beta clipped to [-0.9, 1]); current-limited rows as in local_sop.py.
  --direct (2026-10-04 22:3x): the log-R fit above has the truth current on both sides (R_q I_q ~ H for a voltage-limited
  row), which biases beta towards -1; --direct instead regresses log(I_q / I_ref) on u = log(H / (basis I_ref)) and the
  features (fit_region_direct), i.e. on the quantity that is predicted.
  --gap: headroom H and end voltage less the median gap between the truth pulse's end voltage and the limit, taken
  from the training cells' rows of the same direction (local_sop.with_gap); it matters where H is a few mV (SOC 1.0
  charge).
Output: out/local/<name>/pred.csv (each cell from the fit without it), folds.csv (fold c = the fit without cell c,
predicting every row: cell c out of sample, the training cells in sample, for a deployment-faithful derating k),
coefficients in out/local/<name>/coef.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_sop as ls  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def features(D):
    X = pd.DataFrame(index=D.index)
    X["trend10"] = np.log(D.Roff10 / D.prev_Roff10)
    X["trend1"] = np.log(D.Roff1 / D.prev_Roff1)
    X["diff"] = np.log(D.Roff10 / D.Roff1)
    X["pdiff"] = np.log(D.prev_R10 / D.prev_R1)
    X["prat"] = np.log(D.prev_R10 / D.prev_Roff10)
    X["slope"] = D.slope_mVs
    X["late"] = np.log(D.Roff240 / D.Roff10)
    X["invT"] = 1000.0 / (D.T_pre + 273.15) - 3.354
    X["lRoff"] = np.log(D.Roff10 / 0.03)
    X["trendS"] = np.log(D.Rstep / D.prevRstep)
    X["trendS1"] = np.log(D.Rstep1 / D.prev_Rstep1)
    X["lT"] = np.log(D.off_dur_s / 200.0)
    X["dlT"] = np.log(D.off_dur_s / D.prev_off_dur_s)
    X["sqA"] = np.log(np.clip(D.sq_A * np.sqrt(10) / D.Rstep10, 0.02, None))
    h10 = D.horizon_s >= 5
    p2R = np.where(h10, D.prev2_R10, D.prev2_R2)
    p2S = np.where(h10, D.prev2_Rstep10, D.prev2_Rstep2)
    obs = np.log(D.prevR / p2R)                          # change of the query-pulse R between the last two points
    trS2 = np.log(D.prevRstep / p2S)                     # change of the interruption R between the same two points
    dah = (D.Ah_q - D.prev_Ah) / (D.prev_Ah - D.prev2_Ah)
    X["obs"] = obs
    X["trS2"] = trS2
    X["ext"] = obs * np.clip(dah, 0.0, 3.0)
    X["amp"] = X.trendS * np.clip(np.where(np.abs(trS2) > 0.05, obs / np.where(np.abs(trS2) > 0.05, trS2, 1.0), 1.0), 0.0, 3.0)
    if "t_cv" in D:                                      # CC-CV recharge of the sweep (cv_feats.py)
        X["ltcv"] = np.log(D.t_cv / 1500.0)
        X["ltcc"] = np.log(D.t_cc / 5500.0)
        X["ltau"] = np.log(D.tau_cv_late / 1000.0)
    lam = 0.02                                           # shrinkage of the per-sweep amplification towards 1
    X["ampT"] = X.trendS * np.clip((obs * trS2 + lam) / (trS2 ** 2 + lam), 0.0, 3.0)
    return X


def bas(D, kind):
    if kind == "off":
        return D.Roff, D.I_end.abs()
    if kind == "prev":
        return D.prevR, D.prev_I_last.abs()
    if kind == "step":
        return D.Rstep, D.I_end.abs()
    if kind == "prevamp":                                 # previous pulse R carried by the shrunk amplified trend
        a = features(D)["ampT"]
        return D.prevR * np.exp(a), D.prev_I_last.abs()
    raise ValueError(kind)


def fit_region(T, X, kind, feats):
    B, Iref = bas(T, kind)
    y = np.log(T.Rq / B)
    xi = np.log(T.Iq / Iref)
    A = np.column_stack([np.ones(len(T))] + [X[f].to_numpy() for f in feats] + [xi.to_numpy()])
    ok = np.isfinite(y.to_numpy()) & np.isfinite(A).all(axis=1) & (T.Iq < 0.98 * T.Imax).to_numpy()
    if ok.sum() < max(12, 4 * A.shape[1]):
        return dict(c0=float(np.nanmedian(y[ok])) if ok.any() else 0.0, c={}, beta=0.0, n=int(ok.sum()), fallback=True)
    coef, *_ = np.linalg.lstsq(A[ok], y.to_numpy()[ok], rcond=None)
    if not -0.9 <= coef[-1] <= 1.0:                     # I * R_hat(I) must rise with I: refit with beta clipped
        coef[-1] = float(np.clip(coef[-1], -0.9, 1.0))
        c2, *_ = np.linalg.lstsq(A[ok][:, :-1], y.to_numpy()[ok] - coef[-1] * A[ok][:, -1], rcond=None)
        coef[:-1] = c2
    res = y.to_numpy()[ok] - A[ok] @ coef
    return dict(c0=float(coef[0]), c=dict(zip(feats, map(float, coef[1:-1]))), beta=float(coef[-1]), n=int(ok.sum()),
                sd=float(res.std()), fallback=False)


def fit_region_direct(T, X, kind, feats):
    """--direct: log(I_q / I_ref) = c0 + a u + sum c_j x_j, u = log(H / (basis I_ref)) (the current the basis alone
    would give, relative to I_ref); fitted on voltage-limited training rows, so the target is the predicted quantity."""
    B, Iref = bas(T, kind)
    v = np.log(T.Iq / Iref)
    u = np.log(T.H / (B * Iref))
    A = np.column_stack([np.ones(len(T)), u.to_numpy()] + [X[f].to_numpy() for f in feats])
    ok = np.isfinite(v.to_numpy()) & np.isfinite(A).all(axis=1) & (T.Iq < 0.98 * T.Imax).to_numpy()
    if ok.sum() < max(12, 4 * A.shape[1]):
        return dict(c0=float(np.nanmedian((v - u)[ok])) if ok.any() else 0.0, a=1.0, c={}, n=int(ok.sum()),
                    fallback=True, direct=True)
    coef, *_ = np.linalg.lstsq(A[ok], v.to_numpy()[ok], rcond=None)
    res = v.to_numpy()[ok] - A[ok] @ coef
    return dict(c0=float(coef[0]), a=float(coef[1]), c=dict(zip(feats, map(float, coef[2:]))), n=int(ok.sum()),
                sd=float(res.std()), fallback=False, direct=True)


def predict_rows_direct(D, X, kind, par):
    B, Iref = bas(D, kind)
    out = []
    for i, r in enumerate(D.itertuples()):
        b, ir = B.iat[i], Iref.iat[i]
        lin = par["c0"] + sum(cj * X[f].iat[i] for f, cj in par["c"].items())
        if not (np.isfinite(b) and b > 0 and np.isfinite(ir) and ir > 0 and np.isfinite(lin)):
            out.append((np.nan, np.nan, np.nan))
            continue
        if r.H <= 0:
            out.append((0.0, np.nan, 0.0))
            continue
        I = ir * np.exp(lin + par["a"] * np.log(r.H / (b * ir)))
        Rh = r.H / I                                    # the implied R at the predicted current
        if I > r.Imax:
            Vend = r.V_pre + (1 if r.chg else -1) * r.Imax * Rh
            out.append((r.Imax * Vend, Rh, r.Imax))
        else:
            out.append((r.Vlim * I, Rh, I))
    return pd.DataFrame(out, columns=["pred_power_W", "R_hat", "I_hat"], index=D.index)


def predict_rows(D, X, kind, par):
    B, Iref = bas(D, kind)
    out = []
    for i, r in enumerate(D.itertuples()):
        b, ir = B.iat[i], Iref.iat[i]
        lin = par["c0"] + sum(cj * X[f].iat[i] for f, cj in par["c"].items())
        beta = par["beta"]
        if not (np.isfinite(b) and b > 0 and np.isfinite(ir) and ir > 0 and np.isfinite(lin)):
            out.append((np.nan, np.nan, np.nan))
            continue
        if r.H <= 0:
            out.append((0.0, np.nan, 0.0))
            continue
        k0 = b * np.exp(lin) / ir ** beta
        Rh = lambda I: k0 * max(I, 1e-4) ** beta
        I = (r.H / k0) ** (1.0 / (1.0 + beta))         # closed form of I * R_hat(I) = H (beta > -1)
        if I > r.Imax:
            Vend = r.V_pre + (1 if r.chg else -1) * r.Imax * Rh(r.Imax)
            out.append((r.Imax * Vend, Rh(r.Imax), r.Imax))
        else:
            out.append((r.Vlim * I, Rh(I), I))
    return pd.DataFrame(out, columns=["pred_power_W", "R_hat", "I_hat"], index=D.index)


def run(D, specs, first_split, direct=False, gap=False):
    fitf = fit_region_direct if direct else fit_region
    predf = predict_rows_direct if direct else predict_rows
    X = features(D)
    D = D.copy()
    D0 = D
    first = D0.prev_R10.isna()
    D0["reg"] = D0.region + np.where(first_split & first, np.where(D0.last_sign == "chg", "|firstcv", "|first"), "")
    parts, coefs = [], {}
    for c in sorted(D0.evaluation_group.unique()):
        D = ls.with_gap(D0, D0.evaluation_group != c) if gap else D0
        tr = D.evaluation_group != c
        for reg in D.reg.unique():
            base_reg = reg.split("|")[0]
            spec = specs.get(reg, specs.get(base_reg, specs.get("*")))
            kind, feats = spec
            m_tr = tr & (D.reg == reg)
            m_all = D.reg == reg                     # fold c: the held-out cell c and, in sample, the training cells
            if kind.endswith("!"):                   # no fit: I = H / basis (small-current response at the same SOC)
                kind = kind[:-1]
                par = dict(c0=0.0, a=1.0, c={}, beta=0.0, n=0, fallback=True, fixed=True)
            else:
                par = fitf(D[m_tr], X[m_tr], kind, feats)
            coefs[f"{c}|{reg}"] = dict(par, kind=kind)
            P = predf(D[m_all], X[m_all], kind, par)
            if kind == "prev":                       # no earlier pulse: fall back to the off basis of the same region
                miss = P.pred_power_W.isna()
                if miss.any():
                    par2 = fitf(D[m_tr], X[m_tr], "off", [])
                    P.loc[miss] = predf(D[m_all][miss.to_numpy()], X[m_all][miss.to_numpy()], "off", par2).to_numpy()
            P.insert(0, "fold", c)
            parts.append(pd.concat([D.loc[m_all, ["row_id", "evaluation_group"]], P], axis=1))
    F = pd.concat(parts)
    return F[F.fold == F.evaluation_group].drop(columns=["fold", "evaluation_group"]), coefs, F


def parse_specs(items):
    specs = {}
    for it in items:
        reg, rest = it.split("@", 1)
        kind, _, fs = rest.partition(":")
        specs[reg] = (kind, [f for f in fs.split(",") if f])
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--spec", nargs="+", required=True)
    ap.add_argument("--subregion-first", action="store_true")
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--gap", action="store_true")
    a = ap.parse_args()
    D = ls.table()
    P, coefs, F = run(D, parse_specs(a.spec), a.subregion_first, a.direct, a.gap)
    out = ROOT / "out/local" / a.name
    out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "pred.csv", index=False)
    F.drop(columns="evaluation_group").to_csv(out / "folds.csv", index=False)
    (out / "coef.json").write_text(json.dumps(coefs, indent=1))
    print(f"wrote {out / 'pred.csv'} ({P.pred_power_W.notna().sum()} rows)")


if __name__ == "__main__":
    main()
