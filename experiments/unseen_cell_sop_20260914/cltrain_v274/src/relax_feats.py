"""sop_ident step 3a: model-free features of each RPCWBY row from its allowed window [h0, q) (raw Arbin samples).

    cd ../aging_refit/scripts && python ../../sop_ident/tools/relax_feats.py --workers 8

Per row (query row q; the pre-query rest is the rest run that ends at q - 1):
  last load before the pre-rest (the 3 A / 1/3 A SOC-adjust discharge, the CC-CV end, or an earlier query pulse):
    sign, I_end (current of its last sample), dur_s, Ah, V_end (last sample under load);
    R_off(t) = |V(t) - V_end| / |I_end| at t = 1, 2, 5, 10, 30, 60, 120, 240 s after the interruption (nearest sample),
    and Rstep1/2/10 from a sqrt(t) diffusion superposition fit of R_off(1..30 s) with the load duration (interruption())
  V_pre and T_pre_raw (sample q - 1), rest_s, slope of V over the last 30 s of the rest [mV/s]; rows without an
  event (no >= 60 s pre-rest, e.g. the B 25 C 2 s file) are included with whatever rest precedes q
  previous query pulse of the sweep in the same direction (the previous CP pulse before q, any SOC): its power at its
    last sample, V_pre, R(t) at 1, 2, 10 s (as events.py), t_gap to q
Diagnostics only (the query itself, never a model input): R1_q, R2_q, R10_q of the query pulse (events.csv).
Output: out/feats/relax.csv
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AR = ROOT.parent / "aging_refit"
sys.path.insert(0, str(AR / "tools/rpcwby"))
REST_A = 0.02
T_OFF = (1, 2, 5, 10, 30, 60, 120, 240)


def interruption(t, I, V, rest, a, p, pre=""):
    """Features of the interruption of the load that ends at a - 1 (a = first rest sample, p = last rest sample):
    its current, duration, |V(tau) - V_end| / |I| at T_OFF, and a semi-infinite diffusion superposition fit
    recovery(tau) / |I| = R0 + A (sqrt(T) - sqrt(T + tau) + sqrt(tau)), tau in 1..30 s, T = load duration, which gives
    the step response of a load from rest, Rstep(h) = R0 + A sqrt(h), at h = 1, 2, 10 s."""
    e = a - 1
    b = e
    while b > 0 and not rest[b - 1] and np.sign(I[b - 1]) == np.sign(I[e]):
        b -= 1                                          # b = first sample of the last load run (same sign)
    T = t[e] - t[b]
    o = {f"{pre}I_off": I[e], f"{pre}off_dur_s": T}
    taus, rec = [], []
    for tau in T_OFF:
        k = a + int(np.argmin(np.abs(t[a:p + 1] - t[e] - tau)))
        ok = abs(t[k] - t[e] - tau) <= max(0.35 * tau, 0.5)    # 0.5 s, not 1.1 s: a 0.04 s rest is no 1 s point
        o[f"{pre}Roff{tau}"] = abs(V[k] - V[e]) / abs(I[e]) if ok else np.nan
        if ok and tau <= 30:
            taus.append(t[k] - t[e])
            rec.append(abs(V[k] - V[e]) / abs(I[e]))
    if len(taus) >= 3 and T > 0:
        x = np.sqrt(T) - np.sqrt(T + np.array(taus)) + np.sqrt(np.array(taus))
        A_, R0 = np.polyfit(x, rec, 1)
        o.update({f"{pre}sq_R0": R0, f"{pre}sq_A": A_, f"{pre}Rstep1": R0 + A_, f"{pre}Rstep2": R0 + A_ * np.sqrt(2),
                  f"{pre}Rstep10": R0 + A_ * np.sqrt(10)})
    return o, b


def one(args):
    sid, member, h0, end, rows = args
    import raw
    W = raw.window(member, h0, end + 1, ["t", "st", "cyc", "step", "I", "V", "T_cell"])
    W.index = np.arange(h0, end + 1)
    t, I, V = W.t.to_numpy(), W.I.to_numpy(), W.V.to_numpy()
    key = W.cyc.to_numpy() * 100000 + W.step.to_numpy()
    rest = np.abs(I) < REST_A
    ah = np.zeros(len(I))                               # discharged Ah since h0 (+ = discharge; raw I is + for charge)
    ah[1:] = np.cumsum(-0.5 * (I[1:] + I[:-1]) * np.diff(t)) / 3600.0

    def pulse_before(j, sgn, pre):
        while j > 0:
            if not rest[j] and np.sign(I[j]) == sgn:
                k1 = j
                k0 = j
                while k0 > 0 and not rest[k0 - 1] and key[k0 - 1] == key[j]:
                    k0 -= 1
                if t[k1] - t[k0] <= 12.5 and k0 > 0 and rest[k0 - 1]:
                    t0 = t[k0] - W.st.iat[k0]
                    Vp = V[k0 - 1]
                    dt = t[k0:k1 + 1] - t0
                    R = (V[k0:k1 + 1] - Vp) / I[k0:k1 + 1]
                    o = {f"{pre}P": abs(V[k1] * I[k1]), f"{pre}Vpre": Vp, f"{pre}I_last": I[k1], f"{pre}V_last": V[k1],
                         f"{pre}dt_last": dt[-1], f"{pre}Ah": ah[k0 - 1]}
                    for tau in (1, 2, 10):
                        kk = int(np.argmin(np.abs(dt - tau))) if tau < 10 else int(np.flatnonzero(dt <= 10.1)[-1])
                        o[f"{pre}R{tau}"] = R[kk]
                    a2 = k0 - 1
                    while a2 > 0 and rest[a2 - 1]:
                        a2 -= 1
                    if a2 - 1 >= 0 and not rest[a2 - 1]:
                        f2, _ = interruption(t, I, V, rest, a2, k0 - 1, pre=pre)
                        o.update(f2)
                        o[f"{pre}rest_s"] = t[k0 - 1] - t[a2]
                    return o, k0
                j = k0 - 1
                continue
            j -= 1
        return {}, None

    out = []
    for r in rows:
        q = int(r["q_row"]) - h0
        o = dict(row_id=r["row_id"], sweep_id=sid)
        p = q - 1                                       # last sample before the query
        if p < 1 or not rest[p]:
            o["note"] = "no pre-rest"
            out.append(o)
            continue
        a = p
        while a > 0 and rest[a - 1]:
            a -= 1                                      # a = first rest sample, a - 1 = last load sample
        e = a - 1
        f, b = interruption(t, I, V, rest, a, p)
        o.update(f)
        o.update(V_pre=V[p], T_pre_raw=float(W.T_cell.iat[p]), rest_s=t[p] - t[a], last_sign="chg" if I[e] > 0 else "dis", I_end=I[e],
                 last_dur_s=t[e] - t[b], last_Ah=float(np.sum(np.abs(0.5 * (I[b + 1:e + 1] + I[b:e]) * np.diff(t[b:e + 1])))) / 3600,
                 V_end=V[e], V_rest0=V[a], dt_rest0=t[a] - t[e])
        m = t[a:p + 1] >= t[p] - 30.0
        if m.sum() >= 5:
            o["slope_mVs"] = 1e3 * np.polyfit(t[a:p + 1][m] - t[p], V[a:p + 1][m], 1)[0]
        # previous CP query pulses of the same direction (a load run after >= 60 s rest, <= 12.5 s long): the last one
        # (prev_*) and the one before it (prev2_*), each with the interruption before its pre-rest
        sgn = 1 if r["direction"] == "charge" else -1
        o["Ah_q"] = ah[p]                              # at q - 1: ah[q] would include half of the query's first sample
        f1, k0 = pulse_before(b - 1, sgn, "prev_")
        o.update(f1)
        if k0 is not None:
            o["prev_gap_s"] = t[p] - t[k0]
            b2 = k0 - 1
            while b2 > 0 and rest[b2]:
                b2 -= 1
            while b2 > 0 and not rest[b2 - 1] and np.sign(I[b2 - 1]) == np.sign(I[b2]):
                b2 -= 1                                 # start of the load before the previous pulse's pre-rest
            f2, _ = pulse_before(b2 - 1, sgn, "prev2_")
            o.update(f2)
        out.append(o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    R = pd.read_csv(ROOT.parent / "pybamm/out/rpcwby_bench/rows.csv")
    E = pd.read_csv(AR / "out/v2/events.csv")
    SW = pd.read_csv(AR / "out/v2/sweeps.csv")
    R = R.merge(E[["member", "event_id", "sweep_id", "R1", "R2", "R10"]], left_on=["member", "q_row"],
                right_on=["member", "event_id"], how="left")
    R = R.rename(columns={"R1": "R1_q", "R2": "R2_q", "R10": "R10_q"})
    # rows whose query did not follow >= 60 s of rest have no event (the three B 25 C 2 s rows): their sweep by file,
    # h0_row and direction
    key = ["member", "h0_row", "direction"]
    sid = R[key].merge(SW[key + ["sweep_id"]], on=key, how="left").sweep_id.to_numpy()
    R["sweep_id"] = R.sweep_id.fillna(pd.Series(sid, index=R.index))
    jobs = []
    for sid, g in R[R.sweep_id.notna()].groupby("sweep_id"):
        sw = SW[SW.sweep_id == sid].iloc[0]
        jobs.append((int(sid), sw.member, int(sw.h0_row), int(g.q_row.max()),
                     [dict(row_id=x.row_id, q_row=x.q_row, direction=x.direction) for x in g.itertuples()]))
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        for o in ex.map(one, jobs):
            res += o
    F = pd.DataFrame(res).merge(R[["row_id", "R1_q", "R2_q", "R10_q"]], on="row_id", how="left")
    (ROOT / "out/feats").mkdir(parents=True, exist_ok=True)
    F.to_csv(ROOT / "out/feats/relax.csv", index=False)
    print(f"wrote {len(F)} rows; missing pre-rest {int(F.V_pre.isna().sum())}")


if __name__ == "__main__":
    main()
