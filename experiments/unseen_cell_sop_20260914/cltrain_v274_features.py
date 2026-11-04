"""V274 step 1 (PROTOCOL_CLTRAIN_V274.md): local-predictor features of the current-limited SOP rows of the RPCWBY cells.

- rows: the current-limited rows of all6_measured_v178 (horizon 2 or 10 s), linked 1:1 to the final pulses of the V157 row
  keys by (group, direction, temperature, horizon, SOC point, truth to 1e-4 W) as pybamm's bench.py links the 672
  benchmark rows; query row q = first row of the final pulse, h0 = the V211 history window start (the CC-CV recharge
  that starts the sweep), sweep = aging_refit sweeps.csv by (member, h0_row, direction);
- features: sop_ident relax_feats.one (copy, hash = remote original) on the window [h0, q), through a local stand-in of
  raw.py (cltrain_v274/src/shim) that reads the local RPCWBY zips with raw.py's parsing;
- query-pulse response (training target only, never an input): events.py's per-onset formulas (copy of the remote file in
  cltrain_v274/src/events_v2_reference.py, re-used verbatim in onset() below): R1, R2, R10 = (V - V_pre) / I at the
  samples nearest 1 s and 2 s and the last sample <= 10.1 s of the step after >= 60 s of rest, with the current at those
  samples, I_last, V_min, V_max, T_pre.
Gates: the same code on the 672 benchmark rows must reproduce sop_ident's relax.csv and aging_refit's events.csv.
usage: python cltrain_v274_features.py"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'cltrain_v274'; SRC = OUT / 'src'; V2S = HERE / 'leakfree_v272/src'
REST_A, PRE_REST_MIN_S, WIN_S = 0.02, 60.0, 12.0          # events.py constants


def _init():
    for p in (str(SRC / 'shim'), str(SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)


def onset(t, I, V, st, key, rest, T, i):
    """events.py one(): the event of the load onset at window index i (None without >= 60 s of rest before it)"""
    if i < 1 or rest[i] or not rest[i - 1]:
        return None
    p = i - 1; run0 = p
    while run0 > 0 and rest[run0 - 1]:
        run0 -= 1
    if t[p] - t[run0] < PRE_REST_MIN_S:
        return None
    t0 = t[i] - st[i]; rows = []; j = i
    while j < len(I) and not rest[j] and key[j] == key[i] and t[j] - t0 <= WIN_S:
        rows.append(j); j += 1
    if not rows:
        return None
    Iw, Vw, dt = I[rows], V[rows], t[rows] - t0
    R = (Vw - V[p]) / Iw

    def at(tau):
        k = int(np.argmin(np.abs(dt - tau)))
        return (float(R[k]), float(dt[k]), float(Iw[k])) if abs(dt[k] - tau) <= 0.35 * tau else (np.nan, np.nan, np.nan)
    k10 = np.flatnonzero(dt <= 10.1); r1, r2 = at(1.0), at(2.0)
    return dict(V_pre_ev=float(V[p]), T_pre=float(T[p]), pre_rest_s=float(t[p] - t[run0]), I_first=float(Iw[0]), I_last=float(Iw[-1]),
                I_absmax=float(np.max(np.abs(Iw))), n=len(rows), dt_last=float(dt[-1]), R1=r1[0], t_R1=r1[1], R2=r2[0], t_R2=r2[1],
                I_R2=r2[2], R10=float(R[k10[-1]]) if len(k10) else np.nan, t_R10=float(dt[k10[-1]]) if len(k10) else np.nan,
                I_R10=float(Iw[k10[-1]]) if len(k10) else np.nan, V_min=float(Vw.min()), V_max=float(Vw.max()))


def member_job(args):
    """all sweeps of one member: relax_feats.one on [h0, q) and the query onsets on [h0, last_row]"""
    _init()
    import raw  # the local stand-in
    import relax_feats as rf
    member, sweeps = args
    feats, evs = [], []
    for sid, h0, rows in sweeps:
        end = max(r['q_row'] for r in rows)
        feats += rf.one((sid, member, h0, end, [dict(row_id=r['row_id'], q_row=r['q_row'], direction=r['direction']) for r in rows]))
        last = max(r['last_row'] for r in rows)
        W = raw.window(member, h0, last + 1, ['t', 'st', 'cyc', 'step', 'I', 'V', 'T_cell'])
        t, I, V, st, T = W.t.to_numpy(), W.I.to_numpy(), W.V.to_numpy(), W.st.to_numpy(), W.T_cell.to_numpy()
        key = W.cyc.to_numpy() * 100000 + W.step.to_numpy(); rest = np.abs(I) < REST_A
        for r in rows:
            o = onset(t, I, V, st, key, rest, T, int(r['q_row']) - h0)
            evs.append(dict(row_id=r['row_id'], **(o or {})))
    return feats, evs


def cl_rows():
    """current-limited rows (2 / 10 s) with their raw position, history start and sweep"""
    K = pd.read_csv(HERE.parent / 'same_search_exclusion_v157_20260921/keys/row_keys.csv')
    V = pd.read_csv(HERE / 'all6_measured_v178/predictions.csv'); Fk = K[K.is_final == True].copy()
    for d in (V, Fk):
        d['tr'] = d.truth_power_W.round(4)
    k = ['evaluation_group', 'direction', 'temp_C', 'horizon_s', 'soc_point', 'tr']
    L = V.merge(Fk[k + ['row_id', 'member', 'first_row', 'last_row']], on=k, how='left')
    assert len(L) == len(V) and L.row_id.notna().all() and L.row_id.is_unique and L.groupby(['sheet', 'row']).size().max() == 1
    W = pd.read_csv(HERE / 'mamba_v211/precheck/history_windows.csv')[['row_id', 'h0_row', 'query_first_row']]
    C = L[(L.regime == 'current_limited') & L.horizon_s.isin([2.0, 10.0])].merge(W, on='row_id', how='left', validate='one_to_one')
    assert C.h0_row.notna().all() and (C.query_first_row == C.first_row).all()
    SW = pd.read_csv(SRC / 'sweeps.csv')
    C = C.merge(SW[['member', 'h0_row', 'direction', 'sweep_id']], on=['member', 'h0_row', 'direction'], how='left', validate='many_to_one')
    # sweeps without any voltage-limited row are not in sweeps.csv (3 A17 charge rows): a new id per (member, h0, direction)
    new = C.loc[C.sweep_id.isna(), ['member', 'h0_row', 'direction']].drop_duplicates().reset_index(drop=True)
    new['sweep_new'] = 10000 + np.arange(len(new))
    C = C.merge(new, on=['member', 'h0_row', 'direction'], how='left')
    C['sweep_id'] = C.sweep_id.fillna(C.sweep_new); C = C.drop(columns='sweep_new')
    assert C.sweep_id.notna().all()
    C = C.rename(columns={'first_row': 'q_row'})
    return C[['row_id', 'evaluation_group', 'sheet', 'row', 'member', 'direction', 'temp_C', 'horizon_s', 'soc_point', 'truth_power_W',
              'truth_end_current_A', 'q_row', 'last_row', 'h0_row', 'sweep_id']].astype({'q_row': int, 'last_row': int, 'h0_row': int, 'sweep_id': int})


def compare(a, b, key, cols):
    M = a.merge(b, on=key, how='inner', suffixes=('', '_ref')); out = {}
    for c in cols:
        x, y = M[c].to_numpy(float), M[c + '_ref'].to_numpy(float)
        with np.errstate(all='ignore'):
            r = np.abs(x - y) / np.maximum(np.abs(y), 1e-12)
        r[np.isnan(x) & np.isnan(y)] = 0.0
        out[c] = dict(max_rel=float(np.nanmax(r)) if np.isfinite(r).any() else 0.0, nan_mismatch=int((np.isnan(x) != np.isnan(y)).sum()))
    return len(M), out


def main():
    OUT.mkdir(exist_ok=True)
    Bn = pd.read_csv(V2S / 'pybamm/out/rpcwby_bench/rows.csv')
    SW = pd.read_csv(SRC / 'sweeps.csv')
    Bn = Bn.merge(SW[['member', 'h0_row', 'direction', 'sweep_id']], on=['member', 'h0_row', 'direction'], how='left', validate='many_to_one')
    assert Bn.sweep_id.notna().all()
    C = cl_rows()
    print('current-limited rows (2 / 10 s):', len(C), C.groupby(['evaluation_group', 'direction']).size().to_dict(), flush=True)
    allr = pd.concat([Bn[['row_id', 'member', 'direction', 'q_row', 'last_row', 'h0_row', 'sweep_id']].assign(kind='bench'),
                      C[['row_id', 'member', 'direction', 'q_row', 'last_row', 'h0_row', 'sweep_id']].assign(kind='cl')], ignore_index=True)
    assert allr.row_id.is_unique
    jobs = []
    for m, g in allr.groupby('member'):
        jobs.append((m, [(int(s), int(h), g2.to_dict('records')) for (s, h), g2 in g.groupby(['sweep_id', 'h0_row'])]))
    feats, evs = [], []
    with ProcessPoolExecutor(4, initializer=_init) as ex:
        for f, e in ex.map(member_job, jobs):
            feats += f; evs += e
    Fe = pd.DataFrame(feats); Ev = pd.DataFrame(evs)
    # gates on the 672 benchmark rows
    R0 = pd.read_csv(V2S / 'sop_ident/out/feats/relax.csv'); E0 = pd.read_csv(V2S / 'aging_refit/out/v2/events.csv')
    fcols = [c for c in R0.columns if c not in ('row_id', 'sweep_id', 'note', 'last_sign', 'R1_q', 'R2_q', 'R10_q') and pd.api.types.is_numeric_dtype(R0[c])]
    n1, g1 = compare(Fe[Fe.row_id.isin(Bn.row_id)], R0, 'row_id', fcols)
    sign_ok = bool((Fe.merge(R0[['row_id', 'last_sign']], on='row_id', suffixes=('', '_ref')).pipe(lambda d: (d.last_sign.fillna('') == d.last_sign_ref.fillna('')).all())))
    Eb = Bn[['row_id', 'member', 'q_row']].merge(E0, left_on=['member', 'q_row'], right_on=['member', 'event_id'], how='inner')
    ecols = ['V_pre', 'T_pre', 'I_first', 'I_last', 'I_absmax', 'n', 'dt_last', 'R1', 't_R1', 'R2', 't_R2', 'R10', 't_R10', 'V_min', 'V_max']
    n2, g2 = compare(Ev.rename(columns={'V_pre_ev': 'V_pre'})[['row_id'] + ecols], Eb[['row_id'] + ecols], 'row_id', ecols)
    gates = dict(relax_rows=n1, relax_bench_rows=int(len(R0)), relax=g1, last_sign_equal=sign_ok, events_rows=n2, events_bench_rows_with_event=int(len(Eb)), events=g2)
    # tolerance 1e-9 (relative), the cross-machine reproduction tolerance of V272/V273: the first run had 1e-12 and only
    # slope_mVs (a least-squares slope near zero) exceeded it, 3.5e-11 relative / 1.8e-12 mV/s absolute
    ok = (n1 == len(R0) and all(v['max_rel'] <= 1e-9 and v['nan_mismatch'] == 0 for v in g1.values()) and sign_ok
          and n2 == len(Eb) and all(v['max_rel'] <= 1e-9 and v['nan_mismatch'] == 0 for v in g2.values()))
    gates['ok'] = bool(ok)
    (OUT / 'features_gates.json').write_text(json.dumps(gates, indent=1))
    print('gate relax.csv:', n1, 'rows,', 'max rel', max(v['max_rel'] for v in g1.values()), 'nan mismatch', sum(v['nan_mismatch'] for v in g1.values()), '| last_sign', sign_ok, flush=True)
    print('gate events.csv:', n2, 'rows,', 'max rel', max(v['max_rel'] for v in g2.values()), 'nan mismatch', sum(v['nan_mismatch'] for v in g2.values()), flush=True)
    if not ok:
        raise SystemExit('feature gates failed; current-limited features not written (protocol)')
    X = C.merge(Fe.drop(columns=['sweep_id'], errors='ignore'), on='row_id', how='left').merge(Ev, on='row_id', how='left')
    X.to_csv(OUT / 'features_cl.csv', index=False)
    print('features_cl.csv:', len(X), 'rows; with pre-rest:', int(X.V_pre.notna().sum()), '; with an onset event:', int(X.R10.notna().sum() + X.R2.notna().sum() - (X.R10.notna() & X.R2.notna()).sum()), flush=True)


if __name__ == '__main__':
    main()
