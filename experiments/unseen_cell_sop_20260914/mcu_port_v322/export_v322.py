"""V322 step 2a: everything the C port needs from the frozen Python pipeline (Ls, all audit fixes; V284 adaptation).
  coef_v322.json   per evaluated cell c (= fold c): rest-voltage boundaries, the five local predictors' coefficients for
                   every region / first-query sub-region (the 'off' fallback fit for 'prev'-kind regions, S2), G1's gap
                   constants, starting scale and adaptation constants
  ref_v322.csv     per evaluated row (its own fold): region, sub-region, the five predictions, centre, scale, request
  inter_v322.csv   per evaluated row: the model-table values and features the predictors read (debugging the port)
  stream_v322.bin  the raw samples of the 134 sweep windows [h0, q_max) as exact integers (t, step time in ms; I in
                   1e-5 A; V in 1e-4 V) and the cell temperature as float64 (the B files carry up to 7 decimals), with the
                   query positions; queries_v322.csv the query table
usage: OMP_NUM_THREADS=1 python export_v322.py   (torch_env)"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(OUT))
import v322_stubs  # noqa: E402
v322_stubs.install()
import closedloop_v251 as C1  # noqa: E402
import invariance_v276 as I6  # noqa: E402
import compare_v278 as CV  # noqa: E402
import adaptive_v284 as A4  # noqa: E402
import specaudit_v287 as SA  # noqa: E402
import leakfree_v272 as V2  # noqa: E402
CV.TMP = OUT / 'tmp'
ls, ls2 = V2.ls, V2.ls2
CELLS = SA.CELLS; NAMES = V2.NAMES
BASE_REGS = ('dis<=0.1', 'dis0.15-0.3', 'dis>=0.4', 'chg<=0.7', 'chg>=0.8')
FIRST = ('', '|first', '|firstcv')
FEATS_ALL = ('trend10', 'trend1', 'invT', 'trendS', 'trendS1', 'dlT', 'sqA', 'pdiff', 'slope', 'lT', 'amp', 'ext')
sys.path.insert(0, str(HERE / 'invariance_v276/src/shim_mask'))
import raw  # noqa: E402


def bench():
    B = pd.read_csv(HERE / 'closedloop_v251/src/bench_rows.csv').rename(columns={'truth_power_W': 'truth'})
    assert len(B) == 672
    return B


def regs(D):
    """run_audit's sub-regions (S1: firstcv only after the CC-CV charge, V_end >= 4.19 V)"""
    first = D.prev_R10.isna(); cv = (D.last_sign == 'chg') & (D.V_end >= 4.19)
    return D.region + np.where(first, np.where(cv, '|firstcv', '|first'), '')


def par_json(par, kind, feats):
    c = [float(par['c'].get(f, 0.0)) for f in feats] if par.get('c') else []
    return dict(kind=kind, feats=list(par['c'].keys()) if par.get('c') else [], c=c, c0=float(par['c0']), beta=float(par.get('beta', 0.0)),
                a=float(par.get('a', 1.0)), fallback=bool(par.get('fallback', False)), fixed=bool(par.get('fixed', False)), n=int(par.get('n', 0)))


def export_coef(D):
    out = {}
    with SA.flags(R12=True, L1=True, S1=True, S2=True):
        TH = V2.boundaries(D, 'min_error')
        for c in CELLS:
            Dc = D.copy(); Dc['region'] = V2.assign(Dc, TH[('fold', c)]); tr = Dc.evaluation_group != c
            e = dict(thresholds={k: float(TH[('fold', c)][k]['t']) for k in ('dis1', 'dis2', 'chg')})
            par = ls.fit(Dc[tr], 'prev', True)                                   # fit_audit (L1)
            e['prev'] = {r: [float(k), float(b)] for r, (k, b) in par.items()}
            for name in NAMES[1:]:
                spec, direct, gap = V2.SPECS[name]
                specs = ls2.parse_specs(spec)
                _, coefs, _ = ls2.run(Dc, specs, True, direct, gap)             # run_audit (S1, S2)
                Dg = ls.with_gap(Dc, tr) if gap else Dc.copy()
                Xf = ls2.features(Dg); Dg['reg'] = regs(Dg)
                fitf = ls2.fit_region_direct if direct else ls2.fit_region
                tab = {}
                for reg in sorted(Dg.reg.unique()):
                    p = coefs[f'{c}|{reg}']; kind = p['kind']
                    base = reg.split('|')[0]; sp = specs.get(reg, specs.get(base, specs.get('*')))
                    j = par_json(p, kind, sp[1])
                    if kind == 'prev':                                          # S2 fallback for rows without an earlier pulse
                        m_tr = tr & (Dg.reg == reg)
                        j['par2'] = par_json(fitf(Dg[m_tr], Xf[m_tr], 'off', []), 'off', [])
                    tab[reg] = j
                g = {}
                if gap:
                    for chg in (True, False):
                        m = tr & (Dc.chg == chg) & (Dc.Iq < 0.98 * Dc.Imax)
                        g['chg' if chg else 'dis'] = float(np.nanmedian(Dc.loc[m, 'gap'])) if m.any() else 0.0
                e[name] = dict(direct=bool(direct), gap=bool(gap), g=g, regs=tab)
            out[c] = e
    return out, TH


def main():
    Bm = bench(); src = I6.sources(); J = SA.bench_J(Bm)
    X = SA.build('Ls_v322_export', src, Bm, R12=True, L1=True, S1=True, S2=True)
    D = X['D']
    coef, TH = export_coef(D)
    b0, ad, sc, Jc = SA.requests_all(X, J, 0.05)
    coef['_constants'] = dict(scales={c: float(v['scale']) for c, v in sc.items()}, gamma=A4.GAMMA, tau_star=A4.TAU,
                              log_lo=float(A4.LO), log_hi=float(A4.HI), rest_A=0.02, pre_rest_min_s=60.0, t_off=[1, 2, 5, 10, 30, 60, 120, 240])
    (OUT / 'coef_v322.json').write_text(json.dumps(coef, indent=1))
    # reference predictions: each evaluated row from its own fold
    reqA, scA, evA = A4.adapt(Jc, {c: sc[c]['scale'] for c in CELLS}, A4.GAMMA, A4.TAU, 'cell')
    ref = J[['row_id', 'evaluation_group', 'direction', 'horizon_s', 'temp_C', 'member', 'q_row', 'tkey', 'truth']].copy()
    for n in NAMES:
        Fv = X['Fv'][n]; m = {(f, r): p for f, r, p in zip(Fv.fold, Fv.row_id, Fv.pred_power_W)}
        ref[n] = [m.get((c, r), np.nan) for c, r in zip(ref.evaluation_group, ref.row_id)]
    ref['central'] = Jc.central.to_numpy(); ref['scale'] = scA; ref['request'] = reqA; ref['intervened'] = evA
    inter = []
    for c in CELLS:
        Dc = D.copy(); Dc['region'] = V2.assign(Dc, TH[('fold', c)]); Dc['reg'] = regs(Dc)
        Xf = ls2.features(Dc)
        cols = ['row_id', 'region', 'reg', 'V_pre', 'H', 'Vlim', 'Imax', 'T_pre', 'slope_mVs', 'Roff', 'Rstep', 'prevR', 'prevRoff', 'prevRstep',
                'I_end', 'prev_I_last', 'Ah_q', 'prev_Ah', 'prev2_Ah', 'last_sign', 'V_end', 'off_dur_s']
        Z = pd.concat([Dc[cols], Xf[list(FEATS_ALL) + ['obs', 'trS2']]], axis=1)
        inter.append(Z[(Dc.evaluation_group == c).to_numpy()])
    I = pd.concat(inter); ref = ref.merge(I[['row_id', 'region', 'reg']], on='row_id', how='left')
    ref.to_csv(OUT / 'ref_v322.csv', index=False, float_format='%.17g'); I.to_csv(OUT / 'inter_v322.csv', index=False, float_format='%.17g')
    # sample stream
    R = pd.read_csv(HERE / 'leakfree_v272/src/pybamm/out/rpcwby_bench/rows.csv')
    R = R.merge(J[['row_id', 'tkey', 'truth']], on='row_id', how='left'); assert R.tkey.notna().all()
    cell_ix = {c: i for i, c in enumerate(CELLS)}
    wins = R.groupby(['member', 'h0_row'], sort=False)
    order = sorted(wins.groups.keys(), key=lambda k: (cell_ix[R.loc[wins.groups[k][0], 'evaluation_group']], float(R.loc[wins.groups[k], 'tkey'].min())))
    qrows, nbytes = [], 0
    with open(OUT / 'stream_v322.bin', 'wb') as f:
        f.write(b'V322'); f.write(struct.pack('<I', len(order)))
        for wid, (member, h0) in enumerate(order):
            g = R.loc[wins.groups[(member, h0)]].sort_values('q_row'); qmax = int(g.q_row.max())
            W = raw.window(member, int(h0), qmax, ['t', 'st', 'cyc', 'step', 'I', 'V', 'T_cell'])
            enc = {}
            for col, k, dt in (('t', 1e3, np.int64), ('st', 1e3, np.int64), ('I', 1e5, np.int64), ('V', 1e4, np.int64)):
                v = W[col].to_numpy(float); nan = ~np.isfinite(v); iv = np.where(nan, 0, np.rint(v * k)).astype(dt)
                back = iv.astype(float) / k
                assert np.array_equal(back[~nan], v[~nan]), (member, col)                # exact round trip of every value
                assert not nan.any(), (member, col)
                enc[col] = iv
            key = (W.cyc.to_numpy(np.int64) * 100000 + W.step.to_numpy(np.int64)); assert key.max() < 2 ** 31
            cell = cell_ix[g.evaluation_group.iloc[0]]
            f.write(struct.pack('<IIII', wid, cell, len(W), len(g)))
            for x in g.itertuples():
                qid = int(np.flatnonzero(ref.row_id.to_numpy() == x.row_id)[0])
                f.write(struct.pack('<IIII', int(x.q_row) - int(h0), qid, 1 if x.direction == 'charge' else 0, int(round(x.horizon_s * 1000))))
                qrows.append(dict(query_id=qid, row_id=x.row_id, cell=cell, window_id=wid, offset=int(x.q_row) - int(h0), direction=x.direction,
                                  horizon_s=x.horizon_s, tkey=x.tkey, truth=x.truth))
            rec = np.zeros(len(W), dtype=np.dtype([('t', '<i8'), ('st', '<u4'), ('key', '<i4'), ('I', '<i4'), ('V', '<i4'), ('T', '<f8')]))
            rec['t'] = enc['t']; rec['st'] = enc['st'].astype(np.uint32); rec['key'] = key.astype(np.int32)
            rec['I'] = enc['I'].astype(np.int32); rec['V'] = enc['V'].astype(np.int32); rec['T'] = W.T_cell.to_numpy(float)
            f.write(rec.tobytes()); nbytes += rec.nbytes
    Q = pd.DataFrame(qrows).sort_values('query_id'); assert len(Q) == 672 and Q.query_id.is_unique
    Q.to_csv(OUT / 'queries_v322.csv', index=False, float_format='%.17g')
    print('coef folds', list(coef), '| ref rows', len(ref), '| windows', len(order), '| sample bytes', nbytes)
    print('centre==0 rows', int((ref.central == 0).sum()), '| scales', coef['_constants']['scales'])


if __name__ == '__main__':
    main()
