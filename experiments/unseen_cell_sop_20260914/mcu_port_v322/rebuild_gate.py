"""V322 step 1c (gate): rebuild the main predictor Ls (V287, all audit fixes) and its tau = 5 % online adaptation from the
restored inputs, and compare row by row with the r2 paper package (data/ad_trajectory.csv: central, scale, request, saved
with 10 significant digits). The benchmark frame is read from closedloop_v251/src/bench_rows.csv directly (the V255 loader
also needs tables that this predictor never uses). Gate: relative difference <= 1e-9 for central, scale and request, identical
intervention flags, 11/15 groups under the utilisation-intervention criterion and 4/15 under the strict criterion.
usage: OMP_NUM_THREADS=1 python rebuild_gate.py   (torch_env)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parents[1]; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(OUT))
import v322_stubs  # noqa: E402
v322_stubs.install()
import closedloop_v251 as C1  # noqa: E402
import invariance_v276 as I6  # noqa: E402
import compare_v278 as CV  # noqa: E402
import adaptive_v284 as A4  # noqa: E402
import specaudit_v287 as SA  # noqa: E402
CV.TMP = OUT / 'tmp'
GK = C1.GK


def bench():
    B = pd.read_csv(HERE / 'closedloop_v251/src/bench_rows.csv').rename(columns={'truth_power_W': 'truth'})
    assert len(B) == 672
    return B


def main():
    Bm = bench(); src = I6.sources(); J = SA.bench_J(Bm)
    X = SA.build('Ls_v322', src, Bm, R12=True, L1=True, S1=True, S2=True)
    b0, ad, sc, Jc = SA.requests_all(X, J, 0.05)
    reqA, scA, evA = A4.adapt(Jc, {c: sc[c]['scale'] for c in SA.CELLS}, A4.GAMMA, A4.TAU, 'cell')
    T = pd.read_csv(HERE / 'paper_package_lsad_20261013/data/ad_trajectory.csv', float_precision='round_trip').set_index('row_id').reindex(J.row_id)
    rel = lambda a, b: float(np.max(SA.rel(a, b)))
    res = dict(central_max_rel=rel(Jc.central.to_numpy(), T.central.to_numpy()), scale_max_rel=rel(scA, T.scale.to_numpy()),
               request_max_rel=rel(reqA, T.request.to_numpy()), ad_vs_requests_all=float(np.max(np.abs(reqA - ad))),
               flag_mismatch=int(((reqA / J.truth.to_numpy() > 1 + 1e-9) != T.intervened.to_numpy(bool)).sum()),
               scales={c: v['scale'] for c, v in sc.items()})
    sm, G = SA.score(Jc, reqA)
    res.update(closed_loop=int(sm['closed_loop']), strict=int(sm['original']), int_rate=sm['int_rate'], util_q50=sm['util_q50'])
    res['ok'] = bool(res['central_max_rel'] <= 1e-9 and res['scale_max_rel'] <= 1e-9 and res['request_max_rel'] <= 1e-9 and res['flag_mismatch'] == 0
                     and res['closed_loop'] == 11 and res['strict'] == 4 and res['ad_vs_requests_all'] == 0.0)
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / 'rebuild_gate.json', 'w'), indent=1)
    pd.DataFrame({'row_id': J.row_id, 'cell': J.evaluation_group, 'direction': J.direction, 'tkey': J.tkey, 'truth': J.truth,
                  'central': Jc.central.to_numpy(), 'scale': scA, 'request': reqA}).to_csv(OUT / 'reference_rows.csv', index=False, float_format='%.17g')


if __name__ == '__main__':
    main()
