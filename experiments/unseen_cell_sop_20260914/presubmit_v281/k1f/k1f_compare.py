"""V281 K1f verdict (PROTOCOL_PRESUBMIT_V281_K1F.md): gate K1f-G (R0 = the V279 vT1h n_all output on the 26 sample rows)
and the tests T1, T3 (= R0), with V276's comparison (invariance_v276.remote_compare: relative 1e-9 on pred_power_W, a, b,
c_mOhm, n_events, soc0; equal status; no timeout). Reads the outputs copied back to presubmit_v281/k1f/out/.
usage: python presubmit_v281/k1f/k1f_compare.py"""
import json
import sys
from pathlib import Path

import pandas as pd

K = Path(__file__).resolve().parent; HERE = K.parents[1]
sys.path.insert(0, str(HERE))
import invariance_v276 as I6  # noqa: E402

COLS = ['row_id', 'status'] + I6.RCOLS


def main():
    rd = lambda v: pd.read_csv(K / 'out' / v / 'pred.csv')
    R0, T1, T3 = rd('R0'), rd('T1'), rd('T3')
    ref = pd.read_csv(HERE / 'compare_v278/phys_vT1h/n_all/pred.csv')
    ref = ref[ref.row_id.isin(R0.row_id)]
    out = dict(gate=I6.remote_compare(R0[COLS], ref[COLS], ['row_id']))
    out['gate']['rows_expected'] = int(len(ref)); out['gate']['ok'] = bool(out['gate']['ok'] and out['gate']['rows'] == len(ref) == 26)
    for name, A in (('T1', T1), ('T3', T3)):
        c = I6.remote_compare(A[COLS], R0[COLS], ['row_id']); c['ok'] = bool(c['ok'] and c['rows'] == 26); out[name] = c
    out['all_pass'] = bool(out['gate']['ok'] and out['T1']['ok'] and out['T3']['ok'])
    (K / 'k1f_result.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
