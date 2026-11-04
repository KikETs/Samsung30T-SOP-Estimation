"""V106 feasibility audit (no training, no calibration): can the zero-exceedance and 80 % median goal be met on the measured truth as defined?

1. Perfect-predictor bound. A predictor that knows each pulse's true maximum constant-power SOP at the voltage limit is still scored
   against the measured final pulse, which the Arbin search accepts within +-0.04 V of the limit. First-order resistive estimate of the
   true maximum relative to the measured truth: discharge r = (V_pre - V_lim) V_lim / ((V_pre - V_end) V_end),
   charge r = (V_lim - V_pre) V_lim / ((V_end - V_pre) V_end). Forcing zero exceedance with the best single factor per group (1/max r)
   leaves an achievable median usable of 100 median(r)/max(r).
2. Model bound. The same ideal per-group factor applied to the raw predictions of V98, V101 and V104. It uses the evaluated group's
   truth, so it is an upper bound and never an admissible method.
Small-headroom rows: 0.04 V is more than 20 % of the voltage headroom. Groups follow the V98/V101 gate (voltage-limited rows)."""
import json
import numpy as np
import pandas as pd
from run_envelope import HERE, dump, digest

OUT = HERE / 'feasibility_audit_v106'
KEY = ['evaluation_group', 'direction', 'row', 'pulse_id']
GROUP = ['evaluation_group', 'sheet', 'temp_C', 'horizon_s', 'direction']
MODELS = {'V98': 'measured_cells_scoring_v98/predictions.csv', 'V101': 'lowsoc_floor_scoring_v101/predictions.csv', 'V104': 'pinn_scoring_v104/predictions.csv'}


def bound(ratio):
    ratio = np.asarray(ratio, float)
    return float(100 * np.median(ratio) / ratio.max()) if len(ratio) and ratio.max() > 0 else np.nan


def main():
    OUT.mkdir(exist_ok=False)
    T = pd.read_csv(HERE / 'window_identification_v103' / 'targets.csv')
    V = T[T.regime == 'voltage_limited'].copy()
    lim = np.where(V.direction == 'discharge', 2.55, 4.15)
    vpre, vend = V.v_pre.to_numpy(float), V.truth_end_voltage_V.to_numpy(float)
    dis = (V.direction == 'discharge').to_numpy()
    num = np.where(dis, (vpre - lim) * lim, (lim - vpre) * lim)
    den = np.where(dis, (vpre - vend) * vend, (vend - vpre) * vend)
    assert (den > 0).all() and (num > 0).all()
    V['perfect_ratio'] = num / den
    V['headroom_V'] = np.where(dis, vpre - lim, lim - vpre)
    V['end_offset_V'] = np.where(dis, vend - lim, lim - vend)
    V['small_headroom'] = .04 / V.headroom_V > .2
    V['scope'] = np.where(V.evaluation_group != 'B', 'A_cells', np.where(V.horizon_s == 30., 'B_30s_diagnostic', 'B_goal'))
    for name, path in MODELS.items():
        P = pd.read_csv(HERE / path)[KEY + ['raw_power_W']].rename(columns={'raw_power_W': f'{name}_raw_power_W'})
        V = V.merge(P, on=KEY, how='left', validate='one_to_one')
        assert V[f'{name}_raw_power_W'].notna().all()
        V[f'{name}_ratio'] = V[f'{name}_raw_power_W'] / V.truth_power_W
    rows = []
    for key, g in V.groupby(GROUP):
        keep = g[~g.small_headroom]
        rec = dict(zip(GROUP, key), scope=g.scope.iloc[0], n=len(g), small_headroom_rows=int(g.small_headroom.sum()), headroom_median_V=float(g.headroom_V.median()),
                   end_offset_p05_V=float(g.end_offset_V.quantile(.05)), end_offset_p95_V=float(g.end_offset_V.quantile(.95)),
                   perfect_ratio_median=float(g.perfect_ratio.median()), perfect_ratio_max=float(g.perfect_ratio.max()), perfect_bound_pct=bound(g.perfect_ratio),
                   perfect_bound_without_small_pct=bound(keep.perfect_ratio))
        for name in MODELS:
            rec[f'{name}_bound_pct'] = bound(g[f'{name}_ratio'])
            rec[f'{name}_bound_without_small_pct'] = bound(keep[f'{name}_ratio'])
        rows.append(rec)
    G = pd.DataFrame(rows)
    V.to_csv(OUT / 'rows.csv', index=False)
    G.to_csv(OUT / 'groups.csv', index=False)
    summary = {}
    for scope, s in G.groupby('scope'):
        summary[scope] = dict(groups=len(s), rows=int(s.n.sum()), small_headroom_rows=int(s.small_headroom_rows.sum()),
                              perfect_feasible_groups=int((s.perfect_bound_pct >= 80).sum()), perfect_feasible_without_small=int((s.perfect_bound_without_small_pct >= 80).sum()),
                              **{f'{m}_feasible_groups': int((s[f'{m}_bound_pct'] >= 80).sum()) for m in MODELS},
                              **{f'{m}_feasible_without_small': int((s[f'{m}_bound_without_small_pct'] >= 80).sum()) for m in MODELS})
    summary.update(first_order_resistive_approximation=True, uses_evaluated_truth=True, admissible_method=False, criteria_changed=False, goal_complete=False)
    dump(OUT / 'summary.json', summary)
    dump(OUT / 'audit.json', dict(source_hashes={s: digest(HERE / s) for s in ['feasibility_audit_v106.py', 'window_identification_v103/targets.csv', *MODELS.values()]},
                                  output_hashes={f.name: digest(f) for f in sorted(OUT.iterdir()) if f.is_file() and f.name != 'audit.json'}, goal_complete=False))
    pd.set_option('display.width', 260)
    cols = GROUP + ['n', 'small_headroom_rows', 'headroom_median_V', 'end_offset_p05_V', 'end_offset_p95_V', 'perfect_ratio_max', 'perfect_bound_pct', 'perfect_bound_without_small_pct', 'V101_bound_pct', 'V104_bound_pct']
    print(G[cols].round(3).to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
