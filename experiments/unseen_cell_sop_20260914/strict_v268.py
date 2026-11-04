"""V268 (PROTOCOL_STRICT_V268.md): closed loop with the strict nested calibration of the discharge class. For evaluated cell c
the discharge calibration rows of each other cell d are predicted by the models fitted on the third cell e alone (strict
local five, + strict i3c for the full ensemble); charge and short-rest classes keep the V255 out-of-sample calibration.
Evaluated rows keep their fold-c centres (E3causal for the full ensemble, V267 light centres for the light one).
usage: python strict_v268.py   (inputs: strict_v268/src/local_strict.csv, i3c_strict_<e>.csv from the remote run)"""
import json
import numpy as np, pandas as pd
from run_envelope import HERE, dump
import closedloop_v255 as V5, closedloop_v256 as V6, light_v267 as L7

OUT = HERE / 'strict_v268'; SRC = OUT / 'src'
LOCAL = ['prev', 'L4', 'M6', 'DN6', 'G1']


def strict_centres(with_phys):
    W = pd.read_csv(SRC / 'local_strict.csv').pivot_table(index=['train_cell', 'row_id'], columns='name', values='pred_power_W').reset_index()
    cols = list(LOCAL)
    if with_phys:
        I = pd.concat([pd.read_csv(SRC / f'i3c_strict_{e}.csv').assign(train_cell=e) for e in V5.CELLS])
        W = W.merge(I[['train_cell', 'row_id', 'pred_power_W']].rename(columns={'pred_power_W': 'i3c'}), on=['train_cell', 'row_id'], how='left'); cols.append('i3c')
    L = np.log(W[cols].to_numpy(float)); L[~np.isfinite(L)] = np.nan
    with np.errstate(all='ignore'):
        W['centre'] = np.exp(np.nanmedian(L, axis=1))
    return W.set_index(['train_cell', 'row_id']).centre


def run(B, groups_all, name, predictor, kind, tau, S):
    """closedloop_v255.run with the strict calibration for the discharge class (S: centre by (train cell, row))."""
    K = V5.classes(B, predictor, kind); parts, scales = [], {}
    third = lambda c, d: [x for x in V5.CELLS if x not in (c, d)][0]
    for c in V5.CELLS:
        ev = B[B.evaluation_group == c].copy(); ev['cls'] = K[ev.index]
        ev['central'] = np.where((predictor == 'hybrid') & (ev.cls == 'short'), ev.v242, ev[f'sop_{c}']); ev['pred'] = np.nan
        for k in sorted(ev.cls.unique()):
            dropped = 0
            if S is not None and k == 'discharge':
                m = (B.evaluation_group != c) & (K == k); idx = B.index[m]
                p = np.array([S.get((third(c, B.at[i, 'evaluation_group']), B.at[i, 'row_id']), np.nan) for i in idx]); t = B.truth[m].to_numpy()
                ok = np.isfinite(p); dropped = int((~ok).sum()); p, t = p[ok], t[ok]
            else:
                p, t = V5.calibration(B, K, c, k, 'out')
            s, floor = V5.choose(p, t, tau)
            scales[f'{c}|{k}'] = dict(scale=s, floor_bound=floor, calibration_rows=int(len(p)), dropped_without_centre=dropped)
            ev.loc[ev.cls == k, 'pred'] = s * ev.loc[ev.cls == k, 'central']
        parts.append(ev)
    J = pd.concat(parts); assert len(J) == 672 and J.pred.notna().all()
    summ, G = V5.group_table(J, groups_all)
    return dict(tau=float(tau), groups_pass=int(G.passed.sum()), int_rate=summ['int_rate'], util_q50=summ['util_q50'], scales=scales,
                failed={f'{r.evaluation_group} {r.direction} {r.temp_C} C {r.horizon_s:g} s': r.reason for r in G[~G.passed].itertuples()}), G


def sweep(B, groups_all, label, S):
    per = []
    for tau in V6.TAUS:
        r, G = run(B, groups_all, label, 'hybrid', 'class', tau, S); per.append(r)
        if abs(tau - 0.05) < 1e-9:
            G.to_csv(OUT / f'groups_tau5_{label}.csv', index=False)
    gp = [p['groups_pass'] for p in per]
    return dict(groups_tau_4_6=[int(min(gp)), float(np.median(gp)), int(max(gp))], per_tau=per)


def main():
    OUT.mkdir(exist_ok=True); res = {}
    B = V5.load(); groups_all = B[V5.C1.GK].drop_duplicates().sort_values(V5.C1.GK).reset_index(drop=True); assert len(groups_all) == 15
    res['full_current'] = sweep(B, groups_all, 'full_current', None)
    if res['full_current']['groups_tau_4_6'] != [14, 14.0, 14]:
        raise SystemExit('current full result not reproduced; stopping (protocol)')
    res['full_strict'] = sweep(B, groups_all, 'full_strict', strict_centres(True))
    Bl = V5.load(); Fl = L7.light_folds(); P = Fl.pivot(index='row_id', columns='fold', values='m').reindex(Bl.row_id)
    for c in V5.CELLS:
        Bl[f'sop_{c}'] = np.exp(P[c].to_numpy())
    res['light_current'] = sweep(Bl, groups_all, 'light_current', None)
    if res['light_current']['groups_tau_4_6'] != [12, 12.0, 14]:
        raise SystemExit('current light result (V267) not reproduced; stopping (protocol)')
    res['light_strict'] = sweep(Bl, groups_all, 'light_strict', strict_centres(False))
    for v in ('full', 'light'):
        cur, st = res[f'{v}_current']['groups_tau_4_6'][1], res[f'{v}_strict']['groups_tau_4_6'][1]
        res[f'{v}_verdict'] = '간접 의존 영향 작음' if st >= cur - 1 else '간접 의존 영향 있음'
    rows = [dict(config=k, tau=p['tau'], groups_pass=p['groups_pass'], int_rate=p['int_rate'], util_q50=p['util_q50'],
                 **{f'scale_{s}': v['scale'] for s, v in p['scales'].items()}) for k in ('full_current', 'full_strict', 'light_current', 'light_strict') for p in res[k]['per_tau']]
    pd.DataFrame(rows).to_csv(OUT / 'tau_range.csv', index=False)
    dump(OUT / 'summary.json', res)
    for k in ('full_current', 'full_strict', 'light_current', 'light_strict'):
        t5 = [p for p in res[k]['per_tau'] if abs(p['tau'] - 0.05) < 1e-9][0]
        print(f"{k}: groups {res[k]['groups_tau_4_6']} | tau5 int {100 * t5['int_rate']:.2f}% util {100 * t5['util_q50']:.1f}% "
              f"discharge scales {[v['scale'] for s, v in t5['scales'].items() if s.endswith('discharge')]} failed {list(t5['failed'])}", flush=True)
    print('verdicts:', res['full_verdict'], '/', res['light_verdict'], flush=True)


if __name__ == '__main__':
    main()
