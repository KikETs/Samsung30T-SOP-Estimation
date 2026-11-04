"""V256 (PROTOCOL_CLOSEDLOOP_V256.md): the V255 closed-loop configurations reported over a range of intervention targets
tau = 4-6 % instead of a single value. usage: python closedloop_v256.py"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_envelope import HERE, dump
import closedloop_v255 as V5

OUT = HERE / 'closedloop_v256'
TAUS = np.round(np.arange(0.04, 0.06001, 0.0025), 4)
CONFIGS = [c for c in V5.CONFIGS if c[0] in ('2_sop_single_out', '3_hybrid_two_out', V5.MAIN)]


def v242(B, groups_all, tau):
    """V252 frame: one scale per evaluated cell from the other cells' V242 central predictions."""
    parts, scales = [], {}
    for c in V5.CELLS:
        cal = B[B.evaluation_group != c]; ev = B[B.evaluation_group == c].copy()
        s, floor = V5.choose(cal.v242.to_numpy(), cal.truth.to_numpy(), tau)
        scales[f'{c}|all'] = dict(scale=s, floor_bound=floor); ev['pred'] = s * ev.v242; parts.append(ev)
    J = pd.concat(parts); assert len(J) == 672
    summ, G = V5.group_table(J, groups_all)
    return dict(name='V242', tau=float(tau), scales=scales,
                held_out=dict(int_rate=summ['int_rate'], depth_q95=summ['depth_q95'], depth_max=summ['depth_max'], util_q50=summ['util_q50']),
                groups_pass=int(G.passed.sum()),
                failed={f"{r.evaluation_group} {r.direction} {r.temp_C} C {r.horizon_s:g} s": r.reason for r in G[~G.passed].itertuples()})


def reproduce(res):
    """tau = 5 % must equal V255 (configurations) and V252 (V242) exactly."""
    S5 = json.load(open(HERE / 'closedloop_v255/summary.json')); S2 = json.load(open(HERE / 'closedloop_v252/summary.json'))
    ok = {}
    for name, _, _, _ in CONFIGS:
        a, b = res[(name, 0.05)], S5[f'{name}|tau5']
        ok[name] = bool(a['groups_pass'] == b['groups_pass'] and abs(a['held_out']['int_rate'] - b['held_out']['int_rate']) < 1e-12
                        and abs(a['held_out']['util_q50'] - b['held_out']['util_q50']) < 1e-12)
    a, b = res[('V242', 0.05)], S2['V242']['tau_5pct']['held_out']
    ok['V242'] = bool(a['groups_pass'] == b['groups_util80_int10'] and abs(a['held_out']['int_rate'] - b['int_rate']) < 1e-12
                      and abs(a['held_out']['util_q50'] - b['util_q50']) < 1e-12)
    return ok


def main():
    OUT.mkdir(exist_ok=True)
    B = V5.load(); groups_all = B[V5.C1.GK].drop_duplicates().sort_values(V5.C1.GK).reset_index(drop=True); assert len(groups_all) == 15
    res, rows = {}, []
    for tau in TAUS:
        for name, predictor, kind, calib in CONFIGS:
            r, _, _ = V5.run(B, groups_all, name, predictor, kind, calib, float(tau)); res[(name, float(tau))] = r
        res[('V242', float(tau))] = v242(B, groups_all, float(tau))
        for name in [c[0] for c in CONFIGS] + ['V242']:
            r = res[(name, float(tau))]; h = r['held_out']
            rows.append(dict(tau=float(tau), config=name, groups_pass=r['groups_pass'], int_rate=h['int_rate'], util_q50=h['util_q50'],
                             depth_max=h['depth_max'], scales=' '.join(f"{k}={v['scale']:.2f}" for k, v in sorted(r['scales'].items())),
                             failed='; '.join(f'{g} ({w})' for g, w in r['failed'].items())))
    T = pd.DataFrame(rows); T.to_csv(OUT / 'table.csv', index=False)
    ok = reproduce(res); print('reproduction at tau 5 %:', ok, flush=True)
    if not all(ok.values()):
        dump(OUT / 'summary.json', dict(reproduction=ok)); raise SystemExit('tau 5 % does not reproduce V255/V252; stopping (protocol)')
    P = T.pivot(index='tau', columns='config', values='groups_pass'); I = T.pivot(index='tau', columns='config', values='int_rate')
    print(P.to_string()); print((100 * I).round(2).to_string())
    summ = {c: dict(groups_min=int(P[c].min()), groups_median=float(P[c].median()), groups_max=int(P[c].max()), int_rate_max=float(I[c].max()))
            for c in P.columns}
    m, v = P[V5.MAIN], P['V242']
    keep = bool((m >= 12).all() and (I[V5.MAIN] <= 0.10).all())
    better = bool((m >= v).all() and int((m > v).sum()) >= 5)
    verdict = [x for x, y in (('τ 범위에서 V242 수준 유지', keep), ('V242보다 나음', better)) if y] or ['τ에 민감']
    out = dict(protocol='PROTOCOL_CLOSEDLOOP_V256.md', taus=[float(t) for t in TAUS], reproduction=ok, summary=summ,
               main_vs_V242=dict(main_ge_all=bool((m >= v).all()), main_gt_count=int((m > v).sum()), keep_V242_level=keep, better_than_V242=better),
               verdict=verdict)
    dump(OUT / 'summary.json', out)
    print(json.dumps(summ, indent=1, ensure_ascii=False)); print('verdict (main):', verdict, flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for c, lab in ((V5.MAIN, 'hybrid, per class, out-of-sample (main)'), ('3_hybrid_two_out', 'hybrid, one sop scale, out-of-sample'),
                   ('2_sop_single_out', 'sop alone, out-of-sample'), ('V242', 'V242 (V252 frame)')):
        ax[0].plot(100 * P.index, P[c], marker='o', label=lab); ax[1].plot(100 * I.index, 100 * I[c], marker='o', label=lab)
    ax[0].set_xlabel('intervention target tau on the calibration rows (%)'); ax[0].set_ylabel('groups passed (util >= 80 % and int <= 10 %)')
    ax[0].set_ylim(0, 15.5); ax[0].axhline(12, color='gray', lw=0.8, ls=':'); ax[0].legend(fontsize=7)
    ax[1].set_xlabel('intervention target tau on the calibration rows (%)'); ax[1].set_ylabel('held-out intervention rate (%)')
    ax[1].axhline(10, color='gray', lw=0.8, ls=':')
    fig.suptitle('V256: closed-loop result over tau = 4-6 % (scales chosen without the evaluated cell)'); fig.tight_layout()
    fig.savefig(OUT / 'tau_range.png', dpi=130)


if __name__ == '__main__':
    main()
