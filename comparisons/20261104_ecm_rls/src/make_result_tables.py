"""Markdown tables of the comparator results, generated from the result files (no manual transcription).

    python -I -B src/make_result_tables.py > results/result_tables.md
"""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'


def read(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def pct(x):
    return f'{100 * float(x):.1f} %'


def main():
    prim = read(OUT / 'primary.csv')
    pilot = {(r['method'], r['policy']): r for r in read(ROOT / 'inputs/pilot_primary.csv')}
    gates = json.loads((OUT / 'gates.json').read_text(encoding='utf-8'))
    val = json.loads((OUT / 'validation.json').read_text(encoding='utf-8'))
    pol_ko = {'fixed': '고정 배율', 'AD': '배율 적응'}
    print('| 방법 | 요청 방식 | 합격 그룹 (/15) | 엄격 기준 (/15) | 초과 건수 (/672) | 초과율 | 사용률 중앙 | 최대 초과폭 | 초과폭 95 % | 누락 | 대체 |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in prim:
        print(f"| {r['candidate']} | {pol_ko[r['policy']]} | {r['closed_loop']} | {r['original']} | {r['exceed_rows']} | {pct(r['int_rate'])} | "
              f"{pct(r['util_q50'])} | {float(r['depth_max']):.3f} | {float(r['depth_p95']):.3f} | {r['missing']} | {r['fallback']} |")
    for pol in ('fixed', 'AD'):
        s = gates['main_reproduction'][pol]
        print(f"| (참고) 제안 예측기, 패키지 r3 | {pol_ko[pol]} | {s['closed_loop']} | {s['original']} | {s['exceed_rows']} | {pct(s['int_rate'])} | "
              f"{pct(s['util_q50'])} | {s['depth_max']:.3f} | {s['depth_p95']:.3f} | 0 | — |")
    names = {'PERSISTENCE': '(참고) 앞 펄스 전력', 'RINT_PREVIOUS': '(참고) 앞 펄스 저항'}
    for m in ('PERSISTENCE', 'RINT_PREVIOUS'):
        for pol in ('fixed', 'AD'):
            s = pilot[(m, pol)]
            print(f"| {names[m]} | {pol_ko[pol]} | {s['closed_loop']} | {s['original']} | {s['exceed_rows']} | {pct(s['int_rate'])} | "
                  f"{pct(s['util_q50'])} | {float(s['depth_max']):.3f} | {float(s['depth_p95']):.3f} | {s['missing']} | — |")
    print()
    print('| 방법 | 시작 배율 A1/A17/B (τ = 5 %) | 선택된 설정 A1 / A17 / B |')
    print('|---|---|---|')
    for r in prim:
        if r['policy'] == 'fixed':
            print(f"| {r['candidate']} | {r['scale_A1']}/{r['scale_A17']}/{r['scale_B']} | {r['config_A1']} / {r['config_A17']} / {r['config_B']} |")
    print()
    groups = read(OUT / 'groups.csv')
    cands = sorted({g['candidate'] for g in groups})
    keys = []
    for g in groups:
        k = (g['cell'], g['direction'], g['temp_C'], g['horizon_s'])
        if k not in keys:
            keys.append(k)
    print('| 그룹 | n | ' + ' | '.join(f'{c} {pol_ko[p]}' for c in cands for p in ('fixed', 'AD')) + ' |')
    print('|---|---|' + '---|' * (2 * len(cands)))
    for k in keys:
        cells = []
        n = ''
        for c in cands:
            for p in ('fixed', 'AD'):
                g = next(x for x in groups if x['candidate'] == c and x['policy'] == p and (x['cell'], x['direction'], x['temp_C'], x['horizon_s']) == k)
                n = g['n']
                mark = '합격' if g['passed'] == 'True' else '불합격'
                cells.append(f"{pct(g['util_q50'])}, 초과 {g['exceed']}, {mark}")
        d = '방전' if k[1] == 'discharge' else '충전'
        print(f"| {k[0]} {d} {float(k[2]):g} °C {float(k[3]):g} s | {n} | " + ' | '.join(cells) + ' |')
    print()
    iso = read(OUT / 'iso_budget.csv')
    print('| 방법 | 요청 방식 | 해당 점 수 | 최고 사용률 중앙 | 그때 τ | 그때 초과 건수 | 그때 합격 그룹 | 해당 점의 최대 합격 그룹 |')
    print('|---|---|---|---|---|---|---|---|')
    for r in iso:
        if r['eligible_points'] in ('0', ''):
            print(f"| {r['candidate']} | {pol_ko[r['policy']]} | 0 | — | — | — | — | — |")
        else:
            print(f"| {r['candidate']} | {pol_ko[r['policy']]} | {r['eligible_points']} | {pct(r['best_util'])} | {float(r['best_util_tau']):.4f} | "
                  f"{r['exceed_rows']} | {r['groups_at_best_util']} | {r['maximum_groups']} |")
    print()
    vv = read(OUT / 'voltage_validation.csv')
    print('| 검사 | 방법 | 셀 | 방향 | 온도 | 행 | 누락 | 끝 전압 오차 |중앙| (mV) | 95 % (mV) | 부호 있는 중앙 (mV) | 최대 오차 중앙 (mV) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in vv:
        sel = r['selected_for_this_cell'] == 'True'
        if r['check'] == 'causal_previous_pulse' and not sel:
            continue
        chk = '인과(앞 펄스)' if r['check'] == 'causal_previous_pulse' else '사후(쿼리 펄스)'
        d = '방전' if r['direction'] == 'discharge' else '충전'
        print(f"| {chk} | {r['candidate']} | {r['cell']} | {d} | {float(r['temp_C']):g} °C{'' if float(r['horizon_s']) == 10 else ' (2 s)'} | {r['rows']} | {r['missing']} | "
              f"{float(r['median_abs_end_mV']):.1f} | {float(r['p95_abs_end_mV']):.1f} | {float(r['median_signed_end_mV']):+.1f} | {float(r['median_max_abs_mV']):.1f} |")
    print()
    sens = read(OUT / 'lambda_sensitivity_posthoc.csv')
    print('| 방법 | 설정 (모든 셀) | 요청 방식 | 합격 그룹 | 엄격 | 초과 건수 | 사용률 중앙 | 누락 | 대체 |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in sens:
        print(f"| {r['candidate']} | {r['config_all_cells']} | {pol_ko[r['policy']]} | {r['closed_loop']} | {r['original']} | {r['exceed_rows']} | {pct(r['util_q50'])} | {r['missing']} | {r['fallback']} |")
    print()
    print('```json')
    print(json.dumps(dict(prediction_counts=val['prediction_counts'], solver_convergence=val['solver_convergence'],
                          independent_solver={k: v['max_abs_diff_W'] for k, v in val['independent_solver'].items()},
                          invariance=[(x['test'], x['candidate'], x['passed'], x.get('max_rel_diff_central')) for x in val['invariance']],
                          all_checks_passed=val['all_checks_passed']), indent=1, ensure_ascii=False))
    print('```')


if __name__ == '__main__':
    main()
