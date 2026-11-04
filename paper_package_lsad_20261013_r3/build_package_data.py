"""Builds facts.csv and data/*.csv of the SOP paper package, Ls + AD edition (2026-10-13; revision r2
2026-10-14). Main configuration: the five local predictors in their specification-audited form (V287 Ls, all fixes;
adopted as the main predictor by a rule fixed before its results) with the online derating-scale adaptation V284 AD
(gamma 0.05, tau* 5 %, start at the evaluated-cell-free tau = 5 % scale). Ablation: Ls with the fixed scale (B0).
Sensitivity: the pre-audit implementation L and the single audit fixes. Every number comes from frozen result files; the
only computations here are re-counts of stored rows (marked "re-count"). Post hoc analyses and the information-flow
checks of Ls + AD come from V320 (PROTOCOL_POSTHOC_LSAD_V320.md and its AMENDMENT_1, registered before they ran); r2 adds
V321 (PROTOCOL_POSTHOC_V321.md: adaptation signal, iso-exceedance comparison, B 2 s rows).
usage: python build_package_data.py   (from this folder; PYTHONDONTWRITEBYTECODE=1; the torch_env interpreter)"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent; TOP = PKG.parent / 'experiments' / 'unseen_cell_sop_20260914'; DATA = PKG / 'data'
DATA.mkdir(exist_ok=True)
sys.path.insert(0, str(TOP))
import closedloop_v251 as C1  # noqa: E402

J = lambda p: json.load(open(TOP / p))
facts = []
GK = C1.GK


def fact(fid, value, unit, status, ko, en, src, field):
    facts.append(dict(id=fid, value=value, unit=unit, status=status, description_ko=ko, description_en=en, source_file=src, source_field=field))


def trip(g):
    return f'{g[0]} / {g[1]:g} / {g[2]}'


def gname(x):
    return f'{x.evaluation_group} {x.direction} {x.temp_C} C {x.horizon_s:g} s'


def reason(x):
    r = []
    if x.util_q50 < 0.80:
        r.append('utilisation')
    if x.int_rate > 0.10:
        r.append('interventions')
    return '+'.join(r)


# ---------------- data / benchmark (unchanged from the L edition) ----------------
B = pd.read_csv(TOP / 'closedloop_v251/src/bench_rows.csv')
Gb = B.groupby(GK).size().rename('rows').reset_index(); Gb.to_csv(DATA / 'benchmark_groups.csv', index=False)
fact('D01', len(B), 'rows', 'descriptive', '벤치마크 행 수(전압 한계로 끝난 최종 펄스)', 'benchmark rows (voltage-limited final pulses)', 'closedloop_v251/src/bench_rows.csv', 'row count')
for c, n in B.groupby('evaluation_group').size().items():
    fact(f'D02_{c}', int(n), 'rows', 'descriptive', f'셀 {c}의 행 수', f'rows of cell {c}', 'closedloop_v251/src/bench_rows.csv', f'evaluation_group == {c}')
fact('D03', len(Gb), 'groups', 'descriptive', '판정 그룹 수(셀 × 방향 × 온도 × 펄스 길이)', 'evaluation groups (cell x direction x temperature x horizon)', 'closedloop_v251/src/bench_rows.csv', 'distinct groups')
fact('D04', 0.883, 'ratio', 'descriptive', 'B 셀 노화 정도(대리값): 25 °C 1C 용량 2.600 Ah ÷ A 셀 신품 평균 2.946 Ah', 'cell B ageing proxy: 25 C 1C capacity 2.600 Ah / fresh A-cell mean 2.946 Ah', 'DIAG_V261.md', 'B 셀 용량과 SOH 대리 정답 (정정)')
V178 = pd.read_csv(TOP / 'all6_measured_v178/predictions.csv')
fact('D05', int((V178.regime == 'current_limited').sum()), 'rows', 'descriptive', 'SOP 측정표에서 전류 한계로 끝나 벤치마크에서 뺀 행', 'current-limited rows of the SOP measurement table (not in the benchmark)', 'all6_measured_v178/predictions.csv', 'regime == current_limited')
fact('D06', '660 rows: |error| median 1.54 %, within 3 % 86.7 %, within 5 % 96.5 %, ratio range 0.827-1.363', '', 'descriptive (evidence package recomputation)',
     '정답 펄스 자신의 끝점 저항으로 다시 만든 전력과 정답의 차이(정의에서 나오는 항등 관계이며 예측 성능의 근거가 아님)',
     'reconstruction of the measured SOP from the reference pulse\'s own end-point resistance (a definitional identity, not evidence of predictive skill)',
     '../sop_evidence_handoff/discrepancies.md', 'D17')
RB = pd.read_csv(TOP / 'leakfree_v272/src/pybamm/out/rpcwby_bench/rows.csv')
FR = pd.read_csv(TOP / 'leakfree_v272/src/sop_ident/out/feats/relax.csv')
fq = RB[RB.context_pulses == 0]
fact('D07', f"{len(fq)} (charge {int((fq.direction == 'charge').sum())}, discharge {int((fq.direction == 'discharge').sum())})", 'rows', 'descriptive (re-count)',
     '같은 방향의 앞 펄스가 없는 첫 쿼리 행', 'first queries without a previous same-direction pulse in the sweep', 'leakfree_v272/src/pybamm/out/rpcwby_bench/rows.csv', 'context_pulses == 0')
P20 = J('posthoc_lsad_v320/summary.json')
import specaudit_v287 as SA  # noqa: E402  (spliced_relax: the audited previous-pulse features, a re-count of stored tables)
FS, _ = SA.spliced_relax(FR)
d8 = P20['D08_Ls']; dec = RB.truth_power_W.astype(str).str.split('.').str[1].str.len()
Mx = RB[['row_id', 'member', 'q_row', 'direction', 'truth_power_W']].merge(FS[['row_id', 'prev_P']], on='row_id').sort_values(['member', 'q_row'])
Mx['pt'] = Mx.groupby('member').truth_power_W.shift(1); Mx['pdir'] = Mx.groupby('member').direction.shift(1)
pr = Mx[Mx.prev_P.notna() & Mx.pt.notna() & (Mx.pdir == Mx.direction)]; big = pr[(pr.prev_P - pr.pt).abs() > 0.0005]
assert len(pr) == d8['pairs'] and abs((pr.prev_P - pr.pt).abs().max() - d8['max_abs_diff_W']) < 1e-9
fact('D08', f"{len(pr)} pairs; within 0.0005 W (the truth table's rounding): {len(pr) - len(big)}; the other {len(big)} take an earlier pulse under the 60-s previous-pulse rule: "
            + ', '.join(f'{x.row_id} (prev_P {x.prev_P:.2f} W against {x.pt:.3f} W)' for x in big.itertuples())
            + f"; truth values with 3 decimals: {int((dec == 3).sum())} of {len(RB)}; rows with previous-pulse features: {d8['rows_with_prev_features']}",
     '', 'post hoc re-count (V320 D08_Ls; the L-edition definition applied to the audited features)',
     '국소 입력 prev_P(앞 펄스 전력)와 같은 파일·같은 방향의 바로 앞 쿼리 참조 SOP의 차이(감사한 특징): 앞 SOC 지점의 참조 SOP 펄스가 입력으로 들어감',
     'previous-pulse power input prev_P (audited features) against the reference SOP of the immediately preceding same-direction query of the same file: the reference SOP pulse of the previous SOC point is an input',
     'posthoc_lsad_v320/summary.json; specaudit_v287/feats_fixed.csv; leakfree_v272/src/sop_ident/out/feats/relax.csv', 'D08_Ls; prev_P, truth_power_W')
RAWX = TOP.parents[1] / 'raw/RPCWBY/0_SOP_summary.xlsx'
h_old = pd.read_excel(RAWX, sheet_name='Samsung30T_old', header=None, nrows=1).iloc[0].tolist(); h_a = pd.read_excel(RAWX, sheet_name='Test#1', header=None, nrows=1).iloc[0].tolist()
lim = lambda h: f"current {h[h.index('SOP Limits:') + 2]} A, voltage {h[h.index('SOP Limits:') + 4]} V"
fact('D10', f"other Samsung 30T campaign of the same dataset (summary sheet Samsung30T_old, channel 5): SOP limits {lim(h_old)}; ageing cells A1, A17 (sheet Test#1): {lim(h_a)} ('n' = negative, charge)",
     '', 'descriptive (dataset summary sheet headers)', '같은 데이터셋의 다른 Samsung 30T 캠페인(범위 밖)의 SOP 한계와 A 셀의 한계', 'SOP limits of the other Samsung 30T campaign of the dataset (out of scope) and of the ageing cells',
     'raw/RPCWBY/0_SOP_summary.xlsx (outside this folder: Samsung30T/raw)', 'sheet headers Samsung30T_old, Test#1')
fact('D09', '164 of 672 voltage-limited rows (all charge, SOC set points 0.8-1.0) have a 0.04 V tolerance larger than 20 % of their voltage headroom', 'rows', 'descriptive (record)',
     '탐색 종료 허용폭(0.04 V)이 전압 여유의 20 %를 넘는 행', 'rows whose 0.04 V search tolerance exceeds 20 % of the voltage headroom', 'FEASIBILITY_AUDIT_V106.md', 'line 12')

# ---------------- definitions with numbers (DEF) ----------------
import closedloop_v253 as V3  # noqa: E402
import closedloop_v255 as V5d  # noqa: E402
import closedloop_v256 as V6d  # noqa: E402
import adaptive_v284 as A4  # noqa: E402
dfs = 'definition (code constants / protocol records)'
fact('DEF01', f"median utilisation >= {V5d.GROUP_UTIL:.2f} and intervention rate <= {V5d.GROUP_INT:.2f} per group", '', dfs,
     '그룹 합격(폐루프 기준)의 문턱', 'group-pass thresholds of the utilisation-intervention criterion', 'closedloop_v255.py', 'GROUP_UTIL, GROUP_INT')
fact('DEF02', f"s in {{{V5d.GRID[0]:.2f}, {V5d.GRID[1]:.2f}, ..., {V5d.GRID[-1]:.2f}}} ({len(V5d.GRID)} values); the largest s whose calibration-row intervention rate is <= tau; {V5d.GRID[0]:.2f} when none qualifies",
     '', dfs, '감액 배율의 격자와 고르는 규칙', 'derating-scale grid and selection rule', 'closedloop_v255.py', 'GRID, choose()')
fact('DEF03', f"main tau = 5 %; reported grid {100 * float(V6d.TAUS[0]):.2f}-{100 * float(V6d.TAUS[-1]):.2f} % in 0.25 % steps ({len(V6d.TAUS)} values); "
              'in the main configuration tau sets the starting scale of the online adaptation',
     '', dfs, '보정 목표 τ의 주 값과 보고 격자(주 구성에서는 온라인 적응의 시작 배율을 정함)', 'main calibration target and reported grid (in the main configuration: the starting scale of the adaptation)',
     'PROTOCOL_CLOSEDLOOP_V252.md; closedloop_v256.py; specaudit_v287.py', 'tau line; TAUS; requests_all')
fact('DEF04', 'search convergence at the last sample: |V - V_lim| <= 0.04 V (voltage-limited) or |I| - I_lim <= 0.1 A (current-limited); end voltages of the voltage-limited discharge pulses within -0.040 to +0.039 V of 2.55 V; up to 16 passes observed; records at 1 Hz',
     '', 'definition (dataset fields F_V_error / F_I_error, record)', 'SOP 탐색의 수렴 판정(탐색 종료 허용폭)과 기록 간격', 'convergence test of the SOP search (search termination tolerance) and record rate', 'RPCWBY_AGING_SOP_V84.md', 'lines 11-13, 48-50')
fact('DEF05', f"voltage {V3.VLIM['discharge']:.2f}-{V3.VLIM['charge']:.2f} V; current {V3.ICAP['discharge']:.0f} A discharge, {V3.ICAP['charge']:.0f} A charge", '', 'definition (dataset limits; code constants)',
     '전압·전류 한계', 'voltage and current limits', 'RPCWBY_AGING_SOP_V84.md; closedloop_v253.py', 'limits; VLIM, ICAP')
cond = []
for c, g in B.groupby('evaluation_group'):
    parts = []
    for (d, h), gg in g.groupby(['direction', 'horizon_s']):
        parts.append(f"{d} {h:g} s at {', '.join(f'{t:g}' for t in sorted(gg.temp_C.unique()))} C")
    cond.append(f"{c}: " + '; '.join(parts))
fact('DEF06', ' | '.join(cond), '', 'descriptive (re-count)', '셀별 시험 조건(방향, 펄스 길이, 온도)', 'test conditions per cell (direction, pulse length, temperature)', 'closedloop_v251/src/bench_rows.csv', 'evaluation_group, direction, horizon_s, temp_C')
fact('DEF08', 'R_off(t) = |V(t) - V_end| / |I_end| at t = 1, 2, 5, 10, 30, 60, 120, 240 s after the interruption (nearest sample); Rstep(h) at h = 1, 2, 10 s from the diffusion-superposition fit over tau = 1-30 s; '
              'V_pre and T_pre from the sample before the query; end-of-rest slope over the last 30 s (mV/s); previous-pulse resistance R(t) = (V(t) - V_pre,prev) / I(t) at 1, 2, 10 s, with V_pre,prev the rest voltage before that pulse',
     '', 'definition (feature code)', '특징 추출의 시각, 맞춤 구간, 기준 전압', 'feature extraction: sampling times, fit interval and reference voltages', 'cltrain_v274/src/relax_feats.py; specaudit_v287/src/relax_feats_audit.py', 'docstring, T_OFF, pulse_before')
fact('DEF09', 'invT = 1000 / (T_pre + 273.15) - 3.354; lT = log(load duration / 200 s); sqA = log(max(diffusion share of Rstep10, 0.02)); ext = obs x clip(Ah-gap ratio, 0, 3); amp = trendS x clip(obs / trS2, 0, 3), 1 when |trS2| <= 0.05; '
              'beta clipped to [-0.9, 1]; basis x median ratio when a region has fewer than max(12, 4 x coefficients) training rows; training rows with query current below 0.98 x the current limit; prev predictor: beta fitted with at least 8 rows and spread > 0.05, otherwise beta = 0',
     '', 'definition (local predictor code)', '특징 변환과 국소 예측기 상수', 'feature transforms and local-predictor constants', 'leakfree_v272/src/sop_ident/tools/local_sop2.py; leakfree_v272/src/sop_ident/tools/local_sop.py', 'build features, fit, fit()')
fact('DEF10', 'drift = 1e-3 x slope (mV/s) x horizon (s) in V; charge H = V_lim - V_pre - drift; discharge H = V_pre + drift - V_lim (slope signed, positive when the voltage rises)',
     '', 'definition (local predictor code)', '여유 전압과 휴지 끝 기울기 보정의 부호·단위', 'voltage headroom with the sign and unit of the end-of-rest drift', 'leakfree_v272/src/sop_ident/tools/local_sop.py', 'lines 59-60')
fact('DEF11', 'boundaries between the training cells\' SOC set-point groups discharge <= 0.1 | >= 0.15 and <= 0.3 | >= 0.4, charge <= 0.7 | >= 0.8, learned as V_pre thresholds: the midpoint of neighbouring values with the fewest misassigned rows (V_pre above: upper group); ties: the middle candidate, the lower one of the two middle ones',
     '', 'definition (region-boundary code)', '휴지 전압 구간 경계와 고르는 규칙', 'rest-voltage region boundaries and their selection rule', 'leakfree_v272.py', 'BOUNDS, threshold(min_error)')
fact('DEF12', 'previous pulse: a same-direction load of at most 12.5 s that follows a rest of at least 60 s (rest length from its first rest sample to the sample before the pulse); otherwise the search continues further back; '
              'the search for prev starts at the last load sample before the pre-query rest, prev2 at the last load sample before the rest preceding prev; regression current of the local fit = the measured end current |I_last| of the training query pulses; '
              "first-query region 'firstcv' only when the last load was the CC-CV charge (end voltage >= 4.19 V); interruption-basis fallback only for rows without an earlier pulse; "
              'a row with no prediction from any of the five predictors gets a central prediction of 0 and a request of 0 W (no estimate, no power), and calibration rows without a central prediction are dropped',
     '', 'definition (specification-audit rules of the main predictor, fixed before its results)', '감사한 국소 예측기(주 예측기)의 규칙: 앞 펄스 조건, 검색 시작점, 적합 전류, firstcv 조건, 대체 경로, 예측이 없는 행',
     'rules of the audited local predictors (main predictor): previous-pulse condition, search start, fit current, firstcv condition, fallback, rows without an estimate',
     'AUDIT_V287.md; specaudit_v287.py', 'R1, R2, L1, S1, S2; requests_B0')
fact('DEF13', f"log s_(k+1) = clip(log s_k + gamma (tau* - e_k), log {np.exp(A4.LO):.2f}, log {np.exp(A4.HI):.2f}); e_k = 1 when the k-th executed request of the cell exceeded the reference SOP "
              f"(the benchmark's intervention flag, a proxy for the engagement of the voltage clamp), else 0; gamma = {A4.GAMMA:g}, tau* = {A4.TAU:g}; one scale per evaluated cell for both directions; s_0 = the evaluated-cell-free tau = 5 % scale; "
              'order: the ageing cycle in the file name (cells A1, A17), month and day in the file name (cell B), then the query row within a file',
     '', 'definition (online adaptation rule and constants, registered before its results)', '온라인 배율 적응(AD)의 갱신식, 상수, 처리 순서', 'online derating-scale adaptation (AD): update rule, constants and processing order',
     'adaptive_v284.py; PROTOCOL_ADAPTIVE_V284.md', 'adapt(), LO, HI, GAMMA, TAU, file_order')
fact('DEF14', f"gamma in {{{', '.join(f'{g:g}' for g in A4.GRID_G)}}} x tau* in {{{', '.join(f'{t:g}' for t in A4.GRID_T)}}} ({len(A4.GRID_G) * len(A4.GRID_T)} settings); starting-scale target tau 4-6 % (nine values)",
     '', dfs, '온라인 적응의 민감도 격자', 'sensitivity grid of the online adaptation', 'adaptive_v284.py; closedloop_v256.py', 'GRID_G, GRID_T; TAUS')

# ---------------- main result: Ls + AD (V287), ablation Ls B0 ----------------
S87 = J('specaudit_v287/summary.json'); Ls = S87['Ls']
T87 = pd.read_csv(TOP / 'specaudit_v287/tau_range.csv'); G87 = pd.read_csv(TOP / 'specaudit_v287/groups_tau5.csv')
st = 'preregistered computation (V287 Ls with V284 AD; Ls adopted as the main predictor by a rule fixed before its results)'
ad, b0 = Ls['AD'], Ls['B0']
qa = T87[(T87.config == 'Ls') & (T87.policy == 'AD')]; qb = T87[(T87.config == 'Ls') & (T87.policy == 'B0')]
ga = G87[(G87.config == 'Ls') & (G87.policy == 'AD')].copy(); gb = G87[(G87.config == 'Ls') & (G87.policy == 'B0')].copy()
for g in (ga, gb):
    g['reason'] = [reason(x) if not x.passed else '' for x in g.itertuples()]
fact('M01', ad['closed_loop'], 'groups of 15', st, '주 결과: τ = 5 %에서 Ls + AD의 폐루프 합격 그룹', 'main result: groups passed by Ls + AD under the utilisation-intervention criterion at tau = 5 %', 'specaudit_v287/summary.json', 'Ls.AD.closed_loop')
fact('M02', trip(S87['tau_4_6']['Ls AD']), 'groups of 15', st, '주 결과: 시작 배율의 τ 4~6 %(9개 값)에서 합격 그룹 최소/중앙/최대', 'main result: groups passed over the starting-scale tau 4-6 % (min / median / max)', 'specaudit_v287/summary.json', 'tau_4_6["Ls AD"]')
fact('M03', ' '.join(f'{100 * t:.2f}:{g}' for t, g in zip(qa.tau, qa.closed_loop)), 'tau %:groups', st, '주 결과: τ별 합격 그룹', 'main result: groups passed per tau', 'specaudit_v287/tau_range.csv', 'config == Ls, policy == AD')
fact('M04', f"{100 * ad['int_rate']:.2f}", '%', st, '주 결과: τ 5 % 평가 셀 개입률(참조 SOP 초과율, 672행)', 'main result: held-out intervention rate (exceedance of the reference SOP) at tau 5 % (672 rows)', 'specaudit_v287/summary.json', 'Ls.AD.int_rate')
fact('M05', f"{100 * ad['util_q50']:.2f}", '%', st, '주 결과: τ 5 % 사용률 중앙(전체 행)', 'main result: median utilisation at tau 5 % (all rows)', 'specaudit_v287/summary.json', 'Ls.AD.util_q50')
fact('M06', f"{100 * qa.int_rate.max():.2f}", '%', st, '주 결과: τ 4~6 %에서 평가 셀 개입률 최대', 'main result: maximum held-out intervention rate over tau 4-6 %', 'specaudit_v287/tau_range.csv', 'config == Ls, policy == AD: max int_rate')
fact('M07', ' '.join(f'{c}={v:.2f}' for c, v in Ls['scales_tau5'].items()), 'scale', st, '주 결과: τ 5 % 시작 배율(평가 셀마다 하나, 평가 셀 없는 보정)', 'main result: starting scale per evaluated cell at tau 5 % (evaluated-cell-free calibration)', 'specaudit_v287/summary.json', 'Ls.scales_tau5')
fact('M08', '; '.join(f'{gname(x)} ({x.reason})' for x in ga[~ga.passed].itertuples()), '', st + '; re-count of the reason', '주 결과: τ 5 %에 기준 미달인 그룹(이유)', 'main result: groups below the criterion at tau 5 % (reason)', 'specaudit_v287/groups_tau5.csv', 'config == Ls, policy == AD, passed == False')
D20 = P20['D_AD']
fact('M09', f"{D20['intervened_rows']} / {D20['depth_q95']:.3f} / {D20['depth_max']:.3f}", 'rows / (request/truth - 1) / (request/truth - 1)', 'post hoc re-count (V320 D on the gated recomputation of the registered requests)',
     '주 결과: τ 5 % 개입 행 수 / 개입 깊이 95 % 분위 / 최대', 'main result at tau 5 %: intervened rows / intervention depth q95 / max', 'posthoc_lsad_v320/summary.json', 'D_AD')
fact('M10', '; '.join(f'{gname(x)}: {int(x.interventions)} of {int(x.n)} rows intervened ({100 * x.int_rate:.1f} %), median utilisation {100 * x.util_q50:.1f} %' for x in ga[~ga.passed].itertuples()),
     '', st, '주 결과: 기준 미달 그룹의 값', 'main result: values of the groups below the criterion', 'specaudit_v287/groups_tau5.csv', 'config == Ls, policy == AD')
ct = Ls['calibration_tau5']
fact('M14', ' '.join(f"{c}:{v['calibration_rows']} ({v['dropped_without_centre']} dropped without a centre)" for c, v in ct.items()), 'rows', st,
     '주 결과: 평가 셀마다 보정 행 수(다른 두 셀의 행 가운데 중심값이 있는 행)', 'main result: calibration rows per evaluated cell (rows of the other two cells with a centre)', 'specaudit_v287/summary.json', 'Ls.calibration_tau5')
S2 = J('leakfree_v272/summary.json')
ch = S2['region_changes_evaluated_rows']; nchg = sum(int(v.split('/')[0]) for k, v in ch.items() if k.startswith('vpre_main'))
fact('M17', f'{nchg}/672', 'rows', 'preregistered (V272)', '휴지 전압 경계로 구간이 SOC 설정값 구간과 달라진 평가 행', 'evaluated rows whose rest-voltage region differs from the SOC-set-point region', 'leakfree_v272/summary.json', 'region_changes_evaluated_rows (vpre_main)')
fact('M18', f"{ad['original']} ({ad['exceed_rows']} exceedance rows)", 'groups of 15', st, '같은 요청을 원래 기준(그룹 안 초과 0건, 요청/정답 중앙 80 % 이상)으로 센 합격 그룹',
     'the same requests under the original criterion (no exceedance and median request/truth >= 80 % in a group)', 'specaudit_v287/summary.json', 'Ls.AD.original, exceed_rows')
o10 = P20['O10_Ls']; x2 = ga.set_index(GK).loc[('B', 'discharge', 25, 2.0)]
fact('M19', f"rows without a central prediction: {len(o10['rows_without_centre'])} ({', '.join(o10['rows_without_centre'])}), request 0 W; B discharge 25 C 2 s: {int(x2.interventions)} of {int(x2.n)} rows intervened, median utilisation {100 * x2.util_q50:.1f} %",
     '', st + '; post hoc re-count (V320)', '주 결과: 중심 예측이 없는 행과 B 25 °C 2초 그룹의 값', 'main result: rows without a central prediction and the B 25 C 2 s group', 'posthoc_lsad_v320/summary.json; specaudit_v287/groups_tau5.csv', 'O10_Ls.rows_without_centre; config == Ls, policy == AD')
TRJ = pd.read_csv(TOP / 'posthoc_lsad_v320/ad_trajectory.csv'); TRJ.to_csv(DATA / 'ad_trajectory.csv', index=False)
fact('M20', '; '.join(f"{c}: start {g.scale.iloc[0]:.3f}, end {g.scale.iloc[-1]:.3f}, min {g.scale.min():.3f}, max {g.scale.max():.3f}, interventions {int(g.intervened.sum())} of {len(g)}"
                      for c, g in TRJ.sort_values(['cell', 'order']).groupby('cell')), '', 'post hoc re-count (V320 AD trajectory of the main configuration)',
     '주 결과: 셀마다 적응 배율의 시작·끝·최소·최대와 개입 행', 'main result: adapted scale per cell (start, end, min, max) and intervened rows', 'posthoc_lsad_v320/ad_trajectory.csv', 'scale, intervened')
TG = TRJ.groupby(['cell', 'direction', 'temp_C', 'horizon_s']).agg(rows=('scale', 'size'), first_position=('order', 'min'), last_position=('order', 'max'),
                                                                  scale_min=('scale', 'min'), scale_max=('scale', 'max'), interventions=('intervened', 'sum')).reset_index()
TG.to_csv(DATA / 'ad_scale_by_group.csv', index=False)
fact('M21', '; '.join(f"{x.cell} {x.direction[:3]} {x.temp_C:g} C {x.horizon_s:g} s: positions {x.first_position}-{x.last_position}, scale {x.scale_min:.3f}-{x.scale_max:.3f}" for x in TG.itertuples()),
     '', 'post hoc re-count (V320 AD trajectory of the main configuration)', '주 결과: 그룹마다 처리 순서의 위치와 그 동안의 적응 배율 범위', 'main result: processing positions of each group and the adapted scale range during them',
     'posthoc_lsad_v320/ad_trajectory.csv', 'order, scale per group')
sb = 'preregistered computation (V287 Ls, fixed scale B0); ablation without the online adaptation'
fact('AB01', f"{b0['closed_loop']} (tau 4-6 %: {trip(S87['tau_4_6']['Ls B0'])})", 'groups of 15', sb, '절제: 적응 없이 고정 배율(Ls B0)의 합격 그룹(τ 5 %, τ 4~6 %)', 'ablation: groups passed by Ls with the fixed scale (B0) at tau 5 % and over tau 4-6 %', 'specaudit_v287/summary.json', 'Ls.B0.closed_loop; tau_4_6["Ls B0"]')
fact('AB02', f"intervention {100 * b0['int_rate']:.2f} %, median utilisation {100 * b0['util_q50']:.2f} %, original criterion {b0['original']} ({b0['exceed_rows']} exceedance rows)", '', sb,
     '절제: Ls B0의 개입률, 사용률 중앙, 원래 기준', 'ablation: Ls B0 intervention rate, median utilisation and original criterion', 'specaudit_v287/summary.json', 'Ls.B0')
fact('AB03', '; '.join(f'{gname(x)} ({x.reason}: median utilisation {100 * x.util_q50:.1f} %, {int(x.interventions)} of {int(x.n)} rows intervened)' for x in gb[~gb.passed].itertuples()), '', sb + '; re-count of the reason',
     '절제: Ls B0에서 기준 미달인 그룹(이유, 사용률 중앙, 개입 행)', 'ablation: groups below the criterion with Ls B0 (reason, median utilisation, intervened rows)', 'specaudit_v287/groups_tau5.csv', 'config == Ls, policy == B0')
Db = P20['D_B0']
fact('AB04', f"{Db['intervened_rows']} / {Db['depth_q95']:.3f} / {Db['depth_max']:.3f}", 'rows / (request/truth - 1) / (request/truth - 1)', 'post hoc re-count (V320 D)', '절제: Ls B0의 개입 행 수 / 개입 깊이 95 % 분위 / 최대',
     'ablation: Ls B0 intervened rows / intervention depth q95 / max', 'posthoc_lsad_v320/summary.json', 'D_B0')
xb = gb.set_index(GK)
fact('AB05', '; '.join(f"B discharge {t} C 10 s: median utilisation {100 * xb.loc[('B', 'discharge', t, 10.0), 'util_q50']:.1f} %, {int(xb.loc[('B', 'discharge', t, 10.0), 'interventions'])} of "
                       f"{int(xb.loc[('B', 'discharge', t, 10.0), 'n'])} rows intervened, {'pass' if xb.loc[('B', 'discharge', t, 10.0), 'passed'] else 'fail'}" for t in (-20, -10)), '', sb + '; re-count',
     '절제: 고정 배율에서 B −20 °C, −10 °C 그룹(주 구성에서는 사용률로 미달)', 'ablation: the B -20 C and -10 C groups with the fixed scale (below the criterion on utilisation in the main configuration)',
     'specaudit_v287/groups_tau5.csv', 'config == Ls, policy == B0')
pd.concat([ga.assign(arm='Ls+AD'), gb.assign(arm='Ls B0')], ignore_index=True).drop(columns=['config', 'policy']).to_csv(DATA / 'groups_tau5.csv', index=False)
T87.to_csv(DATA / 'tau_range.csv', index=False)
TH = pd.read_csv(TOP / 'leakfree_v272/thresholds.csv'); TH = TH[TH.rule == 'vpre_main']
TH[['model', 'model_cell', 'training_cells', 'boundary', 'soc_boundary', 'defined', 't_V', 'n_lower', 'n_upper', 'errors', 'ties']].to_csv(DATA / 'vpre_boundaries.csv', index=False)

# ---------------- sensitivity: pre-audit implementation, single fixes, adaptation grid ----------------
L8 = J('compare_v278/L/run.json'); A84 = J('adaptive_v284/summary.json'); P86 = J('s5_v286/posthoc.json')
fact('SE01', f"pre-audit implementation L: fixed scale {L8['tau5']['B0']['closed_loop']} (tau 4-6 %: {trip(L8['tau_4_6']['B0'])}, original {L8['tau5']['B0']['original']}); "
             f"with AD {A84['AD']['closed_loop']} (tau 4-6 %: {trip(P86['tau_4_6']['L AD'])}, original {A84['AD']['original']}, intervention {100 * A84['AD']['int_rate']:.2f} %, median utilisation {100 * A84['AD']['util_q50']:.2f} %, depth max {A84['AD']['depth_max']:.3f})",
     'groups of 15', 'preregistered (V278 L, V284 AD); tau 4-6 % of L + AD from the V286 post hoc sensitivity', '민감도: 감사 전 구현 L의 고정 배율과 AD 결과', 'sensitivity: the pre-audit implementation L with the fixed scale and with AD',
     'compare_v278/L/run.json; adaptive_v284/summary.json; s5_v286/posthoc.json', 'tau5.B0, tau_4_6.B0; AD; tau_4_6["L AD"]')
comp = []
for k in ('only_R12', 'only_L1', 'only_S1', 'only_S2'):
    x = S87[k]; comp.append(dict(config=k.replace('only_', 'only '), B0_tau5=x['B0']['closed_loop'], B0_tau_4_6=trip(S87['tau_4_6'][f'{k} B0']), AD_tau5=x['AD']['closed_loop'], AD_tau_4_6=trip(S87['tau_4_6'][f'{k} AD']),
                                 AD_original=x['AD']['original'], AD_depth_max=round(x['AD']['depth_max'], 4)))
comp = [dict(config='L (pre-audit implementation)', B0_tau5=L8['tau5']['B0']['closed_loop'], B0_tau_4_6=trip(L8['tau_4_6']['B0']), AD_tau5=A84['AD']['closed_loop'], AD_tau_4_6=trip(P86['tau_4_6']['L AD']),
             AD_original=A84['AD']['original'], AD_depth_max=round(A84['AD']['depth_max'], 4))] + comp + \
       [dict(config='Ls (all fixes; main)', B0_tau5=b0['closed_loop'], B0_tau_4_6=trip(S87['tau_4_6']['Ls B0']), AD_tau5=ad['closed_loop'], AD_tau_4_6=trip(S87['tau_4_6']['Ls AD']), AD_original=ad['original'], AD_depth_max=round(ad['depth_max'], 4))]
pd.DataFrame(comp).to_csv(DATA / 'audit_components.csv', index=False)
fact('SE02', '; '.join(f"{r['config']}: fixed {r['B0_tau5']} ({r['B0_tau_4_6']}), AD {r['AD_tau5']} ({r['AD_tau_4_6']})" for r in comp[1:-1]), 'groups of 15 at tau 5 % (tau 4-6 %)',
     'preregistered (V287; descriptive: the single fixes are reported to understand the audit, never to choose among them)', '민감도: 감사 수정을 하나씩만 넣은 판(서술용, 고르지 않음)', 'sensitivity: one audit fix at a time (descriptive; not a selection)',
     'specaudit_v287/summary.json', 'only_R12, only_L1, only_S1, only_S2; tau_4_6')
fact('SE05', '; '.join(S87['Ls']['AD_group_changes_vs_L']), '', 'preregistered (V287)', '민감도: 감사 전 구현 L + AD와 비교해 판정이 바뀐 그룹', 'sensitivity: groups whose verdict changed against the pre-audit L + AD', 'specaudit_v287/summary.json', 'Ls.AD_group_changes_vs_L')
fact('SE06', "in the aged cell's 25 C files (2 s and 10 s) the rests before the intermediate pulses are 0.04-3.4 s, so the 60-s previous-pulse rule reaches back to the first pulse after the full charge "
             '(900-s rest, V_pre 4.156 V, SOC about 1.0); the B 2-s centres become 0 (no estimate), 61.4 and 59.8 W against 63.8, 44.0 and 34.6 W; under the adaptation the two over-requests lowered the B scale, '
             'and the B -20 C and -10 C groups fell below 80 % utilisation; through the A1 fold, which uses the B rows, the A1 starting scale changed from 0.77 to 0.76',
     '', 'descriptive (V287 report, cause analysis of the audit)', '민감도: 감사 판과 감사 전 판의 차이가 난 이유(60초 휴지 규칙과 B 25 °C 파일의 짧은 펄스 사이 휴지)',
     "sensitivity: why the audited and pre-audit predictors differ (the 60-s previous-pulse rule and the short rests between pulses in the aged cell's 25 C files)", 'SPECAUDIT_V287.md', 'section "원인: R1", lines 62-70')
AG = pd.read_csv(TOP / 'specaudit_v287/Ls_adapt_grid.csv'); AG.to_csv(DATA / 'adapt_grid.csv', index=False)
main_cell = AG[(AG.gamma == A4.GAMMA) & (AG.tau_star == A4.TAU)].closed_loop.iloc[0]
fact('SE03', f"{AG.closed_loop.min()}-{AG.closed_loop.max()} over {len(AG)} settings (main setting gamma {A4.GAMMA:g}, tau* {A4.TAU:g}: {main_cell}); "
             + ' '.join(f'{g:g}|{t:g}:{c}' for g, t, c in zip(AG.gamma, AG.tau_star, AG.closed_loop)), 'groups of 15', 'preregistered (V287 adaptation grid; descriptive)',
     '민감도: 적응 격자(γ × τ*)에서 합격 그룹', 'sensitivity: groups passed over the adaptation grid (gamma x tau*)', 'specaudit_v287/Ls_adapt_grid.csv', 'gamma, tau_star, closed_loop')
SF = pd.read_csv(TOP / 'posthoc_lsad_v320/start_scale_factor_AD.csv'); SF.to_csv(DATA / 'start_scale_factor_AD_posthoc.csv', index=False)
fact('SE04', ' '.join(f'{r_.factor:.2f}:{r_.groups_pass}' for r_ in SF.itertuples()), 'factor:groups', 'post hoc (V320 S02_AD; V283 scale-factor sensitivity applied to the AD starting scales)',
     'τ 5 %: 세 셀의 시작 배율에 계수 f를 곱하고 AD를 다시 돌린 합격 그룹', 'tau 5 %: groups passed when every starting scale is multiplied by f and AD is rerun', 'posthoc_lsad_v320/start_scale_factor_AD.csv', 'factor, groups_pass')
TS = pd.read_csv(TOP / 'posthoc_lsad_v320/threshold_sensitivity.csv'); TS.to_csv(DATA / 'threshold_sensitivity_posthoc.csv', index=False)
t1 = TS[(TS.policy == 'AD') & (TS.int_threshold == 0.10)]; t2 = TS[(TS.policy == 'AD') & (TS.util_threshold == 0.80)]
fact('S01', f"intervention <= 10 %, utilisation 70/75/80/85/90 %: {'/'.join(map(str, t1.groups_pass))}; utilisation >= 80 %, intervention 5/10/15/20 %: {'/'.join(map(str, t2.groups_pass))}", 'groups of 15',
     'post hoc (V320 S01; V281 definition)', 'τ 5 % Ls + AD: 문턱을 바꿀 때 합격 그룹', 'tau 5 % Ls + AD: groups passed under other thresholds', 'posthoc_lsad_v320/threshold_sensitivity.csv', 'policy == AD')

# ---------------- information-flow checks of Ls + AD (V320) ----------------
I1 = P20['I1']
mx = max(max(v['max_rel_central'], v['max_rel_pred']) for v in I1.values())
shutil.copy(TOP / 'posthoc_lsad_v320/invariance_tests.csv', DATA / 'invariance_ls.csv')
fact('I01', f"{len(I1)} test x cell comparisons (T1-T4 x A1, A17, B; centres, fixed-scale requests at the nine tau values and scales): max relative difference {mx:g}; scales equal: {all(v['scales_equal'] for v in I1.values())}; all pass: {P20['I1_all_pass']}",
     '', 'post hoc check (V320 I1; V276 tests and tolerance relative 1e-12, applied to Ls)', '정보 흐름 불변 검사(Ls, 672행): 정답 변경, 쿼리 뒤 자료 삭제, SOC·SOH·soc_point 변경, 같은 탐색 이전 반복 삭제',
     'information-flow invariance tests of Ls (672 rows): truth change, removal of post-query data, change of SOC / SOH / set point, removal of earlier iterations of the same search', 'posthoc_lsad_v320/summary.json', 'I1, I1_all_pass')
i2, i3 = P20['I2'], P20['I3']
fact('I02', f"{i2['perturbations']} perturbations (every row of every cell in processing order: that row's and all later rows' truths x U(0.5, 2)); requests up to that row changed in {i2['violations']}",
     '', 'post hoc check (V320 I2)', 'AD 인과성: 행과 그 뒤 행의 정답을 바꿔도 그 행까지의 요청이 바뀌지 않음', 'AD causality: a row\'s request does not change when the truths of that row and of later rows change', 'posthoc_lsad_v320/summary.json', 'I2')
fact('I03', f"truths changed without changing any intervention flag: flags equal {i3['events_equal']}, max |request difference| {i3['max_abs_diff']:g} W", '', 'post hoc check (V320 I3)',
     'AD는 개입 여부만 읽음: 개입 여부를 바꾸지 않고 정답을 바꾸면 요청이 바뀌지 않음', 'AD reads only the intervention flag: truths changed without changing any flag leave the requests unchanged', 'posthoc_lsad_v320/summary.json', 'I3')
hs = {k: hashlib.sha256((TOP / f'posthoc_lsad_v320/feats_{k}_fixed.csv').read_bytes()).hexdigest() for k in ('ref', 'T2', 'T4')}
fact('I04', f"audited features recomputed from the raw files with the query sample removed (T2) and with the earlier iterations of each row's own search removed (T4) are byte-identical to the unmasked recomputation (672 rows): {hs['ref'] == hs['T2'] == hs['T4']}",
     '', 'post hoc check (V320 step 1; V276 masks)', '감사한 특징을 쿼리 표본을 가리고, 또 같은 탐색의 이전 반복을 가리고 다시 계산해도 바이트 단위로 같음', 'audited features are byte-identical when recomputed without the query sample or without earlier same-search iterations',
     'posthoc_lsad_v320/feats_ref_fixed.csv; posthoc_lsad_v320/feats_T2_fixed.csv; posthoc_lsad_v320/feats_T4_fixed.csv', 'sha256')

# ---------------- short-rest rows (B 25 C 2 s) ----------------
Fi = FR.set_index('row_id'); RS = pd.read_csv(TOP / 'specaudit_v287/requests_Ls_B0.csv', float_precision='round_trip').set_index('row_id')
tj = TRJ.set_index('row_id'); rows = []
for r in ('test3:325', 'test3:330', 'test3:335'):
    rows.append(dict(row_id=r, rest_s=Fi.loc[r, 'rest_s'], centre_Ls=tj.loc[r, 'central'], request_AD=tj.loc[r, 'request'], scale_AD=tj.loc[r, 'scale'], request_B0=RS.loc[r, 'request'], truth=tj.loc[r, 'truth'],
                     ratio_AD=tj.loc[r, 'request'] / tj.loc[r, 'truth'], order_in_cell=int(tj.loc[r, 'order'])))
SR = pd.DataFrame(rows); SR.to_csv(DATA / 'short_rest_rows_paths.csv', index=False)
fact('K05', '; '.join(f"{x.row_id}: pre-query rest {round(x.rest_s, 3)} s, Ls centre {x.centre_Ls:.2f} W, AD request {x.request_AD:.2f} W (scale {x.scale_AD:.3f}), truth {x.truth:.3f} W, request/truth {x.ratio_AD:.3f}, position {x.order_in_cell} of cell B"
                      for x in SR.itertuples()), '', 'post hoc (descriptive; V320 trajectory)', 'B 25 °C 2초 세 행: 휴지, 중심 예측, AD 요청, 처리 순서', 'the three B 25 C 2 s rows: rest, centre, AD request and position in the processing order',
     'posthoc_lsad_v320/ad_trajectory.csv; leakfree_v272/src/sop_ident/out/feats/relax.csv', 'central, request, scale, truth, order; rest_s')

# ---------------- post hoc analyses (V320) ----------------
ph = 'post hoc (V320; V281 / V283 definitions applied to Ls + AD)'
fact('O10', f"max {o10['max']:.3f} ({o10['where']}); share above 1.2: {100 * o10['share_above_1_2']:.2f} %; rows above 1.3: {o10['rows_above_1_3']}; q50 / q95 / q99 {o10['q50']:.3f} / {o10['q95']:.3f} / {o10['q99']:.3f}; rows with a centre {o10['rows_with_centre']}",
     'held-out central prediction / truth', ph, '평가 셀 중심 예측(Ls)의 과대 꼬리', 'over-prediction tail of the held-out Ls centres', 'posthoc_lsad_v320/summary.json', 'O10_Ls')
fact('O02', '; '.join(P20['O2_AD']['failed']), '', ph, 'τ 5 % Ls + AD 요청: 원래 기준에서 떨어진 그룹(이유)', 'tau = 5 % Ls + AD requests: groups failing the original criterion (reason)', 'posthoc_lsad_v320/summary.json', 'O2_AD.failed')
d = P20['E_AD']; ep = 'post hoc (V320; V253 response models, ideal voltage clamp)'
shutil.copy(TOP / 'posthoc_lsad_v320/delivery_rows_LsAD.csv', DATA / 'rows_tau5_with_delivery_posthoc.csv')
fact('E01', d['interventions'], 'rows', ep, 'τ 5 % 개입 행 수(전달 모의 대상)', 'intervened rows at tau 5 % (simulated)', 'posthoc_lsad_v320/summary.json', 'E_AD.interventions')
fact('E02', f"{d['del_over_req']['q50']:.3f} / {d['del_over_req']['q05']:.3f} / {d['del_over_req']['min']:.3f}", 'delivered/requested energy', ep, '개입 행의 전달/요청 에너지 중앙/5 % 분위/최소', 'delivered / requested energy of intervened rows (median / q05 / min)', 'posthoc_lsad_v320/summary.json', 'E_AD.del_over_req')
fact('E03', f"{d['del_over_truth']['min']:.4f} ({100 * d['share_del_over_truth_ge_1']:.0f} % >= 1)", 'delivered/truth energy', ep, '개입 행의 전달/정답 에너지 최솟값(1 이상인 비율)', 'minimum delivered / measured-SOP energy of intervened rows (share >= 1)', 'posthoc_lsad_v320/summary.json', 'E_AD.del_over_truth')
DR = pd.read_csv(TOP / 'posthoc_lsad_v320/delivery_rows_LsAD.csv'); itd = DR[DR.intervened]; ncl = int(itd.t_clamp.notna().sum())
fact('E04', f"{np.median(itd.t_clamp.dropna()):.2f} s over the {ncl} of {len(itd)} intervened rows that reached the clamp" if ncl else f'no clamp in {len(itd)} rows', 's', ep,
     '전압 고정 시작 시각 중앙(분모: 고정이 걸린 개입 행)', 'median clamp onset time (denominator: intervened rows that reached the clamp)', 'posthoc_lsad_v320/delivery_rows_LsAD.csv', 't_clamp')
fact('E05', d['groups_delivered_util80_int10'], 'groups of 15', ep, '전달 사용률로 센 합격 그룹(전달 사용률 중앙 ≥ 80 %, 벤치마크 개입률 ≤ 10 %)',
     'groups passing with delivered utilisation (median delivered utilisation >= 80 % and benchmark intervention rate <= 10 %)', 'posthoc_lsad_v320/summary.json', 'E_AD.groups_delivered_util80_int10')
P21 = J('posthoc_v321/summary.json'); p21 = 'post hoc (V321; V253 response models)'
SD = pd.read_csv(TOP / 'posthoc_v321/signal_disagreement.csv'); SD.to_csv(DATA / 'signal_disagreement_posthoc.csv', index=False)
sa = SD[SD.requests == 'AD'].set_index('direction')
SRw = pd.read_csv(TOP / 'posthoc_v321/sigclamp_rows.csv').merge(B[['row_id', 'soc_point']], on='row_id', how='left'); assert len(SRw) == 672
nc = SRw[SRw.AD_bench_event & ~SRw.AD_sim_clamp]; ncr = nc.AD_request / nc.truth
assert len(nc) == int(sa.loc['all', 'bench_only']) and (nc.direction == 'charge').all()
fact('E06', f"{int(sa.loc['all', 'both'])} of {int(sa.loc['all', 'bench_events'])} requests above the reference SOP reached the simulated clamp (discharge {int(sa.loc['discharge', 'both'])} of {int(sa.loc['discharge', 'bench_events'])}, "
            f"charge {int(sa.loc['charge', 'both'])} of {int(sa.loc['charge', 'bench_events'])}); the {len(nc)} that did not are all charge requests at SOC set points {nc.soc_point.min():.2f}-{nc.soc_point.max():.2f}, "
            f"request/reference {ncr.min():.3f}-{ncr.max():.3f} (" + ', '.join(f'{g[0]} {g[1]} {g[2]} C {g[3]:g} s: {len(x)}' for g, x in nc.groupby(GK)) + "); "
            f"requests at or below the reference SOP that reached the simulated clamp: {int(sa.loc['all', 'sim_only'])}",
     '', p21, '주 구성의 요청에서 벤치마크 개입 여부와 모의 전압 고정의 일치: 참조를 넘은 요청 가운데 모의 고정에 닿은 행과 닿지 않은 행',
     'agreement between the benchmark intervention flag and the simulated voltage clamp on the main requests: requests above the reference SOP that did and did not reach the simulated clamp',
     'posthoc_v321/signal_disagreement.csv; posthoc_v321/sigclamp_rows.csv; closedloop_v251/src/bench_rows.csv', "requests == AD; AD_bench_event, AD_sim_clamp; soc_point")
E53 = pd.read_csv(TOP / 'closedloop_v253/ecm_rows.csv'); ce = E53.groupby('direction').clamped_at_truth.agg(['sum', 'size'])
fact('E07', f"{int(E53.clamped_at_truth.sum())} of {len(E53)} rows ({100 * E53.clamped_at_truth.mean():.1f} %; charge {int(ce.loc['charge', 'sum'])} of {int(ce.loc['charge', 'size'])}, discharge {int(ce.loc['discharge', 'sum'])} of {int(ce.loc['discharge', 'size'])})",
     'rows', 'preregistered (V253 fit check)', '응답 모델에 참조 SOP 자체를 걸었을 때 모의 전압이 한계에 닿는 행(참조 펄스는 탐색 종료 허용폭 안에서 한계 조금 안쪽에서 끝나는 경우가 많음)',
     'rows whose response model reaches the voltage limit when driven at the reference SOP itself (consistent with reference pulses ending inside the limit within the search termination tolerance)',
     'closedloop_v253/ecm_rows.csv; closedloop_v253/summary.json', 'clamped_at_truth; fit.share_clamped_at_truth')
S53 = J('closedloop_v253/summary.json')['seed_validation']
fact('E10', S53['n'], 'pulses', 'preregistered (V253)', '응답 모델 검증에 쓴 CC-CV 시드 펄스 수', 'CC-CV seed pulses used to validate the response models', 'closedloop_v253/summary.json', 'seed_validation.n')
fact('E11', f"{100 * S53['median_abs_log_err']:.1f} / {100 * S53['by_direction']['charge']['median_abs']:.1f} / {100 * S53['by_direction']['discharge']['median_abs']:.1f}", '% (|log error| median)', 'preregistered (V253)',
     '모의 끝 전력 오차 중앙: 전체/충전/방전', 'simulated end-power error median: all / charge / discharge', 'closedloop_v253/summary.json', 'seed_validation.median_abs_log_err, by_direction')
fact('E12', round(100 * S53['clamp_agreement'], 1), '%', 'preregistered (V253)', '전압 고정 여부 일치율', 'clamp agreement', 'closedloop_v253/summary.json', 'seed_validation.clamp_agreement')
FQ = pd.read_csv(TOP / 'posthoc_lsad_v320/first_query_LsAD.csv'); FQ.to_csv(DATA / 'first_query_posthoc.csv', index=False)
fa = FQ[(FQ.policy == 'AD') & (FQ.direction == 'all')].set_index('first_query')
fact('F01', '; '.join(f"{'first queries' if k else 'other rows'}: {int(r_.rows)} rows, {int(r_.interventions)} interventions ({100 * r_.int_rate:.2f} %), median utilisation {100 * r_.util_q50:.1f} %, request/truth q95 {r_.ratio_q95:.3f}, max {r_.ratio_max:.3f}"
                      for k, r_ in fa.sort_index(ascending=False).iterrows()), '', 'post hoc (V320 K6, descriptive)', 'τ 5 % Ls + AD: 첫 쿼리와 나머지 행', 'tau 5 % Ls + AD: first queries and the other rows', 'posthoc_lsad_v320/summary.json', 'K6 (policy AD)')

# ---------------- V321 (post hoc) ----------------
GS = pd.read_csv(TOP / 'posthoc_v321/sigclamp_groups.csv'); GS.to_csv(DATA / 'signal_sensitivity_posthoc.csv', index=False)
bs, ba = P21['B']['SIG']['benchmark'], P21['B']['AD']['benchmark']
fact('SG01', f"{bs['closed_loop']} groups (main configuration {ba['closed_loop']}); original criterion {bs['original']} ({bs['exceed_rows']} exceedance rows; main {ba['original']}, {ba['exceed_rows']}); "
             f"intervention {100 * bs['int_rate']:.2f} % (main {100 * ba['int_rate']:.2f} %), median utilisation {100 * bs['util_q50']:.2f} % (main {100 * ba['util_q50']:.2f} %), depth max {bs['depth_max']:.3f}; "
             f"groups below the criterion: {'; '.join(bs['failed_groups'])}; verdict changes: {'; '.join(P21['B']['group_changes'])}",
     'groups of 15', p21, '적응 신호를 모의 전압 고정(응답 모델에 요청을 걸어 전압이 한계에 닿음)으로 바꾼 결과, 벤치마크 지표로 채점(갱신식·상수·순서·시작 배율은 그대로)',
     'adaptation driven by the simulated voltage clamp of the response model instead of the benchmark flag, scored with the benchmark metric (update rule, constants, order and starting scales unchanged)',
     'posthoc_v321/summary.json', 'B.SIG.benchmark, B.AD.benchmark, B.group_changes')
sma, smb = P21['B']['AD']['simulated'], P21['B']['SIG']['simulated']
fact('SG02', f"main configuration {sma['groups_sim_pass']} ({sma['clamp_rows']} clamped rows, {100 * sma['clamp_rate']:.2f} %; below: {'; '.join(sma['failed_sim'])}); "
             f"simulated-clamp signal {smb['groups_sim_pass']} ({smb['clamp_rows']} clamped rows, {100 * smb['clamp_rate']:.2f} %; below: {'; '.join(smb['failed_sim'])})",
     'groups of 15', p21, '모의 고정 지표(전달 사용률 중앙 ≥ 80 %, 모의 고정 비율 ≤ 10 %; 모든 행을 그 요청으로 모의)로 센 합격 그룹',
     'groups passing under the simulated-clamp metric (median delivered utilisation >= 80 % and simulated clamp rate <= 10 %; every row simulated at its request)', 'posthoc_v321/summary.json', 'B.AD.simulated, B.SIG.simulated')
TJ0 = pd.read_csv(TOP / 'posthoc_lsad_v320/ad_trajectory.csv')[['row_id', 'order']]; SRo = SRw.merge(TJ0, on='row_id'); parts3 = []
for c in ('A1', 'A17', 'B'):
    x = SRo[SRo.evaluation_group == c].sort_values('order'); dv = x[np.abs(x.SIG_scale - x.AD_scale) > 1e-15]
    if len(dv):
        y = x[x.order >= dv.order.min()]; dd = (y.SIG_bench_event.astype(int) - y.AD_bench_event.astype(int)).groupby([y.direction, y.temp_C]).sum()
        parts3.append(f"{c}: {int(dd.sum())} more (" + ', '.join(f'{k[0]} {k[1]} C {v:+d}' for k, v in dd.items() if v) + ')')
    else:
        parts3.append(f'{c}: identical scales')
fact('SG03', 'from the first charge request above the reference SOP that did not reach the simulated clamp, the simulated-clamp signal kept the shared scale higher; requests above the reference SOP from then on: ' + '; '.join(parts3),
     'rows', p21 + '; descriptive re-count', '모의 고정 신호에서 배율이 갈라진 뒤 참조 초과가 늘어난 곳(셀, 방향, 온도)', 'where the requests above the reference SOP increased after the two signals diverged (cell, direction, temperature)',
     'posthoc_v321/sigclamp_rows.csv; posthoc_lsad_v320/ad_trajectory.csv', 'AD_scale, SIG_scale, AD_bench_event, SIG_bench_event; order')
TF = pd.read_csv(TOP / 'posthoc_v321/tradeoff_fixed.csv'); TA = pd.read_csv(TOP / 'posthoc_v321/tradeoff_ad.csv'); IS = pd.read_csv(TOP / 'posthoc_v321/iso_comparison.csv')
pd.concat([TF.assign(policy='fixed scale'), TA.assign(policy='online adaptation (tau* = tau)')], ignore_index=True).to_csv(DATA / 'iso_tradeoff_posthoc.csv', index=False)
IS.to_csv(DATA / 'iso_comparison_posthoc.csv', index=False)
TG_ = TF.tau.to_numpy()


def tau_runs(taus):
    """contiguous runs of tau grid values, in %"""
    ix = sorted(int(np.argmin(np.abs(TG_ - t))) for t in taus); out, s = [], ix[0]
    for a, b in zip(ix, ix[1:] + [None]):
        if b is None or b != a + 1:
            out.append(f'{100 * TG_[s]:.2f}' if s == a else f'{100 * TG_[s]:.2f}-{100 * TG_[a]:.2f}'); s = b
    return ', '.join(out)


def rng(s, k=100, f='.2f'):
    return f'{k * s.min():{f}}' if s.min() == s.max() else f'{k * s.min():{f}}-{k * s.max():{f}}'


ie = 'post hoc (V321 C; fixed scale and adaptation over the calibration target 0-20 %; the fixed-scale points are compared with the evaluation data, which favours the fixed scale)'
f11 = TF[TF.closed_loop == TF.closed_loop.max()]; f5 = TF[np.isclose(TF.tau, 0.05)].iloc[0]
fact('IE01', f"tau 0-20 % ({len(TF)} values): realised intervention rate {rng(TF.int_rate)} %, groups {TF.closed_loop.min()}-{TF.closed_loop.max()}; the most groups ({TF.closed_loop.max()}) at tau {tau_runs(f11.tau)} % "
             f"(realised intervention {rng(f11.int_rate)} %, median utilisation {rng(f11.util_q50)} %, original criterion {rng(f11.original, 1, 'd')}); at tau 5 %: {int(f5.closed_loop)} groups, {100 * f5.int_rate:.2f} %, {100 * f5.util_q50:.2f} %",
     '', ie, '고정 배율 곡선(평가 셀 없는 보정의 τ를 0~20 %로 바꿈): 실제 개입률, 합격 그룹, 가장 많은 그룹이 나온 τ', 'fixed-scale curve over the calibration target 0-20 %: realised intervention rate, groups and the targets with the most groups',
     'posthoc_v321/tradeoff_fixed.csv', 'tau, int_rate, util_q50, closed_loop, original')
mp = IS[np.isclose(IS.tau, 0.05)].iloc[0]; Fm = TF[TF.int_rate <= mp.AD_int_rate + 1e-12]
bg = Fm[Fm.closed_loop == Fm.closed_loop.max()]; bu = Fm.loc[Fm.util_q50.idxmax()]
assert int(Fm.closed_loop.max()) == int(mp.fixed_best_closed_loop) and abs(Fm.util_q50.max() - mp.fixed_best_util_q50) < 1e-12 and int(Fm.original.max()) == int(mp.fixed_best_original)
fact('IE02', f"at a realised intervention rate no higher than that of the main configuration ({100 * mp.AD_int_rate:.2f} %), the fixed-scale points reach at most {int(Fm.closed_loop.max())} groups "
             f"(tau {tau_runs(bg.tau)} %: median utilisation {rng(bg.util_q50)} %), at most {100 * Fm.util_q50.max():.2f} % median utilisation (tau {100 * bu.tau:.2f} %: {int(bu.closed_loop)} groups, realised intervention {100 * bu.int_rate:.2f} %) "
             f"and at most {int(Fm.original.max())} groups under the original criterion; the main configuration: {int(mp.AD_closed_loop)} groups, {100 * mp.AD_util_q50:.2f} %, original criterion {int(mp.AD_original)}",
     '', ie, '주 구성과 같은 실제 개입률(이하)에서 고정 배율 점의 최대 합격 그룹·최대 사용률 중앙·엄격 기준 최대(평가 자료로 고른 점)와 주 구성의 비교',
     'at the realised intervention rate of the main configuration or lower: the most groups, the highest median utilisation and the most strict-criterion groups among the fixed-scale points (chosen with the evaluation data), against the main configuration',
     'posthoc_v321/iso_comparison.csv; posthoc_v321/tradeoff_fixed.csv', 'tau == 0.05; int_rate, util_q50, closed_loop, original')
a11 = TA[TA.closed_loop >= 11]; amx = TA[TA.closed_loop == TA.closed_loop.max()]; aor = TA[TA.original == TA.original.max()]; a20 = TA[np.isclose(TA.tau, 0.20)].iloc[0]
fact('IE03', f"tau = tau* 0-20 % (start at the tau scale, gamma {A4.GAMMA:g}): realised intervention rate {rng(TA.int_rate)} %; at least 11 groups at tau {tau_runs(a11.tau)} %; the most groups ({TA.closed_loop.max()}) at tau {tau_runs(amx.tau)} %; "
             f"original criterion at most {TA.original.max()} (tau {tau_runs(aor.tau)} %); at tau 20 %: {int(a20.closed_loop)} group{'s' if int(a20.closed_loop) != 1 else ''}",
     '', ie + '; values that could be chosen after seeing the results', '적응 곡선(τ* = τ, 시작 배율 = 그 τ의 고정 배율): 실제 개입률, 11그룹 이상인 τ, 가장 많은 그룹', 'adaptation curve (tau* = tau, starting scale = the fixed scale of that tau): realised intervention rate, targets with 11 or more groups, the most groups',
     'posthoc_v321/tradeoff_ad.csv', 'tau, int_rate, closed_loop, original')
hv = IS.fixed_points_at_or_below > 0; ua = IS.AD_util_q50 > IS.fixed_best_util_q50 + 1e-12
ga_, ge_, gl_ = IS.AD_closed_loop > IS.fixed_best_closed_loop, IS.AD_closed_loop == IS.fixed_best_closed_loop, IS.AD_closed_loop < IS.fixed_best_closed_loop
gap = 100 * (IS.AD_util_q50 - IS.fixed_best_util_q50)[hv]
fact('IE04', f"over the {len(IS)} tau values ({int(hv.sum())} with a fixed-scale point at the same or a lower realised intervention rate): adaptation median utilisation above the fixed-scale frontier at {int((ua & hv).sum())} "
             f"(difference {gap.min():+.2f} to {gap.max():+.2f} %p); groups above the frontier at {int((ga_ & hv).sum())} (tau {tau_runs(IS.tau[ga_ & hv])} %), equal at {int((ge_ & hv).sum())} (tau {tau_runs(IS.tau[ge_ & hv])} %), "
             f"below at {int((gl_ & hv).sum())} (tau {tau_runs(IS.tau[gl_ & hv])} %)",
     '', ie, '곡선 전체: 같은 실제 개입률(이하)의 고정 배율 경계와 비교한 적응의 사용률 중앙과 합격 그룹', 'whole curve: adaptation median utilisation and groups against the fixed-scale frontier at the same or a lower realised intervention rate',
     'posthoc_v321/iso_comparison.csv', 'AD_util_q50, fixed_best_util_q50, AD_closed_loop, fixed_best_closed_loop')
CB = pd.read_csv(TOP / 'posthoc_v321/counterfactual_b2s_groups.csv'); CB.to_csv(DATA / 'counterfactual_b2s_posthoc.csv', index=False)
Dd = P21['D']; cbk = CB.set_index(['policy', 'temp_C', 'horizon_s'])
u_ = lambda pol, t: 100 * cbk.loc[(pol, t, 10.0), 'util_q50']
fact('CF01', f"total {Dd['no_update_2s']['closed_loop']} groups (main {Dd['AD']['closed_loop']}), original criterion {Dd['no_update_2s']['original']} (main {Dd['AD']['original']}); verdict changes: {'; '.join(Dd['group_changes'])}; "
             f"B -20 C 10 s median utilisation {u_('AD_no_update_2s', -20):.1f} % (main {u_('AD', -20):.1f} %), below 80 %; B scale at its first 10-s query {Dd['B_scale_at_first_10s_row']['no_update']:.3f} (main {Dd['B_scale_at_first_10s_row']['AD']:.3f}); "
             f"B scale range {Dd['B_scale_no_update']['min']:.3f}-{Dd['B_scale_no_update']['max']:.3f} (main {Dd['B_scale_AD']['min']:.3f}-{Dd['B_scale_AD']['max']:.3f})",
     '', 'post hoc diagnostic (V321 D; not a design change)', '진단: B 25 °C 2초 세 행 뒤에 B 배율을 갱신하지 않았다고 둔 재생(세 행은 그대로 요청·채점)',
     'diagnostic: replay in which the three B 25 C 2 s rows do not update the B scale (the three rows are still requested and scored)', 'posthoc_v321/summary.json; posthoc_v321/counterfactual_b2s_groups.csv', 'D; policy, util_q50, passed')

# ---------------- cost ----------------
FC = J('posthoc_lsad_v320/feature_cpu.json')
fact('C02', f"{FC['ref_cpu_ms_per_row']:.1f}", 'ms CPU per row (feature extraction)', 'descriptive (V320 step 1; one measurement; process CPU time summed over three workers, reading and parsing the raw files from the zip archives included)',
     '특징 추출 계산 비용(감사한 추출기, 원자료 zip 읽기 포함, 한 번 잼; 쿼리당 지연 시간이 아님)', 'feature extraction cost of the audited extractor (reading the raw zip files included; one measurement; not a per-query latency)', 'posthoc_lsad_v320/feature_cpu.json', 'ref_cpu_ms_per_row')
T3 = P20['T_Ls']; shutil.copy(TOP / 'posthoc_lsad_v320/timing_Ls.csv', DATA / 'timing_local_predictors.csv')
fact('C03', f"{min(T3['cpu_ms_per_row']):.2f}-{max(T3['cpu_ms_per_row']):.2f} (3 runs); wall {min(T3['wall_s']):.2f}-{max(T3['wall_s']):.2f} s for 672 rows", 'ms CPU per row',
     'descriptive (V320; upper bound: fitting of all folds and single-cell models and temporary CSV input/output included)', '국소 예측기 단계(Ls, 맞춤 포함)의 계산 비용: 세 번 잰 범위(모든 맞춤과 임시 파일 입출력을 672행으로 나눈 값; 쿼리당 지연 시간이 아님)',
     'local-predictor stage cost (Ls) including all fitting, three runs (all fitting and temporary file input/output divided by 672 rows; not a per-query latency)', 'posthoc_lsad_v320/summary.json; posthoc_lsad_v320/timing_Ls.csv', 'T_Ls')

# ---------------- prior-method baseline (V265, V266) ----------------
C6 = J('chen_v266/summary.json')
fact('P01', ' / '.join(f'{x:g}' for x in C6['M_groups_tau_4_6']), 'groups of 15', 'preregistered (V266)', '선행 방법(Chen 2026) 재현, 판정 팔(M): τ 4~6 % 합격 그룹', 'prior method (Chen 2026, our reproduction), judged arm M, tau 4-6 %', 'chen_v266/summary.json', 'M_groups_tau_4_6')
fact('P02', f"{100 * C6['M_tau5']['int_rate']:.1f} / {100 * C6['M_tau5']['util_q50']:.1f}", '%', 'preregistered (V266)', '선행 방법 재현 τ 5 % 개입률 / 사용률 중앙', 'prior method reproduction: intervention / median utilisation at tau 5 %', 'chen_v266/summary.json', 'M_tau5')
fact('P03', f"{C6['M_open_loop']['rel_bias_median_pct']:.0f}", '% median bias', 'preregistered (V266)', '선행 방법 재현 중심 예측 치우침 중앙', 'prior method reproduction: median bias of the central prediction', 'chen_v266/summary.json', 'M_open_loop.rel_bias_median_pct')
if 'R2' in C6:
    fact('P04', f"{C6['R2']['open_loop']['rel_bias_median_pct']:.0f}", '% median bias', 'preregistered (V266)', '논문 설정(R2) 재현의 B 10초 49행 치우침(재현 관문 미달)', 'paper-setting reproduction (R2) bias on 49 B 10-s rows (reproduction gate not met)', 'chen_v266/summary.json', 'R2.open_loop.rel_bias_median_pct')
fact('P05', 1.87, 'W', 'descriptive', '선행 논문이 보고한 10초 SOP RMSE(원문 p9·p10 표 7)', 'prior paper reported 10-s SOP RMSE (paper p9, p10 Table 7)', 'references/SOURCES.md of the evidence package (L1)', 'Chen 2026 p10 Table 7')
RM = pd.read_csv(TOP / 'chen_v266/rows_tau5_M.csv'); assert len(RM) == 672
opM, exM = C1.open_loop(RM, 'pred'); smM, gM = C1.closed_loop(RM, RM.pred.to_numpy())
fact('P06', f"{opM} ({exM} exceedance rows); closed-loop re-count {int(((gM.util_q50 >= 0.80) & (gM.int_rate <= 0.10)).sum())}", 'groups of 15', 'post hoc (re-count of the registered tau = 5 % requests of the judged arm M)',
     '선행 방법 재현(판정 팔 M)의 τ 5 % 요청을 원래 기준으로 센 합격 그룹(초과 행 수), 같은 요청의 폐루프 기준 재집계', 'prior-method reproduction (judged arm M): tau = 5 % requests under the original criterion (exceedance rows), and the closed-loop re-count of the same requests',
     'chen_v266/rows_tau5_M.csv', 'closedloop_v251.open_loop, closed_loop on pred')
K5 = J('presubmit_v281/summary.json')['checks']['K5']['A4_prior_method_M']; z, f5 = K5['zero_exceedance'], K5['int_rate_le_5pct']
zs = f"zero exceedance: s {z['s']:.2f}" if z else 'zero exceedance: no s in 0.20-1.50'
fs = f"intervention <= 5 %: s {f5['s']:.2f}" if f5 else 'intervention <= 5 %: no s in 0.20-1.50'
fact('P07', f"{zs}; {fs}; most groups over s: {K5['max_closed']} (closed loop) / {K5['max_original']} (original)", '', 'post hoc (V281 K5; oracle common scale chosen on the evaluation data, descriptive)',
     '선행 방법 재현: 평가 셀 중심 예측에 공통 배율을 곱한 오라클(평가 자료로 배율을 보므로 성능 주장 아님)', 'prior-method reproduction: oracle common scale on its held-out centres (not a performance claim)', 'presubmit_v281/summary.json', 'checks.K5.A4_prior_method_M')

# ---------------- strict (original) criterion: definition ----------------
fact('H01', 'no exceedance and median predicted/true >= 80 % in every group', '', 'definition (record of the study\'s original pass criterion)',
     '원래 합격 기준(엄격 기준): 그룹마다 초과 0건, 예측/정답 중앙 80 % 이상', 'original (strict) pass criterion: no exceedance and median predicted/true >= 80 % in every group', 'PROTOCOL_RPCWBY_AGING_EVAL_V85.md', 'lines 36-37')

# ---------------- leakage audit (the same five local predictors, pre-audit implementation) ----------------
for k, ko, en in (('R1', 'SOC 설정값 구간, 평가 셀이 들어간 보정(기준)', 'SOC-set-point regions, calibration involving the evaluated cell (reference)'),
                  ('S1', '휴지 전압 구간만 바꿈', 'rest-voltage regions only'),
                  ('S2', '평가 셀 없는 보정만 바꿈', 'evaluated-cell-free calibration only'),
                  ('P', '둘 다 바꿈(혼합 규칙·클래스는 유지)', 'both changed (hybrid rule and classes kept)')):
    fact(f'A02_{k}', trip(S2['configs'][k]['groups_tau_4_6']), 'groups of 15', 'preregistered (V272; pre-audit implementation, fixed scale)', f'누설 분해(같은 국소 예측기, 감사 전 구현): {ko}', f'leakage decomposition (the same local predictors, pre-audit implementation): {en}', 'leakfree_v272/summary.json', f'configs.{k}.groups_tau_4_6')
S8a = J('strict_v268/summary.json')
fact('A03', trip(S8a['light_strict']['groups_tau_4_6']), 'groups of 15', 'preregistered (V268)',
     '방전 보정만 셋째 셀 단독 모델로 바꾼 이전 분석(같은 국소 예측기)', 'earlier analysis: discharge calibration by third-cell-only models only (the same local predictors)', 'strict_v268/summary.json', 'light_strict')
import closedloop_v255 as V5  # noqa: E402
Bl = V5.load()[['row_id', 'evaluation_group', 'direction', 'in_events', 'truth']]
evr = Bl[Bl.in_events & (Bl.direction == 'discharge')]
W2 = pd.read_csv(TOP / 'leakfree_v272/strict_vpre_main.csv').pivot_table(index=['train_cell', 'row_id'], columns='name', values='pred_power_W')
with np.errstate(all='ignore'):
    L2 = np.log(W2.to_numpy(float)); L2[~np.isfinite(L2)] = np.nan
    W2 = W2.assign(m=np.nanmedian(L2, axis=1)).reset_index()
outv = []
for e, dd in (('B', 'A17'), ('B', 'A1')):
    mm = W2[W2.train_cell == e].set_index('row_id').m
    rr = np.exp(mm.reindex(evr[evr.evaluation_group == dd].row_id)).to_numpy() / evr[evr.evaluation_group == dd].truth.to_numpy(); rr = rr[np.isfinite(rr)]
    outv.append(f'{e}->{dd}: {np.median(rr):.3f} / {np.quantile(rr, .95):.3f} (n {len(rr)})')
fact('A04', '; '.join(outv), 'pred/truth median / q95', 'descriptive (V272 saved predictions; pre-audit implementation)', '셀 하나로 맞춘 국소 다섯 개의 다른 셀 방전 행 예측/정답: 노화 셀 B 단독 → 새 셀',
     "local five fitted on one cell: pred/truth on the other cells' discharge rows, aged cell B alone -> fresh cells", 'leakfree_v272/strict_vpre_main.csv; closedloop_v251/src/bench_rows.csv', 'log median of the five, in-event discharge rows')
audit = []
for k in ('R1', 'S1', 'S2', 'P'):
    g = S2['configs'][k]
    audit.append(dict(config=f'V272 {k}', calibration='own-fold models' if k in ('R1', 'S1') else 'evaluated-cell-free', regions='SOC set point' if k in ('R1', 'S2') else 'rest voltage', scale='fixed',
                      groups_tau5=[p['groups_pass'] for p in g['per_tau'] if abs(p['tau'] - 0.05) < 1e-9][0], groups_tau_4_6=trip(g['groups_tau_4_6']),
                      status='information-flow leakage present' if k != 'P' else 'information-flow leakage removed; design leakage', source='leakfree_v272/summary.json'))
audit.append(dict(config='V278 L (pre-audit implementation)', calibration='evaluated-cell-free', regions='rest voltage', scale='fixed', groups_tau5=L8['tau5']['B0']['closed_loop'], groups_tau_4_6=trip(L8['tau_4_6']['B0']),
                  status='information-flow leakage removed; design leakage; hybrid rule and per-direction classes removed', source='compare_v278/L/run.json'))
audit.append(dict(config='V287 Ls (audited)', calibration='evaluated-cell-free', regions='rest voltage', scale='fixed', groups_tau5=b0['closed_loop'], groups_tau_4_6=trip(S87['tau_4_6']['Ls B0']),
                  status='as V278 L; specification audit applied', source='specaudit_v287/summary.json'))
audit.append(dict(config='V287 Ls + V284 AD (main configuration)', calibration='evaluated-cell-free start, online adaptation', regions='rest voltage', scale='adapted', groups_tau5=ad['closed_loop'], groups_tau_4_6=trip(S87['tau_4_6']['Ls AD']),
                  status='as V287 Ls; the adaptation reads the intervention flags of earlier executed queries of the evaluated cell', source='specaudit_v287/summary.json'))
pd.DataFrame(audit).to_csv(DATA / 'leakage_audit.csv', index=False)
inv = [('D1: coefficient region chosen with the test schedule SOC set point (soc_point)', 'information flow', 'removed: rest-voltage boundaries learned on the training cells'),
       ('D2: calibration rows predicted by models fitted with the evaluated cell', 'information flow', 'removed: discharge by third-cell-only models, charge by the fold-c model in sample'),
       ('short-rest predictor M3 and hybrid rule / per-direction classes (oracle exploration)', 'design', 'removed: one predictor set, one scale per evaluated cell'),
       ('local predictor specifications and region SOC boundaries (0.1, 0.3, 0.7); ensemble composition', 'design', 'not removable'),
       ('online adaptation rule and its constants (gamma, tau*, one scale per cell)', 'design', 'not removable: grid sensitivity reported'),
       ('evaluation criterion (utilisation-intervention group pass) chosen while studying these three cells', 'design (evaluation)', 'not removable: original (strict) criterion results reported alongside'),
       ('tau range 4-6 % (reporting range chosen after earlier sensitivity results)', 'design (reporting)', 'main result reported at tau = 5 %; the 4-6 % range reported alongside'),
       ('predictor variant: the specification-audited predictors (Ls) are the main predictor; the pre-audit implementation is reported as sensitivity', 'design (reporting)', 'disclosed: both variants reported'),
       ('inputs: reference SOP pulses of earlier SOC points of the same sweep (previous-pulse features)', 'allowed input by the study rules', 'disclosed: the task is sequential prediction conditioned on this history; the current query\'s response, same-search iterations and later data are not used'),
       ('inputs: intervention flags of earlier executed queries of the evaluated cell (AD); in the offline benchmark computed as request > reference SOP of those queries', 'allowed input by the study rules (a proxy for the voltage-clamp event that a controller observes in operation; the two differ for charge requests near full charge in the response-model simulation)', 'disclosed: the flag of the current and later queries is not used (causality check); sensitivity to a simulated-clamp signal reported (post hoc)')]
pd.DataFrame(inv, columns=['path', 'kind', 'handling']).to_csv(DATA / 'leak_inventory.csv', index=False)

# ---------------- internal only (not in the paper body) ----------------
fact('X01', f"{P20['X01_Ls']['central_abs_rel_err_median_pct']:.2f}", '% |pred/truth-1| median', 'internal only (open loop; V320)', '중심 예측 상대 오차 절댓값 중앙(개방 루프)', 'central prediction median relative error (open loop; not for the manuscript)',
     'posthoc_lsad_v320/summary.json', 'X01_Ls')

F = pd.DataFrame(facts); assert F.id.is_unique
F.to_csv(PKG / 'facts.csv', index=False)
print(f'facts: {len(F)}; data files: {sorted(p.name for p in DATA.glob("*.csv"))}')
