"""Ls + AD manuscript package, revision r3 (2026-10-27): the facts and tables added to r2, and the r3 manifest.
The 96 r2 facts are carried byte for byte from the r2 ZIP (checked by its SHA-256); this script appends the r3 facts,
computed from their source files, and writes the three r3 data tables. Source paths are relative to the parent folder
(experiments/unseen_cell_sop_20260914/), as in facts.csv.
usage: OMP_NUM_THREADS=1 python build_package_r3_additions.py [--manifest]   (torch_env)
       without --manifest: facts.csv and data/{mcu_cost,selection_bias_posthoc,ad_identity_posthoc}.csv
       --manifest: MANIFEST.sha256 (package files, the sources of the r3 facts, the r2 ZIP)"""
import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent
HERE = PKG.parent / 'experiments' / 'unseen_cell_sop_20260914'
R2_ZIP = 'paper_package_lsad_20261013_r2.zip'
R2_SHA = 'f4b15e455c2f9a127eb1765ea37f43d1ca14f9902a6bb8fd663963f5a41733a8'
COLS = ['id', 'value', 'unit', 'status', 'description_ko', 'description_en', 'source_file', 'source_field']
MCU = 'mcu_port_v322/'
GAMMA, TAU_STAR = 0.05, 0.05                    # DEF13
S_LO, S_HI = 0.20, 1.50                         # scale clip of the adaptation (DEF13)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def r2_facts_bytes():
    z = HERE / R2_ZIP
    assert sha(z) == R2_SHA, 'r2 ZIP changed'
    with zipfile.ZipFile(z) as f:
        b = f.read('paper_package_lsad_20261013/facts.csv')
    rows = list(csv.reader(io.StringIO(b.decode('utf-8'))))
    assert rows[0] == COLS and len(rows) == 97, (rows[0], len(rows))
    assert b.endswith(b'\n')
    return b, [r[0] for r in rows[1:]]


def f1(x):
    return f'{x:.1f}'


def mcu_facts():
    S = json.load(open(HERE / MCU / 'v322b/full/summary_v322b.json'))
    run = json.load(open(HERE / MCU / 'v322b/full/board_b_run.json'))
    cmpj = json.load(open(HERE / MCU / 'v322b/full/board_b_compare.json'))
    T = pd.read_csv(HERE / MCU / 'v322b/full/board_b_timing_queries.csv')
    V3 = json.load(open(HERE / MCU / 'full/summary_v322.json'))
    hz = run['info_start']['core_hz']
    assert hz == 250_000_000 and len(T) == 672 and run['queries'] == 672 and run['samples'] == 1902037
    assert all(S['gates'][g] for g in ('G1', 'G2', 'G3', 'G4'))
    q = S['per_query_us']
    ad = (T.cyc_ad_req + T.cyc_ad_event) / hz * 1e6
    m = S['memory_bytes']
    rb_bytes = 1024 * (4 + 4)                                                      # RB_N x (int32 time + float32 voltage)
    assert 'RB_N 1024' in (HERE / MCU / 'c/sop_v322b.h').read_text()
    dc = (HERE / MCU / 'fw_v322b/Build/double_check.txt').read_text().splitlines()
    nones = [dc[i + 1].strip() for i, ln in enumerate(dc) if ln.endswith('soft-double calls:') or ln.endswith('soft-double helpers present:')]
    assert nones == ['none', 'none', 'none'], nones
    dt_med = V3['stream']['dt_s']['median']
    d2 = V3['double']['parity_python']
    status_d = 'descriptive (V322b registered board measurement; recomputation of the main configuration on a microcontroller)'
    F = []
    F.append(['C04', f"mean {f1(q['total']['mean'])}, median {f1(q['total']['median'])}, 99th percentile {f1(q['total']['p99'])}, maximum {f1(q['total']['max'])} (672 queries)",
              'µs per query (MCU computation)', status_d,
              'MCU에서 쿼리 하나의 계산 시간(특징 + 국소 예측기 다섯 개와 로그 중앙값 + 배율 적응의 요청과 갱신; 자료 전송 제외)',
              'per-query computation time on the microcontroller (features, five local predictors and log median, scale adaptation request and update; data transfer excluded)',
              MCU + 'v322b/full/summary_v322b.json', 'per_query_us.total'])
    F.append(['C05', f"features {f1(q['cyc_feat']['mean'])} (maximum {f1(q['cyc_feat']['max'])}); five local predictors and log median {f1(q['cyc_pred']['mean'])} "
              f"(maximum {f1(q['cyc_pred']['max'])}); scale adaptation {ad.mean():.2f} (maximum {ad.max():.2f})",
              'µs per query, mean (maximum)', status_d, 'MCU 쿼리당 계산 시간의 단계별 평균(최대)', 'per-query computation time on the microcontroller by stage, mean (maximum)',
              MCU + 'v322b/full/summary_v322b.json; ' + MCU + 'v322b/full/board_b_timing_queries.csv', 'per_query_us.cyc_feat, cyc_pred; cyc_ad_req + cyc_ad_event'])
    ps = S['per_sample_us']
    F.append(['C06', f"mean {ps['mean']:.2f}, maximum {ps['max']:.2f}, median at most {ps['p50_upper']:.2f}, 99th percentile at most {ps['p99_upper']:.2f} "
              f"over {run['samples']:,} samples; median sample interval {dt_med:.1f} s",
              'µs per sample (MCU)', status_d, 'MCU에서 원시 표본 하나를 받아 특징 상태를 갱신하는 시간(분위수는 64사이클 구간의 위 경계)',
              'time to update the feature state with one raw sample on the microcontroller (percentiles are upper edges of 64-cycle bins)',
              MCU + 'v322b/full/summary_v322b.json; ' + MCU + 'full/summary_v322.json', 'per_sample_us; stream.dt_s.median'])
    F.append(['C07', f"one cell: flash {m['one_cell_flash']:,} B (code {m['sop_code']:,}, math library {m['libm']:,}, compiler runtime {m['libgcc']:,}, "
              f"coefficients of one fold {m['coef_1fold']:,}), RAM {m['one_cell_ram']:,} B (state {m['sop_state']:,}, of which the rest-voltage buffer {rb_bytes:,}; "
              f"stack high-water {m['stack_highwater']:,}); whole test firmware: flash {m['flash_total']:,} B, static RAM {m['ram_static']:,} B",
              'bytes', status_d, 'MCU 메모리: 셀 하나에 배치할 때의 SOP 몫과 측정용 펌웨어 전체(통신·측정 장치·HAL 포함)',
              'microcontroller memory: the share of the SOP computation for one cell, and the whole test firmware (communication, measurement and HAL included)',
              MCU + 'v322b/full/summary_v322b.json; ' + MCU + 'c/sop_v322b.h', 'memory_bytes; RB_N'])
    F.append(['C08', f"intervention flags equal in {672 - cmpj['event_mismatch']} of 672 queries; group verdicts equal "
              f"({cmpj['groups_closed_loop']} groups under the utilisation-intervention criterion, {cmpj['groups_strict']} under the strict criterion); "
              f"region mismatches {cmpj['region_mismatch']}; maximum relative difference of the central prediction {cmpj['centre']['max_rel']:.1e} "
              f"and of the request {cmpj['request']['max_rel']:.1e}",
              '', 'post hoc check (V322b registered gates G1-G3; MCU single-precision computation against the Python implementation)',
              'MCU 계산과 파이썬 구현의 일치(개입 여부, 그룹 판정, 구간, 중심 예측·요청의 최대 상대 차이)',
              'agreement of the microcontroller computation with the Python implementation (intervention flags, group verdicts, regions, central predictions and requests)',
              MCU + 'v322b/full/board_b_compare.json', 'event_mismatch, groups_closed_loop, groups_strict, region_mismatch, centre.max_rel, request.max_rel'])
    F.append(['C09', 'single precision only: no software double-precision routine in the SOP code, in the test harness or anywhere in the linked image; '
              'time kept as integer milliseconds and charge as an integer accumulator',
              '', 'descriptive (V322b registered gate G4: symbol listing of the built objects and image)',
              'MCU 계산의 수치 형식(배정밀도 없음, 시각·전하량은 정수)', 'number format of the microcontroller computation (no double precision; integer time and charge)',
              MCU + 'fw_v322b/Build/double_check.txt; ' + MCU + 'c/sop_v322b.h', 'soft-double listings; header comment'])
    F.append(['C10', f"NUCLEO-H563ZI board (STM32H563ZI, Arm Cortex-M33 at {hz / 1e6:.0f} MHz, single-precision FPU, instruction cache 1-way); "
              'arm-none-eabi-gcc 14.3, -O2, -ffp-contract=off, newlib-nano; cycle counter (DWT) read with interrupts disabled around each measured call; '
              'recorded samples streamed from a PC over UART (921,600 baud); no battery, sensor or current control',
              '', 'definition (measurement setup of the V322b protocol)', 'MCU 측정 조건', 'microcontroller measurement setup',
              'PROTOCOL_MCU_V322B.md; ' + MCU + 'v322b/full/board_b_run.json', 'build and board; info_start.core_hz'])
    F.append(['C11', f"maximum relative difference {d2['centre_max_rel']:.1e} (central prediction) and {d2['request_max_rel']:.1e} (request); "
              f"intervention mismatches {d2['event_mismatch']}",
              '', 'post hoc check (V322 double-precision build of the same C code on the same board)',
              '같은 C 코드의 배정밀도 빌드와 파이썬 구현의 일치(C 이식이 같은 계산임을 확인)',
              'agreement of a double-precision build of the same C code with the Python implementation (the port computes the same quantities)',
              MCU + 'full/summary_v322.json', 'double.parity_python'])
    rows = [dict(section='per query (us)', item=k, mean=q[c]['mean'], median=q[c]['median'], p99=q[c]['p99'], max=q[c]['max'])
            for k, c in (('total', 'total'), ('features', 'cyc_feat'), ('five local predictors and log median', 'cyc_pred'))]
    rows.append(dict(section='per query (us)', item='scale adaptation (request and update)', mean=float(ad.mean()), median=float(ad.median()),
                     p99=float(ad.quantile(0.99)), max=float(ad.max())))
    rows.append(dict(section='per sample (us)', item='feature-state update', mean=ps['mean'], median=ps['p50_upper'], p99=ps['p99_upper'], max=ps['max']))
    mem = [('flash, one cell', m['one_cell_flash']), ('  code', m['sop_code']), ('  math library', m['libm']), ('  compiler runtime', m['libgcc']),
           ('  coefficients of one fold', m['coef_1fold']), ('RAM, one cell', m['one_cell_ram']), ('  state', m['sop_state']), ('    rest-voltage buffer', rb_bytes),
           ('  stack high-water', m['stack_highwater']), ('whole test firmware, flash', m['flash_total']), ('whole test firmware, static RAM', m['ram_static'])]
    rows += [dict(section='memory (bytes)', item=k, value=v) for k, v in mem]
    tab = pd.DataFrame(rows, columns=['section', 'item', 'mean', 'median', 'p99', 'max', 'value'])
    return F, tab


def selbias_facts():
    J = json.load(open(HERE / 'v324/selbias_summary.json'))
    mc, ms = J['main_closed'], J['main_strict']
    st = 'post hoc (V324 analysis 2; resampling of the evaluated rows: per group, sweeps with replacement, then rows within each drawn sweep; adaptation rerun; 1,000 resamples; training cells fixed)'
    F = []
    F.append(['SB01', f"utilisation-intervention criterion: mean {mc['mean']:.2f}, SD {mc['sd']:.2f}, 95 % range {mc['q025']:.0f}-{mc['q975']:.0f}; "
              f"strict criterion: mean {ms['mean']:.2f}, SD {ms['sd']:.2f}, 95 % range {ms['q025']:.0f}-{ms['q975']:.0f}",
              'groups of 15', st, '주 구성의 합격 그룹 수를 평가 행 재표집으로 다시 센 분포(평가 행의 변동만 담음; 학습 셀과 셀 자체의 변동은 담지 않음)',
              'distribution of the passing count of the main configuration under resampling of the evaluated rows (only the variation of the evaluated rows; not that of the training cells or of the cells themselves)',
              'v324/selbias_summary.json', 'main_closed, main_strict'])
    a, b = J['closed']['ls_request_rules'], J['strict']['ls_request_rules']
    assert a['variants'] == 48 and b['variants'] == 48
    F.append(['SB02', f"48 request rules of the main predictor (fixed scale or adaptation; starting tau 4/5/6 %, gamma 0.01-0.2, tau* 0.025-0.1): "
              f"utilisation-intervention criterion: best observed {a['observed_max']}, optimism {a['optimism']:.2f} (95 % {a['optimism_q025']:.2f} to {a['optimism_q975']:.2f}), "
              f"corrected {a['corrected_max']:.1f}; strict criterion: best observed {b['observed_max']}, optimism {b['optimism']:.2f} "
              f"({b['optimism_q025']:.2f} to {b['optimism_q975']:.2f}), corrected {b['corrected_max']:.1f}",
              'groups of 15', st + '; bootstrap optimism of choosing the best variant',
              '주 예측기의 요청 방식 48개 가운데 결과로 최고를 고를 때의 낙관(부트스트랩)',
              'optimism of choosing the best of the 48 request rules of the main predictor after seeing the results (bootstrap)',
              'v324/selbias_summary.json', 'closed.ls_request_rules, strict.ls_request_rules'])
    c, d = J['closed']['all'], J['strict']['all']
    assert c['variants'] == 768 and d['variants'] == 768
    F.append(['SB03', f"768 variants (16 predictor variants with the four audit fixes switched on or off x 48 request rules): strict criterion: best observed {d['observed_max']}, "
              f"optimism {d['optimism']:.2f} (95 % {d['optimism_q025']:.2f} to {d['optimism_q975']:.2f}), corrected {d['corrected_max']:.1f}; "
              f"utilisation-intervention criterion: optimism {c['optimism']:.2f} (95 % {c['optimism_q025']:.2f} to {c['optimism_q975']:.2f})",
              'groups of 15', st + '; bootstrap optimism of choosing the best variant; the best observed variants include single-fix predictors that are not results (PACKAGE 6.11)',
              '예측기 변형과 요청 방식을 모두 합친 768개에서 결과로 최고를 고를 때의 낙관(부트스트랩)',
              'optimism of choosing the best of all 768 predictor-and-request variants after seeing the results (bootstrap)',
              'v324/selbias_summary.json', 'strict.all, closed.all'])
    rows = [dict(criterion='utilisation-intervention', family='main configuration (resampling)', variants=1, observed=11, mean=mc['mean'], sd=mc['sd'], q025=mc['q025'], q975=mc['q975']),
            dict(criterion='strict', family='main configuration (resampling)', variants=1, observed=4, mean=ms['mean'], sd=ms['sd'], q025=ms['q025'], q975=ms['q975'])]
    for crit, key in (('utilisation-intervention', 'closed'), ('strict', 'strict')):
        for fam, fk in (('48 request rules of the main predictor', 'ls_request_rules'), ('768 variants', 'all')):
            x = J[key][fk]
            rows.append(dict(criterion=crit, family=fam, variants=x['variants'], observed=x['observed_max'], optimism=x['optimism'],
                             optimism_q025=x['optimism_q025'], optimism_q975=x['optimism_q975'], corrected=x['corrected_max']))
    return F, pd.DataFrame(rows, columns=['criterion', 'family', 'variants', 'observed', 'mean', 'sd', 'q025', 'q975', 'optimism', 'optimism_q025', 'optimism_q975', 'corrected'])


def identity_fact():
    A = pd.read_csv(PKG / 'data/ad_trajectory.csv')
    rows, parts = [], []
    for cell in ('A1', 'A17', 'B'):
        g = A[A.cell == cell].sort_values('order')
        e = g.intervened.astype(bool).to_numpy(); s = g.scale.to_numpy(float); T = len(g)
        ls = np.log(s)
        # the scale after every update, from the rule itself; check it reproduces the recorded trajectory, then take the last one
        nxt = np.clip(ls + GAMMA * (TAU_STAR - e), np.log(S_LO), np.log(S_HI))
        assert np.allclose(nxt[:-1], ls[1:], rtol=0, atol=1e-8), cell
        clipped = bool(((ls + GAMMA * (TAU_STAR - e)) != nxt).any())
        realised = e.sum() / T; ident = TAU_STAR - (nxt[-1] - ls[0]) / (GAMMA * T)
        rows.append(dict(cell=cell, queries=T, interventions=int(e.sum()), realised_rate=realised, scale_start=s[0], scale_end=float(np.exp(nxt[-1])),
                         identity_rate=ident, clipped=clipped))
        assert not clipped and abs(ident - realised) < 1e-9, (cell, ident, realised)
        parts.append(f'{cell}: {int(e.sum())} of {T} ({100 * realised:.2f} %), scale {s[0]:.3f} -> {np.exp(nxt[-1]):.3f}')
    F = [['AI01', '; '.join(parts) + '; the identity tau* - (ln s_end - ln s_start)/(gamma T) equals the realised rate in every cell (difference below 1e-9; no clipping)',
          '', 'post hoc (exact identity of the update rule when the scale is not clipped; recomputed from data/ad_trajectory.csv)',
          '셀마다 실제 개입률과 배율의 순 이동(갱신식의 항등식: 실제 개입률 = τ* − (ln s_끝 − ln s_시작)/(γT))',
          'realised intervention rate of each cell and the net movement of its scale (identity of the update rule: realised rate = tau* - (ln s_end - ln s_start)/(gamma T))',
          'paper_package_lsad_20261013_r3/data/ad_trajectory.csv', 'cell, order, scale, intervened']]
    return F, pd.DataFrame(rows)


def internal_facts():
    I = pd.read_csv(HERE / 'v324/info_1a.csv')
    a = I[(I.subset == 'all') & (I.protocol == 'loco')].set_index(['direction', 'info', 'model']).rmse
    n = I[(I.subset == 'all') & (I.protocol == 'loco') & (I.model == 'ridge') & (I['info'] == 'I1')].groupby('direction').n.first()
    assert int(n['charge']) + int(n['discharge']) == 658
    name = dict(ridge='ridge', hgb='gradient boosting', knn='kNN', mlp='MLP')

    def blk(d, i, ls=False):
        s = ', '.join(f"{name[m]} {a[(d, i, m)]:.3f}" for m in ('ridge', 'hgb', 'knn', 'mlp'))
        return s + (f", local predictors (central prediction) {a[(d, 'I2 (Ls)', 'Ls_central')]:.3f}" if ls else '')
    v = '; '.join(f"{d}, with low-current relaxation: {blk(d, 'I1')}; with the previous pulses as well: {blk(d, 'I2', True)}" for d in ('charge', 'discharge'))
    F = [['X02', v, 'RMSE of log effective resistance (leave-one-cell-out; 658 rows: 10-s, not current-limited)',
          'internal only (open loop; V324 analysis 1, model settings fixed in its protocol)',
          '유연한 모델과 단순 모델의 새 셀 오차(개방 루프 저항; 원고 본문에 쓰지 않음)',
          'new-cell error of flexible and simple models (open-loop resistance; not for the manuscript)',
          'v324/info_1a.csv', "subset == all, protocol == loco; info, model, rmse"]]
    r = json.load(open(HERE / 'v326/rpc.json'))
    h1, h5 = r['hyb1_delta'], r['hyb0.5_delta']
    F.append(['X03', f"discharge {r['rows']} rows: local predictors {r['ls_rmse']:.4f}; residual-corrected hybrid (lambda 1) {r['hyb1_rmse']:.4f}, difference "
              f"+{h1['observed']:.3f} (95 % +{h1['q025']:.3f} to +{h1['q975']:.3f}); lambda 0.5 {r['hyb0.5_rmse']:.4f} (+{h5['observed']:.3f})",
              'RMSE of log effective resistance (leave-one-cell-out)', 'internal only (open loop; V326 registered design check, verdict: failed)',
              '구조 기본 예측 + 유연한 잔차 보정(하이브리드)과 국소 예측기의 새 셀 오차(개방 루프; 원고 본문에 쓰지 않음)',
              'new-cell error of a hybrid (structural base prediction with a flexible residual correction) against the local predictors (open loop; not for the manuscript)',
              'v326/rpc.json', 'rows, ls_rmse, hyb1_rmse, hyb1_delta, hyb0.5_rmse, hyb0.5_delta'])
    return F


def write_facts():
    b, ids = r2_facts_bytes()
    new = []
    for fn in (mcu_facts, selbias_facts, identity_fact):
        F, tab = fn()
        new += F
        name = {mcu_facts: 'mcu_cost.csv', selbias_facts: 'selection_bias_posthoc.csv', identity_fact: 'ad_identity_posthoc.csv'}[fn]
        tab.to_csv(PKG / 'data' / name, index=False, float_format='%.10g', lineterminator='\n')
    new += internal_facts()
    assert not set(r[0] for r in new) & set(ids) and len({r[0] for r in new}) == len(new)
    for r in new:
        assert len(r) == len(COLS) and all(isinstance(x, str) for x in r), r
    s = io.StringIO(); w = csv.writer(s, lineterminator='\n'); w.writerows(new)
    (PKG / 'facts.csv').write_bytes(b + s.getvalue().encode('utf-8'))
    print(f'facts: {len(ids)} carried + {len(new)} new = {len(ids) + len(new)}')


SOURCES_R3 = ['PROTOCOL_MCU_V322B.md', 'MCU_V322B.md', MCU + 'v322b_prereg.sha256', MCU + 'v322b_results.sha256', MCU + 'v322b/full/summary_v322b.json',
              MCU + 'v322b/full/board_b_run.json', MCU + 'v322b/full/board_b_compare.json', MCU + 'v322b/full/board_b_timing_queries.csv',
              MCU + 'full/summary_v322.json', MCU + 'fw_v322b/Build/double_check.txt', MCU + 'c/sop_v322b.h', MCU + 'c/sop_v322b.c',
              'PROTOCOL_V324.md', 'V324.md', 'v324/AMENDMENT_1.md', 'v324/v324_prereg.sha256', 'v324/selbias_summary.json', 'v324/info_1a.csv',
              'PROTOCOL_V326.md', 'V326.md', 'v326/v326_prereg.sha256', 'v326/rpc.json', 'THEORY_SURVEY_20261024.md',
              'refcheck_r3_20261027/README.md', 'refcheck_r3_20261027/SHA256SUMS', 'refcheck_r3_20261027/gibbs_candes_2021_aci_neurips.pdf',
              'refcheck_r3_20261027/gibbs_candes_2021_neurips_bibtex.bib', 'refcheck_r3_20261027/etsi_ts_136213_v080800.pdf',
              'refcheck_r3_20261027/saxena_2010.08651v2.pdf', 'refcheck_r3_20261027/crossref_10_1109_wpmc_2002_1088198.json',
              'refcheck_r3_20261027/crossref_10_1109_twc_2021_3098972.json', 'NOVELTY_CHECK_20261024.md', R2_ZIP]


def write_manifest():
    files = sorted(p for p in PKG.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256' and '__pycache__' not in p.parts)
    L = ['# Ls + AD edition package manifest, revision r3 (2026-10-27): package files, then the sources of the facts added in r3 '
         '(paths relative to the parent folder), then the r2 ZIP whose MANIFEST.sha256 lists the sources of the 96 facts carried from r2']
    L += [f'{sha(p)}  {p.relative_to(PKG).as_posix()}' for p in files]
    L.append('# sources of the r3 facts and r3 notes (relative to the parent folder)')
    for s in SOURCES_R3:
        L.append(f'{sha(HERE / s)}  ../{s}')
    (PKG / 'MANIFEST.sha256').write_text('\n'.join(L) + '\n')
    print(f'manifest: {len(files)} package files, {len(SOURCES_R3)} sources')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', action='store_true'); a = ap.parse_args()
    write_manifest() if a.manifest else write_facts()
