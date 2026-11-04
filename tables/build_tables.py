"""Build Tables 1, 3, 4 and 5 of the manuscript (LaTeX and CSV) and check them.

Usage: python build_tables.py <repository root> <output dir>
Tables 1, 3 and 4 come from the package data (paper_package_lsad_20261013_r3/data). Table 5 comes from:
  five local predictors    comparisons/20261104_ecm_rls/results/gates.json (re-scored from ad_trajectory.csv)
  previous pulse baselines comparisons/20261014_baseline_pilot/results/primary.csv (PERSISTENCE, RINT_PREVIOUS)
  equivalent circuit       comparisons/20261104_ecm_rls/results/primary.csv (ECM-1RC-RLS-A1, ECM-2RC-RLS-A1)
  Chen et al. reproduction saved tau = 5 % requests of the judged arm (chen_v266/rows_tau5_M.csv), recounted here
                           against truth_power_W of bench_rows.csv and checked against package facts P01, P02, P06
                           and chen_v266/summary.json
Table 2 (predictor specification) is written by hand. Each table is written only after its checks pass.
"""
import csv
import json
import statistics
import sys
from pathlib import Path


def build_tables_1_3_4(DATA, OUT):
    (OUT / "csv").mkdir(parents=True, exist_ok=True)


    def read(name):
        with open(DATA / name, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


    def temp_tex(t):
        t = int(float(t))
        return f"$-${-t}" if t < 0 else str(t)


    def dur(h):
        return str(int(float(h)))


    def run_labels(values):
        """Merge each run of equal labels into one multirow cell on its first row."""
        out, i = [], 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[j + 1] == values[i]:
                j += 1
            n = j - i + 1
            out.append(f"\\multirow{{{n}}}{{*}}{{{values[i]}}}" if n > 1 else values[i])
            out.extend([""] * (n - 1))
            i = j + 1
        return out


    CHECKS = []


    def check(label, got, want):
        ok = got == want
        CHECKS.append((label, got, want, ok))
        if not ok:
            raise SystemExit(f"CHECK FAILED: {label}: got {got!r}, want {want!r}")


    # ---------------------------------------------------------------- Table 1
    groups = read("benchmark_groups.csv")
    check("T1 group count", len(groups), 15)
    CELL_HEAD = {
        "A1": "Cell A1, periodic characterisations during constant-current ageing",
        "A17": "Cell A17, periodic characterisations during US06 ageing",
        "B": "Cell B, aged cell characterised from $-$20 to 40 $^\\circ$C",
    }
    cell_rows = {}
    for g in groups:
        cell_rows.setdefault(g["evaluation_group"], []).append(g)
    sub = {c: sum(int(r["rows"]) for r in rs) for c, rs in cell_rows.items()}
    check("T1 rows per cell", sub, {"A1": 368, "A17": 252, "B": 52})
    check("T1 total rows", sum(sub.values()), 672)

    t1 = [
        "\\begin{table}[width=.78\\linewidth,cols=4,pos=t]",
        "\\caption{Cells, test conditions and row counts of the 15 evaluation groups.}\\label{tbl:groups}",
        "\\begin{tabular*}{\\tblwidth}{@{} LRRR @{}}",
        "\\toprule",
        "Direction & Temperature ($^\\circ$C) & Pulse duration (s) & Rows \\\\",
        "\\midrule",
    ]
    csv1 = [["cell", "direction", "temperature_C", "pulse_duration_s", "rows"]]
    for i, (cell, rs) in enumerate(cell_rows.items()):
        if i:
            t1.append("\\midrule")  # one rule between cells
        t1.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{CELL_HEAD[cell]} ({sub[cell]} rows)}}}} \\\\")
        for r, d_ in zip(rs, run_labels([r["direction"].capitalize() for r in rs])):
            t1.append(f"{d_} & {temp_tex(r['temp_C'])} & {dur(r['horizon_s'])} & {r['rows']} \\\\")
            csv1.append([cell, r["direction"], int(float(r["temp_C"])), dur(r["horizon_s"]), r["rows"]])
    t1 += ["\\midrule", f"Total & & & {sum(sub.values())} \\\\", "\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    (OUT / "T1_groups.tex").write_text("\n".join(t1), encoding="utf-8")
    with open(OUT / "csv" / "T1_groups.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv1)

    # ---------------------------------------------------------------- Table 3
    res = read("groups_tau5.csv")
    arms = {"fixed": [r for r in res if r["arm"] == "Ls B0"], "adapted": [r for r in res if r["arm"] == "Ls+AD"]}
    for k, v in arms.items():
        check(f"T3 {k} group count", len(v), 15)
    key = lambda r: (r["evaluation_group"], r["direction"], r["temp_C"], r["horizon_s"])
    check("T3 group order identical in both arms", [key(r) for r in arms["fixed"]], [key(r) for r in arms["adapted"]])
    check("T3 group order identical to Table 1", [key(r)[:3] + (r["horizon_s"],) for r in arms["fixed"]],
          [(g["evaluation_group"], g["direction"], g["temp_C"], g["horizon_s"]) for g in groups])

    iso = {r["policy"]: r for r in read("iso_tradeoff_posthoc.csv") if abs(float(r["tau"]) - 0.05) < 1e-12}
    tot = {"fixed": iso["fixed scale"], "adapted": iso["online adaptation (tau* = tau)"]}
    for k in arms:
        exc = sum(int(r["exceed"]) for r in arms[k])
        check(f"T3 {k} exceedances sum = overall", exc, int(tot[k]["exceed_rows"]))
        check(f"T3 {k} U&E passes", sum(r["passed"] == "True" for r in arms[k]), int(tot[k]["closed_loop"]))
        check(f"T3 {k} strict passes", sum(r["pass_original"] == "True" for r in arms[k]), int(tot[k]["original"]))
        check(f"T3 {k} rows", sum(int(r["n"]) for r in arms[k]), 672)
    check("T3 overall values (text)", {k: (int(tot[k]["exceed_rows"]), f"{100 * float(tot[k]['int_rate']):.2f}",
                                           f"{100 * float(tot[k]['util_q50']):.2f}", int(tot[k]["closed_loop"]),
                                           int(tot[k]["original"])) for k in tot},
          {"fixed": (7, "1.04", "78.89", 8, 6), "adapted": (28, "4.17", "89.09", 11, 4)})


    def cell3(r):
        n, e = int(r["n"]), int(r["exceed"])
        return f"{e} ({100 * e / n:.1f})", f"{100 * float(r['util_q50']):.1f}", \
            "Pass" if r["passed"] == "True" else "Fail", "Pass" if r["pass_original"] == "True" else "Fail"


    # text values that Table 3 must reproduce
    a = {key(r): cell3(r) for r in arms["adapted"]}
    check("T3 text: A1 charge 25 C", a[("A1", "charge", "25", "10.0")][:2], ("8 (11.0)", "89.5"))
    check("T3 text: B -20 C util", a[("B", "discharge", "-20", "10.0")][1], "73.7")
    check("T3 text: B -10 C util", a[("B", "discharge", "-10", "10.0")][1], "79.9")
    check("T3 text: B 25 C 2 s exceedances", a[("B", "discharge", "25", "2.0")][0], "2 (66.7)")

    t3 = [
        "\\begin{table}[width=\\linewidth,cols=13,pos=t]",
        "\\caption{Exceedances, median utilisation and criterion verdicts of the 15 groups at $\\tau = 5$\\,\\%.}\\label{tbl:results}",
        "\\footnotesize",
        "\\begin{tabular*}{\\tblwidth}{@{} LLRRR RRCC RRCC @{}}",
        "\\toprule",
        " & & & & & \\multicolumn{4}{c}{Fixed scale} & \\multicolumn{4}{c}{Online scale adaptation} \\\\",
        "\\cmidrule(l{4pt}r{4pt}){6-9} \\cmidrule(l{4pt}){10-13}",
        "Cell & Direction & \\makecell[r]{Temp.\\\\($^\\circ$C)} & \\makecell[r]{Duration\\\\(s)} & Rows"
        " & \\makecell[r]{Exceedances,\\\\$n$ (\\%)} & \\makecell[r]{Median\\\\util. (\\%)} & \\makecell{Util.\\\\\\& exc.} & Strict"
        " & \\makecell[r]{Exceedances,\\\\$n$ (\\%)} & \\makecell[r]{Median\\\\util. (\\%)} & \\makecell{Util.\\\\\\& exc.} & Strict \\\\",
        "\\midrule",
    ]
    csv3 = [["cell", "direction", "temperature_C", "pulse_duration_s", "rows",
             "fixed_exceedances", "fixed_exceedance_rate_pct", "fixed_median_utilisation_pct", "fixed_util_exc", "fixed_strict",
             "adapted_exceedances", "adapted_exceedance_rate_pct", "adapted_median_utilisation_pct", "adapted_util_exc", "adapted_strict"]]
    blocks = {}
    for rf, ra in zip(arms["fixed"], arms["adapted"]):
        blocks.setdefault(rf["evaluation_group"], []).append((rf, ra))
    for bi, (cell, pairs) in enumerate(blocks.items()):
        if bi:
            t3.append("\\midrule")  # one rule between cells
        cell_lab = run_labels([cell] * len(pairs))
        dir_lab = run_labels([rf["direction"].capitalize() for rf, _ in pairs])
        for (rf, ra), cl, dl in zip(pairs, cell_lab, dir_lab):
            f_, a_ = cell3(rf), cell3(ra)
            t3.append(" & ".join([cl, dl, temp_tex(rf["temp_C"]), dur(rf["horizon_s"]), rf["n"], *f_, *a_]) + " \\\\")
            n = int(rf["n"])
            csv3.append([cell, rf["direction"], int(float(rf["temp_C"])), dur(rf["horizon_s"]), n,
                         rf["exceed"], f"{100 * int(rf['exceed']) / n:.1f}", f_[1], f_[2], f_[3],
                         ra["exceed"], f"{100 * int(ra['exceed']) / n:.1f}", a_[1], a_[2], a_[3]])


    def tot3(k):
        t = tot[k]
        return [f"{t['exceed_rows']} ({100 * float(t['int_rate']):.2f})", f"{100 * float(t['util_q50']):.2f}",
                f"{t['closed_loop']}/15", f"{t['original']}/15"]


    t3 += ["\\midrule", " & ".join(["All rows", "", "", "", "672", *tot3("fixed"), *tot3("adapted")]) + " \\\\",
           "\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    csv3.append(["all", "", "", "", 672, tot["fixed"]["exceed_rows"], f"{100 * float(tot['fixed']['int_rate']):.2f}",
                 f"{100 * float(tot['fixed']['util_q50']):.2f}", f"{tot['fixed']['closed_loop']}/15", f"{tot['fixed']['original']}/15",
                 tot["adapted"]["exceed_rows"], f"{100 * float(tot['adapted']['int_rate']):.2f}",
                 f"{100 * float(tot['adapted']['util_q50']):.2f}", f"{tot['adapted']['closed_loop']}/15", f"{tot['adapted']['original']}/15"])
    (OUT / "T3_group_results.tex").write_text("\n".join(t3), encoding="utf-8")
    with open(OUT / "csv" / "T3_group_results.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv3)

    # ---------------------------------------------------------------- Table 4
    mcu = read("mcu_cost.csv")
    row = {(r["section"].strip(), r["item"].strip()): r for r in mcu}
    T = lambda item: row[("per query (us)", item)]


    def us(x, d):
        return f"{float(x):.{d}f}"


    time_rows = [
        ("Total computation per query", T("total"), 1),
        ("Feature extraction", T("features"), 1),
        ("Five local predictors and logarithmic median", T("five local predictors and log median"), 1),
        ("Scale adaptation", T("scale adaptation (request and update)"), 2),
    ]
    sample = row[("per sample (us)", "feature-state update")]
    check("T4 text means/max", [us(T("total")["mean"], 1), us(T("total")["max"], 1), us(T("features")["mean"], 1),
                                us(T("features")["max"], 1), us(T("five local predictors and log median")["mean"], 1),
                                us(T("five local predictors and log median")["max"], 1),
                                us(T("scale adaptation (request and update)")["mean"], 2),
                                us(T("scale adaptation (request and update)")["max"], 2), us(sample["mean"], 2), us(sample["max"], 2)],
          ["133.3", "228.4", "77.0", "169.8", "55.6", "73.4", "0.71", "0.90", "0.96", "9.43"])
    M = lambda item: int(row[("memory (bytes)", item)]["value"])
    flash_parts = [("Code", M("code")), ("Math library", M("math library")), ("Compiler runtime", M("compiler runtime")),
                   ("Coefficients", M("coefficients of one fold"))]
    check("T4 flash parts sum", sum(v for _, v in flash_parts), M("flash, one cell"))
    check("T4 flash total (text)", M("flash, one cell"), 16461)
    check("T4 RAM parts sum", M("state") + M("stack high-water"), M("RAM, one cell"))
    check("T4 RAM total (text)", M("RAM, one cell"), 10096)
    check("T4 rest voltage buffer (text)", M("rest-voltage buffer"), 8192)

    t4 = [
        "\\begin{table}[width=.86\\linewidth,cols=5,pos=t]",
        "\\caption{Computation time and memory of the request policy on the Cortex-M33 microcontroller.}\\label{tbl:mcu}",
        "\\begin{tabular*}{\\tblwidth}{@{} LRRRR @{}}",
        "\\toprule",
        "Computation time ($\\mu$s) & Mean & Median & 99th percentile & Maximum \\\\",
        "\\midrule",
    ]
    csv4 = [["block", "item", "mean", "median", "p99", "max", "bytes"]]
    for label, r, d in time_rows:
        t4.append(f"{label} & {us(r['mean'], d)} & {us(r['median'], d)} & {us(r['p99'], d)} & {us(r['max'], d)} \\\\")
        csv4.append(["time per query (us)", label, us(r["mean"], d), us(r["median"], d), us(r["p99"], d), us(r["max"], d), ""])
    t4.append(f"Feature-state update per raw sample & {us(sample['mean'], 2)} & {us(sample['median'], 2)} & "
              f"{us(sample['p99'], 2)} & {us(sample['max'], 2)} \\\\")
    csv4.append(["time per raw sample (us)", "Feature-state update per raw sample", us(sample["mean"], 2),
                 us(sample["median"], 2), us(sample["p99"], 2), us(sample["max"], 2), ""])
    fmt = lambda v: f"{v:,}"
    # Memory block: "Size" and the byte counts are centred across the four value columns.
    # Each count is padded on the left with invisible digits to the width of the widest count, so the centred
    # entries keep their digits right-aligned among themselves.
    mem_values = [M("flash, one cell")] + [v for _, v in flash_parts] + [M("RAM, one cell"), M("state"), M("stack high-water")]
    widest = max(mem_values, key=lambda v: len(fmt(v)))
    template = fmt(widest).translate(str.maketrans("123456789", "000000000"))


    def centred(v):
        s = fmt(v)
        pad = len(template) - len(s)
        inner = (f"\\phantom{{{template[:pad]}}}" if pad else "") + s
        return f"\\multicolumn{{4}}{{C}}{{{inner}}}"


    t4 += ["\\midrule", "Memory for one cell (bytes) & \\multicolumn{4}{C}{Size} \\\\", "\\midrule"]
    t4.append(f"Flash & {centred(M('flash, one cell'))} \\\\")
    csv4.append(["memory (bytes)", "Flash", "", "", "", "", M("flash, one cell")])
    for label, v in flash_parts:
        t4.append(f"\\quad {label} & {centred(v)} \\\\")
        csv4.append(["memory (bytes)", f"Flash: {label}", "", "", "", "", v])
    t4.append(f"RAM & {centred(M('RAM, one cell'))} \\\\")
    csv4.append(["memory (bytes)", "RAM", "", "", "", "", M("RAM, one cell")])
    t4.append(f"\\quad State including the {fmt(M('rest-voltage buffer'))}-byte rest voltage buffer & {centred(M('state'))} \\\\")
    t4.append(f"\\quad Stack high-water mark & {centred(M('stack high-water'))} \\\\")
    csv4.append(["memory (bytes)", "RAM: state including the rest voltage buffer", "", "", "", "", M("state")])
    csv4.append(["memory (bytes)", "RAM: rest voltage buffer (part of state)", "", "", "", "", M("rest-voltage buffer")])
    csv4.append(["memory (bytes)", "RAM: stack high-water mark", "", "", "", "", M("stack high-water")])
    t4 += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    (OUT / "T4_mcu_cost.tex").write_text("\n".join(t4), encoding="utf-8")
    with open(OUT / "csv" / "T4_mcu_cost.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv4)

    with open(OUT / "csv" / "checks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["check", "got", "want", "ok"])
        w.writerows([[c, g, x, ok] for c, g, x, ok in CHECKS])
    print(f"{len(CHECKS)} checks passed")


def build_table5(ROOT, TDIR):
    CHEN = ROOT / 'experiments/unseen_cell_sop_20260914/chen_v266'


    def read(p):
        with open(p, encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))


    gates = json.loads((ROOT / 'comparisons/20261104_ecm_rls/results/gates.json').read_text(encoding='utf-8'))['main_reproduction']
    pilot = {(r['method'], r['policy']): r for r in read(ROOT / 'comparisons/20261014_baseline_pilot/results/primary.csv')}
    ecm = {(r['candidate'], r['policy']): r for r in read(ROOT / 'comparisons/20261104_ecm_rls/results/primary.csv')}
    facts = {r['id']: r for r in read(ROOT / 'paper_package_lsad_20261013_r3/facts.csv')}
    bench = {r['row_id']: r for r in read(ROOT / 'experiments/unseen_cell_sop_20260914/closedloop_v251/src/bench_rows.csv')}
    chen_rows = read(CHEN / 'rows_tau5_M.csv')
    chen_summary = json.loads((CHEN / 'summary.json').read_text(encoding='utf-8'))


    def arm(d):
        return dict(exceed=int(d['exceed_rows']), rate=100 * float(d['int_rate']), util=100 * float(d['util_q50']),
                    ue=int(d['closed_loop']), strict=int(d['original']))


    rows = [('Five local predictors', arm(gates['fixed']), arm(gates['AD'])),
            ('Previous pulse power', arm(pilot[('PERSISTENCE', 'fixed')]), arm(pilot[('PERSISTENCE', 'AD')])),
            ('Previous pulse resistance', arm(pilot[('RINT_PREVIOUS', 'fixed')]), arm(pilot[('RINT_PREVIOUS', 'AD')])),
            ('Equivalent circuit, one RC branch', arm(ecm[('ECM-1RC-RLS-A1', 'fixed')]), arm(ecm[('ECM-1RC-RLS-A1', 'AD')])),
            ('Equivalent circuit, two RC branches', arm(ecm[('ECM-2RC-RLS-A1', 'fixed')]), arm(ecm[('ECM-2RC-RLS-A1', 'AD')]))]

    # ---------------------------------------------------------------- Chen reproduction, recounted from the saved rows
    groups, utils, exceed, ratio_by_scale, no_central = {}, [], 0, {}, []
    for r in chen_rows:
        b = bench[r['row_id']]
        y, p = float(b['truth_power_W']), float(r['pred'])
        e = p > y + 1e-9
        exceed += e
        utils.append(min(p / y, 1.0))
        key = (b['evaluation_group'], b['direction'], b['temp_C'], b['horizon_s'])
        groups.setdefault(key, []).append((e, min(p / y, 1.0), p / y))
        c = float(r['central'])
        if c > 0:
            ratio_by_scale.setdefault(f"{r['evaluation_group']}|{r['direction']}", []).append(p / c)
        else:  # rows without a positive central prediction must request 0 W and carry no scale information
            no_central.append(p)
    ue = sum(statistics.median([u for _, u, _ in v]) >= 0.80 and sum(e for e, _, _ in v) / len(v) <= 0.10 for v in groups.values())
    strict = sum(sum(e for e, _, _ in v) == 0 and statistics.median([q for _, _, q in v]) >= 0.80 for v in groups.values())
    chen = dict(exceed=exceed, rate=100 * exceed / len(chen_rows), util=100 * statistics.median(utils), ue=ue, strict=strict)
    stored_scales = {k: v['scale'] for k, v in chen_summary['M_tau5']['scales'].items()}
    applied_scales = {k: (min(v), max(v)) for k, v in ratio_by_scale.items()}

    p01, p02, p06 = facts['P01']['value'], facts['P02']['value'], facts['P06']['value']

    checks = [
        ('main AD 11/4/28', (rows[0][2]['ue'], rows[0][2]['strict'], rows[0][2]['exceed']) == (11, 4, 28)),
        ('main fixed 8/6/7', (rows[0][1]['ue'], rows[0][1]['strict'], rows[0][1]['exceed']) == (8, 6, 7)),
        ('main reproduction gate passed', bool(gates['passed'])),
        ('previous pulse power 3/3 and 3/1', (rows[1][1]['ue'], rows[1][1]['strict'], rows[1][2]['ue'], rows[1][2]['strict']) == (3, 3, 3, 1)),
        ('previous pulse resistance 3/3 and 3/1', (rows[2][1]['ue'], rows[2][1]['strict'], rows[2][2]['ue'], rows[2][2]['strict']) == (3, 3, 3, 1)),
        ('equivalent circuit 1RC 0/0 and 0/0', (rows[3][1]['ue'], rows[3][1]['strict'], rows[3][2]['ue'], rows[3][2]['strict']) == (0, 0, 0, 0)),
        ('equivalent circuit 2RC 1/1 and 2/1', (rows[4][1]['ue'], rows[4][1]['strict'], rows[4][2]['ue'], rows[4][2]['strict']) == (1, 1, 2, 1)),
        ('Chen P01 0 / 0 / 0', p01.replace(' ', '') == '0/0/0'),
        ('Chen rows: 672 rows, all in the benchmark', len(chen_rows) == 672 and len(groups) == 15),
        (f'Chen rows without a positive central prediction ({len(no_central)}) request 0 W', all(p == 0 for p in no_central)),
        ('Chen recount 99 exceedance rows equals P06', chen['exceed'] == 99 and p06.startswith('0 (99 exceedance rows)')),
        ('Chen recount 0 groups under both criteria equals P01 and P06', chen['ue'] == 0 and chen['strict'] == 0),
        ('Chen recount exceedance rate rounds to P02 (14.7)', round(chen['rate'], 1) == float(p02.split('/')[0])),
        ('Chen recount median utilisation rounds to P02 (56.5)', round(chen['util'], 1) == float(p02.split('/')[1])),
        ('Chen recount median utilisation equals the stored summary', abs(chen['util'] - 100 * chen_summary['M_tau5']['util_q50']) < 1e-9),
        ('Chen request / central equals the stored scale of each cell and direction',
         set(applied_scales) == set(stored_scales)
         and all(abs(lo - stored_scales[k]) < 1e-9 and abs(hi - stored_scales[k]) < 1e-9 for k, (lo, hi) in applied_scales.items())),
        ('Chen judged arm used separate charge and discharge scales (A1 0.59/0.20, A17 0.65/0.20, B 0.20)',
         stored_scales == {'A1|charge': 0.59, 'A1|discharge': 0.2, 'A17|charge': 0.65, 'A17|discharge': 0.2, 'B|discharge': 0.2}),
        ('exceedance rate equals count / 672 for every arm', all(abs(a['rate'] - 100 * a['exceed'] / 672) < 1e-9
                                                                for _, f, d in rows for a in (f, d))),
        ('independent recount of the comparison passed',
         json.loads((ROOT / 'comparisons/20261104_ecm_rls/results/independent_validation.json').read_text(encoding='utf-8'))['passed']),
        ('pilot independent recount passed',
         not json.loads((ROOT / 'comparisons/20261014_baseline_pilot/results/independent_validation.json').read_text(encoding='utf-8'))['score_recount_errors']),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        sys.exit(f'checks failed: {failed}')


    def cell(a):
        return [f"{a['exceed']} ({a['rate']:.2f})", f"{a['util']:.2f}", f"{a['ue']}/15", f"{a['strict']}/15"]


    body = []
    for name, f, d in rows:
        body.append(' & '.join([name] + cell(f) + cell(d)) + r' \\')
    body.append(' & '.join([r'Reproduction of Chen et al. \cite{ref7}'] + cell(chen) + ['--'] * 4) + r' \\')
    tex = '\n'.join([
        r'\begin{table}[width=\linewidth,cols=9,pos=t]',
        r'\caption{Exceedances, median utilisation and passing groups of the five local predictors and four comparators at $\tau = 5$\,\%.}\label{tbl:comparators}',
        r'\footnotesize',
        r'\begin{tabular*}{\tblwidth}{@{} L RRCC RRCC @{}}',
        r'\toprule',
        r' & \multicolumn{4}{c}{Fixed scale} & \multicolumn{4}{c}{Online scale adaptation} \\',
        r'\cmidrule(l{4pt}r{4pt}){2-5} \cmidrule(l{4pt}){6-9}',
        r'Predictor & \makecell[r]{Exceedances,\\$n$ (\%)} & \makecell[r]{Median\\util. (\%)} & \makecell{Util.\\\& exc.} & Strict & '
        r'\makecell[r]{Exceedances,\\$n$ (\%)} & \makecell[r]{Median\\util. (\%)} & \makecell{Util.\\\& exc.} & Strict \\',
        r'\midrule',
        *body,
        r'\bottomrule',
        r'\end{tabular*}',
        r'\end{table}',
    ]) + '\n'
    (TDIR / 'csv').mkdir(parents=True, exist_ok=True)
    (TDIR / 'T5_comparators.tex').write_text(tex, encoding='utf-8')
    with open(TDIR / 'csv/T5_comparators.csv', 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['predictor', 'fixed_exceedances', 'fixed_exceedance_rate_pct', 'fixed_median_utilisation_pct',
                    'fixed_util_exc_groups', 'fixed_strict_groups', 'adapted_exceedances', 'adapted_exceedance_rate_pct',
                    'adapted_median_utilisation_pct', 'adapted_util_exc_groups', 'adapted_strict_groups'])
        for name, f, d in rows:
            w.writerow([name, f['exceed'], f"{f['rate']:.4f}", f"{f['util']:.4f}", f['ue'], f['strict'],
                        d['exceed'], f"{d['rate']:.4f}", f"{d['util']:.4f}", d['ue'], d['strict']])
        w.writerow(['Reproduction of Chen et al. (2026)', chen['exceed'], f"{chen['rate']:.4f}", f"{chen['util']:.4f}",
                    chen['ue'], chen['strict'], '', '', '', '', ''])
    with open(TDIR / 'csv/checks_T5.csv', 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['check', 'passed'])
        w.writerows(checks)
    print(f'T5 written, {len(checks)} checks passed, Chen median utilisation {chen["util"]:.4f} %')


if __name__ == '__main__':
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    build_tables_1_3_4(root / 'paper_package_lsad_20261013_r3' / 'data', out)
    build_table5(root, out)
