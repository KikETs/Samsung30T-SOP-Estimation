"""V281 K1a second step (PROTOCOL_PRESUBMIT_V281.md: absolute-path hits -> static import analysis from the eight entry
tools of the rebuild chain). Written after the first run, as the plan foresaw for the case of hits.
1. The local vT1h working copy must equal the remote rebuild copy file by file (remote_hashes.tsv, sha_rebuild).
2. Static closure: every Import / ImportFrom node of a file (also inside functions) is resolved against the directories
   the tools put on sys.path at run time (each tool's own folder first); ambiguous names are followed in every candidate.
3. Hit files inside the closure are listed with their hit lines (code or comment / docstring).
usage: python presubmit_v281/k1a_imports.py"""
import ast
import hashlib
import json
from pathlib import Path

import pandas as pd

V = Path(__file__).resolve().parent; HERE = V.parent
MIR = HERE.parent / 'physrebuild_vT1h/mirror_vT1h/Samsung30T'
DIRS = ['strictfull_v273/tools', 'sop_ident_premodes/tools', 'sop_ident/tools', 'aging_refit/scripts', 'aging_refit/tools/v2',
        'aging_refit/tools/rpcwby', 'aging_refit/tools/stage2']
ENTRY = ['aging_refit/tools/stage2/fit_rpt_ocv.py', 'aging_refit/tools/v2/thermo.py', 'aging_refit/tools/v2/pulses_uy2.py',
         'aging_refit/tools/v2/law_fit2.py', 'sop_ident_premodes/tools/premodes.py', 'strictfull_v273/tools/strictfull_tables.py',
         'strictfull_v273/tools/run_sweeps.py', 'strictfull_v273/tools/ident7p.py']
PAT = ('/home/', 'Desktop/Samsung30T', 'expanduser(')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def imports(p):
    names = set()
    for n in ast.walk(ast.parse(p.read_text(), str(p))):
        if isinstance(n, ast.Import):
            names |= {a.name.split('.')[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            names.add(n.module.split('.')[0])
    return names


def resolve(name, own):
    out = []
    for d in [own] + [MIR / x for x in DIRS]:
        f = d / f'{name}.py'
        if f.is_file() and f not in out:
            out.append(f)
    return out


def main():
    T = pd.read_csv(V / 'remote_hashes.tsv', sep='\t', dtype=str, keep_default_na=False)
    H = T[T.kind == 'hash']; hits = {f'{d}/{p}' for d, p in T[T.kind == 'abspath'][['dir', 'relpath']].values}
    same = [(f'{d}/{p}', (MIR / d / p).is_file() and sha(MIR / d / p) == h) for d, p, h in H[['dir', 'relpath', 'sha_rebuild']].values]
    differ = [k for k, ok in same if not ok]
    seen, todo, ambiguous = set(), [MIR / e for e in ENTRY], {}
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for nm in imports(f):
            c = resolve(nm, f.parent)
            if len(c) > 1:
                ambiguous[nm] = sorted(str(x.relative_to(MIR)) for x in c)
            todo += c
    closure = sorted(str(f.relative_to(MIR)) for f in seen)
    inter = sorted(set(closure) & hits)
    lines = {}
    for k in inter:
        rows = []
        for i, s in enumerate((MIR / k).read_text().splitlines(), 1):
            if any(p in s for p in PAT):
                st = s.strip(); rows.append(dict(line=i, text=st[:160], comment=st.startswith('#')))
        lines[k] = rows
    out = dict(mirror_equals_remote_copy=not differ, files_compared=len(same), differing=differ, entry_tools=ENTRY,
               closure_size=len(closure), closure=closure, ambiguous_names=ambiguous, hits_total=len(hits),
               hits_in_closure=inter, hit_lines=lines)
    (V / 'k1a_imports.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('closure',)}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
