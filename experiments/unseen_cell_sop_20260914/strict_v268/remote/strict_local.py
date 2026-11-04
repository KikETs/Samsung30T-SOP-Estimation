"""V268 (PROTOCOL_STRICT_V268.md): the five local predictors (prev, L4, M6, DN6, G1; sop_ident/run_local.sh settings) fitted
on ONE cell e: the evaluation-group label is rewritten to e / ZZ so that the original fold loop trains on e only (fold ZZ).
Self-check first: the same wrapper with the original labels must reproduce sop_ident's leave-one-cell-out folds.csv.
Output: out/strict/local_strict.csv (name, train_cell, row_id, pred_power_W) for the rows of the cells other than e."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_sop as ls, local_sop2 as ls2

ROOT = Path(__file__).resolve().parents[1]; SI = ROOT.parent / "sop_ident"
CH = "chg>=0.8@step:pdiff,slope,lT,sqA"
SPECS = {"L4": (["*@prev:trend10,trend1,invT"], False, False),
         "M6": (["*@prev:trendS,trendS1,dlT,invT,sqA", CH, "chg>=0.8|first@off:"], False, False),
         "DN6": (["*@prev:trendS,amp,ext,invT", CH, "chg>=0.8|first@off:"], True, False),
         "G1": (["*@prev:ext", CH, "chg>=0.8|first@off:", "chg>=0.8|firstcv@off!:", "dis>=0.4|firstcv@off!:"], True, True)}


def folds(name, D):
    """All folds of predictor `name` over D's evaluation_group labels: DataFrame(fold, row_id, pred_power_W)."""
    if name == "prev":
        out = []
        for c in sorted(D.evaluation_group.unique()):
            par = ls.fit(D[D.evaluation_group != c], "prev", True)
            P = pd.concat([D[["row_id"]], ls.predict(D, "prev", par)], axis=1)
            out.append(P[["row_id", "pred_power_W"]].assign(fold=c))
        return pd.concat(out)
    spec, direct, gap = SPECS[name]
    _, _, F = ls2.run(D, ls2.parse_specs(spec), True, direct, gap)
    return F[["fold", "row_id", "pred_power_W"]]


def main():
    D = ls.table(); cells = sorted(D.evaluation_group.unique()); res = []
    for name in ("prev", "L4", "M6", "DN6", "G1"):                       # self-check against the frozen outputs
        F = folds(name, D).merge(pd.read_csv(SI / f"out/local/{name}/folds.csv")[["fold", "row_id", "pred_power_W"]], on=["fold", "row_id"], suffixes=("", "_ref"))
        r = (F.pred_power_W / F.pred_power_W_ref - 1).abs(); nan_mis = int((F.pred_power_W.isna() != F.pred_power_W_ref.isna()).sum())
        print(f"self-check {name}: rows {len(F)}, max rel diff {np.nanmax(r):.2e}, nan mismatch {nan_mis}", flush=True)
        if not (np.nanmax(r) <= 1e-9 and nan_mis == 0):
            raise SystemExit(f"wrapper does not reproduce {name}; stopping (protocol)")
    for e in cells:
        Dl = D.copy(); Dl["evaluation_group"] = np.where(D.evaluation_group == e, e, "ZZ")
        own = D.set_index("row_id").evaluation_group
        for name in ("prev", "L4", "M6", "DN6", "G1"):
            F = folds(name, Dl); F = F[F.fold == "ZZ"].copy(); F["cell"] = F.row_id.map(own); F = F[F.cell != e]
            res.append(F.assign(name=name, train_cell=e)[["name", "train_cell", "row_id", "cell", "pred_power_W"]])
        print(f"train on {e}: done", flush=True)
    out = ROOT / "out/strict"; out.mkdir(parents=True, exist_ok=True)
    R = pd.concat(res); R.to_csv(out / "local_strict.csv", index=False)
    print(R.groupby(["train_cell", "name"]).pred_power_W.apply(lambda x: x.notna().sum()).to_string(), flush=True)


if __name__ == "__main__":
    main()
