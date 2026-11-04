"""V268: the reconstructed i3c command must reproduce sop_ident_premodes' i3c predictions on the short test sweeps."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
a = pd.read_csv(ROOT / "out/ident/repro/pred.csv"); b = pd.read_csv(ROOT.parent / "sop_ident_premodes/out/ident/i3c/pred.csv")
m = a.merge(b, on="row_id", suffixes=("", "_ref"))
r = (m.pred_power_W / m.pred_power_W_ref - 1).abs()
print(f"rows {len(a)} matched {len(m)}; max rel diff {np.nanmax(r) if len(r) else float('nan'):.3e}; status equal {bool((m.status == m.status_ref).all())}")
sys.exit(0 if len(m) == len(a) and len(m) > 0 and np.nanmax(r) <= 1e-9 and (m.status == m.status_ref).all() else 1)
