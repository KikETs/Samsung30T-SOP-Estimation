"""V276 masking stand-in for aging_refit/tools/rpcwby/raw.py: cltrain_v274/src/shim/raw.py (the same member parsing,
read from the local RPCWBY zips) plus two things for the invariance tests (PROTOCOL_INVARIANCE_V276.md):
  MASK   list of (first_row, last_row) ranges (absolute 0-based data rows, inclusive) whose samples window() returns
         as NaN in every column: the removed records of a test (T2: the query sample q; T4: the earlier iterations of
         the row's own search);
  READS  every window() call as (member, start, end) so that the first row read can be checked against h0.
Row r = 0-based CSV data line (raw.py rule)."""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[5] / "raw" / "RPCWBY"
ZIPS = {"1_Test#1": RAW / "1_Test_1.zip", "2_Test#2": RAW / "2_Test_2.zip", "Test#3": RAW / "3_Test_3.zip"}
BASE = {"Data_Point": "dp", "Test_Time(s)": "t", "Step_Time(s)": "st", "Cycle_Index": "cyc", "Step_Index": "step",
        "Current(A)": "I", "Voltage(V)": "V", "Power(W)": "P", "Charge_Capacity(Ah)": "qc", "Discharge_Capacity(Ah)": "qd"}
_MEM = {}
MASK = []
READS = []


def load(member):
    if member not in _MEM:
        z = zipfile.ZipFile(ZIPS[member.split("/")[0]])
        with z.open(member) as f:
            cols = list(pd.read_csv(f, nrows=0, encoding="utf-8-sig").columns)
        tcols = [c for c in cols if c.startswith("Aux_Temperature")]
        use = [c for c in cols if c in BASE] + tcols[:2]
        with z.open(member) as f:
            df = pd.read_csv(f, usecols=use, encoding="utf-8-sig", low_memory=False)
        df = df[use]
        ren = dict(BASE)
        ren[tcols[0]] = "T_cell"
        if len(tcols) > 1:
            ren[tcols[1]] = "T_chamber"
        df = df.rename(columns=ren)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        _MEM.clear()                      # one member at a time per process (the jobs are grouped by member)
        _MEM[member] = df
    return _MEM[member]


def columns(member):
    return list(load(member).columns)


def window(member, h0, q, cols=None):
    df = load(member)
    cols = [c for c in (cols or df.columns) if not str(c).startswith("ud")]
    W = df[cols].iloc[int(h0):int(q)].copy()
    READS.append((member, int(h0), int(q)))
    for lo, hi in MASK:
        a, b = max(int(lo), int(h0)), min(int(hi), int(q) - 1)
        if a <= b:
            W = W.astype(float)           # integer columns (cycle, step) cannot hold NaN; values unchanged
            W.iloc[a - int(h0):b - int(h0) + 1] = np.nan
    return W
