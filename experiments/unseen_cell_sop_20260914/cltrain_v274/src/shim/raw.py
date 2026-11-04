"""V274 local stand-in for aging_refit/tools/rpcwby/raw.py: the same member parsing as raw._parse_one (usecols,
utf-8-sig, column renaming, numeric coercion), read directly from the local RPCWBY zips instead of the remote parquet
cache, and the same window(); the tester's UD registers are never read. Row r = 0-based CSV data line (raw.py rule)."""
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[5] / "raw" / "RPCWBY"
ZIPS = {"1_Test#1": RAW / "1_Test_1.zip", "2_Test#2": RAW / "2_Test_2.zip", "Test#3": RAW / "3_Test_3.zip"}
BASE = {"Data_Point": "dp", "Test_Time(s)": "t", "Step_Time(s)": "st", "Cycle_Index": "cyc", "Step_Index": "step",
        "Current(A)": "I", "Voltage(V)": "V", "Power(W)": "P", "Charge_Capacity(Ah)": "qc", "Discharge_Capacity(Ah)": "qd"}
_MEM = {}


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
    return df[cols].iloc[int(h0):int(q)].copy()
