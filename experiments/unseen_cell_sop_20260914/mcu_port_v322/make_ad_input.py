"""V322: ad_v322.bin = per cell (A1, A17, B) the queries in the adaptation's processing order (tkey) with their reference
SOP, for the host-side event (request > reference) of the closed loop. Format: uint32 n, then n x (uint32 qid, float64 truth)."""
import struct
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
Q = pd.read_csv(OUT / 'queries_v322.csv', float_precision='round_trip')
with open(OUT / 'ad_v322.bin', 'wb') as f:
    for cell in (0, 1, 2):
        g = Q[Q.cell == cell].sort_values('tkey', kind='stable')
        f.write(struct.pack('<I', len(g)))
        for x in g.itertuples():
            f.write(struct.pack('<Id', int(x.query_id), float(x.truth)))
print('ad order written', Q.groupby('cell').size().to_dict())
